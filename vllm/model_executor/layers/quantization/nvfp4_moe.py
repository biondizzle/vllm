# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 MoE quantization method for DeepSeek-V4.

Uses the NVFP4 (NVIDIA FP4 with block + global scales) format for MoE
expert weights. Supports CuTeDSL kernel backend for Blackwell (SM100+).
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class NvFp4MoEMethod(FusedMoEMethodBase):
    """NVFP4 MoE quantization method.

    Supports multiple backends (CuTeDSL, FlashInfer+CUTEDSL, etc.)
    selected via --kernel-config moe_backend=cutedsl (or auto-selection).
    """

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.weight_dtype = "nvfp4"
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(moe)

        self.max_capture_size = (
            get_current_vllm_config().compilation_config.max_cudagraph_capture_size
        )

        self.moe_kernel: mk.FusedMoEKernel | None = None

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        # NVFP4 requires alignment to 128 elements for block scales
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        block_size = 128
        if hidden_size % block_size != 0:
            hidden_size = ((hidden_size + block_size - 1) // block_size) * block_size
        if intermediate_size_per_partition % block_size != 0:
            intermediate_size_per_partition = (
                (intermediate_size_per_partition + block_size - 1) // block_size
            ) * block_size
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.float32  # NVFP4 uses float32 global scales
        nvfp4_block = 32  # Same as MXFP4 for weight block scales

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # Per-block weight scales (ue8m0 packed as uint8)
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // nvfp4_block,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w13_weight_scale.quant_method = "block"

        # Per-expert-per-row global scales (float32)
        w13_scale_2 = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale_2", w13_scale_2)
        set_weight_attrs(w13_scale_2, extra_weight_attrs)

        # Activation global scales (float32, per-expert)
        a13_scale = torch.nn.Parameter(
            torch.zeros(num_experts, dtype=scale_dtype),
            requires_grad=False,
        )
        layer.register_parameter("a13_scale", a13_scale)
        set_weight_attrs(a13_scale, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-block weight scales
        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // nvfp4_block,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        w2_weight_scale.quant_method = "block"

        # Per-expert-per-row global scales
        w2_scale_2 = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale_2", w2_scale_2)
        set_weight_attrs(w2_scale_2, extra_weight_attrs)

        # Activation global scales
        a2_scale = torch.nn.Parameter(
            torch.zeros(num_experts, dtype=scale_dtype),
            requires_grad=False,
        )
        layer.register_parameter("a2_scale", a2_scale)
        set_weight_attrs(a2_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_scale_2 = layer.w13_scale_2
        w2_scale_2 = layer.w2_scale_2
        a13_scale = layer.a13_scale
        a2_scale = layer.a2_scale

        if self.nvfp4_backend == NvFp4MoeBackend.CUTEDSL:
            # CuTeDSL handles weight transformation in its own
            # process_weights_after_loading. Store the raw weights
            # and let the expert class handle conversion.
            pass

        # Convert weights to kernel format (for non-CUTEDSL backends)
        (
            w13, w13_scale, w13_scale_2, a13_scale,
            w2, w2_scale, w2_scale_2, a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=w13,
            w13_scale=w13_scale,
            w13_scale_2=w13_scale_2,
            a13_scale=a13_scale,
            w2=w2,
            w2_scale=w2_scale,
            w2_scale_2=w2_scale_2,
            a2_scale=a2_scale,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w13_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_scale_2", w2_scale_2)
        replace_parameter(layer, "a13_scale", a13_scale)
        replace_parameter(layer, "a2_scale", a2_scale)

        # Build quant config
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        # Build kernel
        if self.moe_quant_config is not None and self.experts_cls is not None:
            self.moe_kernel = make_nvfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    def get_fused_moe_quant_config(
        self,
        layer: RoutedExperts,
    ) -> FusedMoEQuantConfig | None:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_scale_2,
            w2_scale_2=layer.w2_scale_2,
            a13_scale=layer.a13_scale,
            a2_scale=layer.a2_scale,
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: RoutedExperts,
    ) -> mk.FusedMoEExpertsModular:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel "
            "initialization logic. This function should not be called."
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            router_logits=router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
