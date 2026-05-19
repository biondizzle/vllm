# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTeDSL NVFP4 MoE experts for DeepSeek-V4.

Integrates the CuTeDSL NVFP4 grouped GEMM kernel into vLLM's FusedMoE
modular kernel framework. This is the proper integration path — no
monkey-patching, no post-load surgery.

The CuTeDSL kernel is a Python-based CUTLASS kernel compiled via MLIR → PTX.
It handles:
  - L1 GEMM (gate + up projections)
  - SiLU activation with optional swiglu_limit clamping
  - L2 GEMM (down projection)
  - All with NVFP4 (float8_e4m3fn block scales + float32 global scales)

CUDA-graph-safe: all intermediate buffers pre-allocated, no CPU-GPU syncs,
no dynamic shapes.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

from cutedsl.runner import CuTeDSLMoERunner


class CuTeDSLMoEExperts(mk.FusedMoEExpertsModular):
    """CuTeDSL NVFP4 MoE experts using the custom CuTeDSL grouped GEMM kernel.

    Uses Standard activation format (non-batched). Handles input quantization
    internally (expects_unquantized_inputs=True).

    Supports expert parallelism: remaps global→local expert IDs.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.quant_dtype == "nvfp4", (
            "CuTeDSL MoE only supports nvfp4 quantization, "
            f"got {quant_config.quant_dtype}"
        )
        self.out_dtype = moe_config.in_dtype
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.topk = moe_config.experts_per_token
        self.local_num_experts = moe_config.num_local_experts
        self.global_num_experts = moe_config.num_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self.local_expert_offset = self.ep_rank * self.local_num_experts
        # max_num_tokens from scheduler config (for buffer pre-allocation)
        self.max_num_tokens = getattr(moe_config, 'max_num_tokens', 8192)

        # swiglu_limit: read from the FusedMoE layer in process_weights_after_loading
        self._swiglu_limit = None

        # Runner is created in process_weights_after_loading
        self._runner: CuTeDSLMoERunner | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert NVFP4 MoE weights into CuTeDSL kernel format.

        Reads w13/w2 weight tensors from the FusedMoE layer, converts them
        to the CuTeDSL runner's expected format, and creates the runner.
        Also folds the activation global scale (input_scale) into the
        weight global scale (weight_scale_2) as the runner's alpha.
        """
        num_experts = layer.w13_weight.shape[0]
        hidden_size = self.hidden_dim
        intermediate_size = self.intermediate_size_per_partition
        device = layer.w13_weight.device

        # NOTE: For the CuTeDSL kernel, we do NOT fold input_scale into
        # weight_scale_2. The CuTeDSL runner uses weight global scale
        # (weight_scale_2) and activation global scale separately.
        # The activation global scale is computed via warmup before first inference.
        # 
        # Also, convert_to_nvfp4_moe_kernel_format already inverted input_scale
        # (1.0 / a13_scale) for the quant config. We undo that inversion here
        # to get the original input_scale, then use it as initial activation gs.
        if layer.w13_input_scale is not None and not isinstance(layer.w13_input_scale, float):
            # input_scale was inverted in convert_to_nvfp4_moe_kernel_format
            # Original: input_scale = amax / (6.0 * 448.0)
            # Inverted: 1.0 / input_scale = 6.0 * 448.0 / amax
            # We need the original for activation gs
            w13_input_scale_orig = 1.0 / layer.w13_input_scale
        else:
            w13_input_scale_orig = None
        if layer.w2_input_scale is not None and not isinstance(layer.w2_input_scale, float):
            w2_input_scale_orig = 1.0 / layer.w2_input_scale
        else:
            w2_input_scale_orig = None

        # Extract weights from the layer — checkpoint format, no copies yet.
        # w13_weight: (E, 2*intermediate, hidden//2) uint8 — gate + up fused
        # w2_weight: (E, hidden, intermediate//2) uint8 — down
        # w13_weight_scale: (E, 2*intermediate, hidden//16) fp8
        # w2_weight_scale: (E, hidden, intermediate//16) fp8
        w13_uint8 = layer.w13_weight.data  # (E, 2*inter, hidden//2)
        w2_uint8 = layer.w2_weight.data    # (E, hidden, intermediate//2)
        w13_sf = layer.w13_weight_scale.data  # (E, 2*inter, hidden//16) = (E, N, K_sf)
        w2_sf = layer.w2_weight_scale.data    # (E, hidden, intermediate//16) = (E, N, K_sf)
        w13_gs = layer.w13_weight_scale_2.data  # (E,) or (E, 2)
        w2_gs = layer.w2_weight_scale_2.data    # (E,) or (E, 2)

        # View as fp4 — byte-preserving, NO copy
        l1_fp4 = w13_uint8.view(torch.float4_e2m1fn_x2)  # (E, N, K_packed)
        l2_fp4 = w2_uint8.view(torch.float4_e2m1fn_x2)    # (E, N, K_packed)

        # Ensure scales are float8_e4m3fn (no copy if already correct dtype)
        if w13_sf.dtype != torch.float8_e4m3fn:
            w13_sf = w13_sf.to(torch.float8_e4m3fn)
        if w2_sf.dtype != torch.float8_e4m3fn:
            w2_sf = w2_sf.to(torch.float8_e4m3fn)

        # Global scales
        l1_gs_list = w13_gs.tolist()
        l2_gs_list = w2_gs.tolist()

        # Free original weight tensors IMMEDIATELY.
        # We have views into the same memory (l1_fp4, l2_fp4), but the runner
        # will create its own copies in _ensure_stacked. Free the layer refs
        # now so the memory can be reclaimed when the views are no longer held.
        # NOTE: The modular kernel framework reads w1.shape[0] in its outer
        # apply() before delegating to our expert impl, so we can't set the
        # weights to None. Instead, replace with a shape-preserving dummy on CPU
        # to free GPU memory while keeping the shape metadata accessible.
        # Free the large weight tensors — they're now in the runner.
        # Keep the scale tensors (small) because the framework's warmup
        # and quant config construction needs them.
        layer.w13_weight = torch.nn.Parameter(torch.empty(
            num_experts, 2 * intermediate_size, hidden_size // 2,
            device='cpu', dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.empty(
            num_experts, hidden_size, intermediate_size // 2,
            device='cpu', dtype=torch.uint8), requires_grad=False)

        # Create the CuTeDSL runner
        self._runner = CuTeDSLMoERunner(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_num_tokens=self.max_num_tokens,
            top_k=self.topk,
            device=str(device),
            experts_start_idx=self.local_expert_offset,
        )
        # Pass stacked tensors in checkpoint format (E, N, K) — no copies needed
        self._runner.prepare_weights_from_stacked(
            l1_fp4, w13_sf, l1_gs_list,
            l2_fp4, w2_sf, l2_gs_list,
        )
        if self._swiglu_limit is not None:
            self._runner.set_swiglu_limit(float(self._swiglu_limit))
        
        # Read swiglu_limit from the FusedMoE layer (set by DeepseekV4MoE)
        swiglu_limit = getattr(layer, 'swiglu_limit', None)
        if swiglu_limit is not None:
            self._swiglu_limit = swiglu_limit
            self._runner.set_swiglu_limit(float(swiglu_limit))

        # Set initial activation global scales from checkpoint input_scale.
        # After undoing the inversion from convert_to_nvfp4_moe_kernel_format,
        # w13_input_scale_orig = amax / (6.0 * 448.0), which IS the activation
        # global scale that quantize_activation_nvfp4 expects.
        # The warmup step (compute_activation_global_scales) will override
        # this with an empirically computed value before the first inference.
        if w13_input_scale_orig is not None:
            # w13_input_scale_orig = amax / (6.0 * 448.0) = activation gs
            # Mean across experts (they should be similar)
            mean_l1_gs = float(w13_input_scale_orig.mean().item())
            if mean_l1_gs > 0:
                self._runner._l1_activation_global_scale = mean_l1_gs
        if w2_input_scale_orig is not None:
            mean_l2_gs = float(w2_input_scale_orig.mean().item())
            if mean_l2_gs > 0:
                self._runner._l2_activation_global_scale = mean_l2_gs

        # Note: activation global scale warmup must be done after
        # process_weights_after_loading, before the first inference.
        # This is handled by the model's load_weights or a separate warmup step.

    @property
    def runner(self) -> CuTeDSLMoERunner | None:
        return self._runner

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # CuTeDSL requires CUDA SM100 (Blackwell)
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(100)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kNvfp4Static, kNvfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # We handle SiLU + swiglu_limit internally
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Our runner handles activation quantization internally
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Our runner manages its own workspace internally (pre-allocated buffers)
        workspace1 = (0,)
        workspace2 = (0,)
        # The output of the full 2-stage MoE pipeline is hidden_dim.
        # K comes from hidden_states.size(-1) (full BF16 dimension, not packed).
        output = (M, self.hidden_dim)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self._runner is not None, (
            "CuTeDSL runner not initialized. "
            "Call process_weights_after_loading first."
        )

        # Our runner expects topk_ids as global expert IDs.
        # The modular kernel framework may pass local IDs with expert_map.
        # We handle remapping internally via experts_start_idx.
        result = self._runner.run(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        # Copy result into output tensor
        output.copy_(result)
