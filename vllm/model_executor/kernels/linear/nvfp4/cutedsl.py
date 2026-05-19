# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTeDSL NVFP4 Linear Kernel for vLLM.

Registers as an NvFp4LinearKernel so that vLLM kernel selection
(init_nvfp4_linear_kernel) picks it up on Blackwell GPUs.
Routes NVFP4 GEMM through CuTeDSL's MLIR-compiled grouped GEMM.

Uses torch.library.custom_op to make Dynamo (torch.compile) treat the
GEMM as opaque. The runner's _run_impl is already cudagraph-safe.
"""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
from cutedsl.custom_ops import register_runner, nvfp4_linear_gemm

logger = init_logger(__name__)


class CuTeDSLNvFp4LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the CuTeDSL framework (Blackwell SM100+)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        cap = compute_capability or current_platform.get_device_capability()
        if cap is not None and cap.major >= 10:
            return True, None
        return False, "CuTeDSL NVFP4 requires SM100+ (Blackwell)"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert NVFP4 weights into CuTeDSL kernel format."""
        from cutedsl.nvfp4_linear import CuTeDSLNvfp4Linear

        w_uint8 = layer.weight.data
        device = w_uint8.device
        out_features = w_uint8.shape[0]
        in_features = w_uint8.shape[1] * 2

        w_fp4 = w_uint8.view(torch.float4_e2m1fn_x2).permute(1, 0).contiguous()

        sf = layer.weight_scale.data
        if sf.dtype != torch.float8_e4m3fn:
            sf = sf.to(torch.float8_e4m3fn)
        sf = sf.permute(1, 0).contiguous()

        gs = layer.weight_global_scale.data.item()

        if layer.weight_global_scale.numel() == 2:
            gs0 = layer.weight_global_scale[0].item()
            gs1 = layer.weight_global_scale[1].item()
            gs = max(gs0, gs1)
            if gs0 != gs1:
                sf_f32 = sf.float()
                logical_widths = getattr(layer, 'logical_widths', None)
                if logical_widths is not None and len(logical_widths) == 2:
                    split_point = logical_widths[0]
                else:
                    split_point = out_features // 2
                sf_f32[:, :split_point] *= (gs0 / gs)
                sf_f32[:, split_point:] *= (gs1 / gs)
                sf = sf_f32.to(torch.float8_e4m3fn)

        runner = CuTeDSLNvfp4Linear(
            in_features=in_features,
            out_features=out_features,
            device=str(device),
        )
        runner.fp4 = [w_fp4]
        runner.sf = [sf]
        runner.gs = [gs]
        runner.finalize_weights()

        # Warmup: compute activation global scale from sample data.
        # The checkpoint's input_scale is a calibration-time value that does NOT
        # match what quantize_activation_nvfp4 expects at runtime. Using it
        # produces garbage output (empty EOS tokens). The correct approach is
        # a warmup forward pass that measures the actual activation distribution.
        # Use only 1 token to minimize GPU memory overhead during weight loading.
        with torch.no_grad():
            sample = torch.randn(
                1, in_features,
                dtype=torch.bfloat16, device=str(device),
            ) * 2.0
            runner.compute_activation_global_scale(sample)
            del sample
            torch.cuda.empty_cache()

        # Register the runner and store the ID (not the runner itself)
        layer._cutedsl_runner_id = register_runner(runner)
        layer._cutedsl_out_features = out_features

        # Replace weight with a GPU dummy (some vLLM code paths like
        # torch.mm(compressor.weight.T) expect weight on GPU).
        layer.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.bfloat16,
                        device=device),
            requires_grad=False,
        )

        for attr in ("weight_scale", "weight_global_scale",
                      "input_global_scale", "input_global_scale_inv",
                      "alpha", "weights_padding_cols", "weight_scale_2",
                      "input_scale"):
            if hasattr(layer, attr):
                try:
                    delattr(layer, attr)
                except Exception:
                    pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = nvfp4_linear_gemm(
            x,
            layer._cutedsl_runner_id,
            layer._cutedsl_out_features,
        )
        if bias is not None:
            result = result + bias
        return result
