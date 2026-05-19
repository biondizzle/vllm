# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MHC (Multi-Head Control) layer — pure PyTorch fallback.

Replaces the TileLang-based implementation with pure PyTorch that works
on all GPU architectures including Blackwell (SM100+). The original
implementation imports tilelang and JIT-compiles kernels which don't
work correctly on SM100.
"""

import torch

from vllm.model_executor.custom_op import CustomOp


# ── Pure PyTorch MHC implementations ──────────────────────────────────

def _mhc_pre_impl(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.view(num_tokens, hc_hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / hc_hidden_size + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (mixes[:, 2 * hc_mult:]
                   .view(num_tokens, hc_mult, hc_mult)
                   * hc_scale[2]
                   + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult))
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_post_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def _mhc_fused_post_pre_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_residual = _mhc_post_impl(x, residual, post_layer_mix, comb_res_mix)
    post_mix, res_mix, layer_input = _mhc_pre_impl(
        new_residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, n_splits,
    )
    return new_residual, post_mix, res_mix, layer_input


def _hc_head_fused_kernel_impl(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    if hs_flat.shape[0] == 0:
        return
    x_flat = hs_flat.reshape(hs_flat.shape[0], hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x_flat, fn.t())
    sqrsum = x_flat.square().sum(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
    result = torch.sum(pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1).to(out.dtype)
    out.copy_(result)


# ── CustomOp wrappers ─────────────────────────────────────────────────

@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block — pure PyTorch implementation."""

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _mhc_pre_impl(
            residual, fn, hc_scale, hc_base, rms_eps,
            hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat, n_splits,
        )

    def forward_native(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block — pure PyTorch implementation."""

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return _mhc_post_impl(x, residual, post_layer_mix, comb_res_mix)

    def forward_native(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post + pre block — pure PyTorch implementation."""

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _mhc_fused_post_pre_impl(
            x, residual, post_layer_mix, comb_res_mix,
            fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
            n_splits, tile_n,
        )

    def forward_native(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


# HC head fused kernel — registered as a torch custom op (mutates out)
from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    op_name="hc_head_fused_kernel",
    op_func=_hc_head_fused_kernel_impl,
    mutates_args=["out"],
)


@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head operation — reduces multi-head residual to single hidden state."""

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        # hidden_states: (num_tokens, hc_mult, hidden_size)
        hc_mult = hidden_states.shape[-2]
        hidden_size = hidden_states.shape[-1]
        x = hidden_states.reshape(hidden_states.shape[0], hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(x, fn.t())
        sqrsum = x.square().sum(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
        pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps
        result = torch.sum(pre_mix.unsqueeze(-1) * hidden_states.to(torch.float32), dim=1).to(hidden_states.dtype)
        return result

    def forward_native(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
