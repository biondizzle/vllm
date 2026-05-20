# SPDX-License-Identifier: Apache-2.0
"""
CSA/HCA attention for Blackwell (SM100+).

Replaces vLLM's FlashMLA + fused CUDA kernels with our own KV cache-based
attention pipeline.

Architecture:
- KV latent: (T, HD=512) single head, shared across 128 Q heads
- KV Cache: fp8_e4m3 paged cache with per-token inverse scale
- SWA Decode: CuTeDSL native kernel (native_swa_decode_attention)
- CSA/HCA Decode: native SWA component + sparse attention (TODO: CuTeDSA sparse)
- Prefill: BF16 causal SDPA
"""

import torch
import torch.nn.functional as F


def apply_gptj_rope(x, cos, sin, nope_dim):
    out = x.clone()
    even = x[..., nope_dim:][..., 0::2]
    odd = x[..., nope_dim:][..., 1::2]
    out[..., nope_dim:][..., 0::2] = even * cos - odd * sin
    out[..., nope_dim:][..., 1::2] = even * sin + odd * cos
    return out


def apply_inv_gptj_rope(x, cos, sin, nope_dim):
    out = x.clone()
    even = x[..., nope_dim:][..., 0::2]
    odd = x[..., nope_dim:][..., 1::2]
    out[..., nope_dim:][..., 0::2] = even * cos + odd * sin
    out[..., nope_dim:][..., 1::2] = -even * sin + odd * cos
    return out


# ── KV Cache Operations ──────────────────────────────────────────────

def kv_quantize_fp8(kv_bf16):
    amax = kv_bf16.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    fp8_max = torch.tensor(448.0, dtype=torch.float32, device=kv_bf16.device)
    scale = fp8_max / amax
    kv_fp8 = (kv_bf16.float() * scale).to(torch.float8_e4m3fn)
    inv_scale = (amax / fp8_max).to(torch.bfloat16)
    return kv_fp8, inv_scale


def kv_dequantize_fp8(kv_fp8, inv_scale):
    return (kv_fp8.to(torch.bfloat16) * inv_scale).to(torch.bfloat16)


def paged_kv_write(kv_data, slot_mapping, cache, block_size):
    if cache.dtype == torch.uint8 and kv_data.dtype == torch.float8_e4m3fn:
        kv_to_write = kv_data.view(torch.uint8)
    else:
        kv_to_write = kv_data
    block_indices = slot_mapping // block_size
    offsets = slot_mapping % block_size
    valid = (block_indices < cache.shape[0]) & (offsets < cache.shape[1])
    if valid.all():
        cache[block_indices, offsets] = kv_to_write
    else:
        for t in range(kv_data.shape[0]):
            bi = block_indices[t].item()
            oi = offsets[t].item()
            if bi < cache.shape[0] and oi < cache.shape[1]:
                cache[bi, oi] = kv_to_write[t]


def paged_kv_read(slot_mapping, cache, block_size, num_tokens, head_dim):
    slots = slot_mapping
    block_indices = slots // block_size
    offsets = slots % block_size
    kv = cache[block_indices, offsets]
    if cache.dtype == torch.uint8:
        kv = kv.view(torch.float8_e4m3fn)
    return kv


# ── Attention ─────────────────────────────────────────────────────────

def causal_prefill_attention(q, kv, scale):
    T, NH, HD = q.shape
    q_t = q.permute(1, 0, 2)
    kv_exp = kv.unsqueeze(0).expand(NH, -1, -1)
    out = F.scaled_dot_product_attention(q_t, kv_exp, kv_exp, is_causal=True, scale=scale)
    return out.permute(1, 0, 2)


def decode_attention(q, kv, scale):
    NH = q.shape[1]
    HD = q.shape[2]
    q_t = q.permute(1, 0, 2)
    kv_exp = kv.unsqueeze(0).expand(NH, -1, -1)
    out = F.scaled_dot_product_attention(q_t, kv_exp, kv_exp, is_causal=False, scale=scale)
    return out.permute(1, 0, 2)


# ── Fused Q norm + RoPE + KV cache write ─────────────────────────────

def fused_qnorm_rope_kv_insert_py(
    q, kv, swa_kv_cache_2d, slot_mapping, positions,
    cos_sin_cache, eps, block_size,
    nope_dim=448, rope_dim=64,
) -> None:
    T = q.shape[0]
    if T == 0:
        return
    half = rope_dim // 2
    cos_q = cos_sin_cache[positions, :half].unsqueeze(1).to(q.dtype)
    sin_q = cos_sin_cache[positions, half:2*half].unsqueeze(1).to(q.dtype)
    q_rope = q[:, :, nope_dim:].clone()
    q[:, :, nope_dim:][:, :, 0::2] = q_rope[:, :, 0::2] * cos_q - q_rope[:, :, 1::2] * sin_q
    q[:, :, nope_dim:][:, :, 1::2] = q_rope[:, :, 0::2] * sin_q + q_rope[:, :, 1::2] * cos_q


def blackwell_attention_kv_write(
    kv, positions, swa_kv_cache, swa_inv_scale,
    slot_mapping, block_size, cos_sin_cache,
    nope_dim=448, rope_dim=64,
) -> None:
    T = kv.shape[0]
    if T == 0:
        return
    half = rope_dim // 2
    cos = cos_sin_cache[positions, :half].to(kv.dtype)
    sin = cos_sin_cache[positions, half:2 * half].to(kv.dtype)
    kv_rope_nope = kv[:, nope_dim:].clone()
    even = kv_rope_nope[:, 0::2]
    odd = kv_rope_nope[:, 1::2]
    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    kv_rope = kv.clone()
    kv_rope[:, nope_dim:][:, 0::2] = new_even
    kv_rope[:, nope_dim:][:, 1::2] = new_odd
    kv_fp8, inv_scale = kv_quantize_fp8(kv_rope)
    paged_kv_write(kv_fp8, slot_mapping, swa_kv_cache, block_size)
    for t in range(T):
        slot = slot_mapping[t].item()
        swa_inv_scale[slot] = inv_scale[t]


def blackwell_attention_decode(
    q, positions, swa_kv_cache, swa_inv_scale,
    slot_mapping, block_size, scale, window_size,
    swa_indices=None, swa_lens=None, decode_token_idx=0,
) -> torch.Tensor:
    """Legacy single-token decode — prefer native_swa_decode_attention."""
    from cutedsl.native_swa_decode import _fallback_batched_sdp

    # Wrap as batched call
    if swa_indices is not None and swa_lens is not None:
        si = swa_indices[decode_token_idx:decode_token_idx+1]
        sl = swa_lens[decode_token_idx:decode_token_idx+1]
    else:
        si = None
        sl = None

    return _fallback_batched_sdp(
        q, swa_kv_cache, swa_inv_scale, si, sl,
        block_size, scale, window_size,
    )


def full_sdpa_attention(q, kv, scale):
    return causal_prefill_attention(q, kv, scale)


# ── CSA/HCA Decode Attention ─────────────────────────────────────────

def blackwell_csa_decode_attention(
    q, positions, swa_kv_cache, swa_inv_scale,
    swa_metadata, flashmla_metadata, compressed_kv_cache,
    compress_ratio, scale, window_size, nope_dim, rope_dim,
    cos_sin_cache, attn_sink, max_model_len,
) -> torch.Tensor:
    """CSA/HCA decode: native sparse + SWA decode with sink weight merge.

    Uses the CuTeDSL sparse decode kernel that processes both SWA and
    compressed KV in a single pass with online softmax, then applies
    per-head sink weights.
    """
    num_tokens, NH, HD = q.shape
    device = q.device
    block_size = swa_metadata.block_size

    from cutedsl.native_sparse_decode import native_sparse_decode_attention

    # Get topk indices from metadata
    topk_indices = None
    topk_lens = None
    is_valid = swa_metadata.is_valid_token[:num_tokens]
    if flashmla_metadata is not None:
        comp_block_size = flashmla_metadata.block_size // compress_ratio
        if compress_ratio == 4:
            # C4A: need to compute global indices
            # This is handled by the caller (attention.py) which has access
            # to topk_indices_buffer. Fall back to SWA-only if not available.
            pass
        else:
            # C128A: pre-computed
            topk_indices = getattr(flashmla_metadata, "c128a_global_decode_topk_indices", None)
            topk_lens = getattr(flashmla_metadata, "c128a_decode_topk_lens", None)

    if topk_indices is None:
        # No sparse indices available — fall back to SWA-only
        from cutedsl.native_swa_decode import native_swa_decode_attention
        return native_swa_decode_attention(
            q, swa_kv_cache, swa_inv_scale,
            swa_metadata.decode_swa_indices[:num_tokens],
            swa_metadata.decode_swa_lens[:num_tokens],
            block_size, scale, window_size,
        )

    # Per-head inverse scale for compressed KV cache
    max_comp_slots = compressed_kv_cache.shape[0] * compressed_kv_cache.shape[1]
    comp_inv_scale = torch.ones(max_comp_slots, 1, dtype=torch.bfloat16, device=device)

    return native_sparse_decode_attention(
        q,
        swa_kv_cache,
        swa_inv_scale,
        swa_metadata.decode_swa_indices[:num_tokens],
        swa_metadata.decode_swa_lens[:num_tokens],
        compressed_kv_cache,
        comp_inv_scale,
        topk_indices,
        topk_lens,
        attn_sink,
        block_size,
        scale,
        window_size,
        compress_ratio=compress_ratio,
    )
