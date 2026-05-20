"""Batched SWA decode attention for Blackwell.

Replaces the per-token Python for-loop with a single batched operation.
Each decode token attends to its SWA window (up to window_size KV entries).
We gather all KV at once using the pre-computed swa_indices, pad to uniform
length, and use flash attention with a mask.
"""
import torch
import torch.nn.functional as F


def blackwell_batched_swa_decode(
    q,               # (num_decode_tokens, NH, HD) with RoPE
    positions,       # (num_decode_tokens,)
    swa_kv_cache,    # (num_blocks, block_size, D) fp8 SWA cache (uint8)
    swa_inv_scale,   # (max_slots, 1) per-token inv scale
    swa_metadata,    # DeepseekSparseSWAMetadata
    block_size,      # tokens per block
    scale,           # 1/sqrt(HD)
    window_size,     # 128
) -> torch.Tensor:
    """Batched SWA decode attention.
    
    Instead of a Python for-loop per token, we:
    1. Gather ALL KV entries for ALL decode tokens at once using swa_indices
    2. Dequantize fp8 -> bf16 in one batch
    3. Run batched SDPA with proper masking
    
    Returns: (num_decode_tokens, NH, HD)
    """
    num_tokens, NH, HD = q.shape
    device = q.device
    
    # swa_indices: (num_decode_tokens, window_size) - pre-computed paged slot indices
    # swa_lens: (num_decode_tokens,) - number of valid entries per token
    swa_indices = swa_metadata.decode_swa_indices[:num_tokens]  # (T, W)
    swa_lens = swa_metadata.decode_swa_lens[:num_tokens]  # (T,)
    max_len = window_size  # All padded to window_size
    
    # Gather KV from paged cache using swa_indices
    # swa_indices may contain -1 (invalid), clamp to 0 for safe indexing
    safe_indices = swa_indices.clamp(min=0)  # (T, W)
    block_indices = safe_indices // block_size  # (T, W)
    offsets = safe_indices % block_size  # (T, W)
    
    # Read KV from cache: (T, W, D)
    kv_raw = swa_kv_cache[block_indices, offsets]  # (T, W, D)
    if swa_kv_cache.dtype == torch.uint8:
        kv_raw = kv_raw.view(torch.float8_e4m3fn)
    
    # Gather inv_scales: (T, W, 1)
    inv_scales = swa_inv_scale[safe_indices]  # (T, W, 1)
    
    # Dequantize fp8 -> bf16
    kv_bf16 = (kv_raw.to(torch.bfloat16) * inv_scales).to(torch.bfloat16)  # (T, W, D)
    
    # Build attention mask: True = masked out (invalid)
    # Invalid where swa_indices < 0 or beyond swa_lens
    positions_in_window = torch.arange(max_len, device=device).unsqueeze(0)  # (1, W)
    len_mask = positions_in_window >= swa_lens.unsqueeze(1)  # (T, W) - True where beyond valid
    invalid_mask = swa_indices < 0  # (T, W) - True where slot is -1
    attn_mask = len_mask | invalid_mask  # (T, W) - True = ignore
    
    # For SDPA: q is (T, NH, HD), kv_bf16 is (T, W, D)
    # We need to reshape for batched attention:
    # q: (NH, T, HD), k=v=kv_bf16: (NH, T, W, HD)
    q_t = q.permute(1, 0, 2)  # (NH, T, HD)
    kv_expanded = kv_bf16.unsqueeze(0).expand(NH, -1, -1, -1)  # (NH, T, W, HD)
    
    # Expand mask for all heads: (NH, T, W) 
    # SDPA expects mask shape compatible with (..., T, W)
    # For SDPA with is_causal=False, attn_mask shape should be (T, W) broadcastable
    # Actually, let's use the mask directly with SDPA
    # SDPA mask: True = allow, False = block (when using attn_mask)
    # But PyTorch SDPA's attn_mask convention varies. Let's use the additive mask approach.
    
    # Create additive mask: 0 for valid, -inf for invalid
    float_mask = torch.zeros(attn_mask.shape, dtype=torch.bfloat16, device=device)
    float_mask[attn_mask] = float('-inf')  # (T, W)
    
    # Expand for heads: (NH, T, W) -> needs to be (NH * T, 1, W) for batched SDPA
    # Actually, use the (NH, T, 1, W) shape for expand
    float_mask_exp = float_mask.unsqueeze(0).unsqueeze(2)  # (1, T, 1, W)
    
    # Reshape q and kv for batched SDPA
    # q: (NH, T, HD) -> (NH*T, 1, HD)
    # kv: (NH, T, W, HD) -> (NH*T, W, HD)
    q_batch = q_t.reshape(NH * num_tokens, 1, HD)
    k_batch = kv_expanded.reshape(NH * num_tokens, max_len, HD)
    v_batch = k_batch  # K = V in MLA/CSA
    
    # Expand mask: (1, T, 1, W) -> (NH*T, 1, W)
    mask_batch = float_mask_exp.expand(NH, num_tokens, 1, max_len).reshape(NH * num_tokens, 1, max_len)
    
    # Batched SDPA
    out = F.scaled_dot_product_attention(
        q_batch, k_batch, v_batch,
        attn_mask=mask_batch,
        is_causal=False,
        scale=scale,
    )  # (NH*T, 1, HD)
    
    # Reshape back: (NH*T, 1, HD) -> (NH, T, HD) -> (T, NH, HD)
    out = out.reshape(NH, num_tokens, HD).permute(1, 0, 2)
    
    return out


def blackwell_batched_swa_decode(
    q,               # (num_decode_tokens, NH, HD) with RoPE
    positions,       # (num_decode_tokens,)
    swa_kv_cache,    # (num_blocks, block_size, D) fp8 SWA cache (uint8)
    swa_inv_scale,   # (max_slots, 1) per-token inv scale
    swa_metadata,    # DeepseekSparseSWAMetadata
    block_size,      # tokens per block
    scale,           # 1/sqrt(HD)
    window_size,     # 128
) -> torch.Tensor:
    """Batched SWA decode attention — NO Python for-loop.
    
    All decode tokens are processed in a single batched SDPA call:
    1. Gather ALL KV entries for ALL decode tokens at once using swa_indices
    2. Dequantize fp8 -> bf16 in one batch
    3. Run batched SDPA with proper masking
    """
    num_tokens, NH, HD = q.shape
    device = q.device
    
    swa_indices = swa_metadata.decode_swa_indices[:num_tokens]  # (T, W)
    swa_lens = swa_metadata.decode_swa_lens[:num_tokens]  # (T,)
    max_len = window_size
    
    # Clamp invalid indices to 0 for safe gather
    safe_indices = swa_indices.clamp(min=0)
    block_indices = safe_indices // block_size
    offsets = safe_indices % block_size
    
    # Batched KV read from paged cache
    kv_raw = swa_kv_cache[block_indices, offsets]  # (T, W, D)
    if swa_kv_cache.dtype == torch.uint8:
        kv_raw = kv_raw.view(torch.float8_e4m3fn)
    
    # Batched dequantize
    inv_scales = swa_inv_scale[safe_indices]  # (T, W, 1)
    kv_bf16 = (kv_raw.to(torch.bfloat16) * inv_scales).to(torch.bfloat16)
    
    # Attention mask: -inf for invalid positions
    positions_in_window = torch.arange(max_len, device=device).unsqueeze(0)
    len_mask = positions_in_window >= swa_lens.unsqueeze(1)
    invalid_mask = swa_indices < 0
    attn_mask = len_mask | invalid_mask  # (T, W) True = ignore
    
    float_mask = torch.zeros(attn_mask.shape, dtype=torch.bfloat16, device=device)
    float_mask[attn_mask] = float('-inf')
    
    # Reshape for batched SDPA:
    # q: (T, NH, HD) -> (NH*T, 1, HD)
    # kv: (T, W, HD) -> (NH*T, W, HD)  (K=V in MLA)
    q_t = q.permute(1, 0, 2)  # (NH, T, HD)
    q_batch = q_t.reshape(NH * num_tokens, 1, HD)
    
    kv_expanded = kv_bf16.unsqueeze(0).expand(NH, -1, -1, -1)  # (NH, T, W, HD)
    k_batch = kv_expanded.reshape(NH * num_tokens, max_len, HD)
    v_batch = k_batch  # K = V in MLA
    
    mask_batch = float_mask.unsqueeze(0).unsqueeze(2).expand(
        NH, num_tokens, 1, max_len
    ).reshape(NH * num_tokens, 1, max_len)
    
    out = F.scaled_dot_product_attention(
        q_batch, k_batch, v_batch,
        attn_mask=mask_batch,
        is_causal=False,
        scale=scale,
    )
    
    return out.reshape(NH, num_tokens, HD).permute(1, 0, 2)
