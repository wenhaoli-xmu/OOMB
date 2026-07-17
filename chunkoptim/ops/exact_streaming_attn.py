import math

import torch


def _validate_attention_inputs(query, key, value, query_block_size, kv_block_size):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be shaped [batch, seq, heads, dim]")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key/value batch sizes must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key/value head dimensions must match")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if query_block_size < 1:
        raise ValueError("query_block_size must be >= 1")
    if kv_block_size < 1:
        raise ValueError("kv_block_size must be >= 1")


def _expand_gqa(tensor, num_query_heads):
    groups = num_query_heads // tensor.shape[2]
    if groups == 1:
        return tensor
    return tensor.repeat_interleave(groups, dim=2)


def rectangular_streaming_attention(
        query, key, value, *, query_block_size=128, kv_block_size=128,
        causal=True, q_start=0, kv_start=0, scale=None, return_lse=False):
    """Exact dense attention computed as rectangular KV tiles.

    The implementation keeps per-query online-softmax state `(m, l, acc)` and
    never materializes the full score matrix. It is intentionally written in
    PyTorch so the math is easy to validate and differentiable; production
    kernels can later reuse this API contract.
    """
    _validate_attention_inputs(
        query, key, value, query_block_size, kv_block_size)
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    batch, q_len, num_heads, head_dim = query.shape
    kv_len = key.shape[1]
    out_blocks = []
    lse_blocks = []
    key_expanded = _expand_gqa(key, num_heads)
    value_expanded = _expand_gqa(value, num_heads)

    for q0 in range(0, q_len, query_block_size):
        q1 = min(q0 + query_block_size, q_len)
        q_blk = query[:, q0:q1]
        q_blk_len = q1 - q0
        m = torch.full(
            (batch, num_heads, q_blk_len), -float("inf"),
            device=query.device, dtype=torch.float32)
        l = torch.zeros(
            (batch, num_heads, q_blk_len),
            device=query.device, dtype=torch.float32)
        acc = torch.zeros(
            (batch, q_blk_len, num_heads, head_dim),
            device=query.device, dtype=torch.float32)

        q_pos = None
        if causal:
            q_pos = torch.arange(
                q_start + q0, q_start + q1,
                device=query.device, dtype=torch.long)

        for k0 in range(0, kv_len, kv_block_size):
            k1 = min(k0 + kv_block_size, kv_len)
            k_blk = key_expanded[:, k0:k1]
            v_blk = value_expanded[:, k0:k1]
            scores = torch.einsum(
                "bqhd,bkhd->bhqk", q_blk.float(), k_blk.float())
            scores = scores * scale

            if causal:
                kv_pos = torch.arange(
                    kv_start + k0, kv_start + k1,
                    device=query.device, dtype=torch.long)
                allowed = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
                scores = scores.masked_fill(
                    ~allowed.unsqueeze(0).unsqueeze(0), -float("inf"))

            tile_has_value = torch.isfinite(scores).any(dim=-1)
            tile_m = torch.max(scores, dim=-1).values
            tile_m_safe = torch.where(
                tile_has_value, tile_m, torch.zeros_like(tile_m))
            exp_scores = torch.exp(scores - tile_m_safe.unsqueeze(-1))
            exp_scores = torch.where(
                torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))
            tile_l = exp_scores.sum(dim=-1)
            tile_acc = torch.einsum(
                "bhqk,bkhd->bqhd", exp_scores, v_blk.float())

            new_m = torch.maximum(m, tile_m)
            new_valid = torch.isfinite(new_m)
            m_scale = torch.where(
                torch.isfinite(m),
                torch.exp(m - torch.where(new_valid, new_m, torch.zeros_like(new_m))),
                torch.zeros_like(m),
            )
            tile_scale = torch.where(
                tile_has_value,
                torch.exp(tile_m - torch.where(new_valid, new_m, torch.zeros_like(new_m))),
                torch.zeros_like(tile_m),
            )
            acc = (
                acc * m_scale.permute(0, 2, 1).unsqueeze(-1)
                + tile_acc * tile_scale.permute(0, 2, 1).unsqueeze(-1)
            )
            l = l * m_scale + tile_l * tile_scale
            m = new_m

        denom = l.clamp_min(1e-30)
        out = acc / denom.permute(0, 2, 1).unsqueeze(-1)
        valid = l > 0
        out = torch.where(
            valid.permute(0, 2, 1).unsqueeze(-1),
            out,
            torch.zeros_like(out),
        )
        out_blocks.append(out.to(query.dtype))

        lse = torch.where(valid, m + torch.log(denom), torch.full_like(m, -float("inf")))
        lse_blocks.append(lse)

    out = torch.cat(out_blocks, dim=1)
    if not return_lse:
        return out
    return out, torch.cat(lse_blocks, dim=-1).to(torch.float32)


def materialize_paged_kv(manager):
    """Concatenate a paged KV cache into contiguous `[B, K, H, D]` tensors."""
    if manager.num_kv < 1:
        raise ValueError("manager must contain at least one KV token")
    if len(manager.key_gpu) != len(manager.val_gpu):
        raise ValueError("manager key/value page counts differ")
    if not manager.key_gpu:
        raise ValueError("manager has no resident KV pages")
    key = torch.cat(manager.key_gpu, dim=1)[:, :manager.num_kv].contiguous()
    value = torch.cat(manager.val_gpu, dim=1)[:, :manager.num_kv].contiguous()
    return key, value


class PagedRectangularAttention(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, query, key_current, value_current, manager,
            query_block_size, kv_block_size):
        key_full, value_full = materialize_paged_kv(manager)
        query_detached = query.detach().requires_grad_(True)
        key_full_detached = key_full.detach().requires_grad_(True)
        value_full_detached = value_full.detach().requires_grad_(True)
        with torch.enable_grad():
            output = rectangular_streaming_attention(
                query_detached, key_full_detached, value_full_detached,
                query_block_size=query_block_size,
                kv_block_size=kv_block_size,
                causal=True,
                q_start=manager.num_kv - query.shape[1],
                kv_start=0,
            )
        ctx.save_for_backward(
            query_detached, key_full_detached, value_full_detached, output)
        ctx.current_tokens = key_current.shape[1]
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        query, key_full, value_full, output = ctx.saved_tensors
        dq, dk_full, dv_full = torch.autograd.grad(
            output,
            (query, key_full, value_full),
            grad_output,
            retain_graph=False,
            allow_unused=False,
        )
        current = ctx.current_tokens
        dk_current = dk_full[:, -current:].to(dtype=grad_output.dtype)
        dv_current = dv_full[:, -current:].to(dtype=grad_output.dtype)
        return dq.to(dtype=grad_output.dtype), dk_current, dv_current, None, None, None


def paged_rectangular_attention(
        query, key_current, value_current, manager, *,
        query_block_size=128, kv_block_size=128):
    """Autograd-compatible rectangular exact attention over a paged manager."""
    return PagedRectangularAttention.apply(
        query, key_current, value_current, manager,
        int(query_block_size), int(kv_block_size))
