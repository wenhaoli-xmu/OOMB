import math

import torch

from chunkoptim.ops.ring_context_parallel import (
    finalize_online_attention,
    init_online_attention_state,
    online_attention_update,
    ring_kv_owner,
)


def dense_attention(q, k, v, q_start, k_start, causal=True):
    if q.shape[2] != k.shape[2]:
        rep = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float())
    scores = scores / math.sqrt(q.shape[-1])
    if causal:
        q_pos = torch.arange(q_start, q_start + q.shape[1], device=q.device)
        k_pos = torch.arange(k_start, k_start + k.shape[1], device=q.device)
        scores = scores.masked_fill(
            k_pos.view(1, 1, 1, -1) > q_pos.view(1, 1, -1, 1),
            -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    return torch.einsum("bhqk,bkhd->bqhd", probs, v.float())


def test_online_attention_update_matches_dense_attention_for_gqa():
    torch.manual_seed(0)
    q = torch.randn(1, 3, 4, 5, dtype=torch.bfloat16)
    k0 = torch.randn(1, 4, 2, 5, dtype=torch.bfloat16)
    v0 = torch.randn(1, 4, 2, 5, dtype=torch.bfloat16)
    k1 = torch.randn(1, 5, 2, 5, dtype=torch.bfloat16)
    v1 = torch.randn(1, 5, 2, 5, dtype=torch.bfloat16)

    state = init_online_attention_state(
        batch_size=1, q_len=3, num_heads=4, head_dim=5,
        device=q.device, dtype=torch.float32)
    state = online_attention_update(state, q, k0, v0, q_start=6, k_start=0)
    state = online_attention_update(state, q, k1, v1, q_start=6, k_start=4)
    out, lse = finalize_online_attention(state, output_dtype=torch.float32)

    ref = dense_attention(
        q, torch.cat([k0, k1], dim=1), torch.cat([v0, v1], dim=1),
        q_start=6, k_start=0)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    assert torch.isfinite(lse).all()


def test_online_attention_update_keeps_empty_rows_finite():
    q = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    state = init_online_attention_state(
        batch_size=1, q_len=2, num_heads=2, head_dim=4,
        device=q.device, dtype=torch.float32)

    state = online_attention_update(state, q, k, v, q_start=0, k_start=5)
    out, lse = finalize_online_attention(state, output_dtype=torch.float32)

    assert torch.equal(out, torch.zeros_like(out))
    assert torch.equal(lse, torch.full_like(lse, -float("inf")))


def test_ring_kv_owner_rotates_from_left_neighbor():
    assert [ring_kv_owner(rank=3, step=i, world_size=8) for i in range(8)] == [
        3, 2, 1, 0, 7, 6, 5, 4]
