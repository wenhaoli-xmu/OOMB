import math

import torch

from chunkoptim.cache.kv_cache import SimpleCacheManager
from chunkoptim.ops.cp_transport import (
    rectangular_kv_schedule,
    ring_kv_schedule,
    usp_kv_schedule,
)
from chunkoptim.ops.exact_streaming_attn import (
    materialize_paged_kv,
    paged_rectangular_attention,
    rectangular_streaming_attention,
)


def dense_attention_reference(query, key, value, *, causal=True,
                              q_start=0, kv_start=0):
    num_heads = query.shape[2]
    num_kv_heads = key.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by kv heads")
    groups = num_heads // num_kv_heads
    if groups != 1:
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)

    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    scores = scores / math.sqrt(query.shape[-1])
    if causal:
        q_pos = torch.arange(
            q_start, q_start + query.shape[1], device=query.device)
        kv_pos = torch.arange(
            kv_start, kv_start + key.shape[1], device=query.device)
        allowed = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(
            ~allowed.unsqueeze(0).unsqueeze(0), -float("inf"))

    valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    out = torch.einsum("bhqk,bkhd->bqhd", probs, value.float())
    return out.to(query.dtype)


def test_rectangular_streaming_attention_matches_dense_causal_gqa():
    torch.manual_seed(0)
    query = torch.randn(2, 5, 4, 8)
    key = torch.randn(2, 7, 2, 8)
    value = torch.randn(2, 7, 2, 8)

    out, lse = rectangular_streaming_attention(
        query, key, value,
        query_block_size=2,
        kv_block_size=3,
        causal=True,
        q_start=3,
        kv_start=0,
        return_lse=True,
    )

    ref = dense_attention_reference(
        query, key, value, causal=True, q_start=3, kv_start=0)
    assert out.shape == query.shape
    assert lse.shape == (2, 4, 5)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_rectangular_streaming_attention_handles_all_masked_rows():
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)

    out, lse = rectangular_streaming_attention(
        query, key, value,
        query_block_size=1,
        kv_block_size=2,
        causal=True,
        q_start=0,
        kv_start=5,
        return_lse=True,
    )

    assert torch.equal(out, torch.zeros_like(out))
    assert torch.isneginf(lse).all()


def test_rectangular_streaming_attention_backward_matches_dense():
    torch.manual_seed(1)
    query = torch.randn(1, 4, 2, 5, requires_grad=True)
    key = torch.randn(1, 6, 2, 5, requires_grad=True)
    value = torch.randn(1, 6, 2, 5, requires_grad=True)
    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)

    out = rectangular_streaming_attention(
        query, key, value,
        query_block_size=3,
        kv_block_size=4,
        causal=True,
        q_start=2,
        kv_start=0,
    )
    ref = dense_attention_reference(
        query_ref, key_ref, value_ref, causal=True, q_start=2, kv_start=0)

    out.square().sum().backward()
    ref.square().sum().backward()

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(query.grad, query_ref.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(key.grad, key_ref.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(value.grad, value_ref.grad, atol=2e-5, rtol=2e-5)


def test_rectangular_kv_schedule_covers_non_divisible_range_once():
    blocks = rectangular_kv_schedule(num_kv=10, kv_block_size=4)

    assert [(b.start, b.end, b.backend, b.owner_rank, b.hop) for b in blocks] == [
        (0, 4, "rectangular", None, 0),
        (4, 8, "rectangular", None, 0),
        (8, 10, "rectangular", None, 0),
    ]


def test_ring_kv_schedule_orders_blocks_by_ring_hop():
    blocks = ring_kv_schedule(
        num_kv=10, kv_block_size=3, world_size=3, rank=1)

    assert [(b.start, b.end, b.owner_rank, b.hop, b.backend) for b in blocks] == [
        (3, 6, 1, 0, "ring"),
        (0, 3, 0, 1, "ring"),
        (9, 10, 0, 1, "ring"),
        (6, 9, 2, 2, "ring"),
    ]


def test_usp_kv_schedule_preserves_global_block_order_with_owners():
    blocks = usp_kv_schedule(
        num_kv=10, kv_block_size=3, world_size=3, rank=1)

    assert [(b.start, b.end, b.owner_rank, b.hop, b.backend) for b in blocks] == [
        (0, 3, 0, 1, "usp"),
        (3, 6, 1, 0, "usp"),
        (6, 9, 2, 1, "usp"),
        (9, 10, 0, 1, "usp"),
    ]


def test_materialize_paged_kv_trims_padding_tokens():
    manager = SimpleCacheManager(
        batch_size=1, page_size=4, num_kv_heads=1, head_dim=2, local_rank=0)
    key = torch.arange(12, dtype=torch.bfloat16).view(1, 6, 1, 2)
    value = key + 100
    manager.update(key, value, stage=1)

    key_full, value_full = materialize_paged_kv(manager)

    assert key_full.shape == (1, 6, 1, 2)
    assert value_full.shape == (1, 6, 1, 2)
    assert torch.equal(key_full, key)
    assert torch.equal(value_full, value)


def test_cache_manager_exposes_global_page_indices_for_indexed_attention():
    manager = SimpleCacheManager(
        batch_size=1, page_size=2, num_kv_heads=1, head_dim=2, local_rank=0)
    key = torch.arange(12, dtype=torch.bfloat16).view(1, 6, 1, 2)
    manager.update(key, key + 100, stage=1)

    page_indices = manager.page_indices_tensor(device="cpu")

    assert page_indices.dtype == torch.int64
    assert page_indices.tolist() == [0, 1, 2]

    autograd_indices = manager.page_indices_tensor(
        device="cpu", for_autograd=True)
    assert not torch.is_inference(autograd_indices)
    assert autograd_indices.tolist() == [0, 1, 2]


def test_paged_rectangular_attention_returns_current_kv_grad_from_manager():
    torch.manual_seed(2)
    manager = SimpleCacheManager(
        batch_size=1, page_size=2, num_kv_heads=1, head_dim=4, local_rank=0)
    key = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 4, 1, 4, dtype=torch.bfloat16)
    manager.update(key, value, stage=1)
    query = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    key_cur = key[:, 2:].detach().clone().requires_grad_(True)
    value_cur = value[:, 2:].detach().clone().requires_grad_(True)
    query_ref = query.detach().float().requires_grad_(True)
    key_ref = key.detach().float().requires_grad_(True)
    value_ref = value.detach().float().requires_grad_(True)

    out = paged_rectangular_attention(
        query, key_cur, value_cur, manager,
        query_block_size=1,
        kv_block_size=2,
    )
    ref = dense_attention_reference(
        query_ref, key_ref, value_ref, q_start=2, kv_start=0)

    out.float().square().sum().backward()
    ref.square().sum().backward()

    assert torch.allclose(out.float(), ref, atol=5e-3, rtol=5e-3)
    assert torch.allclose(query.grad.float(), query_ref.grad, atol=5e-3, rtol=5e-3)
    assert torch.allclose(key_cur.grad.float(), key_ref.grad[:, 2:],
                          atol=5e-3, rtol=5e-3)
    assert torch.allclose(value_cur.grad.float(), value_ref.grad[:, 2:],
                          atol=5e-3, rtol=5e-3)
