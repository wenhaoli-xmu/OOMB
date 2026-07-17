import torch

from chunkoptim.modifiers.blockwise_attention import (
    resolve_attention_config,
    run_blockwise_attention,
)
from chunkoptim.cache.kv_cache import KVCache


class DummyManager:
    def __init__(self):
        self.called = None


def test_resolve_attention_config_defaults_to_paged():
    manager = DummyManager()

    config = resolve_attention_config(manager)

    assert config["backend"] == "paged"
    assert config["merge_backend"] == "allreduce"
    assert config["reduce_dtype"] == torch.float32


def test_resolve_attention_config_accepts_manager_config_aliases():
    manager = DummyManager()
    manager.attention_conf = {
        "backend": "distributed-paged",
        "merge_backend": "allgather_ref",
        "reduce_dtype": "bf16",
    }

    config = resolve_attention_config(manager)

    assert config["backend"] == "distributed_paged"
    assert config["merge_backend"] == "allgather_ref"
    assert config["reduce_dtype"] == torch.bfloat16


def test_run_blockwise_attention_dispatches_paged_backend(monkeypatch):
    calls = []

    def fake_paged(q, k, v, manager):
        calls.append(("paged", q, k, v, manager))
        return q + 1

    monkeypatch.setattr(
        "chunkoptim.modifiers.blockwise_attention.flash_paged_attn_func",
        fake_paged)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    v = torch.zeros(1, 2, 1, 4)
    manager = DummyManager()

    out = run_blockwise_attention(q, k, v, manager)

    assert torch.equal(out, torch.ones_like(q))
    assert calls == [("paged", q, k, v, manager)]


def test_resolve_attention_config_parses_string_booleans():
    manager = DummyManager()
    manager.attention_conf = {
        "backend": "distributed_paged",
        "fallback_to_local": "false",
    }

    config = resolve_attention_config(manager)

    assert config["fallback_to_local"] is False


def test_run_blockwise_attention_dispatches_distributed_backend(monkeypatch):
    calls = []

    def fake_distributed(
            q, k, v, manager, group, reduce_dtype, merge_backend,
            fallback_to_local):
        calls.append((
            q, k, v, manager, group, reduce_dtype, merge_backend,
            fallback_to_local))
        return q + 2

    monkeypatch.setattr(
        "chunkoptim.modifiers.blockwise_attention.flash_paged_attn_distributed_func",
        fake_distributed)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    v = torch.zeros(1, 2, 1, 4)
    manager = DummyManager()
    manager.attention_conf = {
        "backend": "distributed_paged",
        "merge_backend": "allgather_ref",
        "reduce_dtype": torch.bfloat16,
    }

    out = run_blockwise_attention(q, k, v, manager)

    assert torch.equal(out, torch.full_like(q, 2))
    assert calls == [(
        q, k, v, manager, None, torch.bfloat16, "allgather_ref", True)]


def test_run_blockwise_attention_passes_distributed_fallback(monkeypatch):
    calls = []

    def fake_distributed(
            q, k, v, manager, group, reduce_dtype, merge_backend,
            fallback_to_local):
        calls.append(fallback_to_local)
        return q + 4

    monkeypatch.setattr(
        "chunkoptim.modifiers.blockwise_attention.flash_paged_attn_distributed_func",
        fake_distributed)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    v = torch.zeros(1, 2, 1, 4)
    manager = DummyManager()
    manager.attention_conf = {"backend": "distributed_paged"}

    out = run_blockwise_attention(q, k, v, manager)

    assert torch.equal(out, torch.full_like(q, 4))
    assert calls == [True]


def test_run_blockwise_attention_dispatches_rectangular_reference(monkeypatch):
    calls = []

    def fake_rectangular(q, k, v, manager, query_block_size, kv_block_size):
        calls.append((q, k, v, manager, query_block_size, kv_block_size))
        return q + 3

    monkeypatch.setattr(
        "chunkoptim.modifiers.blockwise_attention.paged_rectangular_attention",
        fake_rectangular)
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    v = torch.zeros(1, 2, 1, 4)
    manager = DummyManager()
    manager.attention_conf = {
        "backend": "rectangular",
        "query_block_size": 4,
        "kv_block_size": 8,
    }

    out = run_blockwise_attention(q, k, v, manager)

    assert torch.equal(out, torch.full_like(q, 3))
    assert calls == [(q, k, v, manager, 4, 8)]


def test_run_blockwise_attention_rejects_unknown_backend():
    q = torch.zeros(1, 2, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    v = torch.zeros(1, 2, 1, 4)
    manager = DummyManager()
    manager.attention_conf = {"backend": "bad"}

    try:
        run_blockwise_attention(q, k, v, manager)
    except ValueError as exc:
        assert "unsupported blockwise attention backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_kv_cache_attaches_attention_config_to_managers():
    cache = KVCache(
        num_layers=2,
        batch_size=1,
        page_size=2,
        num_heads=1,
        head_dim=4,
        cpu_offload=None,
        local_rank=0,
        attention_conf={
            "backend": "distributed_paged",
            "merge_backend": "allgather_ref",
        },
    )

    assert cache.attention_conf["backend"] == "distributed_paged"
    assert [m.attention_conf for m in cache.managers] == [
        cache.attention_conf,
        cache.attention_conf,
    ]
