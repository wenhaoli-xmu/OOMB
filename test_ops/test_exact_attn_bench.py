import math

import torch

from chunkoptim.ops.exact_attn_bench import run_rectangular_attention_benchmark


def test_rectangular_attention_benchmark_reports_timing_and_error():
    result = run_rectangular_attention_benchmark(
        batch_size=1,
        q_len=3,
        kv_len=5,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        query_block_size=2,
        kv_block_size=2,
        q_start=2,
        device="cpu",
        dtype=torch.float32,
        seed=123,
        warmup=0,
        repeat=1,
        check_dense=True,
    )

    assert result["backend"] == "rectangular_exact"
    assert result["device"] == "cpu"
    assert result["dtype"] == "torch.float32"
    assert result["shape"] == {
        "batch_size": 1,
        "q_len": 3,
        "kv_len": 5,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
    }
    assert result["tiles"] == {
        "query_block_size": 2,
        "kv_block_size": 2,
        "query_blocks": 2,
        "kv_blocks": 3,
        "tile_count": 6,
    }
    assert result["score_cells"] == 1 * 2 * 3 * 5
    assert result["timing_s"]["rectangular_forward_avg"] >= 0.0
    assert result["timing_s"]["materialize_avg"] >= 0.0
    assert result["dense_check"]["enabled"] is True
    assert result["dense_check"]["max_abs_error"] < 1e-5
    assert result["dense_check"]["max_rel_error"] < 1e-5
    assert math.isfinite(result["dense_check"]["dense_forward_s"])
