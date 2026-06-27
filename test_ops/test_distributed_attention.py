import torch
import torch.multiprocessing as mp

from chunkoptim.ops.distributed_attention import (
    combine_attention_outputs_from_lse,
    distributed_lse_merge,
    sanitize_empty_attention_output,
)


def test_lse_merge_matches_manual_weighted_sum():
    local_outputs = torch.tensor(
        [
            [[[[1.0, 2.0], [10.0, 20.0]],
              [[3.0, 4.0], [30.0, 40.0]]]],
            [[[[5.0, 6.0], [50.0, 60.0]],
              [[7.0, 8.0], [70.0, 80.0]]]],
        ],
        dtype=torch.bfloat16,
    )
    local_lses = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 3.0]]],
            [[[4.0, 5.0], [6.0, 7.0]]],
        ],
        dtype=torch.float32,
    )

    combined, global_lse = combine_attention_outputs_from_lse(
        local_outputs, local_lses)

    expected_lse = torch.logsumexp(local_lses, dim=0)
    weights = torch.exp(local_lses - expected_lse.unsqueeze(0))
    expected = (
        local_outputs.float()
        * weights.permute(0, 1, 3, 2).unsqueeze(-1)
    ).sum(dim=0)
    torch.testing.assert_close(global_lse, expected_lse)
    torch.testing.assert_close(combined.float(), expected, rtol=4e-3, atol=4e-3)


def test_lse_merge_ignores_empty_local_shards():
    local_outputs = torch.tensor(
        [
            [[[[float("nan"), float("nan")], [10.0, 20.0]]]],
            [[[[5.0, 6.0], [50.0, 60.0]]]],
        ],
        dtype=torch.float32,
    )
    local_lses = torch.tensor(
        [
            [[[-float("inf")], [2.0]]],
            [[[4.0], [6.0]]],
        ],
        dtype=torch.float32,
    )

    combined, global_lse = combine_attention_outputs_from_lse(
        local_outputs, local_lses)

    expected_lse = torch.logsumexp(local_lses, dim=0)
    torch.testing.assert_close(global_lse, expected_lse)
    torch.testing.assert_close(combined[0, 0, 0], torch.tensor([5.0, 6.0]))
    assert torch.isfinite(combined).all()


def test_sanitize_empty_attention_output_supports_stacked_and_unstacked():
    output = torch.ones(1, 2, 3, 4)
    lse = torch.tensor([[[0.0, -float("inf")],
                         [-float("inf"), 1.0],
                         [2.0, 3.0]]])
    cleaned = sanitize_empty_attention_output(output, lse)
    assert cleaned[0, 1, 0].abs().sum().item() == 0.0
    assert cleaned[0, 0, 1].abs().sum().item() == 0.0
    assert cleaned[0, 0, 0].abs().sum().item() > 0.0

    stacked = sanitize_empty_attention_output(output.unsqueeze(0), lse.unsqueeze(0))
    torch.testing.assert_close(stacked[0], cleaned)


def test_distributed_lse_merge_single_rank_uses_same_math():
    local_output = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    local_lse = torch.randn(1, 2, 3, dtype=torch.float32)

    combined, global_lse = distributed_lse_merge(
        local_output, local_lse, backend="allreduce", reduce_dtype=torch.float32)

    torch.testing.assert_close(combined.float(), local_output.float())
    torch.testing.assert_close(global_lse, local_lse)


def test_distributed_lse_merge_can_fallback_to_local_when_dist_uninitialized():
    local_output = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    local_lse = torch.randn(1, 2, 3, dtype=torch.float32)

    combined, global_lse = distributed_lse_merge(
        local_output, local_lse,
        backend="allreduce",
        reduce_dtype=torch.float32,
        fallback_to_local=True)

    torch.testing.assert_close(combined.float(), local_output.float())
    torch.testing.assert_close(global_lse, local_lse)


def test_distributed_lse_merge_rejects_unknown_backend():
    local_output = torch.randn(1, 1, 1, 1)
    local_lse = torch.randn(1, 1, 1)
    try:
        distributed_lse_merge(local_output, local_lse, backend="not-a-backend")
    except ValueError as exc:
        assert "unsupported distributed attention merge backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _distributed_lse_merge_worker(rank, world_size, init_file, queue):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size)
    try:
        local_output = torch.tensor(
            [[[[1.0 + rank, 2.0 + rank],
               [3.0 + rank, 4.0 + rank]]]],
            dtype=torch.float32)
        local_lse = torch.tensor(
            [[[float(rank), float(rank + 1)]]],
            dtype=torch.float32)
        combined, global_lse = distributed_lse_merge(
            local_output, local_lse, backend="allreduce")

        stacked_outputs = torch.stack([
            torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32),
            torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]], dtype=torch.float32),
        ])
        stacked_lses = torch.stack([
            torch.tensor([[[0.0, 1.0]]], dtype=torch.float32),
            torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
        ])
        expected, expected_lse = combine_attention_outputs_from_lse(
            stacked_outputs, stacked_lses)
        torch.testing.assert_close(combined, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(global_lse, expected_lse, rtol=1e-6, atol=1e-6)
        queue.put("ok")
    finally:
        dist.destroy_process_group()


def test_distributed_lse_merge_allreduce_matches_reference_across_ranks(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    init_file = tmp_path / "dist_init"
    processes = [
        ctx.Process(
            target=_distributed_lse_merge_worker,
            args=(rank, 2, str(init_file), queue))
        for rank in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    exitcodes = [process.exitcode for process in processes]
    assert exitcodes == [0, 0]
    assert [queue.get(timeout=5) for _ in processes] == ["ok", "ok"]
