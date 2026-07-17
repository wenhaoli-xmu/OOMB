"""Exact distributed attention output merge helpers.

The local paged attention kernel returns a normalized output for the KV shard
owned by one rank plus that shard's row-wise log-sum-exp. These helpers merge
those shard-local results into the exact dense-attention output without moving
KV blocks between ranks.
"""
from __future__ import annotations

import torch


def sanitize_empty_attention_output(local_output: torch.Tensor,
                                    local_lse: torch.Tensor) -> torch.Tensor:
    """Zero rows whose local causal shard is empty.

    Empty rows have local LSE = -inf. Some kernels may leave their corresponding
    output values undefined, so they must be masked before finite checks or
    weighted sums.
    """
    empty = torch.isneginf(local_lse.float())
    if local_output.ndim == 4:
        empty = empty.permute(0, 2, 1).unsqueeze(-1)
    elif local_output.ndim == 5:
        empty = empty.permute(0, 1, 3, 2).unsqueeze(-1)
    else:
        raise ValueError(
            "local_output must be shaped [batch, query, heads, dim] "
            "or [shard, batch, query, heads, dim]")
    return torch.where(empty, torch.zeros_like(local_output), local_output)


def combine_attention_outputs_from_lse(
        local_outputs: torch.Tensor,
        local_lses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable exact merge of normalized local attention outputs.

    Args:
        local_outputs: [shard, batch, query, heads, dim]
        local_lses: [shard, batch, heads, query]

    Returns:
        (global_output, global_lse), where global_output is normalized by the
        full KV set represented by all shards.
    """
    if local_outputs.ndim != 5:
        raise ValueError(
            "local_outputs must be shaped [shard, batch, query, heads, dim]")
    if local_lses.ndim != 4:
        raise ValueError(
            "local_lses must be shaped [shard, batch, heads, query]")
    if local_outputs.shape[0] != local_lses.shape[0]:
        raise ValueError(
            "local_outputs and local_lses must have same shard count")

    local_lses_f = local_lses.float()
    global_lse = torch.logsumexp(local_lses_f, dim=0)
    weights = torch.exp(local_lses_f - global_lse.unsqueeze(0))
    weights = torch.where(
        torch.isneginf(local_lses_f), torch.zeros_like(weights), weights)
    weights = weights.permute(0, 1, 3, 2).unsqueeze(-1)
    local_outputs_f = sanitize_empty_attention_output(
        local_outputs.float(), local_lses_f)
    combined = (local_outputs_f * weights).sum(dim=0)
    return combined.to(local_outputs.dtype), global_lse.to(local_lses.dtype)


def distributed_lse_merge_allreduce(
        local_output: torch.Tensor,
        local_lse: torch.Tensor,
        group=None,
        reduce_dtype: torch.dtype = torch.float32,
        fallback_to_local: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one rank's local attention shard via exact LSE all-reduces."""
    if reduce_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("reduce_dtype must be torch.float32 or torch.bfloat16")

    import torch.distributed as dist

    local_lse_f = local_lse.float()
    if fallback_to_local and (
            (not dist.is_available()) or (not dist.is_initialized())):
        return combine_attention_outputs_from_lse(
            local_output.unsqueeze(0), local_lse_f.unsqueeze(0))

    max_lse = local_lse_f.clone()
    dist.all_reduce(max_lse, op=dist.ReduceOp.MAX, group=group)
    shifted_exp = torch.exp(local_lse_f - max_lse)
    shifted_exp = torch.where(
        torch.isneginf(local_lse_f), torch.zeros_like(shifted_exp), shifted_exp)
    output = sanitize_empty_attention_output(local_output.float(), local_lse_f)
    numerator = (
        output * shifted_exp.permute(0, 2, 1).unsqueeze(-1)
    ).to(reduce_dtype).contiguous()
    denominator = shifted_exp.contiguous()
    dist.all_reduce(numerator, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    denominator = denominator.clamp_min(1e-30)
    combined = numerator.float() / denominator.permute(0, 2, 1).unsqueeze(-1)
    global_lse = max_lse + torch.log(denominator)
    return combined.to(local_output.dtype), global_lse.to(local_lse.dtype)


def distributed_lse_merge_allgather_reference(
        local_output: torch.Tensor,
        local_lse: torch.Tensor,
        group=None,
        fallback_to_local: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backend: gather all local shard outputs, then merge locally.

    This is intentionally not a performance backend. It is useful for debugging
    the all-reduce backend because it uses the same mathematical recomposition
    as the single-process dense-sharded path.
    """
    import torch.distributed as dist

    if fallback_to_local and (
            (not dist.is_available()) or (not dist.is_initialized())):
        return combine_attention_outputs_from_lse(
            local_output.unsqueeze(0), local_lse.unsqueeze(0))

    world = dist.get_world_size(group=group)
    local_output = local_output.contiguous()
    local_lse = local_lse.contiguous()
    gathered_outputs = [torch.empty_like(local_output) for _ in range(world)]
    gathered_lses = [torch.empty_like(local_lse) for _ in range(world)]
    dist.all_gather(gathered_outputs, local_output, group=group)
    dist.all_gather(gathered_lses, local_lse, group=group)
    return combine_attention_outputs_from_lse(
        torch.stack(gathered_outputs, dim=0),
        torch.stack(gathered_lses, dim=0))


def distributed_lse_merge(
        local_output: torch.Tensor,
        local_lse: torch.Tensor,
        group=None,
        *,
        backend: str = "allreduce",
        reduce_dtype: torch.dtype = torch.float32,
        fallback_to_local: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch exact distributed attention merge backend."""
    if backend == "allreduce":
        return distributed_lse_merge_allreduce(
            local_output, local_lse, group=group, reduce_dtype=reduce_dtype,
            fallback_to_local=fallback_to_local)
    if backend == "allgather_ref":
        return distributed_lse_merge_allgather_reference(
            local_output, local_lse, group=group,
            fallback_to_local=fallback_to_local)
    raise ValueError(
        f"unsupported distributed attention merge backend: {backend}")
