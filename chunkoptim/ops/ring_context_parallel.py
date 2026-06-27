"""Reference exact ring context-parallel attention primitives.

This module is intentionally small and PyTorch-first. It provides the exact
online-softmax state update needed by ring/USP style context parallelism:
query blocks stay local, KV blocks rotate across ranks, and every rank merges
each visited KV block into its local output without materializing full scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class OnlineAttentionState:
    max_score: torch.Tensor
    denominator: torch.Tensor
    numerator: torch.Tensor


def ring_kv_owner(*, rank: int, step: int, world_size: int) -> int:
    """Return the original KV owner seen by rank after `step` ring receives."""
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return (rank - step) % world_size


def init_online_attention_state(
        *,
        batch_size: int,
        q_len: int,
        num_heads: int,
        head_dim: int,
        device,
        dtype: torch.dtype = torch.float32) -> OnlineAttentionState:
    """Create empty online softmax accumulators."""
    max_score = torch.full(
        (batch_size, num_heads, q_len), -float("inf"),
        device=device, dtype=torch.float32)
    denominator = torch.zeros_like(max_score)
    numerator = torch.zeros(
        batch_size, q_len, num_heads, head_dim,
        device=device, dtype=dtype)
    return OnlineAttentionState(max_score, denominator, numerator)


def _expand_gqa_kv(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape[2] == key.shape[2]:
        return key, value
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    repeats = query.shape[2] // key.shape[2]
    return (
        key.repeat_interleave(repeats, dim=2),
        value.repeat_interleave(repeats, dim=2),
    )


def online_attention_update(
        state: OnlineAttentionState,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_start: int,
        k_start: int,
        causal: bool = True) -> OnlineAttentionState:
    """Merge one KV block into an online exact attention state.

    Shapes:
        query: [batch, q_len, heads, dim]
        key/value: [batch, kv_len, kv_heads, dim]
    """
    if key.shape != value.shape:
        raise ValueError("key and value must have matching shape")
    key, value = _expand_gqa_kv(query, key, value)
    if query.shape[0] != key.shape[0]:
        raise ValueError("query and key must share batch size")
    if query.shape[2:] != key.shape[2:]:
        raise ValueError("query/key head and dim shapes must match after GQA")

    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    scores = scores / math.sqrt(query.shape[-1])
    if causal:
        q_pos = torch.arange(
            q_start, q_start + query.shape[1], device=query.device)
        k_pos = torch.arange(
            k_start, k_start + key.shape[1], device=query.device)
        causal_mask = k_pos.view(1, 1, 1, -1) > q_pos.view(1, 1, -1, 1)
        scores = scores.masked_fill(causal_mask, -float("inf"))

    block_max = scores.max(dim=-1).values
    new_max = torch.maximum(state.max_score, block_max)
    old_scale = torch.exp(state.max_score - new_max)
    old_scale = torch.where(
        torch.isneginf(state.max_score), torch.zeros_like(old_scale), old_scale)
    block_exp = torch.exp(scores - new_max.unsqueeze(-1))
    block_exp = torch.where(
        torch.isfinite(scores), block_exp, torch.zeros_like(block_exp))
    block_denominator = block_exp.sum(dim=-1)
    block_numerator = torch.einsum("bhqk,bkhd->bqhd", block_exp, value.float())

    old_numerator = (
        state.numerator.float()
        * old_scale.permute(0, 2, 1).unsqueeze(-1))
    numerator = old_numerator + block_numerator
    denominator = state.denominator * old_scale + block_denominator
    return OnlineAttentionState(
        max_score=new_max,
        denominator=denominator,
        numerator=numerator.to(state.numerator.dtype))


def finalize_online_attention(
        state: OnlineAttentionState,
        *,
        output_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize online attention accumulators into output and global LSE."""
    denom = state.denominator.clamp_min(1e-30)
    output = state.numerator.float() / denom.permute(0, 2, 1).unsqueeze(-1)
    empty = state.denominator == 0
    output = torch.where(
        empty.permute(0, 2, 1).unsqueeze(-1),
        torch.zeros_like(output),
        output)
    lse = state.max_score + torch.log(denom)
    lse = torch.where(empty, torch.full_like(lse, -float("inf")), lse)
    return output.to(output_dtype), lse
