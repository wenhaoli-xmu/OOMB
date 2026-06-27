import argparse
import json
import math
import time

import torch

from chunkoptim.cache.kv_cache import SimpleCacheManager
from chunkoptim.ops.exact_streaming_attn import (
    materialize_paged_kv,
    rectangular_streaming_attention,
)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _dtype_from_name(name):
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in table:
        raise ValueError("dtype must be one of float32,bfloat16,float16")
    return table[name]


def _dense_attention_reference(query, key, value, *, q_start, kv_start=0):
    groups = query.shape[2] // key.shape[2]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float())
    scores = scores / math.sqrt(query.shape[-1])
    q_pos = torch.arange(q_start, q_start + query.shape[1], device=query.device)
    kv_pos = torch.arange(kv_start, kv_start + key.shape[1], device=query.device)
    allowed = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), -float("inf"))
    valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(valid, probs, torch.zeros_like(probs))
    return torch.einsum("bhqk,bkhd->bqhd", probs, value.float()).to(query.dtype)


def _time_call(fn, repeat, device):
    elapsed = []
    result = None
    for _ in range(repeat):
        _sync(device)
        start = time.perf_counter()
        result = fn()
        _sync(device)
        elapsed.append(time.perf_counter() - start)
    return result, sum(elapsed) / max(len(elapsed), 1)


def _peak_memory_gb(device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev) / 1e9


def _make_manager(key, value, page_size):
    dev = key.device
    local_rank = dev.index if dev.type == "cuda" else 0
    manager = SimpleCacheManager(
        key.shape[0], page_size, key.shape[2], key.shape[3],
        local_rank=local_rank)
    manager.update(key.to(torch.bfloat16), value.to(torch.bfloat16), stage=1)
    return manager


def run_rectangular_attention_benchmark(
        *, batch_size, q_len, kv_len, num_heads, num_kv_heads, head_dim,
        query_block_size, kv_block_size, q_start=None, page_size=None,
        device="cpu", dtype=torch.float32, seed=0, warmup=1, repeat=3,
        check_dense=False):
    if q_start is None:
        q_start = kv_len - q_len
    if page_size is None:
        page_size = kv_block_size
    if q_start < 0:
        raise ValueError("q_start must be >= 0")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    torch.manual_seed(seed)
    dev = torch.device(device)
    query = torch.randn(
        batch_size, q_len, num_heads, head_dim, device=dev, dtype=dtype)
    key = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, device=dev, dtype=dtype)
    value = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, device=dev, dtype=dtype)
    manager = _make_manager(key, value, page_size)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(warmup):
        key_full, value_full = materialize_paged_kv(manager)
        rectangular_streaming_attention(
            query, key_full.to(dtype), value_full.to(dtype),
            query_block_size=query_block_size,
            kv_block_size=kv_block_size,
            causal=True,
            q_start=q_start,
            kv_start=0)
    _sync(dev)

    (key_full, value_full), materialize_avg = _time_call(
        lambda: materialize_paged_kv(manager), repeat, dev)
    key_full = key_full.to(dtype)
    value_full = value_full.to(dtype)

    out, rectangular_avg = _time_call(
        lambda: rectangular_streaming_attention(
            query, key_full, value_full,
            query_block_size=query_block_size,
            kv_block_size=kv_block_size,
            causal=True,
            q_start=q_start,
            kv_start=0),
        repeat,
        dev)

    dense_check = {"enabled": False}
    if check_dense:
        ref, dense_s = _time_call(
            lambda: _dense_attention_reference(
                query, key_full, value_full, q_start=q_start),
            1,
            dev)
        diff = (out.float() - ref.float()).abs()
        denom = ref.float().abs().clamp_min(1e-8)
        dense_check = {
            "enabled": True,
            "dense_forward_s": dense_s,
            "max_abs_error": float(diff.max().item()),
            "max_rel_error": float((diff / denom).max().item()),
        }

    query_blocks = math.ceil(q_len / query_block_size)
    kv_blocks = math.ceil(kv_len / kv_block_size)
    return {
        "backend": "rectangular_exact",
        "device": str(dev),
        "dtype": str(dtype),
        "seed": seed,
        "shape": {
            "batch_size": batch_size,
            "q_len": q_len,
            "kv_len": kv_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "tiles": {
            "query_block_size": query_block_size,
            "kv_block_size": kv_block_size,
            "query_blocks": query_blocks,
            "kv_blocks": kv_blocks,
            "tile_count": query_blocks * kv_blocks,
        },
        "page_size": page_size,
        "q_start": q_start,
        "score_cells": batch_size * num_heads * q_len * kv_len,
        "timing_s": {
            "materialize_avg": materialize_avg,
            "rectangular_forward_avg": rectangular_avg,
        },
        "peak_memory_gb": _peak_memory_gb(dev),
        "dense_check": dense_check,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=2048)
    parser.add_argument("--kv-len", type=int, default=131072)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--query-block-size", type=int, default=128)
    parser.add_argument("--kv-block-size", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=None)
    parser.add_argument("--q-start", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--check-dense", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = run_rectangular_attention_benchmark(
        batch_size=args.batch_size,
        q_len=args.q_len,
        kv_len=args.kv_len,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        query_block_size=args.query_block_size,
        kv_block_size=args.kv_block_size,
        q_start=args.q_start,
        page_size=args.page_size,
        device=args.device,
        dtype=_dtype_from_name(args.dtype),
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
        check_dense=args.check_dense,
    )
    print("EXACT_ATTN_BENCH_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
