import torch

from ..ops import (
    flash_paged_attn_distributed_func,
    flash_paged_attn_func,
)
from ..ops.exact_streaming_attn import paged_rectangular_attention


def _normalize_backend(name):
    return str(name).replace("-", "_").lower()


def _dtype_from_config(value):
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return torch.float32
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return table[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            "attention reduce_dtype must be one of float32, fp32, "
            "bfloat16, bf16") from exc


def _bool_from_config(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError("attention fallback_to_local must be a boolean")


def resolve_attention_config(manager):
    raw = getattr(manager, "attention_conf", None) or {}
    return {
        "backend": _normalize_backend(raw.get("backend", "paged")),
        "merge_backend": raw.get("merge_backend", "allreduce"),
        "reduce_dtype": _dtype_from_config(raw.get("reduce_dtype")),
        "group": raw.get("group", None),
        "fallback_to_local": _bool_from_config(raw.get("fallback_to_local", True)),
        "query_block_size": int(raw.get("query_block_size", 128)),
        "kv_block_size": int(raw.get("kv_block_size", 128)),
    }


def run_blockwise_attention(query, key, value, manager):
    config = resolve_attention_config(manager)
    backend = config["backend"]
    if backend == "paged":
        return flash_paged_attn_func(query, key, value, manager)
    if backend == "distributed_paged":
        return flash_paged_attn_distributed_func(
            query, key, value, manager,
            config["group"],
            config["reduce_dtype"],
            config["merge_backend"],
            config["fallback_to_local"])
    if backend == "rectangular":
        return paged_rectangular_attention(
            query, key, value, manager,
            query_block_size=config["query_block_size"],
            kv_block_size=config["kv_block_size"])
    raise ValueError(f"unsupported blockwise attention backend: {backend}")
