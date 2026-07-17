from dataclasses import dataclass


@dataclass(frozen=True)
class KVBlock:
    start: int
    end: int
    backend: str
    owner_rank: int | None = None
    hop: int = 0

    @property
    def tokens(self):
        return self.end - self.start


def _validate_schedule_args(num_kv, kv_block_size, world_size=None, rank=None):
    if num_kv < 1:
        raise ValueError("num_kv must be >= 1")
    if kv_block_size < 1:
        raise ValueError("kv_block_size must be >= 1")
    if world_size is not None and world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank is not None and (rank < 0 or rank >= world_size):
        raise ValueError("rank must be in [0, world_size)")


def rectangular_kv_schedule(num_kv, kv_block_size):
    """Return contiguous KV tiles for exact rectangular streaming attention."""
    _validate_schedule_args(num_kv, kv_block_size)
    return [
        KVBlock(start, min(start + kv_block_size, num_kv), "rectangular")
        for start in range(0, num_kv, kv_block_size)
    ]


def _owned_blocks(num_kv, kv_block_size, world_size, backend):
    blocks = []
    for block_idx, start in enumerate(range(0, num_kv, kv_block_size)):
        owner = block_idx % world_size
        blocks.append(KVBlock(
            start=start,
            end=min(start + kv_block_size, num_kv),
            backend=backend,
            owner_rank=owner,
            hop=0,
        ))
    return blocks


def ring_kv_schedule(num_kv, kv_block_size, world_size, rank):
    """Return KV blocks in the owner order a rank would see in a ring pass.

    This is a transport schedule only: it describes which global KV tile is
    processed at each ring hop. The caller is responsible for the actual
    send/recv or all-gather implementation.
    """
    _validate_schedule_args(num_kv, kv_block_size, world_size, rank)
    blocks = _owned_blocks(num_kv, kv_block_size, world_size, "ring")
    ordered = []
    for hop in range(world_size):
        owner = (rank - hop) % world_size
        for block in blocks:
            if block.owner_rank == owner:
                ordered.append(KVBlock(
                    block.start, block.end, block.backend, block.owner_rank, hop))
    return ordered


def usp_kv_schedule(num_kv, kv_block_size, world_size, rank):
    """Return a Ulysses-style schedule with global tile order and rank owners.

    USP-style implementations can keep global block order while using separate
    collectives to materialize remote KV tiles. The `hop` field is therefore a
    distance hint rather than a strict ring step.
    """
    _validate_schedule_args(num_kv, kv_block_size, world_size, rank)
    blocks = _owned_blocks(num_kv, kv_block_size, world_size, "usp")
    scheduled = []
    for block in blocks:
        distance = min(
            (rank - block.owner_rank) % world_size,
            (block.owner_rank - rank) % world_size,
        )
        scheduled.append(KVBlock(
            block.start, block.end, block.backend, block.owner_rank, distance))
    return scheduled
