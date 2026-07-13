"""Distributed sharding helpers for sampling and dataset iteration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    """Distributed sharding metadata for one runtime consumer."""

    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    num_workers: int = 1
    epoch: int = 0
    shard_mode: str = "stride"

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}.")
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must satisfy 0 <= rank < world_size, got rank={self.rank}, world_size={self.world_size}."
            )
        if not 0 <= self.worker_id < self.num_workers:
            raise ValueError(
                "worker_id must satisfy 0 <= worker_id < num_workers, "
                f"got worker_id={self.worker_id}, num_workers={self.num_workers}."
            )
        if self.shard_mode != "stride":
            raise ValueError(f"Unsupported shard_mode '{self.shard_mode}'.")

    @property
    def shard_index(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    @property
    def num_shards(self) -> int:
        return self.world_size * self.num_workers

    def with_worker(self, worker_id: int, num_workers: int) -> "DistributedContext":
        """Return a new context with worker metadata filled in."""

        return DistributedContext(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            num_workers=num_workers,
            epoch=self.epoch,
            shard_mode=self.shard_mode,
        )

    def seed_offset(self, base_seed: int) -> int:
        """Return a deterministic epoch-aware seed."""

        return int(base_seed) + self.epoch * 1_000_003


@dataclass
class DistributedRuntime:
    """Mutable distributed runtime state for training loops.

    This object is intended for epoch-aware streaming datasets. Update
    ``epoch`` between epochs, and the next iteration will observe the
    new distributed context.
    """

    rank: int = 0
    world_size: int = 1
    epoch: int = 0
    shard_mode: str = "stride"

    def __post_init__(self) -> None:
        DistributedContext(
            rank=self.rank,
            world_size=self.world_size,
            epoch=self.epoch,
            shard_mode=self.shard_mode,
        )

    @classmethod
    def from_environment(
        cls,
        *,
        epoch: int = 0,
        shard_mode: str = "stride",
    ) -> "DistributedRuntime":
        """Construct runtime metadata from common distributed environment variables."""

        rank = 0
        world_size = 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
        return cls(rank=rank, world_size=world_size, epoch=epoch, shard_mode=shard_mode)

    def set_epoch(self, epoch: int) -> None:
        """Update epoch for the next dataset iteration."""

        self.epoch = int(epoch)

    def to_context(self) -> DistributedContext:
        """Freeze the mutable runtime into one immutable context."""

        return DistributedContext(
            rank=self.rank,
            world_size=self.world_size,
            epoch=self.epoch,
            shard_mode=self.shard_mode,
        )


def resolve_distributed_context(
    distributed_context: DistributedContext | DistributedRuntime | None = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    shard_mode: str = "stride",
    infer_worker: bool = True,
) -> DistributedContext:
    """Resolve a full distributed context, optionally inferring worker metadata."""

    if distributed_context is None:
        context = DistributedContext(rank=rank, world_size=world_size, epoch=epoch, shard_mode=shard_mode)
    elif isinstance(distributed_context, DistributedRuntime):
        context = distributed_context.to_context()
    else:
        context = distributed_context

    if not infer_worker:
        return context

    try:
        from torch.utils.data import get_worker_info
    except Exception:
        return context

    worker_info = get_worker_info()
    if worker_info is None:
        return context
    return context.with_worker(worker_id=int(worker_info.id), num_workers=int(worker_info.num_workers))


def shard_iterable(
    iterable: Iterable[T],
    distributed_context: DistributedContext | None,
    *,
    index_fn: Callable[[T], int] | None = None,
) -> Iterator[T]:
    """Yield only the records assigned to one distributed shard."""

    if distributed_context is None or distributed_context.num_shards == 1:
        yield from iterable
        return

    shard_index = distributed_context.shard_index
    num_shards = distributed_context.num_shards

    if index_fn is None:
        for item_index, item in enumerate(iterable):
            if item_index % num_shards == shard_index:
                yield item
        return

    for item in iterable:
        if int(index_fn(item)) % num_shards == shard_index:
            yield item
