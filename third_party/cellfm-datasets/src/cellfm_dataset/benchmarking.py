"""Benchmark helpers for streaming group datasets."""

from __future__ import annotations

import time
from typing import Any

from .distributed import DistributedRuntime
from .hf import load_hf_group_dataset


def benchmark_streaming_group_dataset(dataset, *, steps: int) -> dict[str, Any]:
    """Measure throughput for a finite prefix of a streaming group dataset."""

    n_groups = 0
    n_cells = 0
    n_nnz = 0
    start = time.perf_counter()
    for item in dataset:
        n_groups += 1
        n_cells += len(item["source_cell_indices"])
        n_nnz += len(item["gene_indices"])
        if n_groups >= steps:
            break
    elapsed = time.perf_counter() - start
    return {
        "steps": n_groups,
        "elapsed_sec": elapsed,
        "groups_per_sec": (n_groups / elapsed) if elapsed > 0 else None,
        "cells_per_sec": (n_cells / elapsed) if elapsed > 0 else None,
        "nnz_per_sec": (n_nnz / elapsed) if elapsed > 0 else None,
    }


def benchmark_group_sampling(
    *,
    dataset_root: str,
    sampler,
    steps: int = 100,
    include_obs: bool = False,
    infer_env_distributed: bool = False,
    epoch: int = 0,
) -> dict[str, Any]:
    """Build a streaming HF group dataset and benchmark it."""

    runtime = (
        DistributedRuntime.from_environment(epoch=epoch)
        if infer_env_distributed
        else DistributedRuntime(epoch=epoch)
    )
    dataset = load_hf_group_dataset(
        dataset_root=dataset_root,
        sampler=sampler,
        streaming=True,
        include_obs=include_obs,
        distributed_context=runtime,
    )
    payload = benchmark_streaming_group_dataset(dataset, steps=steps)
    payload.update(
        {
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "epoch": runtime.epoch,
        }
    )
    return payload
