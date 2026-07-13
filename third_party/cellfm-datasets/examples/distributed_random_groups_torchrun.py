"""Minimal torchrun example for distributed random group streaming.

Run with:
  torchrun --nproc_per_node=8 examples/distributed_random_groups_torchrun.py \
    --dataset-root /path/to/memmap_dataset
"""

from __future__ import annotations

import argparse
import json

from cellfm_dataset import DistributedRuntime, RandomCellSampler, load_hf_group_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed random-group streaming example.")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--cells-per-group", type=int, default=256)
    parser.add_argument("--num-groups", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = DistributedRuntime.from_environment(epoch=0)
    sampler = RandomCellSampler(
        cells_per_group=args.cells_per_group,
        num_groups=args.num_groups,
        seed=7,
        sample_weight_mode="n_cells",
    )

    for epoch in range(2):
        runtime.set_epoch(epoch)
        dataset = load_hf_group_dataset(
            dataset_root=args.dataset_root,
            sampler=sampler,
            streaming=True,
            include_obs=False,
            distributed_context=runtime,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "rank": runtime.rank,
                    "world_size": runtime.world_size,
                }
            ),
            flush=True,
        )
        for step, batch_item in enumerate(dataset):
            if step >= args.steps:
                break
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "rank": runtime.rank,
                        "step": step,
                        "group_id": batch_item["group_id"],
                        "sample_id": batch_item["sample_id"],
                        "n_cells": len(batch_item["source_cell_indices"]),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
