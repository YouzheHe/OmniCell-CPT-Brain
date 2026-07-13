"""PyTorch DataLoader example for fixed-size random groups.

Run with:
  python examples/dataloader_random_groups.py \
    --dataset-root /path/to/memmap_dataset \
    --num-workers 4
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from cellfm_dataset import CellFMRegionDataCollator, RandomCellSampler, load_hf_group_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataLoader example for fixed-size random groups.")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--cells-per-group", type=int, default=256)
    parser.add_argument("--num-groups", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampler = RandomCellSampler(
        cells_per_group=args.cells_per_group,
        num_groups=args.num_groups,
        seed=7,
        sample_weight_mode="n_cells",
    )

    dataset = load_hf_group_dataset(
        dataset_root=args.dataset_root,
        sampler=sampler,
        streaming=True,
        include_obs=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=CellFMRegionDataCollator(),
    )

    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        print(
            {
                "step": step,
                "batch_size": int(batch["coords"].shape[0]),
                "cells_per_group": int(batch["coords"].shape[1]),
                "coord_dim": int(batch["coords"].shape[2]),
                "nnz": int(batch["gene_indices"].shape[0]),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
