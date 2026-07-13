#!/usr/bin/env python
"""Embed all vascular anchors with modality-specific nonzero-HVG OmniCell pooling."""

from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
PROJECT = WORK_ROOT / "projects" / "nvu_vascular"
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_CHECKPOINT = WORK_ROOT / "checkpoint" / "OmniCell_CPT_336687" / "checkpoint-245000"
DEFAULT_INDEX = PROJECT / "results" / "vascular_index" / "vascular_cells.parquet"
DEFAULT_OUTPUT = PROJECT / "results" / "vascular_omnicell_cpt_nonzero_hvg_all_data"

sys.path.insert(0, str(WORK_ROOT / "OmniCell-HF"))
sys.path.insert(0, str(WORK_ROOT / "OmniCell-HF" / "scripts"))
sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets" / "src"))

from cellfm_dataset.memmap import MemmapDataset  # noqa: E402
from omnicell_hf.modeling_omnicell import OmniCellForUnsupervisedFineTuning  # noqa: E402
from train_memmap_pretrain import OmniCellMemmapCollator  # noqa: E402


class HVGFilteredMemmapCollator(OmniCellMemmapCollator):
    """Filter each cell to nonzero genes in a supplied HVG token set."""

    def __init__(self, *args, hvg_token_ids: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ids = np.asarray(hvg_token_ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids <= self.max_expression_token_id) & (ids < self.vocab_size)]
        if ids.size == 0:
            raise ValueError("No HVG token ids remain after filtering.")
        self.hvg_token_ids = np.unique(ids)
        self._allowed = np.zeros(self.vocab_size, dtype=bool)
        self._allowed[self.hvg_token_ids] = True

    def _fallback_gene_ids(self, sample_id: str) -> np.ndarray:  # noqa: ARG002
        return self.hvg_token_ids

    def _select_cell_tokens(
        self,
        *,
        sample_id: str,
        row_indices: np.ndarray,
        row_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        row_values = np.asarray(row_values, dtype=np.float32)
        if row_indices.size == 0:
            return super()._select_cell_tokens(sample_id=sample_id, row_indices=row_indices, row_values=row_values)
        clipped = np.clip(row_indices, 0, self._allowed.shape[0] - 1)
        valid = (row_indices >= 0) & (row_indices < self._allowed.shape[0]) & self._allowed[clipped] & (row_values > 0)
        return super()._select_cell_tokens(sample_id=sample_id, row_indices=row_indices[valid], row_values=row_values[valid])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index-parquet", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hvg-top", type=int, default=15000)
    parser.add_argument("--hvg-mode", choices=["by_modality", "global"], default="by_modality")
    parser.add_argument("--min-detected-cells", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--selection-strategy", choices=["top_expression", "input_order"], default="top_expression")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_gene_vocab(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def sample_dir_for_sample_id(dataset_root: Path, sample_id: str) -> Path:
    if sample_id == "AD_sc":
        return dataset_root / "AD_Hip_sc"
    return dataset_root / sample_id


def load_index(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"sample_id", "cell_index", "modality"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Index lacks required columns: {missing}")
    frame = frame.reset_index(drop=True).copy()
    frame["embedding_row"] = np.arange(len(frame), dtype=np.int64)
    frame["cell_index"] = frame["cell_index"].astype(np.int64)
    return frame


def rows_to_groups(dataset: MemmapDataset, rows: pd.DataFrame) -> list[dict[str, Any]]:
    groups = []
    for i, row in enumerate(rows.itertuples(index=False)):
        sample_id = str(getattr(row, "sample_id"))
        cell_index = int(getattr(row, "cell_index"))
        groups.append(
            dataset.build_group(
                sample_id=sample_id,
                source_cell_indices=np.asarray([cell_index], dtype=np.int64),
                group_id=f"{sample_id}:{cell_index}",
                group_type="indexed_cell",
                group_index=i,
                sample_group_index=cell_index,
                normalize_coords=True,
                include_obs=False,
            )
        )
    return groups


def iter_sample_rows(index: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = []
    for sample_id, frame in index.groupby("sample_id", sort=False):
        order = np.argsort(frame["cell_index"].to_numpy(dtype=np.int64), kind="mergesort")
        groups.append(
            (
                str(sample_id),
                frame.index.to_numpy(dtype=np.int64)[order],
                frame["cell_index"].to_numpy(dtype=np.int64)[order],
            )
        )
    return groups


def accumulate_hvg_stats(
    *,
    index: pd.DataFrame,
    dataset_root: Path,
    n_genes: int,
    min_detected_cells: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    sums = np.zeros(n_genes, dtype=np.float64)
    sum_sq = np.zeros(n_genes, dtype=np.float64)
    detected = np.zeros(n_genes, dtype=np.int64)
    n_obs = int(len(index))
    print(f"[hvg] accumulating variance over {n_obs:,} vascular rows", flush=True)

    for sample_id, _, cell_indices in iter_sample_rows(index):
        sample_dir = sample_dir_for_sample_id(dataset_root, sample_id)
        if not sample_dir.exists():
            raise FileNotFoundError(f"Missing sample directory for {sample_id}: {sample_dir}")
        indptr = np.load(sample_dir / "indptr.npy", mmap_mode="r")
        indices = np.load(sample_dir / "indices.npy", mmap_mode="r")
        values = np.load(sample_dir / "values.npy", mmap_mode="r")
        for cell_i in cell_indices:
            start = int(indptr[cell_i])
            end = int(indptr[cell_i + 1])
            if end <= start:
                continue
            idx = np.asarray(indices[start:end], dtype=np.int64)
            val = np.nan_to_num(np.asarray(values[start:end], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            mask = (idx >= 0) & (idx < n_genes) & (val > 0)
            if not np.any(mask):
                continue
            idx = idx[mask]
            val = val[mask].astype(np.float64, copy=False)
            np.add.at(sums, idx, val)
            np.add.at(sum_sq, idx, val * val)
            np.add.at(detected, idx, 1)

    mean = sums / max(n_obs, 1)
    variance = sum_sq / max(n_obs, 1) - mean * mean
    valid = detected >= int(min_detected_cells)
    variance[~valid] = -np.inf
    order = np.argsort(-variance, kind="mergesort")
    order = order[np.isfinite(variance[order])]
    return order.astype(np.int64), pd.DataFrame(
        {
            "token_id": np.arange(n_genes, dtype=np.int64),
            "mean": mean,
            "variance": variance,
            "detected_cells": detected,
            "detected_fraction": detected / max(n_obs, 1),
        }
    )


def compute_hvgs(
    index: pd.DataFrame,
    dataset_root: Path,
    gene_vocab: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_genes = len(gene_vocab)
    if args.hvg_mode == "global":
        partitions = {"global": index}
    else:
        partitions = {str(mod): frame for mod, frame in index.groupby("modality", sort=False)}

    hvg_by_partition: dict[str, np.ndarray] = {}
    for name, frame in partitions.items():
        slug = name.replace("/", "_")
        hvg_path = output_dir / f"hvg_token_ids_{slug}.npy"
        hvg_table_path = output_dir / f"hvg_gene_tokens_{slug}.csv"
        if hvg_path.exists() and hvg_table_path.exists() and not args.force:
            hvg_ids = np.load(hvg_path)
            print(f"[hvg] using cached {name}: {len(hvg_ids):,} genes", flush=True)
            hvg_by_partition[name] = hvg_ids.astype(np.int64)
            continue

        order, stats = accumulate_hvg_stats(
            index=frame,
            dataset_root=dataset_root,
            n_genes=n_genes,
            min_detected_cells=args.min_detected_cells,
        )
        order = order[: min(int(args.hvg_top), len(order))]
        hvg_table = stats.iloc[order].copy()
        hvg_table.insert(0, "rank", np.arange(1, len(hvg_table) + 1))
        hvg_table.insert(2, "gene_id", [gene_vocab[int(i)] for i in hvg_table["token_id"].to_numpy()])
        hvg_table.to_csv(hvg_table_path, index=False)
        np.save(hvg_path, order.astype(np.int64))
        hvg_by_partition[name] = order.astype(np.int64)
        print(f"[hvg] {name}: selected {len(order):,} HVGs -> {hvg_table_path}", flush=True)
    return hvg_by_partition


def build_collator(dataset_root: Path, config: Any, selection_strategy: str, hvg_ids: np.ndarray) -> HVGFilteredMemmapCollator:
    max_expression_token_id = min(config.vocab_size - 1, config.start_token_id - 1)
    return HVGFilteredMemmapCollator(
        dataset_root=dataset_root,
        token_per_cell=config.token_per_cell,
        vocab_size=config.vocab_size,
        start_token_id=config.start_token_id,
        end_token_id=config.end_token_id,
        max_expression_token_id=max_expression_token_id,
        use_smooth_rank=True,
        smooth_rank_range=(0.0, 5.0),
        selection_strategy=selection_strategy,
        hvg_token_ids=hvg_ids,
    )


def autocast_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    emb_path = output_dir / "embedding.npy"
    if emb_path.exists() and not args.force:
        print(f"[skip] {emb_path} exists; use --force to regenerate", flush=True)
        return

    index = load_index(args.index_parquet.expanduser().resolve())
    gene_vocab = read_gene_vocab(dataset_root / "gene_vocab.txt")
    hvg_by_partition = compute_hvgs(index, dataset_root, gene_vocab, args)

    run_config = {
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "index_path": str(args.index_parquet.expanduser().resolve()),
        "output_dir": str(output_dir),
        "n_cells": int(len(index)),
        "hvg_top": int(args.hvg_top),
        "hvg_mode": args.hvg_mode,
        "min_detected_cells": int(args.min_detected_cells),
        "pooling": "OmniCell hidden-state pooling over nonzero expressed HVG gene tokens via nonzero_mask",
        "batch_size": int(args.batch_size),
        "device": args.device,
        "dtype": args.dtype,
        "selection_strategy": args.selection_strategy,
        "partitions": {key: int(len(value)) for key, value in hvg_by_partition.items()},
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
    index.to_parquet(output_dir / "embedding_meta.parquet", index=False)
    index.to_csv(output_dir / "embedding_meta.csv", index=False)
    print(json.dumps(run_config, indent=2, ensure_ascii=False), flush=True)
    if args.check_only:
        return

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OmniCell CPT inference requires CUDA.")
    model = OmniCellForUnsupervisedFineTuning.from_pretrained(str(checkpoint))
    model.eval().to(device)
    dataset = MemmapDataset(dataset_root)
    dtype = autocast_dtype(args.dtype)

    embedding = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(len(index), model.config.d_model))
    nonzero_counts = np.lib.format.open_memmap(output_dir / "nonzero_hvg_counts.npy", mode="w+", dtype=np.int16, shape=(len(index),))

    if args.hvg_mode == "global":
        modality_partitions = {"global": index}
    else:
        modality_partitions = {str(mod): frame for mod, frame in index.groupby("modality", sort=False)}

    with torch.inference_mode():
        for partition_name, frame in modality_partitions.items():
            hvg_key = "global" if args.hvg_mode == "global" else partition_name
            collator = build_collator(dataset_root, model.config, args.selection_strategy, hvg_by_partition[hvg_key])
            frame = frame.sort_values("embedding_row", kind="mergesort")
            print(f"[embed] {partition_name}: {len(frame):,} rows", flush=True)
            for start in range(0, len(frame), args.batch_size):
                end = min(start + args.batch_size, len(frame))
                batch_rows = frame.iloc[start:end]
                batch = to_device(collator(rows_to_groups(dataset, batch_rows)), device)
                mask = batch["nonzero_mask"][:, 1:-1].sum(dim=1).detach().cpu().numpy()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                    outputs = model.omnicell(**batch)
                row_ids = batch_rows["embedding_row"].to_numpy(dtype=np.int64)
                embedding[row_ids] = outputs.cell_embeddings[:, 0, :].detach().float().cpu().numpy()
                nonzero_counts[row_ids] = mask.astype(np.int16)
                if end % (args.batch_size * 100) == 0 or end == len(frame):
                    embedding.flush()
                    nonzero_counts.flush()
                    print(f"[embed] {partition_name}: {end:,}/{len(frame):,}", flush=True)

    embedding.flush()
    nonzero_counts.flush()
    print(f"[done] wrote {emb_path}", flush=True)


if __name__ == "__main__":
    main()
