#!/usr/bin/env python
"""Embed indexed cells with the latest OmniCell CPT checkpoint.

The default index is the vascular-cell table used by the original Figure 2
workflow, but the script also accepts all-cell indexes with sample_id and
cell_index columns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


WORK_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_CHECKPOINT_ROOT = WORK_ROOT / "checkpoint" / "OmniCell_CPT_336687"
DEFAULT_INDEX = WORK_ROOT / "projects" / "nvu_vascular" / "results" / "vascular_index" / "vascular_cells.parquet"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "projects" / "nvu_vascular" / "results" / "omnicell_cpt_latest_embeddings"

sys.path.insert(0, str(WORK_ROOT / "OmniCell-HF"))
sys.path.insert(0, str(WORK_ROOT / "OmniCell-HF" / "scripts"))
sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets" / "src"))

from cellfm_dataset.memmap import MemmapDataset  # noqa: E402
from omnicell_hf.modeling_omnicell import OmniCellForUnsupervisedFineTuning  # noqa: E402
from train_memmap_pretrain import OmniCellMemmapCollator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--index-parquet", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--vascular-classes", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--selection-strategy", choices=["top_expression", "input_order"], default="top_expression")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def split_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def latest_checkpoint(root: Path) -> Path:
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
            checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories with model weights found under {root}")
    return sorted(checkpoints, key=lambda item: item[0])[-1][1]


def load_index(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def filter_index(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    sample_ids = split_csv(args.sample_ids)
    classes = split_csv(args.vascular_classes)
    selected = frame
    if sample_ids is not None:
        selected = selected[selected["sample_id"].astype(str).isin(sample_ids)]
    if classes is not None:
        selected = selected[selected["vascular_class"].astype(str).isin(classes)]
    if args.limit is not None:
        selected = selected.head(args.limit)
    return selected.reset_index(drop=True)


def build_collator(dataset_root: Path, config: Any, selection_strategy: str) -> OmniCellMemmapCollator:
    max_expression_token_id = min(config.vocab_size - 1, config.start_token_id - 1)
    return OmniCellMemmapCollator(
        dataset_root=dataset_root,
        token_per_cell=config.token_per_cell,
        vocab_size=config.vocab_size,
        start_token_id=config.start_token_id,
        end_token_id=config.end_token_id,
        max_expression_token_id=max_expression_token_id,
        use_smooth_rank=True,
        smooth_rank_range=(0.0, 5.0),
        selection_strategy=selection_strategy,
    )


def rows_to_groups(dataset: MemmapDataset, rows: pd.DataFrame) -> list[dict[str, Any]]:
    groups = []
    for row in rows.itertuples(index=False):
        sample_id = str(getattr(row, "sample_id"))
        cell_index = int(getattr(row, "cell_index"))
        groups.append(
            dataset.build_group(
                sample_id=sample_id,
                source_cell_indices=np.asarray([cell_index], dtype=np.int64),
                group_id=f"{sample_id}:{cell_index}",
                group_type="indexed_cell",
                group_index=cell_index,
                sample_group_index=cell_index,
                normalize_coords=True,
                include_obs=False,
            )
        )
    return groups


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def write_meta(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        print(f"[WARN] Could not write {path.name} as parquet: {exc}. Wrote {fallback}.")
        return fallback


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint = (args.checkpoint.expanduser().resolve() if args.checkpoint else latest_checkpoint(args.checkpoint_root.expanduser().resolve()))
    index_path = args.index_parquet.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index = filter_index(load_index(index_path), args)
    if index.empty:
        raise ValueError("No indexed cells remain after filtering.")

    run_config = {
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "index_path": str(index_path),
        "output_dir": str(output_dir),
        "n_cells": int(len(index)),
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "sample_ids": args.sample_ids,
        "vascular_classes": args.vascular_classes,
        "limit": args.limit,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
    meta_path = write_meta(index, output_dir / "embedding_meta.parquet")
    print(json.dumps({**run_config, "embedding_meta": str(meta_path)}, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if device.type != "cuda":
        raise RuntimeError("OmniCell CPT inference requires CUDA because this model uses flash-attn.")

    dataset = MemmapDataset(dataset_root)
    model = OmniCellForUnsupervisedFineTuning.from_pretrained(str(checkpoint))
    model.eval().to(device)
    collator = build_collator(dataset_root, model.config, args.selection_strategy)

    embedding = np.lib.format.open_memmap(
        output_dir / "embedding.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(index), model.config.d_model),
    )
    dtype = autocast_dtype(args.dtype)

    with torch.inference_mode():
        for start in range(0, len(index), args.batch_size):
            end = min(start + args.batch_size, len(index))
            groups = rows_to_groups(dataset, index.iloc[start:end])
            batch = to_device(collator(groups), device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                outputs = model.omnicell(**batch)
            values = outputs.cell_embeddings[:, 0, :].detach().float().cpu().numpy()
            embedding[start:end] = values
            embedding.flush()
            print(f"[INFO] embedded {end}/{len(index)}", flush=True)

    print(f"[INFO] wrote embeddings: {output_dir / 'embedding.npy'}", flush=True)


if __name__ == "__main__":
    main()
