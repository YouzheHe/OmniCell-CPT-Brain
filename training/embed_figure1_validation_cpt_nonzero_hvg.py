#!/usr/bin/env python
"""Embed Figure 1 validation anchors with an OmniCell CPT checkpoint.

This is intentionally scoped to Figure 1C.  It reads the exact
atlas_validation_full_ridge/validation_cells.csv table, computes an HVG token
set from those anchors, and writes an embedding matrix in the same row order as
the validation CSV.
"""

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
DEFAULT_VALIDATION = PROJECT / "results" / "atlas_validation_full_ridge" / "validation_cells.csv"
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_CHECKPOINT = PROJECT / "results" / "figure1_multitask_cpt_alignment_full" / "backbone"
DEFAULT_OUT = PROJECT / "results" / "figure1_multitask_cpt_alignment_validation_embedding"

sys.path.insert(0, str(WORK_ROOT / "OmniCell-HF"))
sys.path.insert(0, str(WORK_ROOT / "OmniCell-HF" / "scripts"))
sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets" / "src"))

from cellfm_dataset.memmap import MemmapDataset  # noqa: E402
from omnicell_hf.configuration_omnicell import OmniCellConfig  # noqa: E402
from omnicell_hf.modeling_omnicell import OmniCellForUnsupervisedFineTuning  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from train_memmap_pretrain import OmniCellMemmapCollator  # noqa: E402


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")


class HVGFilteredMemmapCollator(OmniCellMemmapCollator):
    def __init__(self, *args: Any, hvg_token_ids: np.ndarray, **kwargs: Any) -> None:
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
        clipped = np.clip(row_indices, 0, self._allowed.shape[0] - 1)
        valid = (row_indices >= 0) & (row_indices < self._allowed.shape[0]) & self._allowed[clipped]
        return super()._select_cell_tokens(
            sample_id=sample_id,
            row_indices=row_indices[valid],
            row_values=row_values[valid],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hvg-top", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--selection-strategy", choices=["top_expression", "input_order"], default="top_expression")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def autocast_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def load_omnicell_model(checkpoint: Path, device: torch.device) -> OmniCellForUnsupervisedFineTuning:
    config = OmniCellConfig.from_pretrained(str(checkpoint))
    model = OmniCellForUnsupervisedFineTuning(config)
    if (checkpoint / "model.safetensors").exists():
        state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    elif (checkpoint / "pytorch_model.bin").exists():
        state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {checkpoint}")

    if any(k.startswith("backbone.omnicell.") for k in state):
        state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_as = "multitask_backbone"
    elif any(k.startswith("omnicell.") for k in state):
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_as = "unsupervised"
    else:
        missing, unexpected = model.omnicell.load_state_dict(state, strict=False)
        loaded_as = "backbone"
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "loaded_as": loaded_as,
                "missing_n": len(missing),
                "unexpected_n": len(unexpected),
            },
            indent=2,
        ),
        flush=True,
    )
    return model.eval().to(device)


def validation_source_index(meta: pd.DataFrame) -> np.ndarray:
    if "source_cell_index" in meta.columns:
        return meta["source_cell_index"].to_numpy(dtype=np.int64)
    if "cell_index" in meta.columns:
        return meta["cell_index"].to_numpy(dtype=np.int64)
    raise ValueError("validation CSV must contain either source_cell_index or cell_index")


def resolve_sample_id(dataset: MemmapDataset, sample_id: str) -> str:
    if sample_id in dataset.samples:
        return sample_id
    aliases = {
        "AD_Hip_sc": "AD_sc",
        "AD_sc": "AD_Hip_sc",
    }
    alt = aliases.get(sample_id)
    if alt and alt in dataset.samples:
        return alt
    raise KeyError(f"Sample {sample_id!r} is not present in the memmap dataset.")


def compute_hvg_token_ids(
    dataset: MemmapDataset,
    meta: pd.DataFrame,
    max_expression_token_id: int,
    n_top: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_genes = len(dataset.gene_vocab)
    sums = np.zeros(n_genes, dtype=np.float64)
    sum_sq = np.zeros(n_genes, dtype=np.float64)
    detected = np.zeros(n_genes, dtype=np.int64)
    source_idx = validation_source_index(meta)

    for i, (sample_id, cell_index) in enumerate(zip(meta["sample_id"].astype(str), source_idx, strict=False), start=1):
        sample_key = resolve_sample_id(dataset, str(sample_id))
        sample = dataset.samples[sample_key]
        start = int(sample.indptr[int(cell_index)])
        end = int(sample.indptr[int(cell_index) + 1])
        gene_ids = np.asarray(sample.indices[start:end], dtype=np.int64)
        values = np.asarray(sample.values[start:end], dtype=np.float32)
        good = np.isfinite(values) & (values > 0) & (gene_ids >= 0) & (gene_ids < n_genes) & (gene_ids <= max_expression_token_id)
        gene_ids = gene_ids[good]
        values = values[good]
        np.add.at(sums, gene_ids, values)
        np.add.at(sum_sq, gene_ids, values.astype(np.float64) ** 2)
        np.add.at(detected, gene_ids, 1)
        if i % 10000 == 0:
            print(f"[hvg] scanned {i:,}/{len(meta):,}", flush=True)

    mean = sums / max(len(meta), 1)
    var = sum_sq / max(len(meta), 1) - mean * mean
    var[(detected < 10) | ~np.isfinite(var)] = -np.inf
    order = np.argsort(-var, kind="mergesort")
    order = order[np.isfinite(var[order])][: min(int(n_top), int(np.isfinite(var).sum()))]
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "gene": np.asarray(dataset.gene_vocab, dtype=str)[order],
            "token_id": order.astype(np.int64),
            "variance": var[order],
            "mean": mean[order],
            "detected": detected[order],
        }
    )
    return order.astype(np.int64), table


def build_collator(dataset_root: Path, config: Any, hvg_ids: np.ndarray, selection_strategy: str) -> HVGFilteredMemmapCollator:
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


def rows_to_groups(dataset: MemmapDataset, rows: pd.DataFrame) -> list[dict[str, Any]]:
    source_idx = validation_source_index(rows)
    groups = []
    for i, (row, cell_index) in enumerate(zip(rows.itertuples(index=False), source_idx, strict=False)):
        sample_id = resolve_sample_id(dataset, str(getattr(row, "sample_id")))
        groups.append(
            dataset.build_group(
                sample_id=sample_id,
                source_cell_indices=np.asarray([int(cell_index)], dtype=np.int64),
                group_id=f"{sample_id}:{int(cell_index)}",
                group_type="figure1_validation_anchor",
                group_index=i,
                sample_group_index=int(cell_index),
                normalize_coords=True,
                include_obs=False,
            )
        )
    return groups


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embedding.npy"
    if emb_path.exists() and not args.force:
        print(f"[skip] {emb_path} exists; use --force to regenerate", flush=True)
        return

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OmniCell CPT inference requires CUDA.")

    meta = pd.read_csv(args.validation_csv.expanduser().resolve(), low_memory=False)
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset = MemmapDataset(dataset_root)
    model = load_omnicell_model(args.checkpoint.expanduser().resolve(), device)
    max_expression_token_id = min(model.config.vocab_size - 1, model.config.start_token_id - 1)
    hvg_ids, hvg_table = compute_hvg_token_ids(dataset, meta, max_expression_token_id, args.hvg_top)
    hvg_table.to_csv(out_dir / "hvg_gene_tokens.csv", index=False)
    np.save(out_dir / "hvg_token_ids.npy", hvg_ids)
    meta.to_parquet(out_dir / "embedding_meta.parquet", index=False)
    meta.to_csv(out_dir / "embedding_meta.csv", index=False)

    collator = build_collator(dataset_root, model.config, hvg_ids, args.selection_strategy)
    embedding = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(len(meta), model.config.d_model))
    nonzero_counts = np.lib.format.open_memmap(out_dir / "nonzero_hvg_counts.npy", mode="w+", dtype=np.int16, shape=(len(meta),))
    dtype = autocast_dtype(args.dtype)
    run_config = {
        "validation_csv": str(args.validation_csv.expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "output_dir": str(out_dir),
        "hvg_top": int(args.hvg_top),
        "hvg_tokens_available": int(len(hvg_ids)),
        "selection_strategy": args.selection_strategy,
        "pooling": "OmniCell nonzero-mask pooling after HVG token filtering",
        "batch_size": int(args.batch_size),
        "device": str(device),
        "dtype": args.dtype,
        "n_rows": int(len(meta)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps(run_config, indent=2), flush=True)

    with torch.inference_mode():
        for start in range(0, len(meta), args.batch_size):
            end = min(start + args.batch_size, len(meta))
            batch_rows = meta.iloc[start:end]
            groups = rows_to_groups(dataset, batch_rows)
            batch = to_device(collator(groups), device)
            mask = batch["nonzero_mask"][:, 1:-1].sum(dim=1).detach().cpu().numpy()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                outputs = model.omnicell(**batch)
            embedding[start:end] = outputs.cell_embeddings[:, 0, :].detach().float().cpu().numpy()
            nonzero_counts[start:end] = mask.astype(np.int16)
            if end % (args.batch_size * 100) == 0 or end == len(meta):
                embedding.flush()
                nonzero_counts.flush()
                print(f"[embed] {end:,}/{len(meta):,}", flush=True)

    embedding.flush()
    nonzero_counts.flush()
    print(f"[done] wrote {emb_path}", flush=True)


if __name__ == "__main__":
    main()
