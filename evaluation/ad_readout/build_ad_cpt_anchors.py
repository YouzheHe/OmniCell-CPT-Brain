#!/usr/bin/env python
"""Build AD/Control anchor tables for BI CPT representation learning.

The script starts from the AD hippocampus all-cell index and writes separate
single-cell and spatial anchor CSVs. Each anchor is a single cell/spot with
sample_id, source_cell_index, condition, modality, and broad cell type columns
that can be consumed by train_memmap_multitask_alignment.py.
"""

from __future__ import annotations
import os

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
DEFAULT_INDEX = WORK_ROOT / "projects/nvu_vascular/results/ad_hip_allcell_index/ad_hip_all_cells.parquet"
DEFAULT_OUT = WORK_ROOT / "projects/BI/results/ad_cpt_anchors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modalities", default="single_cell,spatial")
    parser.add_argument("--max-per-sample", type=int, default=20000)
    parser.add_argument("--max-per-condition-celltype", type=int, default=12000)
    parser.add_argument("--max-per-modality", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def clean_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "na", "n/a", "unknown"}:
        return ""
    return text


def infer_condition(row: pd.Series) -> str:
    candidates = [
        row.get("condition_inferred", ""),
        row.get("condition", ""),
        row.get("disease", ""),
        row.get("batch_id", ""),
        row.get("sample_id", ""),
        row.get("obs_name", ""),
    ]
    text = " ".join(clean_text(x) for x in candidates).replace("_", " ")
    if re.search(r"(^|[/\s-])con\d*|(^|[/\s-])ctrl\d*|control|healthy|normal", text, re.IGNORECASE):
        return "Control"
    if re.search(r"(^|[/\s-])ad\d*|alzheimer|dementia", text, re.IGNORECASE):
        return "AD"
    return "Unknown"


def canonical_modality(value: object) -> str:
    text = clean_text(value).lower()
    if "spatial" in text or "saptial" in text:
        return "spatial"
    if "single" in text or "snrna" in text or "scrna" in text:
        return "single_cell"
    return text or "unknown"


def choose_celltype(row: pd.Series) -> str:
    for col in ["ground_truth_celltype", "broad_celltype", "cell_class", "celltype", "cell_type", "ground_truth_label"]:
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return "Other/unknown"


def sample_balanced(frame: pd.DataFrame, group_cols: list[str], max_n: int, rng: np.random.Generator) -> pd.DataFrame:
    if max_n <= 0:
        return frame
    parts = []
    for _, group in frame.groupby(group_cols, dropna=False, sort=False):
        if len(group) > max_n:
            parts.append(group.sample(n=max_n, random_state=int(rng.integers(0, 2**31 - 1))))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def final_condition_balance(frame: pd.DataFrame, max_total: int, rng: np.random.Generator) -> pd.DataFrame:
    counts = frame["condition_inferred"].value_counts()
    if not {"AD", "Control"}.issubset(set(counts.index)):
        return frame
    target = int(counts[["AD", "Control"]].min())
    if max_total > 0:
        target = min(target, max_total // 2)
    parts = []
    for condition in ["AD", "Control"]:
        group = frame[frame["condition_inferred"].eq(condition)]
        if len(group) > target:
            parts.append(group.sample(n=target, random_state=int(rng.integers(0, 2**31 - 1))))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True)


def prepare_base(index: pd.DataFrame) -> pd.DataFrame:
    frame = index.copy()
    if "cell_index" not in frame.columns:
        raise ValueError("Input index must contain a cell_index column.")
    if "sample_id" not in frame.columns:
        raise ValueError("Input index must contain a sample_id column.")
    frame["source_cell_index"] = pd.to_numeric(frame["cell_index"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["sample_id", "source_cell_index"]).copy()
    frame["source_cell_index"] = frame["source_cell_index"].astype(int)
    frame["condition_inferred"] = frame.apply(infer_condition, axis=1)
    frame["modality_inferred"] = frame.get("modality", "unknown").map(canonical_modality)
    frame["ground_truth_celltype"] = frame.apply(choose_celltype, axis=1)
    if "batch_id" not in frame.columns:
        frame["batch_id"] = frame["sample_id"].astype(str)
    frame["batch_id"] = frame["batch_id"].map(clean_text).where(lambda s: s.ne(""), frame["sample_id"].astype(str))
    frame["condition_celltype_stratum"] = (
        "cond=" + frame["condition_inferred"].astype(str) + "|cell=" + frame["ground_truth_celltype"].astype(str)
    )
    frame["condition_sample_stratum"] = (
        "cond=" + frame["condition_inferred"].astype(str) + "|sample=" + frame["batch_id"].astype(str)
    )
    frame["condition_modality_celltype_stratum"] = (
        "cond="
        + frame["condition_inferred"].astype(str)
        + "|mod="
        + frame["modality_inferred"].astype(str)
        + "|cell="
        + frame["ground_truth_celltype"].astype(str)
    )
    keep_cols = [
        "sample_id",
        "source_cell_index",
        "cell_index",
        "obs_name",
        "condition_inferred",
        "modality_inferred",
        "ground_truth_celltype",
        "ground_truth_label",
        "batch_id",
        "dataset_source",
        "age_years",
        "coord_x",
        "coord_y",
        "condition_celltype_stratum",
        "condition_sample_stratum",
        "condition_modality_celltype_stratum",
    ]
    keep_cols = [col for col in keep_cols if col in frame.columns]
    return frame[keep_cols].reset_index(drop=True)


def write_modality(
    base: pd.DataFrame,
    modality: str,
    output_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, object]:
    out_csv = output_dir / f"bi_ad_{modality}_cpt_anchors.csv"
    if out_csv.exists() and not args.force:
        existing = pd.read_csv(out_csv, usecols=["condition_inferred", "ground_truth_celltype", "sample_id"])
        return {
            "modality": modality,
            "path": str(out_csv),
            "n_anchors": int(len(existing)),
            "condition_counts": existing["condition_inferred"].value_counts().to_dict(),
        }

    frame = base[base["modality_inferred"].eq(modality)].copy()
    frame = frame[frame["condition_inferred"].isin(["AD", "Control"])].copy()
    if frame.empty:
        raise ValueError(f"No AD/Control anchors found for modality={modality}")

    frame = sample_balanced(frame, ["condition_inferred", "batch_id"], args.max_per_sample, rng)
    frame = sample_balanced(
        frame,
        ["condition_inferred", "ground_truth_celltype"],
        args.max_per_condition_celltype,
        rng,
    )
    frame = final_condition_balance(frame, args.max_per_modality, rng)
    frame = frame.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    frame.to_csv(out_csv, index=False)

    summary = {
        "modality": modality,
        "path": str(out_csv),
        "n_anchors": int(len(frame)),
        "n_samples": int(frame["sample_id"].nunique()),
        "n_batches": int(frame["batch_id"].nunique()),
        "condition_counts": frame["condition_inferred"].value_counts().to_dict(),
        "celltype_counts": frame["ground_truth_celltype"].value_counts().head(30).to_dict(),
        "sample_counts": frame.groupby(["condition_inferred", "batch_id"], dropna=False).size().reset_index(name="n").to_dict("records"),
    }
    (output_dir / f"bi_ad_{modality}_cpt_anchor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    frame.groupby(["condition_inferred", "ground_truth_celltype"], dropna=False).size().reset_index(name="n").to_csv(
        output_dir / f"bi_ad_{modality}_condition_celltype_counts.csv",
        index=False,
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    index = read_table(args.index)
    base = prepare_base(index)
    summaries = []
    for modality in split_csv(args.modalities):
        summaries.append(write_modality(base, modality, args.output_dir, args, rng))
    payload = {
        "input_index": str(args.index),
        "output_dir": str(args.output_dir),
        "modalities": summaries,
        "max_per_sample": args.max_per_sample,
        "max_per_condition_celltype": args.max_per_condition_celltype,
        "max_per_modality": args.max_per_modality,
    }
    (args.output_dir / "bi_ad_cpt_anchor_build_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
