#!/usr/bin/env python
"""Build age-balanced anchors for Figure 1 multi-task OmniCell fine-tuning."""

from __future__ import annotations
import os

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
CELLFM_SRC = WORK_ROOT / "cellfm-datasets" / "src"
sys.path.insert(0, str(CELLFM_SRC))

from cellfm_dataset.memmap import MemmapDataset  # noqa: E402


DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "projects" / "nvu_vascular" / "results" / "figure1_agebalanced_training_anchors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--max-per-stratum", type=int, default=5000)
    parser.add_argument("--max-per-sample", type=int, default=35000)
    parser.add_argument("--n-age-bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def first_present(row: dict[str, Any], columns: list[str]) -> str:
    for col in columns:
        if col in row and row[col] is not None:
            value = str(row[col]).strip()
            if value and value.lower() not in {"nan", "na", "none", "<na>", "unknown"}:
                return value
    return ""


def parse_age(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    age = float(match.group(0))
    if not np.isfinite(age) or age < 0 or age > 120:
        return None
    return age


def infer_condition(row: dict[str, Any], sample_id: str) -> str:
    value = first_present(
        row,
        [
            "condition_inferred",
            "condition",
            "disease",
            "ROIGroupFine",
            "Diagnosis",
            "diagnosis",
            "sample",
            "chip",
            "chipID",
            "batch_id",
        ],
    )
    text = f"{value} {sample_id}".lower()
    if re.search(r"\bad\b|alzheimer|dement|(?:^|[/_\-\s])ad\d", text):
        return "AD"
    if re.search(r"\bcon\b|control|normal|ctrl|healthy|(?:^|[/_\-\s])con\d", text):
        return "Control"
    return "Other"


def infer_modality(sample_id: str, coord_dim: int) -> str:
    text = sample_id.lower()
    if coord_dim > 0 or "spatial" in text or "saptial" in text or "stereo" in text:
        return "spatial"
    return "single_cell"


def infer_cell_class(row: dict[str, Any]) -> str:
    value = first_present(
        row,
        [
            "ground_truth_celltype",
            "cell_class",
            "CellType_m",
            "celltype",
            "cell_type",
            "CellType",
            "subclass.v4",
            "ground_truth_label",
            "cell_label_original",
        ],
    )
    text = value.lower()
    if not text:
        return "Other"
    if any(x in text for x in ["excit", "ex_", "glut", "ca1", "ca2", "ca3", "dg", "sub", "it neuron", "et neuron", "ct neuron"]):
        return "Excitatory neuron"
    if any(x in text for x in ["inhib", "in_", "gaba", "pvalb", "vip", "sst", "lamp5"]):
        return "Inhibitory neuron"
    if "astro" in text:
        return "Astrocyte"
    if any(x in text for x in ["oligo", "oligodendrocyte"]):
        return "Oligodendrocyte"
    if "opc" in text or "precursor" in text:
        return "OPC"
    if any(x in text for x in ["micro", "immune", "macrophage", "myeloid", "monocyte"]):
        return "Microglia/immune"
    if any(x in text for x in ["endo", "vascular", "pericyte", "mural", "smc", "vsmc", "fibro", "vlmc"]):
        return "Vascular"
    if any(x in text for x in ["ependymal", "choroid"]):
        return "Ependymal/choroid"
    return "Other"


def infer_cohort(sample_id: str) -> str:
    if "AD_Hip_Saptial" in sample_id:
        return "AD_Hip_Saptial"
    if sample_id == "AD_sc" or "AD_Hip_sc" in sample_id:
        return "AD_sc"
    if "Cortex_Spatial" in sample_id:
        return "Cortex_Spatial"
    if sample_id == "Cortex_sc":
        return "Cortex_sc"
    if "AD_Cortex_Spatial" in sample_id:
        return "AD_Cortex_Spatial"
    if "39402379" in sample_id:
        return "39402379"
    if sample_id.isdigit():
        return "Public_snRNA_PMID"
    return "Other"


def available_age_columns(obs: pd.DataFrame) -> list[str]:
    return [col for col in ["age_years", "ageNum", "age", "Age", "age_months", "donor_age"] if col in obs.columns]


def choose_sample_ids(dataset: MemmapDataset, requested: list[str]) -> list[str]:
    if requested:
        missing = [sid for sid in requested if sid not in dataset.samples]
        if missing:
            raise KeyError(f"Unknown sample ids: {missing[:10]}")
        return requested
    selected: list[str] = []
    for sid, sample in dataset.samples.items():
        if not sample.has_obs_table:
            continue
        obs = sample.load_obs_frame()
        age_cols = available_age_columns(obs)
        if not age_cols:
            continue
        ages = obs[age_cols[0]].map(parse_age)
        if int(ages.notna().sum()) > 0:
            selected.append(sid)
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "agebalanced_training_anchors.csv.gz"
    out_summary = args.output_dir / "agebalanced_training_anchors_summary.json"
    if out_csv.exists() and out_summary.exists() and not args.force:
        print(out_csv)
        return

    rng = np.random.default_rng(args.seed)
    dataset = MemmapDataset(args.dataset_root)
    sample_ids = choose_sample_ids(dataset, split_csv(args.sample_ids))
    frames: list[pd.DataFrame] = []
    sample_summary: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        sample = dataset.samples[sample_id]
        obs = sample.load_obs_frame()
        age_cols = available_age_columns(obs)
        if not age_cols:
            continue
        age_col = age_cols[0]
        ages = obs[age_col].map(parse_age)
        keep = ages.notna()
        if not keep.any():
            continue
        indices = np.flatnonzero(keep.to_numpy())
        if len(indices) > args.max_per_sample:
            indices = rng.choice(indices, size=args.max_per_sample, replace=False)
        sub = obs.iloc[indices].copy()
        sub["source_cell_index"] = indices.astype(int)
        sub["sample_id"] = sample_id
        sub["age_years"] = ages.iloc[indices].astype(float).to_numpy()
        coord_dim = int(getattr(sample.metadata, "coord_dim", 0) or 0)
        sub["modality_inferred"] = infer_modality(sample_id, coord_dim)
        sub["condition_inferred"] = [infer_condition(row, sample_id) for row in sub.to_dict("records")]
        sub["cell_class_inferred"] = [infer_cell_class(row) for row in sub.to_dict("records")]
        sub["cohort_inferred"] = infer_cohort(sample_id)
        frames.append(
            sub[
                [
                    "sample_id",
                    "source_cell_index",
                    "age_years",
                    "condition_inferred",
                    "modality_inferred",
                    "cell_class_inferred",
                    "cohort_inferred",
                ]
            ]
        )
        sample_summary.append(
            {
                "sample_id": sample_id,
                "n_age_cells": int(keep.sum()),
                "n_sampled": int(len(indices)),
                "age_min": float(np.nanmin(ages.iloc[indices])),
                "age_max": float(np.nanmax(ages.iloc[indices])),
                "modality": infer_modality(sample_id, coord_dim),
            }
        )

    if not frames:
        raise RuntimeError("No age-labeled anchors could be built.")
    anchors = pd.concat(frames, ignore_index=True)
    anchors["age_bin"] = pd.qcut(
        anchors["age_years"].rank(method="first"),
        q=min(args.n_age_bins, anchors["age_years"].nunique()),
        labels=False,
        duplicates="drop",
    ).astype(int)
    anchors["age_condition_modality_cell_class_stratum"] = (
        "age" + anchors["age_bin"].astype(str)
        + "|cond=" + anchors["condition_inferred"].astype(str)
        + "|mod=" + anchors["modality_inferred"].astype(str)
        + "|cell=" + anchors["cell_class_inferred"].astype(str)
    )
    anchors["age_condition_modality_stratum"] = (
        "age" + anchors["age_bin"].astype(str)
        + "|cond=" + anchors["condition_inferred"].astype(str)
        + "|mod=" + anchors["modality_inferred"].astype(str)
    )

    selected = []
    for _, group in anchors.groupby("age_condition_modality_cell_class_stratum", sort=False):
        if len(group) > args.max_per_stratum:
            selected.append(group.sample(n=args.max_per_stratum, random_state=int(rng.integers(0, 2**31 - 1))))
        else:
            selected.append(group)
    anchors = pd.concat(selected, ignore_index=True)
    anchors = anchors.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    anchors.to_csv(out_csv, index=False)

    summary = {
        "dataset_root": str(args.dataset_root),
        "n_selected_samples": len(sample_summary),
        "n_anchors": int(len(anchors)),
        "n_strata": int(anchors["age_condition_modality_cell_class_stratum"].nunique()),
        "max_per_stratum": args.max_per_stratum,
        "max_per_sample": args.max_per_sample,
        "age_min": float(anchors["age_years"].min()),
        "age_max": float(anchors["age_years"].max()),
        "by_age_bin": anchors["age_bin"].value_counts().sort_index().to_dict(),
        "by_modality": anchors["modality_inferred"].value_counts().to_dict(),
        "by_condition": anchors["condition_inferred"].value_counts().to_dict(),
        "by_cell_class": anchors["cell_class_inferred"].value_counts().to_dict(),
        "samples": sample_summary,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print(out_csv)


if __name__ == "__main__":
    main()
