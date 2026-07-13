#!/usr/bin/env python
"""Build a standardized vascular-cell index from the NVU memmap dataset."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WORK_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "projects" / "nvu_vascular" / "results" / "vascular_index"

CELL_LABEL_COLUMNS = [
    "subcelltype",
    "celltype_unit",
    "CellType",
    "celltype",
    "cell_type",
    "Subclass",
    "subclass.v4",
    "CellType_m",
    "Class",
    "Supertype",
]
CONTEXT_COLUMNS = [
    "condition",
    "disease",
    "sample",
    "sample_id",
    "chip",
    "chipID",
    "chipId",
    "age",
    "ageNum",
    "region",
    "region1",
    "area",
    "gender",
    "sex",
    "tissue",
    "brain_region1_en",
    "brain_region2",
    "ROIGroupFine",
    "batch",
]
VASCULAR_REGEX = re.compile(
    r"endo|endothelial|pericyte|smooth|smc|vsmc|mural|vascular|vlmc|"
    r"fibro|leptomeningeal|perivascular|capillary|arter|ven",
    re.IGNORECASE,
)
CLASS_RULES = [
    ("endothelial", re.compile(r"endo|endothelial|capillary|arter|ven", re.IGNORECASE)),
    ("pericyte", re.compile(r"pericyte", re.IGNORECASE)),
    ("smooth_muscle", re.compile(r"smooth|smc|vsmc|mural", re.IGNORECASE)),
    ("vlmc_fibroblast", re.compile(r"vlmc|fibro|leptomeningeal", re.IGNORECASE)),
    ("vascular_unknown", re.compile(r"vascular", re.IGNORECASE)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--sample-annotation-csv", type=Path, default=None)
    parser.add_argument("--max-cells-per-sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-coords", action="store_true")
    return parser.parse_args()


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame = frame.copy()
        for column in frame.select_dtypes(include=["object"]).columns:
            frame[column] = frame[column].astype("string")
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception as exc:
            fallback = path.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            print(f"[WARN] Could not write {path.name} as parquet: {exc}. Wrote {fallback}.")
            return fallback
    frame.to_csv(path, index=False)
    return path


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "dataset_manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def cohort_from_sample(sample_id: str) -> str:
    if sample_id.startswith("AD_Cortex_Spatial/"):
        return "AD_Cortex_Spatial"
    if sample_id.startswith("AD_Hip_Saptial/"):
        return "AD_Hip_Saptial"
    if sample_id.startswith("Cortex_Spatial/"):
        return "Cortex_Spatial"
    if sample_id.startswith("39402379"):
        return "39402379"
    if sample_id in {"AD_sc", "Cortex_sc"}:
        return sample_id
    return "Public_snRNA_PMID"


def modality_from_coord_dim(coord_dim: int) -> str:
    return "spatial" if int(coord_dim) > 0 else "single_cell"


def parse_age_years(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "/"}:
        return float("nan")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return float("nan")
    return float(match.group(1))


def infer_condition(sample_id: str, row: pd.Series) -> str:
    candidates = []
    for col in ("condition", "disease", "sample", "obs_sample_id", "chip", "chipID", "chipId"):
        if col in row and pd.notna(row[col]):
            candidates.append(str(row[col]))
    candidates.append(sample_id)
    text = " ".join(candidates)
    if re.search(r"\bAD\b|Alzheimer|dementia", text, re.IGNORECASE):
        return "AD"
    if re.search(r"\bCon|Control|CTRL|healthy|normal", text, re.IGNORECASE):
        return "Control"
    return "Unknown"


def classify_label(label: Any) -> str | None:
    text = "" if label is None or pd.isna(label) else str(label)
    for class_name, pattern in CLASS_RULES:
        if pattern.search(text):
            return class_name
    return None


def choose_vascular_label(row: pd.Series, label_columns: list[str]) -> tuple[str | None, str | None, str | None]:
    fallback_label = None
    fallback_column = None
    fallback_class = None
    for col in label_columns:
        value = row.get(col)
        if pd.isna(value):
            continue
        label = str(value)
        vascular_class = classify_label(label)
        if vascular_class is None:
            continue
        if vascular_class != "vascular_unknown":
            return label, col, vascular_class
        if fallback_label is None:
            fallback_label = label
            fallback_column = col
            fallback_class = vascular_class
    return fallback_label, fallback_column, fallback_class


def vascular_mask(frame: pd.DataFrame, label_columns: list[str]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for col in label_columns:
        values = frame[col].astype("string")
        mask = mask | values.str.contains(VASCULAR_REGEX, na=False)
    return mask


def load_sample_annotations(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError("--sample-annotation-csv must contain a sample_id column.")
    return frame


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    selected_sample_ids = split_csv(args.sample_ids)
    sample_annotations = load_sample_annotations(args.sample_annotation_csv)

    manifest = load_manifest(dataset_root)
    samples = manifest["samples"]
    rows: list[pd.DataFrame] = []
    sample_summary: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = str(sample["sample_id"])
        if selected_sample_ids is not None and sample_id not in selected_sample_ids:
            continue
        sample_dir = dataset_root / sample_id
        obs_path = sample_dir / "obs.parquet"
        label_columns = [col for col in CELL_LABEL_COLUMNS if col in sample.get("obs_columns", [])]
        context_columns = [col for col in CONTEXT_COLUMNS if col in sample.get("obs_columns", [])]
        read_columns = ["obs_name", *label_columns, *context_columns]
        read_columns = list(dict.fromkeys([col for col in read_columns if col]))

        summary = {
            "sample_id": sample_id,
            "cohort": cohort_from_sample(sample_id),
            "modality": modality_from_coord_dim(sample.get("coord_dim", 0)),
            "coord_dim": int(sample.get("coord_dim", 0)),
            "n_cells": int(sample["n_cells"]),
            "n_present_genes": int(sample.get("n_present_genes", 0)),
            "obs_columns": "|".join(sample.get("obs_columns", [])),
            "vascular_cells": 0,
            "source_h5ad": sample.get("source_h5ad", ""),
        }

        if not obs_path.exists() or not label_columns:
            sample_summary.append(summary)
            continue

        obs = pd.read_parquet(obs_path, columns=read_columns)
        mask = vascular_mask(obs, label_columns)
        vascular = obs.loc[mask].copy()
        if vascular.empty:
            sample_summary.append(summary)
            continue

        if args.max_cells_per_sample is not None and len(vascular) > args.max_cells_per_sample:
            selected_positions = rng.choice(len(vascular), size=args.max_cells_per_sample, replace=False)
            vascular = vascular.iloc[np.sort(selected_positions)].copy()

        if "sample_id" in vascular.columns:
            vascular["obs_sample_id"] = vascular["sample_id"]
            vascular = vascular.drop(columns=["sample_id"])
            context_columns = ["obs_sample_id" if col == "sample_id" else col for col in context_columns]

        labels = vascular.apply(lambda row: choose_vascular_label(row, label_columns), axis=1)
        vascular["cell_label_original"] = [item[0] for item in labels]
        vascular["cell_label_column"] = [item[1] for item in labels]
        vascular["vascular_class"] = [item[2] for item in labels]
        vascular = vascular[vascular["vascular_class"].notna()].copy()
        vascular["cell_index"] = vascular.index.astype(np.int64)
        vascular["sample_id"] = sample_id
        vascular["cohort"] = summary["cohort"]
        vascular["modality"] = summary["modality"]
        vascular["coord_dim"] = summary["coord_dim"]
        vascular["source_h5ad"] = summary["source_h5ad"]
        vascular["condition_inferred"] = vascular.apply(
            lambda row: infer_condition(sample_id, row), axis=1
        )

        age_source = None
        for col in ("ageNum", "age"):
            if col in vascular.columns:
                age_source = col
                break
        vascular["age_years"] = (
            vascular[age_source].map(parse_age_years) if age_source else np.nan
        )
        vascular["age_source_column"] = age_source or ""

        if not args.no_coords and summary["coord_dim"] > 0 and (sample_dir / "coords.npy").exists():
            coords = np.load(sample_dir / "coords.npy", mmap_mode="r")
            selected_coords = np.asarray(coords[vascular["cell_index"].to_numpy()], dtype=np.float32)
            vascular["coord_x"] = selected_coords[:, 0]
            vascular["coord_y"] = selected_coords[:, 1] if selected_coords.shape[1] > 1 else np.nan
        else:
            vascular["coord_x"] = np.nan
            vascular["coord_y"] = np.nan

        if sample_annotations is not None:
            vascular = vascular.merge(sample_annotations, how="left", on="sample_id", suffixes=("", "_annot"))

        summary["vascular_cells"] = int(len(vascular))
        sample_summary.append(summary)
        keep_columns = [
            "sample_id",
            "cell_index",
            "obs_name",
            "cohort",
            "modality",
            "coord_dim",
            "coord_x",
            "coord_y",
            "cell_label_original",
            "cell_label_column",
            "vascular_class",
            "condition_inferred",
            "age_years",
            "age_source_column",
            "source_h5ad",
            *[col for col in context_columns if col in vascular.columns],
        ]
        keep_columns = list(dict.fromkeys([col for col in keep_columns if col in vascular.columns]))
        rows.append(vascular[keep_columns])
        print(f"[INFO] {sample_id}: vascular_cells={len(vascular)}", flush=True)

    vascular_index = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    sample_summary_frame = pd.DataFrame(sample_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    vascular_path = write_table(vascular_index, output_dir / "vascular_cells.parquet")
    write_table(sample_summary_frame, output_dir / "sample_summary.csv")
    if not vascular_index.empty:
        label_counts = (
            vascular_index.groupby(["cohort", "modality", "vascular_class", "cell_label_original"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        write_table(label_counts, output_dir / "vascular_label_counts.csv")
    figure1 = (
        sample_summary_frame.groupby(["cohort", "modality", "coord_dim"], dropna=False)
        .agg(n_samples=("sample_id", "count"), n_cells=("n_cells", "sum"), vascular_cells=("vascular_cells", "sum"))
        .reset_index()
        .sort_values(["cohort", "modality"])
    )
    write_table(figure1, output_dir / "figure1_dataset_summary.csv")
    config = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "sample_ids": sorted(selected_sample_ids) if selected_sample_ids is not None else None,
        "max_cells_per_sample": args.max_cells_per_sample,
        "vascular_cells_path": str(vascular_path),
        "n_vascular_cells": int(len(vascular_index)),
        "n_samples_processed": int(len(sample_summary_frame)),
    }
    (output_dir / "build_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
