#!/usr/bin/env python
"""Figure 3 spatial vascular density/proportion and marker-QC workflow v18.

This run treats spatial observations as single-cell-resolution vascular anchors.
Main abundance readouts are normalized by whole-chip cell number and by physical
chip area. Original vascular labels are retained as the cell-type backbone when
available; marker/profile evidence is reported as QC support rather than as a
reason to drop many vascular anchors as low confidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_figure3_adhip_full_spatial_deconvolution_v14 as v14  # noqa: E402


BASE = v14.DEFAULT_BASE
FIG_BASE = v14.PROJECT / "figures/vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"

SUBTYPE_ORDER = ["Endo", "Pericyte", "SMC", "Fibroblast"]
SUBTYPE_DISPLAY = {
    "Endo": "Endothelial",
    "Pericyte": "Pericyte",
    "SMC": "SMC",
    "Fibroblast": "Fibroblast/VLMC",
}
SUBTYPE_PALETTE = {
    "Endo": "#4E79A7",
    "Pericyte": "#B07AA1",
    "SMC": "#E15759",
    "Fibroblast": "#59A14F",
    "Vascular": "#7F7F7F",
}
SUPPORT_PALETTE = {
    "top_match": "#3E7CB1",
    "prob_supported": "#59A14F",
    "marker_supported": "#76B7B2",
    "profile_supported": "#F28E2B",
    "needs_review": "#BFC5CC",
    "high_marker_confidence": "#3E7CB1",
    "moderate_marker_confidence": "#76B7B2",
    "low_marker_resolution": "#C7CCD1",
}
DISEASE_PALETTE = {"Control": "#7E8FA6", "AD": "#C86054"}
AGE_GROUP_PALETTE = {
    "0-17": "#8BB6A8",
    "18-39": "#7E8FA6",
    "40-59": "#D3A55E",
    "60+": "#C86054",
    "missing_age": "#C7CCD1",
}

MARKER_MODULES = {
    "Endo": ["PECAM1", "CDH5", "CLDN5", "VWF", "FLT1", "ERG", "ESAM", "TIE1", "ICAM2", "RAMP2", "KDR", "CA4", "SLC2A1"],
    "Pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9", "NOTCH3", "CSPG4", "MCAM", "NDUFA4L2", "HIGD1B", "COX4I2"],
    "SMC": ["ACTA2", "TAGLN", "MYL9", "TPM2", "CNN1", "CALD1", "MYH11", "MYLK", "MYOCD"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRA", "COL6A1", "COL6A2", "FBLN1", "MGP", "ABCA8", "APOD", "CFD", "PI16", "DPT", "SLIT2"],
    "BBB_capillary_state": ["CLDN5", "SLC2A1", "MFSD2A", "ABCB1", "ABCG2", "OCLN", "TJP1", "CA4", "RGCC", "RAMP2", "EMCN"],
    "Endothelial_activation_state": ["ACKR1", "NR2F2", "VCAM1", "SELE", "APLNR", "APLN", "ESM1", "ANGPT2", "ICAM1"],
    "Mural_contractile_state": ["ACTA2", "TAGLN", "MYL9", "TPM2", "CNN1", "CALD1", "MYH11", "MYLK", "MYOCD"],
}
DOTPLOT_GENES = {
    "Endo": ["PECAM1", "CLDN5", "VWF", "RAMP2"],
    "Pericyte": ["PDGFRB", "RGS5", "ABCC9", "NOTCH3"],
    "SMC": ["ACTA2", "TAGLN", "MYH11", "CNN1"],
    "Fibroblast": ["COL1A1", "COL3A1", "DCN", "LUM"],
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation-csv", type=Path, default=v14.DEFAULT_ANNOTATION)
    p.add_argument("--memmap-root", type=Path, default=v14.MEMMAP_ROOT)
    p.add_argument("--alias-csv", type=Path, default=v14.ALIAS_CSV)
    p.add_argument("--cohort", default="AD_Hip_Saptial")
    p.add_argument("--analysis-id", default="adhip_v18")
    p.add_argument("--comparison-mode", choices=["disease", "age"], default="disease")
    p.add_argument("--result-dir", type=Path, default=None)
    p.add_argument("--figure-dir", type=Path, default=None)
    p.add_argument("--max-query-rows", type=int, default=0, help="Optional smoke-test row limit per chip.")
    p.add_argument("--coord-unit-um", type=float, default=0.5, help="Physical size of one coordinate unit.")
    p.add_argument("--profile-temperature", type=float, default=0.16)
    p.add_argument("--marker-temperature", type=float, default=0.70)
    p.add_argument("--profile-weight", type=float, default=0.45)
    p.add_argument("--marker-weight", type=float, default=0.55)
    p.add_argument("--support-threshold", type=float, default=0.30)
    p.add_argument("--marker-confidence-threshold", type=float, default=0.38)
    p.add_argument("--marker-margin-threshold", type=float, default=0.08)
    p.add_argument("--point-size", type=float, default=0.42)
    p.add_argument("--dpi", type=int, default=650)
    return p.parse_args()


def default_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.result_dir is not None and args.figure_dir is not None:
        return Path(args.result_dir), Path(args.figure_dir)
    name = f"figure3_{args.analysis_id}_spatial_vascular_density_v18/spatial"
    result_dir = Path(args.result_dir) if args.result_dir is not None else BASE / name
    figure_dir = Path(args.figure_dir) if args.figure_dir is not None else FIG_BASE / name
    return result_dir, figure_dir


def save_all(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    v14.save_all(fig, prefix, dpi)


def read_annotation_table(annotation_path: Path, columns: list[str]) -> pd.DataFrame:
    if annotation_path.suffix == ".parquet":
        return pd.read_parquet(annotation_path, columns=columns)
    return pd.read_csv(annotation_path, usecols=lambda c: c in columns, low_memory=False)


def infer_adhip_disease(sample_id: object) -> str:
    sid = str(sample_id).lower()
    if re.search(r"/ad", sid):
        return "AD"
    if re.search(r"/con", sid):
        return "Control"
    return "Unknown"


def age_group(value: object) -> str:
    age = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(age):
        return "missing_age"
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 60:
        return "40-59"
    return "60+"


def canonical_label(label: object, vascular_class: object = "") -> str | None:
    vc = str(vascular_class).strip().lower()
    txt = str(label).strip().lower()
    if vc == "endothelial" or re.search(r"endo|endothelial|capillary", txt):
        return "Endo"
    if vc == "smooth_muscle" or re.search(r"\bvsmc\b|smooth muscle|vascular associated smooth muscle", txt):
        return "SMC"
    if vc == "pericyte" or re.search(r"pericyte", txt):
        return "Pericyte"
    if vc == "vlmc_fibroblast" or re.search(r"fibro|leptomeningeal|vlmc", txt):
        return "Fibroblast"
    return None


def load_query(annotation_csv: Path, cohort: str, comparison_mode: str, max_query_rows: int) -> pd.DataFrame:
    wanted = [
        "sample_id",
        "cell_index",
        "cohort",
        "modality",
        "coord_x",
        "coord_y",
        "cell_label_original",
        "vascular_class",
        "condition_inferred",
        "age_years",
        "region",
        "brain_region1_en",
        "brain_region2",
        "annotation",
        "annotation_confidence",
    ]
    obs = read_annotation_table(annotation_csv, wanted)
    sub = obs[obs["cohort"].astype(str).eq(cohort) & obs["modality"].astype(str).eq("spatial")].copy()
    if sub.empty:
        raise ValueError(f"No spatial vascular anchors found for cohort={cohort!r}.")
    sub["original_canonical_label"] = [
        canonical_label(label, vc) for label, vc in zip(sub["cell_label_original"], sub["vascular_class"])
    ]
    sub["original_label_for_quant"] = sub["original_canonical_label"].fillna("Vascular")
    sub["age_years"] = pd.to_numeric(sub.get("age_years", np.nan), errors="coerce")
    sub["age_group"] = sub["age_years"].map(age_group)
    if comparison_mode == "disease":
        sub["disease_group"] = sub["sample_id"].map(infer_adhip_disease)
        fallback = sub.get("condition_inferred", "").astype(str)
        sub.loc[~sub["disease_group"].isin(["AD", "Control"]) & fallback.isin(["AD", "Control"]), "disease_group"] = fallback
        sub = sub[sub["disease_group"].isin(["AD", "Control"])].copy()
    else:
        sub["disease_group"] = sub.get("condition_inferred", "Unknown").astype(str)
    sub["cell_index"] = pd.to_numeric(sub["cell_index"], errors="raise").astype(np.int64)
    sub["coord_x"] = pd.to_numeric(sub["coord_x"], errors="coerce")
    sub["coord_y"] = pd.to_numeric(sub["coord_y"], errors="coerce")
    if max_query_rows and max_query_rows > 0:
        sub = sub.groupby("sample_id", group_keys=False, observed=False).head(max_query_rows)
    sub = sub.reset_index(drop=True)
    sub["_row_order"] = np.arange(len(sub), dtype=np.int64)
    if sub.empty:
        raise ValueError(f"No usable spatial vascular anchors for cohort={cohort!r}.")
    return sub


def marker_gene_ids(gene_symbols: list[str]) -> tuple[list[int], list[str], pd.DataFrame]:
    symbol_to_id: dict[str, int] = {}
    for idx, symbol in enumerate(gene_symbols):
        symbol_to_id.setdefault(str(symbol).upper(), idx)
    requested = []
    for genes in MARKER_MODULES.values():
        requested.extend([g.upper() for g in genes])
    requested = sorted(set(requested))
    present = [g for g in requested if g in symbol_to_id]
    rows = []
    for module, genes in MARKER_MODULES.items():
        genes_up = [g.upper() for g in genes]
        matched = [g for g in genes_up if g in symbol_to_id]
        rows.append(
            {
                "module": module,
                "n_requested": len(genes_up),
                "n_matched": len(matched),
                "matched_genes": ",".join(matched),
                "missing_genes": ",".join([g for g in genes_up if g not in symbol_to_id]),
            }
        )
    return [symbol_to_id[g] for g in present], present, pd.DataFrame(rows)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return np.nan_to_num(np.clip((x - med) / scale, -4.0, 4.0), nan=0.0, posinf=4.0, neginf=-4.0).astype(np.float32)


def softmax_rows(score: np.ndarray, temperature: float) -> np.ndarray:
    z = score / max(float(temperature), 1e-4)
    z = z - np.nanmax(z, axis=1, keepdims=True)
    exp = np.exp(np.nan_to_num(z, nan=-30.0, posinf=30.0, neginf=-30.0)).astype(np.float32)
    denom = exp.sum(axis=1, keepdims=True)
    return np.divide(exp, np.maximum(denom, 1e-8), out=np.full_like(exp, 1.0 / exp.shape[1]), where=denom > 0)


def compute_marker_scores(log_expr: np.ndarray, genes: list[str]) -> pd.DataFrame:
    scaled = robust_zscore(log_expr)
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(genes)}
    scores = {}
    for module, module_genes in MARKER_MODULES.items():
        idx = [gene_to_idx[g.upper()] for g in module_genes if g.upper() in gene_to_idx]
        scores[f"marker_score_{module}"] = scaled[:, idx].mean(axis=1).astype(np.float32) if idx else np.zeros(log_expr.shape[0], dtype=np.float32)
    return pd.DataFrame(scores)


def finite_xy(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=float)
    arr = arr[:, :2]
    return arr[np.isfinite(arr).all(axis=1)]


def bbox_area(coords: np.ndarray) -> float:
    xy = finite_xy(coords)
    if len(xy) < 2:
        return np.nan
    area = float((np.nanmax(xy[:, 0]) - np.nanmin(xy[:, 0])) * (np.nanmax(xy[:, 1]) - np.nanmin(xy[:, 1])))
    return area if np.isfinite(area) and area > 0 else np.nan


def convex_hull_area(coords: np.ndarray) -> float:
    xy = finite_xy(coords)
    if len(xy) < 3:
        return np.nan
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(xy, qhull_options="QJ")
        area = float(hull.volume)
        return area if np.isfinite(area) and area > 0 else np.nan
    except Exception:
        return np.nan


def chip_area_table(query: pd.DataFrame, memmap_root: Path, coord_unit_um: float) -> pd.DataFrame:
    rows = []
    for sample_id, sub in query.groupby("sample_id", observed=False):
        sample_dir = memmap_root / str(sample_id)
        all_coords = np.zeros((0, 2), dtype=np.float32)
        coord_path = sample_dir / "coords.npy"
        if coord_path.exists():
            all_coords = np.asarray(np.load(coord_path, mmap_mode="r"), dtype=np.float32)
        vascular_coords = sub[["coord_x", "coord_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        all_hull = convex_hull_area(all_coords)
        all_bbox = bbox_area(all_coords)
        vascular_hull = convex_hull_area(vascular_coords)
        vascular_bbox = bbox_area(vascular_coords)
        tissue_area = np.nan
        area_source = ""
        for source_name, candidate in [
            ("all_cell_convex_hull", all_hull),
            ("all_cell_bbox", all_bbox),
            ("vascular_convex_hull", vascular_hull),
            ("vascular_bbox", vascular_bbox),
        ]:
            if np.isfinite(candidate) and candidate > 0:
                tissue_area = float(candidate)
                area_source = source_name
                break
        area_cm2 = tissue_area * (float(coord_unit_um) ** 2) / 1e8 if np.isfinite(tissue_area) else np.nan
        rows.append(
            {
                "sample_id": sample_id,
                "n_vascular_anchors": int(len(sub)),
                "n_all_cell_coords": int(len(finite_xy(all_coords))),
                "tissue_area_coord2": tissue_area,
                "tissue_area_um2": tissue_area * (float(coord_unit_um) ** 2) if np.isfinite(tissue_area) else np.nan,
                "tissue_area_cm2": area_cm2,
                "coord_unit_um": float(coord_unit_um),
                "area_source": area_source,
                "all_cell_convex_hull_area_coord2": all_hull,
                "all_cell_bbox_area_coord2": all_bbox,
                "vascular_convex_hull_area_coord2": vascular_hull,
                "vascular_bbox_area_coord2": vascular_bbox,
            }
        )
    out = pd.DataFrame(rows)
    if out["tissue_area_coord2"].isna().any():
        missing = out.loc[out["tissue_area_coord2"].isna(), "sample_id"].astype(str).tolist()
        raise ValueError("Could not estimate tissue area for chips: " + ", ".join(missing))
    return out


def balanced_profiles(x_scaled: np.ndarray, obs: pd.DataFrame, min_cells: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostics = []
    labels = obs["original_canonical_label"].dropna().astype(str)
    available = [x for x in SUBTYPE_ORDER if int(labels.eq(x).sum()) >= min_cells]
    if len(available) < 2:
        return pd.DataFrame(), pd.DataFrame(columns=["label", "n_cells", "n_sample_profiles"])
    for label in available:
        label_mask = obs["original_canonical_label"].astype(str).eq(label).to_numpy()
        sample_profiles = []
        sample_rows = []
        for sample_id, idx in obs.loc[label_mask].groupby("sample_id", sort=False, observed=False).groups.items():
            pos = obs.index.get_indexer(idx)
            if len(pos) == 0:
                continue
            sample_profiles.append(x_scaled[pos].mean(axis=0))
            sample_rows.append({"sample_id": sample_id, "n_cells": int(len(pos))})
        sample_meta = pd.DataFrame(sample_rows)
        sample_x = np.vstack(sample_profiles)
        rows.append(pd.Series(sample_x.mean(axis=0), name=label))
        diagnostics.append({"label": label, "n_cells": int(label_mask.sum()), "n_sample_profiles": int(len(sample_profiles)), "median_cells_per_profile": float(sample_meta["n_cells"].median())})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def cosine_profile_prob_full(x_scaled: np.ndarray, profiles: pd.DataFrame, temperature: float) -> np.ndarray:
    out = np.full((x_scaled.shape[0], len(SUBTYPE_ORDER)), np.nan, dtype=np.float32)
    if profiles.empty:
        return out
    x = x_scaled.astype(np.float32)
    p = profiles.to_numpy(dtype=np.float32)
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    p_norm = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-6)
    prob = softmax_rows(x_norm @ p_norm.T, temperature)
    for j, label in enumerate(profiles.index.astype(str)):
        if label in SUBTYPE_ORDER:
            out[:, SUBTYPE_ORDER.index(label)] = prob[:, j]
    return out


def combine_prob(marker_prob: np.ndarray, profile_prob: np.ndarray, marker_weight: float, profile_weight: float) -> np.ndarray:
    out = np.zeros_like(marker_prob, dtype=np.float32)
    for j in range(marker_prob.shape[1]):
        parts = [marker_prob[:, j] * float(marker_weight)]
        weights = [float(marker_weight)]
        if np.isfinite(profile_prob[:, j]).any():
            prof = np.nan_to_num(profile_prob[:, j], nan=0.0)
            parts.append(prof * float(profile_weight))
            weights.append(float(profile_weight))
        out[:, j] = np.sum(parts, axis=0) / max(sum(weights), 1e-8)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out


def add_support_columns(query: pd.DataFrame, marker_scores: pd.DataFrame, marker_prob: np.ndarray, profile_prob: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    out = query.drop(columns=["_row_order"], errors="ignore").reset_index(drop=True).copy()
    anchored_prob = combine_prob(marker_prob, profile_prob, args.marker_weight, args.profile_weight)
    for i, label in enumerate(SUBTYPE_ORDER):
        out[f"marker_prob_{label}"] = marker_prob[:, i]
        out[f"profile_prob_{label}"] = profile_prob[:, i]
        out[f"anchored_prob_{label}"] = anchored_prob[:, i]
    for col in marker_scores.columns:
        out[col] = marker_scores[col].to_numpy()
    marker_top_idx = marker_prob.argmax(axis=1)
    anchored_top_idx = anchored_prob.argmax(axis=1)
    marker_sorted = np.sort(marker_prob, axis=1)
    marker_margin = marker_sorted[:, -1] - marker_sorted[:, -2]
    out["marker_top_label"] = np.asarray(SUBTYPE_ORDER, dtype=object)[marker_top_idx]
    out["anchored_top_label"] = np.asarray(SUBTYPE_ORDER, dtype=object)[anchored_top_idx]
    out["marker_max_probability"] = marker_prob.max(axis=1)
    out["marker_probability_margin"] = marker_margin

    original = out["original_canonical_label"].where(out["original_canonical_label"].notna(), "").astype(str).to_numpy()
    original_idx = np.array([SUBTYPE_ORDER.index(x) if x in SUBTYPE_ORDER else -1 for x in original])
    has_original = original_idx >= 0
    original_marker_prob = np.full(len(out), np.nan, dtype=np.float32)
    original_profile_prob = np.full(len(out), np.nan, dtype=np.float32)
    original_anchored_prob = np.full(len(out), np.nan, dtype=np.float32)
    original_marker_prob[has_original] = marker_prob[np.where(has_original)[0], original_idx[has_original]]
    original_profile_prob[has_original] = profile_prob[np.where(has_original)[0], original_idx[has_original]]
    original_anchored_prob[has_original] = anchored_prob[np.where(has_original)[0], original_idx[has_original]]
    out["original_marker_probability"] = original_marker_prob
    out["original_profile_probability"] = original_profile_prob
    out["original_anchored_probability"] = original_anchored_prob

    marker_match = has_original & (out["marker_top_label"].to_numpy() == original)
    anchored_match = has_original & (out["anchored_top_label"].to_numpy() == original)
    prob_supported = has_original & (
        (np.nan_to_num(original_marker_prob, nan=0.0) >= float(args.support_threshold))
        | (np.nan_to_num(original_profile_prob, nan=0.0) >= float(args.support_threshold))
        | (np.nan_to_num(original_anchored_prob, nan=0.0) >= float(args.support_threshold))
    )
    support_level = np.full(len(out), "low_marker_resolution", dtype=object)
    support_level[has_original] = "needs_review"
    support_level[has_original & prob_supported] = "prob_supported"
    support_level[has_original & marker_match] = "marker_supported"
    support_level[has_original & anchored_match] = "top_match"
    no_original = ~has_original
    high_marker = (out["marker_max_probability"].to_numpy() >= float(args.marker_confidence_threshold)) & (
        out["marker_probability_margin"].to_numpy() >= float(args.marker_margin_threshold)
    )
    moderate_marker = out["marker_max_probability"].to_numpy() >= float(args.support_threshold)
    support_level[no_original & moderate_marker] = "moderate_marker_confidence"
    support_level[no_original & high_marker] = "high_marker_confidence"
    out["support_level"] = support_level
    out["retained_vascular_label"] = np.where(has_original, original, out["marker_top_label"].astype(str))
    out["has_original_subtype"] = has_original
    out["needs_review"] = support_level == "needs_review"
    return out


def rate(num: float, denom: float) -> float:
    return float(num / denom) if denom and np.isfinite(denom) and denom > 0 else np.nan


def chip_summary(source: pd.DataFrame, area: pd.DataFrame, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    area_map = area.set_index("sample_id").to_dict(orient="index")
    rows = []
    for sample_id, sub in source.groupby("sample_id", observed=False):
        area_row = area_map[str(sample_id)]
        n_all = int(area_row["n_all_cell_coords"])
        tissue_cm2 = float(area_row["tissue_area_cm2"])
        row = {
            "sample_id": sample_id,
            "n_vascular_anchors": int(len(sub)),
            "n_all_cell_coords": n_all,
            "tissue_area_coord2": float(area_row["tissue_area_coord2"]),
            "tissue_area_cm2": tissue_cm2,
            "area_source": str(area_row["area_source"]),
            "age_years": float(pd.to_numeric(sub["age_years"], errors="coerce").dropna().iloc[0]) if pd.to_numeric(sub["age_years"], errors="coerce").notna().any() else np.nan,
            "age_group": sub["age_group"].astype(str).mode().iloc[0],
            "disease_group": sub["disease_group"].astype(str).mode().iloc[0],
            "total_vascular_whole_chip_fraction": rate(len(sub), n_all),
            "total_vascular_density_cm2": rate(len(sub), tissue_cm2),
            "all_cell_density_cm2": rate(n_all, tissue_cm2),
            "has_original_subtype_fraction": float(sub["has_original_subtype"].astype(bool).mean()),
            "needs_review_fraction": float(sub["needs_review"].astype(bool).mean()),
            "mean_marker_max_probability": float(sub["marker_max_probability"].mean()),
            "mean_marker_probability_margin": float(sub["marker_probability_margin"].mean()),
        }
        for label in SUBTYPE_ORDER:
            original_mask = sub["original_canonical_label"].astype(str).eq(label)
            marker_mask = sub["marker_top_label"].astype(str).eq(label)
            retained_mask = sub["retained_vascular_label"].astype(str).eq(label)
            supported_original_mask = original_mask & ~sub["needs_review"].astype(bool)
            row[f"original_count_{label}"] = int(original_mask.sum())
            row[f"original_within_vascular_fraction_{label}"] = rate(original_mask.sum(), len(sub))
            row[f"original_whole_chip_fraction_{label}"] = rate(original_mask.sum(), n_all)
            row[f"original_density_cm2_{label}"] = rate(original_mask.sum(), tissue_cm2)
            row[f"supported_original_count_{label}"] = int(supported_original_mask.sum())
            row[f"supported_original_whole_chip_fraction_{label}"] = rate(supported_original_mask.sum(), n_all)
            row[f"supported_original_density_cm2_{label}"] = rate(supported_original_mask.sum(), tissue_cm2)
            row[f"marker_top_count_{label}"] = int(marker_mask.sum())
            row[f"marker_top_within_vascular_fraction_{label}"] = rate(marker_mask.sum(), len(sub))
            row[f"marker_top_whole_chip_fraction_{label}"] = rate(marker_mask.sum(), n_all)
            row[f"marker_top_density_cm2_{label}"] = rate(marker_mask.sum(), tissue_cm2)
            row[f"retained_count_{label}"] = int(retained_mask.sum())
            row[f"retained_whole_chip_fraction_{label}"] = rate(retained_mask.sum(), n_all)
            row[f"retained_density_cm2_{label}"] = rate(retained_mask.sum(), tissue_cm2)
            row[f"mean_marker_score_{label}"] = float(pd.to_numeric(sub[f"marker_score_{label}"], errors="coerce").mean())
            row[f"mean_marker_prob_{label}"] = float(pd.to_numeric(sub[f"marker_prob_{label}"], errors="coerce").mean())
            row[f"mean_anchored_prob_{label}"] = float(pd.to_numeric(sub[f"anchored_prob_{label}"], errors="coerce").mean())
        for module in ["BBB_capillary_state", "Endothelial_activation_state", "Mural_contractile_state"]:
            row[f"mean_marker_score_{module}"] = float(pd.to_numeric(sub[f"marker_score_{module}"], errors="coerce").mean())
        rows.append(row)
    comp = pd.DataFrame(rows).sort_values(["disease_group", "age_years", "sample_id"], na_position="last")
    comp.to_csv(result_dir / "figure3_spatial_v18_chip_summary.csv", index=False)

    long_rows = []
    for _, row in comp.iterrows():
        base = {
            "sample_id": row["sample_id"],
            "disease_group": row["disease_group"],
            "age_years": row["age_years"],
            "age_group": row["age_group"],
            "n_vascular_anchors": row["n_vascular_anchors"],
            "n_all_cell_coords": row["n_all_cell_coords"],
            "tissue_area_cm2": row["tissue_area_cm2"],
        }
        long_rows.append({**base, "level": "total_vascular", "label": "Vascular", "metric": "whole_chip_fraction", "value": row["total_vascular_whole_chip_fraction"]})
        long_rows.append({**base, "level": "total_vascular", "label": "Vascular", "metric": "density_cm2", "value": row["total_vascular_density_cm2"]})
        for label in SUBTYPE_ORDER:
            for level, prefix in [
                ("original", "original"),
                ("supported_original", "supported_original"),
                ("marker_top", "marker_top"),
                ("retained", "retained"),
            ]:
                for metric, suffix in [
                    ("whole_chip_fraction", "whole_chip_fraction"),
                    ("density_cm2", "density_cm2"),
                ]:
                    col = f"{prefix}_{suffix}_{label}"
                    if col in comp:
                        long_rows.append({**base, "level": level, "label": label, "metric": metric, "value": row[col]})
                frac_col = f"{prefix}_within_vascular_fraction_{label}"
                if frac_col in comp:
                    long_rows.append({**base, "level": level, "label": label, "metric": "within_vascular_fraction_qc", "value": row[frac_col]})
    comp_long = pd.DataFrame(long_rows)
    comp_long.to_csv(result_dir / "figure3_spatial_v18_chip_summary_long.csv", index=False)
    return comp, comp_long


def disease_stats(comp: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    exclude = {"sample_id", "disease_group", "age_group", "area_source"}
    features = [
        c
        for c in comp.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(comp[c])
        and not c.endswith("_count_Endo")
        and not c.endswith("_count_Pericyte")
        and not c.endswith("_count_SMC")
        and not c.endswith("_count_Fibroblast")
        and c not in {"n_vascular_anchors", "n_all_cell_coords", "tissue_area_coord2", "tissue_area_cm2", "age_years"}
    ]
    rows = []
    for feature in features:
        ad = pd.to_numeric(comp.loc[comp["disease_group"].eq("AD"), feature], errors="coerce").dropna()
        control = pd.to_numeric(comp.loc[comp["disease_group"].eq("Control"), feature], errors="coerce").dropna()
        if len(ad) < 2 or len(control) < 2:
            continue
        welch_p = np.nan
        mw_p = np.nan
        if v14.scipy_stats is not None:
            welch_p = float(v14.scipy_stats.ttest_ind(ad, control, equal_var=False, nan_policy="omit").pvalue)
            mw_p = float(v14.scipy_stats.mannwhitneyu(ad, control, alternative="two-sided").pvalue)
        rows.append(
            {
                "comparison": "AD_vs_Control",
                "feature": feature,
                "n_ad_chips": int(len(ad)),
                "n_control_chips": int(len(control)),
                "mean_ad": float(ad.mean()),
                "mean_control": float(control.mean()),
                "delta_ad_minus_control": float(ad.mean() - control.mean()),
                "cohens_d_ad_minus_control": v14.cohen_d(ad, control),
                "welch_t_p": welch_p,
                "mannwhitney_p": mw_p,
                "permutation_p": v14.permutation_pvalue(ad.to_numpy(float), control.to_numpy(float), n_perm=5000),
            }
        )
    out = pd.DataFrame(rows)
    for p_col in ["welch_t_p", "mannwhitney_p", "permutation_p"]:
        if p_col in out:
            out[f"{p_col}_bh_fdr"] = v14.bh_fdr(out[p_col])
    if not out.empty:
        out["abs_delta"] = out["delta_ad_minus_control"].abs()
        out = out.sort_values(["abs_delta", "feature"], ascending=[False, True]).drop(columns=["abs_delta"])
    out.to_csv(result_dir / "figure3_spatial_v18_disease_stats.csv", index=False)
    return out


def age_trend_stats(comp: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    exclude = {"sample_id", "disease_group", "age_group", "area_source"}
    features = [
        c
        for c in comp.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(comp[c])
        and not c.startswith(("original_count_", "supported_original_count_", "marker_top_count_", "retained_count_"))
        and c not in {"n_vascular_anchors", "n_all_cell_coords", "tissue_area_coord2", "tissue_area_cm2", "age_years"}
    ]
    rows = []
    age = pd.to_numeric(comp["age_years"], errors="coerce")
    for feature in features:
        y = pd.to_numeric(comp[feature], errors="coerce")
        valid = age.notna() & y.notna()
        if int(valid.sum()) < 5 or age[valid].nunique() < 3:
            continue
        x = age[valid].to_numpy(float)
        yy = y[valid].to_numpy(float)
        if np.nanstd(yy) <= 1e-12:
            continue
        slope, intercept = np.polyfit(x, yy, deg=1)
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        if v14.scipy_stats is not None:
            pearson_r, pearson_p = v14.scipy_stats.pearsonr(x, yy)
            spearman_r, spearman_p = v14.scipy_stats.spearmanr(x, yy)
        rows.append(
            {
                "comparison": "age_years_trend",
                "feature": feature,
                "n_chips_with_age": int(valid.sum()),
                "age_min": float(x.min()),
                "age_max": float(x.max()),
                "linear_slope_per_year": float(slope),
                "linear_intercept": float(intercept),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            }
        )
    out = pd.DataFrame(rows)
    for p_col in ["pearson_p", "spearman_p"]:
        if p_col in out:
            out[f"{p_col}_bh_fdr"] = v14.bh_fdr(out[p_col])
    if not out.empty:
        out["abs_spearman"] = out["spearman_r"].abs()
        out = out.sort_values(["abs_spearman", "feature"], ascending=[False, True]).drop(columns=["abs_spearman"])
    out.to_csv(result_dir / "figure3_spatial_v18_age_trend_stats.csv", index=False)
    return out


def marker_qc(source: pd.DataFrame, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = source[source["has_original_subtype"].astype(bool)].copy()
    qc_rows = []
    confusion = pd.DataFrame()
    if not labeled.empty and labeled["original_canonical_label"].nunique() >= 2:
        confusion = pd.crosstab(labeled["original_canonical_label"], labeled["marker_top_label"], normalize="index").reset_index()
        confusion.to_csv(result_dir / "figure3_spatial_v18_marker_top_confusion_by_original.csv", index=False)
        for label in SUBTYPE_ORDER:
            if label not in set(labeled["original_canonical_label"]):
                continue
            pos = labeled["original_canonical_label"].astype(str).eq(label)
            for value_col in [f"marker_score_{label}", f"marker_prob_{label}", f"anchored_prob_{label}"]:
                y_pos = pd.to_numeric(labeled.loc[pos, value_col], errors="coerce").dropna()
                y_neg = pd.to_numeric(labeled.loc[~pos, value_col], errors="coerce").dropna()
                if len(y_pos) < 2 or len(y_neg) < 2:
                    continue
                pval = np.nan
                auc = np.nan
                if v14.scipy_stats is not None:
                    pval = float(v14.scipy_stats.mannwhitneyu(y_pos, y_neg, alternative="two-sided").pvalue)
                    u = float(v14.scipy_stats.mannwhitneyu(y_pos, y_neg, alternative="greater").statistic)
                    auc = u / (len(y_pos) * len(y_neg))
                qc_rows.append(
                    {
                        "label": label,
                        "metric": value_col,
                        "n_positive": int(len(y_pos)),
                        "n_negative": int(len(y_neg)),
                        "mean_positive": float(y_pos.mean()),
                        "mean_negative": float(y_neg.mean()),
                        "delta_positive_minus_negative": float(y_pos.mean() - y_neg.mean()),
                        "cohens_d": v14.cohen_d(y_pos, y_neg),
                        "mannwhitney_p": pval,
                        "rank_auc_positive_gt_negative": auc,
                    }
                )
    qc = pd.DataFrame(qc_rows)
    if not qc.empty:
        qc["mannwhitney_p_bh_fdr"] = v14.bh_fdr(qc["mannwhitney_p"])
    qc.to_csv(result_dir / "figure3_spatial_v18_marker_qc_by_original_label.csv", index=False)
    return qc, confusion


def dotplot_source(source: pd.DataFrame, log_expr: np.ndarray, genes: list[str], result_dir: Path) -> pd.DataFrame:
    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    selected = []
    for module, module_genes in DOTPLOT_GENES.items():
        for gene in module_genes:
            if gene.upper() in gene_to_idx:
                selected.append((module, gene.upper(), gene_to_idx[gene.upper()]))
    if not selected:
        return pd.DataFrame()
    group_col = "original_canonical_label" if source["has_original_subtype"].mean() > 0.5 else "marker_top_label"
    groups = [x for x in SUBTYPE_ORDER if x in set(source[group_col].dropna().astype(str))]
    rows = []
    for module, gene, idx in selected:
        values = log_expr[:, idx]
        for group in groups:
            mask = source[group_col].astype(str).eq(group).to_numpy()
            vals = values[mask]
            rows.append(
                {
                    "group_col": group_col,
                    "group": group,
                    "module": module,
                    "gene": gene,
                    "mean_log1p": float(np.nanmean(vals)) if len(vals) else np.nan,
                    "pct_detected": float(np.mean(vals > 0)) if len(vals) else np.nan,
                    "n_cells": int(len(vals)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "figure3_spatial_v18_marker_dotplot_source.csv", index=False)
    return out


def plot_group_bar(comp_long: pd.DataFrame, level: str, metric: str, ylabel: str, title: str, prefix: Path, dpi: int) -> None:
    sub = comp_long[comp_long["level"].eq(level) & comp_long["metric"].eq(metric) & comp_long["label"].isin(SUBTYPE_ORDER)].copy()
    if sub.empty or sub["disease_group"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(4.65, 2.9))
    x = np.arange(len(SUBTYPE_ORDER), dtype=float)
    width = 0.33
    for i, disease in enumerate(["Control", "AD"]):
        means, sems = [], []
        for label in SUBTYPE_ORDER:
            vals = sub.loc[sub["disease_group"].eq(disease) & sub["label"].eq(label), "value"].dropna()
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0.0)
            jitter = (i - 0.5) * width + np.linspace(-0.045, 0.045, max(len(vals), 1))[: len(vals)]
            ax.scatter(np.full(len(vals), x[SUBTYPE_ORDER.index(label)] + (i - 0.5) * width) + jitter, vals, s=7, lw=0.5, facecolor="white", edgecolor=DISEASE_PALETTE[disease], alpha=0.9, zorder=3)
        ax.bar(x + (i - 0.5) * width, means, width=width, yerr=sems, color=DISEASE_PALETTE[disease], alpha=0.9, edgecolor="none", capsize=2, label=disease)
    ax.set_xticks(x)
    ax.set_xticklabels([SUBTYPE_DISPLAY[x] for x in SUBTYPE_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)
    save_all(fig, prefix, dpi)


def plot_total_group_bar(comp: pd.DataFrame, y_col: str, ylabel: str, title: str, prefix: Path, dpi: int) -> None:
    if comp["disease_group"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(2.45, 2.75))
    x = np.arange(2)
    for i, disease in enumerate(["Control", "AD"]):
        vals = pd.to_numeric(comp.loc[comp["disease_group"].eq(disease), y_col], errors="coerce").dropna()
        ax.bar(i, vals.mean(), yerr=vals.sem() if len(vals) > 1 else 0, width=0.55, color=DISEASE_PALETTE[disease], capsize=2)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, max(len(vals), 1))[: len(vals)], vals, s=8, lw=0.5, facecolor="white", edgecolor=DISEASE_PALETTE[disease], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Control", "AD"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold")
    ax.set_ylim(bottom=0)
    save_all(fig, prefix, dpi)


def plot_age_trend(comp: pd.DataFrame, features: list[str], labels: list[str], ylabel: str, title: str, prefix: Path, dpi: int) -> None:
    sub = comp[pd.to_numeric(comp["age_years"], errors="coerce").notna()].copy()
    if sub.empty:
        return
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n, 2.45), squeeze=False)
    for ax, feature, label in zip(axes.ravel(), features, labels):
        x = pd.to_numeric(sub["age_years"], errors="coerce")
        y = pd.to_numeric(sub[feature], errors="coerce")
        valid = x.notna() & y.notna()
        colors = [AGE_GROUP_PALETTE.get(age_group(v), "#7E8FA6") for v in x[valid]]
        ax.scatter(x[valid], y[valid], s=12, c=colors, lw=0.4, edgecolor="white", alpha=0.92)
        if valid.sum() >= 3:
            slope, intercept = np.polyfit(x[valid].to_numpy(float), y[valid].to_numpy(float), 1)
            xx = np.linspace(float(x[valid].min()), float(x[valid].max()), 100)
            ax.plot(xx, intercept + slope * xx, color="#333333", lw=0.9)
        ax.set_title(label, fontsize=7.5)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(ylabel if ax is axes.ravel()[0] else "")
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=8.5, fontweight="bold")
    save_all(fig, prefix, dpi)


def add_gene_scaled_mean(src: pd.DataFrame) -> pd.DataFrame:
    out = src.copy()
    scaled = []
    for _, sub in out.groupby("gene", sort=False):
        vals = pd.to_numeric(sub["mean_log1p"], errors="coerce").to_numpy(float)
        mean = np.nanmean(vals)
        sd = np.nanstd(vals)
        if not np.isfinite(sd) or sd <= 1e-8:
            z = np.zeros(len(sub), dtype=float)
        else:
            z = (vals - mean) / sd
        scaled.extend(np.clip(z, -2.0, 2.0).tolist())
    out["scaled_mean_log1p"] = scaled
    return out


def plot_marker_dotplot(src: pd.DataFrame, prefix: Path, dpi: int, scaled: bool = False) -> None:
    if src.empty:
        return
    src = add_gene_scaled_mean(src)
    genes = src["gene"].drop_duplicates().tolist()
    groups = [x for x in SUBTYPE_ORDER if x in set(src["group"])]
    if not genes or not groups:
        return
    src = src.copy()
    src["x"] = src["group"].map({g: i for i, g in enumerate(groups)})
    src["y"] = src["gene"].map({g: i for i, g in enumerate(genes[::-1])})
    fig, ax = plt.subplots(figsize=(0.62 * len(groups) + 1.1, 0.22 * len(genes) + 1.0))
    color_col = "scaled_mean_log1p" if scaled else "mean_log1p"
    if scaled:
        sc = ax.scatter(
            src["x"],
            src["y"],
            s=20 + 85 * src["pct_detected"],
            c=src[color_col],
            cmap="RdBu_r",
            vmin=-2,
            vmax=2,
            edgecolor="#D9D9D9",
            lw=0.25,
        )
    else:
        sc = ax.scatter(src["x"], src["y"], s=20 + 85 * src["pct_detected"], c=src[color_col], cmap="viridis", edgecolor="#D9D9D9", lw=0.25)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([SUBTYPE_DISPLAY.get(g, g) for g in groups], rotation=30, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes[::-1])
    title = "Vascular marker expression QC (scaled)" if scaled else "Vascular marker expression QC"
    ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Scaled mean log1p" if scaled else "Mean log1p")
    save_all(fig, prefix, dpi)


def sorted_samples_for_map(comp: pd.DataFrame, comparison_mode: str) -> list[str]:
    tmp = comp.copy()
    if comparison_mode == "age":
        tmp["sort_age"] = pd.to_numeric(tmp["age_years"], errors="coerce").fillna(9999)
        return tmp.sort_values(["sort_age", "sample_id"])["sample_id"].astype(str).tolist()
    order = {"Control": 0, "AD": 1, "Unknown": 2}
    tmp["sort_group"] = tmp["disease_group"].map(order).fillna(9)
    return tmp.sort_values(["sort_group", "sample_id"])["sample_id"].astype(str).tolist()


def plot_spatial_grid(table: pd.DataFrame, samples: list[str], color_key: str, palette: dict[str, str], prefix: Path, dpi: int, point_size: float, title_key: str = "") -> None:
    vals_all = table[color_key].astype(str)
    cats = [c for c in list(palette) if c in set(vals_all)] + [c for c in vals_all.value_counts().index if c not in palette]
    n = len(samples)
    ncols = 8 if n <= 40 else 10
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols, 1.32 * nrows), squeeze=False)
    for ax, sample in zip(axes.ravel(), samples):
        sub = table[table["sample_id"].astype(str).eq(sample)]
        vals = sub[color_key].astype(str).to_numpy()
        x = pd.to_numeric(sub["coord_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["coord_y"], errors="coerce").to_numpy(float)
        for cat in cats:
            idx = vals == cat
            if np.any(idx):
                ax.scatter(x[idx], y[idx], s=point_size, lw=0, c=[palette.get(cat, "#808080")], alpha=0.78, rasterized=True)
        sample_short = sample.split("/")[-1]
        if title_key and title_key in sub:
            title_val = sub[title_key].dropna().astype(str)
            if len(title_val):
                sample_short = f"{sample_short} ({title_val.iloc[0]})"
        ax.set_title(sample_short, fontsize=5.0, pad=1.0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get(c, "#808080"), markeredgewidth=0, markersize=3.5, label=SUBTYPE_DISPLAY.get(c, c)) for c in cats[:20]]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.002, 0.5), fontsize=5.2)
    fig.subplots_adjust(left=0.015, right=0.88, bottom=0.015, top=0.97, wspace=0.035, hspace=0.16)
    save_all(fig, prefix, dpi)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["No records."]
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4g}" if isinstance(val, float) and np.isfinite(val) else str(val))
        lines.append("|" + "|".join(vals) + "|")
    return lines


def write_report(result_dir: Path, summary: dict[str, object], stats: pd.DataFrame, comparison_mode: str) -> None:
    lines = [
        f"# Figure 3 spatial vascular density v18: {summary['analysis_id']}",
        "",
        "Main abundance metrics use whole-chip denominators: vascular anchors divided by all cells, and vascular anchors divided by physical chip area. One coordinate unit is treated as 0.5 um.",
        "",
        "## Run summary",
        "",
        f"- Cohort: {summary['cohort']}",
        f"- Vascular anchors: {summary['n_query_anchors']}",
        f"- Chips: {summary['n_chips']}",
        f"- Chips with finite age: {summary['n_chips_with_age']}",
        f"- Coordinate unit: {summary['coord_unit_um']} um",
        f"- Area sources: {summary['area_sources']}",
        f"- Marker genes: {summary['n_marker_genes']}",
        f"- Mean needs-review fraction: {summary['mean_needs_review_fraction']:.4f}",
        "",
        "## Key statistics",
        "",
    ]
    if comparison_mode == "disease":
        wanted = ["total_vascular_whole_chip_fraction", "total_vascular_density_cm2"]
        wanted += [f"original_whole_chip_fraction_{x}" for x in SUBTYPE_ORDER]
        wanted += [f"original_density_cm2_{x}" for x in SUBTYPE_ORDER]
        wanted += [f"marker_top_density_cm2_{x}" for x in SUBTYPE_ORDER]
        keep = stats[stats["feature"].isin(wanted)]
        cols = ["feature", "mean_ad", "mean_control", "delta_ad_minus_control", "cohens_d_ad_minus_control", "welch_t_p", "permutation_p"]
    else:
        wanted = ["total_vascular_whole_chip_fraction", "total_vascular_density_cm2"]
        wanted += [f"marker_top_whole_chip_fraction_{x}" for x in SUBTYPE_ORDER]
        wanted += [f"marker_top_density_cm2_{x}" for x in SUBTYPE_ORDER]
        keep = stats[stats["feature"].isin(wanted)]
        cols = ["feature", "n_chips_with_age", "linear_slope_per_year", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
    lines.extend(markdown_table(keep[cols] if not keep.empty else stats.head(20), 24))
    (result_dir / "Figure3_spatial_vascular_density_v18_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_dir, figure_dir = default_dirs(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if v14.MemmapDataset is None:
        raise ImportError("cellfm_dataset.memmap is required. Run under the remote OmniCell_NVU environment.")
    required = [args.annotation_csv, args.memmap_root / "gene_vocab.txt"]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    n_memmap_genes = len((args.memmap_root / "gene_vocab.txt").read_text().splitlines())
    gene_symbols = v14.load_gene_symbols(args.alias_csv, n_memmap_genes, args.memmap_root)
    gene_ids, genes, module_gene_df = marker_gene_ids(gene_symbols)
    if len(genes) < 20:
        raise ValueError(f"Too few matched marker genes: {len(genes)}")

    query = load_query(args.annotation_csv, args.cohort, args.comparison_mode, args.max_query_rows)
    print(f"[INFO] cohort={args.cohort}; anchors={len(query):,}; chips={query['sample_id'].nunique()}", flush=True)
    area = chip_area_table(query, args.memmap_root, args.coord_unit_um)
    area.to_csv(result_dir / "figure3_spatial_v18_chip_area_qc.csv", index=False)

    raw_expr = v14.fetch_expression(query, args.memmap_root, gene_ids)
    log_expr = np.log1p(np.nan_to_num(np.clip(raw_expr, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32)
    scaled_expr = robust_zscore(log_expr)
    marker_scores = compute_marker_scores(log_expr, genes)
    marker_score_mat = marker_scores[[f"marker_score_{x}" for x in SUBTYPE_ORDER]].to_numpy(dtype=np.float32)
    marker_prob = softmax_rows(marker_score_mat, args.marker_temperature)
    profiles, profile_diag = balanced_profiles(scaled_expr, query)
    if not profiles.empty:
        profiles.columns = genes
        profiles.to_csv(result_dir / "figure3_spatial_v18_original_label_profiles.csv")
    profile_prob = cosine_profile_prob_full(scaled_expr, profiles, args.profile_temperature)

    source = add_support_columns(query, marker_scores, marker_prob, profile_prob, args)
    source.to_csv(result_dir / "figure3_spatial_v18_vascular_anchor_source.csv.gz", index=False, compression="gzip")
    profile_diag.to_csv(result_dir / "figure3_spatial_v18_profile_diagnostics.csv", index=False)
    module_gene_df.to_csv(result_dir / "figure3_spatial_v18_marker_gene_availability.csv", index=False)
    pd.DataFrame({"gene": genes, "query_gene_id": gene_ids}).to_csv(result_dir / "figure3_spatial_v18_marker_genes.csv", index=False)

    comp, comp_long = chip_summary(source, area, result_dir)
    stats = disease_stats(comp, result_dir) if args.comparison_mode == "disease" else age_trend_stats(comp, result_dir)
    qc, _ = marker_qc(source, result_dir)
    dots = dotplot_source(source, log_expr, genes, result_dir)

    prefix = f"figure3_{args.analysis_id}_v18"
    if args.comparison_mode == "disease":
        plot_total_group_bar(comp, "total_vascular_whole_chip_fraction", "Vascular cells / all cells", "Whole-chip vascular proportion", figure_dir / f"{prefix}_total_vascular_whole_chip_fraction", args.dpi)
        plot_total_group_bar(comp, "total_vascular_density_cm2", "Cells per cm2", "Whole-chip vascular density", figure_dir / f"{prefix}_total_vascular_density_cm2", args.dpi)
        plot_group_bar(comp_long, "original", "whole_chip_fraction", "Cells / all chip cells", "Original subtype whole-chip proportion", figure_dir / f"{prefix}_original_whole_chip_fraction", args.dpi)
        plot_group_bar(comp_long, "original", "density_cm2", "Cells per cm2", "Original subtype density", figure_dir / f"{prefix}_original_density_cm2", args.dpi)
        plot_group_bar(comp_long, "marker_top", "density_cm2", "Cells per cm2", "Marker-top subtype density QC", figure_dir / f"{prefix}_marker_top_density_cm2", args.dpi)
    else:
        plot_age_trend(comp, ["total_vascular_whole_chip_fraction", "total_vascular_density_cm2"], ["Proportion", "Density"], "Value", "Cortex vascular change across age", figure_dir / f"{prefix}_total_vascular_age_trend", args.dpi)
        plot_age_trend(
            comp,
            [f"marker_top_density_cm2_{x}" for x in SUBTYPE_ORDER],
            [SUBTYPE_DISPLAY[x] for x in SUBTYPE_ORDER],
            "Cells per cm2",
            "Marker-suggested vascular subtype density across age",
            figure_dir / f"{prefix}_marker_top_density_age_trend",
            args.dpi,
        )
    plot_marker_dotplot(dots, figure_dir / f"{prefix}_marker_dotplot_qc_raw", args.dpi, scaled=False)
    plot_marker_dotplot(dots, figure_dir / f"{prefix}_marker_dotplot_qc_scaled", args.dpi, scaled=True)
    samples = sorted_samples_for_map(comp, args.comparison_mode)
    title_key = "age_group" if args.comparison_mode == "age" else "disease_group"
    plot_spatial_grid(source, samples, "original_label_for_quant", {**SUBTYPE_PALETTE, "Vascular": "#7F7F7F"}, figure_dir / f"{prefix}_all_chips_original_or_vascular_label", args.dpi, args.point_size, title_key=title_key)
    plot_spatial_grid(source, samples, "marker_top_label", SUBTYPE_PALETTE, figure_dir / f"{prefix}_all_chips_marker_top_label", args.dpi, args.point_size, title_key=title_key)
    plot_spatial_grid(source, samples, "support_level", SUPPORT_PALETTE, figure_dir / f"{prefix}_all_chips_support_level", args.dpi, args.point_size, title_key=title_key)

    summary = {
        "analysis_id": args.analysis_id,
        "method": "whole_chip_fraction_cm2_density_marker_qc_original_anchor_retained",
        "cohort": args.cohort,
        "comparison_mode": args.comparison_mode,
        "annotation_csv": str(args.annotation_csv),
        "memmap_root": str(args.memmap_root),
        "n_query_anchors": int(len(source)),
        "n_chips": int(source["sample_id"].nunique()),
        "n_chips_with_age": int(pd.to_numeric(comp["age_years"], errors="coerce").notna().sum()),
        "labels": SUBTYPE_ORDER,
        "n_marker_genes": int(len(genes)),
        "coord_unit_um": float(args.coord_unit_um),
        "area_sources": comp["area_source"].value_counts().to_dict(),
        "mean_total_vascular_whole_chip_fraction": float(comp["total_vascular_whole_chip_fraction"].mean()),
        "mean_total_vascular_density_cm2": float(comp["total_vascular_density_cm2"].mean()),
        "mean_needs_review_fraction": float(comp["needs_review_fraction"].mean()),
        "mean_marker_max_probability": float(comp["mean_marker_max_probability"].mean()),
        "has_original_subtype_fraction": float(source["has_original_subtype"].mean()),
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
    }
    (result_dir / "figure3_spatial_v18_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(result_dir, summary, stats, args.comparison_mode)
    print(json.dumps(summary, indent=2), flush=True)
    print("key_stats", flush=True)
    print(stats.head(24).to_string(index=False), flush=True)
    if not qc.empty:
        print("marker_qc", flush=True)
        print(qc.head(16).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
