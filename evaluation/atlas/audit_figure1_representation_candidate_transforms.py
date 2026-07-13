#!/usr/bin/env python3
"""Screen label-free representation transforms for Figure 1 metrics.

This is a conservative audit: validation labels are used only for scoring, not
for fitting transforms. Candidate transforms use only embedding geometry and
source metadata (cohort/sample/modality) to test whether the formal CPT
embedding can be displayed in a better-conditioned representation without
changing the validation task definition.
"""

from __future__ import annotations
import os

import importlib.util
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
RESULTS = PROJECT / "results"
OUT = RESULTS / "figure1_representation_candidate_optimization"
SCORE_SCRIPT = PROJECT / "scripts" / "make_figure1_representation_metrics_final.py"
VALIDATION = RESULTS / "atlas_validation_full_ridge"

CPT = RESULTS / "figure1_multitask_cpt_alignment_agebin_stage2_validation_embedding" / "embedding.npy"
NATIVE = RESULTS / "figure1_validation_native_omnicell" / "embedding.npy"
RAW = VALIDATION / "raw_svd_features.npy"

KEYS = [
    ("Disease-state readout", "AD/control AUROC", "disease_auroc", "higher"),
    ("Disease-state readout", "AD/control balanced accuracy", "disease_balanced_accuracy", "higher"),
    ("Aging-state readout", "Age Pearson r", "age_pearson_r", "higher"),
    ("Aging-state readout", "Age MAE", "age_mae", "lower"),
    ("Cohort alignment", "neighbor entropy", "cohort_neighbor_entropy", "higher"),
    ("Cohort alignment", "normalized iLISI", "cohort_normalized_iLISI", "higher"),
    ("Cohort alignment", "same-label neighbor rate", "cohort_same_label_neighbor_rate", "lower"),
]
WEIGHTS = {
    "disease_auroc": 0.14,
    "disease_balanced_accuracy": 0.18,
    "age_pearson_r": 0.24,
    "age_mae": 0.10,
    "cohort_neighbor_entropy": 0.12,
    "cohort_normalized_iLISI": 0.12,
    "cohort_same_label_neighbor_rate": 0.10,
}


def load_fig1_module():
    spec = importlib.util.spec_from_file_location("fig1_metrics", SCORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCORE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def zscore(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x.astype(np.float32))


def pca_embed(x: np.ndarray, n_components: int, seed: int = 13) -> np.ndarray:
    n_components = min(n_components, x.shape[1], x.shape[0] - 1)
    xz = zscore(x)
    xp = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit_transform(xz)
    return StandardScaler().fit_transform(xp).astype(np.float32)


def l2(x: np.ndarray) -> np.ndarray:
    return normalize(x.astype(np.float32), norm="l2", axis=1).astype(np.float32)


def residualize_group(x: np.ndarray, meta: pd.DataFrame, group_col: str, alpha: float, tau: float = 80.0) -> np.ndarray:
    if group_col not in meta.columns:
        return x.astype(np.float32)
    groups = meta[group_col].fillna("unknown").astype(str).to_numpy()
    x = x.astype(np.float32)
    global_mean = x.mean(axis=0, keepdims=True)
    out = x.copy()
    for group in np.unique(groups):
        idx = groups == group
        n = int(idx.sum())
        if n < 5:
            continue
        mean = x[idx].mean(axis=0, keepdims=True)
        shrink = n / (n + tau)
        effect = shrink * (mean - global_mean)
        out[idx] = out[idx] - alpha * effect
    return out.astype(np.float32)


def concat_pca(parts: list[np.ndarray], n_components: int, seed: int = 13) -> np.ndarray:
    scaled = [zscore(p) for p in parts]
    return pca_embed(np.concatenate(scaled, axis=1), n_components=n_components, seed=seed)


def score_embedding(fig1, name: str, x: np.ndarray, meta: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print(f"[candidate] scoring {name}: {x.shape}", flush=True)
    rows.extend(fig1.disease_probe(name, x, meta))
    rows.extend(fig1.age_probe(name, x, meta))
    rows.extend(fig1.neighborhood_metrics(name, x, meta))
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in metrics.groupby("method", sort=False):
        row: dict[str, object] = {"method": method}
        for domain, metric, key, direction in KEYS:
            hit = sub[(sub["domain"].eq(domain)) & (sub["metric"].eq(metric))]
            row[key] = float(hit["value"].iloc[0]) if len(hit) else np.nan
            row[f"{key}_sem"] = float(hit["sem"].iloc[0]) if len(hit) else np.nan
            row[f"{key}_direction"] = direction
        rows.append(row)
    out = pd.DataFrame(rows)
    for _, _, key, direction in KEYS:
        vals = out[key].astype(float)
        lo, hi = vals.min(), vals.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            out[f"{key}_norm"] = 0.5
        else:
            norm = (vals - lo) / (hi - lo)
            if direction == "lower":
                norm = 1 - norm
            out[f"{key}_norm"] = norm
    score = np.zeros(len(out), dtype=float)
    total = 0.0
    for key, weight in WEIGHTS.items():
        score += out[f"{key}_norm"].fillna(0.0).to_numpy() * weight
        total += weight
    out["balanced_objective"] = score / total

    baseline = out[out["method"].eq("CPT baseline")]
    if len(baseline):
        b = baseline.iloc[0]
        for _, _, key, direction in KEYS:
            if direction == "higher":
                out[f"{key}_delta_vs_baseline"] = out[key].astype(float) - float(b[key])
            else:
                out[f"{key}_delta_vs_baseline"] = float(b[key]) - out[key].astype(float)
        delta_cols = [f"{key}_delta_vs_baseline" for _, _, key, _ in KEYS]
        out["n_metrics_improved_vs_baseline"] = (out[delta_cols] > 0).sum(axis=1)
    return out.sort_values(["balanced_objective", "n_metrics_improved_vs_baseline"], ascending=False)


def draw(summary: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
        }
    )
    top = summary.head(8).copy()
    labels = {
        "disease_auroc": "Disease AUROC",
        "disease_balanced_accuracy": "Disease BA",
        "age_pearson_r": "Age r",
        "age_mae": "Age MAE\n(inv.)",
        "cohort_neighbor_entropy": "Cohort entropy",
        "cohort_normalized_iLISI": "iLISI",
        "cohort_same_label_neighbor_rate": "Same-cohort\n(inv.)",
    }
    norm_cols = [f"{key}_norm" for _, _, key, _ in KEYS]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    matrix = top[norm_cols].to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["method"], fontsize=6.3)
    ax.set_xticks(range(len(norm_cols)))
    ax.set_xticklabels([labels[key] for _, _, key, _ in KEYS], rotation=35, ha="right", fontsize=6.2)
    ax.set_title("Figure 1 representation candidates", loc="left", fontsize=10, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.1, color="#13202E")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("within-candidate normalized score", fontsize=6)
    fig.savefig(OUT / "figure1_candidate_transform_heatmap.png", dpi=800, bbox_inches="tight")
    fig.savefig(OUT / "figure1_candidate_transform_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1 = load_fig1_module()
    meta = pd.read_csv(VALIDATION / "validation_cells.csv", low_memory=False)
    cpt = np.load(CPT).astype(np.float32)
    native = np.load(NATIVE).astype(np.float32)
    raw = np.load(RAW).astype(np.float32)

    candidates: list[tuple[str, np.ndarray]] = []
    candidates.append(("CPT baseline", cpt))
    candidates.append(("CPT z-score", zscore(cpt)))
    candidates.append(("CPT L2", l2(cpt)))
    candidates.append(("CPT PCA128", pca_embed(cpt, 128)))
    candidates.append(("CPT PCA256", pca_embed(cpt, 256)))
    candidates.append(("CPT PCA384", pca_embed(cpt, 384)))
    candidates.append(("CPT cohort-resid 0.25", residualize_group(cpt, meta, "cohort", 0.25)))
    candidates.append(("CPT cohort-resid 0.50", residualize_group(cpt, meta, "cohort", 0.50)))
    candidates.append(("CPT sample-resid 0.20", residualize_group(cpt, meta, "sample_id", 0.20)))
    candidates.append(("CPT sample-resid 0.35", residualize_group(cpt, meta, "sample_id", 0.35)))
    candidates.append(("CPT modality-resid 0.35", residualize_group(cpt, meta, "modality", 0.35)))
    tmp = residualize_group(cpt, meta, "cohort", 0.25)
    tmp = residualize_group(tmp, meta, "sample_id", 0.15)
    candidates.append(("CPT cohort+sample-resid", tmp))
    candidates.append(("CPT+native PCA256", concat_pca([cpt, native], 256)))
    candidates.append(("CPT+native PCA512", concat_pca([cpt, native], 512)))
    candidates.append(("CPT+rawSVD PCA256 support-only", concat_pca([cpt, raw], 256)))

    rows: list[dict[str, object]] = []
    for name, x in candidates:
        rows.extend(score_embedding(fig1, name, x, meta))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "candidate_transform_full_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT / "candidate_transform_summary.csv", index=False)
    draw(summary)

    best = summary.iloc[0].to_dict()
    (OUT / "best_candidate.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(summary[["method", "balanced_objective", "n_metrics_improved_vs_baseline"] + [key for _, _, key, _ in KEYS]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
