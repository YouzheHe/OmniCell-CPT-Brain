#!/usr/bin/env python
"""Reference-profile deconvolution of spatial vascular anchors.

The single-cell reference is the v11 clean vascular annotation.  The spatial
input stores OmniCell embeddings in X, so this script uses the shared
expanded-marker expression matrix rather than the 512-d embedding as the
deconvolution signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.lines import Line2D
from scipy.optimize import nnls
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score


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


BASE = (
    "${OMNICELL_NVU_ROOT}/projects/nvu_vascular/results/"
    "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
)
FIG_BASE = (
    "${OMNICELL_NVU_ROOT}/projects/nvu_vascular/figures/"
    "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
)
DEFAULT_REF = f"{BASE}/vascular_clean_diagonal_v11/single_cell/single_cell_clean_vascular_v11.h5ad"
DEFAULT_SPATIAL = f"{BASE}/biology_annotation_v2/spatial/spatial_biology_annotation_v2.h5ad"
DEFAULT_RESULT_DIR = f"{BASE}/spatial_vascular_deconvolution_v12/spatial"
DEFAULT_FIGURE_DIR = f"{FIG_BASE}/spatial_vascular_deconvolution_v12/spatial"

MODULE_ORDER = ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"]
CLASS_PALETTE = {
    "Endothelial": "#4E79A7",
    "Pericyte": "#B07AA1",
    "SMC": "#E15759",
    "Fibroblast_VLMC": "#59A14F",
    "Low_confidence": "#BFC5CC",
}
DISTINCT_CLUSTER_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
]
EXCLUDE_GENES = {
    "GFAP",
    "AQP4",
    "PLP1",
    "MBP",
    "P2RY12",
    "CX3CR1",
    "RBFOX3",
    "SNAP25",
    "HBB",
    "HBA1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference-h5ad", default=DEFAULT_REF)
    p.add_argument("--spatial-h5ad", default=DEFAULT_SPATIAL)
    p.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    p.add_argument("--matrix-key", default="expanded_marker_log1p")
    p.add_argument("--gene-key", default="expanded_marker_genes")
    p.add_argument("--cluster-key", default="v11_clean_cluster")
    p.add_argument("--class-key", default="v11_marker_class")
    p.add_argument("--confidence-threshold", type=float, default=0.38)
    p.add_argument("--residual-threshold", type=float, default=0.78)
    p.add_argument("--dpi", type=int, default=800)
    return p.parse_args()


def save_all(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)


def get_marker_matrix(adata: sc.AnnData, matrix_key: str, gene_key: str) -> tuple[np.ndarray, list[str]]:
    if matrix_key not in adata.obsm:
        raise KeyError(f"{matrix_key} missing from obsm")
    if gene_key not in adata.uns:
        raise KeyError(f"{gene_key} missing from uns")
    return np.asarray(adata.obsm[matrix_key], dtype=np.float32), list(map(str, adata.uns[gene_key]))


def shared_genes(ref_genes: list[str], spatial_genes: list[str]) -> tuple[list[str], list[int], list[int]]:
    ref_idx = {g: i for i, g in enumerate(ref_genes)}
    sp_idx = {g: i for i, g in enumerate(spatial_genes)}
    genes = [g for g in ref_genes if g in sp_idx and g not in EXCLUDE_GENES]
    return genes, [ref_idx[g] for g in genes], [sp_idx[g] for g in genes]


def scale_matrices(ref_x: np.ndarray, sp_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = np.vstack([ref_x, sp_x])
    scale = np.nanpercentile(sample, 99, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-3)
    return np.clip(ref_x / scale, 0, 5), np.clip(sp_x / scale, 0, 5), scale


def build_profiles(ref: sc.AnnData, ref_x: np.ndarray, cluster_key: str, class_key: str) -> tuple[pd.DataFrame, pd.Series]:
    clusters = list(ref.obs[cluster_key].cat.categories) if hasattr(ref.obs[cluster_key], "cat") else sorted(ref.obs[cluster_key].astype(str).unique())
    labels = ref.obs[cluster_key].astype(str).to_numpy()
    rows = []
    cluster_class = {}
    for cluster in clusters:
        mask = labels == str(cluster)
        if not np.any(mask):
            continue
        rows.append(pd.Series(ref_x[mask].mean(axis=0), name=str(cluster)))
        klass = ref.obs.loc[mask, class_key].astype(str).mode()
        cluster_class[str(cluster)] = klass.iloc[0] if len(klass) else "Unknown"
    profile = pd.DataFrame(rows)
    return profile, pd.Series(cluster_class, name="class")


def run_nnls(sp_x: np.ndarray, profile: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = profile.to_numpy(dtype=np.float32).T
    n_spots = sp_x.shape[0]
    n_clusters = profile.shape[0]
    weights = np.zeros((n_spots, n_clusters), dtype=np.float32)
    residual = np.zeros(n_spots, dtype=np.float32)
    fitted_norm = np.zeros(n_spots, dtype=np.float32)
    for i in range(n_spots):
        y = sp_x[i].astype(np.float32)
        w, _ = nnls(a, y)
        pred = a @ w
        denom = max(float(np.linalg.norm(y)), 1e-6)
        residual[i] = float(np.linalg.norm(y - pred) / denom)
        fitted_norm[i] = float(np.linalg.norm(pred) / denom)
        s = float(w.sum())
        if s > 0:
            weights[i] = (w / s).astype(np.float32)
        if (i + 1) % 10000 == 0:
            print(f"NNLS {i + 1}/{n_spots}", flush=True)
    return weights, residual, fitted_norm


def entropy_rows(w: np.ndarray) -> np.ndarray:
    p = np.clip(w, 1e-8, 1.0)
    ent = -(p * np.log(p)).sum(axis=1)
    return ent / np.log(w.shape[1])


def add_deconv_obs(
    spatial: sc.AnnData,
    weights: np.ndarray,
    profile: pd.DataFrame,
    cluster_class: pd.Series,
    residual: np.ndarray,
    fitted_norm: np.ndarray,
    confidence_threshold: float,
    residual_threshold: float,
) -> pd.DataFrame:
    clusters = profile.index.astype(str).tolist()
    class_by_cluster = cluster_class.reindex(clusters).astype(str).to_dict()
    for j, cluster in enumerate(clusters):
        spatial.obs[f"deconv_cluster_{cluster}"] = weights[:, j]
    for klass in MODULE_ORDER:
        cols = [j for j, c in enumerate(clusters) if class_by_cluster.get(c) == klass]
        spatial.obs[f"deconv_class_{klass}"] = weights[:, cols].sum(axis=1) if cols else 0.0

    max_idx = weights.argmax(axis=1)
    max_w = weights[np.arange(weights.shape[0]), max_idx]
    dom_cluster = np.array([clusters[i] for i in max_idx], dtype=object)
    dom_class = np.array([class_by_cluster.get(c, "Unknown") for c in dom_cluster], dtype=object)
    ent = entropy_rows(weights)
    confident = (max_w >= confidence_threshold) & (residual <= residual_threshold)
    spatial.obs["deconv_dominant_cluster_raw"] = dom_cluster
    spatial.obs["deconv_dominant_class_raw"] = dom_class
    spatial.obs["deconv_dominant_cluster"] = np.where(confident, dom_cluster, "Low_confidence")
    spatial.obs["deconv_dominant_class"] = np.where(confident, dom_class, "Low_confidence")
    spatial.obs["deconv_confidence"] = max_w
    spatial.obs["deconv_entropy"] = ent
    spatial.obs["deconv_residual"] = residual
    spatial.obs["deconv_fitted_norm"] = fitted_norm
    spatial.obs["deconv_is_confident"] = confident

    out_cols = [
        "sample_id",
        "cohort",
        "condition_inferred",
        "coord_x",
        "coord_y",
        "deconv_dominant_cluster",
        "deconv_dominant_class",
        "deconv_confidence",
        "deconv_entropy",
        "deconv_residual",
    ]
    out_cols += [f"deconv_class_{k}" for k in MODULE_ORDER]
    out_cols += [f"deconv_cluster_{c}" for c in clusters]
    return spatial.obs[[c for c in out_cols if c in spatial.obs.columns]].copy()


def evaluate(spatial: sc.AnnData, result_dir: Path) -> pd.DataFrame:
    rows = []
    pred = spatial.obs["deconv_dominant_class"].astype(str)
    for key in ["bio_macro_label", "vascular_class", "cell_label_original", "cluster_annotation"]:
        if key not in spatial.obs:
            continue
        truth = spatial.obs[key].astype(str)
        mask = (~truth.isin(["nan", "Unknown", "Possible_contaminant", "Mixed_spot"])) & (pred != "Low_confidence")
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "reference_label": key,
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(truth[mask], pred[mask])),
                "balanced_accuracy": float(balanced_accuracy_score(truth[mask], pred[mask])),
                "macro_f1": float(f1_score(truth[mask], pred[mask], average="macro")),
                "nmi": float(normalized_mutual_info_score(truth[mask], pred[mask])),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "spatial_vascular_deconv_v12_label_agreement.csv", index=False)
    return df


def plot_umap(spatial: sc.AnnData, color_key: str, prefix: Path, title: str, dpi: int, palette: dict[str, str] | None = None) -> None:
    coords = spatial.obsm["X_umap"] if "X_umap" in spatial.obsm else spatial.obsm["X_omnicell"][:, :2]
    vals = spatial.obs[color_key].astype(str)
    cats = vals.value_counts().index.tolist()
    if color_key == "deconv_dominant_class":
        cats = [c for c in MODULE_ORDER + ["Low_confidence"] if c in set(vals)]
    colors = {}
    if palette:
        colors = {c: palette.get(c, "#808080") for c in cats}
    else:
        cmap = plt.get_cmap("tab20")
        colors = {c: cmap(i % 20) for i, c in enumerate(cats)}
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    for cat in cats:
        mask = vals.to_numpy() == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], s=1.0, lw=0, c=[colors[cat]], alpha=0.75, rasterized=True)
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=2)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[c], markeredgewidth=0, markersize=4, label=c) for c in cats[:24]]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.8)
    save_all(fig, prefix, dpi)
    plt.close(fig)


def selected_samples(spatial: sc.AnnData) -> list[str]:
    samples = set(spatial.obs["sample_id"].astype(str))
    desired = [
        "AD_Hip_Saptial/AD2.1",
        "AD_Hip_Saptial/AD2.2",
        "AD_Hip_Saptial/Con2.1",
        "AD_Hip_Saptial/Con2.2",
        "Cortex_Spatial/T917",
        "Cortex_Spatial/T991",
        "Cortex_Spatial/T989",
        "Cortex_Spatial/T988",
    ]
    found = [s for s in desired if s in samples]
    if found:
        return found
    return spatial.obs["sample_id"].astype(str).value_counts().head(8).index.tolist()


def plot_spatial_grid(
    spatial: sc.AnnData,
    samples: list[str],
    color_key: str,
    prefix: Path,
    title: str,
    dpi: int,
    palette: dict[str, str] | None = None,
) -> None:
    n = len(samples)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    vals_all = spatial.obs[color_key].astype(str)
    cats = vals_all.value_counts().index.tolist()
    if color_key == "deconv_dominant_class":
        cats = [c for c in MODULE_ORDER + ["Low_confidence"] if c in set(vals_all)]
    colors = palette or {c: plt.get_cmap("tab20")(i % 20) for i, c in enumerate(cats)}
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 2.0 * nrows), squeeze=False)
    for ax, sample in zip(axes.ravel(), samples):
        mask = spatial.obs["sample_id"].astype(str).to_numpy() == sample
        sub = spatial[mask]
        vals = sub.obs[color_key].astype(str).to_numpy()
        x = sub.obs["coord_x"].astype(float).to_numpy()
        y = sub.obs["coord_y"].astype(float).to_numpy()
        for cat in cats:
            idx = vals == cat
            if np.any(idx):
                ax.scatter(x[idx], y[idx], s=4.0, lw=0, c=[colors.get(cat, "#808080")], alpha=0.82, rasterized=True)
        ax.set_title(sample.split("/")[-1], fontsize=6.8, pad=1.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors.get(c, "#808080"), markeredgewidth=0, markersize=4, label=c) for c in cats[:24]]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=5.8)
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=8.4, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.02, top=0.88, wspace=0.06, hspace=0.13)
    save_all(fig, prefix, dpi)
    plt.close(fig)


def summarize_samples(spatial: sc.AnnData, result_dir: Path) -> pd.DataFrame:
    rows = []
    for sample, obs in spatial.obs.groupby(spatial.obs["sample_id"].astype(str), observed=False):
        row = {
            "sample_id": sample,
            "n_spots": int(len(obs)),
            "cohort": obs["cohort"].astype(str).mode().iloc[0] if "cohort" in obs else "",
            "condition_inferred": obs["condition_inferred"].astype(str).mode().iloc[0] if "condition_inferred" in obs else "",
            "mean_confidence": float(obs["deconv_confidence"].mean()),
            "mean_residual": float(obs["deconv_residual"].mean()),
            "confident_fraction": float(obs["deconv_is_confident"].mean()),
        }
        for klass in MODULE_ORDER:
            row[f"mean_{klass}"] = float(obs[f"deconv_class_{klass}"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "spatial_vascular_deconv_v12_sample_summary.csv", index=False)
    return df


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    ref = sc.read_h5ad(args.reference_h5ad)
    spatial = sc.read_h5ad(args.spatial_h5ad)
    ref_x, ref_genes = get_marker_matrix(ref, args.matrix_key, args.gene_key)
    sp_x, sp_genes = get_marker_matrix(spatial, args.matrix_key, args.gene_key)
    genes, ref_idx, sp_idx = shared_genes(ref_genes, sp_genes)
    ref_x = ref_x[:, ref_idx]
    sp_x = sp_x[:, sp_idx]
    ref_scaled, sp_scaled, scale = scale_matrices(ref_x, sp_x)

    profile, cluster_class = build_profiles(ref, ref_scaled, args.cluster_key, args.class_key)
    profile.columns = genes
    profile.to_csv(result_dir / "spatial_vascular_deconv_v12_reference_profiles.csv")
    pd.DataFrame({"gene": genes, "scale_p99": scale}).to_csv(result_dir / "spatial_vascular_deconv_v12_gene_scaling.csv", index=False)
    cluster_class.to_csv(result_dir / "spatial_vascular_deconv_v12_reference_cluster_classes.csv", header=True)

    weights, residual, fitted_norm = run_nnls(sp_scaled, profile)
    source = add_deconv_obs(
        spatial,
        weights,
        profile,
        cluster_class,
        residual,
        fitted_norm,
        args.confidence_threshold,
        args.residual_threshold,
    )
    source.to_csv(result_dir / "spatial_vascular_deconv_v12_source.csv.gz", index=True, compression="gzip")
    sample_summary = summarize_samples(spatial, result_dir)
    agreement = evaluate(spatial, result_dir)

    spatial.uns["deconv_v12_reference_clusters"] = profile.index.astype(str).tolist()
    spatial.uns["deconv_v12_reference_classes"] = cluster_class.reindex(profile.index).astype(str).tolist()
    spatial.uns["deconv_v12_genes"] = genes
    spatial.uns["deconv_v12_method"] = "NNLS on shared expanded-marker log1p profiles from clean v11 single-cell annotation"
    spatial.write_h5ad(result_dir / "spatial_vascular_deconv_v12.h5ad", compression="gzip")

    plot_umap(
        spatial,
        "deconv_dominant_class",
        figure_dir / "spatial_vascular_deconv_v12_umap_dominant_class",
        "Spatial vascular deconvolution: dominant class",
        args.dpi,
        palette=CLASS_PALETTE,
    )
    cluster_palette = {c: DISTINCT_CLUSTER_COLORS[i % len(DISTINCT_CLUSTER_COLORS)] for i, c in enumerate(profile.index.astype(str))}
    cluster_palette["Low_confidence"] = "#BFC5CC"
    plot_umap(
        spatial,
        "deconv_dominant_cluster",
        figure_dir / "spatial_vascular_deconv_v12_umap_dominant_cluster",
        "Spatial vascular deconvolution: dominant subtype",
        args.dpi,
        palette=cluster_palette,
    )
    samples = selected_samples(spatial)
    plot_spatial_grid(
        spatial,
        samples,
        "deconv_dominant_class",
        figure_dir / "spatial_vascular_deconv_v12_selected_chips_dominant_class",
        "Selected chips: dominant vascular class",
        args.dpi,
        palette=CLASS_PALETTE,
    )
    plot_spatial_grid(
        spatial,
        samples,
        "deconv_dominant_cluster",
        figure_dir / "spatial_vascular_deconv_v12_selected_chips_dominant_cluster",
        "Selected chips: dominant vascular subtype",
        args.dpi,
        palette=cluster_palette,
    )

    summary = {
        "reference_h5ad": args.reference_h5ad,
        "spatial_h5ad": args.spatial_h5ad,
        "n_reference_cells": int(ref.n_obs),
        "n_spatial_spots": int(spatial.n_obs),
        "n_reference_clusters": int(profile.shape[0]),
        "n_genes": int(len(genes)),
        "mean_confidence": float(spatial.obs["deconv_confidence"].mean()),
        "mean_residual": float(spatial.obs["deconv_residual"].mean()),
        "confident_fraction": float(spatial.obs["deconv_is_confident"].mean()),
        "selected_samples": samples,
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
    }
    (result_dir / "spatial_vascular_deconv_v12_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not agreement.empty:
        print("agreement")
        print(agreement.to_string(index=False), flush=True)
    print("sample_summary_head")
    print(sample_summary.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
