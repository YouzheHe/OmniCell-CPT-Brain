#!/usr/bin/env python
"""Build Figure 3 vascular-identification panels from final_results.

The script is intentionally strict about inputs: it reads the final Figure 3
result package and writes a self-contained set of 12 Python/matplotlib panels,
source data, marker-confirmation tables and a compact analysis report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


FINAL_FIGURE3_DIRNAME = "Figure3_vascular_final_20260706"

MACRO_ORDER = ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"]
FINE_ORDER = [
    "BBB_EC",
    "Capillary_EC",
    "Arterial_EC",
    "Venous_EC",
    "Activated_EC",
    "Endothelial_core",
    "Pericyte",
    "Contractile_mural",
    "SMC",
    "Fibroblast_VLMC",
]
CLUSTER_ORDER = [
    "EC00",
    "EC01",
    "EC02",
    "EC03",
    "EC04",
    "PC00",
    "PC01",
    "PC02",
    "SMC00",
    "VLMC00",
    "VLMC01",
    "VLMC02",
    "VLMC03",
]

MACRO_COLORS = {
    "Endothelial": "#4E79A7",
    "Pericyte": "#B07AA1",
    "SMC": "#E15759",
    "Fibroblast_VLMC": "#59A14F",
    "Mixed_spot": "#CFCFCF",
    "Unknown": "#8C8C8C",
}
FINE_COLORS = {
    "BBB_EC": "#36699F",
    "Capillary_EC": "#76A5D8",
    "Arterial_EC": "#9BC2E6",
    "Venous_EC": "#2F6F9F",
    "Activated_EC": "#E7A6A1",
    "Endothelial_core": "#4E79A7",
    "Pericyte": "#B07AA1",
    "Contractile_mural": "#D66DA1",
    "SMC": "#E15759",
    "Fibroblast_VLMC": "#59A14F",
}
CLUSTER_COLORS = {
    "EC00": "#36699F",
    "EC01": "#5B8CC0",
    "EC02": "#76A5D8",
    "EC03": "#2F6F9F",
    "EC04": "#9BC2E6",
    "PC00": "#B07AA1",
    "PC01": "#C799B8",
    "PC02": "#8E5D88",
    "SMC00": "#E15759",
    "VLMC00": "#59A14F",
    "VLMC01": "#86BC86",
    "VLMC02": "#3F8B4C",
    "VLMC03": "#9FCA7F",
}
GENE_MODULES = {
    "Endothelial": ["PECAM1", "CLDN5", "VWF", "RAMP2", "FLT1", "KDR", "CDH5", "ESAM", "SLC2A1"],
    "Pericyte": ["PDGFRB", "RGS5", "ABCC9", "NOTCH3", "CSPG4", "KCNJ8", "MCAM"],
    "SMC": ["ACTA2", "TAGLN", "MYH11", "CNN1", "MYLK", "MYOCD", "SMTN"],
    "VLMC/Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "APOD"],
    "BBB/Capillary EC": ["ABCB1", "MFSD2A", "RGCC", "EMCN", "PLVAP", "CA4"],
    "Arterial EC": ["GJA5", "HEY1", "EFNB2", "SOX17", "CXCR4", "DLL4"],
    "Venous/activated EC": ["ACKR1", "NR2F2", "ICAM1", "VCAM1", "SELE", "PLVAP"],
}
GENE_TO_MODULE = {gene: module for module, genes in GENE_MODULES.items() for gene in genes}
CANONICAL_MACRO_GENES = [
    "PECAM1",
    "CLDN5",
    "VWF",
    "RAMP2",
    "PDGFRB",
    "RGS5",
    "ABCC9",
    "NOTCH3",
    "ACTA2",
    "TAGLN",
    "MYH11",
    "CNN1",
    "COL1A1",
    "COL3A1",
    "DCN",
    "LUM",
]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Final Figure 3 results directory. Parent final_results is also accepted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <input-dir>/figure3_vascular_identification_python.",
    )
    parser.add_argument("--strict-final-results", action="store_true")
    parser.add_argument("--skip-findmarkers", action="store_true")
    parser.add_argument("--max-umap-cells", type=int, default=120_000)
    parser.add_argument("--max-spots-per-sample", type=int, default=16_000)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def resolve_input_dir(root: Path, input_dir: Path | None, strict: bool) -> Path:
    candidates: list[Path] = []
    if input_dir is not None:
        candidates.append(input_dir)
        candidates.append(input_dir / FINAL_FIGURE3_DIRNAME)
    candidates.extend(
        [
            root / "projects" / "nvu_vascular" / "final_results" / FINAL_FIGURE3_DIRNAME,
            root / "projects" / "nvu_vascular" / "final_results",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.name == "final_results":
            nested = candidate / FINAL_FIGURE3_DIRNAME
            if nested.exists():
                candidate = nested
        if (candidate / "01_single_cell").exists() and (candidate / "03_spatial_deconvolution_QC").exists():
            return candidate
    message = "Could not locate the final Figure 3 result directory."
    if strict:
        raise FileNotFoundError(message)
    raise FileNotFoundError(message + " Pass --input-dir explicitly.")


def rel_path(base: Path, rel: str, required: bool = True) -> Path | None:
    path = base / rel
    if path.exists():
        return path
    if required:
        raise FileNotFoundError(f"Required input not found: {path}")
    return None


def read_csv_required(base: Path, rel: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(rel_path(base, rel, required=True), **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_pub_py(fig: plt.Figure, prefix: Path, dpi: int = 450) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(prefix.with_suffix(f".{ext}"), **kwargs)
    return prefix.with_suffix(".png")


def clean_label(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return str(value)


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "item"


def ordered_values(values: Iterable[object], preferred: list[str]) -> list[str]:
    present = [clean_label(v) for v in pd.Series(list(values)).dropna().astype(str).unique()]
    ordered = [v for v in preferred if v in present]
    ordered.extend(sorted(v for v in present if v not in ordered))
    return ordered


def downsample(df: pd.DataFrame, max_n: int, seed: int = 17) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed)


def draw_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.06,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def placeholder(ax: plt.Axes, title: str, message: str) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
        wrap=True,
    )


def prepare_umap(umap: pd.DataFrame) -> pd.DataFrame:
    required = {"UMAP1", "UMAP2", "v11_marker_class", "v11_clean_cluster", "bio_fine_label"}
    missing = required - set(umap.columns)
    if missing:
        raise ValueError(f"UMAP source is missing columns: {sorted(missing)}")
    out = umap.copy()
    out["broad_class"] = out["v11_marker_class"].map(clean_label)
    out["subtype_label"] = out["bio_fine_label"].map(clean_label)
    out["cluster_label"] = out["v11_clean_cluster"].map(clean_label)
    return out


def normalize_dotplot(dot: pd.DataFrame, group_order: list[str] | None = None) -> pd.DataFrame:
    out = dot.copy()
    rename = {
        "fraction_positive": "pct_expressing",
        "scaled_mean_log1p": "scaled_mean",
        "mean_log1p": "mean_log1p",
    }
    out = out.rename(columns=rename)
    if "pct_expressing" not in out.columns:
        out["pct_expressing"] = 0.0
    if "scaled_mean" not in out.columns:
        out["scaled_mean"] = out.groupby("gene")["mean_log1p"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        )
    out["group"] = out["group"].map(clean_label)
    out["gene"] = out["gene"].map(clean_label)
    if group_order is not None:
        out["group"] = pd.Categorical(out["group"], categories=[g for g in group_order if g in set(out["group"])], ordered=True)
        out = out.sort_values(["group", "module", "gene"])
        out["group"] = out["group"].astype(str)
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if finite.sum() == 0:
        return out
    p = pvalues[finite]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[finite] = tmp
    return out


def marker_panel_findmarkers(
    h5ad_path: Path,
    source_dir: Path,
    group_cols: list[str],
    min_cells: int = 50,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """Run a marker-panel FindAllMarkers-like confirmation on expanded markers.

    The final h5ad stores marker expression in obsm["expanded_marker_log1p"] and
    the gene names in uns["expanded_marker_genes"]. This is a marker-panel
    confirmation, not a genome-wide DE run.
    """
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("scanpy is required for marker confirmation") from exc

    try:
        from scipy.stats import ranksums
    except Exception:  # pragma: no cover - environment specific
        ranksums = None

    adata = sc.read_h5ad(h5ad_path)
    if "expanded_marker_log1p" not in adata.obsm or "expanded_marker_genes" not in adata.uns:
        raise ValueError("h5ad lacks expanded marker expression required for marker confirmation.")
    expr = np.asarray(adata.obsm["expanded_marker_log1p"], dtype=np.float32)
    genes = np.asarray(adata.uns["expanded_marker_genes"], dtype=str)
    obs = adata.obs.copy()

    marker_tables: dict[str, pd.DataFrame] = {}
    dot_tables: dict[str, pd.DataFrame] = {}
    summary_records: list[dict[str, object]] = []

    for group_col in group_cols:
        if group_col not in obs.columns:
            continue
        labels = obs[group_col].astype("string").fillna("Unknown").astype(str).to_numpy()
        groups = [g for g in ordered_values(labels, FINE_ORDER + CLUSTER_ORDER + MACRO_ORDER) if (labels == g).sum() >= min_cells]
        records: list[dict[str, object]] = []
        for group in groups:
            mask = labels == group
            n_in = int(mask.sum())
            n_out = int((~mask).sum())
            if n_in < min_cells or n_out < min_cells:
                continue
            x_in = expr[mask]
            x_out = expr[~mask]
            mean_in = x_in.mean(axis=0)
            mean_out = x_out.mean(axis=0)
            pct_in = (x_in > 0).mean(axis=0)
            pct_out = (x_out > 0).mean(axis=0)
            delta = mean_in - mean_out
            pvals = np.full(len(genes), np.nan, dtype=float)
            if ranksums is not None:
                for i in range(len(genes)):
                    try:
                        pvals[i] = ranksums(x_in[:, i], x_out[:, i]).pvalue
                    except Exception:
                        pvals[i] = np.nan
            padj = bh_adjust(pvals)
            score = delta * np.sqrt(np.maximum(pct_in, 1e-6)) * np.maximum(pct_in - pct_out + 0.05, 0.05)
            for i, gene in enumerate(genes):
                records.append(
                    {
                        "group_col": group_col,
                        "group": group,
                        "gene": gene,
                        "module": GENE_TO_MODULE.get(gene, "Other"),
                        "n_in": n_in,
                        "n_out": n_out,
                        "mean_log1p_in": float(mean_in[i]),
                        "mean_log1p_out": float(mean_out[i]),
                        "avg_log1p_diff": float(delta[i]),
                        "pct_in": float(pct_in[i]),
                        "pct_out": float(pct_out[i]),
                        "pct_diff": float(pct_in[i] - pct_out[i]),
                        "wilcoxon_p": float(pvals[i]) if np.isfinite(pvals[i]) else np.nan,
                        "p_adj_bh": float(padj[i]) if np.isfinite(padj[i]) else np.nan,
                        "marker_score": float(score[i]),
                    }
                )
        table = pd.DataFrame(records)
        if table.empty:
            continue
        table = table.sort_values(
            ["group", "p_adj_bh", "marker_score", "avg_log1p_diff"],
            ascending=[True, True, False, False],
            na_position="last",
        )
        table["rank_in_group"] = table.groupby("group").cumcount() + 1
        marker_tables[group_col] = table
        write_csv(table, source_dir / f"findmarkers_like_{safe_filename(group_col)}.csv")

        selected_genes = []
        for _, sub in table.groupby("group", sort=False):
            top = sub[(sub["avg_log1p_diff"] > 0) & (sub["pct_in"] >= 0.05)].head(2)
            selected_genes.extend(top["gene"].tolist())
        selected_genes = list(dict.fromkeys(selected_genes))[:30]
        if len(selected_genes) < 12:
            selected_genes.extend([g for g in CANONICAL_MACRO_GENES if g in genes and g not in selected_genes])
            selected_genes = selected_genes[:24]
        dot = make_dot_source_from_expr(expr, genes, labels, group_col, groups, selected_genes)
        dot_tables[group_col] = dot
        write_csv(dot, source_dir / f"findmarkers_like_dotplot_{safe_filename(group_col)}.csv")

        top = table[table["rank_in_group"] <= 3].copy()
        summary_records.append(
            {
                "group_col": group_col,
                "n_groups": len(groups),
                "n_tests": len(table),
                "median_top3_log1p_diff": float(top["avg_log1p_diff"].median()),
                "median_top3_pct_diff": float(top["pct_diff"].median()),
                "n_top3_adj_p_lt_0p05": int((top["p_adj_bh"] < 0.05).sum()),
            }
        )
    summary = pd.DataFrame(summary_records)
    write_csv(summary, source_dir / "findmarkers_like_summary.csv")
    return marker_tables, dot_tables, summary


def make_dot_source_from_expr(
    expr: np.ndarray,
    genes: np.ndarray,
    labels: np.ndarray,
    group_col: str,
    groups: list[str],
    selected_genes: list[str],
) -> pd.DataFrame:
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    selected = [gene for gene in selected_genes if gene in gene_to_idx]
    records: list[dict[str, object]] = []
    for group in groups:
        mask = labels == group
        if mask.sum() == 0:
            continue
        for gene in selected:
            values = expr[mask, gene_to_idx[gene]]
            records.append(
                {
                    "group_col": group_col,
                    "group": group,
                    "gene": gene,
                    "module": GENE_TO_MODULE.get(gene, "Other"),
                    "n_cells": int(mask.sum()),
                    "mean_log1p": float(np.mean(values)),
                    "fraction_positive": float(np.mean(values > 0)),
                }
            )
    dot = pd.DataFrame(records)
    if not dot.empty:
        dot["scaled_mean_log1p"] = dot.groupby("gene")["mean_log1p"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        )
    return dot


def compute_subtype_nmi_from_h5ad(h5ad_path: Path, source_dir: Path) -> pd.DataFrame:
    try:
        import scanpy as sc
        from sklearn.metrics import normalized_mutual_info_score
    except Exception:
        return pd.DataFrame()
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    cluster_keys = [
        "leiden_raw_clean_v10",
        "leiden_resid_sample_v10",
        "leiden_resid_sample_condition_v10",
        "leiden_resid_sample_condition_marker_v10",
        "v11_leiden_recluster",
        "v11_clean_cluster",
    ]
    fields = ["sample_id", "cohort", "condition_inferred", "v11_marker_class", "bio_fine_label"]
    records: list[dict[str, object]] = []
    for cluster_key in cluster_keys:
        if cluster_key not in obs.columns:
            continue
        cluster = obs[cluster_key].astype("string").fillna("Unknown").astype(str)
        if cluster.nunique() < 2:
            continue
        for field in fields:
            if field not in obs.columns:
                continue
            values = obs[field].astype("string").fillna("Unknown").astype(str)
            if values.nunique() < 2:
                continue
            records.append(
                {
                    "cluster_key": cluster_key,
                    "field": field,
                    "nmi": float(normalized_mutual_info_score(values, cluster)),
                }
            )
    out = pd.DataFrame(records)
    write_csv(out, source_dir / "single_cell_subtype_nmi_qc.csv")
    return out


def plot_umap_categories(
    df: pd.DataFrame,
    color_col: str,
    title: str,
    out_prefix: Path,
    palette: dict[str, str],
    order: list[str],
    max_cells: int,
    dpi: int,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.1, 3.6))
    plot_df = downsample(df.dropna(subset=["UMAP1", "UMAP2"]), max_cells)
    cats = ordered_values(plot_df[color_col], order)
    for cat in cats:
        sub = plot_df[plot_df[color_col].astype(str) == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=0.25,
            alpha=0.68,
            c=palette.get(cat, "#8C8C8C"),
            linewidths=0,
            rasterized=True,
            label=cat,
        )
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        markerscale=8,
        fontsize=5.6,
        handletextpad=0.2,
        borderaxespad=0,
    )
    return save_pub_py(fig, out_prefix, dpi=dpi)


def plot_umap_scores(
    df: pd.DataFrame,
    score_cols: list[str],
    labels: list[str],
    title: str,
    out_prefix: Path,
    max_cells: int,
    dpi: int,
) -> Path:
    score_cols = [c for c in score_cols if c in df.columns]
    if not score_cols:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        placeholder(ax, title, "No score columns were available.")
        return save_pub_py(fig, out_prefix, dpi=dpi)
    ncols = min(4, len(score_cols))
    nrows = math.ceil(len(score_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.55 * nrows), squeeze=False)
    plot_df = downsample(df.dropna(subset=["UMAP1", "UMAP2"]), max_cells)
    finite = plot_df[score_cols].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    vmin, vmax = (-1, 1) if finite.size == 0 else (np.nanpercentile(finite, 2), np.nanpercentile(finite, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    scatter = None
    for ax, col, label in zip(axes.ravel(), score_cols, labels):
        scatter = ax.scatter(
            plot_df["UMAP1"],
            plot_df["UMAP2"],
            c=plot_df[col],
            s=0.22,
            cmap="magma",
            norm=norm,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(label, fontsize=7.3, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    for ax in axes.ravel()[len(score_cols) :]:
        ax.axis("off")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
        cbar.set_label("marker score", fontsize=6.5)
    fig.suptitle(title, x=0.02, y=1.01, ha="left", fontsize=8.5, fontweight="bold")
    return save_pub_py(fig, out_prefix, dpi=dpi)


def plot_dotplot(
    dot: pd.DataFrame,
    title: str,
    out_prefix: Path,
    dpi: int,
    genes: list[str] | None = None,
    group_order: list[str] | None = None,
    max_genes: int = 28,
    max_groups: int = 16,
) -> Path:
    dot = normalize_dotplot(dot, group_order=group_order)
    if genes is not None:
        selected_genes = [g for g in genes if g in set(dot["gene"])]
    else:
        selected_genes = (
            dot.groupby("gene")["scaled_mean"]
            .max()
            .sort_values(ascending=False)
            .head(max_genes)
            .index.tolist()
        )
    selected_genes = selected_genes[:max_genes]
    groups = ordered_values(dot["group"], group_order or [])[:max_groups]
    plot_df = dot[dot["gene"].isin(selected_genes) & dot["group"].isin(groups)].copy()

    fig, ax = plt.subplots(figsize=(max(5.1, len(groups) * 0.42), max(3.2, len(selected_genes) * 0.18)))
    if plot_df.empty:
        placeholder(ax, title, "No dotplot rows were available.")
        return save_pub_py(fig, out_prefix, dpi=dpi)

    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=selected_genes[::-1], ordered=True)
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=groups, ordered=True)
    x = plot_df["group"].cat.codes
    y = plot_df["gene"].cat.codes
    sizes = 10 + 110 * plot_df["pct_expressing"].astype(float).clip(0, 1)
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=plot_df["scaled_mean"].astype(float),
        cmap="coolwarm",
        vmin=-2.2,
        vmax=2.2,
        edgecolors="#222222",
        linewidths=0.16,
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=5.8)
    ax.set_yticks(range(len(selected_genes)))
    ax.set_yticklabels(selected_genes[::-1], fontsize=6.0)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("scaled mean", fontsize=6.5)
    handles = [
        ax.scatter([], [], s=10 + 110 * v, facecolor="white", edgecolor="#222222", linewidth=0.2)
        for v in [0.25, 0.5, 0.75]
    ]
    ax.legend(handles, ["25%", "50%", "75%"], title="pct.", loc="upper left", bbox_to_anchor=(1.13, 1), fontsize=6)
    return save_pub_py(fig, out_prefix, dpi=dpi)


def plot_singlecell_nmi(nmi: pd.DataFrame, title: str, out_prefix: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    if nmi.empty or not {"representation", "field", "nmi"}.issubset(nmi.columns):
        placeholder(ax, title, "NMI QC table was not available.")
        return save_pub_py(fig, out_prefix, dpi=dpi)
    rep_order = [
        "raw_clean",
        "resid_sample",
        "resid_sample_condition",
        "resid_sample_condition_marker",
    ]
    field_order = ["sample_id", "source_short", "condition_inferred", "bio_macro_label", "vascular_class"]
    table = nmi.pivot_table(index="field", columns="representation", values="nmi", aggfunc="mean")
    rows = [r for r in field_order if r in table.index] + [r for r in table.index if r not in field_order]
    cols = [c for c in rep_order if c in table.columns] + [c for c in table.columns if c not in rep_order]
    table = table.loc[rows, cols]
    values = table.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=max(0.8, np.nanmax(values) if np.isfinite(values).any() else 0.8), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=5.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.replace("_", " ") for r in rows], fontsize=6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5.3, color="white" if values[i, j] > 0.35 else "#222222")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("NMI", fontsize=6.5)
    return save_pub_py(fig, out_prefix, dpi=dpi)


def plot_subtype_evaluation(
    subtype_nmi: pd.DataFrame,
    umap: pd.DataFrame,
    marker_summary: pd.DataFrame,
    title: str,
    out_prefix: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.1), gridspec_kw={"width_ratios": [1.2, 1.0]})

    if subtype_nmi.empty:
        placeholder(axes[0], "cluster-label NMI", "Subtype NMI could not be computed.")
    else:
        keep_fields = ["sample_id", "cohort", "condition_inferred", "v11_marker_class", "bio_fine_label"]
        table = subtype_nmi.pivot_table(index="field", columns="cluster_key", values="nmi", aggfunc="mean")
        rows = [r for r in keep_fields if r in table.index]
        cols = [c for c in [
            "leiden_raw_clean_v10",
            "leiden_resid_sample_v10",
            "leiden_resid_sample_condition_v10",
            "leiden_resid_sample_condition_marker_v10",
            "v11_leiden_recluster",
            "v11_clean_cluster",
        ] if c in table.columns]
        table = table.loc[rows, cols]
        values = table.to_numpy(dtype=float)
        im = axes[0].imshow(values, cmap="viridis", vmin=0, vmax=max(0.85, np.nanmax(values) if np.isfinite(values).any() else 0.85), aspect="auto")
        axes[0].set_xticks(range(len(cols)))
        axes[0].set_xticklabels([c.replace("leiden_", "").replace("_v10", "").replace("_", "\n") for c in cols], fontsize=5.0)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels([r.replace("_", " ") for r in rows], fontsize=5.8)
        axes[0].set_title("subtype / batch NMI", loc="left", fontsize=7.5, fontweight="bold")
        fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.01)

    score_cols = ["score_Endothelial", "score_Pericyte", "score_SMC", "score_VLMC_Fibroblast"]
    if all(c in umap.columns for c in score_cols):
        scores = umap[score_cols].to_numpy(dtype=float)
        top2 = np.sort(scores, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        tmp = umap[["subtype_label"]].copy()
        tmp["score_margin"] = margin
        groups = [g for g in FINE_ORDER if g in set(tmp["subtype_label"])]
        data = [tmp.loc[tmp["subtype_label"] == g, "score_margin"].dropna().values for g in groups]
        axes[1].boxplot(
            data,
            patch_artist=True,
            widths=0.65,
            medianprops={"color": "#222222", "linewidth": 0.8},
            boxprops={"facecolor": "#D8E6F3", "edgecolor": "#555555", "linewidth": 0.6},
            whiskerprops={"color": "#555555", "linewidth": 0.6},
            capprops={"color": "#555555", "linewidth": 0.6},
            flierprops={"markersize": 1.2, "markerfacecolor": "#999999", "markeredgewidth": 0, "alpha": 0.3},
        )
        axes[1].set_xticks(range(1, len(groups) + 1))
        axes[1].set_xticklabels(groups, rotation=45, ha="right", fontsize=5.5)
        axes[1].set_ylabel("top-two marker score margin")
        axes[1].set_title("module separation", loc="left", fontsize=7.5, fontweight="bold")
    elif not marker_summary.empty:
        axes[1].barh(marker_summary["group_col"], marker_summary["median_top3_log1p_diff"], color="#4E79A7")
        axes[1].set_xlabel("median top3 log1p diff")
        axes[1].set_title("marker confirmation", loc="left", fontsize=7.5, fontweight="bold")
    else:
        placeholder(axes[1], "module separation", "Marker-score margin was not available.")

    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=8.5, fontweight="bold")
    return save_pub_py(fig, out_prefix, dpi=dpi)


def select_spatial_samples(spatial: pd.DataFrame, max_per_cohort: int = 2, max_total: int = 4) -> list[str]:
    counts = spatial.groupby(["cohort", "sample_id"], dropna=False).size().reset_index(name="n")
    preferred_cohorts = ["AD_Hip_Saptial", "Cortex_Spatial"]
    samples: list[str] = []
    for cohort in preferred_cohorts:
        sub = counts[counts["cohort"].astype(str) == cohort].sort_values("n", ascending=False).head(max_per_cohort)
        samples.extend(sub["sample_id"].astype(str).tolist())
    if len(samples) < max_total:
        extra = counts[~counts["sample_id"].astype(str).isin(samples)].sort_values("n", ascending=False).head(max_total - len(samples))
        samples.extend(extra["sample_id"].astype(str).tolist())
    return samples[:max_total]


def plot_spatial_facets(
    spatial: pd.DataFrame,
    label_col: str,
    title: str,
    out_prefix: Path,
    palette: dict[str, str],
    order: list[str],
    max_spots_per_sample: int,
    dpi: int,
) -> Path:
    required = {"sample_id", "coord_x", "coord_y", label_col}
    missing = required - set(spatial.columns)
    if missing:
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        placeholder(ax, title, f"Missing spatial columns: {sorted(missing)}")
        return save_pub_py(fig, out_prefix, dpi=dpi)

    samples = select_spatial_samples(spatial)
    ncols = 2 if len(samples) <= 4 else min(3, max(1, len(samples)))
    nrows = math.ceil(max(1, len(samples)) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False)
    cats = ordered_values(spatial[label_col], order)
    for ax, sample in zip(axes.ravel(), samples):
        sub = spatial[spatial["sample_id"].astype(str) == sample].copy()
        sub = downsample(sub.dropna(subset=["coord_x", "coord_y"]), max_spots_per_sample, seed=abs(hash(sample)) % 100000)
        for cat in cats:
            one = sub[sub[label_col].astype(str) == cat]
            if one.empty:
                continue
            ax.scatter(
                one["coord_x"],
                one["coord_y"],
                s=1.1,
                c=palette.get(cat, "#8C8C8C"),
                alpha=0.86,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(sample.split("/")[-1], fontsize=7, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.ravel()[len(samples) :]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get(cat, "#8C8C8C"), markeredgewidth=0, markersize=4, label=cat)
        for cat in cats[:18]
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5), fontsize=5.8, borderaxespad=0)
    fig.suptitle(title, x=0.02, y=1.01, ha="left", fontsize=8.5, fontweight="bold")
    return save_pub_py(fig, out_prefix, dpi=dpi)


def plot_marker_correlations(cor: pd.DataFrame, title: str, out_prefix: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    if cor.empty or "marker_score_vs_deconv_prop_r" not in cor.columns:
        placeholder(ax, title, "Correlation table was not available.")
        return save_pub_py(fig, out_prefix, dpi=dpi)
    plot_df = cor.rename(columns={"marker_score_vs_deconv_prop_r": "r"}).copy()
    plot_df = plot_df[np.isfinite(plot_df["r"])].sort_values("r", ascending=True)
    ax.barh(
        plot_df["class"],
        plot_df["r"],
        color=[MACRO_COLORS.get(str(c), "#8C8C8C") for c in plot_df["class"]],
        edgecolor="white",
        linewidth=0.4,
    )
    for y, r in enumerate(plot_df["r"]):
        ax.text(r + 0.012, y, f"r={r:.2f}", va="center", fontsize=6)
    ax.axvline(0.5, color="#666666", linewidth=0.7, linestyle="--")
    ax.set_xlim(0, max(0.82, float(plot_df["r"].max()) + 0.12))
    ax.set_xlabel("marker-derived score vs deconvolution proportion")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    return save_pub_py(fig, out_prefix, dpi=dpi)


def plot_marker_support_heatmap(support: pd.DataFrame, title: str, out_prefix: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(5.1, 3.9))
    if support.empty or not {"label", "module", "module_mean"}.issubset(support.columns):
        placeholder(ax, title, "Subtype marker-support table was not available.")
        return save_pub_py(fig, out_prefix, dpi=dpi)
    module_rows = support[pd.isna(support.get("gene"))].copy()
    if module_rows.empty:
        module_rows = support.groupby(["label", "module"], as_index=False)["mean_log1p"].mean().rename(columns={"mean_log1p": "module_mean"})
    mat = module_rows.pivot_table(index="label", columns="module", values="module_mean", aggfunc="mean")
    rows = [r for r in CLUSTER_ORDER if r in mat.index] + [r for r in mat.index if r not in CLUSTER_ORDER]
    modules = [m for m in ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"] if m in mat.columns] + [m for m in mat.columns if m not in {"Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"}]
    mat = mat.loc[rows, modules].fillna(0)
    values = mat.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels([m.replace("_", "\n") for m in modules], rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=4.9, color="white" if values[i, j] > np.nanmax(values) * 0.45 else "#222222")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("module mean", fontsize=6.5)
    return save_pub_py(fig, out_prefix, dpi=dpi)


def trim_white(img: np.ndarray, pad: int = 14) -> np.ndarray:
    rgb = img[..., :3]
    if rgb.dtype.kind != "f":
        rgb = rgb.astype(float) / 255.0
    mask = np.any(rgb < 0.985, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[max(y0 - pad, 0) : min(y1 + pad, img.shape[0]), max(x0 - pad, 0) : min(x1 + pad, img.shape[1])]


def compose_overview(panel_pngs: list[tuple[str, str, Path]], out_prefix: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(14.2, 9.7))
    for ax, (letter, title, path) in zip(axes.ravel(), panel_pngs):
        ax.axis("off")
        if path.exists():
            ax.imshow(trim_white(mpimg.imread(path)))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        draw_panel_label(ax, letter)
        ax.set_title(title, fontsize=6.5, loc="left", pad=2)
    for ax in axes.ravel()[len(panel_pngs) :]:
        ax.axis("off")
    fig.suptitle(
        "Figure 3: vascular-cell identification and spatial deconvolution validation",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return save_pub_py(fig, out_prefix, dpi=dpi)


def write_report(
    out_dir: Path,
    input_dir: Path,
    used_files: dict[str, str],
    panel_records: list[dict[str, str]],
    marker_summary: pd.DataFrame,
    cor: pd.DataFrame,
) -> None:
    report = out_dir / "Figure3_python_analysis_summary.md"
    lines = [
        "# Figure 3 Python analysis summary",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: vascular identities and vascular subtypes are supported by single-cell marker structure and remain consistent with spatial deconvolution evidence.",
        "- Evidence chain: UMAP annotation, marker-score projection, marker dotplots, single-cell QC, spatial deconvolution maps, and marker/deconvolution consistency.",
        "- Archetype: asymmetric mixed-modality figure.",
        "- Backend: Python/matplotlib only.",
        f"- Input directory: `{input_dir}`.",
        f"- Output directory: `{out_dir}`.",
        "",
        "## Panel outputs",
        "",
        "| Panel | Content | File | Status |",
        "| --- | --- | --- | --- |",
    ]
    for rec in panel_records:
        lines.append(f"| {rec['panel']} | {rec['content']} | `{Path(rec['file']).name}` | {rec['status']} |")
    lines.extend(["", "## Marker confirmation", ""])
    if marker_summary.empty:
        lines.append("- Marker-panel FindAllMarkers-like confirmation was not run or did not return results.")
    else:
        for row in marker_summary.to_dict("records"):
            lines.append(
                f"- `{row['group_col']}`: {row['n_groups']} groups, {row['n_tests']} tests, "
                f"median top-3 log1p diff {row['median_top3_log1p_diff']:.3f}, "
                f"top-3 adjusted P<0.05 count {row['n_top3_adj_p_lt_0p05']}."
            )
        lines.append("- Confirmation is marker-panel based, using `expanded_marker_log1p` and `expanded_marker_genes` from the final h5ad.")
    if not cor.empty and "marker_score_vs_deconv_prop_r" in cor.columns:
        cor_text = ", ".join(
            f"{row['class']} r={row['marker_score_vs_deconv_prop_r']:.2f}" for row in cor.to_dict("records")
        )
        lines.extend(["", "## Spatial marker-deconvolution consistency", "", f"- Broad-class correlations: {cor_text}."])
    lines.extend(["", "## Resolved source files", "", "| Input | File |", "| --- | --- |"])
    for key, value in sorted(used_files.items()):
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Review notes",
            "",
            "- Panel 11 uses the final v12b marker-score versus deconvolution-proportion Pearson correlation table.",
            "- Panel 12 uses final v12b cluster-level spatial marker support as subtype-level consistency evidence.",
            "- The single-cell marker confirmation is limited to the expanded vascular marker panel available in the final h5ad, not genome-wide differential expression.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = resolve_input_dir(root, args.input_dir, args.strict_final_results)
    out_dir = (args.out_dir or input_dir / "figure3_vascular_identification_python").resolve()
    panels_dir = out_dir / "panels"
    source_dir = out_dir / "source_data"
    panels_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    used_files: dict[str, str] = {}

    def load_csv(key: str, rel: str, **kwargs) -> pd.DataFrame:
        path = rel_path(input_dir, rel, required=True)
        used_files[key] = str(path)
        return pd.read_csv(path, **kwargs)

    umap = prepare_umap(
        load_csv("single_cell_v12_umap_scores", "01_single_cell/v12_score_dotplot_subtype/source_data/figure3_sc_v12_umap_scores_and_annotations.csv")
    )
    macro_dot = load_csv("single_cell_v12_macro_dotplot", "01_single_cell/v12_score_dotplot_subtype/source_data/figure3_sc_v12_selected_marker_dotplot_by_macro.csv")
    subtype_dot_final = load_csv("single_cell_v12_subtype_dotplot", "01_single_cell/v12_score_dotplot_subtype/source_data/figure3_sc_v12_selected_marker_dotplot_by_subtype.csv")
    nmi = load_csv("single_cell_v10_metadata_nmi", "01_single_cell/v10_batch_residual_umap/results/single_cell_batch_residual_v10_metadata_nmi.csv")
    spatial = load_csv("spatial_v12b_all_chip_annotations", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_all_chip_annotations.csv.gz")
    cor = load_csv("spatial_v12b_marker_deconv_correlations", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_marker_deconv_correlations.csv")
    support_cluster = load_csv("spatial_v12b_marker_support_by_cluster", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_marker_support_by_cluster.csv")
    h5ad_path = rel_path(input_dir, "01_single_cell/v11_clean_annotation/results/single_cell_clean_vascular_v11.h5ad", required=True)
    used_files["single_cell_v11_h5ad"] = str(h5ad_path)

    write_csv(umap, source_dir / "figure3_single_cell_umap_scores_annotations_normalized.csv")
    write_csv(spatial, source_dir / "figure3_spatial_v12b_all_chip_annotations_source.csv")
    write_csv(cor, source_dir / "figure3_spatial_marker_deconv_correlations_source.csv")

    marker_tables: dict[str, pd.DataFrame] = {}
    marker_dot_tables: dict[str, pd.DataFrame] = {}
    marker_summary = pd.DataFrame()
    if not args.skip_findmarkers:
        try:
            marker_tables, marker_dot_tables, marker_summary = marker_panel_findmarkers(
                h5ad_path,
                source_dir,
                ["v11_marker_class", "bio_fine_label", "v11_clean_cluster"],
            )
        except Exception as exc:
            warnings.warn(f"Marker confirmation failed; using final v12 dotplot source. Reason: {exc}")

    subtype_nmi = compute_subtype_nmi_from_h5ad(h5ad_path, source_dir)
    subtype_dot = marker_dot_tables.get("bio_fine_label", subtype_dot_final)

    panel_records: list[dict[str, str]] = []

    def record(panel: str, content: str, path: Path, status: str = "generated") -> None:
        panel_records.append({"panel": panel, "content": content, "file": str(path), "status": status})

    p1 = plot_umap_categories(
        umap,
        "broad_class",
        "1. Vascular cells in single-cell UMAP",
        panels_dir / "fig3_01_vascular_umap_macro",
        MACRO_COLORS,
        MACRO_ORDER,
        args.max_umap_cells,
        args.dpi,
    )
    record("1", "broad vascular-cell UMAP", p1)

    p2 = plot_umap_scores(
        umap,
        ["score_Endothelial", "score_Pericyte", "score_SMC", "score_VLMC_Fibroblast"],
        ["Endothelial", "Pericyte", "SMC", "VLMC/Fibroblast"],
        "2. Broad vascular gene-score UMAP",
        panels_dir / "fig3_02_gene_score_umap_macro",
        args.max_umap_cells,
        args.dpi,
    )
    record("2", "macro marker-score UMAP", p2)

    p3 = plot_dotplot(
        macro_dot,
        "3. Broad vascular marker-gene dot plot",
        panels_dir / "fig3_03_marker_gene_dotplot_macro",
        args.dpi,
        genes=CANONICAL_MACRO_GENES,
        group_order=MACRO_ORDER,
        max_genes=18,
        max_groups=6,
    )
    record("3", "broad marker-gene dotplot", p3)

    p4 = plot_singlecell_nmi(
        nmi,
        "4. Single-cell representation QC",
        panels_dir / "fig3_04_singlecell_model_evaluation",
        args.dpi,
    )
    record("4", "Figure 2-style single-cell NMI evaluation", p4)

    p5 = plot_umap_categories(
        umap,
        "subtype_label",
        "5. Vascular subtype UMAP",
        panels_dir / "fig3_05_vascular_subtype_umap",
        FINE_COLORS,
        FINE_ORDER,
        args.max_umap_cells,
        args.dpi,
    )
    record("5", "vascular subtype UMAP", p5)

    p6 = plot_umap_scores(
        umap,
        ["score_Endothelial", "score_Pericyte", "score_SMC", "score_VLMC_Fibroblast"],
        ["EC module", "Pericyte module", "SMC module", "VLMC/Fibro module"],
        "6. Subtype gene-score UMAP",
        panels_dir / "fig3_06_subtype_gene_score_umap",
        args.max_umap_cells,
        args.dpi,
    )
    record("6", "subtype marker-score UMAP", p6)

    p7 = plot_dotplot(
        subtype_dot,
        "7. Subtype marker-gene dot plot",
        panels_dir / "fig3_07_subtype_marker_gene_dotplot",
        args.dpi,
        genes=None,
        group_order=FINE_ORDER,
        max_genes=28,
        max_groups=12,
    )
    record(
        "7",
        "subtype marker dotplot with marker-panel FindAllMarkers-like confirmation",
        p7,
        "generated from confirmation table" if "bio_fine_label" in marker_dot_tables else "generated from final v12 source",
    )

    p8 = plot_subtype_evaluation(
        subtype_nmi,
        umap,
        marker_summary,
        "8. Single-cell subtype QC",
        panels_dir / "fig3_08_singlecell_subtype_evaluation",
        args.dpi,
    )
    record("8", "subtype NMI and marker-score margin evaluation", p8)

    p9 = plot_spatial_facets(
        spatial,
        "deconv_dominant_class",
        "9. Spatial vascular deconvolution map",
        panels_dir / "fig3_09_spatial_vascular_deconv_annotation_map",
        MACRO_COLORS,
        MACRO_ORDER,
        args.max_spots_per_sample,
        args.dpi,
    )
    record("9", "spatial broad-class deconvolution map", p9)

    p10 = plot_spatial_facets(
        spatial,
        "deconv_dominant_cluster",
        "10. Spatial vascular subtype deconvolution map",
        panels_dir / "fig3_10_spatial_subtype_deconv_annotation_map",
        CLUSTER_COLORS,
        CLUSTER_ORDER,
        args.max_spots_per_sample,
        args.dpi,
    )
    record("10", "spatial subtype/cluster deconvolution map", p10)

    p11 = plot_marker_correlations(
        cor,
        "11. Spatial deconvolution-marker consistency",
        panels_dir / "fig3_11_spatial_deconv_marker_consistency_macro",
        args.dpi,
    )
    record("11", "broad deconvolution vs marker-score correlation", p11)

    p12 = plot_marker_support_heatmap(
        support_cluster,
        "12. Subtype marker consistency in space",
        panels_dir / "fig3_12_spatial_subtype_marker_consistency",
        args.dpi,
    )
    record("12", "subtype deconvolution cluster marker-support heatmap", p12)

    panel_pngs = [
        ("a", "vascular UMAP", p1),
        ("b", "macro score UMAP", p2),
        ("c", "macro dotplot", p3),
        ("d", "single-cell QC", p4),
        ("e", "subtype UMAP", p5),
        ("f", "subtype score UMAP", p6),
        ("g", "subtype dotplot", p7),
        ("h", "subtype QC", p8),
        ("i", "spatial macro map", p9),
        ("j", "spatial subtype map", p10),
        ("k", "marker/deconv r", p11),
        ("l", "subtype marker support", p12),
    ]
    overview = compose_overview(panel_pngs, out_dir / "figure3_vascular_identification_12panel_overview", args.dpi)
    record("overview", "12-panel overview", overview)

    write_csv(pd.DataFrame(panel_records), source_dir / "figure3_panel_manifest.csv")
    (source_dir / "resolved_source_files.json").write_text(json.dumps(used_files, indent=2), encoding="utf-8")
    write_report(out_dir, input_dir, used_files, panel_records, marker_summary, cor)
    print(f"Wrote Figure 3 Python outputs to: {out_dir}")
    print(f"Overview: {overview}")


if __name__ == "__main__":
    main()
