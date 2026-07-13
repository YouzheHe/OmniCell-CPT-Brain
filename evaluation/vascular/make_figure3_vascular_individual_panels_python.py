#!/usr/bin/env python
"""Create individual Nature-style Figure 3 vascular panels.

This v2 script uses only the final Figure 3 result package. It removes QC-only
panels, recomputes marker-panel FindAllMarkers-like genes for subtype dotplots
and score UMAPs, and exports each panel independently at 1000 dpi.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
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
    "Endothelial": "#2C5A8A",
    "Pericyte": "#B55AA0",
    "SMC": "#D73027",
    "Fibroblast_VLMC": "#238B45",
    "Low_confidence": "#BDBDBD",
    "Unknown": "#8A8A8A",
}
FINE_COLORS = {
    # Endothelial family, blue-teal gradient.
    "BBB_EC": "#08306B",
    "Capillary_EC": "#2171B5",
    "Arterial_EC": "#6BAED6",
    "Venous_EC": "#1B9E77",
    "Activated_EC": "#A50F15",
    "Endothelial_core": "#4292C6",
    # Mural/pericyte family, purple-magenta gradient.
    "Pericyte": "#7A0177",
    "Contractile_mural": "#C51B8A",
    # SMC and VLMC/fibroblast.
    "SMC": "#D73027",
    "Fibroblast_VLMC": "#238B45",
}
CLUSTER_COLORS = {
    "EC00": "#08306B",
    "EC01": "#2171B5",
    "EC02": "#6BAED6",
    "EC03": "#1B9E77",
    "EC04": "#A50F15",
    "PC00": "#7A0177",
    "PC01": "#9E4FA8",
    "PC02": "#C51B8A",
    "SMC00": "#D73027",
    "VLMC00": "#006D2C",
    "VLMC01": "#238B45",
    "VLMC02": "#41AB5D",
    "VLMC03": "#74C476",
    "Low_confidence": "#BDBDBD",
}

GENE_MODULES = {
    "Endothelial": ["PECAM1", "CLDN5", "VWF", "RAMP2", "FLT1", "KDR", "CDH5", "ESAM", "SLC2A1", "ERG", "TIE1"],
    "BBB/Capillary EC": ["ABCB1", "ABCG2", "MFSD2A", "RGCC", "EMCN", "PLVAP", "CA4", "LSR", "TJP1", "OCLN"],
    "Arterial EC": ["GJA5", "HEY1", "EFNB2", "SOX17", "CXCR4", "DLL4", "BMX"],
    "Venous/activated EC": ["ACKR1", "NR2F2", "ICAM1", "VCAM1", "SELE", "ANGPT2", "BTNL9"],
    "Pericyte": ["PDGFRB", "RGS5", "ABCC9", "NOTCH3", "CSPG4", "KCNJ8", "MCAM", "HIGD1B", "NDUFA4L2"],
    "SMC": ["ACTA2", "TAGLN", "MYH11", "CNN1", "MYLK", "MYOCD", "SMTN", "LMOD1", "MYL9"],
    "VLMC/Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "COL6A2", "APOD", "FBLN1", "MGP", "PI16"],
}
GENE_TO_MODULE = {gene: module for module, genes in GENE_MODULES.items() for gene in genes}
EXPECTED_MODULES = {
    "Endothelial": ["Endothelial", "BBB/Capillary EC", "Arterial EC", "Venous/activated EC"],
    "Pericyte": ["Pericyte"],
    "SMC": ["SMC"],
    "Fibroblast_VLMC": ["VLMC/Fibroblast"],
    "BBB_EC": ["BBB/Capillary EC", "Endothelial"],
    "Capillary_EC": ["BBB/Capillary EC", "Endothelial"],
    "Arterial_EC": ["Arterial EC", "Endothelial"],
    "Venous_EC": ["Venous/activated EC", "Endothelial"],
    "Activated_EC": ["Venous/activated EC", "Endothelial"],
    "Endothelial_core": ["Endothelial"],
    "Contractile_mural": ["SMC", "Pericyte"],
}
FALLBACK_GENES = {
    "Endothelial": ["CLDN5", "PECAM1", "VWF", "RAMP2"],
    "Pericyte": ["PDGFRB", "RGS5", "ABCC9", "NOTCH3"],
    "SMC": ["ACTA2", "TAGLN", "MYH11", "CNN1"],
    "Fibroblast_VLMC": ["COL1A1", "COL3A1", "DCN", "LUM"],
    "BBB_EC": ["ABCB1", "MFSD2A", "CLDN5"],
    "Capillary_EC": ["CA4", "EMCN", "RGCC"],
    "Arterial_EC": ["GJA5", "EFNB2", "HEY1"],
    "Venous_EC": ["ACKR1", "NR2F2", "VWF"],
    "Activated_EC": ["ICAM1", "VCAM1", "SELE"],
    "Endothelial_core": ["PECAM1", "CLDN5", "RAMP2"],
    "Contractile_mural": ["ACTA2", "MYL9", "TAGLN"],
}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=1000)
    parser.add_argument("--max-umap-cells", type=int, default=120_000)
    parser.add_argument("--hip-sample", default="AD_Hip_Saptial/AD2.1")
    parser.add_argument("--cortex-sample", default="Cortex_Spatial/T991")
    parser.add_argument("--strict-final-results", action="store_true")
    return parser.parse_args()


def resolve_input_dir(root: Path, input_dir: Path | None, strict: bool) -> Path:
    candidates = []
    if input_dir is not None:
        candidates += [input_dir, input_dir / FINAL_FIGURE3_DIRNAME]
    candidates += [
        root / "projects" / "nvu_vascular" / "final_results" / FINAL_FIGURE3_DIRNAME,
        root / "projects" / "nvu_vascular" / "final_results",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.name == "final_results" and (candidate / FINAL_FIGURE3_DIRNAME).exists():
            candidate = candidate / FINAL_FIGURE3_DIRNAME
        if (candidate / "01_single_cell").exists() and (candidate / "03_spatial_deconvolution_QC").exists():
            return candidate
    if strict:
        raise FileNotFoundError("Final Figure 3 directory not found.")
    raise FileNotFoundError("Final Figure 3 directory not found; pass --input-dir.")


def req(base: Path, rel: str) -> Path:
    path = base / rel
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return path


def save_panel(fig: plt.Figure, prefix: Path, dpi: int, pdf_dir: Path) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "dpi": dpi}
        fig.savefig(prefix.with_suffix(f".{ext}"), **kwargs)
    shutil.copy2(prefix.with_suffix(".pdf"), pdf_dir / prefix.with_suffix(".pdf").name)
    plt.close(fig)
    return prefix.with_suffix(".png")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def ordered_values(values: pd.Series | np.ndarray, order: list[str]) -> list[str]:
    present = pd.Series(values).dropna().astype(str).unique().tolist()
    out = [x for x in order if x in present]
    out += sorted([x for x in present if x not in out])
    return out


def downsample(df: pd.DataFrame, max_n: int, seed: int = 17) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed)


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    mask = np.isfinite(pvalues)
    if mask.sum() == 0:
        return out
    p = pvalues[mask]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    inv = np.empty_like(adj)
    inv[order] = np.clip(adj, 0, 1)
    out[mask] = inv
    return out


def read_marker_expression(h5ad_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    expr = np.asarray(adata.obsm["expanded_marker_log1p"], dtype=np.float32)
    genes = np.asarray(adata.uns["expanded_marker_genes"], dtype=str)
    obs = adata.obs.copy()
    return obs, expr, genes


def findmarkers_marker_panel(
    obs: pd.DataFrame,
    expr: np.ndarray,
    genes: np.ndarray,
    group_col: str,
    group_order: list[str],
    min_cells: int = 50,
) -> pd.DataFrame:
    from scipy.stats import ranksums

    labels = obs[group_col].astype("string").fillna("Unknown").astype(str).to_numpy()
    groups = [g for g in ordered_values(labels, group_order) if int((labels == g).sum()) >= min_cells]
    records: list[dict[str, object]] = []
    for group in groups:
        mask = labels == group
        x_in = expr[mask]
        x_out = expr[~mask]
        mean_in = x_in.mean(axis=0)
        mean_out = x_out.mean(axis=0)
        pct_in = (x_in > 0).mean(axis=0)
        pct_out = (x_out > 0).mean(axis=0)
        delta = mean_in - mean_out
        pvals = np.array([ranksums(x_in[:, i], x_out[:, i]).pvalue for i in range(len(genes))], dtype=float)
        padj = bh_adjust(pvals)
        score = delta * np.sqrt(np.maximum(pct_in, 1e-6)) * np.maximum(pct_in - pct_out + 0.05, 0.05)
        for i, gene in enumerate(genes):
            records.append(
                {
                    "group_col": group_col,
                    "group": group,
                    "gene": gene,
                    "module": GENE_TO_MODULE.get(gene, "Other"),
                    "n_in": int(mask.sum()),
                    "n_out": int((~mask).sum()),
                    "mean_log1p_in": float(mean_in[i]),
                    "mean_log1p_out": float(mean_out[i]),
                    "avg_log1p_diff": float(delta[i]),
                    "pct_in": float(pct_in[i]),
                    "pct_out": float(pct_out[i]),
                    "pct_diff": float(pct_in[i] - pct_out[i]),
                    "p_adj_bh": float(padj[i]) if np.isfinite(padj[i]) else np.nan,
                    "marker_score": float(score[i]),
                }
            )
    table = pd.DataFrame(records)
    table = table.sort_values(
        ["group", "p_adj_bh", "marker_score", "avg_log1p_diff"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    table["rank_in_group"] = table.groupby("group").cumcount() + 1
    return table


def select_group_genes(table: pd.DataFrame, groups: list[str], n_per_group: int) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for group in groups:
        sub = table[table["group"] == group].copy()
        expected = EXPECTED_MODULES.get(group, [group])
        filt = sub[
            (sub["avg_log1p_diff"] > 0)
            & (sub["pct_in"] >= 0.05)
            & (sub["pct_diff"] >= -0.02)
            & (sub["module"].isin(expected))
        ].copy()
        if len(filt) < n_per_group:
            extra = sub[
                (sub["avg_log1p_diff"] > 0)
                & (sub["pct_in"] >= 0.05)
                & (~sub["gene"].isin(filt["gene"]))
            ]
            filt = pd.concat([filt, extra], ignore_index=True)
        genes = filt.sort_values(["p_adj_bh", "marker_score", "avg_log1p_diff"], ascending=[True, False, False])[
            "gene"
        ].drop_duplicates().head(n_per_group).tolist()
        for gene in FALLBACK_GENES.get(group, []):
            if len(genes) >= n_per_group:
                break
            if gene not in genes and gene in set(table["gene"]):
                genes.append(gene)
        selected[group] = genes
    return selected


def add_marker_scores(umap: pd.DataFrame, expr: np.ndarray, genes: np.ndarray, gene_sets: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    z = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + 1e-6)
    out = umap.copy()
    for group, marker_genes in gene_sets.items():
        idx = [gene_to_idx[g] for g in marker_genes if g in gene_to_idx]
        col = f"{prefix}{safe_name(group)}"
        out[col] = z[:, idx].mean(axis=1) if idx else np.nan
    return out


def make_dot_source(
    obs: pd.DataFrame,
    expr: np.ndarray,
    genes: np.ndarray,
    group_col: str,
    group_order: list[str],
    selected_gene_sets: dict[str, list[str]],
) -> pd.DataFrame:
    labels = obs[group_col].astype("string").fillna("Unknown").astype(str).to_numpy()
    groups = [g for g in group_order if g in set(labels)]
    selected = []
    for group in groups:
        selected.extend(selected_gene_sets.get(group, []))
    selected = list(dict.fromkeys(selected))
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    records = []
    for group in groups:
        mask = labels == group
        for gene in selected:
            if gene not in gene_to_idx:
                continue
            values = expr[mask, gene_to_idx[gene]]
            records.append(
                {
                    "group_col": group_col,
                    "group": group,
                    "gene": gene,
                    "module": GENE_TO_MODULE.get(gene, "Other"),
                    "n_cells": int(mask.sum()),
                    "mean_log1p": float(values.mean()),
                    "fraction_positive": float((values > 0).mean()),
                }
            )
    dot = pd.DataFrame(records)
    if not dot.empty:
        dot["scaled_mean_log1p"] = dot.groupby("gene")["mean_log1p"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        )
    return dot


def prep_umap(umap: pd.DataFrame) -> pd.DataFrame:
    out = umap.rename(columns={"UMAP1": "umap_1", "UMAP2": "umap_2"}).copy()
    out["macro_label"] = out["v11_marker_class"].astype(str)
    out["subtype_label"] = out["bio_fine_label"].astype(str)
    out["cluster_label"] = out["v11_clean_cluster"].astype(str)
    return out


def nature_umap_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.annotate("", xy=(0.19, 0.08), xytext=(0.08, 0.08), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555"))
    ax.annotate("", xy=(0.08, 0.20), xytext=(0.08, 0.08), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555"))
    ax.text(0.20, 0.045, "UMAP_1", transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="#555555")
    ax.text(0.045, 0.20, "UMAP_2", transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=6.5, color="#555555")


def plot_umap_categories(
    umap: pd.DataFrame,
    label_col: str,
    palette: dict[str, str],
    order: list[str],
    title: str,
    out_prefix: Path,
    pdf_dir: Path,
    dpi: int,
    max_cells: int,
    point_size: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    plot_df = downsample(umap.dropna(subset=["umap_1", "umap_2"]), max_cells)
    cats = ordered_values(plot_df[label_col], order)
    for cat in cats:
        sub = plot_df[plot_df[label_col].astype(str) == cat]
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=point_size,
            alpha=0.82,
            c=palette.get(cat, "#808080"),
            linewidths=0,
            rasterized=True,
            label=cat,
        )
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=2)
    ax.set_aspect("equal", adjustable="datalim")
    nature_umap_axes(ax)
    ncol = 2 if len(cats) <= 5 else 3
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.95), fontsize=6, markerscale=5, handletextpad=0.2, ncols=1 if len(cats) <= 6 else ncol)
    return save_panel(fig, out_prefix, dpi, pdf_dir)


def plot_score_umap_grid(
    umap: pd.DataFrame,
    score_cols: list[str],
    labels: list[str],
    title: str,
    out_prefix: Path,
    pdf_dir: Path,
    dpi: int,
    max_cells: int,
    ncols: int,
) -> Path:
    score_cols = [c for c in score_cols if c in umap.columns]
    labels = [lab for c, lab in zip(score_cols, labels) if c in umap.columns]
    ncols = min(ncols, len(score_cols))
    nrows = math.ceil(len(score_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 2.1 * nrows), squeeze=False)
    plot_df = downsample(umap.dropna(subset=["umap_1", "umap_2"]), max_cells)
    values = plot_df[score_cols].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin, vmax = (-1, 1) if finite.size == 0 else (np.nanpercentile(finite, 2), np.nanpercentile(finite, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    scatter = None
    for ax, col, label in zip(axes.ravel(), score_cols, labels):
        scatter = ax.scatter(
            plot_df["umap_1"],
            plot_df["umap_2"],
            c=plot_df[col],
            s=0.7,
            cmap="magma",
            norm=norm,
            alpha=0.86,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(label, fontsize=7, fontweight="bold", pad=1)
        ax.set_aspect("equal", adjustable="datalim")
        nature_umap_axes(ax)
    for ax in axes.ravel()[len(score_cols) :]:
        ax.axis("off")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.018, pad=0.01)
        cbar.set_label("FindAllMarkers score", fontsize=6.5)
    fig.suptitle(title, x=0.01, y=1.02, ha="left", fontsize=8.5, fontweight="bold")
    return save_panel(fig, out_prefix, dpi, pdf_dir)


def plot_dotplot(
    dot: pd.DataFrame,
    selected_gene_sets: dict[str, list[str]],
    group_order: list[str],
    title: str,
    out_prefix: Path,
    pdf_dir: Path,
    dpi: int,
) -> Path:
    groups = [g for g in group_order if g in set(dot["group"])]
    gene_order = []
    for group in groups:
        gene_order.extend(selected_gene_sets.get(group, []))
    gene_order = list(dict.fromkeys(gene_order))
    plot_df = dot[dot["group"].isin(groups) & dot["gene"].isin(gene_order)].copy()
    fig, ax = plt.subplots(figsize=(max(4.8, len(groups) * 0.42), max(3.2, len(gene_order) * 0.16)))
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=groups, ordered=True)
    plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=gene_order[::-1], ordered=True)
    x = plot_df["group"].cat.codes
    y = plot_df["gene"].cat.codes
    sc = ax.scatter(
        x,
        y,
        s=12 + 130 * plot_df["fraction_positive"].clip(0, 1),
        c=plot_df["scaled_mean_log1p"],
        cmap="coolwarm",
        vmin=-2.2,
        vmax=2.2,
        edgecolors="#222222",
        linewidths=0.18,
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(gene_order)))
    ax.set_yticklabels(gene_order[::-1], fontsize=6)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("scaled mean", fontsize=6.5)
    handles = [ax.scatter([], [], s=12 + 130 * v, facecolor="white", edgecolor="#222222", linewidth=0.2) for v in [0.25, 0.5, 0.75]]
    ax.legend(handles, ["25%", "50%", "75%"], title="pct.", loc="upper left", bbox_to_anchor=(1.12, 1), fontsize=6)
    return save_panel(fig, out_prefix, dpi, pdf_dir)


def sample_mask(df: pd.DataFrame, sample: str) -> pd.Series:
    s = df["sample_id"].astype(str)
    return (s == sample) | s.str.endswith("/" + sample.split("/")[-1])


def plot_spatial_pair(
    spatial: pd.DataFrame,
    label_col: str,
    palette: dict[str, str],
    order: list[str],
    samples: list[str],
    title: str,
    out_prefix: Path,
    pdf_dir: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.45), squeeze=False)
    cats = ordered_values(spatial[label_col], order)
    for ax, sample in zip(axes.ravel(), samples):
        sub = spatial[sample_mask(spatial, sample)].copy()
        for cat in cats:
            one = sub[sub[label_col].astype(str) == cat]
            if one.empty:
                continue
            ax.scatter(one["coord_x"], one["coord_y"], s=4.5, c=palette.get(cat, "#808080"), alpha=0.9, linewidths=0, rasterized=True)
        ax.set_title(sample.split("/")[-1], fontsize=7.5, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get(cat, "#808080"), markeredgewidth=0, markersize=4, label=cat) for cat in cats if cat in set(spatial[label_col].astype(str))]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=5.8)
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=8.5, fontweight="bold")
    return save_panel(fig, out_prefix, dpi, pdf_dir)


def plot_spatial_macro_validation(cor: pd.DataFrame, agreement: pd.DataFrame, out_prefix: Path, pdf_dir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0]})
    plot_df = cor.rename(columns={"marker_score_vs_deconv_prop_r": "r"}).copy().sort_values("r")
    axes[0].barh(plot_df["class"], plot_df["r"], color=[MACRO_COLORS.get(x, "#808080") for x in plot_df["class"]])
    for y, r in enumerate(plot_df["r"]):
        axes[0].text(r + 0.01, y, f"{r:.2f}", va="center", fontsize=6)
    axes[0].axvline(0.5, color="#666666", linestyle="--", linewidth=0.7)
    axes[0].set_xlim(0, max(0.82, float(plot_df["r"].max()) + 0.12))
    axes[0].set_xlabel("marker score vs deconv proportion")
    axes[0].set_title("marker consistency", loc="left", fontsize=7.5, fontweight="bold")

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "nmi"]
    keep = agreement[agreement["truth_column"].isin(["bio_macro_label", "cell_label_original", "cluster_annotation"])].copy()
    keep = keep.set_index("truth_column")[metrics]
    im = axes[1].imshow(keep.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=0.8, aspect="auto")
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=5.8)
    axes[1].set_yticks(range(keep.shape[0]))
    axes[1].set_yticklabels([x.replace("_", " ") for x in keep.index], fontsize=6)
    for i in range(keep.shape[0]):
        for j in range(len(metrics)):
            axes[1].text(j, i, f"{keep.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if keep.iloc[i, j] > 0.35 else "#222222")
    axes[1].set_title("label agreement", loc="left", fontsize=7.5, fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.01)
    fig.suptitle("Spatial validation: broad vascular classes", x=0.02, y=1.02, ha="left", fontsize=8.5, fontweight="bold")
    return save_panel(fig, out_prefix, dpi, pdf_dir)


def plot_spatial_subtype_validation(support: pd.DataFrame, out_prefix: Path, pdf_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    module_rows = support[pd.isna(support["gene"])].copy()
    mat = module_rows.pivot_table(index="label", columns="module", values="module_mean", aggfunc="mean")
    rows = [r for r in CLUSTER_ORDER if r in mat.index] + [r for r in mat.index if r not in CLUSTER_ORDER]
    cols = [c for c in ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"] if c in mat.columns]
    mat = mat.loc[rows, cols].fillna(0)
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=6)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            value = float(mat.loc[row, col])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5, color="white" if value > mat.to_numpy().max() * 0.45 else "#222222")
    ax.set_title("Spatial validation: vascular subtype clusters", loc="left", fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label("marker module mean", fontsize=6.5)
    return save_panel(fig, out_prefix, dpi, pdf_dir)


def write_report(
    out_dir: Path,
    input_dir: Path,
    panel_records: list[dict[str, str]],
    selected_macro: dict[str, list[str]],
    selected_fine: dict[str, list[str]],
    used: dict[str, str],
) -> None:
    lines = [
        "# Figure 3 individual panels v2",
        "",
        "- Core conclusion: vascular classes and subtypes are identifiable in single-cell space and are supported by spatial deconvolution validation.",
        "- Removed QC-only panels.",
        "- UMAP panels use larger points and axis-arrow styling.",
        "- Marker genes and subtype score genes were recomputed from final h5ad marker-panel FindAllMarkers-like tests.",
        "- PDF exports were rendered with dpi=1000 for rasterized scatter layers.",
        f"- Input: `{input_dir}`.",
        f"- Output: `{out_dir}`.",
        "",
        "## Panels",
        "",
        "| Panel | Content | PNG |",
        "| --- | --- | --- |",
    ]
    for rec in panel_records:
        lines.append(f"| {rec['panel']} | {rec['content']} | `{Path(rec['file']).name}` |")
    lines += ["", "## Selected macro score genes", ""]
    for group, genes in selected_macro.items():
        lines.append(f"- {group}: {', '.join(genes)}")
    lines += ["", "## Selected subtype score genes", ""]
    for group, genes in selected_fine.items():
        lines.append(f"- {group}: {', '.join(genes)}")
    lines += ["", "## Source files", ""]
    for key, value in sorted(used.items()):
        lines.append(f"- {key}: `{value}`")
    (out_dir / "Figure3_individual_panels_v2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_dir = resolve_input_dir(root, args.input_dir, args.strict_final_results)
    out_dir = (args.out_dir or input_dir / "figure3_vascular_individual_panels_python_v2").resolve()
    panels_dir = out_dir / "panels"
    pdf_dir = out_dir / "pdf_1000dpi"
    source_dir = out_dir / "source_data"
    panels_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    used: dict[str, str] = {}

    def load_csv(key: str, rel: str, **kwargs) -> pd.DataFrame:
        path = req(input_dir, rel)
        used[key] = str(path)
        return pd.read_csv(path, **kwargs)

    umap = prep_umap(load_csv("single_cell_v12_umap_scores", "01_single_cell/v12_score_dotplot_subtype/source_data/figure3_sc_v12_umap_scores_and_annotations.csv"))
    spatial = load_csv("spatial_v12b_all_chip_annotations", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_all_chip_annotations.csv.gz")
    cor = load_csv("spatial_v12b_marker_deconv_correlations", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_marker_deconv_correlations.csv")
    agreement = load_csv("spatial_v12b_harmonized_label_agreement", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_harmonized_label_agreement.csv")
    support_cluster = load_csv("spatial_v12b_marker_support_by_cluster", "03_spatial_deconvolution_QC/v12b/results/spatial_vascular_deconv_v12b_marker_support_by_cluster.csv")
    h5ad_path = req(input_dir, "01_single_cell/v11_clean_annotation/results/single_cell_clean_vascular_v11.h5ad")
    used["single_cell_v11_h5ad"] = str(h5ad_path)

    obs, expr, genes = read_marker_expression(h5ad_path)
    if len(umap) != expr.shape[0]:
        raise ValueError(f"UMAP source rows ({len(umap)}) do not match h5ad expression rows ({expr.shape[0]}).")

    macro_markers = findmarkers_marker_panel(obs, expr, genes, "v11_marker_class", MACRO_ORDER)
    fine_markers = findmarkers_marker_panel(obs, expr, genes, "bio_fine_label", FINE_ORDER)
    macro_markers.to_csv(source_dir / "findmarkers_like_macro_v11_marker_class.csv", index=False)
    fine_markers.to_csv(source_dir / "findmarkers_like_subtype_bio_fine_label.csv", index=False)

    selected_macro = select_group_genes(macro_markers, MACRO_ORDER, n_per_group=4)
    selected_fine = select_group_genes(fine_markers, FINE_ORDER, n_per_group=3)
    pd.DataFrame([{"group": g, "genes": ",".join(v)} for g, v in selected_macro.items()]).to_csv(source_dir / "selected_macro_findmarker_score_genes.csv", index=False)
    pd.DataFrame([{"group": g, "genes": ",".join(v)} for g, v in selected_fine.items()]).to_csv(source_dir / "selected_subtype_findmarker_score_genes.csv", index=False)
    (source_dir / "selected_findmarker_score_genes.json").write_text(json.dumps({"macro": selected_macro, "subtype": selected_fine}, indent=2), encoding="utf-8")

    umap_scored = add_marker_scores(umap, expr, genes, selected_macro, "fm_macro_")
    umap_scored = add_marker_scores(umap_scored, expr, genes, selected_fine, "fm_subtype_")
    umap_scored.to_csv(source_dir / "single_cell_umap_with_findmarker_scores.csv", index=False)

    macro_dot = make_dot_source(obs, expr, genes, "v11_marker_class", MACRO_ORDER, selected_macro)
    fine_dot = make_dot_source(obs, expr, genes, "bio_fine_label", FINE_ORDER, selected_fine)
    macro_dot.to_csv(source_dir / "dotplot_macro_findmarker_source.csv", index=False)
    fine_dot.to_csv(source_dir / "dotplot_subtype_findmarker_source.csv", index=False)

    panel_records: list[dict[str, str]] = []

    def record(panel: str, content: str, path: Path) -> None:
        panel_records.append({"panel": panel, "content": content, "file": str(path)})

    p = plot_umap_categories(umap_scored, "macro_label", MACRO_COLORS, MACRO_ORDER, "Vascular classes", panels_dir / "fig3a_singlecell_vascular_class_umap", pdf_dir, args.dpi, args.max_umap_cells, point_size=1.25)
    record("a", "single-cell broad vascular UMAP", p)

    macro_cols = [f"fm_macro_{safe_name(g)}" for g in MACRO_ORDER]
    p = plot_score_umap_grid(umap_scored, macro_cols, MACRO_ORDER, "FindAllMarkers-derived vascular scores", panels_dir / "fig3b_macro_findmarker_score_umap", pdf_dir, args.dpi, args.max_umap_cells, ncols=4)
    record("b", "FindAllMarkers-derived macro score UMAP", p)

    p = plot_dotplot(macro_dot, selected_macro, MACRO_ORDER, "Broad vascular marker genes", panels_dir / "fig3c_macro_findmarker_dotplot", pdf_dir, args.dpi)
    record("c", "macro marker dotplot from recomputed markers", p)

    p = plot_umap_categories(umap_scored, "subtype_label", FINE_COLORS, FINE_ORDER, "Vascular subtypes", panels_dir / "fig3d_singlecell_vascular_subtype_umap", pdf_dir, args.dpi, args.max_umap_cells, point_size=1.2)
    record("d", "single-cell vascular subtype UMAP", p)

    fine_cols = [f"fm_subtype_{safe_name(g)}" for g in FINE_ORDER]
    p = plot_score_umap_grid(umap_scored, fine_cols, FINE_ORDER, "FindAllMarkers-derived subtype scores", panels_dir / "fig3e_subtype_findmarker_score_umap", pdf_dir, args.dpi, args.max_umap_cells, ncols=5)
    record("e", "FindAllMarkers-derived subtype score UMAP", p)

    p = plot_dotplot(fine_dot, selected_fine, FINE_ORDER, "Subtype marker genes", panels_dir / "fig3f_subtype_findmarker_dotplot", pdf_dir, args.dpi)
    record("f", "subtype marker dotplot from recomputed markers", p)

    samples = [args.hip_sample, args.cortex_sample]
    p = plot_spatial_pair(spatial, "deconv_dominant_class", MACRO_COLORS, MACRO_ORDER, samples, "Spatial deconvolution: broad classes", panels_dir / "fig3g_spatial_macro_map_AD2_1_T991", pdf_dir, args.dpi)
    record("g", "spatial broad deconvolution map for AD2.1 and T991", p)

    p = plot_spatial_pair(spatial, "deconv_dominant_cluster", CLUSTER_COLORS, CLUSTER_ORDER, samples, "Spatial deconvolution: vascular subtypes", panels_dir / "fig3h_spatial_subtype_map_AD2_1_T991", pdf_dir, args.dpi)
    record("h", "spatial subtype deconvolution map for AD2.1 and T991", p)

    p = plot_spatial_macro_validation(cor, agreement, panels_dir / "fig3i_spatial_macro_validation", pdf_dir, args.dpi)
    record("i", "spatial validation for broad vascular classes", p)

    p = plot_spatial_subtype_validation(support_cluster, panels_dir / "fig3j_spatial_subtype_validation", pdf_dir, args.dpi)
    record("j", "spatial validation for vascular subtype clusters", p)

    pd.DataFrame(panel_records).to_csv(source_dir / "figure3_individual_panel_manifest.csv", index=False)
    spatial[sample_mask(spatial, args.hip_sample) | sample_mask(spatial, args.cortex_sample)].to_csv(source_dir / "spatial_selected_samples_AD2_1_T991.csv", index=False)
    (source_dir / "resolved_source_files.json").write_text(json.dumps(used, indent=2), encoding="utf-8")
    write_report(out_dir, input_dir, panel_records, selected_macro, selected_fine, used)
    print(f"Wrote individual Figure 3 panels to: {out_dir}")


if __name__ == "__main__":
    main()
