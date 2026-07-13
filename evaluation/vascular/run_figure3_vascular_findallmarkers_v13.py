#!/usr/bin/env python
"""Run Figure 3 vascular subtype marker discovery on the remote v11 clean h5ad.

This is the rerun requested for the Figure 3 subtype annotation layer. It uses
the v10 fine-tuned/v11 cleaned single-cell vascular result as input, converts
the saved expanded-marker expression matrix into a temporary AnnData object,
and runs Scanpy rank_genes_groups for both clean subclusters and broad classes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


BASE = (
    "${OMNICELL_NVU_ROOT}/projects/nvu_vascular/results/"
    "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
)
FIG_BASE = (
    "${OMNICELL_NVU_ROOT}/projects/nvu_vascular/figures/"
    "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
)
DEFAULT_H5AD = f"{BASE}/vascular_clean_diagonal_v11/single_cell/single_cell_clean_vascular_v11.h5ad"
DEFAULT_RESULT_DIR = f"{BASE}/figure3_vascular_findallmarkers_v13/single_cell"
DEFAULT_FIGURE_DIR = f"{FIG_BASE}/figure3_vascular_findallmarkers_v13/single_cell"

CLASS_ORDER = ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"]
MODULES = {
    "Endothelial_core": ["CLDN5", "PECAM1", "CDH5", "VWF", "ESAM", "ERG", "ICAM2", "FLT1", "KDR"],
    "BBB_capillary": ["SLC2A1", "MFSD2A", "ABCB1", "ABCG2", "CA4", "RGCC", "SLC7A5"],
    "Arterial_EC": ["EFNB2", "GJA5", "SOX17", "DLL4", "NOTCH4", "HEY1"],
    "Venous_activated_EC": ["ACKR1", "VCAM1", "SELE", "NR2F2", "VWF"],
    "Pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9", "NOTCH3", "CSPG4", "MCAM", "NDUFA4L2"],
    "SMC": ["ACTA2", "TAGLN", "MYH11", "CNN1", "MYLK", "MYOCD", "SMTN", "CALD1"],
    "Fibroblast_VLMC": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "DCN", "LUM", "APOD", "CFD", "SLC6A13"],
    "Contaminant_glia_neuron": ["GFAP", "AQP4", "PLP1", "MBP", "P2RY12", "CX3CR1", "RBFOX3", "SNAP25", "HBB", "HBA1"],
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
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    p.add_argument("--matrix-key", default="expanded_marker_log1p")
    p.add_argument("--gene-key", default="expanded_marker_genes")
    p.add_argument("--cluster-key", default="v11_clean_cluster")
    p.add_argument("--class-key", default="v11_marker_class")
    p.add_argument("--method", default="wilcoxon", choices=["wilcoxon", "t-test_overestim_var", "t-test"])
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--dotplot-top-per-cluster", type=int, default=4)
    p.add_argument("--min-pct-in", type=float, default=0.08)
    p.add_argument("--min-pct-delta", type=float, default=0.03)
    p.add_argument("--dpi", type=int, default=800)
    return p.parse_args()


def save_all(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def nuisance_gene(gene: str) -> bool:
    g = str(gene).upper()
    if g.startswith(("MT-", "RPL", "RPS", "MRPL", "MRPS", "HLA-")):
        return True
    if g.startswith("ENSG") or g in {"MALAT1", "NEAT1", "XIST", "FTX"}:
        return True
    return bool(re.match(r"^(AC|AL|AP|RP11|LINC|MIR|SNHG|RN7|RNU|RNVU|SCARNA|SMIM|C\d+ORF)", g))


def make_expression_adata(adata: sc.AnnData, matrix_key: str, gene_key: str, obs_cols: list[str]) -> sc.AnnData:
    if matrix_key not in adata.obsm:
        raise KeyError(f"{matrix_key} missing from input obsm")
    if gene_key not in adata.uns:
        raise KeyError(f"{gene_key} missing from input uns")
    x = np.asarray(adata.obsm[matrix_key], dtype=np.float32)
    genes = list(map(str, adata.uns[gene_key]))
    obs = adata.obs[[c for c in obs_cols if c in adata.obs]].copy()
    out = sc.AnnData(X=x, obs=obs)
    out.var_names = pd.Index(genes)
    out.var_names_make_unique()
    return out


def pct_by_group(marker: sc.AnnData, groupby: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    labels = marker.obs[groupby].astype(str)
    groups = sorted(labels.unique(), key=lambda z: (len(z), z))
    x = marker.X
    out = {}
    for group in groups:
        mask = labels.eq(group).to_numpy()
        if int(mask.sum()) == 0:
            continue
        pct_in = np.asarray((x[mask] > 0).mean(axis=0)).ravel()
        pct_out = np.asarray((x[~mask] > 0).mean(axis=0)).ravel()
        out[group] = (pct_in, pct_out)
    return out


def rank_to_table(marker: sc.AnnData, groupby: str, args: argparse.Namespace) -> pd.DataFrame:
    marker.obs[groupby] = marker.obs[groupby].astype(str).astype("category")
    sc.tl.rank_genes_groups(marker, groupby=groupby, method=args.method, pts=False, tie_correct=True)
    res = marker.uns["rank_genes_groups"]
    groups = list(res["names"].dtype.names)
    pct = pct_by_group(marker, groupby)
    gene_to_idx = {g: i for i, g in enumerate(marker.var_names.astype(str))}
    rows = []
    for group in groups:
        for i in range(min(args.top_n, len(res["names"][group]))):
            gene = str(res["names"][group][i])
            idx = gene_to_idx.get(gene)
            pct_in = float(pct[group][0][idx]) if idx is not None and group in pct else np.nan
            pct_out = float(pct[group][1][idx]) if idx is not None and group in pct else np.nan
            rows.append(
                {
                    "group": group,
                    "rank": i + 1,
                    "gene": gene,
                    "score": float(res["scores"][group][i]),
                    "logfoldchange": float(res["logfoldchanges"][group][i]) if "logfoldchanges" in res else np.nan,
                    "pval": float(res["pvals"][group][i]) if "pvals" in res else np.nan,
                    "pval_adj": float(res["pvals_adj"][group][i]) if "pvals_adj" in res else np.nan,
                    "pct_in": pct_in,
                    "pct_out": pct_out,
                    "pct_delta": pct_in - pct_out if np.isfinite(pct_in) and np.isfinite(pct_out) else np.nan,
                    "nuisance_gene": nuisance_gene(gene),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["passes_filter"] = (
        (~df["nuisance_gene"])
        & (df["pct_in"] >= args.min_pct_in)
        & (df["pct_delta"] >= args.min_pct_delta)
        & (df["pval_adj"].fillna(1.0) <= 0.05)
    )
    return df


def filtered_top(df: pd.DataFrame, n_per_group: int) -> pd.DataFrame:
    if df.empty:
        return df
    keep = df[df["passes_filter"]].copy()
    if keep.empty:
        keep = df[~df["nuisance_gene"]].copy()
    return keep.sort_values(["group", "rank"]).groupby("group", group_keys=False).head(n_per_group).copy()


def dotplot_source(marker: sc.AnnData, groupby: str, genes: list[str]) -> pd.DataFrame:
    genes = [g for g in dict.fromkeys(genes) if g in marker.var_names]
    labels = marker.obs[groupby].astype(str)
    groups = sorted(labels.unique(), key=lambda z: (len(z), z))
    gene_idx = [marker.var_names.get_loc(g) for g in genes]
    rows = []
    x = marker.X[:, gene_idx]
    for group in groups:
        mask = labels.eq(group).to_numpy()
        for j, gene in enumerate(genes):
            vals = np.asarray(x[mask, j]).ravel()
            rows.append(
                {
                    "group": group,
                    "gene": gene,
                    "mean_log1p": float(vals.mean()) if len(vals) else np.nan,
                    "pct_expr": float((vals > 0).mean()) if len(vals) else np.nan,
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_dotplot(src: pd.DataFrame, title: str, path: Path, dpi: int) -> None:
    if src.empty:
        return
    groups = src["group"].drop_duplicates().tolist()
    genes = src["gene"].drop_duplicates().tolist()
    mean = src.pivot(index="group", columns="gene", values="mean_log1p").reindex(index=groups, columns=genes)
    pct = src.pivot(index="group", columns="gene", values="pct_expr").reindex(index=groups, columns=genes)
    z = mean.copy()
    for gene in genes:
        vals = z[gene].to_numpy(float)
        sd = np.nanstd(vals)
        z[gene] = (vals - np.nanmean(vals)) / (sd if sd > 1e-6 else 1.0)
    z = z.clip(-1.8, 1.8)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.16 * len(genes) + 1.5), max(2.7, 0.22 * len(groups) + 0.8)))
    for yi, group in enumerate(groups):
        for xi, gene in enumerate(genes):
            ax.scatter(xi, yi, s=5 + 60 * float(pct.loc[group, gene]), c=float(z.loc[group, gene]), cmap="viridis", vmin=-1.8, vmax=1.8, lw=0)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=6.5)
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=-1.8, vmax=1.8))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("gene-wise z-score", fontsize=6)
    save_all(fig, path, dpi)


def module_dotplot(marker: sc.AnnData, groupby: str, result_dir: Path, figure_dir: Path, dpi: int) -> pd.DataFrame:
    genes = list(marker.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows = []
    labels = marker.obs[groupby].astype(str)
    groups = sorted(labels.unique(), key=lambda z: (len(z), z))
    for group in groups:
        mask = labels.eq(group).to_numpy()
        for module, module_genes in MODULES.items():
            keep = [gene_to_idx[g] for g in module_genes if g in gene_to_idx]
            if not keep:
                continue
            x = np.asarray(marker.X[mask][:, keep], dtype=np.float32)
            rows.append(
                {
                    "group": group,
                    "module": module,
                    "genes_found": ",".join([g for g in module_genes if g in gene_to_idx]),
                    "score_mean": float(x.mean()),
                    "score_pos_pct": float((x > 0).mean()),
                    "n": int(mask.sum()),
                }
            )
    src = pd.DataFrame(rows)
    src.to_csv(result_dir / f"figure3_vascular_v13_{groupby}_module_dotplot_source.csv", index=False)
    if src.empty:
        return src
    modules = list(src["module"].drop_duplicates())
    mean = src.pivot(index="group", columns="module", values="score_mean").reindex(index=groups, columns=modules)
    pct = src.pivot(index="group", columns="module", values="score_pos_pct").reindex(index=groups, columns=modules)
    fig, ax = plt.subplots(figsize=(max(5.2, 0.35 * len(modules) + 1.2), max(2.7, 0.22 * len(groups) + 0.8)))
    vmax = max(0.6, float(np.nanmax(np.abs(mean.to_numpy()))))
    for yi, group in enumerate(groups):
        for xi, module in enumerate(modules):
            ax.scatter(xi, yi, s=6 + 55 * float(pct.loc[group, module]), c=float(mean.loc[group, module]), cmap="coolwarm", vmin=-vmax, vmax=vmax, lw=0)
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(modules, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=6.5)
    ax.set_title(f"Figure 3 vascular marker modules: {groupby}", loc="left", fontsize=9, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("mean log1p module score", fontsize=6)
    save_all(fig, figure_dir / f"figure3_vascular_v13_{groupby}_module_dotplot", dpi)
    return src


def cluster_annotation_summary(marker: sc.AnnData, cluster_markers: pd.DataFrame, result_dir: Path) -> pd.DataFrame:
    obs = marker.obs.copy()
    rows = []
    for group, sub in obs.groupby("v11_clean_cluster", observed=False):
        top = cluster_markers[(cluster_markers["group"].astype(str) == str(group)) & cluster_markers["passes_filter"]]
        top_genes = ", ".join(top.sort_values("rank")["gene"].astype(str).head(12))
        klass = sub["v11_marker_class"].astype(str).mode()
        rows.append(
            {
                "v11_clean_cluster": str(group),
                "v11_marker_class": klass.iloc[0] if len(klass) else "",
                "n": int(len(sub)),
                "top_filtered_markers": top_genes,
                "sample_top3": "; ".join(sub["sample_id"].astype(str).value_counts().head(3).index.tolist()) if "sample_id" in sub else "",
                "condition_composition": "; ".join([f"{k}:{v}" for k, v in sub.get("condition_inferred", pd.Series(["Unknown"] * len(sub), index=sub.index)).astype(str).value_counts().head(4).items()]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(result_dir / "figure3_vascular_v13_cluster_annotation_summary.csv", index=False)
    return out


def run_one(marker: sc.AnnData, groupby: str, args: argparse.Namespace, result_dir: Path, figure_dir: Path) -> pd.DataFrame:
    table = rank_to_table(marker, groupby, args)
    table.to_csv(result_dir / f"figure3_vascular_v13_findallmarkers_{groupby}_top{args.top_n}.csv", index=False)
    top = filtered_top(table, args.dotplot_top_per_cluster)
    top.to_csv(result_dir / f"figure3_vascular_v13_findallmarkers_{groupby}_filtered_top{args.dotplot_top_per_cluster}.csv", index=False)
    genes = top.sort_values(["group", "rank"])["gene"].astype(str).tolist()
    src = dotplot_source(marker, groupby, genes)
    src.to_csv(result_dir / f"figure3_vascular_v13_findallmarkers_{groupby}_dotplot_source.csv", index=False)
    plot_dotplot(src, f"Figure 3 vascular FindAllMarker: {groupby}", figure_dir / f"figure3_vascular_v13_findallmarkers_{groupby}_dotplot", args.dpi)
    module_dotplot(marker, groupby, result_dir, figure_dir, args.dpi)
    return table


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    required = [args.cluster_key, args.class_key, "sample_id", "condition_inferred"]
    marker = make_expression_adata(adata, args.matrix_key, args.gene_key, required)
    marker.obs[args.cluster_key] = marker.obs[args.cluster_key].astype(str).astype("category")
    marker.obs[args.class_key] = marker.obs[args.class_key].astype(str).astype("category")

    cluster_markers = run_one(marker, args.cluster_key, args, result_dir, figure_dir)
    class_markers = run_one(marker, args.class_key, args, result_dir, figure_dir)
    cluster_summary = cluster_annotation_summary(marker, cluster_markers, result_dir)

    xlsx = result_dir / "figure3_vascular_v13_findallmarkers_workbook.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        cluster_markers.to_excel(writer, sheet_name="cluster_all_top", index=False)
        filtered_top(cluster_markers, 20).to_excel(writer, sheet_name="cluster_filtered_top20", index=False)
        class_markers.to_excel(writer, sheet_name="class_all_top", index=False)
        filtered_top(class_markers, 30).to_excel(writer, sheet_name="class_filtered_top30", index=False)
        cluster_summary.to_excel(writer, sheet_name="cluster_summary", index=False)

    summary = {
        "input_h5ad": str(args.h5ad),
        "n_cells": int(marker.n_obs),
        "n_genes": int(marker.n_vars),
        "cluster_key": args.cluster_key,
        "class_key": args.class_key,
        "method": args.method,
        "top_n": args.top_n,
        "cluster_marker_rows": int(len(cluster_markers)),
        "class_marker_rows": int(len(class_markers)),
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
        "workbook": str(xlsx),
    }
    (result_dir / "figure3_vascular_v13_findallmarkers_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
