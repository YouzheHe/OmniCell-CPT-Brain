#!/usr/bin/env python
"""Align single-cell and spatial vascular clusters by marker signatures."""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
PROJECT = WORK_ROOT / "projects" / "nvu_vascular"
BASE_RESULTS = PROJECT / "results" / "vascular_omnicell_cpt_nonzero_hvg_all_data" / "k05_k10_scanpy_findallmarkers"
BASE_FIGURES = PROJECT / "figures" / "vascular_omnicell_cpt_nonzero_hvg_all_data" / "k05_k10_scanpy_findallmarkers"


PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
}

ALIGNED_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#9C755F",
    "#FF9DA7",
    "#BAB0AC",
]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 6.4,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "legend.frameon": False,
        "agg.path.chunksize": 20000,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=BASE_RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=BASE_FIGURES)
    parser.add_argument("--k-values", default="5,6,7,8,9,10")
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=900)
    return parser.parse_args()


def save(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def parse_k_values(text: str) -> list[int]:
    return sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})


def signature_matrix(markers: pd.DataFrame, modality: str, k: int, top_n: int) -> tuple[list[str], list[str], np.ndarray, dict[str, set[str]]]:
    sub = markers[(markers["modality"].astype(str) == modality) & (markers["k"].astype(int) == int(k))].copy()
    sub = sub[sub["rank"].astype(int) <= int(top_n)].copy()
    sub = sub[np.isfinite(sub["score"].astype(float))]
    clusters = sorted(sub["cluster_id"].astype(str).unique())
    genes = sorted(sub["gene_symbol"].fillna(sub["gene_id"]).astype(str).unique())
    gene_to_i = {gene: i for i, gene in enumerate(genes)}
    mat = np.zeros((len(clusters), len(genes)), dtype=np.float32)
    top_sets: dict[str, set[str]] = {}
    cluster_to_i = {cluster: i for i, cluster in enumerate(clusters)}
    for row in sub.itertuples(index=False):
        cluster = str(row.cluster_id)
        gene = str(row.gene_symbol) if pd.notna(row.gene_symbol) else str(row.gene_id)
        rank = int(row.rank)
        score = max(float(row.score), 0.0)
        logfc = max(float(row.logfoldchanges), 0.0) if np.isfinite(float(row.logfoldchanges)) else 0.0
        weight = (top_n - rank + 1) / top_n
        mat[cluster_to_i[cluster], gene_to_i[gene]] = max(mat[cluster_to_i[cluster], gene_to_i[gene]], weight * (1.0 + logfc) * np.log1p(score))
        top_sets.setdefault(cluster, set()).add(gene)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = np.divide(mat, np.maximum(norm, 1e-8), out=np.zeros_like(mat), where=norm > 0)
    return clusters, genes, mat, top_sets


def align_one_k(markers: pd.DataFrame, k: int, top_n: int) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    sc_clusters, sc_genes, sc_mat, sc_sets = signature_matrix(markers, "single_cell", k, top_n)
    sp_clusters, sp_genes, sp_mat, sp_sets = signature_matrix(markers, "spatial", k, top_n)
    genes = sorted(set(sc_genes).union(sp_genes))
    gene_to_i = {gene: i for i, gene in enumerate(genes)}

    def expand(mat: np.ndarray, old_genes: list[str]) -> np.ndarray:
        out = np.zeros((mat.shape[0], len(genes)), dtype=np.float32)
        for old_i, gene in enumerate(old_genes):
            out[:, gene_to_i[gene]] = mat[:, old_i]
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, np.maximum(norm, 1e-8), out=np.zeros_like(out), where=norm > 0)

    sc = expand(sc_mat, sc_genes)
    sp = expand(sp_mat, sp_genes)
    sim = sc @ sp.T
    row_ind, col_ind = linear_sum_assignment(-sim)

    rows = []
    sc_label_map: dict[str, str] = {}
    sp_label_map: dict[str, str] = {}
    for aligned_i, (r, c) in enumerate(zip(row_ind, col_ind)):
        sc_cluster = sc_clusters[int(r)]
        sp_cluster = sp_clusters[int(c)]
        aligned = f"VC{aligned_i:02d}"
        sc_label_map[sc_cluster] = aligned
        sp_label_map[sp_cluster] = aligned
        shared = sorted(sc_sets.get(sc_cluster, set()).intersection(sp_sets.get(sp_cluster, set())))
        rows.append(
            {
                "k": int(k),
                "aligned_label": aligned,
                "single_cell_cluster": sc_cluster,
                "spatial_cluster": sp_cluster,
                "marker_cosine_similarity": float(sim[int(r), int(c)]),
                "n_shared_top_markers": int(len(shared)),
                "shared_top_markers": ";".join(shared[:40]),
            }
        )
    return pd.DataFrame(rows).sort_values("aligned_label"), sc_label_map, sp_label_map


def draw_aligned_grid(umap: pd.DataFrame, mapping: pd.DataFrame, k_values: list[int], fig_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, len(k_values), figsize=(2.45 * len(k_values), 4.95), squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.90, bottom=0.07, wspace=0.08, hspace=0.18)
    for col_i, k in enumerate(k_values):
        map_k = mapping[mapping["k"].astype(int) == int(k)].copy()
        color_map = {row.aligned_label: ALIGNED_COLORS[i % len(ALIGNED_COLORS)] for i, row in enumerate(map_k.itertuples(index=False))}
        for row_i, modality in enumerate(["single_cell", "spatial"]):
            ax = axes[row_i, col_i]
            frame = umap[(umap["k"].astype(int) == int(k)) & (umap["modality"].astype(str) == modality)].copy()
            for aligned in sorted(frame["aligned_label"].dropna().astype(str).unique()):
                sub = frame[frame["aligned_label"].astype(str) == aligned]
                ax.scatter(
                    sub["umap_1"],
                    sub["umap_2"],
                    s=0.25 if modality == "spatial" else 0.32,
                    color=color_map.get(aligned, "#C7C9CC"),
                    alpha=0.76,
                    linewidths=0,
                    rasterized=True,
                )
                if len(sub) > 0:
                    ax.text(sub["umap_1"].median(), sub["umap_2"].median(), aligned.replace("VC", ""), fontsize=3.9, fontweight="bold", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            if row_i == 0:
                ax.set_title(f"k={k}", loc="left", fontsize=6.8, fontweight="bold")
            if col_i == 0:
                ax.text(-0.12, 0.5, modality, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=6.6, fontweight="bold")
            for spn in ax.spines.values():
                spn.set_linewidth(0.45)
    fig.text(0.035, 0.965, "Marker-aligned OmniCell vascular clusters across single-cell and spatial anchors", ha="left", va="top", fontsize=8.0, fontweight="bold")
    fig.text(0.035, 0.925, "Colors and labels are aligned within each k by cross-modality marker-signature similarity.", ha="left", va="top", fontsize=5.4, color=PALETTE["muted"])
    save(fig, fig_dir / "vascular_omnicell_k05_k10_umap_grid_sc_spatial_marker_aligned", dpi=dpi)


def main() -> None:
    args = parse_args()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.figure_dir = args.figure_dir.expanduser().resolve()
    k_values = parse_k_values(args.k_values)
    markers = pd.read_csv(args.results_dir / "scanpy_findallmarkers_all_k05_k10_hvg_t-test_overestim_var.csv")
    umap_source = pd.read_csv(args.figure_dir / "vascular_nonzero_hvg_k05_k10_umap_source_data.csv")

    mapping_rows = []
    label_maps: dict[int, dict[str, dict[str, str]]] = {}
    for k in k_values:
        map_k, sc_map, sp_map = align_one_k(markers, k, args.top_n)
        mapping_rows.append(map_k)
        label_maps[int(k)] = {"single_cell": sc_map, "spatial": sp_map}
    mapping = pd.concat(mapping_rows, ignore_index=True)
    mapping.to_csv(args.results_dir / "vascular_k05_k10_sc_spatial_marker_alignment.csv", index=False)

    long_rows = []
    for k in k_values:
        for modality in ["single_cell", "spatial"]:
            label_col = f"modality_cluster_k{k:02d}"
            frame = umap_source[umap_source["modality"].astype(str) == modality].copy()
            frame["k"] = int(k)
            frame["original_cluster"] = frame[label_col].astype(str)
            frame["aligned_label"] = frame["original_cluster"].map(label_maps[int(k)][modality])
            long_rows.append(frame[["sample_id", "cell_index", "modality", "k", "original_cluster", "aligned_label", "umap_1", "umap_2"]])
    aligned_umap = pd.concat(long_rows, ignore_index=True)
    aligned_umap.to_csv(args.figure_dir / "vascular_k05_k10_umap_marker_aligned_source_data.csv", index=False)
    draw_aligned_grid(aligned_umap, mapping, k_values, args.figure_dir, args.dpi)

    contract = {
        "reason": "Original cluster numbers/colors were independently assigned by modality and k; this run aligns single-cell and spatial cluster labels by marker-signature cosine similarity.",
        "marker_table": str(args.results_dir / "scanpy_findallmarkers_all_k05_k10_hvg_t-test_overestim_var.csv"),
        "alignment_table": str(args.results_dir / "vascular_k05_k10_sc_spatial_marker_alignment.csv"),
        "aligned_figure": str(args.figure_dir / "vascular_omnicell_k05_k10_umap_grid_sc_spatial_marker_aligned.png"),
        "top_n_markers_per_cluster_used": int(args.top_n),
        "k_values": k_values,
    }
    (args.results_dir / "vascular_k05_k10_marker_alignment_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
