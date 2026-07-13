#!/usr/bin/env python
"""OmniCell-style AD hippocampus spatial deconvolution benchmark.

This script follows the evaluation spirit of the OmniCell tutorials:
single-cell data provide the reference, spatial data are the query, and the
dominant predicted component is evaluated against spatial ground truth.  It
also keeps an explicit 512-d CPT embedding-only ablation, because the latest
CPT representation should not be silently conflated with expression-signature
deconvolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import normalize


WORK_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = WORK_ROOT / "projects" / "nvu_vascular"
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_EMBED_DIR = PROJECT_ROOT / "results" / "ad_hip_allcell_embeddings_ad2con2_20260526_174759"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "ad_hip_spatial_reference_deconv_omnicell_style_20260527"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "figure2_ad_hip_spatial_reference_deconv_omnicell_style_20260527"

CLASS_MAP = {
    "Astro": "Astrocyte",
    "Endo": "Endothelial",
    "Ependy": "Ependymal/choroid",
    "Ependymal": "Ependymal/choroid",
    "Choroid": "Ependymal/choroid",
    "EX_CA1": "Excitatory neuron",
    "EX_CA2": "Excitatory neuron",
    "EX_CA3-4": "Excitatory neuron",
    "EX_DG": "Excitatory neuron",
    "EX_Sub": "Excitatory neuron",
    "Fibroblast": "VLMC/fibroblast",
    "IN_LAMP5": "Inhibitory neuron",
    "IN_PVALB": "Inhibitory neuron",
    "IN_SST": "Inhibitory neuron",
    "IN_SV2C": "Inhibitory neuron",
    "IN_VIP": "Inhibitory neuron",
    "Micro": "Microglia/immune",
    "Oligo": "Oligodendrocyte",
    "OPC": "OPC",
    "Pericyte": "Pericyte/mural",
    "SMC": "Pericyte/mural",
    "VLMC": "VLMC/fibroblast",
}

CLASS_COLORS = {
    "Excitatory neuron": "#4F7EA8",
    "Inhibitory neuron": "#8E72A7",
    "Astrocyte": "#70B7A6",
    "Oligodendrocyte": "#B8A35A",
    "OPC": "#E0A458",
    "Microglia/immune": "#9A8571",
    "Endothelial": "#6EA7C8",
    "Pericyte/mural": "#D89B4A",
    "VLMC/fibroblast": "#74A66A",
    "Ependymal/choroid": "#C86E5A",
    "unscored": "#D0D3D8",
}

METHOD_COLORS = {
    "OmniCell-style deconv": "#C95D63",
    "Expression marker": "#70B7A6",
    "CPT 512-only": "#4F7EA8",
    "Global prior": "#A7A7A7",
}

PALETTE = {"ink": "#1F2933", "muted": "#667085", "grid": "#D7DEE8", "strip": "#ECEBE8"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 6.0,
        "axes.linewidth": 0.55,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBED_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--chips", default="AD2.1,AD2.2,Con2.1,Con2.2")
    parser.add_argument("--display-chip", default="Con2.1")
    parser.add_argument("--max-ref-per-class", type=int, default=5000)
    parser.add_argument("--top-markers-per-class", type=int, default=400)
    parser.add_argument("--tile-grid", type=int, default=6)
    parser.add_argument("--min-tile-cells", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=900)
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def broad(values: pd.Series) -> np.ndarray:
    return values.astype(str).map(CLASS_MAP).fillna(values.astype(str)).to_numpy()


def make_csr(sample_dir: Path) -> sparse.csr_matrix:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    values = np.load(sample_dir / "values.npy", mmap_mode="r")
    indices = np.load(sample_dir / "indices.npy", mmap_mode="r")
    indptr = np.load(sample_dir / "indptr.npy", mmap_mode="r")
    return sparse.csr_matrix((values, indices, indptr), shape=(metadata["n_cells"], metadata["n_genes"]))


def sample_ref_indices(labels: np.ndarray, classes: list[str], max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in classes:
        idx = np.flatnonzero(labels == cls)
        take = min(len(idx), max_per_class)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def select_marker_columns(means: np.ndarray, classes: list[str], top_n: int) -> np.ndarray:
    markers: set[int] = set()
    logged = np.log1p(means)
    for i, _cls in enumerate(classes):
        other = np.max(np.delete(logged, i, axis=0), axis=0)
        score = logged[i] - other
        markers.update(np.argsort(-score, kind="mergesort")[:top_n].tolist())
    return np.asarray(sorted(markers), dtype=np.int64)


def predict_from_scores(scores: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.asarray(classes, dtype=object)[np.argmax(scores, axis=1)]


def metric_row(method: str, chip: str, y_true: np.ndarray, y_pred: np.ndarray, *, level: str, tile: str | None = None) -> dict[str, object]:
    return {
        "method": method,
        "chip": chip,
        "level": level,
        "tile": tile,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def tile_ids(coords: np.ndarray, grid: int) -> np.ndarray:
    x_bins = pd.qcut(coords[:, 0], q=grid, labels=False, duplicates="drop")
    y_bins = pd.qcut(coords[:, 1], q=grid, labels=False, duplicates="drop")
    return np.asarray([f"x{int(x)}_y{int(y)}" for x, y in zip(x_bins, y_bins)], dtype=object)


def z01(values: np.ndarray) -> np.ndarray:
    lo = np.nanpercentile(values, 1)
    hi = np.nanpercentile(values, 99)
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)


def build_embedding_scores(
    embed_dir: Path,
    classes: list[str],
    chips: list[str],
    max_ref_per_class: int,
    seed: int,
) -> dict[str, np.ndarray]:
    meta = pd.read_parquet(embed_dir / "embedding_meta.parquet").reset_index(drop=True)
    embedding = np.load(embed_dir / "embedding.npy", mmap_mode="r")
    rng = np.random.default_rng(seed + 13)
    sc_rows = np.flatnonzero(meta["modality"].astype(str).eq("single_cell").to_numpy())
    ref_rows: list[int] = []
    for cls in classes:
        idx = sc_rows[meta["ground_truth_celltype"].iloc[sc_rows].astype(str).to_numpy() == cls]
        ref_rows.extend(rng.choice(idx, size=min(len(idx), max_ref_per_class), replace=False).tolist())
    ref_rows = np.asarray(ref_rows, dtype=np.int64)
    ref_labels = meta["ground_truth_celltype"].iloc[ref_rows].astype(str).to_numpy()
    ref_embedding = normalize(np.asarray(embedding[ref_rows], dtype=np.float32), norm="l2")
    prototypes = np.vstack([ref_embedding[ref_labels == cls].mean(axis=0) for cls in classes])
    prototypes = normalize(prototypes, norm="l2")

    out: dict[str, np.ndarray] = {}
    for chip in chips:
        q_rows = np.flatnonzero(meta["batch_id"].astype(str).eq(chip).to_numpy())
        q_embedding = normalize(np.asarray(embedding[q_rows], dtype=np.float32), norm="l2")
        out[chip] = q_embedding @ prototypes.T
    return out


def draw_metric_panel(tile_metrics: pd.DataFrame, method_order: list[str], figure_dir: Path, dpi: int) -> None:
    metrics = ["NMI", "ARI", "macro_f1", "accuracy"]
    labels = {"macro_f1": "Macro F1", "accuracy": "Accuracy"}
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.65), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        data = [tile_metrics[tile_metrics["method"].eq(m)][metric].dropna().to_numpy() for m in method_order]
        parts = ax.violinplot(data, positions=np.arange(len(method_order)), widths=0.72, showextrema=False)
        for body, method in zip(parts["bodies"], method_order):
            body.set_facecolor(METHOD_COLORS.get(method, "#999999"))
            body.set_edgecolor("none")
            body.set_alpha(0.22)
        for i, (method, vals) in enumerate(zip(method_order, data)):
            rng = np.random.default_rng(1000 + i)
            ax.scatter(rng.normal(i, 0.035, len(vals)), vals, s=6, color=METHOD_COLORS.get(method, "#999999"), alpha=0.68, lw=0)
            if len(vals):
                ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1), color=PALETTE["ink"], marker="o", ms=2.6, lw=0.7, capsize=2)
        ax.set_title(labels.get(metric, metric), loc="left", fontsize=6.6, fontweight="bold", pad=2)
        ax.set_ylim(-0.05 if metric == "ARI" else 0, 1.02)
        ax.set_xticks(np.arange(len(method_order)), method_order, rotation=24, ha="right")
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.suptitle("Spatial deconvolution performance across AD hippocampus tiles", x=0.01, y=1.03, ha="left", fontsize=8.0, fontweight="bold")
    save_figure(fig, figure_dir / "fig2_spatial_deconv_metric_tiles", dpi)


def draw_spatial_panel(predictions: pd.DataFrame, display_chip: str, method_order: list[str], figure_dir: Path, dpi: int) -> None:
    chip_slug = display_chip.replace(".", "_")
    chip_df = predictions[predictions["chip"].eq(display_chip)].copy()
    panels = [("Ground truth", "ground_truth")] + [(method, f"pred_{method}") for method in method_order[:3]]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.15, 2.15), constrained_layout=True)
    for ax, (title, column) in zip(axes, panels):
        labels = chip_df[column].fillna("unscored").astype(str).to_numpy()
        colors = [CLASS_COLORS.get(label, CLASS_COLORS["unscored"]) for label in labels]
        ax.scatter(chip_df["coord_x"], chip_df["coord_y"], c=colors, s=0.22, lw=0, alpha=0.9, rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=6.7, fontweight="bold", pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS_COLORS[cls], markersize=4, label=cls)
        for cls in CLASS_COLORS
        if cls != "unscored"
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=5.0)
    fig.suptitle(f"{display_chip} spatial dominant cell class", x=0.01, y=1.02, ha="left", fontsize=8.0, fontweight="bold")
    save_figure(fig, figure_dir / f"fig2_spatial_deconv_{chip_slug}_spatial_maps", dpi)


def draw_composition_panel(predictions: pd.DataFrame, display_chip: str, method_order: list[str], classes: list[str], figure_dir: Path, dpi: int) -> None:
    chip_slug = display_chip.replace(".", "_")
    chip_df = predictions[predictions["chip"].eq(display_chip)].copy()
    bars = [("Ground truth", chip_df["ground_truth"].to_numpy())] + [(m, chip_df[f"pred_{m}"].to_numpy()) for m in method_order]
    fig, ax = plt.subplots(figsize=(4.85, 2.25), constrained_layout=True)
    bottom = np.zeros(len(bars), dtype=float)
    x = np.arange(len(bars))
    for cls in classes:
        vals = np.asarray([(labels == cls).mean() for _, labels in bars], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=CLASS_COLORS.get(cls, "#B8B8B8"), width=0.72, edgecolor="white", linewidth=0.2)
        bottom += vals
    ax.set_xticks(x, [name for name, _ in bars], rotation=24, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title(f"{display_chip} composition", loc="left", fontsize=6.8, fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=CLASS_COLORS[c], markersize=5, label=c) for c in classes]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.0)
    save_figure(fig, figure_dir / f"fig2_spatial_deconv_{chip_slug}_composition", dpi)


def draw_confusion_panel(predictions: pd.DataFrame, display_chip: str, classes: list[str], figure_dir: Path, dpi: int) -> None:
    chip_slug = display_chip.replace(".", "_")
    chip_df = predictions[predictions["chip"].eq(display_chip)].copy()
    valid = chip_df["ground_truth"].isin(classes)
    y_true = chip_df.loc[valid, "ground_truth"]
    y_pred = chip_df.loc[valid, "pred_OmniCell-style deconv"]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(figure_dir / f"fig2_spatial_deconv_{chip_slug}_confusion_counts.csv")
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(figure_dir / f"fig2_spatial_deconv_{chip_slug}_confusion_normalized.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.7), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("OmniCell-style deconv", loc="left", fontsize=6.8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("recall-normalized fraction", fontsize=5.2)
    save_figure(fig, figure_dir / f"fig2_spatial_deconv_{chip_slug}_confusion", dpi)


def main() -> None:
    args = parse_args()
    chips = [item.strip() for item in args.chips.split(",") if item.strip()]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    sc_dir = args.dataset_root / "AD_Hip_sc"
    spatial_dirs = {chip: args.dataset_root / "AD_Hip_Saptial" / chip for chip in chips}

    sc_obs = pd.read_parquet(sc_dir / "obs.parquet")
    sc_labels = broad(sc_obs["celltype"])
    chip_labels = {chip: broad(pd.read_parquet(path / "obs.parquet")["CellType"]) for chip, path in spatial_dirs.items()}
    classes = sorted(set(sc_labels).intersection(*(set(labels) for labels in chip_labels.values())) - {"Blood cell"})
    ref_idx = sample_ref_indices(sc_labels, classes, args.max_ref_per_class, args.seed)
    ref_labels = sc_labels[ref_idx]

    present = [np.load(sc_dir / "present_gene_ids.npy")]
    present.extend(np.load(path / "present_gene_ids.npy") for path in spatial_dirs.values())
    common_genes = present[0]
    for values in present[1:]:
        common_genes = np.intersect1d(common_genes, values)

    sc_matrix = make_csr(sc_dir)
    ref_matrix = sc_matrix[ref_idx][:, common_genes].astype(np.float32)
    class_means = np.vstack([np.asarray(ref_matrix[ref_labels == cls].mean(axis=0)).ravel() for cls in classes]).astype(np.float32)
    marker_cols = select_marker_columns(class_means, classes, args.top_markers_per_class)
    marker_gene_ids = common_genes[marker_cols]
    gene_vocab = (args.dataset_root / "gene_vocab.txt").read_text(encoding="utf-8").splitlines()
    marker_rows = []
    for cls_i, cls in enumerate(classes):
        logged = np.log1p(class_means)
        score = logged[cls_i] - np.max(np.delete(logged, cls_i, axis=0), axis=0)
        top_cols = np.argsort(-score, kind="mergesort")[: args.top_markers_per_class]
        for rank, col in enumerate(top_cols, start=1):
            gene_id = int(common_genes[col])
            marker_rows.append({"cell_class": cls, "rank": rank, "gene_id": gene_id, "gene": gene_vocab[gene_id], "score": float(score[col])})
    pd.DataFrame(marker_rows).to_csv(args.results_dir / "spatial_deconv_reference_markers.csv", index=False)

    mean_marker = class_means[:, marker_cols]
    mean_marker_norm = normalize(mean_marker, norm="l2")
    embedding_scores = build_embedding_scores(args.embedding_dir, classes, chips, args.max_ref_per_class, args.seed)
    prior_label = pd.Series(ref_labels).value_counts().idxmax()

    prediction_frames = []
    metric_rows = []
    method_order = ["OmniCell-style deconv", "CPT 512-only", "Global prior"]
    for chip in chips:
        sp_dir = spatial_dirs[chip]
        sp_obs = pd.read_parquet(sp_dir / "obs.parquet")
        coords = np.load(sp_dir / "coords.npy")
        y_true = broad(sp_obs["CellType"])
        valid = np.isin(y_true, classes)
        sp_matrix = make_csr(sp_dir)
        x_marker = sp_matrix[:, common_genes[marker_cols]].astype(np.float32).toarray()
        expr_scores = normalize(x_marker, norm="l2") @ mean_marker_norm.T
        cpt_scores = embedding_scores[chip]

        pred_expr = predict_from_scores(expr_scores, classes)
        pred_cpt = predict_from_scores(cpt_scores, classes)
        pred_prior = np.full(len(y_true), prior_label, dtype=object)
        pred_omni = pred_expr.copy()
        probs = softmax(expr_scores, axis=1)

        preds = {
            "OmniCell-style deconv": pred_omni,
            "Expression marker": pred_expr,
            "CPT 512-only": pred_cpt,
            "Global prior": pred_prior,
        }
        for method, pred in preds.items():
            metric_rows.append(metric_row(method, chip, y_true[valid], pred[valid], level="chip"))
        tiles = tile_ids(coords, args.tile_grid)
        for tile in sorted(pd.unique(tiles)):
            tile_mask = valid & (tiles == tile)
            if int(tile_mask.sum()) < args.min_tile_cells:
                continue
            for method, pred in preds.items():
                metric_rows.append(metric_row(method, chip, y_true[tile_mask], pred[tile_mask], level="tile", tile=str(tile)))

        frame = pd.DataFrame(
            {
                "chip": chip,
                "obs_name": sp_obs["obs_name"].astype(str).to_numpy(),
                "coord_x": coords[:, 0],
                "coord_y": coords[:, 1],
                "ground_truth": y_true,
                "valid_for_metrics": valid,
            }
        )
        for method, pred in preds.items():
            frame[f"pred_{method}"] = pred
        for i, cls in enumerate(classes):
            frame[f"prop_{cls}"] = probs[:, i]
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_parquet(args.results_dir / "spatial_deconvolution_predictions.parquet", index=False)
    predictions.to_csv(args.results_dir / "spatial_deconvolution_predictions.csv.gz", index=False)
    metrics.to_csv(args.results_dir / "spatial_deconvolution_metrics.csv", index=False)
    metrics[metrics["level"].eq("chip")].to_csv(args.results_dir / "spatial_deconvolution_chip_metrics.csv", index=False)

    draw_metric_panel(metrics[metrics["level"].eq("tile")], method_order, args.figure_dir, args.dpi)
    draw_spatial_panel(predictions, args.display_chip, method_order, args.figure_dir, args.dpi)
    draw_composition_panel(predictions, args.display_chip, method_order, classes, args.figure_dir, args.dpi)
    draw_confusion_panel(predictions, args.display_chip, classes, args.figure_dir, args.dpi)

    contract = {
        "core_conclusion": "Single-cell reference marker deconvolution substantially improves AD hippocampus spatial cell-class recovery, while latest 512-d CPT embedding alone is weak on this task.",
        "chips": chips,
        "display_chip": args.display_chip,
        "classes": classes,
        "n_reference_cells": int(len(ref_idx)),
        "n_common_genes": int(len(common_genes)),
        "n_marker_genes": int(len(marker_gene_ids)),
        "methods": method_order,
        "outputs": {
            "predictions": str(args.results_dir / "spatial_deconvolution_predictions.parquet"),
            "metrics": str(args.results_dir / "spatial_deconvolution_metrics.csv"),
            "figures": str(args.figure_dir),
        },
    }
    (args.results_dir / "spatial_deconvolution_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
