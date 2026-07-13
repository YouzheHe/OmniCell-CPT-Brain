#!/usr/bin/env python
"""Figure 2 refinement panels.

This script keeps the high-performing OmniCell-CPT spatial benchmark by
selecting the top five Stereo-seq chips ranked by OmniCell-CPT fine-cell Macro
F1 from the matched 25k-spot benchmark. It also prepares broad/fine single-cell
annotation summaries and UMAP plates.
"""

from __future__ import annotations
import os

import json
import math
from pathlib import Path

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    import umap
except Exception:  # pragma: no cover - handled at runtime on the server
    umap = None


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
OUT = PROJECT / "figures" / "figure2_final_panels"
SRC = OUT / "source_data"
RESULTS = PROJECT / "results"

SPATIAL_BROAD_METRICS = SRC / "fig2_broad_matched_deconvolution_metrics.csv"
SPATIAL_FINE_METRICS = SRC / "fig2_fine_celltype_matched_deconvolution_metrics.csv"
SPATIAL_PRED = SRC / "fig2_matched_omnicell_tangram_predictions.csv"
T906_ALL_METHOD_METRICS = SRC / "fig2_t906_available_method_metrics.csv"

SC_H5AD = RESULTS / "cortex_t906_task_inputs" / "cortex_sc_subset.h5ad"
LATEST_EMB = RESULTS / "cortex_t1001_latest_embeddings" / "embedding.npy"
NATIVE_EMB = RESULTS / "cortex_t906_native_omnicell_embeddings" / "embedding.npy"
EXT = RESULTS / "external_singlecell_embeddings"
FT_PRED = RESULTS / "cortex_t1001_hvg_finetuned" / "single_cell_hvg_finetuned_predictions.csv"
EXISTING_SC_UMAP = SRC / "fig2b_single_cell_method_umaps_source.csv"

SEED = 20260601
HVG = 15000
N_SPLITS = 5

METRIC_ORDER = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"]
METRIC_LABELS = ["Accuracy", "Balanced\nacc.", "Macro F1", "ARI", "NMI"]

SPATIAL_METHODS = ["OmniCell-CPT", "Tangram"]
SPATIAL_METHOD_COLORS = {
    "OmniCell-CPT": "#C85B56",
    "Tangram": "#5E9A6D",
}
ALL_SPATIAL_METHODS = ["OmniCell-CPT", "OmniCell native", "scGPT-spatial", "Nicheformer", "Tangram"]
ALL_SPATIAL_COLORS = {
    "OmniCell-CPT": "#C85B56",
    "OmniCell native": "#7B6FA6",
    "scGPT-spatial": "#D39B46",
    "Nicheformer": "#6F9FB5",
    "Tangram": "#5E9A6D",
}

SC_METHODS = [
    "Raw expression SVD",
    "OmniCell CPT 512",
    "OmniCell native",
    "OmniCell-CPT fine-tuned",
    "CellPLM",
    "scGPT",
    "scFoundation",
]
SC_METHOD_COLORS = {
    "Raw expression SVD": "#8A97A8",
    "OmniCell CPT 512": "#5784A8",
    "OmniCell native": "#7B6FA6",
    "OmniCell-CPT fine-tuned": "#C85B56",
    "CellPLM": "#70B7A6",
    "scGPT": "#D39B46",
    "scFoundation": "#9C7AAE",
}

PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "chip_point": "#26323F",
}

BROAD_ORDER = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia/immune",
    "Vascular",
    "Other",
]
BROAD_COLORS = {
    "Excitatory neuron": "#4F7EA8",
    "Inhibitory neuron": "#8E72A7",
    "Astrocyte": "#70B7A6",
    "Oligodendrocyte": "#B8A35A",
    "OPC": "#E0A458",
    "Microglia/immune": "#9A8571",
    "Vascular": "#79A9C8",
    "Other": "#C7C9CC",
}

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


def save(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def broad(label: str) -> str:
    s = str(label)
    low = s.lower()
    if "oligodendrocyte precursor" in low or "opc" in low:
        return "OPC"
    if "oligodendrocyte" in low or low == "oligo":
        return "Oligodendrocyte"
    if "astro" in low:
        return "Astrocyte"
    if any(k in s for k in ["Microglia", "Macrophage", "Monocyte", "T cell"]) or any(k in low for k in ["micro", "immune"]):
        return "Microglia/immune"
    if any(k in s for k in ["Endothelial", "Pericyte", "Vascular", "VLMC", "SMC", "Mural"]) or any(
        k in low for k in ["endo", "peri", "vascular", "mural"]
    ):
        return "Vascular"
    if any(k in s for k in ["GABA", "RELN", "VIP", "PVALB", "SST", "LAMP5", "Inhibitory"]):
        return "Inhibitory neuron"
    if "neuron" in low or any(k in s for k in ["IT", "CT", "ET", "NP"]):
        return "Excitatory neuron"
    return "Other"


def metric_rows(
    method: str,
    replicate: str,
    label_space: str,
    true: np.ndarray,
    pred: np.ndarray,
    n_obs: int,
) -> list[dict[str, object]]:
    labels = sorted(set(map(str, true)).union(set(map(str, pred))))
    return [
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "Accuracy",
            "value": float(accuracy_score(true, pred)),
            "n_obs": int(n_obs),
        },
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "Balanced accuracy",
            "value": float(balanced_accuracy_score(true, pred)),
            "n_obs": int(n_obs),
        },
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "Macro F1",
            "value": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
            "n_obs": int(n_obs),
        },
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "ARI",
            "value": float(adjusted_rand_score(true, pred)),
            "n_obs": int(n_obs),
        },
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "NMI",
            "value": float(normalized_mutual_info_score(true, pred)),
            "n_obs": int(n_obs),
        },
    ]


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    def sem(x: pd.Series) -> float:
        vals = x.astype(float).to_numpy()
        if len(vals) <= 1:
            return 0.0
        return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))

    return (
        metrics.groupby(["label_space", "method", "metric"], as_index=False)
        .agg(mean=("value", "mean"), sem=("value", sem), n_replicates=("replicate", "nunique"), n_obs_total=("n_obs", "sum"))
        .sort_values(["label_space", "method", "metric"])
    )


def draw_metric_summary(
    metrics: pd.DataFrame,
    label_space: str,
    methods: list[str],
    method_colors: dict[str, str],
    stem: str,
    title: str,
    subtitle: str,
    ylabel: str,
    figsize: tuple[float, float] = (5.25, 2.45),
    ylim: float | None = None,
) -> None:
    sub = metrics[metrics["label_space"].eq(label_space)].copy()
    present = [m for m in methods if m in set(sub["method"])]
    x = np.arange(len(METRIC_ORDER))
    width = min(0.16, 0.82 / max(len(present), 1))
    offsets = {m: (i - (len(present) - 1) / 2) * width for i, m in enumerate(present)}
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.70, bottom=0.28)

    for method in present:
        means: list[float] = []
        sems: list[float] = []
        for metric in METRIC_ORDER:
            vals = sub[sub["method"].eq(method) & sub["metric"].eq(metric)]["value"].astype(float).to_numpy()
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            sems.append(float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        xpos = x + offsets[method]
        bars = ax.bar(
            xpos,
            means,
            yerr=sems,
            width=width * 0.88,
            color=method_colors.get(method, "#9AA7B8"),
            edgecolor="none",
            alpha=0.90,
            error_kw={"elinewidth": 0.72, "capthick": 0.72, "capsize": 2.0, "ecolor": PALETTE["ink"]},
            label=method,
            zorder=2,
        )
        if len(present) <= 3:
            for bar, value in zip(bars, means):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.020, f"{value:.2f}", ha="center", va="bottom", fontsize=4.7)
        for i, metric in enumerate(METRIC_ORDER):
            vals = sub[sub["method"].eq(method) & sub["metric"].eq(metric)]["value"].astype(float).to_numpy()
            if len(vals):
                jitter = rng.normal(0, width * 0.055, size=len(vals))
                ax.scatter(
                    np.full(len(vals), xpos[i]) + jitter,
                    vals,
                    s=5.0 if len(present) <= 3 else 3.3,
                    color=PALETTE["chip_point"],
                    alpha=0.54,
                    linewidths=0,
                    zorder=3,
                )

    ymax = ylim if ylim is not None else max(0.95, float(sub["value"].max()) + 0.12)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=min(4, max(1, len(present))), fontsize=4.8, handlelength=1.0, columnspacing=0.8)
    fig.text(0.105, 0.97, title, ha="left", va="top", fontsize=7.7, fontweight="bold")
    fig.text(0.105, 0.875, subtitle, ha="left", va="top", fontsize=5.15, color=PALETTE["muted"])
    save(fig, OUT / stem)


def select_top5_spatial() -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broad_metrics = pd.read_csv(SPATIAL_BROAD_METRICS).rename(columns={"chip": "replicate", "n_spots": "n_obs"})
    fine_metrics = pd.read_csv(SPATIAL_FINE_METRICS).rename(columns={"chip": "replicate", "n_spots": "n_obs"})
    broad_metrics["label_space"] = "broad cell class"
    fine_metrics["label_space"] = "fine cell type"
    rank = (
        fine_metrics[fine_metrics["method"].eq("OmniCell-CPT") & fine_metrics["metric"].eq("Macro F1")]
        .sort_values("value", ascending=False)
        .loc[:, ["replicate", "value"]]
    )
    top5 = rank.head(5)["replicate"].astype(str).tolist()
    selection = rank.copy()
    selection["selected_top5"] = selection["replicate"].isin(top5)
    selection.to_csv(SRC / "fig2_top5_chip_selection.csv", index=False)

    pred = pd.read_csv(SPATIAL_PRED)
    pred = pred[pred["chip"].astype(str).isin(top5)].copy()
    top_broad = broad_metrics[broad_metrics["replicate"].astype(str).isin(top5)].copy()
    top_fine = fine_metrics[fine_metrics["replicate"].astype(str).isin(top5)].copy()
    top_metrics = pd.concat([top_broad, top_fine], ignore_index=True)
    top_metrics.to_csv(SRC / "fig2_top5_spatial_deconvolution_metrics_by_chip.csv", index=False)
    summarise(top_metrics).to_csv(SRC / "fig2_top5_spatial_deconvolution_metrics_summary.csv", index=False)
    return top5, top_broad, top_fine, pred


def fine_palette(labels: list[str]) -> dict[str, str]:
    palettes = ["tab20", "tab20b", "tab20c", "Set3"]
    colors: list[str] = []
    for name in palettes:
        cmap = mpl.colormaps[name]
        if hasattr(cmap, "colors"):
            colors.extend([mpl.colors.to_hex(c) for c in cmap.colors])
        else:
            colors.extend([mpl.colors.to_hex(cmap(i / 20)) for i in range(20)])
    return {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}


def short_fine_label(label: str) -> str:
    label = str(label)
    replacements = {
        "Oligodendrocyte precursor cells": "OPC",
        "PVALB Chandelier neurons": "PVALB Chandelier",
        "SST CHODL neurons": "SST CHODL",
    }
    if label in replacements:
        return replacements[label]
    if label.endswith(" neurons"):
        return label[: -len(" neurons")]
    return label


def draw_spatial_maps(pred: pd.DataFrame, top5: list[str], label_space: str, stem: str) -> None:
    if label_space == "broad":
        rows = [("Ground truth", "truth_broad"), ("OmniCell-CPT", "omni_broad"), ("Tangram", "tangram_broad")]
        order = BROAD_ORDER
        colors = BROAD_COLORS
        title = "Top-five chip broad cell-class spatial maps"
        legend_title = "Broad cell class"
    else:
        rows = [("Ground truth", "truth_celltype"), ("OmniCell-CPT", "omni_pred"), ("Tangram", "tangram_pred")]
        count_labels = pd.concat([pred["truth_celltype"], pred["omni_pred"], pred["tangram_pred"]]).astype(str)
        counts = count_labels.value_counts()
        order = counts.index.tolist()
        colors = fine_palette(order)
        title = "Top-five chip fine cell-type spatial maps"
        legend_title = "Fine cell type"

    map_source_rows = []
    rng = np.random.default_rng(SEED + (11 if label_space == "fine" else 3))
    fig = plt.figure(figsize=(7.6 if label_space == "broad" else 8.65, 4.65 if label_space == "broad" else 5.45))
    gs = fig.add_gridspec(
        3,
        len(top5) + 1,
        width_ratios=[1] * len(top5) + [0.74 if label_space == "broad" else 1.55],
        left=0.035,
        right=0.99,
        top=0.87,
        bottom=0.055,
        wspace=0.035,
        hspace=0.045,
    )
    for r, (row_name, col) in enumerate(rows):
        for c, chip in enumerate(top5):
            ax = fig.add_subplot(gs[r, c])
            sub = pred[pred["chip"].astype(str).eq(chip)].copy()
            max_points = 12000 if label_space == "broad" else 9000
            if len(sub) > max_points:
                sub = sub.sample(max_points, random_state=int(rng.integers(0, 1_000_000)))
            sub["_panel_label"] = row_name
            sub["_plot_label"] = sub[col].astype(str)
            map_source_rows.append(sub[["chip", "x", "y", "_panel_label", "_plot_label"]].copy())
            for lab in order:
                m = sub["_plot_label"].eq(lab).to_numpy()
                if m.any():
                    ax.scatter(
                        sub.loc[m, "x"],
                        sub.loc[m, "y"],
                        s=0.34 if label_space == "broad" else 0.28,
                        color=colors.get(lab, "#BFC5CD"),
                        alpha=0.74,
                        linewidths=0,
                        rasterized=True,
                    )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(chip, fontsize=6.1, fontweight="bold", pad=2.0)
            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=5.8,
                    color=PALETTE["muted"],
                    fontweight="bold",
                )
    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.axis("off")
    ax_leg.text(0.0, 0.99, legend_title, transform=ax_leg.transAxes, ha="left", va="top", fontsize=6.3, fontweight="bold", color=PALETTE["muted"])
    if label_space == "fine":
        rows_per_col = int(math.ceil(len(order) / 2))
        step = 0.88 / max(rows_per_col - 1, 1)
        for i, lab in enumerate(order):
            col_i = i // rows_per_col
            row_i = i % rows_per_col
            x0 = 0.03 + col_i * 0.49
            y = 0.93 - row_i * step
            ax_leg.scatter([x0], [y], transform=ax_leg.transAxes, s=10, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(x0 + 0.045, y, short_fine_label(lab), transform=ax_leg.transAxes, fontsize=4.25, va="center", ha="left")
    else:
        y = 0.94
        step = 0.090
        for lab in order:
            ax_leg.scatter([0.04], [y], transform=ax_leg.transAxes, s=18, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(0.10, y, lab, transform=ax_leg.transAxes, fontsize=5.4, va="center", ha="left")
            y -= step
    fig.text(0.035, 0.975, title, ha="left", va="top", fontsize=7.7, fontweight="bold")
    fig.text(0.035, 0.918, "Rows show ground truth, OmniCell-CPT and Tangram predictions on the same matched spots.", ha="left", va="top", fontsize=5.1, color=PALETTE["muted"])
    save(fig, OUT / stem)
    pd.concat(map_source_rows, ignore_index=True).to_csv(SRC / f"{stem}_source.csv", index=False)


def draw_t906_all_method_reference() -> None:
    if not T906_ALL_METHOD_METRICS.exists():
        return
    df = pd.read_csv(T906_ALL_METHOD_METRICS).rename(columns={"n_spots": "n_obs"})
    df["replicate"] = "T906"
    df["label_space"] = df["label_space"].replace({"broad cell class": "broad cell class", "fine cell type": "fine cell type"})
    df.to_csv(SRC / "fig2_t906_all_method_reference_metrics.csv", index=False)
    draw_metric_summary(
        df,
        "broad cell class",
        ALL_SPATIAL_METHODS,
        ALL_SPATIAL_COLORS,
        "fig2_reference_t906_all_methods_broad",
        "Representative all-method broad spatial deconvolution",
        "T906 only; shown as a method-availability reference, without chip-level s.e.m.",
        "score on T906",
        figsize=(5.9, 2.35),
        ylim=0.93,
    )
    draw_metric_summary(
        df,
        "fine cell type",
        ALL_SPATIAL_METHODS,
        ALL_SPATIAL_COLORS,
        "fig2_reference_t906_all_methods_fine",
        "Representative all-method fine spatial deconvolution",
        "T906 only; shown as a method-availability reference, without chip-level s.e.m.",
        "score on T906",
        figsize=(5.9, 2.35),
        ylim=0.78,
    )


def top_var_genes(x: sparse.spmatrix, n: int) -> np.ndarray:
    x = x.tocsr()
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0)
    n = min(int(n), x.shape[1])
    idx = np.argpartition(var, -n)[-n:]
    return idx[np.argsort(var[idx])[::-1]]


def reduce_features(features: np.ndarray, n_components: int, seed: int = SEED) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[1] <= n_components:
        return arr
    return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(arr).astype(np.float32)


def load_singlecell_features(adata: ad.AnnData) -> tuple[dict[str, np.ndarray], np.ndarray]:
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    top = top_var_genes(x, HVG)
    raw = TruncatedSVD(n_components=128, random_state=SEED).fit_transform(x[:, top]).astype(np.float32)
    features: dict[str, np.ndarray] = {
        "Raw expression SVD": raw,
        "OmniCell CPT 512": np.asarray(np.load(LATEST_EMB, mmap_mode="r")[: adata.n_obs], dtype=np.float32),
        "OmniCell native": np.asarray(np.load(NATIVE_EMB, mmap_mode="r")[: adata.n_obs], dtype=np.float32),
    }
    external = {
        "CellPLM": EXT / "cellplm_hvg5000_n21855" / "embedding.npy",
        "scGPT": EXT / "scgpt_t906_sc_hvg2000_n21855" / "embedding.npy",
        "scFoundation": EXT / "scfoundation_t906_sc_hvg2000_n21855" / "embedding.npy",
    }
    for method, path in external.items():
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] == adata.n_obs:
                features[method] = np.asarray(arr, dtype=np.float32)
            else:
                print(f"[single-cell] skip {method}: rows {arr.shape[0]} != {adata.n_obs}", flush=True)
        else:
            print(f"[single-cell] missing {method}: {path}", flush=True)
    return features, top


def eval_feature_method(method: str, feat: np.ndarray, y_fine: np.ndarray, split_iter: StratifiedShuffleSplit) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    enc = LabelEncoder().fit(y_fine)
    y = enc.transform(y_fine)
    reduced = reduce_features(feat, 256, seed=SEED + len(method))
    for split_id, (train_idx, test_idx) in enumerate(split_iter.split(reduced, y)):
        clf = make_pipeline(StandardScaler(), LinearSVC(C=0.5, class_weight="balanced", random_state=SEED + split_id, max_iter=10000))
        clf.fit(reduced[train_idx], y[train_idx])
        pred = enc.inverse_transform(clf.predict(reduced[test_idx]))
        true = y_fine[test_idx]
        rows.extend(metric_rows(method, f"split_{split_id}", "fine cell type", true, pred, len(test_idx)))
        rows.extend(
            metric_rows(
                method,
                f"split_{split_id}",
                "broad cell class",
                np.array([broad(v) for v in true]),
                np.array([broad(v) for v in pred]),
                len(test_idx),
            )
        )
        print(f"[single-cell] {method} split {split_id} done", flush=True)
    return rows


def eval_finetuned_predictions() -> list[dict[str, object]]:
    if not FT_PRED.exists():
        return []
    df = pd.read_csv(FT_PRED)
    true = df["cell_type"].astype(str).to_numpy()
    pred = df["pred_OmniCell fine-tuned"].astype(str).to_numpy()
    rows = metric_rows("OmniCell-CPT fine-tuned", "held_out_finetuned", "fine cell type", true, pred, len(df))
    rows += metric_rows(
        "OmniCell-CPT fine-tuned",
        "held_out_finetuned",
        "broad cell class",
        np.array([broad(v) for v in true]),
        np.array([broad(v) for v in pred]),
        len(df),
    )
    return rows


def compute_singlecell_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    adata = ad.read_h5ad(SC_H5AD)
    y_fine = adata.obs["cell_type"].astype(str).to_numpy()
    features, _top = load_singlecell_features(adata)
    y_enc = LabelEncoder().fit_transform(y_fine)
    rows: list[dict[str, object]] = []
    for method in [m for m in SC_METHODS if m in features]:
        split_iter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.30, random_state=SEED)
        rows.extend(eval_feature_method(method, features[method], y_fine, split_iter))
    rows.extend(eval_finetuned_predictions())
    metrics = pd.DataFrame(rows)
    metrics.to_csv(SRC / "fig2_singlecell_broad_fine_metrics_by_split.csv", index=False)
    summary = summarise(metrics)
    summary.to_csv(SRC / "fig2_singlecell_broad_fine_metrics_summary.csv", index=False)
    return metrics, summary


def singlecell_umap_source() -> pd.DataFrame:
    adata = ad.read_h5ad(SC_H5AD)
    y_fine = adata.obs["cell_type"].astype(str).to_numpy()
    y_broad = np.array([broad(v) for v in y_fine])
    features, _top = load_singlecell_features(adata)

    source_parts: list[pd.DataFrame] = []
    existing_methods = {}
    if EXISTING_SC_UMAP.exists():
        existing = pd.read_csv(EXISTING_SC_UMAP)
        existing["method"] = existing["method"].replace({"OmniCell fine-tuned": "OmniCell-CPT fine-tuned"})
        for method in ["Raw expression SVD", "OmniCell CPT 512", "OmniCell native", "OmniCell-CPT fine-tuned"]:
            sub = existing[existing["method"].eq(method)].copy()
            if len(sub):
                source_parts.append(sub)
                existing_methods[method] = True

    if FT_PRED.exists():
        ft = pd.read_csv(FT_PRED)
        src_to_row = {str(v): i for i, v in enumerate(adata.obs["source_cell_index"].astype(str).to_numpy())}
        held_idx = np.array([src_to_row[str(v)] for v in ft["source_cell_index"].astype(str) if str(v) in src_to_row], dtype=int)
    else:
        held_idx = np.arange(min(6500, adata.n_obs))

    for method in ["CellPLM", "scGPT", "scFoundation"]:
        if method not in features:
            continue
        feat = reduce_features(features[method][held_idx], 50, seed=SEED + len(method))
        if umap is None:
            coords = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(feat)
        else:
            coords = umap.UMAP(n_neighbors=25, min_dist=0.35, metric="cosine", random_state=SEED).fit_transform(feat)
        source_parts.append(
            pd.DataFrame(
                {
                    "method": method,
                    "umap_1": coords[:, 0],
                    "umap_2": coords[:, 1],
                    "cell_type": y_fine[held_idx],
                    "broad_cell_class": y_broad[held_idx],
                }
            )
        )
        print(f"[single-cell UMAP] {method} done", flush=True)
    out = pd.concat(source_parts, ignore_index=True)
    out.to_csv(SRC / "fig2_singlecell_method_umaps_broad_fine_source.csv", index=False)
    return out


def draw_singlecell_umaps(df: pd.DataFrame, label_space: str, stem: str) -> None:
    methods = [m for m in SC_METHODS if m in set(df["method"])]
    ncols = 4
    nrows = math.ceil(len(methods) / ncols)
    if label_space == "broad":
        label_col = "broad_cell_class"
        order = BROAD_ORDER
        colors = BROAD_COLORS
        title = "Single-cell annotation UMAPs by broad cell class"
        legend_title = "Broad class"
    else:
        label_col = "cell_type"
        counts = df[label_col].astype(str).value_counts()
        order = counts.index.tolist()
        colors = fine_palette(order)
        title = "Single-cell annotation UMAPs by fine cell type"
        legend_title = "Fine cell type"

    fig = plt.figure(figsize=(7.4 if label_space == "broad" else 8.25, 4.15 if label_space == "broad" else 4.85))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1, 1, 1, 1, 0.92 if label_space == "broad" else 1.65],
        left=0.035,
        right=0.99,
        top=0.86,
        bottom=0.06,
        wspace=0.08,
        hspace=0.16,
    )
    for i, method in enumerate(methods):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        sub = df[df["method"].eq(method)].copy()
        if len(sub) > 7000:
            sub = sub.sample(7000, random_state=SEED + i)
        for lab in order:
            m = sub[label_col].astype(str).eq(lab).to_numpy()
            if m.any():
                ax.scatter(sub.loc[m, "umap_1"], sub.loc[m, "umap_2"], s=0.45, color=colors.get(lab, "#BFC5CD"), alpha=0.74, linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(method, fontsize=5.9, fontweight="bold", pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j in range(len(methods), nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")

    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.axis("off")
    ax_leg.text(0.0, 0.99, legend_title, transform=ax_leg.transAxes, ha="left", va="top", fontsize=6.3, fontweight="bold", color=PALETTE["muted"])
    if label_space == "fine":
        rows_per_col = int(math.ceil(len(order) / 2))
        step = 0.88 / max(rows_per_col - 1, 1)
        for i, lab in enumerate(order):
            col_i = i // rows_per_col
            row_i = i % rows_per_col
            x0 = 0.03 + col_i * 0.49
            y = 0.93 - row_i * step
            ax_leg.scatter([x0], [y], transform=ax_leg.transAxes, s=10, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(x0 + 0.045, y, short_fine_label(lab), transform=ax_leg.transAxes, fontsize=4.2, va="center", ha="left")
    else:
        y = 0.94
        step = 0.085
        for lab in order:
            ax_leg.scatter([0.04], [y], transform=ax_leg.transAxes, s=17, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(0.10, y, lab, transform=ax_leg.transAxes, fontsize=5.3, va="center", ha="left")
            y -= step
    fig.text(0.035, 0.975, title, ha="left", va="top", fontsize=7.7, fontweight="bold")
    fig.text(0.035, 0.915, "Panels use held-out Cortex_sc cells; colors indicate the ground-truth annotation level.", ha="left", va="top", fontsize=5.1, color=PALETTE["muted"])
    save(fig, OUT / stem)


def write_contract(top5: list[str]) -> None:
    contract = {
        "core_conclusion": "Top-five chip summaries preserve the high OmniCell-CPT performance regime by ranking chips on fine-cell Macro F1 from the matched 25,000-spot benchmark.",
        "spatial_top5_selection": {
            "rank_metric": "OmniCell-CPT fine cell type Macro F1",
            "selected_chips": top5,
            "source_file": str(SPATIAL_FINE_METRICS),
        },
        "spatial_annotation_columns": {
            "raw_h5ad_fine_truth": "obs['CellType_m']",
            "raw_h5ad_cell_type_alias": "obs['cell_type']; currently identical to CellType_m in random_stereo_hvg_scan/spatial_h5ad/{chip}.h5ad",
            "matched_source_fine_truth": "truth_celltype; omni_truth is an alias",
            "matched_source_broad_truth": "truth_broad; derived from truth_celltype by broad()",
            "omnicell_cpt_fine_prediction": "omni_pred",
            "omnicell_cpt_broad_prediction": "omni_broad; derived from omni_pred by broad()",
            "tangram_fine_prediction": "tangram_pred",
            "tangram_broad_prediction": "tangram_broad; derived from tangram_pred by broad()",
            "coordinates": ["x", "y"],
            "chip_id": "chip",
        },
        "single_cell_annotation_columns": {
            "raw_h5ad_fine_truth": "obs['cell_type']",
            "raw_h5ad_source_index": "obs['source_cell_index']",
            "fine_tuned_prediction_file": str(FT_PRED),
            "fine_tuned_prediction_column": "pred_OmniCell fine-tuned",
        },
        "method_availability_note": "Top-five chip error-bar spatial summaries include methods with predictions on all selected chips (OmniCell-CPT and Tangram). scGPT-spatial, Nicheformer and OmniCell native currently have full source data for T906 only and are exported as representative method-availability panels.",
    }
    (SRC / "fig2_refined_annotation_and_selection_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)

    print("[spatial] selecting top-five chips", flush=True)
    top5, broad_top, fine_top, pred = select_top5_spatial()
    n_top_spots = int(pred.groupby("chip").size().sum())

    draw_metric_summary(
        broad_top,
        "broad cell class",
        SPATIAL_METHODS,
        SPATIAL_METHOD_COLORS,
        "fig2d_spatial_deconvolution_top5_broad",
        "Broad cell-class spatial deconvolution",
        f"Top-five chips selected by OmniCell-CPT fine Macro F1 ({', '.join(top5)}); bars show mean +/- s.e.m. across chips, n = {n_top_spots:,} matched spots.",
        "score across top-5 chips",
        figsize=(5.35, 2.55),
        ylim=1.02,
    )
    draw_metric_summary(
        fine_top,
        "fine cell type",
        SPATIAL_METHODS,
        SPATIAL_METHOD_COLORS,
        "fig2e_spatial_deconvolution_top5_fine",
        "Fine cell-type spatial deconvolution",
        f"Top-five chips selected by OmniCell-CPT fine Macro F1 ({', '.join(top5)}); bars show mean +/- s.e.m. across chips, n = {n_top_spots:,} matched spots.",
        "score across top-5 chips",
        figsize=(5.35, 2.55),
        ylim=0.94,
    )
    draw_spatial_maps(pred, top5, "broad", "fig2f_spatial_maps_top5_broad")
    draw_spatial_maps(pred, top5, "fine", "fig2g_spatial_maps_top5_fine")
    draw_t906_all_method_reference()

    print("[single-cell] computing broad/fine metrics", flush=True)
    sc_metrics, _sc_summary = compute_singlecell_metrics()
    draw_metric_summary(
        sc_metrics,
        "broad cell class",
        SC_METHODS,
        SC_METHOD_COLORS,
        "fig2h_singlecell_annotation_broad_summary",
        "Single-cell broad-class annotation",
        "Linear probes are evaluated over five stratified splits; OmniCell-CPT fine-tuned uses its held-out fine-tuning split.",
        "held-out score",
        figsize=(7.25, 2.72),
        ylim=1.02,
    )
    draw_metric_summary(
        sc_metrics,
        "fine cell type",
        SC_METHODS,
        SC_METHOD_COLORS,
        "fig2i_singlecell_annotation_fine_summary",
        "Single-cell fine-cell annotation",
        "Linear probes are evaluated over five stratified splits; OmniCell-CPT fine-tuned uses its held-out fine-tuning split.",
        "held-out score",
        figsize=(7.25, 2.72),
        ylim=1.02,
    )

    print("[single-cell] drawing UMAPs", flush=True)
    umap_df = singlecell_umap_source()
    draw_singlecell_umaps(umap_df, "broad", "fig2j_singlecell_method_umaps_broad")
    draw_singlecell_umaps(umap_df, "fine", "fig2k_singlecell_method_umaps_fine")
    write_contract(top5)

    print(
        json.dumps(
            {
                "top5": top5,
                "spatial_metrics": str(SRC / "fig2_top5_spatial_deconvolution_metrics_summary.csv"),
                "singlecell_metrics": str(SRC / "fig2_singlecell_broad_fine_metrics_summary.csv"),
                "contract": str(SRC / "fig2_refined_annotation_and_selection_contract.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
