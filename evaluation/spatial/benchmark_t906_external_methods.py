#!/usr/bin/env python
"""T906 15k-HVG spatial benchmark against OmniCell native and external models."""

from __future__ import annotations
import os

import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
INPUT = PROJECT / "results" / "cortex_t906_task_inputs"
FINE_DIR = PROJECT / "figures" / "figure2_t906_hvg15000_benchmark"
NATIVE = PROJECT / "results" / "cortex_t906_native_omnicell_embeddings" / "embedding.npy"
CELLPLM_SC = PROJECT / "results" / "external_singlecell_embeddings" / "cellplm_hvg5000_n21855" / "embedding.npy"
CELLPLM_SP = PROJECT / "results" / "external_singlecell_embeddings" / "cellplm_t906_hvg5000_n61147" / "embedding.npy"
SCGPT_SPATIAL_SC = PROJECT / "results" / "external_singlecell_embeddings" / "scgpt_spatial_t906_sc_hvg2000_n21855_b64" / "embedding.npy"
SCGPT_SPATIAL_SP = PROJECT / "results" / "external_singlecell_embeddings" / "scgpt_spatial_t906_hvg2000_n61147_b64" / "embedding.npy"
OUT = PROJECT / "figures" / "figure2_t906_external_method_comparison"

PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "raw": "#8A97A8",
    "native": "#7C6AA6",
    "fine": "#C86054",
    "cellplm": "#70B7A6",
    "scgpt_spatial": "#D09B45",
}
METHOD_COLORS = {
    "Raw expression SVD transfer": PALETTE["raw"],
    "OmniCell native transfer": PALETTE["native"],
    "OmniCell-CPT fine-tuned": PALETTE["fine"],
    "CellPLM transfer": PALETTE["cellplm"],
    "scGPT-spatial transfer": PALETTE["scgpt_spatial"],
}
BROAD_COLORS = {
    "Excitatory neuron": "#4F7EA8",
    "Inhibitory neuron": "#8E72A7",
    "Astrocyte": "#70B7A6",
    "Oligodendrocyte": "#B8A35A",
    "OPC": "#E0A458",
    "Microglia": "#9A8571",
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


def broad(label: str) -> str:
    s = str(label)
    if "Oligodendrocyte precursor" in s:
        return "OPC"
    if "Oligodendrocyte" in s:
        return "Oligodendrocyte"
    if "Astro" in s:
        return "Astrocyte"
    if any(k in s for k in ["Microglia", "Macrophage", "Monocyte", "T cell"]):
        return "Microglia"
    if any(k in s for k in ["Endothelial", "Pericyte", "Vascular", "VLMC", "SMC"]):
        return "Vascular"
    if any(k in s for k in ["GABA", "RELN", "VIP", "PVALB", "SST", "LAMP5"]):
        return "Inhibitory neuron"
    if "neuron" in s or "IT" in s or "CT" in s or "ET" in s or "NP" in s:
        return "Excitatory neuron"
    return "Other"


def top_var_genes(x: sparse.csr_matrix, rows: np.ndarray, n: int) -> np.ndarray:
    xt = x[rows]
    mean = np.asarray(xt.mean(axis=0)).ravel()
    mean_sq = np.asarray(xt.multiply(xt).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0)
    idx = np.argpartition(var, -min(n, x.shape[1]))[-min(n, x.shape[1]) :]
    return idx[np.argsort(var[idx])[::-1]]


def metric_rows(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    task = "T906 spatial deconvolution"
    return [
        {"task": task, "method": method, "metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "Balanced accuracy", "value": balanced_accuracy_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "Macro F1", "value": f1_score(y_true, y_pred, average="macro", zero_division=0)},
        {"task": task, "method": method, "metric": "ARI", "value": adjusted_rand_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "NMI", "value": normalized_mutual_info_score(y_true, y_pred)},
    ]


def save(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def split_indices(scad: ad.AnnData, spad: ad.AnnData, seed: int = 20260528) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
    labels_text = np.r_[scad.obs["cell_type"].astype(str).to_numpy(), spad.obs["cell_type"].astype(str).to_numpy()]
    enc = LabelEncoder().fit(labels_text)
    y = enc.transform(labels_text)
    sc_idx = np.arange(scad.n_obs, dtype=np.int64)
    sp_idx = np.arange(scad.n_obs, scad.n_obs + spad.n_obs, dtype=np.int64)
    sp_y = y[sp_idx]
    cal_local, held_local = train_test_split(
        np.arange(spad.n_obs),
        test_size=0.50,
        random_state=seed + 15000 + sum(ord(c) for c in "T906"),
        stratify=sp_y,
    )
    return sc_idx, sp_idx, cal_local, held_local, enc, y


def linear_transfer(name: str, features: np.ndarray, sc_idx: np.ndarray, sp_idx: np.ndarray, held_local: np.ndarray, y: np.ndarray, enc: LabelEncoder) -> tuple[list[dict], np.ndarray]:
    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.5, class_weight="balanced", random_state=20260528, max_iter=10000),
    )
    clf.fit(features[sc_idx], y[sc_idx])
    held_idx = sp_idx[held_local]
    pred_held = clf.predict(features[held_idx])
    pred_all = clf.predict(features[sp_idx])
    return metric_rows(name, y[held_idx], pred_held), enc.inverse_transform(pred_all)


def raw_svd_transfer(scad: ad.AnnData, spad: ad.AnnData, sc_idx: np.ndarray, sp_idx: np.ndarray, held_local: np.ndarray, y: np.ndarray, enc: LabelEncoder) -> tuple[list[dict], np.ndarray]:
    x = sparse.vstack([scad.X.tocsr(), spad.X.tocsr()], format="csr")
    top = top_var_genes(x, sc_idx, 15000)
    svd = TruncatedSVD(n_components=128, random_state=20260528)
    features = svd.fit_transform(x[:, top]).astype(np.float32)
    return linear_transfer("Raw expression SVD transfer", features, sc_idx, sp_idx, held_local, y, enc)


def plot_metrics(metrics: pd.DataFrame) -> None:
    show = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"]
    methods = list(dict.fromkeys(metrics["method"].tolist()))
    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.78, bottom=0.34)
    x = np.arange(len(show))
    width = min(0.16, 0.82 / len(methods))
    for i, method in enumerate(methods):
        vals = []
        for metric in show:
            v = metrics.loc[metrics["method"].eq(method) & metrics["metric"].eq(metric), "value"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width * 0.92, color=METHOD_COLORS.get(method, "#999"), label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(show)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45, alpha=0.65)
    fig.text(0.075, 0.95, "T906 spatial deconvolution benchmark", ha="left", va="top", fontsize=9.0, fontweight="bold", color=PALETTE["ink"])
    fig.text(
        0.075,
        0.87,
        "OmniCell-CPT uses 15,000 HVG and spatial calibration; transfer baselines are trained from Cortex_sc labels.",
        ha="left",
        va="top",
        fontsize=6.3,
        color=PALETTE["muted"],
    )
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.28), columnspacing=0.95, handlelength=1.05)
    save(fig, OUT / "figure2_t906_external_metrics")


def plot_maps(pred: pd.DataFrame, coords: np.ndarray) -> None:
    panels = [
        ("Ground truth", "ground_truth_celltype"),
        ("OmniCell-CPT fine-tuned", "pred_OmniCell-CPT fine-tuned"),
        ("scGPT-spatial transfer", "pred_scGPT-spatial transfer"),
        ("OmniCell native transfer", "pred_OmniCell native transfer"),
        ("CellPLM transfer", "pred_CellPLM transfer"),
    ]
    panels = [(title, col) for title, col in panels if col in pred.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(9.0, 1.9 * len(panels)), 2.15), constrained_layout=False)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.76, bottom=0.25, wspace=0.06)
    axes = np.atleast_1d(axes)
    for ax, (title, col) in zip(axes, panels):
        vals = pred[col].map(broad).to_numpy()
        for klass, color in BROAD_COLORS.items():
            m = vals == klass
            if m.any():
                ax.scatter(coords[m, 0], coords[m, 1], s=0.75, c=color, linewidths=0, rasterized=True)
        ax.set_title(title, loc="left", fontsize=7.1, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [mpl.lines.Line2D([0], [0], marker="o", lw=0, ms=4, color=BROAD_COLORS[k], label=k) for k in BROAD_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.03), fontsize=5.9, handletextpad=0.25, columnspacing=0.65)
    fig.text(0.01, 0.96, "T906 broad cell-class maps", ha="left", va="top", fontsize=8.8, fontweight="bold", color=PALETTE["ink"])
    save(fig, OUT / "figure2_t906_external_spatial_maps")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scad = ad.read_h5ad(INPUT / "cortex_sc_subset.h5ad")
    spad = ad.read_h5ad(INPUT / "t1001_spatial.h5ad")
    sc_idx, sp_idx, cal_local, held_local, enc, y = split_indices(scad, spad)
    rows: list[dict] = []
    predictions = spad.obs[["sample_id", "source_cell_index", "cell_type"]].copy().reset_index(drop=True)
    predictions["ground_truth_celltype"] = spad.obs["cell_type"].astype(str).to_numpy()
    predictions["split"] = "held_out"
    predictions.loc[cal_local, "split"] = "calibration"

    raw_rows, raw_pred = raw_svd_transfer(scad, spad, sc_idx, sp_idx, held_local, y, enc)
    rows.extend(raw_rows)
    predictions["pred_Raw expression SVD transfer"] = raw_pred

    native = np.load(NATIVE, mmap_mode="r")
    native_rows, native_pred = linear_transfer("OmniCell native transfer", np.asarray(native, dtype=np.float32), sc_idx, sp_idx, held_local, y, enc)
    rows.extend(native_rows)
    predictions["pred_OmniCell native transfer"] = native_pred

    cellplm = np.vstack([np.load(CELLPLM_SC), np.load(CELLPLM_SP)]).astype(np.float32)
    cellplm_rows, cellplm_pred = linear_transfer("CellPLM transfer", cellplm, sc_idx, sp_idx, held_local, y, enc)
    rows.extend(cellplm_rows)
    predictions["pred_CellPLM transfer"] = cellplm_pred

    if SCGPT_SPATIAL_SC.exists() and SCGPT_SPATIAL_SP.exists():
        scgpt_spatial = np.vstack([np.load(SCGPT_SPATIAL_SC), np.load(SCGPT_SPATIAL_SP)]).astype(np.float32)
        scgpt_rows, scgpt_pred = linear_transfer("scGPT-spatial transfer", scgpt_spatial, sc_idx, sp_idx, held_local, y, enc)
        rows.extend(scgpt_rows)
        predictions["pred_scGPT-spatial transfer"] = scgpt_pred

    fine_metrics = pd.read_csv(FINE_DIR / "random_stereo_hvg_scan_metrics.csv").iloc[0]
    rename = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro F1",
        "ari": "ARI",
        "nmi": "NMI",
    }
    for src, dst in rename.items():
        rows.append({"task": "T906 spatial deconvolution", "method": "OmniCell-CPT fine-tuned", "metric": dst, "value": float(fine_metrics[src])})
    fine_pred = pd.read_csv(FINE_DIR / "random_stereo_best_predictions.csv")
    predictions["pred_OmniCell-CPT fine-tuned"] = fine_pred["predicted_celltype"].astype(str).to_numpy()

    metrics = pd.DataFrame(rows)
    order = [
        "Raw expression SVD transfer",
        "OmniCell native transfer",
        "CellPLM transfer",
        "scGPT-spatial transfer",
        "OmniCell-CPT fine-tuned",
    ]
    order = [m for m in order if m in set(metrics["method"])]
    metrics["method"] = pd.Categorical(metrics["method"], categories=order, ordered=True)
    metrics = metrics.sort_values(["method", "metric"]).reset_index(drop=True)
    metrics["method"] = metrics["method"].astype(str)
    metrics.to_csv(OUT / "figure2_t906_external_metrics.csv", index=False)
    predictions.to_csv(OUT / "figure2_t906_external_predictions.csv", index=False)
    plot_metrics(metrics)
    plot_maps(predictions, np.asarray(spad.obsm["spatial"]))

    contract = {
        "core_conclusion": "T906 replaces T1001 as the main Stereo-seq chip; 15,000-HVG OmniCell-CPT fine-tuning is compared with raw expression transfer, original OmniCell native transfer, CellPLM transfer and scGPT-spatial transfer when embeddings are available.",
        "input_dir": str(INPUT),
        "output_dir": str(OUT),
        "fine_tuned_metrics": str(FINE_DIR / "random_stereo_hvg_scan_metrics.csv"),
        "scgpt_spatial_reference_embedding": str(SCGPT_SPATIAL_SC),
        "scgpt_spatial_spot_embedding": str(SCGPT_SPATIAL_SP),
        "metrics": str(OUT / "figure2_t906_external_metrics.csv"),
        "predictions": str(OUT / "figure2_t906_external_predictions.csv"),
    }
    (OUT / "figure2_t906_external_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
