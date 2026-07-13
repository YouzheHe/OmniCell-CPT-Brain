#!/usr/bin/env python
"""Evaluate CellPLM embeddings for T1001 spatial deconvolution/transfer."""

from __future__ import annotations
import os

import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
INPUT = PROJECT / "results" / "cortex_t1001_task_inputs"
EXT = PROJECT / "results" / "external_singlecell_embeddings"
V2 = PROJECT / "figures" / "figure2_cortex_t1001_finetune_benchmark_v2"
V2_RES = PROJECT / "results" / "cortex_t1001_hvg_finetuned"
OUT = PROJECT / "figures" / "figure2_external_spatial_cellplm"

SC_EMB = EXT / "cellplm_hvg5000_n21855" / "embedding.npy"
SP_EMB = EXT / "cellplm_t1001_hvg5000_n46826" / "embedding.npy"

PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "raw": "#8A97A8",
    "cpt": "#5784A8",
    "native": "#7C6AA6",
    "fine": "#C86054",
    "cellplm": "#70B7A6",
}
METHOD_COLORS = {
    "Raw expression SVD transfer": PALETTE["raw"],
    "OmniCell CPT 512 transfer": PALETTE["cpt"],
    "OmniCell native transfer": PALETTE["native"],
    "OmniCell fine-tuned": PALETTE["fine"],
    "CellPLM transfer": PALETTE["cellplm"],
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


def rows(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    task = "T1001 spatial deconvolution"
    return [
        {"task": task, "method": method, "metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "Balanced accuracy", "value": balanced_accuracy_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "Macro F1", "value": f1_score(y_true, y_pred, average="macro")},
        {"task": task, "method": method, "metric": "ARI", "value": adjusted_rand_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "NMI", "value": normalized_mutual_info_score(y_true, y_pred)},
    ]


def save(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def centroid_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    query_x = scaler.transform(query_x)
    classes = np.unique(train_y)
    centroids = []
    for c in classes:
        centroids.append(train_x[train_y == c].mean(axis=0))
    centroids = np.vstack(centroids).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
    query = query_x.astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
    return classes[np.argmax(query @ centroids.T, axis=1)]


def plot_metrics(metrics: pd.DataFrame) -> None:
    show = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"]
    methods = list(dict.fromkeys(metrics["method"].tolist()))
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    x = np.arange(len(show))
    width = min(0.13, 0.82 / max(1, len(methods)))
    for i, method in enumerate(methods):
        vals = []
        for metric in show:
            v = metrics.loc[metrics["method"].eq(method) & metrics["metric"].eq(metric), "value"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width * 0.9, color=METHOD_COLORS.get(method, "#9AA7B8"), label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(show)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45, alpha=0.65)
    ax.set_title("T1001 spatial deconvolution benchmark with external CellPLM transfer", loc="left", fontsize=8.6, fontweight="bold")
    ax.text(0, 1.08, "Held-out T1001 spots; CellPLM uses the same calibration split as the OmniCell transfer task.", transform=ax.transAxes, color=PALETTE["muted"], fontsize=6.4)
    ax.legend(ncol=3, bbox_to_anchor=(0, -0.25), loc="upper left", handlelength=1.1, columnspacing=1.0)
    save(fig, OUT / "figure2_external_spatial_cellplm_metrics")


def plot_maps(pred: pd.DataFrame, fine: pd.DataFrame, spad: ad.AnnData) -> None:
    coords = np.asarray(spad.obsm["spatial"])
    panels = [
        ("Ground truth", pred["ground_truth_celltype"]),
        ("OmniCell fine-tuned", fine["pred_OmniCell fine-tuned"]),
        ("CellPLM transfer", pred["pred_CellPLM transfer"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    for ax, (title, vals) in zip(axes, panels):
        bvals = vals.map(broad) if hasattr(vals, "map") else pd.Series(vals).map(broad)
        for klass, color in BROAD_COLORS.items():
            m = bvals.to_numpy() == klass
            if m.any():
                ax.scatter(coords[m, 0], coords[m, 1], s=1.0, c=color, linewidths=0, rasterized=True)
        ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [mpl.lines.Line2D([0], [0], marker="o", lw=0, ms=4, color=BROAD_COLORS[k], label=k) for k in BROAD_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=6.0, handletextpad=0.25, columnspacing=0.8)
    fig.suptitle("Spatial distribution of broad cell classes on T1001", x=0.01, y=1.06, ha="left", fontsize=8.6, fontweight="bold")
    save(fig, OUT / "figure2_external_spatial_cellplm_maps")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scad = ad.read_h5ad(INPUT / "cortex_sc_subset.h5ad")
    spad = ad.read_h5ad(INPUT / "t1001_spatial.h5ad")
    x_sc = np.load(SC_EMB)
    x_sp = np.load(SP_EMB)
    labels = np.r_[scad.obs["cell_type"].astype(str).to_numpy(), spad.obs["cell_type"].astype(str).to_numpy()]
    enc = LabelEncoder().fit(labels)
    y = enc.transform(labels)
    sc_idx = np.arange(scad.n_obs)
    sp_idx = np.arange(scad.n_obs, scad.n_obs + spad.n_obs)
    sp_y = y[sp_idx]
    cal_local, held_local = train_test_split(
        np.arange(spad.n_obs),
        test_size=0.50,
        random_state=20260527 + 13,
        stratify=sp_y,
    )
    train_idx = np.r_[sc_idx, sp_idx[cal_local]]
    held_idx = sp_idx[held_local]
    features = np.vstack([x_sc, x_sp]).astype(np.float32)
    pred_held = centroid_predict(features[train_idx], y[train_idx], features[held_idx])
    pred_all = centroid_predict(features[train_idx], y[train_idx], features[sp_idx])

    ext_metrics = pd.DataFrame(rows("CellPLM transfer", y[held_idx], pred_held))
    base = pd.read_csv(V2 / "figure2_cortex_t1001_metrics_v2.csv")
    base = base[base["task"].eq("T1001 spatial deconvolution") & base["metric"].isin(["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"])]
    metrics = pd.concat([base, ext_metrics], ignore_index=True)
    metrics.to_csv(OUT / "external_spatial_cellplm_metrics.csv", index=False)

    pred = spad.obs[["sample_id", "source_cell_index", "cell_type"]].copy().reset_index(drop=True)
    pred["ground_truth_celltype"] = spad.obs["cell_type"].astype(str).to_numpy()
    pred["pred_CellPLM transfer"] = enc.inverse_transform(pred_all)
    pred["split"] = "held_out"
    pred.loc[cal_local, "split"] = "calibration"
    pred.to_csv(OUT / "external_spatial_cellplm_predictions.csv", index=False)

    fine = pd.read_csv(V2_RES / "t1001_hvg_finetuned_predictions.csv")
    plot_metrics(metrics)
    plot_maps(pred, fine, spad)
    contract = {
        "core_conclusion": "CellPLM was deployed as an external spatial transfer baseline on T1001 and compared against raw, frozen OmniCell, native OmniCell and task-tuned OmniCell metrics.",
        "sc_embedding": str(SC_EMB),
        "spatial_embedding": str(SP_EMB),
        "metrics": str(OUT / "external_spatial_cellplm_metrics.csv"),
        "predictions": str(OUT / "external_spatial_cellplm_predictions.csv"),
        "output_dir": str(OUT),
    }
    (OUT / "figure2_external_spatial_cellplm_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
