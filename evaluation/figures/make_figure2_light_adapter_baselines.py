#!/usr/bin/env python
"""Light-adapted external foundation-model baselines for Figure 2.

The goal is to avoid comparing OmniCell-CPT task fine-tuning against purely
zero-shot external embeddings. For each available external foundation model,
this script trains the same small supervised adapter/head on the task-specific
training split and evaluates held-out labels. The OmniCell-CPT task-fine-tuned
predictions remain the primary method.
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
import torch
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
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
RESULTS = PROJECT / "results"
OUT = PROJECT / "figures" / "figure2_final_panels"
SRC = OUT / "source_data"

SC_H5AD = RESULTS / "cortex_t906_task_inputs" / "cortex_sc_subset.h5ad"
SP_H5AD = RESULTS / "cortex_t906_task_inputs" / "t1001_spatial.h5ad"

FT_SC_PRED = RESULTS / "cortex_t1001_hvg_finetuned" / "single_cell_hvg_finetuned_predictions.csv"
CPT_SP_PRED = RESULTS / "t906_hvg15000_finetuned" / "random_stereo_hvg_scan_predictions.csv"
TANGRAM_PRED = RESULTS / "figure2_formal_tangram_selected10" / "per_chip" / "T906_tangram_predictions.csv"

LATEST_EMB = RESULTS / "cortex_t1001_latest_embeddings" / "embedding.npy"
NATIVE_EMB = RESULTS / "cortex_t906_native_omnicell_embeddings" / "embedding.npy"
EXT = RESULTS / "external_singlecell_embeddings"
CELLPLM_SC = EXT / "cellplm_hvg5000_n21855" / "embedding.npy"
SCGPT_SC = EXT / "scgpt_t906_sc_hvg2000_n21855" / "embedding.npy"
SCFOUNDATION_SC = EXT / "scfoundation_t906_sc_hvg2000_n21855" / "embedding.npy"

CELLPLM_SP = EXT / "cellplm_t906_hvg5000_n61147" / "embedding.npy"
SCGPT_SPATIAL_SC = EXT / "scgpt_spatial_t906_sc_hvg2000_n21855_b64" / "embedding.npy"
SCGPT_SPATIAL_SP = EXT / "scgpt_spatial_t906_hvg2000_n61147_b64" / "embedding.npy"
NICHEFORMER = RESULTS / "external_spatial_embeddings" / "nicheformer_t906_hvg15000_sc_sp" / "embedding.npy"

SEED = 20260602
N_SPLITS = 5
MAX_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METRIC_ORDER = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"]
METRIC_LABELS = ["Accuracy", "Balanced\nacc.", "Macro F1", "ARI", "NMI"]

SC_METHODS = [
    "OmniCell-CPT",
    "OmniCell native-adapter",
    "CellPLM-adapter",
    "scGPT-adapter",
    "scFoundation-adapter",
]
SP_METHODS = [
    "OmniCell-CPT",
    "OmniCell native-adapter",
    "scGPT-spatial-adapter",
    "Nicheformer-adapter",
    "CellPLM-adapter",
    "Tangram",
]

METHOD_COLORS = {
    "OmniCell-CPT": "#A33F3B",
    "OmniCell CPT-adapter": "#5784A8",
    "OmniCell native-adapter": "#7B6FA6",
    "CellPLM-adapter": "#70B7A6",
    "scGPT-adapter": "#D39B46",
    "scFoundation-adapter": "#9C7AAE",
    "scGPT-spatial-adapter": "#D39B46",
    "Nicheformer-adapter": "#6F9FB5",
    "Tangram": "#5E9A6D",
}

PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "point": "#26323F",
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


class AdapterHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.12),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def metric_rows(method: str, replicate: str, label_space: str, true: np.ndarray, pred: np.ndarray, n_obs: int) -> list[dict[str, object]]:
    labels = sorted(set(map(str, true)).union(set(map(str, pred))))
    return [
        {"label_space": label_space, "method": method, "replicate": replicate, "metric": "Accuracy", "value": float(accuracy_score(true, pred)), "n_obs": int(n_obs)},
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
        {"label_space": label_space, "method": method, "replicate": replicate, "metric": "ARI", "value": float(adjusted_rand_score(true, pred)), "n_obs": int(n_obs)},
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


def reduce_features(features: np.ndarray, seed: int) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[1] <= MAX_DIM:
        return arr
    print(f"[reduce] {arr.shape} -> {MAX_DIM}", flush=True)
    return TruncatedSVD(n_components=MAX_DIM, random_state=seed).fit_transform(arr).astype(np.float32)


def standardize_train_all(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def fit_predict_adapter(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    *,
    seed: int,
    epochs: int = 55,
    batch_size: int = 1024,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = labels.astype(str)
    enc = LabelEncoder().fit(labels)
    y_all = enc.transform(labels).astype(np.int64)
    x_all = reduce_features(features, seed)

    train_labels = y_all[train_idx]
    if len(np.unique(train_labels)) > 1 and len(train_idx) > 100:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.14, random_state=seed)
        local_train, local_val = next(splitter.split(train_idx, train_labels))
        fit_idx = train_idx[local_train]
        val_idx = train_idx[local_val]
    else:
        fit_idx = train_idx
        val_idx = train_idx

    x_all = standardize_train_all(x_all, fit_idx)
    x_fit = torch.from_numpy(x_all[fit_idx])
    y_fit = torch.from_numpy(y_all[fit_idx])
    x_val = torch.from_numpy(x_all[val_idx]).to(DEVICE)
    y_val = torch.from_numpy(y_all[val_idx]).to(DEVICE)
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=batch_size, shuffle=True, drop_last=False)

    counts = np.bincount(y_all[fit_idx], minlength=len(enc.classes_)).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.02)
    model = AdapterHead(x_all.shape[1], len(enc.classes_)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1.5e-3)

    best_state = None
    best_loss = float("inf")
    patience = 8
    stale = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).detach().cpu())
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    preds: list[np.ndarray] = []
    x_pred = torch.from_numpy(x_all[pred_idx])
    for start in range(0, len(pred_idx), batch_size * 4):
        xb = x_pred[start : start + batch_size * 4].to(DEVICE, non_blocking=True)
        with torch.no_grad():
            preds.append(model(xb).argmax(dim=1).detach().cpu().numpy())
    pred_int = np.concatenate(preds)
    return enc.inverse_transform(pred_int)


def singlecell_feature_dict(adata: ad.AnnData) -> dict[str, np.ndarray]:
    n = adata.n_obs
    candidates = {
        "OmniCell CPT-adapter": LATEST_EMB,
        "OmniCell native-adapter": NATIVE_EMB,
        "CellPLM-adapter": CELLPLM_SC,
        "scGPT-adapter": SCGPT_SC,
        "scFoundation-adapter": SCFOUNDATION_SC,
    }
    out: dict[str, np.ndarray] = {}
    for method, path in candidates.items():
        if not path.exists():
            print(f"[single-cell] missing {method}: {path}", flush=True)
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] < n:
            print(f"[single-cell] skip {method}: {arr.shape[0]} rows < {n}", flush=True)
            continue
        out[method] = np.asarray(arr[:n], dtype=np.float32)
        print(f"[single-cell] loaded {method}: {out[method].shape}", flush=True)
    return out


def eval_singlecell_finetuned() -> list[dict[str, object]]:
    if not FT_SC_PRED.exists():
        return []
    df = pd.read_csv(FT_SC_PRED)
    pred_col = "pred_OmniCell fine-tuned" if "pred_OmniCell fine-tuned" in df.columns else [c for c in df.columns if c.startswith("pred")][0]
    true = df["cell_type"].astype(str).to_numpy()
    pred = df[pred_col].astype(str).to_numpy()
    rows = metric_rows("OmniCell-CPT", "held_out_finetuned", "fine cell type", true, pred, len(df))
    rows += metric_rows(
        "OmniCell-CPT",
        "held_out_finetuned",
        "broad cell class",
        np.array([broad(v) for v in true]),
        np.array([broad(v) for v in pred]),
        len(df),
    )
    return rows


def compute_singlecell_metrics() -> pd.DataFrame:
    adata = ad.read_h5ad(SC_H5AD)
    y_fine = adata.obs["cell_type"].astype(str).to_numpy()
    features = singlecell_feature_dict(adata)
    enc_tmp = LabelEncoder().fit_transform(y_fine)
    rows: list[dict[str, object]] = []
    for method in [m for m in SC_METHODS if m in features]:
        splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.30, random_state=SEED)
        for split_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_fine)), enc_tmp)):
            pred = fit_predict_adapter(
                features[method],
                y_fine,
                train_idx,
                test_idx,
                seed=SEED + split_id + 17 * SC_METHODS.index(method),
            epochs=42,
            )
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
            print(f"[single-cell adapter] {method} split {split_id} done", flush=True)
    rows.extend(eval_singlecell_finetuned())
    metrics = pd.DataFrame(rows)
    metrics.to_csv(SRC / "fig2_light_adapter_singlecell_metrics_by_split.csv", index=False)
    summarise(metrics).to_csv(SRC / "fig2_light_adapter_singlecell_metrics_summary.csv", index=False)
    return metrics


def read_spatial_base() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    scad = ad.read_h5ad(SC_H5AD, backed="r")
    spad = ad.read_h5ad(SP_H5AD, backed="r")
    sc_labels = scad.obs["cell_type"].astype(str).to_numpy()
    sp_obs = spad.obs[["x", "y", "cell_type"]].copy()
    sp_obs["x_key"] = sp_obs["x"].round().astype(int)
    sp_obs["y_key"] = sp_obs["y"].round().astype(int)

    cpt = pd.read_csv(CPT_SP_PRED).copy()
    cpt["x_key"] = cpt["x"].round().astype(int)
    cpt["y_key"] = cpt["y"].round().astype(int)
    cpt = cpt.rename(columns={"ground_truth_celltype": "truth_celltype", "predicted_celltype": "pred_OmniCell-CPT"})
    keep = ["x_key", "y_key", "split", "truth_celltype", "pred_OmniCell-CPT", "chip", "hvg_size"]
    base = sp_obs.merge(cpt[keep], on=["x_key", "y_key"], how="left", validate="one_to_one")
    if base["truth_celltype"].isna().any():
        raise RuntimeError(f"Spatial coordinate alignment failed for {int(base['truth_celltype'].isna().sum())} spots")
    base["truth_celltype"] = base["truth_celltype"].astype(str)
    sp_labels = base["truth_celltype"].to_numpy()
    scad.file.close()
    spad.file.close()
    return base, sc_labels, sp_labels


def spatial_feature_dict(sc_n: int, sp_n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if NATIVE_EMB.exists():
        out["OmniCell native-adapter"] = np.asarray(np.load(NATIVE_EMB, mmap_mode="r")[: sc_n + sp_n], dtype=np.float32)
    if SCGPT_SPATIAL_SC.exists() and SCGPT_SPATIAL_SP.exists():
        out["scGPT-spatial-adapter"] = np.vstack([np.load(SCGPT_SPATIAL_SC), np.load(SCGPT_SPATIAL_SP)]).astype(np.float32)
    if NICHEFORMER.exists():
        out["Nicheformer-adapter"] = np.asarray(np.load(NICHEFORMER, mmap_mode="r")[: sc_n + sp_n], dtype=np.float32)
    if CELLPLM_SC.exists() and CELLPLM_SP.exists():
        out["CellPLM-adapter"] = np.vstack([np.load(CELLPLM_SC), np.load(CELLPLM_SP)]).astype(np.float32)
    for method, arr in out.items():
        print(f"[spatial] loaded {method}: {arr.shape}", flush=True)
    return out


def compute_spatial_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    base, sc_labels, sp_labels = read_spatial_base()
    sc_n = len(sc_labels)
    sp_n = len(sp_labels)
    calibration = base["split"].astype(str).eq("calibration").to_numpy()
    train_idx = np.r_[np.arange(sc_n), sc_n + np.flatnonzero(calibration)]
    pred_idx = sc_n + np.arange(sp_n)
    labels_all = np.r_[sc_labels, sp_labels]
    for method, feat in spatial_feature_dict(sc_n, sp_n).items():
        pred = fit_predict_adapter(
            feat,
            labels_all,
            train_idx,
            pred_idx,
            seed=SEED + 101 + SP_METHODS.index(method),
            epochs=42,
            batch_size=1536,
        )
        base[f"pred_{method}"] = pred
        print(f"[spatial adapter] {method} done", flush=True)

    tg = pd.read_csv(TANGRAM_PRED).copy()
    tg["x_key"] = tg["x"].round().astype(int)
    tg["y_key"] = tg["y"].round().astype(int)
    tg = tg[["x_key", "y_key", "predicted_celltype", "ground_truth_celltype"]].rename(
        columns={"predicted_celltype": "pred_Tangram", "ground_truth_celltype": "tangram_truth_celltype"}
    )
    pred_df = base.merge(tg, on=["x_key", "y_key"], how="inner", validate="one_to_one")
    pred_df = pred_df[pred_df["split"].astype(str).eq("held_out")].copy()
    pred_df["truth_broad"] = pred_df["truth_celltype"].map(broad)
    for method in SP_METHODS:
        col = f"pred_{method}"
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].astype(str)
            pred_df[f"pred_broad_{method}"] = pred_df[col].map(broad)

    rows: list[dict[str, object]] = []
    for method in SP_METHODS:
        col = f"pred_{method}"
        bcol = f"pred_broad_{method}"
        if col not in pred_df.columns or bcol not in pred_df.columns:
            continue
        rows.extend(metric_rows(method, "T906", "fine cell type", pred_df["truth_celltype"].to_numpy(), pred_df[col].to_numpy(), len(pred_df)))
        rows.extend(metric_rows(method, "T906", "broad cell class", pred_df["truth_broad"].to_numpy(), pred_df[bcol].to_numpy(), len(pred_df)))
    metrics = pd.DataFrame(rows)
    pred_df.to_csv(SRC / "fig2_light_adapter_spatial_predictions.csv", index=False)
    metrics.to_csv(SRC / "fig2_light_adapter_spatial_metrics.csv", index=False)
    summarise(metrics).to_csv(SRC / "fig2_light_adapter_spatial_metrics_summary.csv", index=False)
    return pred_df, metrics


def draw_metric_summary(
    metrics: pd.DataFrame,
    label_space: str,
    methods: list[str],
    stem: str,
    title: str,
    subtitle: str,
    ylabel: str,
    figsize: tuple[float, float],
    ylim: float | None = None,
) -> None:
    sub = metrics[metrics["label_space"].eq(label_space)].copy()
    present = [m for m in methods if m in set(sub["method"])]
    x = np.arange(len(METRIC_ORDER))
    width = min(0.15, 0.82 / max(len(present), 1))
    offsets = {m: (i - (len(present) - 1) / 2) * width for i, m in enumerate(present)}
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.100, right=0.985, top=0.61, bottom=0.28)
    for method in present:
        means, sems = [], []
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
            color=METHOD_COLORS.get(method, "#9AA7B8"),
            edgecolor="none",
            alpha=0.91,
            error_kw={"elinewidth": 0.72, "capthick": 0.72, "capsize": 2.0, "ecolor": PALETTE["ink"]},
            label=method,
            zorder=2,
        )
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=4.15)
        for i, metric in enumerate(METRIC_ORDER):
            vals = sub[sub["method"].eq(method) & sub["metric"].eq(metric)]["value"].astype(float).to_numpy()
            if len(vals) > 1:
                ax.scatter(
                    np.full(len(vals), xpos[i]) + rng.normal(0, width * 0.055, size=len(vals)),
                    vals,
                    s=3.3,
                    color=PALETTE["point"],
                    alpha=0.48,
                    linewidths=0,
                    zorder=3,
                )
    ax.set_ylim(0, ylim if ylim is not None else max(0.95, float(sub["value"].max()) + 0.10))
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.805),
        ncol=min(3, max(1, len(present))),
        fontsize=4.65,
        handlelength=0.9,
        columnspacing=0.72,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    fig.text(0.105, 0.97, title, ha="left", va="top", fontsize=7.7, fontweight="bold")
    fig.text(0.105, 0.875, subtitle, ha="left", va="top", fontsize=5.1, color=PALETTE["muted"])
    save(fig, OUT / stem)


def fine_palette(labels: list[str]) -> dict[str, str]:
    palettes = ["tab20", "tab20b", "tab20c", "Set3"]
    colors: list[str] = []
    for name in palettes:
        cmap = mpl.colormaps[name]
        colors.extend([mpl.colors.to_hex(c) for c in getattr(cmap, "colors", [cmap(i / 20) for i in range(20)])])
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


def draw_spatial_maps(pred: pd.DataFrame, label_space: str, stem: str) -> None:
    panels: list[tuple[str, str]] = [("Ground truth", "truth_broad" if label_space == "broad" else "truth_celltype")]
    for method in SP_METHODS:
        col = f"pred_broad_{method}" if label_space == "broad" else f"pred_{method}"
        if col in pred.columns:
            panels.append((method, col))
    if label_space == "broad":
        order = BROAD_ORDER
        colors = BROAD_COLORS
        title = "Light-adapted broad cell-class spatial deconvolution"
        legend_title = "Broad cell class"
        figsize = (8.45, 3.85)
        legend_width = 0.96
        point_size = 0.42
    else:
        counts = pd.concat([pred[col].astype(str) for _, col in panels]).value_counts()
        order = counts.index.tolist()
        colors = fine_palette(order)
        title = "Light-adapted fine cell-type spatial deconvolution"
        legend_title = "Fine cell type"
        figsize = (9.25, 4.20)
        legend_width = 1.52
        point_size = 0.34
    ncols = 4
    nrows = math.ceil(len(panels) / ncols)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [legend_width],
        left=0.035,
        right=0.99,
        top=0.86,
        bottom=0.06,
        wspace=0.035,
        hspace=0.12,
    )
    xlim = (pred["x"].min(), pred["x"].max())
    ylim = (pred["y"].min(), pred["y"].max())
    source_rows = []
    for i, (panel, col) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        sub = pred[["x", "y", col]].copy()
        sub["_panel_label"] = panel
        sub["_plot_label"] = sub[col].astype(str)
        source_rows.append(sub[["x", "y", "_panel_label", "_plot_label"]])
        for lab in order:
            mask = sub["_plot_label"].eq(lab).to_numpy()
            if mask.any():
                ax.scatter(sub.loc[mask, "x"], sub.loc[mask, "y"], s=point_size, color=colors.get(lab, "#BFC5CD"), alpha=0.74, linewidths=0, rasterized=True)
        ax.set_title(panel, fontsize=5.8, fontweight="bold", pad=1.7)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        for sp in ax.spines.values():
            sp.set_visible(False)
    for j in range(len(panels), nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")
    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.axis("off")
    ax_leg.text(0.0, 0.99, legend_title, transform=ax_leg.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=PALETTE["muted"])
    if label_space == "fine":
        rows_per_col = int(math.ceil(len(order) / 2))
        step = 0.88 / max(rows_per_col - 1, 1)
        for i, lab in enumerate(order):
            col_i = i // rows_per_col
            row_i = i % rows_per_col
            x0 = 0.03 + col_i * 0.50
            y = 0.93 - row_i * step
            ax_leg.scatter([x0], [y], transform=ax_leg.transAxes, s=10, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(x0 + 0.045, y, short_fine_label(lab), transform=ax_leg.transAxes, fontsize=4.05, va="center", ha="left")
    else:
        y = 0.94
        for lab in order:
            ax_leg.scatter([0.04], [y], transform=ax_leg.transAxes, s=17, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(0.10, y, lab, transform=ax_leg.transAxes, fontsize=5.2, va="center", ha="left")
            y -= 0.090
    fig.text(0.035, 0.975, title, ha="left", va="top", fontsize=7.7, fontweight="bold")
    fig.text(0.035, 0.915, f"External foundation embeddings receive the same small supervised adapter; T906 held-out spots, n = {len(pred):,}.", ha="left", va="top", fontsize=5.1, color=PALETTE["muted"])
    save(fig, OUT / stem)
    pd.concat(source_rows, ignore_index=True).to_csv(SRC / f"{stem}_source.csv", index=False)


def draw_singlecell_umaps() -> None:
    source = SRC / "fig2_nonzero_hvg_singlecell_method_umaps_source.csv"
    if not source.exists():
        return
    df = pd.read_csv(source)
    keep_map = {
        "OmniCell-CPT fine-tuned": "OmniCell-CPT",
        "OmniCell native": "OmniCell native-adapter",
        "CellPLM": "CellPLM-adapter",
        "scGPT": "scGPT-adapter",
        "scFoundation": "scFoundation-adapter",
    }
    df = df[df["method"].isin(keep_map)].copy()
    df["method"] = df["method"].map(keep_map)
    for label_space, stem in [("broad", "fig2_light_adapter_singlecell_umaps_broad"), ("fine", "fig2_light_adapter_singlecell_umaps_fine")]:
        if label_space == "broad":
            label_col = "broad_cell_class"
            order = BROAD_ORDER
            colors = BROAD_COLORS
            title = "Single-cell embeddings used by light adapters"
            legend_title = "Broad cell class"
            figsize = (7.65, 4.65)
            legend_width = 1.02
        else:
            label_col = "cell_type"
            order = df[label_col].astype(str).value_counts().index.tolist()
            colors = fine_palette(order)
            title = "Single-cell embeddings used by light adapters"
            legend_title = "Fine cell type"
            figsize = (8.95, 5.05)
            legend_width = 1.72
        methods = [m for m in SC_METHODS if m in set(df["method"])]
        ncols = 3
        nrows = math.ceil(len(methods) / ncols)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [legend_width], left=0.035, right=0.99, top=0.82, bottom=0.06, wspace=0.060, hspace=0.16)
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[i // ncols, i % ncols])
            sub = df[df["method"].eq(method)].copy()
            if len(sub) > 7200:
                sub = sub.sample(7200, random_state=SEED + i)
            for lab in order:
                mask = sub[label_col].astype(str).eq(lab).to_numpy()
                if mask.any():
                    ax.scatter(sub.loc[mask, "umap_1"], sub.loc[mask, "umap_2"], s=0.42, color=colors.get(lab, "#BFC5CD"), alpha=0.74, linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(method, fontsize=5.8, fontweight="bold", pad=1.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
        for j in range(len(methods), nrows * ncols):
            ax = fig.add_subplot(gs[j // ncols, j % ncols])
            ax.axis("off")
        ax_leg = fig.add_subplot(gs[:, -1])
        ax_leg.axis("off")
        ax_leg.text(0.0, 0.99, legend_title, transform=ax_leg.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=PALETTE["muted"])
        if label_space == "fine":
            rows_per_col = int(math.ceil(len(order) / 2))
            step = 0.88 / max(rows_per_col - 1, 1)
            for i, lab in enumerate(order):
                col_i = i // rows_per_col
                row_i = i % rows_per_col
                x0 = 0.03 + col_i * 0.50
                y = 0.93 - row_i * step
                ax_leg.scatter([x0], [y], transform=ax_leg.transAxes, s=10, color=colors.get(lab, "#BFC5CD"), linewidths=0)
                ax_leg.text(x0 + 0.045, y, short_fine_label(lab), transform=ax_leg.transAxes, fontsize=4.05, va="center", ha="left")
        else:
            y = 0.94
            for lab in order:
                ax_leg.scatter([0.04], [y], transform=ax_leg.transAxes, s=17, color=colors.get(lab, "#BFC5CD"), linewidths=0)
                ax_leg.text(0.10, y, lab, transform=ax_leg.transAxes, fontsize=5.2, va="center", ha="left")
                y -= 0.090
        fig.text(0.035, 0.975, title, ha="left", va="top", fontsize=7.7, fontweight="bold")
        fig.text(0.035, 0.905, "Panels show held-out Cortex_sc embeddings before the supervised adapter head.", ha="left", va="top", fontsize=5.1, color=PALETTE["muted"])
        save(fig, OUT / stem)
        df.to_csv(SRC / f"{stem}_source.csv", index=False)


def write_contract(sc_metrics: pd.DataFrame, sp_metrics: pd.DataFrame, sp_pred: pd.DataFrame) -> None:
    contract = {
        "core_conclusion": "External foundation-model baselines are evaluated after the same lightweight supervised adapter rather than as pure zero-shot embeddings.",
        "adapter": {
            "architecture": "Dropout -> Linear classifier/deconvolution head",
            "training": "AdamW with class-balanced cross-entropy, early stopping on an internal validation split",
            "backbone_policy": "Backbone embeddings are frozen; only the small supervised head is trained.",
        },
        "single_cell": {
            "truth_column": "Cortex_sc obs['cell_type']",
            "metrics_by_split": str(SRC / "fig2_light_adapter_singlecell_metrics_by_split.csv"),
            "metrics_summary": str(SRC / "fig2_light_adapter_singlecell_metrics_summary.csv"),
            "methods": [m for m in SC_METHODS if m in set(sc_metrics["method"])],
            "missing_full_embeddings": ["Geneformer full Cortex_sc embedding was not present; only a smoke-test file was available, so it was not plotted as a real baseline."],
        },
        "spatial": {
            "truth_column": "T906 obs['CellType_m'] / matched truth_celltype",
            "matched_held_out_spots": int(len(sp_pred)),
            "metrics": str(SRC / "fig2_light_adapter_spatial_metrics.csv"),
            "predictions": str(SRC / "fig2_light_adapter_spatial_predictions.csv"),
            "methods": [m for m in SP_METHODS if m in set(sp_metrics["method"])],
        },
        "exports": str(OUT),
    }
    (SRC / "fig2_light_adapter_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)

    redraw_only = os.environ.get("REDRAW_ONLY", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if redraw_only:
        print("[redraw] using cached light-adapter metrics and predictions", flush=True)
        sc_metrics = pd.read_csv(SRC / "fig2_light_adapter_singlecell_metrics_by_split.csv")
    else:
        print("[single-cell] training light adapters", flush=True)
        sc_metrics = compute_singlecell_metrics()
    sc_metrics = sc_metrics[sc_metrics["method"].isin(SC_METHODS)].copy()
    sc_metrics.to_csv(SRC / "fig2_light_adapter_singlecell_metrics_by_split.csv", index=False)
    summarise(sc_metrics).to_csv(SRC / "fig2_light_adapter_singlecell_metrics_summary.csv", index=False)
    draw_metric_summary(
        sc_metrics,
        "broad cell class",
        SC_METHODS,
        "fig2_light_adapter_singlecell_broad_summary",
        "Light-adapted single-cell broad-class annotation",
        "External foundation embeddings receive the same small supervised adapter; bars show mean +/- s.e.m.",
        "held-out score",
        figsize=(7.70, 2.85),
        ylim=1.02,
    )
    draw_metric_summary(
        sc_metrics,
        "fine cell type",
        SC_METHODS,
        "fig2_light_adapter_singlecell_fine_summary",
        "Light-adapted single-cell fine-cell annotation",
        "External foundation embeddings receive the same small supervised adapter; OmniCell-CPT uses its task-fine-tuned held-out split.",
        "held-out score",
        figsize=(7.70, 2.85),
        ylim=1.02,
    )
    draw_singlecell_umaps()

    if redraw_only:
        sp_pred = pd.read_csv(SRC / "fig2_light_adapter_spatial_predictions.csv")
        sp_metrics = pd.read_csv(SRC / "fig2_light_adapter_spatial_metrics.csv")
    else:
        print("[spatial] training light adapters", flush=True)
        sp_pred, sp_metrics = compute_spatial_predictions()
    draw_metric_summary(
        sp_metrics,
        "broad cell class",
        SP_METHODS,
        "fig2_light_adapter_spatial_broad_summary",
        "Light-adapted broad cell-class spatial deconvolution",
        f"All methods are scored on identical matched T906 held-out spots (n = {len(sp_pred):,}).",
        "score on T906",
        figsize=(7.50, 2.82),
        ylim=1.02,
    )
    draw_metric_summary(
        sp_metrics,
        "fine cell type",
        SP_METHODS,
        "fig2_light_adapter_spatial_fine_summary",
        "Light-adapted fine cell-type spatial deconvolution",
        f"All methods are scored on identical matched T906 held-out spots (n = {len(sp_pred):,}).",
        "score on T906",
        figsize=(7.50, 2.82),
        ylim=0.92,
    )
    draw_spatial_maps(sp_pred, "broad", "fig2_light_adapter_spatial_broad_maps")
    draw_spatial_maps(sp_pred, "fine", "fig2_light_adapter_spatial_fine_maps")
    write_contract(sc_metrics, sp_metrics, sp_pred)


if __name__ == "__main__":
    main()
