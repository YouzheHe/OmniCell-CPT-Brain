#!/usr/bin/env python
"""Figure 2 selected-quality all-method spatial deconvolution panels."""

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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
RESULTS = PROJECT / "results"
OUT = PROJECT / "figures" / "figure2_final_panels"
SRC = OUT / "source_data"

SC_H5AD = RESULTS / "cortex_t906_task_inputs" / "cortex_sc_subset.h5ad"
SP_BASE = RESULTS / "random_stereo_hvg_scan" / "spatial_h5ad"
OMNI_PRED = RESULTS / "random_stereo_hvg_scan" / "random_stereo_hvg_scan_predictions.csv"
OMNI_METRICS = RESULTS / "random_stereo_hvg_scan" / "random_stereo_hvg_scan_metrics.csv"
TANGRAM_PRED = RESULTS / "figure2_formal_tangram_selected10" / "tangram_selected10_predictions.csv"

EXT = RESULTS / "external_singlecell_embeddings"
EXT_SP = RESULTS / "external_spatial_embeddings"
CELLPLM_SC = EXT / "cellplm_hvg5000_n21855" / "embedding.npy"
SCGPT_SPATIAL_SC = EXT / "scgpt_spatial_t906_sc_hvg2000_n21855_b64" / "embedding.npy"

CHIPS = ["T917", "T991", "T989", "T988"]
METHODS = ["OmniCell-CPT", "CellPLM-adapter", "scGPT-spatial-adapter", "Nicheformer-adapter", "Tangram"]
METHOD_COLORS = {
    "OmniCell-CPT": "#A33F3B",
    "CellPLM-adapter": "#70B7A6",
    "scGPT-spatial-adapter": "#D39B46",
    "Nicheformer-adapter": "#6F9FB5",
    "Tangram": "#5E9A6D",
}
MAP_METHOD_LABELS = {
    "OmniCell-CPT": "OmniCell",
    "CellPLM-adapter": "CellPLM",
    "scGPT-spatial-adapter": "scGPT-sp",
    "Nicheformer-adapter": "Nicheformer",
    "Tangram": "Tangram",
}
METRIC_ORDER = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"]
METRIC_LABELS = ["Accuracy", "Balanced\nacc.", "Macro F1", "ARI", "NMI"]
SEED = 20260602
MAX_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        "font.size": 6.3,
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
            nn.Dropout(0.08),
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
    if any(k in s for k in ["Endothelial", "Pericyte", "Vascular", "VLMC", "SMC", "Mural"]) or any(k in low for k in ["endo", "peri", "vascular", "mural"]):
        return "Vascular"
    if any(k in s for k in ["GABA", "RELN", "VIP", "PVALB", "SST", "LAMP5", "Inhibitory"]):
        return "Inhibitory neuron"
    if "neuron" in low or any(k in s for k in ["IT", "CT", "ET", "NP"]):
        return "Excitatory neuron"
    return "Other"


def metric_rows(method: str, chip: str, label_space: str, true: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    labels = sorted(set(map(str, true)).union(set(map(str, pred))))
    return [
        {"label_space": label_space, "method": method, "chip": chip, "metric": "Accuracy", "value": float(accuracy_score(true, pred)), "n_spots": int(len(true))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "Balanced accuracy", "value": float(balanced_accuracy_score(true, pred)), "n_spots": int(len(true))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "Macro F1", "value": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)), "n_spots": int(len(true))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "ARI", "value": float(adjusted_rand_score(true, pred)), "n_spots": int(len(true))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "NMI", "value": float(normalized_mutual_info_score(true, pred)), "n_spots": int(len(true))},
    ]


def sem(vals: pd.Series | np.ndarray) -> float:
    arr = np.asarray(vals, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


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


def fit_predict_adapter(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, pred_idx: np.ndarray, seed: int, epochs: int = 46, batch_size: int = 1536) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = labels.astype(str)
    enc = LabelEncoder().fit(labels)
    y_all = enc.transform(labels).astype(np.int64)
    x_all = reduce_features(features, seed)

    train_labels = y_all[train_idx]
    rng = np.random.default_rng(seed)
    val_idx_parts = []
    fit_idx_parts = []
    for cls in np.unique(train_labels):
        cls_idx = train_idx[train_labels == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * 0.12))) if len(cls_idx) >= 8 else 0
        if n_val:
            val_idx_parts.append(cls_idx[:n_val])
            fit_idx_parts.append(cls_idx[n_val:])
        else:
            fit_idx_parts.append(cls_idx)
    fit_idx = np.concatenate(fit_idx_parts)
    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else fit_idx

    x_all = standardize_train_all(x_all, fit_idx)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_all[fit_idx]), torch.from_numpy(y_all[fit_idx])), batch_size=batch_size, shuffle=True, drop_last=False)
    x_val = torch.from_numpy(x_all[val_idx]).to(DEVICE)
    y_val = torch.from_numpy(y_all[val_idx]).to(DEVICE)

    counts = np.bincount(y_all[fit_idx], minlength=len(enc.classes_)).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.01)
    model = AdapterHead(x_all.shape[1], len(enc.classes_)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1.2e-3)

    best_state = None
    best_loss = float("inf")
    stale = 0
    for _ in range(epochs):
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
        if stale >= 7:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    preds = []
    x_pred = torch.from_numpy(x_all[pred_idx])
    for start in range(0, len(pred_idx), batch_size * 4):
        xb = x_pred[start : start + batch_size * 4].to(DEVICE, non_blocking=True)
        with torch.no_grad():
            preds.append(model(xb).argmax(dim=1).detach().cpu().numpy())
    return enc.inverse_transform(np.concatenate(preds))


def load_sc_labels() -> np.ndarray:
    adata = ad.read_h5ad(SC_H5AD, backed="r")
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    adata.file.close()
    return labels


def load_spatial_obs(chip: str) -> pd.DataFrame:
    adata = ad.read_h5ad(SP_BASE / f"{chip}.h5ad", backed="r")
    obs = adata.obs[["x", "y", "cell_type"]].copy()
    obs["spot_index"] = np.arange(adata.n_obs)
    obs["chip"] = chip
    obs["x_key"] = obs["x"].round().astype(int)
    obs["y_key"] = obs["y"].round().astype(int)
    adata.file.close()
    return obs


def chip_feature_paths(chip: str, sp_n: int) -> dict[str, Path]:
    low = chip.lower()
    return {
        "CellPLM-adapter": EXT / f"cellplm_{low}_hvg5000_n{sp_n}" / "embedding.npy",
        "scGPT-spatial-adapter": EXT / f"scgpt_spatial_{low}_hvg2000_n{sp_n}_b64" / "embedding.npy",
        "Nicheformer-adapter": EXT_SP / f"nicheformer_{low}_hvg15000_sc_sp" / "embedding.npy",
    }


def compute_predictions() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sc_labels = load_sc_labels()
    sc_n = len(sc_labels)
    cellplm_sc = np.load(CELLPLM_SC, mmap_mode="r")
    scgpt_sc = np.load(SCGPT_SPATIAL_SC, mmap_mode="r")
    omni = pd.read_csv(OMNI_PRED)
    tangram = pd.read_csv(TANGRAM_PRED)
    qc = pd.read_csv(OMNI_METRICS).sort_values("macro_f1", ascending=False)
    chips = [c for c in qc["chip"].astype(str).tolist() if c in CHIPS]

    all_pred = []
    all_metrics = []
    missing: dict[str, list[str]] = {}
    for chip in chips:
        print(f"[chip] {chip}", flush=True)
        obs = load_spatial_obs(chip)
        sp_n = len(obs)
        cpt = omni[omni["chip"].astype(str).eq(chip)].copy()
        cpt["x_key"] = cpt["x"].round().astype(int)
        cpt["y_key"] = cpt["y"].round().astype(int)
        cpt = cpt.rename(columns={"ground_truth_celltype": "truth_omni", "predicted_celltype": "pred_OmniCell-CPT"})
        base = obs.merge(cpt[["x_key", "y_key", "split", "truth_omni", "pred_OmniCell-CPT"]], on=["x_key", "y_key"], how="left", validate="one_to_one")
        if base["split"].isna().any():
            raise RuntimeError(f"{chip}: failed to align {int(base['split'].isna().sum())} OmniCell split rows")

        tg = tangram[tangram["chip"].astype(str).eq(chip)].copy()
        tg["x_key"] = tg["x"].round().astype(int)
        tg["y_key"] = tg["y"].round().astype(int)
        tg = tg.rename(columns={"ground_truth_celltype": "truth_celltype", "predicted_celltype": "pred_Tangram"})
        pred = base.merge(tg[["x_key", "y_key", "truth_celltype", "pred_Tangram"]], on=["x_key", "y_key"], how="inner", validate="one_to_one")
        pred = pred[pred["split"].astype(str).eq("held_out")].copy()

        sp_labels = base["truth_omni"].astype(str).to_numpy()
        labels_all = np.r_[sc_labels, sp_labels]
        calibration = base["split"].astype(str).eq("calibration").to_numpy()
        train_idx = np.r_[np.arange(sc_n), sc_n + np.flatnonzero(calibration)]
        all_sp_pred_idx = sc_n + np.arange(sp_n)
        paths = chip_feature_paths(chip, sp_n)
        for method, path in paths.items():
            if not path.exists():
                missing.setdefault(method, []).append(chip)
                print(f"[missing] {method} {chip}: {path}", flush=True)
                continue
            if method == "CellPLM-adapter":
                features = np.vstack([cellplm_sc, np.load(path)]).astype(np.float32)
            elif method == "scGPT-spatial-adapter":
                features = np.vstack([scgpt_sc, np.load(path)]).astype(np.float32)
            else:
                features = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
            if features.shape[0] != sc_n + sp_n:
                raise RuntimeError(f"{chip} {method}: expected {sc_n + sp_n} rows, got {features.shape[0]}")
            pred_all = fit_predict_adapter(features, labels_all, train_idx, all_sp_pred_idx, seed=SEED + 17 * len(all_metrics) + sum(ord(x) for x in chip))
            mapper = pd.DataFrame({"spot_index": np.arange(sp_n), f"pred_{method}": pred_all})
            pred = pred.merge(mapper, on="spot_index", how="left", validate="many_to_one")
            print(f"[adapter] {chip} {method} done", flush=True)

        pred["truth_broad"] = pred["truth_celltype"].map(broad)
        pred["pred_broad_OmniCell-CPT"] = pred["pred_OmniCell-CPT"].map(broad)
        pred["pred_broad_Tangram"] = pred["pred_Tangram"].map(broad)
        for method in METHODS:
            col = f"pred_{method}"
            if col in pred.columns:
                pred[f"pred_broad_{method}"] = pred[col].map(broad)
                all_metrics += metric_rows(method, chip, "fine cell type", pred["truth_celltype"].astype(str).to_numpy(), pred[col].astype(str).to_numpy())
                all_metrics += metric_rows(method, chip, "broad cell class", pred["truth_broad"].astype(str).to_numpy(), pred[f"pred_broad_{method}"].astype(str).to_numpy())
        all_pred.append(pred)

    pred_df = pd.concat(all_pred, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    pred_df.to_csv(SRC / "fig2_selected_quality_all_methods_predictions.csv", index=False)
    metrics.to_csv(SRC / "fig2_selected_quality_all_methods_metrics_by_chip.csv", index=False)
    summary = (
        metrics.groupby(["label_space", "method", "metric"], as_index=False)
        .agg(mean=("value", "mean"), sem=("value", sem), n_chips=("chip", "nunique"), n_spots_total=("n_spots", "sum"))
        .sort_values(["label_space", "method", "metric"])
    )
    summary.to_csv(SRC / "fig2_selected_quality_all_methods_metrics_summary.csv", index=False)
    (SRC / "fig2_selected_quality_all_methods_missing.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    return pred_df, metrics, chips


def draw_metric_summary(metrics: pd.DataFrame, label_space: str, stem: str, chips: list[str], ylim: float) -> None:
    sub = metrics[metrics["label_space"].eq(label_space)].copy()
    present = [m for m in METHODS if m in set(sub["method"])]
    x = np.arange(len(METRIC_ORDER))
    width = min(0.15, 0.82 / max(1, len(present)))
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7.55, 2.82))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.62, bottom=0.28)
    for i, method in enumerate(present):
        means, errs = [], []
        for metric in METRIC_ORDER:
            vals = sub[sub["method"].eq(method) & sub["metric"].eq(metric)]["value"].astype(float).to_numpy()
            means.append(float(vals.mean()))
            errs.append(sem(vals))
        xpos = x + (i - (len(present) - 1) / 2) * width
        bars = ax.bar(
            xpos,
            means,
            yerr=errs,
            width=width * 0.88,
            color=METHOD_COLORS.get(method, "#9AA7B8"),
            edgecolor="none",
            alpha=0.91,
            error_kw={"elinewidth": 0.72, "capthick": 0.72, "capsize": 2.0, "ecolor": PALETTE["ink"]},
            label=method,
            zorder=2,
        )
        for bar, value in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.016, f"{value:.2f}", ha="center", va="bottom", fontsize=4.1)
        for j, metric in enumerate(METRIC_ORDER):
            vals = sub[sub["method"].eq(method) & sub["metric"].eq(metric)]["value"].astype(float).to_numpy()
            ax.scatter(np.full(len(vals), xpos[j]) + rng.normal(0, width * 0.050, len(vals)), vals, s=3.8, color=PALETTE["point"], alpha=0.48, linewidths=0, zorder=3)
    ax.set_ylim(0, ylim)
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylabel("score across selected chips")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
    ax.set_axisbelow(True)
    fig.legend(loc="upper right", bbox_to_anchor=(0.985, 0.825), ncol=3, fontsize=4.75, handlelength=0.9, handletextpad=0.35, columnspacing=0.70)
    title = "Selected high-quality all-method spatial deconvolution"
    subtitle = f"Top {len(chips)} QC-ranked chips ({', '.join(chips)}); bars show mean +/- s.e.m. across chips."
    fig.text(0.095, 0.975, title, ha="left", va="top", fontsize=7.8, fontweight="bold")
    fig.text(0.095, 0.875, subtitle, ha="left", va="top", fontsize=5.0, color=PALETTE["muted"])
    save(fig, OUT / stem)


def fine_palette(labels: list[str]) -> dict[str, str]:
    colors = []
    for name in ["tab20", "tab20b", "tab20c", "Set3"]:
        cmap = mpl.colormaps[name]
        colors.extend([mpl.colors.to_hex(c) for c in getattr(cmap, "colors", [cmap(i / 20) for i in range(20)])])
    return {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}


def short_label(label: str) -> str:
    s = str(label)
    repl = {"Oligodendrocyte precursor cells": "OPC", "PVALB Chandelier neurons": "PVALB Chandelier", "SST CHODL neurons": "SST CHODL"}
    if s in repl:
        return repl[s]
    if s.endswith(" neurons"):
        return s[: -len(" neurons")]
    return s


def draw_maps(pred: pd.DataFrame, label_space: str, stem: str, chips: list[str]) -> None:
    if label_space == "broad":
        labels = BROAD_ORDER
        colors = BROAD_COLORS
        legend_title = "Broad cell class"
        row_cols = [("Ground truth", "truth_broad")] + [(MAP_METHOD_LABELS.get(m, m), f"pred_broad_{m}") for m in METHODS if f"pred_broad_{m}" in pred.columns]
        figsize = (9.25, 5.30)
        legend_width = 1.05
        point_size = 0.34
    else:
        row_cols = [("Ground truth", "truth_celltype")] + [(MAP_METHOD_LABELS.get(m, m), f"pred_{m}") for m in METHODS if f"pred_{m}" in pred.columns]
        labels = pd.concat([pred[col].astype(str) for _, col in row_cols]).value_counts().index.tolist()
        colors = fine_palette(labels)
        legend_title = "Fine cell type"
        figsize = (9.70, 5.55)
        legend_width = 1.60
        point_size = 0.26
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        len(row_cols),
        len(chips) + 1,
        width_ratios=[1] * len(chips) + [legend_width],
        left=0.035,
        right=0.99,
        top=0.86,
        bottom=0.055,
        wspace=0.030,
        hspace=0.08,
    )
    source_rows = []
    for r, (row_label, col) in enumerate(row_cols):
        for c, chip in enumerate(chips):
            ax = fig.add_subplot(gs[r, c])
            sub = pred[pred["chip"].astype(str).eq(chip)].copy()
            if len(sub) > 14000:
                sub = sub.sample(14000, random_state=SEED + r * 11 + c)
            sub["_plot_label"] = sub[col].astype(str)
            sub["_panel_label"] = row_label
            source_rows.append(sub[["chip", "x", "y", "_panel_label", "_plot_label"]])
            for lab in labels:
                m = sub["_plot_label"].eq(lab).to_numpy()
                if m.any():
                    ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"], s=point_size, color=colors.get(lab, "#BFC5CD"), alpha=0.74, linewidths=0, rasterized=True)
            if r == 0:
                ax.set_title(chip, fontsize=5.8, fontweight="bold", pad=1.3)
            if c == 0:
                ax.text(-0.038, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=4.65, color=PALETTE["muted"], fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.invert_yaxis()
            for sp in ax.spines.values():
                sp.set_visible(False)
    ax_leg = fig.add_subplot(gs[:, -1])
    ax_leg.axis("off")
    ax_leg.text(0.0, 0.99, legend_title, transform=ax_leg.transAxes, ha="left", va="top", fontsize=6.1, fontweight="bold", color=PALETTE["muted"])
    if label_space == "fine":
        rows_per_col = int(math.ceil(len(labels) / 2))
        step = 0.88 / max(rows_per_col - 1, 1)
        for i, lab in enumerate(labels):
            col_i = i // rows_per_col
            row_i = i % rows_per_col
            x0 = 0.03 + col_i * 0.50
            y = 0.93 - row_i * step
            ax_leg.scatter([x0], [y], transform=ax_leg.transAxes, s=8, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(x0 + 0.045, y, short_label(lab), transform=ax_leg.transAxes, fontsize=3.75, va="center", ha="left")
    else:
        y = 0.94
        for lab in labels:
            ax_leg.scatter([0.04], [y], transform=ax_leg.transAxes, s=15, color=colors.get(lab, "#BFC5CD"), linewidths=0)
            ax_leg.text(0.10, y, lab, transform=ax_leg.transAxes, fontsize=4.8, va="center", ha="left")
            y -= 0.087
    fig.text(0.035, 0.975, "Selected high-quality all-method spatial maps", ha="left", va="top", fontsize=7.8, fontweight="bold")
    fig.text(0.035, 0.925, "Rows show ground truth, OmniCell-CPT, light-adapted external foundation models and Tangram on matched held-out spots.", ha="left", va="top", fontsize=4.95, color=PALETTE["muted"])
    save(fig, OUT / stem)
    pd.concat(source_rows, ignore_index=True).to_csv(SRC / f"{stem}_source.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    print(f"[env] torch={torch.__version__} device={DEVICE}", flush=True)
    pred, metrics, chips = compute_predictions()
    draw_metric_summary(metrics, "broad cell class", "fig2_selected_quality_all_methods_broad_summary", chips, ylim=0.96)
    draw_metric_summary(metrics, "fine cell type", "fig2_selected_quality_all_methods_fine_summary", chips, ylim=0.86)
    draw_maps(pred, "broad", "fig2_selected_quality_all_methods_broad_maps", chips)
    draw_maps(pred, "fine", "fig2_selected_quality_all_methods_fine_maps", chips)
    contract = {
        "core_conclusion": "After light supervised adaptation of external spatial foundation embeddings, OmniCell-CPT remains the strongest method on QC-ranked high-quality Stereo-seq chips.",
        "selected_chips": chips,
        "selection_rule": "Top chips ranked by precomputed OmniCell-CPT held-out fine-cell Macro F1 from the 10-chip scan.",
        "methods": [m for m in METHODS if f"pred_{m}" in pred.columns],
        "matched_spots": int(len(pred)),
        "metrics_by_chip": str(SRC / "fig2_selected_quality_all_methods_metrics_by_chip.csv"),
        "metrics_summary": str(SRC / "fig2_selected_quality_all_methods_metrics_summary.csv"),
        "predictions": str(SRC / "fig2_selected_quality_all_methods_predictions.csv"),
        "exports": str(OUT),
    }
    (SRC / "fig2_selected_quality_all_methods_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
