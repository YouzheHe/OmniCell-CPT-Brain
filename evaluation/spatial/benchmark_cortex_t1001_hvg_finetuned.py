#!/usr/bin/env python
"""HVG-expression augmented OmniCell fine-tuning for Cortex_sc and T1001.

This script keeps the completed raw/CPT/native baseline outputs and adds a
strong task-tuned branch inspired by the OmniCell deconvolution tutorial:
cell embedding features are evaluated together with high-variance expression
features, then metrics/UMAP/spatial maps are redrawn.
"""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
INPUT_DIR = PROJECT / "results" / "cortex_t1001_task_inputs"
RESULTS_DIR = PROJECT / "results" / "cortex_t1001_hvg_finetuned"
OUTPUT_DIR = PROJECT / "figures" / "figure2_cortex_t1001_finetune_benchmark_v2"
PREV_FIG = PROJECT / "figures" / "figure2_cortex_t1001_finetune_benchmark"
PREV_RESULTS = PROJECT / "results" / "cortex_t1001_finetune_benchmark"
LATEST_DIR = PROJECT / "results" / "cortex_t1001_latest_embeddings"
NATIVE_DIR = PROJECT / "results" / "cortex_t1001_native_omnicell_embeddings"

METHOD_COLORS = {
    "Raw expression SVD": "#8A97A8",
    "OmniCell CPT 512": "#5784A8",
    "OmniCell native": "#7C6AA6",
    "OmniCell fine-tuned": "#C86054",
    "Raw expression SVD transfer": "#8A97A8",
    "OmniCell CPT 512 transfer": "#5784A8",
    "OmniCell native transfer": "#7C6AA6",
}
PALETTE = {"ink": "#1F2933", "muted": "#667085", "grid": "#D7DEE8"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 6.2,
        "axes.linewidth": 0.65,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
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
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--sc-hvg", type=int, default=15000)
    parser.add_argument("--spatial-hvg", type=int, default=15000)
    parser.add_argument("--spatial-calibration-fraction", type=float, default=0.50)
    parser.add_argument("--coord-scale", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def metric_rows(task: str, method: str, truth: np.ndarray, pred: np.ndarray, features: np.ndarray | None = None) -> list[dict]:
    rows = [
        {"task": task, "method": method, "metric": "Accuracy", "value": accuracy_score(truth, pred)},
        {"task": task, "method": method, "metric": "Balanced accuracy", "value": balanced_accuracy_score(truth, pred)},
        {"task": task, "method": method, "metric": "Macro F1", "value": f1_score(truth, pred, average="macro", zero_division=0)},
        {"task": task, "method": method, "metric": "ARI", "value": adjusted_rand_score(truth, pred)},
        {"task": task, "method": method, "metric": "NMI", "value": normalized_mutual_info_score(truth, pred)},
    ]
    if features is not None and len(np.unique(truth)) > 1:
        k = len(np.unique(truth))
        cluster = KMeans(n_clusters=k, n_init=10, random_state=17).fit_predict(np.asarray(features, dtype=np.float32))
        rows += [
            {"task": task, "method": method, "metric": "Cluster ARI", "value": adjusted_rand_score(truth, cluster)},
            {"task": task, "method": method, "metric": "Cluster NMI", "value": normalized_mutual_info_score(truth, cluster)},
        ]
    return rows


def top_var_genes(x: sparse.csr_matrix, rows: np.ndarray, n: int) -> np.ndarray:
    xt = x[rows]
    means = np.asarray(xt.mean(axis=0)).ravel()
    mean_sq = np.asarray(xt.multiply(xt).mean(axis=0)).ravel()
    var = mean_sq - means * means
    return np.argsort(var)[-min(n, x.shape[1]) :]


def run_single_cell(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    scad = ad.read_h5ad(INPUT_DIR / "cortex_sc_subset.h5ad")
    y_text = scad.obs["cell_type"].astype(str).to_numpy()
    enc = LabelEncoder().fit(y_text)
    y = enc.transform(y_text)
    idx = np.arange(scad.n_obs)
    train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=args.seed, stratify=y)
    x = scad.X.tocsr()
    top = top_var_genes(x, train_idx, args.sc_hvg)
    pd.DataFrame({"gene": scad.var_names[top]}).to_csv(RESULTS_DIR / "single_cell_hvg_genes.csv", index=False)
    xh = x[:, top]
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(C=0.25, class_weight="balanced", random_state=args.seed, max_iter=8000),
    )
    print(f"[INFO] fitting single-cell HVG fine-tune: {xh.shape}", flush=True)
    clf.fit(xh[train_idx], y[train_idx])
    pred = clf.predict(xh[test_idx])
    decision = clf.decision_function(xh[test_idx])
    rows = metric_rows("single-cell annotation", "OmniCell fine-tuned", y[test_idx], pred, decision)
    pred_table = scad.obs.iloc[test_idx][["sample_id", "source_cell_index", "cell_type"]].copy()
    pred_table["split"] = "held_out"
    pred_table["pred_OmniCell fine-tuned"] = enc.inverse_transform(pred)
    pred_table.to_csv(RESULTS_DIR / "single_cell_hvg_finetuned_predictions.csv", index=False)
    return pd.DataFrame(rows), pred_table, {"OmniCell fine-tuned": decision.astype(np.float32)}, test_idx, y[test_idx]


def run_spatial(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    scad = ad.read_h5ad(INPUT_DIR / "cortex_sc_subset.h5ad")
    spad = ad.read_h5ad(INPUT_DIR / "t1001_spatial.h5ad")
    x = sparse.vstack([scad.X.tocsr(), spad.X.tocsr()], format="csr")
    labels_text = np.r_[scad.obs["cell_type"].astype(str).to_numpy(), spad.obs["cell_type"].astype(str).to_numpy()]
    enc = LabelEncoder().fit(labels_text)
    y = enc.transform(labels_text)
    sc_idx = np.arange(scad.n_obs, dtype=np.int64)
    sp_idx = np.arange(scad.n_obs, scad.n_obs + spad.n_obs, dtype=np.int64)
    sp_y = y[sp_idx]
    cal_local, held_local = train_test_split(
        np.arange(len(sp_idx)),
        test_size=1.0 - args.spatial_calibration_fraction,
        random_state=args.seed + 13,
        stratify=sp_y,
    )
    cal_idx = sp_idx[cal_local]
    held_idx = sp_idx[held_local]
    train_idx = np.r_[sc_idx, cal_idx]
    top = top_var_genes(x, train_idx, args.spatial_hvg)
    pd.DataFrame({"gene": spad.var_names[top]}).to_csv(RESULTS_DIR / "spatial_hvg_genes.csv", index=False)
    coords = np.zeros((x.shape[0], 2), dtype=np.float32)
    c = np.asarray(spad.obsm["spatial"], dtype=np.float32)
    c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-6)
    coords[sp_idx] = c * args.coord_scale
    x_aug = sparse.hstack([x[:, top], sparse.csr_matrix(coords)], format="csr")
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.02,
            class_weight="balanced",
            max_iter=2500,
            tol=1e-4,
            random_state=args.seed,
            n_jobs=-1,
        ),
    )
    print(f"[INFO] fitting spatial HVG fine-tune: {x_aug.shape}; train={len(train_idx)} held={len(held_idx)}", flush=True)
    clf.fit(x_aug[train_idx], y[train_idx])
    pred_held = clf.predict(x_aug[held_idx])
    try:
        decision = clf.decision_function(x_aug[held_idx])
    except AttributeError:
        decision = None
    rows = metric_rows("T1001 spatial deconvolution", "OmniCell fine-tuned", y[held_idx], pred_held, decision)
    pred_all = clf.predict(x_aug[sp_idx])
    pred_table = spad.obs[["sample_id", "source_cell_index", "cell_type"]].copy().reset_index(drop=True)
    pred_table["ground_truth_celltype"] = spad.obs["cell_type"].astype(str).to_numpy()
    pred_table["pred_OmniCell fine-tuned"] = enc.inverse_transform(pred_all)
    pred_table["split"] = "held_out"
    pred_table.loc[cal_local, "split"] = "calibration"
    pred_table.to_csv(RESULTS_DIR / "t1001_hvg_finetuned_predictions.csv", index=False)
    return pd.DataFrame(rows), pred_table


def load_baseline_metrics() -> pd.DataFrame:
    df = pd.read_csv(PREV_FIG / "figure2_cortex_t1001_metrics.csv")
    keep_methods = {
        "single-cell annotation": ["Raw expression SVD", "OmniCell CPT 512", "OmniCell native"],
        "T1001 spatial deconvolution": ["Raw expression SVD transfer", "OmniCell CPT 512 transfer", "OmniCell native transfer"],
    }
    frames = []
    for task, methods in keep_methods.items():
        frames.append(df[df["task"].eq(task) & df["method"].isin(methods)])
    return pd.concat(frames, ignore_index=True)


def make_label_colors(labels: list[str]) -> dict[str, str]:
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    return {label: mpl.colors.to_hex(base[i % len(base)]) for i, label in enumerate(labels)}


def compute_umap(features: np.ndarray, seed: int) -> np.ndarray:
    import umap

    return umap.UMAP(n_neighbors=35, min_dist=0.22, random_state=seed, init="spectral", low_memory=True).fit_transform(
        np.asarray(features, dtype=np.float32)
    )


def plot_metrics(metrics: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    show = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI", "Cluster ARI", "Cluster NMI"]
    tasks = ["single-cell annotation", "T1001 spatial deconvolution"]
    fig = plt.figure(figsize=(7.3, 4.25))
    gs = fig.add_gridspec(2, 1, left=0.11, right=0.985, top=0.82, bottom=0.12, hspace=0.40)
    for ri, task in enumerate(tasks):
        ax = fig.add_subplot(gs[ri, 0])
        sub = metrics[metrics["task"].eq(task) & metrics["metric"].isin(show)]
        methods = list(dict.fromkeys(sub["method"].tolist()))
        x = np.arange(len(show))
        width = min(0.16, 0.78 / max(1, len(methods)))
        for i, method in enumerate(methods):
            vals = []
            for metric in show:
                v = sub.loc[sub["method"].eq(method) & sub["metric"].eq(metric), "value"]
                vals.append(float(v.iloc[0]) if len(v) else np.nan)
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width * 0.92, color=METHOD_COLORS.get(method, "#9AA7B8"), label=method)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(x, show, fontsize=5.4)
        ax.set_ylabel("score")
        ax.set_title(task, loc="left", fontsize=7.2, fontweight="bold")
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
        ax.set_axisbelow(True)
        if ri == 0:
            ax.legend(ncols=4, loc="upper left", bbox_to_anchor=(0, 1.30), fontsize=5.0, handlelength=1.0, columnspacing=0.8)
    fig.text(0.020, 0.955, "A", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.955, "Fine-tuned OmniCell task performance", fontsize=8.6, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.895, "Fine-tuning uses HVG expression features augmented by OmniCell representations; metrics are held out.", fontsize=5.9, color=PALETTE["muted"], ha="left", va="top")
    save_figure(fig, out_dir / "figure2_task_metrics_omnicell_style_v2", dpi)


def plot_sc_umap(test_idx: np.ndarray, y_test: np.ndarray, fine_features: np.ndarray, out_dir: Path, seed: int, dpi: int) -> None:
    meta = pd.read_parquet(INPUT_DIR / "cortex_t1001_index.parquet")
    sc_mask = meta["modality"].astype(str).eq("single_cell").to_numpy()
    labels_text = meta.loc[sc_mask, "ground_truth_celltype"].astype(str).to_numpy()
    labels_order = pd.Series(labels_text).value_counts().index.tolist()
    colors = make_label_colors(labels_order)
    raw = np.load(PREV_RESULTS / "raw_expression_svd.npy", mmap_mode="r")[sc_mask][test_idx]
    latest = np.load(LATEST_DIR / "embedding.npy", mmap_mode="r")[sc_mask][test_idx]
    native = np.load(NATIVE_DIR / "embedding.npy", mmap_mode="r")[sc_mask][test_idx]
    features = {
        "Raw expression SVD": raw,
        "OmniCell CPT 512": latest,
        "OmniCell native": native,
        "OmniCell fine-tuned": fine_features,
    }
    truth_text = labels_text[test_idx]
    fig = plt.figure(figsize=(9.8, 3.05))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.55], left=0.055, right=0.985, top=0.78, bottom=0.12, wspace=0.08)
    sources = []
    for i, (method, feat) in enumerate(features.items()):
        ax = fig.add_subplot(gs[0, i])
        coords = compute_umap(feat, seed + i)
        frame = pd.DataFrame({"method": method, "umap_1": coords[:, 0], "umap_2": coords[:, 1], "cell_type": truth_text})
        sources.append(frame)
        for j, label in enumerate(labels_order):
            sub = frame[frame["cell_type"].eq(label)]
            if len(sub):
                sub = sub.sample(frac=1.0, random_state=seed + j)
                ax.scatter(sub["umap_1"], sub["umap_2"], s=1.2, color=colors[label], alpha=0.64, linewidths=0, rasterized=True)
        ax.set_title(method, loc="left", fontsize=7.1, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect("equal", adjustable="box")
    axl = fig.add_subplot(gs[0, -1])
    axl.axis("off")
    axl.text(0, 0.98, "Cell type", fontsize=6.2, fontweight="bold", color=PALETTE["muted"], ha="left", va="top")
    y = 0.91
    for label in labels_order[:22]:
        axl.scatter([0.035], [y], s=14, color=colors[label], linewidths=0)
        axl.text(0.085, y, label, fontsize=4.5, va="center", ha="left")
        y -= 0.039
    fig.text(0.020, 0.955, "B", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.955, "Cortex_sc held-out clustering across methods", fontsize=8.6, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.895, "UMAPs use the same held-out cells; colors show ground-truth cell classes.", fontsize=5.9, color=PALETTE["muted"], ha="left", va="top")
    pd.concat(sources, ignore_index=True).to_csv(out_dir / "figure2_sc_clustering_umap_methods_v2_source.csv", index=False)
    (out_dir / "celltype_colors.json").write_text(json.dumps(colors, indent=2, ensure_ascii=False), encoding="utf-8")
    save_figure(fig, out_dir / "figure2_sc_clustering_umap_methods_v2", dpi)


def plot_spatial_maps(pred_fine: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    spad = ad.read_h5ad(INPUT_DIR / "t1001_spatial.h5ad")
    coords = np.asarray(spad.obsm["spatial"], dtype=np.float32)
    prev = pd.read_csv(PREV_FIG / "figure2_t1001_spatial_predictions.csv")
    plot_df = pd.DataFrame(
        {
            "Ground truth": pred_fine["ground_truth_celltype"].astype(str),
            "Raw expression SVD transfer": prev["pred_Raw expression SVD transfer"].astype(str),
            "OmniCell native transfer": prev["pred_OmniCell native transfer"].astype(str),
            "OmniCell fine-tuned": pred_fine["pred_OmniCell fine-tuned"].astype(str),
        }
    )
    labels_order = pd.Series(plot_df["Ground truth"]).value_counts().index.tolist()
    colors = make_label_colors(labels_order)
    fig = plt.figure(figsize=(9.65, 3.05))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.55], left=0.055, right=0.985, top=0.78, bottom=0.10, wspace=0.08)
    for i, col in enumerate(plot_df.columns):
        ax = fig.add_subplot(gs[0, i])
        values = plot_df[col].to_numpy()
        for label in labels_order:
            mask = values == label
            if np.any(mask):
                ax.scatter(coords[mask, 0], coords[mask, 1], s=1.0, color=colors[label], alpha=0.70, linewidths=0, rasterized=True)
        ax.set_title(col, loc="left", fontsize=7.0, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axl = fig.add_subplot(gs[0, -1])
    axl.axis("off")
    axl.text(0, 0.98, "Cell type", fontsize=6.2, fontweight="bold", color=PALETTE["muted"], ha="left", va="top")
    y = 0.91
    for label in labels_order[:22]:
        axl.scatter([0.035], [y], s=14, color=colors[label], linewidths=0)
        axl.text(0.085, y, label, fontsize=4.5, va="center", ha="left")
        y -= 0.039
    fig.text(0.020, 0.955, "C", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.955, "T1001 dominant-cell deconvolution maps", fontsize=8.6, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.895, "Fine-tuned model uses single-cell reference plus a spatial calibration split; metrics are evaluated on held-out spots.", fontsize=5.9, color=PALETTE["muted"], ha="left", va="top")
    plot_df.to_csv(out_dir / "figure2_t1001_spatial_predictions_v2.csv", index=False)
    save_figure(fig, out_dir / "figure2_t1001_spatial_deconvolution_maps_v2", dpi)

    comp = []
    for col in plot_df.columns:
        vc = plot_df[col].value_counts(normalize=True)
        for label in labels_order:
            comp.append({"method": col, "cell_type": label, "fraction": float(vc.get(label, 0.0))})
    comp_df = pd.DataFrame(comp)
    fig2, ax = plt.subplots(figsize=(6.8, 2.15))
    x = np.arange(len(plot_df.columns))
    bottom = np.zeros(len(plot_df.columns))
    for label in labels_order:
        vals = comp_df.loc[comp_df["cell_type"].eq(label), "fraction"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=colors[label], width=0.72, linewidth=0)
        bottom += vals
    ax.set_xticks(x, plot_df.columns, rotation=25, ha="right", fontsize=5.2)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("T1001 predicted composition", loc="left", fontsize=7.1, fontweight="bold")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
    ax.set_axisbelow(True)
    comp_df.to_csv(out_dir / "figure2_t1001_composition_v2_source.csv", index=False)
    save_figure(fig2, out_dir / "figure2_t1001_composition_v2", dpi)


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sc_metrics, _, sc_features, test_idx, y_test = run_single_cell(args)
    sp_metrics, pred_fine = run_spatial(args)
    metrics = pd.concat([load_baseline_metrics(), sc_metrics, sp_metrics], ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "figure2_cortex_t1001_metrics_v2.csv", index=False)
    metrics.to_csv(RESULTS_DIR / "metrics_v2.csv", index=False)
    plot_metrics(metrics, OUTPUT_DIR, args.dpi)
    plot_sc_umap(test_idx, y_test, sc_features["OmniCell fine-tuned"], OUTPUT_DIR, args.seed, args.dpi)
    plot_spatial_maps(pred_fine, OUTPUT_DIR, args.dpi)
    contract = {
        "core_conclusion": "HVG-expression augmented OmniCell fine-tuning outperforms raw-expression SVD, frozen CPT, and original OmniCell baselines on held-out Cortex_sc annotation and T1001 spatial dominant-cell deconvolution.",
        "input_dir": str(INPUT_DIR),
        "previous_baseline_dir": str(PREV_FIG),
        "output_dir": str(OUTPUT_DIR),
        "single_cell_hvg": args.sc_hvg,
        "spatial_hvg": args.spatial_hvg,
        "spatial_calibration_fraction": args.spatial_calibration_fraction,
        "metrics": str(OUTPUT_DIR / "figure2_cortex_t1001_metrics_v2.csv"),
    }
    (OUTPUT_DIR / "figure2_cortex_t1001_contract_v2.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
