#!/usr/bin/env python
"""Scan HVG sizes on three random Cortex Stereo-seq chips."""

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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
DATASET_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/NVU_hyz"))
SC_H5AD = PROJECT / "results" / "cortex_t1001_task_inputs" / "cortex_sc_subset.h5ad"
RESULTS_DIR = PROJECT / "results" / "random_stereo_hvg_scan"
OUTPUT_DIR = PROJECT / "figures" / "figure2_random_stereo_hvg_scan"

PALETTE = {"ink": "#1F2933", "muted": "#667085", "grid": "#D7DEE8", "signal": "#C86054"}
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
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chips", default="T765,T906,T1008")
    p.add_argument("--hvg-sizes", default="3000,5000")
    p.add_argument("--calibration-fraction", type=float, default=0.50)
    p.add_argument("--coord-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--dpi", type=int, default=900)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def extract_rows_csr(sample_dir: Path, rows: np.ndarray, n_genes: int) -> sparse.csr_matrix:
    indptr = np.load(sample_dir / "indptr.npy", mmap_mode="r")
    indices = np.load(sample_dir / "indices.npy", mmap_mode="r")
    values = np.load(sample_dir / "values.npy", mmap_mode="r")
    nnz = int(np.sum(indptr[rows + 1] - indptr[rows]))
    out_indptr = np.empty(len(rows) + 1, dtype=np.int64)
    out_indices = np.empty(nnz, dtype=np.int32)
    out_values = np.empty(nnz, dtype=np.float32)
    out_indptr[0] = 0
    cur = 0
    for i, row in enumerate(rows):
        s = int(indptr[row])
        e = int(indptr[row + 1])
        w = e - s
        out_indices[cur : cur + w] = indices[s:e]
        out_values[cur : cur + w] = values[s:e]
        cur += w
        out_indptr[i + 1] = cur
    return sparse.csr_matrix((out_values, out_indices, out_indptr), shape=(len(rows), n_genes))


def spatial_h5ad_for_chip(chip: str, force: bool = False) -> Path:
    out = RESULTS_DIR / "spatial_h5ad" / f"{chip}.h5ad"
    if out.exists() and not force:
        return out
    sample_id = f"Cortex_Spatial/{chip}"
    sample_dir = DATASET_ROOT / sample_id
    obs = pd.read_parquet(sample_dir / "obs.parquet").reset_index(drop=True)
    obs["source_cell_index"] = np.arange(len(obs), dtype=np.int64)
    obs["sample_id"] = sample_id
    obs["modality"] = "spatial"
    obs["cell_type"] = obs["CellType_m"].astype("object").where(pd.notna(obs["CellType_m"]), "").astype(str)
    if "obs_name" in obs:
        obs.index = obs["obs_name"].astype(str).to_numpy()
    rows = np.arange(len(obs), dtype=np.int64)
    genes = (DATASET_ROOT / "gene_vocab.txt").read_text(encoding="utf-8").splitlines()
    x = extract_rows_csr(sample_dir, rows, len(genes))
    coords = np.load(sample_dir / "coords.npy", mmap_mode="r")
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame({"feature_name": genes}, index=genes))
    adata.raw = adata
    adata.obsm["spatial"] = np.asarray(coords, dtype=np.float32)
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")
    return out


def top_var_genes(x: sparse.csr_matrix, rows: np.ndarray, n: int) -> np.ndarray:
    xt = x[rows]
    means = np.asarray(xt.mean(axis=0)).ravel()
    mean_sq = np.asarray(xt.multiply(xt).mean(axis=0)).ravel()
    var = mean_sq - means * means
    return np.argsort(var)[-min(n, x.shape[1]) :]


def evaluate_chip(scad: ad.AnnData, spad: ad.AnnData, chip: str, hvg_size: int, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    x = sparse.vstack([scad.X.tocsr(), spad.X.tocsr()], format="csr")
    labels_text = np.r_[scad.obs["cell_type"].astype(str).to_numpy(), spad.obs["cell_type"].astype(str).to_numpy()]
    enc = LabelEncoder().fit(labels_text)
    y = enc.transform(labels_text)
    sc_idx = np.arange(scad.n_obs, dtype=np.int64)
    sp_idx = np.arange(scad.n_obs, scad.n_obs + spad.n_obs, dtype=np.int64)
    sp_y = y[sp_idx]
    cal_local, held_local = train_test_split(
        np.arange(len(sp_idx)),
        test_size=1.0 - args.calibration_fraction,
        random_state=args.seed + hvg_size + sum(ord(c) for c in chip),
        stratify=sp_y,
    )
    cal_idx = sp_idx[cal_local]
    held_idx = sp_idx[held_local]
    train_idx = np.r_[sc_idx, cal_idx]
    top = top_var_genes(x, train_idx, hvg_size)
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
    clf.fit(x_aug[train_idx], y[train_idx])
    pred_held = clf.predict(x_aug[held_idx])
    pred_all = clf.predict(x_aug[sp_idx])
    metrics = {
        "chip": chip,
        "n_spots": int(spad.n_obs),
        "region": str(spad.obs.get("region1", pd.Series([""])).astype(str).mode().iloc[0]) if "region1" in spad.obs else "",
        "age": str(spad.obs.get("age", pd.Series([""])).astype(str).mode().iloc[0]) if "age" in spad.obs else "",
        "hvg_size": int(hvg_size),
        "calibration_fraction": float(args.calibration_fraction),
        "n_train": int(len(train_idx)),
        "n_heldout": int(len(held_idx)),
        "accuracy": accuracy_score(y[held_idx], pred_held),
        "balanced_accuracy": balanced_accuracy_score(y[held_idx], pred_held),
        "macro_f1": f1_score(y[held_idx], pred_held, average="macro", zero_division=0),
        "ari": adjusted_rand_score(y[held_idx], pred_held),
        "nmi": normalized_mutual_info_score(y[held_idx], pred_held),
    }
    pred_df = pd.DataFrame(
        {
            "chip": chip,
            "hvg_size": hvg_size,
            "x": spad.obsm["spatial"][:, 0],
            "y": spad.obsm["spatial"][:, 1],
            "ground_truth_celltype": spad.obs["cell_type"].astype(str).to_numpy(),
            "predicted_celltype": enc.inverse_transform(pred_all),
            "split": "held_out",
        }
    )
    pred_df.loc[cal_local, "split"] = "calibration"
    return metrics, pred_df


def make_label_colors(labels: list[str]) -> dict[str, str]:
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    return {label: mpl.colors.to_hex(base[i % len(base)]) for i, label in enumerate(labels)}


def plot_scan(metrics: pd.DataFrame, pred_best: pd.DataFrame, args: argparse.Namespace) -> None:
    chips = metrics["chip"].drop_duplicates().tolist()
    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 3, left=0.10, right=0.98, top=0.78, bottom=0.19, wspace=0.34)
    for i, metric in enumerate(["accuracy", "macro_f1", "nmi"]):
        ax = fig.add_subplot(gs[0, i])
        for hvg, sub in metrics.groupby("hvg_size"):
            sub = sub.set_index("chip").loc[chips].reset_index()
            ax.plot(np.arange(len(chips)), sub[metric], marker="o", lw=1.2, ms=3.5, label=f"{hvg:,} HVG")
        ax.set_xticks(np.arange(len(chips)), chips)
        ax.set_ylim(0, max(0.75, metrics[metric].max() * 1.15))
        ax.set_title(metric.replace("_", " ").title(), loc="left", fontsize=7.1, fontweight="bold")
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(fontsize=5.4, loc="upper left")
    fig.text(0.020, 0.955, "A", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.060, 0.955, "Random Cortex Stereo-seq chip scan", fontsize=8.5, fontweight="bold", ha="left", va="top")
    fig.text(0.060, 0.895, "Held-out dominant cell-type deconvolution after spatial calibration; three chips were selected by a fixed random seed.", fontsize=5.8, color=PALETTE["muted"], ha="left", va="top")
    save_figure(fig, OUTPUT_DIR / "random_stereo_hvg_scan_metrics", args.dpi)

    labels = pd.Series(pred_best["ground_truth_celltype"]).value_counts().index.tolist()
    colors = make_label_colors(labels)
    fig2 = plt.figure(figsize=(7.35, 4.75))
    gs2 = fig2.add_gridspec(2, len(chips), left=0.035, right=0.84, top=0.83, bottom=0.08, wspace=0.06, hspace=0.12)
    for col, chip in enumerate(chips):
        sub = pred_best[pred_best["chip"].eq(chip)]
        for row, field in enumerate(["ground_truth_celltype", "predicted_celltype"]):
            ax = fig2.add_subplot(gs2[row, col])
            values = sub[field].astype(str).to_numpy()
            for label in labels:
                m = values == label
                if np.any(m):
                    ax.scatter(sub.loc[m, "x"], sub.loc[m, "y"], s=0.8, color=colors[label], alpha=0.72, linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(chip, fontsize=7.1, fontweight="bold", pad=2)
            if col == 0:
                ax.text(-0.03, 0.5, "Ground truth" if row == 0 else "Fine-tuned", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.2, fontweight="bold", color=PALETTE["muted"])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axl = fig2.add_axes([0.86, 0.12, 0.13, 0.70])
    axl.axis("off")
    axl.text(0, 1, "Cell type", fontsize=6.2, fontweight="bold", color=PALETTE["muted"], ha="left", va="top")
    y = 0.93
    for label in labels[:18]:
        axl.scatter([0.04], [y], s=13, color=colors[label], linewidths=0)
        axl.text(0.10, y, label, fontsize=4.5, va="center", ha="left")
        y -= 0.049
    fig2.text(0.020, 0.965, "B", fontsize=11, fontweight="bold", ha="left", va="top")
    fig2.text(0.060, 0.965, "Spatial maps for the best HVG setting per chip", fontsize=8.5, fontweight="bold", ha="left", va="top")
    save_figure(fig2, OUTPUT_DIR / "random_stereo_best_spatial_maps", args.dpi)


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
    chips = [x.strip() for x in args.chips.split(",") if x.strip()]
    hvgs = [int(x) for x in args.hvg_sizes.split(",") if x.strip()]
    scad = ad.read_h5ad(SC_H5AD)
    metrics = []
    predictions = []
    for chip in chips:
        spad = ad.read_h5ad(spatial_h5ad_for_chip(chip, force=args.force))
        for hvg in hvgs:
            print(f"[INFO] evaluating {chip} with {hvg} HVGs", flush=True)
            row, pred = evaluate_chip(scad, spad, chip, hvg, args)
            metrics.append(row)
            predictions.append(pred)
            pd.DataFrame(metrics).to_csv(RESULTS_DIR / "random_stereo_hvg_scan_metrics.csv", index=False)
    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_csv(RESULTS_DIR / "random_stereo_hvg_scan_predictions.csv", index=False)
    best = metrics_df.sort_values(["chip", "accuracy"], ascending=[True, False]).groupby("chip", as_index=False).head(1)
    pred_best = pred_df.merge(best[["chip", "hvg_size"]], on=["chip", "hvg_size"], how="inner")
    metrics_df.to_csv(OUTPUT_DIR / "random_stereo_hvg_scan_metrics.csv", index=False)
    pred_best.to_csv(OUTPUT_DIR / "random_stereo_best_predictions.csv", index=False)
    plot_scan(metrics_df, pred_best, args)
    contract = {
        "chips": chips,
        "hvg_sizes": hvgs,
        "calibration_fraction": args.calibration_fraction,
        "metrics": str(OUTPUT_DIR / "random_stereo_hvg_scan_metrics.csv"),
        "figures": ["random_stereo_hvg_scan_metrics.*", "random_stereo_best_spatial_maps.*"],
    }
    (OUTPUT_DIR / "random_stereo_hvg_scan_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
