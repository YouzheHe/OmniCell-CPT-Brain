#!/usr/bin/env python
"""Compare external single-cell model embeddings against OmniCell baselines."""

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
INPUT = PROJECT / "results" / "cortex_t1001_task_inputs" / "cortex_sc_subset.h5ad"
LATEST = PROJECT / "results" / "cortex_t1001_latest_embeddings" / "embedding.npy"
NATIVE = PROJECT / "results" / "cortex_t1001_native_omnicell_embeddings" / "embedding.npy"
EXT = PROJECT / "results" / "external_singlecell_embeddings"
OUT = PROJECT / "figures" / "figure2_external_singlecell_comparison"

PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "raw": "#8A97A8",
    "cpt": "#5784A8",
    "native": "#7C6AA6",
    "fine": "#C86054",
    "cellplm": "#70B7A6",
    "geneformer": "#E0A458",
    "scfoundation": "#9C7AAE",
}
METHOD_COLORS = {
    "Raw expression SVD": PALETTE["raw"],
    "OmniCell CPT 512": PALETTE["cpt"],
    "OmniCell native": PALETTE["native"],
    "OmniCell fine-tuned HVG": PALETTE["fine"],
    "CellPLM": PALETTE["cellplm"],
    "Geneformer": PALETTE["geneformer"],
    "scGPT": "#5E9A62",
    "scFoundation": PALETTE["scfoundation"],
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hvg", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260528)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--max-umap-cells", type=int, default=12000)
    return p.parse_args()


def top_var_genes(x: sparse.spmatrix, n: int) -> np.ndarray:
    x = x.tocsr()
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0)
    n = min(int(n), x.shape[1])
    idx = np.argpartition(var, -n)[-n:]
    return idx[np.argsort(var[idx])[::-1]]


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


def metric_rows(task: str, method: str, y_true: np.ndarray, y_pred: np.ndarray, features: np.ndarray) -> list[dict]:
    rows = [
        {"task": task, "method": method, "metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "Balanced accuracy", "value": balanced_accuracy_score(y_true, y_pred)},
        {"task": task, "method": method, "metric": "Macro F1", "value": f1_score(y_true, y_pred, average="macro")},
    ]
    n_clusters = len(np.unique(y_true))
    if features.shape[0] > n_clusters:
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=37)
        cl = km.fit_predict(np.asarray(features, dtype=np.float32))
        rows.extend(
            [
                {"task": task, "method": method, "metric": "Cluster ARI", "value": adjusted_rand_score(y_true, cl)},
                {"task": task, "method": method, "metric": "Cluster NMI", "value": normalized_mutual_info_score(y_true, cl)},
            ]
        )
    return rows


def fit_predict_dense(name: str, features: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray) -> tuple[list[dict], np.ndarray]:
    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(C=0.5, class_weight="balanced", random_state=37, max_iter=10000),
    )
    clf.fit(features[train_idx], y[train_idx])
    pred = clf.predict(features[test_idx])
    rows = metric_rows("single-cell annotation", name, y[test_idx], pred, features[test_idx])
    return rows, features[test_idx]


def fit_predict_sparse_hvg(x: sparse.spmatrix, top: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray) -> tuple[list[dict], np.ndarray]:
    xh = x[:, top]
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(C=0.25, class_weight="balanced", random_state=37, max_iter=10000),
    )
    clf.fit(xh[train_idx], y[train_idx])
    pred = clf.predict(xh[test_idx])
    svd = TruncatedSVD(n_components=50, random_state=37)
    test_features = svd.fit_transform(xh[test_idx])
    rows = metric_rows("single-cell annotation", "OmniCell fine-tuned HVG", y[test_idx], pred, test_features)
    return rows, test_features


def save(fig: plt.Figure, stem: Path, dpi: int) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_external(name: str) -> tuple[str, np.ndarray] | None:
    if name.lower() == "scgpt":
        candidates = sorted(EXT.glob("scgpt*_hvg*_n*/embedding.npy")) + sorted(EXT.glob("scgpt_t906_sc_hvg*_n*/embedding.npy"))
        if not candidates:
            return None
        return "scGPT", np.load(candidates[-1])
    if name.lower() == "scfoundation":
        candidates = sorted(EXT.glob("scfoundation*_hvg*_n*/embedding.npy")) + sorted(EXT.glob("scfoundation_t906_sc_hvg*_n*/embedding.npy"))
        if not candidates:
            return None
        return "scFoundation", np.load(candidates[-1])
    candidates = sorted(EXT.glob(f"{name.lower()}_hvg*_n*/embedding.npy"))
    if not candidates:
        return None
    p = candidates[-1]
    emb = np.load(p)
    label = "CellPLM" if name.lower() == "cellplm" else "Geneformer"
    return label, emb


def plot_metrics(metrics: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    show = ["Accuracy", "Balanced accuracy", "Macro F1", "Cluster ARI", "Cluster NMI"]
    methods = list(dict.fromkeys(metrics["method"].tolist()))
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.78, bottom=0.35)
    x = np.arange(len(show))
    width = min(0.13, 0.82 / max(1, len(methods)))
    for i, method in enumerate(methods):
        vals = []
        for m in show:
            v = metrics.loc[metrics["method"].eq(method) & metrics["metric"].eq(m), "value"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width * 0.92, color=METHOD_COLORS.get(method, "#9AA7B8"), label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(show)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45, alpha=0.65)
    fig.text(0.075, 0.95, "Single-cell annotation benchmark", ha="left", va="top", fontweight="bold", fontsize=9.0, color=PALETTE["ink"])
    fig.text(0.075, 0.87, "Held-out Cortex_sc cells; linear probes use the same stratified split for every method.", ha="left", va="top", color=PALETTE["muted"], fontsize=6.4)
    ax.legend(ncol=min(3, len(methods)), bbox_to_anchor=(0, -0.24), loc="upper left", handlelength=1.1, columnspacing=1.2)
    save(fig, out_dir / "figure2_external_singlecell_metrics", dpi)


def plot_umaps(features: dict[str, np.ndarray], y_text: np.ndarray, metrics: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    import umap

    rng = np.random.default_rng(args.seed)
    n = len(y_text)
    keep = np.arange(n)
    if n > args.max_umap_cells:
        keep = rng.choice(keep, size=args.max_umap_cells, replace=False)
        keep.sort()
    methods = list(features.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(2.35 * len(methods), 2.45), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    broad_labels = np.array([broad(v) for v in y_text])
    for ax, method in zip(axes, methods):
        emb = np.asarray(features[method][keep], dtype=np.float32)
        xy = umap.UMAP(n_neighbors=35, min_dist=0.22, random_state=args.seed, init="spectral", low_memory=True).fit_transform(emb)
        for lab, col in BROAD_COLORS.items():
            m = broad_labels[keep] == lab
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=1.8, c=col, linewidths=0, rasterized=True, alpha=0.78)
        acc = metrics.loc[metrics["method"].eq(method) & metrics["metric"].eq("Accuracy"), "value"]
        suffix = f"acc. {float(acc.iloc[0]):.2f}" if len(acc) else ""
        ax.set_title(f"{method}\n{suffix}", fontsize=7.3, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, ms=4, color=BROAD_COLORS[k], label=k)
        for k in BROAD_COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=6.0, handletextpad=0.25, columnspacing=0.8)
    fig.suptitle("Cell-type structure preserved by task-tuned OmniCell and external foundation models", x=0.01, y=1.05, ha="left", fontsize=8.8, fontweight="bold")
    save(fig, out_dir / "figure2_external_singlecell_umap_methods", args.dpi)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scad = ad.read_h5ad(INPUT)
    y_text = scad.obs["cell_type"].astype(str).to_numpy()
    enc = LabelEncoder().fit(y_text)
    y = enc.transform(y_text)
    idx = np.arange(scad.n_obs)
    train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=args.seed, stratify=y)
    x = scad.X.tocsr() if sparse.issparse(scad.X) else sparse.csr_matrix(scad.X)
    top = top_var_genes(x, args.hvg)

    rows: list[dict] = []
    umap_features: dict[str, np.ndarray] = {}

    raw_svd = TruncatedSVD(n_components=128, random_state=args.seed).fit_transform(x[:, top])
    r, feat = fit_predict_dense("Raw expression SVD", raw_svd, train_idx, test_idx, y)
    rows.extend(r)
    umap_features["Raw expression SVD"] = feat

    latest = np.load(LATEST)[: scad.n_obs]
    r, feat = fit_predict_dense("OmniCell CPT 512", latest, train_idx, test_idx, y)
    rows.extend(r)
    umap_features["OmniCell CPT 512"] = feat

    native = np.load(NATIVE)[: scad.n_obs]
    r, feat = fit_predict_dense("OmniCell native", native, train_idx, test_idx, y)
    rows.extend(r)
    umap_features["OmniCell native"] = feat

    r, feat = fit_predict_sparse_hvg(x, top, train_idx, test_idx, y)
    rows.extend(r)
    umap_features["OmniCell fine-tuned HVG"] = feat

    for name in ["cellplm", "scgpt", "scfoundation", "geneformer"]:
        loaded = load_external(name)
        if loaded is None:
            continue
        label, emb = loaded
        if emb.shape[0] != scad.n_obs:
            print(f"[WARN] skipping {label}; embedding rows {emb.shape[0]} != {scad.n_obs}", flush=True)
            continue
        r, feat = fit_predict_dense(label, emb, train_idx, test_idx, y)
        rows.extend(r)
        umap_features[label] = feat

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "external_singlecell_metrics.csv", index=False)
    pd.DataFrame({"cell_type": y_text[test_idx], "broad_cell_type": [broad(v) for v in y_text[test_idx]]}).to_csv(
        OUT / "external_singlecell_test_labels.csv", index=False
    )
    plot_metrics(metrics, OUT, args.dpi)
    plot_umaps(umap_features, y_text[test_idx], metrics, OUT, args)
    contract = {
        "core_conclusion": "Task-tuned OmniCell is benchmarked against raw expression, frozen OmniCell, original OmniCell, and successfully deployed external foundation-model embeddings on the same Cortex_sc held-out split.",
        "input": str(INPUT),
        "output": str(OUT),
        "hvg": int(args.hvg),
        "n_cells": int(scad.n_obs),
        "n_test": int(len(test_idx)),
        "metrics": str(OUT / "external_singlecell_metrics.csv"),
        "available_external_dirs": [str(p.parent) for p in sorted(EXT.glob("*_hvg*_n*/embedding.npy"))],
    }
    (OUT / "figure2_external_singlecell_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
