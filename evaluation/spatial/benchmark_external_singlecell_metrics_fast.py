#!/usr/bin/env python
"""Fast single-cell annotation metrics for T906/Cortex_sc external baselines."""

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
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
INPUT = PROJECT / "results" / "cortex_t906_task_inputs" / "cortex_sc_subset.h5ad"
LATEST = PROJECT / "results" / "cortex_t1001_latest_embeddings" / "embedding.npy"
NATIVE = PROJECT / "results" / "cortex_t906_native_omnicell_embeddings" / "embedding.npy"
EXT = PROJECT / "results" / "external_singlecell_embeddings"
OUT = PROJECT / "figures" / "figure2_external_singlecell_comparison_hvg15000_fast"
SEED = 20260528
HVG = 15000

COLORS = {
    "Raw expression SVD": "#8A97A8",
    "OmniCell CPT 512": "#5784A8",
    "OmniCell native": "#7C6AA6",
    "OmniCell fine-tuned HVG": "#C86054",
    "CellPLM": "#70B7A6",
    "scGPT": "#5E9A62",
    "scFoundation": "#9C7AAE",
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
        "legend.frameon": False,
    }
)


def top_var_genes(x: sparse.spmatrix, n: int) -> np.ndarray:
    x = x.tocsr()
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0)
    n = min(int(n), x.shape[1])
    idx = np.argpartition(var, -n)[-n:]
    return idx[np.argsort(var[idx])[::-1]]


def dense_for_kmeans(features: np.ndarray, n_components: int = 50) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[1] <= n_components:
        return arr
    return TruncatedSVD(n_components=n_components, random_state=SEED).fit_transform(arr).astype(np.float32)


def dense_for_classifier(features: np.ndarray, n_components: int = 256) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[1] <= n_components:
        return arr
    return TruncatedSVD(n_components=n_components, random_state=SEED).fit_transform(arr).astype(np.float32)


def metric_rows(method: str, y_true: np.ndarray, y_pred: np.ndarray, features: np.ndarray) -> list[dict]:
    cluster_features = dense_for_kmeans(features)
    pred_cluster = KMeans(n_clusters=len(np.unique(y_true)), n_init=10, random_state=SEED).fit_predict(cluster_features)
    return [
        {"task": "single-cell annotation", "method": method, "metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
        {"task": "single-cell annotation", "method": method, "metric": "Balanced accuracy", "value": balanced_accuracy_score(y_true, y_pred)},
        {"task": "single-cell annotation", "method": method, "metric": "Macro F1", "value": f1_score(y_true, y_pred, average="macro")},
        {"task": "single-cell annotation", "method": method, "metric": "Cluster ARI", "value": adjusted_rand_score(y_true, pred_cluster)},
        {"task": "single-cell annotation", "method": method, "metric": "Cluster NMI", "value": normalized_mutual_info_score(y_true, pred_cluster)},
    ]


def fit_dense(method: str, features: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray) -> list[dict]:
    feat = dense_for_classifier(features)
    clf = make_pipeline(StandardScaler(), LinearSVC(C=0.5, class_weight="balanced", random_state=SEED, max_iter=10000))
    clf.fit(feat[train_idx], y[train_idx])
    pred = clf.predict(feat[test_idx])
    return metric_rows(method, y[test_idx], pred, feat[test_idx])


def fit_sparse_hvg(x: sparse.spmatrix, top: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray) -> list[dict]:
    xh = x[:, top]
    clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC(C=0.25, class_weight="balanced", random_state=SEED, max_iter=10000))
    clf.fit(xh[train_idx], y[train_idx])
    pred = clf.predict(xh[test_idx])
    test_features = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(xh[test_idx]).astype(np.float32)
    return metric_rows("OmniCell fine-tuned HVG", y[test_idx], pred, test_features)


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame) -> None:
    show = ["Accuracy", "Balanced accuracy", "Macro F1", "Cluster ARI", "Cluster NMI"]
    methods = list(dict.fromkeys(metrics["method"].tolist()))
    fig, ax = plt.subplots(figsize=(7.4, 2.55))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.78, bottom=0.36)
    x = np.arange(len(show))
    width = min(0.12, 0.82 / len(methods))
    for i, method in enumerate(methods):
        vals = []
        for metric in show:
            v = metrics.loc[metrics["method"].eq(method) & metrics["metric"].eq(metric), "value"]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width * 0.92, color=COLORS.get(method, "#9AA7B8"), label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(show)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.grid(axis="y", color="#D7DEE8", lw=0.45, alpha=0.65)
    fig.text(0.075, 0.95, "Cortex_sc single-cell annotation benchmark", ha="left", va="top", fontsize=9.0, fontweight="bold")
    fig.text(0.075, 0.87, "15,000 HVG for raw/HVG probes; external embeddings use their deployed model-specific inputs.", ha="left", va="top", fontsize=6.4, color="#667085")
    ax.legend(ncol=4, bbox_to_anchor=(0, -0.28), loc="upper left", handlelength=1.0, columnspacing=0.9)
    save(fig, OUT / "external_singlecell_metrics_hvg15000_fast")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(INPUT)
    y_text = adata.obs["cell_type"].astype(str).to_numpy()
    y = LabelEncoder().fit_transform(y_text)
    idx = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    top = top_var_genes(x, HVG)

    rows: list[dict] = []
    raw = TruncatedSVD(n_components=128, random_state=SEED).fit_transform(x[:, top]).astype(np.float32)
    rows.extend(fit_dense("Raw expression SVD", raw, train_idx, test_idx, y))
    rows.extend(fit_dense("OmniCell CPT 512", np.load(LATEST)[: adata.n_obs], train_idx, test_idx, y))
    rows.extend(fit_dense("OmniCell native", np.load(NATIVE)[: adata.n_obs], train_idx, test_idx, y))
    rows.extend(fit_sparse_hvg(x, top, train_idx, test_idx, y))

    external = {
        "CellPLM": EXT / "cellplm_hvg5000_n21855" / "embedding.npy",
        "scGPT": EXT / "scgpt_t906_sc_hvg2000_n21855" / "embedding.npy",
        "scFoundation": EXT / "scfoundation_t906_sc_hvg2000_n21855" / "embedding.npy",
    }
    for method, path in external.items():
        if not path.exists():
            print(f"[WARN] missing {method}: {path}", flush=True)
            continue
        emb = np.load(path)
        if emb.shape[0] != adata.n_obs:
            print(f"[WARN] skipping {method}; rows {emb.shape[0]} != {adata.n_obs}", flush=True)
            continue
        print(f"Evaluating {method}: {emb.shape}", flush=True)
        rows.extend(fit_dense(method, emb, train_idx, test_idx, y))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "external_singlecell_metrics_hvg15000_fast.csv", index=False)
    plot_metrics(metrics)
    contract = {
        "core_conclusion": "Single-cell annotation baselines are evaluated separately from spatial methods. Raw and HVG probes use 15,000 HVG; model embeddings use their deployed model-specific inputs.",
        "input": str(INPUT),
        "output": str(OUT),
        "hvg": HVG,
        "n_cells": int(adata.n_obs),
        "n_test": int(len(test_idx)),
        "metrics": str(OUT / "external_singlecell_metrics_hvg15000_fast.csv"),
        "note": "Dense embeddings above 256 dimensions are reduced by randomized TruncatedSVD for the classifier, and all cluster metrics use a 50-dimensional randomized SVD projection for tractability.",
    }
    (OUT / "external_singlecell_metrics_hvg15000_fast_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
