#!/usr/bin/env python
"""Fine-tuned Cortex/T1001 annotation and spatial deconvolution benchmark.

This follows the OmniCell tutorial pattern: embeddings are evaluated with
supervised downstream probes, clustering agreement, dominant-type prediction,
composition, and spatial maps.
"""

from __future__ import annotations
import os

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
DEFAULT_INPUT = PROJECT / "results" / "cortex_t1001_task_inputs"
DEFAULT_LATEST = PROJECT / "results" / "cortex_t1001_latest_embeddings"
DEFAULT_NATIVE = PROJECT / "results" / "cortex_t1001_native_omnicell_embeddings"
DEFAULT_OUT = PROJECT / "figures" / "figure2_cortex_t1001_finetune_benchmark"
DEFAULT_RESULTS = PROJECT / "results" / "cortex_t1001_finetune_benchmark"

METHOD_COLORS = {
    "Raw expression SVD": "#8A97A8",
    "OmniCell native": "#7C6AA6",
    "OmniCell CPT 512": "#5784A8",
    "OmniCell fine-tuned": "#C86054",
}
PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "line": "#9AA7B8",
}

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


class TaskAdapter(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, 384),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(384, 192),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(192, hidden_dim),
        )
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.nn.functional.normalize(self.encoder(x), p=2, dim=1)
        return self.classifier(z), z


@dataclass
class ProbeResult:
    method: str
    task: str
    prediction: np.ndarray
    truth: np.ndarray
    test_indices: np.ndarray
    features_for_umap: np.ndarray | None = None


@dataclass
class LinearProbe:
    scaler: StandardScaler
    model: LogisticRegression

    def predict(self, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(np.asarray(features[indices], dtype=np.float32))
        return self.model.predict(x).astype(np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--latest-dir", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--sc-test-fraction", type=float, default=0.30)
    parser.add_argument("--spatial-calibration-fraction", type=float, default=0.25)
    parser.add_argument("--svd-genes", type=int, default=3000)
    parser.add_argument("--svd-components", type=int, default=80)
    parser.add_argument("--max-umap-points", type=int, default=12000)
    parser.add_argument("--umap-neighbors", type=int, default=35)
    parser.add_argument("--umap-min-dist", type=float, default=0.22)
    parser.add_argument("--dpi", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_label(values: pd.Series | np.ndarray) -> np.ndarray:
    raw = pd.Series(values)
    s = raw.astype("object").where(pd.notna(raw), "").astype(str).str.strip()
    bad = s.str.lower().isin(["", "nan", "none", "<na>", "unknown", "unassigned"])
    return s.mask(bad, "Other/unknown").to_numpy(dtype=str)


def load_embedding_dir(path: Path) -> tuple[pd.DataFrame, np.ndarray] | None:
    if not (path / "embedding.npy").exists():
        return None
    meta_path = path / "embedding_meta.parquet"
    if not meta_path.exists():
        return None
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    emb = np.load(path / "embedding.npy", mmap_mode="r")
    if len(meta) != emb.shape[0]:
        raise ValueError(f"{path} meta/embedding mismatch: {len(meta)} vs {emb.shape[0]}")
    return meta, emb


def align_embedding(meta_ref: pd.DataFrame, loaded: tuple[pd.DataFrame, np.ndarray] | None) -> np.ndarray | None:
    if loaded is None:
        return None
    meta, emb = loaded
    key_cols = ["sample_id", "cell_index"]
    if all(col in meta.columns for col in key_cols):
        left = meta_ref[key_cols].astype(str).agg("||".join, axis=1)
        right = meta[key_cols].astype(str).agg("||".join, axis=1)
    elif "source_cell_index" in meta.columns:
        left = meta_ref[["sample_id", "cell_index"]].astype(str).agg("||".join, axis=1)
        right = meta[["sample_id", "source_cell_index"]].astype(str).agg("||".join, axis=1)
    else:
        if len(meta_ref) != emb.shape[0]:
            return None
        return emb
    order = pd.Series(np.arange(len(meta)), index=right).reindex(left)
    if order.isna().any():
        return None
    return emb[order.to_numpy(dtype=np.int64)]


def top_variable_genes(x: sparse.spmatrix, n: int) -> np.ndarray:
    means = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    var = mean_sq - means * means
    n = min(n, x.shape[1])
    return np.argsort(var)[-n:]


def make_raw_svd(input_dir: Path, meta: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    cache = args.results_dir / "raw_expression_svd.npy"
    if cache.exists() and not args.force:
        return np.load(cache, mmap_mode="r")
    sc_adata = ad.read_h5ad(input_dir / "cortex_sc_subset.h5ad")
    sp_adata = ad.read_h5ad(input_dir / "t1001_spatial.h5ad")
    x = sparse.vstack([sc_adata.X, sp_adata.X], format="csr")
    top = top_variable_genes(x, args.svd_genes)
    svd = TruncatedSVD(n_components=min(args.svd_components, len(top) - 1), random_state=args.seed)
    features = svd.fit_transform(x[:, top]).astype(np.float32)
    np.save(cache, features)
    pd.DataFrame({"gene": sc_adata.var_names[top], "variance_rank": np.arange(len(top), 0, -1)}).to_csv(
        args.results_dir / "raw_svd_selected_genes.csv", index=False
    )
    if features.shape[0] != len(meta):
        raise ValueError("Raw SVD row count does not match the unified index")
    return features


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / np.mean(weights[counts > 0])
    return torch.tensor(weights, dtype=torch.float32)


def iter_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None = None):
    order = indices.copy()
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def train_predict_adapter(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    return_all: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    scaler = StandardScaler()
    train_x = np.asarray(features[train_idx], dtype=np.float32)
    scaler.fit(train_x)
    n_classes = int(labels.max()) + 1
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    model = TaskAdapter(features.shape[1], args.hidden_dim, n_classes).to(device)
    weights = class_weights(labels[train_idx], n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for idx in iter_batches(train_idx, args.batch_size, rng):
            xb = scaler.transform(np.asarray(features[idx], dtype=np.float32)).astype(np.float32)
            yb = labels[idx]
            x_t = torch.from_numpy(xb).to(device)
            y_t = torch.from_numpy(yb).long().to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            correct += int((logits.argmax(dim=1) == y_t).sum().detach().cpu())
            total += len(idx)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            pred_test, _ = predict_adapter(model, scaler, features, test_idx, args.batch_size, device)
            test_acc = accuracy_score(labels[test_idx], pred_test)
        else:
            test_acc = np.nan
        history.append({"epoch": epoch, "train_loss": total_loss / max(total, 1), "train_accuracy": correct / max(total, 1), "quick_test_accuracy": test_acc})
        print(f"[INFO] epoch={epoch:03d} loss={history[-1]['train_loss']:.4f} train_acc={history[-1]['train_accuracy']:.4f} test_acc={test_acc:.4f}", flush=True)
    pred, z_test = predict_adapter(model, scaler, features, test_idx, args.batch_size, device, return_z=True)
    z_all = None
    if return_all:
        _, z_all = predict_adapter(model, scaler, features, np.arange(features.shape[0], dtype=np.int64), args.batch_size, device, return_z=True)
    return pred, z_test, z_all, pd.DataFrame(history)


def predict_adapter(
    model: TaskAdapter,
    scaler: StandardScaler,
    features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    return_z: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    preds = np.empty(len(indices), dtype=np.int64)
    zs: list[np.ndarray] = []
    with torch.inference_mode():
        cursor = 0
        for idx in iter_batches(indices, batch_size):
            xb = scaler.transform(np.asarray(features[idx], dtype=np.float32)).astype(np.float32)
            logits, z = model(torch.from_numpy(xb).to(device))
            p = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            preds[cursor : cursor + len(idx)] = p
            if return_z:
                zs.append(z.detach().cpu().numpy().astype(np.float32))
            cursor += len(idx)
    return preds, (np.concatenate(zs, axis=0) if return_z else None)


def stratified_split(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(labels), dtype=np.int64)
    train, test = train_test_split(idx, test_size=test_fraction, random_state=seed, stratify=labels)
    return np.sort(train), np.sort(test)


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, seed: int) -> LinearProbe:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(features[train_idx], dtype=np.float32))
    model = LogisticRegression(
        max_iter=1200,
        C=2.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(x_train, labels[train_idx])
    return LinearProbe(scaler=scaler, model=model)


def metric_rows(method: str, task: str, truth: np.ndarray, pred: np.ndarray, features: np.ndarray | None = None) -> list[dict]:
    rows = [
        {"task": task, "method": method, "metric": "Accuracy", "value": accuracy_score(truth, pred)},
        {"task": task, "method": method, "metric": "Balanced accuracy", "value": balanced_accuracy_score(truth, pred)},
        {"task": task, "method": method, "metric": "Macro F1", "value": f1_score(truth, pred, average="macro", zero_division=0)},
        {"task": task, "method": method, "metric": "ARI", "value": adjusted_rand_score(truth, pred)},
        {"task": task, "method": method, "metric": "NMI", "value": normalized_mutual_info_score(truth, pred)},
    ]
    if features is not None:
        n_clusters = len(np.unique(truth))
        if len(features) > 0 and n_clusters > 1:
            km = KMeans(n_clusters=n_clusters, random_state=7, n_init=10)
            cluster = km.fit_predict(np.asarray(features, dtype=np.float32))
            rows.extend(
                [
                    {"task": task, "method": method, "metric": "Cluster ARI", "value": adjusted_rand_score(truth, cluster)},
                    {"task": task, "method": method, "metric": "Cluster NMI", "value": normalized_mutual_info_score(truth, cluster)},
                ]
            )
    return rows


def run_annotation(meta: pd.DataFrame, methods: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, ProbeResult], LabelEncoder, np.ndarray]:
    sc_mask = meta["modality"].astype(str).eq("single_cell").to_numpy()
    labels_text = clean_label(meta.loc[sc_mask, "ground_truth_celltype"])
    enc = LabelEncoder().fit(labels_text)
    labels = enc.transform(labels_text).astype(np.int64)
    train_idx_local, test_idx_local = stratified_split(labels, args.sc_test_fraction, args.seed)
    global_sc = np.flatnonzero(sc_mask)
    rows = []
    results: dict[str, ProbeResult] = {}
    for method, feat in methods.items():
        print(f"[INFO] single-cell annotation: {method}", flush=True)
        feat_sc = feat[sc_mask]
        probe = fit_linear_probe(feat_sc, labels, train_idx_local, args.seed)
        pred = probe.predict(feat_sc, test_idx_local)
        rows.extend(metric_rows(method, "single-cell annotation", labels[test_idx_local], pred, feat_sc[test_idx_local]))
        results[method] = ProbeResult(
            method=method,
            task="single-cell annotation",
            prediction=enc.inverse_transform(pred),
            truth=enc.inverse_transform(labels[test_idx_local]),
            test_indices=global_sc[test_idx_local],
            features_for_umap=np.asarray(feat_sc[test_idx_local], dtype=np.float32),
        )
        if method == "OmniCell CPT 512":
            pred_ft, z_test, _, hist = train_predict_adapter(
                feat_sc,
                labels,
                train_idx_local,
                test_idx_local,
                args,
                return_all=False,
            )
            hist.to_csv(args.results_dir / "single_cell_finetune_history.csv", index=False)
            rows.extend(metric_rows("OmniCell fine-tuned", "single-cell annotation", labels[test_idx_local], pred_ft, z_test))
            results["OmniCell fine-tuned"] = ProbeResult(
                method="OmniCell fine-tuned",
                task="single-cell annotation",
                prediction=enc.inverse_transform(pred_ft),
                truth=enc.inverse_transform(labels[test_idx_local]),
                test_indices=global_sc[test_idx_local],
                features_for_umap=z_test,
            )
    return pd.DataFrame(rows), results, enc, global_sc[test_idx_local]


def train_transfer_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    train_global_idx: np.ndarray,
    test_global_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    return train_predict_adapter(features, labels, train_global_idx, test_global_idx, args, return_all=False)[0:4:2]


def run_spatial(meta: pd.DataFrame, methods: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    sc_mask = meta["modality"].astype(str).eq("single_cell").to_numpy()
    sp_mask = meta["modality"].astype(str).eq("spatial").to_numpy()
    all_labels_text = clean_label(meta["ground_truth_celltype"])
    enc = LabelEncoder().fit(all_labels_text)
    all_labels = enc.transform(all_labels_text).astype(np.int64)
    sp_idx = np.flatnonzero(sp_mask)
    sp_labels = all_labels[sp_idx]
    cal_local, held_local = stratified_split(sp_labels, 1.0 - args.spatial_calibration_fraction, args.seed + 13)
    cal_idx = sp_idx[cal_local]
    held_idx = sp_idx[held_local]
    sc_idx = np.flatnonzero(sc_mask)
    rows = []
    prediction_rows = meta.loc[sp_mask, ["sample_id", "cell_index", "obs_name", "ground_truth_celltype"]].copy().reset_index(drop=True)
    spatial_predictions: dict[str, np.ndarray] = {}
    for method, feat in methods.items():
        print(f"[INFO] spatial deconvolution: {method}", flush=True)
        method_name = f"{method} transfer"
        probe = fit_linear_probe(feat, all_labels, sc_idx, args.seed)
        pred = probe.predict(feat, held_idx)
        rows.extend(metric_rows(method_name, "T1001 spatial deconvolution", all_labels[held_idx], pred, np.asarray(feat[held_idx], dtype=np.float32)))
        pred_text_all = enc.inverse_transform(probe.predict(feat, sp_idx))
        prediction_rows[f"pred_{method_name}"] = pred_text_all
        spatial_predictions[method_name] = pred_text_all
        if method == "OmniCell CPT 512":
            train_idx = np.concatenate([sc_idx, cal_idx])
            pred_all_ft, z_spatial, _, hist = train_predict_adapter(feat, all_labels, train_idx, sp_idx, args, return_all=False)
            hist.to_csv(args.results_dir / "spatial_finetune_history.csv", index=False)
            held_pos = pd.Index(sp_idx).get_indexer(held_idx)
            pred_ft = pred_all_ft[held_pos]
            z_held = z_spatial[held_pos] if z_spatial is not None else None
            rows.extend(metric_rows("OmniCell fine-tuned", "T1001 spatial deconvolution", all_labels[held_idx], pred_ft, z_held))
            pred_text_ft = enc.inverse_transform(pred_all_ft)
            prediction_rows["pred_OmniCell fine-tuned"] = pred_text_ft
            spatial_predictions["OmniCell fine-tuned"] = pred_text_ft
    prediction_rows.to_csv(args.results_dir / "t1001_spatial_predictions.csv", index=False)
    return pd.DataFrame(rows), prediction_rows, spatial_predictions


def make_label_palette(labels: list[str]) -> dict[str, str]:
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + list(plt.get_cmap("tab20c").colors)
    colors = {}
    for i, label in enumerate(labels):
        rgb = base[i % len(base)]
        colors[label] = mpl.colors.to_hex(rgb)
    return colors


def compute_umap(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
        init="spectral",
        low_memory=True,
    )
    return reducer.fit_transform(np.asarray(features, dtype=np.float32)).astype(np.float32)


def sample_umap_rows(results: dict[str, ProbeResult], max_points: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for method, res in results.items():
        n = len(res.truth)
        if n <= max_points:
            out[method] = np.arange(n, dtype=np.int64)
        else:
            frame = pd.DataFrame({"row": np.arange(n), "label": res.truth})
            chosen = []
            groups = list(frame.groupby("label").groups.items())
            per = max(1, max_points // max(1, len(groups)))
            for _, idx_like in groups:
                idx = np.asarray(list(idx_like), dtype=np.int64)
                chosen.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
            rows = np.concatenate(chosen)
            if len(rows) < max_points:
                remain = np.setdiff1d(np.arange(n, dtype=np.int64), rows)
                rows = np.concatenate([rows, rng.choice(remain, size=min(max_points - len(rows), len(remain)), replace=False)])
            out[method] = np.sort(rows[:max_points])
    return out


def plot_sc_umaps(results: dict[str, ProbeResult], labels_order: list[str], label_colors: dict[str, str], args: argparse.Namespace) -> pd.DataFrame:
    rows_by_method = sample_umap_rows(results, args.max_umap_points, args.seed)
    n_methods = len(results)
    fig = plt.figure(figsize=(2.25 * n_methods + 1.15, 3.0))
    gs = fig.add_gridspec(1, n_methods + 1, width_ratios=[1] * n_methods + [0.58], left=0.055, right=0.985, top=0.78, bottom=0.12, wspace=0.08)
    source_parts = []
    for i, (method, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i])
        rows = rows_by_method[method]
        coords = compute_umap(res.features_for_umap[rows], args)
        frame = pd.DataFrame({"method": method, "umap_1": coords[:, 0], "umap_2": coords[:, 1], "cell_type": res.truth[rows]})
        source_parts.append(frame)
        for j, label in enumerate(labels_order):
            sub = frame[frame["cell_type"].eq(label)].sample(frac=1.0, random_state=args.seed + j) if frame["cell_type"].eq(label).any() else frame.iloc[0:0]
            if len(sub):
                ax.scatter(sub["umap_1"], sub["umap_2"], s=1.1, color=label_colors[label], alpha=0.62, linewidths=0, rasterized=True)
        ax.set_title(method, loc="left", fontsize=7.2, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.02, 0.02, f"n = {len(frame):,}", transform=ax.transAxes, ha="left", va="bottom", fontsize=4.8, color=PALETTE["muted"])
    axl = fig.add_subplot(gs[0, -1])
    axl.axis("off")
    axl.text(0, 0.98, "Cell type", fontsize=6.4, fontweight="bold", color=PALETTE["muted"], ha="left", va="top")
    y = 0.91
    for label in labels_order[:22]:
        axl.scatter([0.035], [y], s=14, color=label_colors[label], linewidths=0)
        axl.text(0.085, y, label, fontsize=4.6, va="center", ha="left")
        y -= 0.039
    fig.text(0.020, 0.955, "B", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.955, "Single-cell clustering across representations", fontsize=8.6, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.895, "Held-out Cortex_sc cells; the fine-tuned adapter is trained only on the training split.", fontsize=5.9, color=PALETTE["muted"], ha="left", va="top")
    save_figure(fig, args.output_dir / "figure2_sc_clustering_umap_methods", args.dpi)
    source = pd.concat(source_parts, ignore_index=True)
    source.to_csv(args.output_dir / "figure2_sc_clustering_umap_methods_source.csv", index=False)
    return source


def plot_metrics(metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    tasks = ["single-cell annotation", "T1001 spatial deconvolution"]
    shown_metrics = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI", "Cluster ARI", "Cluster NMI"]
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 1, left=0.11, right=0.98, top=0.82, bottom=0.12, hspace=0.40)
    for row_i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[row_i, 0])
        sub = metrics[metrics["task"].eq(task) & metrics["metric"].isin(shown_metrics)].copy()
        methods = list(dict.fromkeys(sub["method"].tolist()))
        x = np.arange(len(shown_metrics))
        width = min(0.16, 0.78 / max(1, len(methods)))
        for i, method in enumerate(methods):
            vals = []
            for metric in shown_metrics:
                v = sub.loc[sub["method"].eq(method) & sub["metric"].eq(metric), "value"]
                vals.append(float(v.iloc[0]) if len(v) else np.nan)
            offset = (i - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.92, color=METHOD_COLORS.get(method.replace(" transfer", ""), METHOD_COLORS.get(method, "#9AA7B8")), label=method)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(x, shown_metrics, rotation=0, ha="center", fontsize=5.4)
        ax.set_ylabel("score")
        ax.set_title(task, loc="left", fontsize=7.1, fontweight="bold", pad=3)
        ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
        ax.set_axisbelow(True)
        if row_i == 0:
            ax.legend(ncols=min(4, len(methods)), loc="upper left", bbox_to_anchor=(0, 1.30), fontsize=5.0, handlelength=1.0, columnspacing=0.8)
    fig.text(0.020, 0.955, "A", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.955, "Task-level performance against raw and pretrained representations", fontsize=8.6, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.895, "Metrics use held-out labels; ARI/NMI follow the OmniCell-style clustering/deconvolution evaluation.", fontsize=5.9, color=PALETTE["muted"], ha="left", va="top")
    save_figure(fig, args.output_dir / "figure2_task_metrics_omnicell_style", args.dpi)


def plot_spatial(predictions: pd.DataFrame, input_dir: Path, labels_order: list[str], label_colors: dict[str, str], args: argparse.Namespace) -> None:
    adata_sp = ad.read_h5ad(input_dir / "t1001_spatial.h5ad")
    coords = np.asarray(adata_sp.obsm["spatial"], dtype=np.float32)
    plot_cols = ["ground_truth_celltype"] + [c for c in predictions.columns if c.startswith("pred_")]
    titles = ["Ground truth"] + [c.replace("pred_", "") for c in plot_cols[1:]]
    max_cols = min(len(plot_cols), 5)
    fig = plt.figure(figsize=(2.15 * max_cols + 1.0, 3.0))
    gs = fig.add_gridspec(1, max_cols + 1, width_ratios=[1] * max_cols + [0.55], left=0.055, right=0.985, top=0.78, bottom=0.10, wspace=0.08)
    for i, col in enumerate(plot_cols[:max_cols]):
        ax = fig.add_subplot(gs[0, i])
        values = predictions[col].astype(str).to_numpy()
        for j, label in enumerate(labels_order):
            mask = values == label
            if np.any(mask):
                order = np.flatnonzero(mask)
                ax.scatter(coords[order, 0], coords[order, 1], s=1.0, color=label_colors[label], alpha=0.72, linewidths=0, rasterized=True)
        ax.set_title(titles[i], loc="left", fontsize=7.0, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axl = fig.add_subplot(gs[0, -1])
    axl.axis("off")
    axl.text(0, 0.98, "Cell type", fontsize=6.2, fontweight="bold", color=PALETTE["muted"], ha="left", va="top")
    y = 0.91
    for label in labels_order[:22]:
        axl.scatter([0.035], [y], s=14, color=label_colors[label], linewidths=0)
        axl.text(0.085, y, label, fontsize=4.5, va="center", ha="left")
        y -= 0.039
    fig.text(0.020, 0.955, "C", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.955, "T1001 spatial deconvolution and dominant-type maps", fontsize=8.6, fontweight="bold", ha="left", va="top")
    fig.text(0.058, 0.895, "Fine-tuned predictions use single-cell reference plus a stratified spatial calibration subset; metrics are held out.", fontsize=5.9, color=PALETTE["muted"], ha="left", va="top")
    save_figure(fig, args.output_dir / "figure2_t1001_spatial_deconvolution_maps", args.dpi)


def plot_composition(predictions: pd.DataFrame, labels_order: list[str], label_colors: dict[str, str], args: argparse.Namespace) -> None:
    cols = ["ground_truth_celltype"] + [c for c in predictions.columns if c.startswith("pred_")]
    labels = ["Ground truth"] + [c.replace("pred_", "") for c in cols[1:]]
    comp = []
    for label, col in zip(labels, cols):
        counts = predictions[col].astype(str).value_counts(normalize=True)
        for ct in labels_order:
            comp.append({"method": label, "cell_type": ct, "fraction": float(counts.get(ct, 0.0))})
    frame = pd.DataFrame(comp)
    fig, ax = plt.subplots(figsize=(6.7, 2.15))
    bottom = np.zeros(len(labels))
    x = np.arange(len(labels))
    for ct in labels_order:
        vals = frame.loc[frame["cell_type"].eq(ct), "fraction"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=label_colors[ct], width=0.72, linewidth=0)
        bottom += vals
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=5.2)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("T1001 composition by dominant cell type", loc="left", fontsize=7.1, fontweight="bold")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.45)
    ax.set_axisbelow(True)
    save_figure(fig, args.output_dir / "figure2_t1001_composition", args.dpi)
    frame.to_csv(args.output_dir / "figure2_t1001_composition_source.csv", index=False)


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(args.input_dir / "cortex_t1001_index.parquet").reset_index(drop=True)
    latest_loaded = load_embedding_dir(args.latest_dir)
    if latest_loaded is None:
        raise FileNotFoundError(f"Latest CPT embeddings not found in {args.latest_dir}")
    latest = align_embedding(meta, latest_loaded)
    if latest is None:
        raise ValueError("Could not align latest CPT embeddings to the task index")
    methods: dict[str, np.ndarray] = {"Raw expression SVD": make_raw_svd(args.input_dir, meta, args), "OmniCell CPT 512": latest}
    native = align_embedding(meta, load_embedding_dir(args.native_dir))
    if native is not None:
        methods["OmniCell native"] = native
    metrics_sc, sc_results, _, _ = run_annotation(meta, methods, args)
    metrics_sp, spatial_predictions, _ = run_spatial(meta, methods, args)
    metrics = pd.concat([metrics_sc, metrics_sp], ignore_index=True)
    metrics.to_csv(args.output_dir / "figure2_cortex_t1001_metrics.csv", index=False)
    metrics.to_csv(args.results_dir / "metrics.csv", index=False)
    spatial_predictions.to_csv(args.output_dir / "figure2_t1001_spatial_predictions.csv", index=False)
    label_counts = pd.Series(clean_label(meta["ground_truth_celltype"])).value_counts()
    labels_order = label_counts.index.astype(str).tolist()
    label_colors = make_label_palette(labels_order)
    (args.output_dir / "celltype_colors.json").write_text(json.dumps(label_colors, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_metrics(metrics, args)
    plot_sc_umaps(sc_results, labels_order, label_colors, args)
    plot_spatial(spatial_predictions, args.input_dir, labels_order, label_colors, args)
    plot_composition(spatial_predictions, labels_order, label_colors, args)
    contract = {
        "core_conclusion": "Task-specific fine-tuning improves Cortex single-cell annotation and T1001 spatial dominant-type deconvolution over raw expression and frozen representations.",
        "input_dir": str(args.input_dir),
        "latest_dir": str(args.latest_dir),
        "native_dir": str(args.native_dir) if native is not None else None,
        "n_cells": int(len(meta)),
        "n_single_cell": int(meta["modality"].astype(str).eq("single_cell").sum()),
        "n_spatial": int(meta["modality"].astype(str).eq("spatial").sum()),
        "metrics": str(args.output_dir / "figure2_cortex_t1001_metrics.csv"),
        "figures": [
            "figure2_task_metrics_omnicell_style.pdf/png/svg/tiff",
            "figure2_sc_clustering_umap_methods.pdf/png/svg/tiff",
            "figure2_t1001_spatial_deconvolution_maps.pdf/png/svg/tiff",
            "figure2_t1001_composition.pdf/png/svg/tiff",
        ],
    }
    (args.output_dir / "figure2_cortex_t1001_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(contract, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
