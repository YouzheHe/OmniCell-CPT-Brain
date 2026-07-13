#!/usr/bin/env python
"""Validate OmniCell CPT embeddings against raw expression baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORK_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_EMBED_DIR = WORK_ROOT / "projects" / "nvu_vascular" / "results" / "omnicell_cpt_latest_embeddings"
DEFAULT_INDEX = DEFAULT_EMBED_DIR / "embedding_meta.parquet"
DEFAULT_EMBEDDING = DEFAULT_EMBED_DIR / "embedding.npy"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "projects" / "nvu_vascular" / "results" / "atlas_validation"

sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets" / "src"))
from cellfm_dataset.memmap import MemmapDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index-parquet", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--embedding-npy", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cells", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--raw-svd", action="store_true")
    parser.add_argument("--raw-components", type=int, default=50)
    parser.add_argument("--probe-max-iter", type=int, default=1000)
    parser.add_argument("--probe-model", choices=["logistic", "ridge"], default="logistic")
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def balanced_sample(frame: pd.DataFrame, max_cells: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["_embedding_row"] = np.arange(len(frame), dtype=np.int64)
    if len(frame) <= max_cells:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    group_cols = [col for col in ["vascular_class", "cohort"] if col in frame.columns]
    if not group_cols:
        chosen = rng.choice(len(frame), size=max_cells, replace=False)
        return frame.iloc[np.sort(chosen)].reset_index(drop=True)
    pieces = []
    grouped = list(frame.groupby(group_cols, dropna=False))
    per_group = max(1, max_cells // len(grouped))
    for _, group in grouped:
        take = min(len(group), per_group)
        pieces.append(group.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    sampled = pd.concat(pieces, ignore_index=False)
    if len(sampled) < max_cells:
        remaining = frame.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            extra = remaining.sample(
                n=min(max_cells - len(sampled), len(remaining)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
            sampled = pd.concat([sampled, extra], ignore_index=False)
    return sampled.sort_values("_embedding_row").reset_index(drop=True)


def neighbor_entropy(labels: np.ndarray, indices: np.ndarray) -> float:
    unique = pd.Series(labels).dropna().astype(str).unique()
    if len(unique) <= 1:
        return float("nan")
    denom = math.log(len(unique))
    scores = []
    for row in indices:
        counts = pd.Series(labels[row]).astype(str).value_counts(normalize=True).to_numpy()
        entropy = -float(np.sum(counts * np.log(np.maximum(counts, 1e-12))))
        scores.append(entropy / denom)
    return float(np.mean(scores))


def neighbor_purity(labels: np.ndarray, indices: np.ndarray) -> float:
    labels = pd.Series(labels).astype(str).to_numpy()
    scores = []
    for i, row in enumerate(indices):
        scores.append(float(np.mean(labels[row] == labels[i])))
    return float(np.mean(scores))


def space_metrics(name: str, features: np.ndarray, meta: pd.DataFrame, n_neighbors: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] < 10:
        return rows
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    k = min(n_neighbors + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]

    for label_col in ["vascular_class", "cohort", "condition_inferred"]:
        if label_col not in meta.columns:
            continue
        labels = meta[label_col].fillna("Unknown").astype(str).to_numpy()
        if len(set(labels)) < 2:
            continue
        metric_name = "neighbor_purity" if label_col == "vascular_class" else "neighbor_entropy"
        value = neighbor_purity(labels, indices) if label_col == "vascular_class" else neighbor_entropy(labels, indices)
        rows.append({"space": name, "metric": metric_name, "label": label_col, "value": value})
        counts = pd.Series(labels).value_counts()
        if (counts >= 2).sum() >= 2:
            sample_size = min(10000, x.shape[0])
            try:
                score = silhouette_score(x, labels, metric="cosine", sample_size=sample_size, random_state=13)
                rows.append({"space": name, "metric": "silhouette", "label": label_col, "value": float(score)})
            except Exception as exc:
                rows.append({"space": name, "metric": "silhouette_error", "label": label_col, "value": str(exc)})
    return rows


def probe_metrics(
    name: str,
    features: np.ndarray,
    meta: pd.DataFrame,
    seed: int,
    max_iter: int,
    probe_model: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    groups = meta["sample_id"].astype(str).to_numpy() if "sample_id" in meta.columns else np.arange(len(meta))
    for target in ["vascular_class", "condition_inferred"]:
        if target not in meta.columns:
            continue
        labels = meta[target].fillna("Unknown").astype(str)
        labels = labels[labels != "Unknown"]
        if labels.nunique() < 2 or len(labels) < 20:
            continue
        idx = labels.index.to_numpy()
        y = labels.to_numpy()
        x = features[idx]
        target_groups = groups[idx]
        if probe_model == "ridge":
            classifier = RidgeClassifier(class_weight="balanced")
        else:
            classifier = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        model = make_pipeline(StandardScaler(), classifier)
        scores = []
        if len(set(target_groups)) >= 3:
            splitter = GroupKFold(n_splits=min(5, len(set(target_groups))))
            splits = splitter.split(x, y, groups=target_groups)
        else:
            splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            splits = splitter.split(x, y)
        for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
            if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
                continue
            print(f"[INFO] probe {name}.{target} fold {fold_i}", flush=True)
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            scores.append(balanced_accuracy_score(y[test_idx], pred))
        if scores:
            rows.append({"space": name, "target": target, "metric": "balanced_accuracy", "value": float(np.mean(scores))})

    if "age_years" in meta.columns:
        age = pd.to_numeric(meta["age_years"], errors="coerce")
        valid = age.notna()
        if valid.sum() >= 30 and age[valid].nunique() >= 3:
            idx = np.flatnonzero(valid.to_numpy())
            x = features[idx]
            y = age[valid].to_numpy(dtype=np.float32)
            target_groups = groups[idx]
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            scores = []
            if len(set(target_groups)) >= 3:
                splitter = GroupKFold(n_splits=min(5, len(set(target_groups))))
                splits = splitter.split(x, y, groups=target_groups)
            else:
                splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                bins = pd.qcut(y, q=min(3, len(set(y))), duplicates="drop", labels=False)
                splits = splitter.split(x, bins)
            for train_idx, test_idx in splits:
                model.fit(x[train_idx], y[train_idx])
                pred = model.predict(x[test_idx])
                scores.append(r2_score(y[test_idx], pred))
            if scores:
                rows.append({"space": name, "target": "age_years", "metric": "r2", "value": float(np.mean(scores))})
    return rows


def raw_svd_features(dataset_root: Path, meta: pd.DataFrame, n_components: int) -> np.ndarray:
    dataset = MemmapDataset(dataset_root)
    nnz = 0
    cached: list[tuple[np.ndarray, np.ndarray]] = []
    for out_i, row in enumerate(meta.itertuples(index=False)):
        sample = dataset.samples[str(getattr(row, "sample_id"))]
        cell_index = int(getattr(row, "cell_index"))
        start = int(sample.indptr[cell_index])
        end = int(sample.indptr[cell_index + 1])
        gene_ids = np.asarray(sample.indices[start:end], dtype=np.int64)
        values = np.asarray(sample.values[start:end], dtype=np.float32)
        good = np.isfinite(values) & (gene_ids >= 0) & (gene_ids < len(dataset.gene_vocab))
        gene_ids = gene_ids[good]
        values = values[good]
        cached.append((gene_ids, values))
        nnz += len(gene_ids)
        if (out_i + 1) % 5000 == 0:
            print(f"[INFO] raw matrix scan {out_i + 1}/{len(meta)} cells, nnz={nnz}", flush=True)

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    offset = 0
    for out_i, (gene_ids, values) in enumerate(cached):
        end = offset + len(gene_ids)
        rows[offset:end] = out_i
        cols[offset:end] = gene_ids.astype(np.int32, copy=False)
        data[offset:end] = values.astype(np.float32, copy=False)
        offset = end
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(meta), len(dataset.gene_vocab)),
        dtype=np.float32,
    )
    n_components = min(n_components, max(2, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=17)
    return svd.fit_transform(matrix).astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_all = load_table(args.index_parquet.expanduser().resolve())
    embedding_all = np.load(args.embedding_npy.expanduser().resolve(), mmap_mode="r")
    meta = balanced_sample(meta_all, args.max_cells, args.seed)
    print(f"[INFO] validation cells: {len(meta)}", flush=True)
    selected_positions = meta["_embedding_row"].to_numpy(dtype=np.int64)
    embedding = np.asarray(embedding_all[selected_positions], dtype=np.float32)
    meta = meta.reset_index(drop=True)

    metrics = []
    probes = []
    print("[INFO] computing OmniCell atlas metrics", flush=True)
    metrics.extend(space_metrics("omnicell_cpt", embedding, meta, args.neighbors))
    print("[INFO] computing OmniCell probe metrics", flush=True)
    probes.extend(probe_metrics("omnicell_cpt", embedding, meta, args.seed, args.probe_max_iter, args.probe_model))

    pd.DataFrame(metrics).to_csv(output_dir / "atlas_metrics.partial.csv", index=False)
    pd.DataFrame(probes).to_csv(output_dir / "probe_metrics.partial.csv", index=False)
    meta.to_csv(output_dir / "validation_cells.csv", index=False)

    if args.raw_svd:
        print("[INFO] computing raw expression SVD baseline", flush=True)
        raw = raw_svd_features(args.dataset_root.expanduser().resolve(), meta, args.raw_components)
        np.save(output_dir / "raw_svd_features.npy", raw)
        print("[INFO] computing raw expression atlas metrics", flush=True)
        metrics.extend(space_metrics("raw_expression_svd", raw, meta, args.neighbors))
        print("[INFO] computing raw expression probe metrics", flush=True)
        probes.extend(probe_metrics("raw_expression_svd", raw, meta, args.seed, args.probe_max_iter, args.probe_model))

    metric_frame = pd.DataFrame(metrics)
    probe_frame = pd.DataFrame(probes)
    metric_frame.to_csv(output_dir / "atlas_metrics.csv", index=False)
    probe_frame.to_csv(output_dir / "probe_metrics.csv", index=False)
    meta.to_csv(output_dir / "validation_cells.csv", index=False)
    config = {
        "index": str(args.index_parquet),
        "embedding": str(args.embedding_npy),
        "output_dir": str(output_dir),
        "n_cells": int(len(meta)),
        "raw_svd": bool(args.raw_svd),
        "max_cells": args.max_cells,
        "probe_model": args.probe_model,
    }
    (output_dir / "validation_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    print(metric_frame.to_string(index=False), flush=True)
    if not probe_frame.empty:
        print(probe_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
