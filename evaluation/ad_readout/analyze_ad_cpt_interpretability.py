#!/usr/bin/env python
"""Analyze AD/Control CPT embeddings and interpretable module associations.

This script intentionally uses conservative, reviewable interpretability:
linear AD probes on CPT embeddings plus module scores from raw expression.
It reports whether the learned disease axis is supported by cell-type-specific
modules rather than treating the embedding as a black box.
"""

from __future__ import annotations
import os

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
DEFAULT_DATASET = WORK_ROOT / "NVU_hyz"
DEFAULT_MODULE_CONFIG = WORK_ROOT / "projects/BI/config/ad_interpretability_config.json"
DEFAULT_OUT = WORK_ROOT / "projects/BI/results/ad_representation_interpretability"
DEFAULT_ALIAS = WORK_ROOT / "OmniCell-HF/assets/vocab/new_genes_homo_sapiens.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--module-config", type=Path, default=DEFAULT_MODULE_CONFIG)
    parser.add_argument("--alias-csv", type=Path, default=DEFAULT_ALIAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--analysis-name", default="")
    parser.add_argument("--condition-column", default="condition_inferred")
    parser.add_argument("--celltype-column", default="ground_truth_celltype")
    parser.add_argument("--group-column", default="batch_id")
    parser.add_argument("--max-cells", type=int, default=120000)
    parser.add_argument("--raw-svd-components", type=int, default=80)
    parser.add_argument("--cpt-svd-components", type=int, default=32)
    parser.add_argument("--selected-gene-top-k", type=int, default=160)
    parser.add_argument("--selected-gene-candidate-top-k", type=int, default=2500)
    parser.add_argument("--selected-gene-min-mean", type=float, default=0.01)
    parser.add_argument("--min-celltype-cells", type=int, default=50)
    parser.add_argument("--allow-ungrouped-fallback", action="store_true")
    parser.add_argument("--allow-confounder-genes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_meta(path: Path) -> pd.DataFrame:
    parquet = path / "embedding_meta.parquet"
    csv = path / "embedding_meta.csv"
    if parquet.exists():
        return pd.read_parquet(parquet).reset_index(drop=True)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False).reset_index(drop=True)
    raise FileNotFoundError(f"No embedding_meta.parquet/csv found under {path}")


def clean_label(series: pd.Series, fallback: str) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.mask(out.str.lower().isin(["", "nan", "none", "<na>", "na", "n/a"]), fallback)


def sample_dir_for_sample_id(dataset_root: Path, sample_id: str) -> Path:
    if sample_id == "AD_sc" and not (dataset_root / sample_id).exists():
        return dataset_root / "AD_Hip_sc"
    return dataset_root / sample_id


def load_gene_vocab(dataset_root: Path) -> tuple[list[str], dict[str, int]]:
    genes = [line.strip() for line in (dataset_root / "gene_vocab.txt").read_text(encoding="utf-8").splitlines()]
    lookup: dict[str, int] = {}
    for i, gene in enumerate(genes):
        lookup.setdefault(gene.upper(), i)
        lookup.setdefault(gene.split(".", 1)[0].upper(), i)
    return genes, lookup


def load_symbol_to_gene_index(alias_csv: Path, gene_lookup: dict[str, int]) -> dict[str, int]:
    if not alias_csv.exists():
        return {}
    frame = pd.read_csv(alias_csv, header=None, usecols=[0, 1], names=["ensembl", "symbol"]).dropna()
    frame["ensembl_norm"] = frame["ensembl"].astype(str).str.split(".", n=1).str[0].str.upper()
    frame["symbol_norm"] = frame["symbol"].astype(str).str.upper()
    out: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        gene_index = gene_lookup.get(str(row.ensembl_norm))
        if gene_index is not None:
            out.setdefault(str(row.symbol_norm), int(gene_index))
    return out


def load_gene_symbols(alias_csv: Path, genes: list[str]) -> list[str]:
    if not alias_csv.exists():
        return genes
    frame = pd.read_csv(alias_csv, header=None, usecols=[0, 1], names=["ensembl", "symbol"]).dropna()
    frame["ensembl_norm"] = frame["ensembl"].astype(str).str.split(".", n=1).str[0].str.upper()
    frame["symbol"] = frame["symbol"].astype(str)
    mapping = dict(zip(frame["ensembl_norm"], frame["symbol"]))
    return [mapping.get(gene.split(".", 1)[0].upper(), gene) for gene in genes]


def log1p_sparse(raw: sparse.csr_matrix) -> sparse.csr_matrix:
    out = raw.copy()
    out.data = np.log1p(out.data)
    return out


def is_confounder_gene(symbol: str) -> bool:
    gene = str(symbol).upper()
    if gene in {
        "XIST",
        "TSIX",
        "FTX",
        "UTY",
        "USP9Y",
        "DDX3Y",
        "KDM5D",
        "RPS4Y1",
        "EIF1AY",
        "ZFY",
        "NLGN4Y",
        "TTTY14",
        "MALAT1",
        "NEAT1",
    }:
        return True
    prefixes = ("MT-", "RPL", "RPS", "HB", "MTRNR", "MTATP", "MTND", "MTCO")
    return gene.startswith(prefixes)


def allowed_gene_mask(gene_symbols: list[str], allow_confounders: bool = False) -> np.ndarray:
    if allow_confounders:
        return np.ones(len(gene_symbols), dtype=bool)
    return np.array([not is_confounder_gene(gene) for gene in gene_symbols], dtype=bool)


def select_analysis_rows(meta: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    cond = clean_label(meta[args.condition_column], "Unknown")
    keep = cond.isin(["AD", "Control"]).to_numpy()
    indices = np.flatnonzero(keep)
    if args.max_cells and len(indices) > args.max_cells:
        rng = np.random.default_rng(args.seed)
        selected = []
        frame = meta.iloc[indices].copy()
        frame["_row"] = indices
        for _, group in frame.groupby([args.condition_column, args.celltype_column], dropna=False, sort=False):
            frac = min(1.0, args.max_cells / max(len(indices), 1))
            n = max(1, int(round(len(group) * frac)))
            if len(group) > n:
                selected.extend(group.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))["_row"].tolist())
            else:
                selected.extend(group["_row"].tolist())
        indices = np.array(sorted(set(selected)), dtype=np.int64)
        if len(indices) > args.max_cells:
            indices = np.sort(rng.choice(indices, size=args.max_cells, replace=False))
    return indices


def split_by_group(groups: np.ndarray, y: np.ndarray, n_splits: int = 5, allow_ungrouped_fallback: bool = False):
    groups = pd.Series(groups).fillna("unknown").astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return iter([])
    group_frame = pd.DataFrame({"group": groups, "label": y}).drop_duplicates()
    group_label_counts = group_frame.groupby("label")["group"].nunique()
    min_label_groups = int(group_label_counts.min()) if not group_label_counts.empty else 0
    if min_label_groups >= 2:
        folds = min(n_splits, min_label_groups)
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=13).split(np.zeros(len(y)), y, groups)
    if not allow_ungrouped_fallback:
        return iter([])
    codes = pd.factorize(y)[0]
    min_class = np.bincount(codes).min()
    if int(min_class) < 2:
        return iter([])
    folds = max(2, min(n_splits, int(min_class))) if min_class > 1 else 2
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=13).split(np.zeros(len(y)), y)


def fit_probe(
    name: str,
    x: np.ndarray,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    analysis_scope: str = "All cells",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    labels = clean_label(meta[args.condition_column], "Unknown")
    y = labels.map({"Control": 0, "AD": 1}).to_numpy(dtype=int)
    groups = clean_label(meta[args.group_column] if args.group_column in meta.columns else meta["sample_id"], "unknown").to_numpy()
    scores = []
    oof = np.full(len(meta), np.nan, dtype=np.float32)
    if len(meta) < 4 or len(np.unique(y)) < 2:
        return pd.DataFrame(scores), oof, np.zeros(x.shape[1], dtype=np.float32)

    for fold_i, (train, test) in enumerate(split_by_group(groups, y, 5, args.allow_ungrouped_fallback), start=1):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        )
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof[test] = prob.astype(np.float32)
        scores.append(
            {
                "analysis_scope": analysis_scope,
                "feature_space": name,
                "fold": fold_i,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_test_groups": int(len(np.unique(groups[test]))),
                "auroc": float(roc_auc_score(y[test], prob)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro")),
            }
        )

    if not scores:
        return pd.DataFrame(scores), oof, np.zeros(x.shape[1], dtype=np.float32)
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    final_model.fit(x, y)
    coef = final_model.named_steps["logisticregression"].coef_.ravel().astype(np.float32)
    return pd.DataFrame(scores), oof, coef


def build_raw_matrix(dataset_root: Path, meta: pd.DataFrame) -> sparse.csr_matrix:
    genes, _ = load_gene_vocab(dataset_root)
    n_genes = len(genes)
    total_nnz = 0
    groups = []
    for sample_id, frame in meta.groupby("sample_id", sort=False):
        sample_dir = sample_dir_for_sample_id(dataset_root, str(sample_id))
        indptr = np.load(sample_dir / "indptr.npy", mmap_mode="r")
        cell_indices = frame["cell_index"].to_numpy(dtype=np.int64)
        lengths = np.asarray(indptr[cell_indices + 1] - indptr[cell_indices], dtype=np.int64)
        total_nnz += int(lengths.sum())
        groups.append((str(sample_id), sample_dir, frame.index.to_numpy(dtype=np.int64), cell_indices))

    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float32)
    offset = 0
    for sample_id, sample_dir, row_numbers, cell_indices in groups:
        indptr = np.load(sample_dir / "indptr.npy", mmap_mode="r")
        indices = np.load(sample_dir / "indices.npy", mmap_mode="r")
        values = np.load(sample_dir / "values.npy", mmap_mode="r")
        for out_row, cell_i in zip(row_numbers, cell_indices):
            start = int(indptr[cell_i])
            end = int(indptr[cell_i + 1])
            gene_ids = np.asarray(indices[start:end], dtype=np.int64)
            vals = np.nan_to_num(np.asarray(values[start:end], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            good = (gene_ids >= 0) & (gene_ids < n_genes) & np.isfinite(vals) & (vals != 0)
            n = int(good.sum())
            if n == 0:
                continue
            rows[offset : offset + n] = int(out_row)
            cols[offset : offset + n] = gene_ids[good].astype(np.int32, copy=False)
            data[offset : offset + n] = vals[good].astype(np.float32, copy=False)
            offset += n
    return sparse.csr_matrix((data[:offset], (rows[:offset], cols[:offset])), shape=(len(meta), n_genes), dtype=np.float32)


def module_score_table(
    raw: sparse.csr_matrix,
    dataset_root: Path,
    alias_csv: Path,
    modules: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes, lookup = load_gene_vocab(dataset_root)
    symbol_lookup = load_symbol_to_gene_index(alias_csv, lookup)
    module_values: dict[str, np.ndarray] = {}
    availability = []
    log_raw = raw.copy()
    log_raw.data = np.log1p(log_raw.data)
    for module, gene_list in modules.items():
        idx = []
        resolved_symbols = []
        missing_symbols = []
        for gene in gene_list:
            key = gene.upper()
            gene_index = lookup.get(key)
            if gene_index is None:
                gene_index = symbol_lookup.get(key)
            if gene_index is None:
                missing_symbols.append(gene)
            else:
                idx.append(int(gene_index))
                resolved_symbols.append(gene)
        idx = sorted(set(idx))
        availability.append(
            {
                "module": module,
                "n_requested": len(gene_list),
                "n_available": len(idx),
                "available_genes": ";".join([genes[i] for i in idx]),
                "resolved_symbols": ";".join(resolved_symbols),
                "missing_genes": ";".join(missing_symbols),
            }
        )
        if idx:
            vals = np.asarray(log_raw[:, idx].mean(axis=1)).ravel().astype(np.float32)
        else:
            vals = np.zeros(log_raw.shape[0], dtype=np.float32)
        module_values[module] = vals
    scores = pd.DataFrame(module_values)
    for col in scores.columns:
        values = scores[col].to_numpy(dtype=np.float32)
        sd = float(np.nanstd(values))
        scores[col] = (values - float(np.nanmean(values))) / sd if sd > 0 else 0.0
    return scores, pd.DataFrame(availability)


def module_gene_indices(dataset_root: Path, alias_csv: Path, modules: dict[str, list[str]]) -> dict[str, list[int]]:
    _, lookup = load_gene_vocab(dataset_root)
    symbol_lookup = load_symbol_to_gene_index(alias_csv, lookup)
    out: dict[str, list[int]] = {}
    for module, gene_list in modules.items():
        idx = []
        for gene in gene_list:
            key = gene.upper()
            gene_index = lookup.get(key)
            if gene_index is None:
                gene_index = symbol_lookup.get(key)
            if gene_index is not None:
                idx.append(int(gene_index))
        out[module] = sorted(set(idx))
    return out


def select_fold_ad_genes(
    raw_log: sparse.csr_matrix,
    y: np.ndarray,
    train: np.ndarray,
    candidate_idx: list[int] | None,
    gene_symbols: list[str],
    gene_allowed: np.ndarray,
    args: argparse.Namespace,
    analysis_scope: str,
    fold: int,
    selection_space: str,
) -> tuple[list[int], pd.DataFrame]:
    train_ad = train[y[train] == 1]
    train_control = train[y[train] == 0]
    if len(train_ad) == 0 or len(train_control) == 0:
        return [], pd.DataFrame()
    if candidate_idx:
        sub = raw_log[:, candidate_idx]
        ad_mean = np.asarray(sub[train_ad, :].mean(axis=0)).ravel()
        control_mean = np.asarray(sub[train_control, :].mean(axis=0)).ravel()
        candidate_array = np.asarray(candidate_idx, dtype=np.int64)
        allowed = gene_allowed[candidate_array]
    else:
        ad_mean = np.asarray(raw_log[train_ad, :].mean(axis=0)).ravel()
        control_mean = np.asarray(raw_log[train_control, :].mean(axis=0)).ravel()
        candidate_array = np.arange(raw_log.shape[1], dtype=np.int64)
        allowed = gene_allowed
    delta = ad_mean - control_mean
    pooled_mean = (ad_mean + control_mean) / 2.0
    valid = (pooled_mean >= args.selected_gene_min_mean) & allowed
    order = np.argsort(np.abs(delta))[::-1][: args.selected_gene_candidate_top_k]
    selected_local = [int(i) for i in order if valid[int(i)]][: args.selected_gene_top_k]
    selected_gene_idx = [int(candidate_array[i]) for i in selected_local]
    rows = []
    for rank, local_i in enumerate(selected_local, start=1):
        gene_idx = int(candidate_array[local_i])
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "fold": int(fold),
                "selection_space": selection_space,
                "confounder_genes_allowed": bool(args.allow_confounder_genes),
                "rank": int(rank),
                "gene_index": int(gene_idx),
                "gene_symbol": gene_symbols[gene_idx],
                "train_ad_mean_log1p": float(ad_mean[local_i]),
                "train_control_mean_log1p": float(control_mean[local_i]),
                "train_delta_ad_minus_control": float(delta[local_i]),
                "abs_delta": float(abs(delta[local_i])),
            }
        )
    return selected_gene_idx, pd.DataFrame(rows)


def fit_ad_informed_probe(
    name: str,
    cpt_features: np.ndarray,
    raw_log: sparse.csr_matrix,
    module_scores: pd.DataFrame,
    meta: pd.DataFrame,
    candidate_gene_idx: list[int] | None,
    gene_symbols: list[str],
    gene_allowed: np.ndarray,
    args: argparse.Namespace,
    analysis_scope: str = "All cells",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    labels = clean_label(meta[args.condition_column], "Unknown")
    y = labels.map({"Control": 0, "AD": 1}).to_numpy(dtype=int)
    groups = clean_label(meta[args.group_column] if args.group_column in meta.columns else meta["sample_id"], "unknown").to_numpy()
    module_x = module_scores.to_numpy(dtype=np.float32)
    scores = []
    oof = np.full(len(meta), np.nan, dtype=np.float32)
    selected_tables = []
    if len(meta) < 4 or len(np.unique(y)) < 2:
        return pd.DataFrame(scores), oof, pd.DataFrame()

    for fold_i, (train, test) in enumerate(split_by_group(groups, y, 5, args.allow_ungrouped_fallback), start=1):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        selected_idx, selected_table = select_fold_ad_genes(
            raw_log,
            y,
            train,
            candidate_gene_idx,
            gene_symbols,
            gene_allowed,
            args,
            analysis_scope,
            fold_i,
            "curated_modules" if candidate_gene_idx else "genome_wide",
        )
        if not selected_table.empty:
            selected_tables.append(selected_table)
        if selected_idx:
            gene_x = np.asarray(raw_log[:, selected_idx].toarray(), dtype=np.float32)
            x = np.hstack([cpt_features, module_x, gene_x])
        else:
            x = np.hstack([cpt_features, module_x])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        )
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof[test] = prob.astype(np.float32)
        scores.append(
            {
                "analysis_scope": analysis_scope,
                "feature_space": name,
                "fold": fold_i,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_test_groups": int(len(np.unique(groups[test]))),
                "n_selected_genes": int(len(selected_idx)),
                "auroc": float(roc_auc_score(y[test], prob)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro")),
            }
        )
    selected = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    return pd.DataFrame(scores), oof, selected


def curated_module_feature_table(
    module_scores: pd.DataFrame,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    analysis_scope: str = "All cells",
) -> tuple[pd.DataFrame, np.ndarray]:
    scores, oof, _ = fit_probe(
        "Curated AD/NVU module scores",
        module_scores.to_numpy(dtype=np.float32),
        meta,
        args,
        analysis_scope=analysis_scope,
    )
    return scores, oof


def run_celltype_stratified_probes(
    cpt: np.ndarray,
    cpt_lowdim: np.ndarray,
    raw_svd: np.ndarray,
    raw_log: sparse.csr_matrix,
    module_scores: pd.DataFrame,
    meta: pd.DataFrame,
    candidate_gene_idx: list[int] | None,
    gene_symbols: list[str],
    gene_allowed: np.ndarray,
    focus_celltypes: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_tables = []
    gene_tables = []
    celltypes = clean_label(meta[args.celltype_column], "Other/unknown")
    for celltype in focus_celltypes:
        mask = celltypes.eq(celltype).to_numpy()
        if int(mask.sum()) < args.min_celltype_cells:
            continue
        sub_meta = meta.loc[mask].reset_index(drop=True).copy()
        if sub_meta[args.condition_column].nunique() < 2:
            continue
        scope = f"Cell type: {celltype}"
        rows = np.flatnonzero(mask)
        cpt_scores, _, _ = fit_probe("CPT embedding", cpt[rows], sub_meta, args, analysis_scope=scope)
        raw_scores, _, _ = fit_probe("Raw expression SVD", raw_svd[rows], sub_meta, args, analysis_scope=scope)
        module_scores_df, _ = curated_module_feature_table(module_scores.loc[mask].reset_index(drop=True), sub_meta, args, scope)
        informed_scores, _, selected = fit_ad_informed_probe(
            "AD-informed CPT + modules + fold-internal genes",
            cpt_lowdim[rows],
            raw_log[rows, :],
            module_scores.loc[mask].reset_index(drop=True),
            sub_meta,
            candidate_gene_idx,
            gene_symbols,
            gene_allowed,
            args,
            analysis_scope=scope,
        )
        metric_tables.extend([cpt_scores, raw_scores, module_scores_df, informed_scores])
        if not selected.empty:
            gene_tables.append(selected)
    metrics = pd.concat([x for x in metric_tables if not x.empty], ignore_index=True) if metric_tables else pd.DataFrame()
    genes = pd.concat(gene_tables, ignore_index=True) if gene_tables else pd.DataFrame()
    return metrics, genes


def summarize_probe(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores
    group_cols = ["feature_space"]
    if "analysis_scope" in scores.columns:
        group_cols = ["analysis_scope", "feature_space"]
    return (
        scores.groupby(group_cols, dropna=False)
        .agg(
            n_folds=("fold", "count"),
            mean_auroc=("auroc", "mean"),
            sem_auroc=("auroc", lambda s: float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else np.nan),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            sem_balanced_accuracy=("balanced_accuracy", lambda s: float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else np.nan),
            mean_macro_f1=("macro_f1", "mean"),
            mean_selected_genes=("n_selected_genes", "mean") if "n_selected_genes" in scores.columns else ("fold", "count"),
        )
        .reset_index()
    )


def module_correlations(module_scores: pd.DataFrame, meta: pd.DataFrame, score_col: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    celltypes = clean_label(meta[args.celltype_column], "Other/unknown")
    axis = pd.to_numeric(meta[score_col], errors="coerce").to_numpy(dtype=float)
    for celltype in sorted(celltypes.unique()):
        mask = celltypes.eq(celltype).to_numpy() & np.isfinite(axis)
        if int(mask.sum()) < 50:
            continue
        for module in module_scores.columns:
            vals = module_scores.loc[mask, module].to_numpy(dtype=float)
            if np.nanstd(vals) <= 0 or np.nanstd(axis[mask]) <= 0:
                continue
            pear = stats.pearsonr(axis[mask], vals)
            spear = stats.spearmanr(axis[mask], vals)
            rows.append(
                {
                    "celltype": celltype,
                    "module": module,
                    "n_cells": int(mask.sum()),
                    "pearson_r": float(pear.statistic),
                    "pearson_p": float(pear.pvalue),
                    "spearman_r": float(spear.statistic),
                    "spearman_p": float(spear.pvalue),
                    "mean_module_score": float(np.nanmean(vals)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_spearman_r"] = out["spearman_r"].abs()
        out = out.sort_values(["abs_spearman_r", "n_cells"], ascending=[False, False])
    return out


def sample_level_summary(meta: pd.DataFrame, module_scores: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = meta.copy()
    for module in module_scores.columns:
        frame[f"module_{module}"] = module_scores[module].to_numpy()
    group_col = args.group_column if args.group_column in frame.columns else "sample_id"
    agg = {
        "n_cells": ("sample_id", "size"),
        "mean_ad_axis_score": ("cpt_ad_axis_score", "mean"),
    }
    for prob_col in [
        "cpt_oof_ad_probability",
        "raw_svd_oof_ad_probability",
        "module_oof_ad_probability",
        "ad_informed_oof_ad_probability",
    ]:
        if prob_col in frame.columns:
            agg[f"mean_{prob_col}"] = (prob_col, "mean")
    for module in module_scores.columns:
        agg[f"mean_module_{module}"] = (f"module_{module}", "mean")
    return frame.groupby(["condition_inferred", "modality_inferred", group_col], dropna=False).agg(**agg).reset_index()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.analysis_name or args.embedding_dir.name
    out_dir = args.output_dir / name
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    meta_all = read_meta(args.embedding_dir)
    embedding_all = np.load(args.embedding_dir / "embedding.npy", mmap_mode="r")
    if len(meta_all) != embedding_all.shape[0]:
        raise ValueError(f"Embedding/meta row mismatch: {embedding_all.shape[0]} vs {len(meta_all)}")
    for col, fallback in [
        (args.condition_column, "Unknown"),
        (args.celltype_column, "Other/unknown"),
        (args.group_column, "unknown"),
        ("modality_inferred", "unknown"),
    ]:
        if col not in meta_all.columns:
            if col == "modality_inferred" and "modality" in meta_all.columns:
                meta_all[col] = meta_all["modality"]
            else:
                meta_all[col] = fallback
        meta_all[col] = clean_label(meta_all[col], fallback)

    rows = select_analysis_rows(meta_all, args)
    meta = meta_all.iloc[rows].reset_index(drop=True).copy()
    cpt = np.asarray(embedding_all[rows], dtype=np.float32)
    cpt_lowdim = TruncatedSVD(
        n_components=min(args.cpt_svd_components, cpt.shape[1] - 1),
        random_state=args.seed,
    ).fit_transform(cpt)
    cpt_scores, cpt_oof, cpt_coef = fit_probe("CPT embedding", cpt, meta, args, analysis_scope="All cells")
    cpt_center = cpt.mean(axis=0)
    cpt_scale = cpt.std(axis=0)
    cpt_scale[cpt_scale == 0] = 1.0
    meta["cpt_ad_axis_score"] = ((cpt - cpt_center) / cpt_scale).dot(cpt_coef).astype(np.float32)
    meta["cpt_oof_ad_probability"] = cpt_oof

    raw = build_raw_matrix(args.dataset_root, meta)
    raw_log = log1p_sparse(raw)
    raw_svd = TruncatedSVD(n_components=min(args.raw_svd_components, raw.shape[1] - 1), random_state=args.seed).fit_transform(raw)
    raw_scores, raw_oof, _ = fit_probe("Raw expression SVD", raw_svd.astype(np.float32), meta, args, analysis_scope="All cells")
    meta["raw_svd_oof_ad_probability"] = raw_oof

    config = json.loads(args.module_config.read_text(encoding="utf-8"))
    modules = config.get("modules", {})
    module_scores, module_availability = module_score_table(raw, args.dataset_root, args.alias_csv, modules)
    module_probe_scores, module_oof = curated_module_feature_table(module_scores, meta, args, analysis_scope="All cells")
    meta["module_oof_ad_probability"] = module_oof
    genes, _ = load_gene_vocab(args.dataset_root)
    gene_symbols = load_gene_symbols(args.alias_csv, genes)
    gene_allowed = allowed_gene_mask(gene_symbols, args.allow_confounder_genes)
    ad_informed_scores, ad_informed_oof, selected_gene_table = fit_ad_informed_probe(
        "AD-informed CPT + modules + fold-internal genes",
        cpt_lowdim.astype(np.float32),
        raw_log,
        module_scores,
        meta,
        None,
        gene_symbols,
        gene_allowed,
        args,
        analysis_scope="All cells",
    )
    meta["ad_informed_oof_ad_probability"] = ad_informed_oof
    focus_celltypes = config.get(
        "cell_type_focus",
        [
            "Microglia/immune",
            "Astrocyte",
            "Endothelial",
            "Pericyte/mural",
            "VLMC/fibroblast",
            "Excitatory neuron",
            "Inhibitory neuron",
        ],
    )
    celltype_fold_metrics, celltype_selected_genes = run_celltype_stratified_probes(
        cpt,
        cpt_lowdim.astype(np.float32),
        raw_svd.astype(np.float32),
        raw_log,
        module_scores,
        meta,
        None,
        gene_symbols,
        gene_allowed,
        focus_celltypes,
        args,
    )
    correlations = module_correlations(module_scores, meta, "cpt_ad_axis_score", args)
    sample_summary = sample_level_summary(meta, module_scores, args)

    probe_fold_metrics = pd.concat(
        [cpt_scores, raw_scores, module_probe_scores, ad_informed_scores, celltype_fold_metrics],
        ignore_index=True,
    )
    selected_gene_tables = [selected_gene_table]
    if not celltype_selected_genes.empty:
        selected_gene_tables.append(celltype_selected_genes)
    all_selected_gene_table = (
        pd.concat([x for x in selected_gene_tables if not x.empty], ignore_index=True)
        if any(not x.empty for x in selected_gene_tables)
        else pd.DataFrame()
    )
    probe_summary = summarize_probe(probe_fold_metrics)

    meta.to_csv(source_dir / "bi_ad_cpt_cell_axis_scores.csv.gz", index=False)
    probe_fold_metrics.to_csv(source_dir / "bi_ad_cpt_probe_fold_metrics.csv", index=False)
    probe_summary.to_csv(source_dir / "bi_ad_cpt_probe_summary.csv", index=False)
    all_selected_gene_table.to_csv(source_dir / "bi_ad_cpt_fold_internal_selected_genes.csv", index=False)
    module_scores.to_csv(source_dir / "bi_ad_cpt_module_scores.csv.gz", index=False)
    module_availability.to_csv(source_dir / "bi_ad_cpt_module_gene_availability.csv", index=False)
    correlations.to_csv(source_dir / "bi_ad_cpt_module_axis_correlations.csv", index=False)
    sample_summary.to_csv(source_dir / "bi_ad_cpt_sample_level_summary.csv", index=False)
    np.save(source_dir / "bi_ad_cpt_probe_coefficients.npy", cpt_coef)

    summary = {
        "analysis_name": name,
        "embedding_dir": str(args.embedding_dir),
        "n_cells_analyzed": int(len(meta)),
        "condition_counts": meta[args.condition_column].value_counts().to_dict(),
        "celltype_counts": meta[args.celltype_column].value_counts().head(30).to_dict(),
        "probe_summary": probe_summary.to_dict("records"),
        "top_module_axis_correlations": correlations.head(20).to_dict("records") if not correlations.empty else [],
        "outputs": {
            "source_data": str(source_dir),
            "probe_summary": str(source_dir / "bi_ad_cpt_probe_summary.csv"),
            "probe_fold_metrics": str(source_dir / "bi_ad_cpt_probe_fold_metrics.csv"),
            "fold_internal_selected_genes": str(source_dir / "bi_ad_cpt_fold_internal_selected_genes.csv"),
            "module_axis_correlations": str(source_dir / "bi_ad_cpt_module_axis_correlations.csv"),
            "sample_level_summary": str(source_dir / "bi_ad_cpt_sample_level_summary.csv"),
        },
    }
    (out_dir / "bi_ad_cpt_interpretability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
