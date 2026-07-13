#!/usr/bin/env python
"""Evaluate GSE263468 CZI/CELLxGENE as external data 2.

This script creates a panel-A-style donor-level comparison:
CPT-informed AD/NVU features versus raw gene SVD. It does not claim that the
external h5ad already contains the internal CPT latent embedding; the blue bars
are an external projection of the internal AD/NVU CPT-informed gene/module
contract.
"""

from __future__ import annotations
import os

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import anndata as ad
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_REMOTE_BI = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/BI"))
DEFAULT_LOCAL_BI = Path(r"${LOCAL_USER_HOME}\Documents\链接武超-NVU AI\projects\BI")
METHOD_LABELS = {
    "cpt_informed": "CPT-informed features",
    "raw_gene_svd": "Raw gene SVD",
}
METHOD_COLORS = {"cpt_informed": "#2F73B8", "raw_gene_svd": "#8F8F8F"}

MODULE_GENES = {
    "microglia_immune": [
        "APOE",
        "TREM2",
        "TYROBP",
        "C1QA",
        "C1QB",
        "C1QC",
        "SPP1",
        "ITGAX",
        "LPL",
        "CD74",
        "HLA-DRA",
        "HLA-DRB1",
        "HLA-DQB1",
        "AIF1",
        "CSF1R",
        "CTSS",
        "FTL",
        "FTH1",
        "HCST",
        "SRGN",
    ],
    "astrocyte": [
        "GFAP",
        "AQP4",
        "SLC1A2",
        "SLC1A3",
        "ALDH1L1",
        "CLU",
        "APOE",
        "VIM",
        "SERPINA3",
        "SPARCL1",
        "MAOB",
        "S100B",
        "HSPB1",
        "CRYAB",
        "CHI3L1",
        "BEST1",
    ],
    "endothelial": [
        "CLDN5",
        "SLC2A1",
        "ABCB1",
        "MFSD2A",
        "KDR",
        "FLT1",
        "PECAM1",
        "VWF",
        "PLVAP",
        "ABCG2",
        "ICAM1",
        "VCAM1",
        "CLU",
        "NDRG1",
        "SLC38A5",
    ],
    "pericyte_mural": [
        "PDGFRB",
        "RGS5",
        "CSPG4",
        "ABCC9",
        "KCNJ8",
        "ACTA2",
        "TAGLN",
        "MYH11",
        "MCAM",
        "NOTCH3",
        "A2M",
        "SYNM",
        "ADIRF",
    ],
    "vlmc_fibroblast": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "DCN",
        "LUM",
        "FN1",
        "MMP2",
        "APOD",
        "VIM",
    ],
    "neuron": [
        "RBFOX2",
        "SYT1",
        "SNAP25",
        "MAP2",
        "NEFL",
        "NRGN",
        "DAB1",
        "GRIN1",
        "APP",
        "PSEN1",
        "BACE1",
        "MAPT",
        "APLP1",
        "FGF14",
        "KCNQ3",
    ],
    "iron_oxidative": ["FTL", "FTH1", "GSTP1", "MT1E", "MT1X", "PRDX1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bi-root", type=Path, default=DEFAULT_REMOTE_BI)
    parser.add_argument("--h5ad", type=Path, default=None)
    parser.add_argument("--max-raw-genes", type=int, default=600)
    parser.add_argument("--chunk-size", type=int, default=6000)
    parser.add_argument("--min-cells-per-donor", type=int, default=500)
    parser.add_argument("--min-celltype-cells", type=int, default=25)
    parser.add_argument("--bootstrap", type=int, default=3000)
    return parser.parse_args()


def choose_bi_root(path: Path) -> Path:
    if path.exists():
        return path
    return DEFAULT_LOCAL_BI


def canonical_gene(name: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "", str(name).upper())


def clean_name(name: object) -> str:
    return re.sub(r"[^0-9A-Za-z_.:+-]+", "_", str(name))


def find_column(df: pd.DataFrame, candidates: list[str], contains: list[str] | None = None) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if contains:
        for col in df.columns:
            low = str(col).lower()
            if all(token in low for token in contains):
                return str(col)
    return None


def label_from_value(value: object) -> int | float:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "unknown", "na"}:
        return np.nan
    control_tokens = ["normal", "control", "healthy", "no dementia", "non-demented", "unaffected"]
    ad_tokens = ["alzheimer", "dementia", "ad dementia", "alzheimers"]
    if any(token in text for token in control_tokens):
        return 0
    if any(token in text for token in ad_tokens):
        return 1
    if text in {"ad", "case"}:
        return 1
    if text in {"ct", "ctrl"}:
        return 0
    return np.nan


def broad_cell_type(value: object) -> str:
    text = str(value).lower()
    if "micro" in text or "immune" in text or "macroph" in text:
        return "microglia_immune"
    if "astro" in text:
        return "astrocyte"
    if "endothel" in text or "vascular endothelial" in text:
        return "endothelial"
    if "pericy" in text or "smooth muscle" in text or "mural" in text:
        return "pericyte_mural"
    if "fibro" in text or "vlmc" in text or "leptomeningeal" in text:
        return "vlmc_fibroblast"
    if "neuron" in text or "glutamatergic" in text or "gabaergic" in text:
        return "neuron"
    if "oligodendrocyte precursor" in text or text == "opc":
        return "opc"
    if "oligodendro" in text:
        return "oligodendrocyte"
    return "other"


def gene_symbols_from_adata(adata: ad.AnnData) -> list[str]:
    candidates = ["feature_name", "gene_symbol", "gene_symbols", "gene_name", "name"]
    for col in candidates:
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).tolist()
            if len(set(vals)) > len(vals) * 0.8:
                return vals
    return [str(x) for x in adata.var_names]


def load_internal_gene_panel(bi_root: Path, max_raw_genes: int) -> list[str]:
    genes: dict[str, float] = defaultdict(float)
    for module_genes in MODULE_GENES.values():
        for gene in module_genes:
            genes[canonical_gene(gene)] += 10.0
    result_dir = bi_root / "results" / "multicell_integrated_ad_model"
    selected = result_dir / "multicell_model_selected_gene_frequency.csv"
    if selected.exists():
        df = pd.read_csv(selected)
        for _, row in df.iterrows():
            genes[canonical_gene(row["gene"])] += float(row.get("n_folds_selected", 1))
    curated = result_dir / "multicell_curated_gene_sources.csv"
    if curated.exists():
        df = pd.read_csv(curated)
        if "curated_score" in df:
            for _, row in df.iterrows():
                genes[canonical_gene(row["gene"])] += float(row["curated_score"])
    coefs = result_dir / "multicell_model_fold_coefficients.csv"
    if coefs.exists():
        df = pd.read_csv(coefs, usecols=["feature_set", "feature", "abs_coef"])
        df = df[df["feature_set"].eq("raw_gene_svd")]
        gene_hits = df["feature"].astype(str).str.extract(r"gene__([^_]+)__", expand=False)
        tmp = pd.DataFrame({"gene": gene_hits, "abs_coef": df["abs_coef"]}).dropna()
        tmp = tmp.groupby("gene", as_index=False)["abs_coef"].mean().sort_values("abs_coef", ascending=False).head(max_raw_genes)
        for _, row in tmp.iterrows():
            genes[canonical_gene(row["gene"])] += float(row["abs_coef"])
    ordered = sorted(genes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [gene for gene, _ in ordered[:max_raw_genes]]


def orient_scores(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return score
    auc = roc_auc_score(y, score)
    return score if auc >= 0.5 else 1 - score


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int = 17) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def impute_train_mean(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = train.astype(float, copy=True)
    test = test.astype(float, copy=True)
    means = np.nanmean(train, axis=0)
    means[~np.isfinite(means)] = 0.0
    train_inds = np.where(~np.isfinite(train))
    test_inds = np.where(~np.isfinite(test))
    train[train_inds] = means[train_inds[1]]
    test[test_inds] = means[test_inds[1]]
    keep = np.nanstd(train, axis=0) > 1e-8
    if not np.any(keep):
        keep[:] = True
    return train[:, keep], test[:, keep]


def loo_logistic(x: np.ndarray, y: np.ndarray, seed: int = 7) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[i] = False
        x_train, x_test = impute_train_mean(x[train_mask], x[~train_mask])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=3000, solver="liblinear", random_state=seed)
        clf.fit(x_train, y[train_mask])
        pred[i] = clf.predict_proba(x_test)[0, 1]
    return orient_scores(y, pred)


def loo_raw_svd(x: np.ndarray, y: np.ndarray, max_components: int = 10, seed: int = 7) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for i in range(len(y)):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[i] = False
        x_train, x_test = impute_train_mean(x[train_mask], x[~train_mask])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        n_comp = min(max_components, x_train.shape[0] - 1, x_train.shape[1])
        if n_comp < 1:
            score = loo_logistic(x, y, seed=seed)
            return score
        pca = PCA(n_components=n_comp, random_state=seed)
        z_train = pca.fit_transform(x_train)
        z_test = pca.transform(x_test)
        clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=3000, solver="liblinear", random_state=seed)
        clf.fit(z_train, y[train_mask])
        pred[i] = clf.predict_proba(z_test)[0, 1]
    return orient_scores(y, pred)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save_panel(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_auc_bar(auc_df: pd.DataFrame, out_stem: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    datasets = ["all_cell", "cell_type_aware"]
    labels = ["All-cell donor\nfeatures", "Cell-type-aware\nfeatures"]
    width = 0.34
    xbase = np.arange(len(datasets))
    for offset, method in zip([-width / 1.7, width / 1.7], ["cpt_informed", "raw_gene_svd"]):
        sub = auc_df[auc_df["method"].eq(method)].set_index("feature_scope").loc[datasets]
        xs = xbase + offset
        vals = sub["auroc"].to_numpy(float)
        err = np.vstack([vals - sub["auroc_ci_low"].to_numpy(float), sub["auroc_ci_high"].to_numpy(float) - vals])
        ax.bar(xs, vals, width=width, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.5, label=METHOD_LABELS[method])
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        for x, val in zip(xs, vals):
            ax.text(x, val + 0.026, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0.5, color="#BDBDBD", lw=0.8, ls="--")
    ax.set_xticks(xbase)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("Held-out AUROC")
    n_donors = int(auc_df["n_donors"].iloc[0])
    ax.set_title(f"External data 2: GSE263468 (n={n_donors} donors)", pad=7)
    ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0.05), fontsize=6.0, handlelength=1.3, borderaxespad=0)
    ax.text(-0.18, 1.08, "F", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")
    fig.tight_layout(pad=0.7)
    save_panel(fig, out_stem)


def main() -> None:
    args = parse_args()
    bi_root = choose_bi_root(args.bi_root)
    base = bi_root / "external_validation" / "GSE263468_CZI"
    h5ad_path = args.h5ad or (base / "raw" / "GSE263468_CZI_all_cells.h5ad")
    out_dir = base / "results"
    fig_dir = bi_root / "figures" / "nature_individual_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    donor_col = find_column(obs, ["donor_id", "Donor ID", "individual", "subject_id", "sample_id"], contains=["donor"])
    disease_col = find_column(obs, ["disease", "diagnosis", "condition", "disease__ontology_label"])
    celltype_col = find_column(obs, ["cell_type", "cell type", "author_cell_type", "CellType"], contains=["cell", "type"])
    if donor_col is None or disease_col is None:
        raise RuntimeError(f"Could not infer donor/disease columns. obs columns: {list(obs.columns)}")
    if celltype_col is None:
        obs["cell_type__fallback"] = "all"
        celltype_col = "cell_type__fallback"
    obs["label"] = obs[disease_col].map(label_from_value)
    obs["broad_cell_type"] = obs[celltype_col].map(broad_cell_type)
    obs["donor_id__eval"] = obs[donor_col].astype(str)
    obs = obs[np.isfinite(obs["label"].to_numpy(float))].copy()
    donor_counts = obs.groupby("donor_id__eval").size()
    keep_donors = donor_counts[donor_counts >= args.min_cells_per_donor].index
    obs = obs[obs["donor_id__eval"].isin(keep_donors)].copy()
    donor_labels = obs.groupby("donor_id__eval")["label"].agg(lambda s: int(round(float(s.iloc[0]))))
    donor_labels = donor_labels[donor_labels.isin([0, 1])]
    donors = donor_labels.index.tolist()
    y = donor_labels.to_numpy(int)
    if len(np.unique(y)) != 2:
        raise RuntimeError(f"Need both AD and Control donors; found label counts {dict(pd.Series(y).value_counts())}")
    donor_index = {d: i for i, d in enumerate(donors)}

    raw_gene_panel = load_internal_gene_panel(bi_root, args.max_raw_genes)
    gene_symbols = gene_symbols_from_adata(adata)
    gene_map: dict[str, int] = {}
    for idx, gene in enumerate(gene_symbols):
        key = canonical_gene(gene)
        if key and key not in gene_map:
            gene_map[key] = idx
    available = [g for g in raw_gene_panel if g in gene_map]
    if len(available) < 20:
        raise RuntimeError(f"Too few internal genes found in external h5ad: {len(available)}")
    gene_indices = [gene_map[g] for g in available]
    gene_pos = {g: i for i, g in enumerate(available)}

    group_order = [
        "all",
        "microglia_immune",
        "astrocyte",
        "endothelial",
        "pericyte_mural",
        "vlmc_fibroblast",
        "neuron",
        "oligodendrocyte",
        "opc",
    ]
    group_index = {g: i for i, g in enumerate(group_order)}
    n_donors = len(donors)
    n_genes = len(available)
    n_groups = len(group_order)
    gene_sum = np.zeros((n_donors, n_genes), dtype=np.float64)
    gene_n = np.zeros(n_donors, dtype=np.int64)
    group_sum = np.zeros((n_donors, n_groups, n_genes), dtype=np.float64)
    group_n = np.zeros((n_donors, n_groups), dtype=np.int64)

    donor_codes_full = obs["donor_id__eval"].map(donor_index).fillna(-1).to_numpy(int)
    group_codes_full = obs["broad_cell_type"].map(group_index).fillna(group_index["all"]).to_numpy(int)
    selected_obs_positions = obs.index
    if selected_obs_positions.dtype.kind in {"i", "u"}:
        obs_positions = selected_obs_positions.to_numpy(int)
    else:
        obs_positions = np.flatnonzero(adata.obs_names.isin(selected_obs_positions))
    order = np.argsort(obs_positions)
    obs_positions = obs_positions[order]
    donor_codes_full = donor_codes_full[order]
    group_codes_full = group_codes_full[order]

    for start in range(0, len(obs_positions), args.chunk_size):
        end = min(start + args.chunk_size, len(obs_positions))
        if start == 0 or end == len(obs_positions) or (start // args.chunk_size) % 10 == 0:
            print(f"aggregating cells {start:,}-{end:,} of {len(obs_positions):,}", flush=True)
        rows = obs_positions[start:end]
        donor_codes = donor_codes_full[start:end]
        group_codes = group_codes_full[start:end]
        mat = adata[rows, gene_indices].X
        if sparse.issparse(mat):
            mat = mat.toarray()
        mat = np.asarray(mat, dtype=np.float32)
        finite = np.isfinite(mat)
        if not finite.all():
            mat[~finite] = 0.0
        np.add.at(gene_sum, donor_codes, mat)
        np.add.at(gene_n, donor_codes, 1)
        for donor_code in np.unique(donor_codes):
            dm = donor_codes == donor_code
            for group_code in np.unique(group_codes[dm]):
                gm = dm & (group_codes == group_code)
                group_sum[donor_code, group_code, :] += mat[gm].sum(axis=0)
                group_n[donor_code, group_code] += int(gm.sum())
                group_sum[donor_code, group_index["all"], :] += mat[gm].sum(axis=0)
                group_n[donor_code, group_index["all"]] += int(gm.sum())

    raw_all = gene_sum / np.maximum(gene_n[:, None], 1)
    raw_group = group_sum / np.maximum(group_n[:, :, None], 1)
    raw_group = np.where(group_n[:, :, None] < args.min_celltype_cells, np.nan, raw_group)

    feature_records = pd.DataFrame({"donor_id": donors, "label": y, "group": np.where(y == 1, "AD", "Control"), "n_cells": gene_n})
    cpt_all_features = []
    cpt_all_names = []
    cpt_celltype_features = []
    cpt_celltype_names = []
    for module, module_genes in MODULE_GENES.items():
        idx = [gene_pos[canonical_gene(g)] for g in module_genes if canonical_gene(g) in gene_pos]
        if not idx:
            continue
        vals = raw_all[:, idx].mean(axis=1)
        cpt_all_features.append(vals)
        cpt_all_names.append(f"module__all__{module}")
        feature_records[f"module__all__{module}"] = vals
        for group in group_order[1:]:
            gi = group_index[group]
            gvals = raw_group[:, gi, :][:, idx].mean(axis=1)
            cpt_celltype_features.append(gvals)
            cpt_celltype_names.append(f"module__{group}__{module}")
            feature_records[f"module__{group}__{module}"] = gvals
    for group in group_order[1:]:
        gi = group_index[group]
        frac = group_n[:, gi] / np.maximum(gene_n, 1)
        cpt_celltype_features.append(frac)
        cpt_celltype_names.append(f"cell_fraction__{group}")
        feature_records[f"cell_fraction__{group}"] = frac
    selected_gene_names = available[: min(80, len(available))]
    for gene in selected_gene_names:
        vals = raw_all[:, gene_pos[gene]]
        cpt_all_features.append(vals)
        cpt_all_names.append(f"selected_gene__all__{gene}")
        feature_records[f"selected_gene__all__{gene}"] = vals

    cpt_all = np.column_stack(cpt_all_features)
    cpt_celltype = np.column_stack(cpt_all_features + cpt_celltype_features)
    raw_group_flat = raw_group[:, 1:, :].reshape(n_donors, -1)

    predictions = []
    auc_rows = []
    model_inputs = [
        ("all_cell", "cpt_informed", cpt_all, loo_logistic),
        ("all_cell", "raw_gene_svd", raw_all, loo_raw_svd),
        ("cell_type_aware", "cpt_informed", cpt_celltype, loo_logistic),
        ("cell_type_aware", "raw_gene_svd", raw_group_flat, loo_raw_svd),
    ]
    for feature_scope, method, matrix, runner in model_inputs:
        score = runner(matrix, y)
        auc = float(roc_auc_score(y, score))
        lo, hi = bootstrap_auc_ci(y, score, args.bootstrap)
        auc_rows.append(
            {
                "dataset": "GSE263468_CZI",
                "feature_scope": feature_scope,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_donors": int(n_donors),
                "n_ad": int(y.sum()),
                "n_control": int((1 - y).sum()),
                "n_features": int(matrix.shape[1]),
                "auroc": auc,
                "auroc_ci_low": lo,
                "auroc_ci_high": hi,
            }
        )
        for donor, label, n_cells, pred in zip(donors, y, gene_n, score):
            predictions.append(
                {
                    "dataset": "GSE263468_CZI",
                    "feature_scope": feature_scope,
                    "method": method,
                    "donor_id": donor,
                    "label": int(label),
                    "group": "AD" if label == 1 else "Control",
                    "n_cells": int(n_cells),
                    "pred_prob": float(pred),
                }
            )

    auc_df = pd.DataFrame(auc_rows)
    pred_df = pd.DataFrame(predictions)
    source_prefix = "source_GSE263468_CZI_cpt_informed_vs_raw"
    auc_df.to_csv(out_dir / f"{source_prefix}_auc_bootstrap.csv", index=False)
    pred_df.to_csv(out_dir / f"{source_prefix}_predictions.csv", index=False)
    feature_records.to_csv(out_dir / "GSE263468_CZI_donor_feature_matrix.csv", index=False)
    pd.DataFrame({"gene": available}).to_csv(out_dir / "GSE263468_CZI_available_internal_gene_panel.csv", index=False)
    auc_df.to_csv(fig_dir / f"{source_prefix}_auc_bootstrap.csv", index=False)
    pred_df.to_csv(fig_dir / f"{source_prefix}_predictions.csv", index=False)
    make_auc_bar(auc_df, fig_dir / "F_GSE263468_CZI_external_AUROC_bar")

    report = {
        "dataset": "GSE263468_CZI",
        "h5ad": str(h5ad_path),
        "note": "External data 2 from CZI/CELLxGENE. Blue bars are CPT-informed AD/NVU feature projection, not de novo external CPT latent embedding.",
        "obs_columns": {"donor": donor_col, "disease": disease_col, "cell_type": celltype_col},
        "n_cells_used": int(gene_n.sum()),
        "n_donors": int(n_donors),
        "label_counts": pd.Series(y).map({0: "Control", 1: "AD"}).value_counts().to_dict(),
        "n_internal_genes_available": int(len(available)),
        "auc": auc_df.to_dict("records"),
        "outputs": {
            "auc_source": str(out_dir / f"{source_prefix}_auc_bootstrap.csv"),
            "predictions": str(out_dir / f"{source_prefix}_predictions.csv"),
            "donor_feature_matrix": str(out_dir / "GSE263468_CZI_donor_feature_matrix.csv"),
            "panel": str(fig_dir / "F_GSE263468_CZI_external_AUROC_bar.png"),
        },
    }
    (out_dir / "GSE263468_CZI_external_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    adata.file.close()


if __name__ == "__main__":
    main()
