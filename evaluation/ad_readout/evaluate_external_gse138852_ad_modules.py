#!/usr/bin/env python
"""External single-cell AD validation using GSE138852.

This is a lightweight external sanity check for the AD/NVU gene-module axes
learned in the internal BI model. GSE138852 provides cell-level AD/Control
labels and broad cell types for human entorhinal cortex nuclei, but not clean
donor-level metadata in the GEO covariates table. Therefore the primary output
is cell-level and cell-type-stratified external support, with bootstrap CIs;
it should not be described as donor-level validation.
"""

from __future__ import annotations
import os

import argparse
import gzip
import json
from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_BI_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/BI"))
LOCAL_BI_ROOT = Path(r"${LOCAL_USER_HOME}\Documents\链接武超-NVU AI\projects\BI")

MODULE_GENES = {
    "microglia_immune": ["APOE", "TYROBP", "FCER1G", "C1QA", "C1QB", "C1QC", "CD74", "HLA-DRA", "HLA-DQB1", "LGALS3", "SERPING1"],
    "astrocyte_stress": ["GFAP", "AQP4", "CLU", "VIM", "BEST1", "HSPB1", "CRYAB", "SERPINA3"],
    "endothelial_pericyte": ["CLDN5", "SLC2A1", "NDRG1", "RGS5", "PDGFRB", "ADIRF", "VWF", "PECAM1"],
    "iron_oxidative": ["FTL", "FTH1", "GSTP1", "MT1E", "MT1X", "PRDX1"],
    "neuron_ad": ["APP", "PSEN1", "BACE1", "MAPT", "SNAP25", "SYT1", "NRGN"],
    "selected_internal": ["ADIRF", "BEST1", "CAPS", "FABP5", "FCER1G", "FTH1", "FTL", "GSTP1", "HCST", "LGALS1", "TRIP6", "TYROBP", "VIM", "LGALS3", "SERPING1", "HLA-DQB1"],
}

CELLTYPE_MAP = {
    "mg": "microglia",
    "astro": "astrocyte",
    "endo": "endothelial",
    "oligo": "oligodendrocyte",
    "OPC": "OPC",
    "neuron": "neuron",
    "unID": "unID",
    "doublet": "doublet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bi-root", type=Path, default=DEFAULT_BI_ROOT)
    parser.add_argument("--min-cells-per-class", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def canonical_gene(gene: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "", str(gene).upper())


def choose_bi_root(path: Path) -> Path:
    if path.exists():
        return path
    return LOCAL_BI_ROOT


def load_covariates(raw_dir: Path) -> pd.DataFrame:
    cov = pd.read_csv(raw_dir / "GSE138852_covariates.csv.gz")
    cov = cov.rename(columns={cov.columns[0]: "cell_id"})
    cov["label"] = cov["oupSample.batchCond"].map({"AD": 1, "ct": 0}).astype(int)
    cov["group"] = cov["oupSample.batchCond"].map({"AD": "AD", "ct": "Control"})
    cov["cell_type"] = cov["oupSample.cellType"].map(CELLTYPE_MAP).fillna(cov["oupSample.cellType"].astype(str))
    cov["barcode_suffix"] = cov["cell_id"].str.extract(r"_([^_]+(?:_[^_]+)?)$", expand=False)
    return cov


def count_rows(counts_path: Path) -> int:
    with gzip.open(counts_path, "rt", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle) - 1


def find_module_gene_rows(counts_path: Path, wanted: set[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    with gzip.open(counts_path, "rt", encoding="utf-8", errors="replace") as handle:
        _ = handle.readline()
        for idx, line in enumerate(handle):
            gene = line.split(",", 1)[0].strip().strip('"')
            key = canonical_gene(gene)
            if key in wanted and key not in found:
                found[key] = idx
    return found


def read_selected_gene_counts(counts_path: Path, gene_rows: dict[str, int]) -> tuple[list[str], np.ndarray]:
    if not gene_rows:
        return [], np.empty((0, 0), dtype=np.float32)
    row_to_gene = {row: gene for gene, row in gene_rows.items()}
    max_row = max(row_to_gene)
    genes: list[str] = []
    rows: list[np.ndarray] = []
    with gzip.open(counts_path, "rt", encoding="utf-8", errors="replace") as handle:
        _ = handle.readline()
        for idx, line in enumerate(handle):
            if idx > max_row:
                break
            if idx not in row_to_gene:
                continue
            parts = line.rstrip("\n").split(",")
            genes.append(row_to_gene[idx])
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    return genes, np.vstack(rows)


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int = 13) -> tuple[float, float]:
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


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    auc = roc_auc_score(y, score)
    return float(max(auc, 1 - auc))


def module_scores(genes: list[str], counts: np.ndarray, cov: pd.DataFrame) -> pd.DataFrame:
    gene_index = {g: i for i, g in enumerate(genes)}
    lib = counts.sum(axis=0)
    lib[lib <= 0] = 1
    log_cpm = np.log1p((counts / lib) * 1e4)
    features = pd.DataFrame(index=cov["cell_id"])
    for module, module_genes in MODULE_GENES.items():
        idx = [gene_index[canonical_gene(g)] for g in module_genes if canonical_gene(g) in gene_index]
        if idx:
            features[module] = log_cpm[idx, :].mean(axis=0)
        else:
            features[module] = 0.0
    return features


def run_probe(features: pd.DataFrame, cov: pd.DataFrame, n_splits: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = cov["label"].to_numpy(int)
    x = features.to_numpy(float)
    n_splits = min(n_splits, int(np.bincount(y).min()))
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, solver="liblinear", random_state=7),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    prob = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba")[:, 1]
    if roc_auc_score(y, prob) < 0.5:
        prob = 1 - prob
    pred = cov[["cell_id", "group", "label", "cell_type", "barcode_suffix"]].copy()
    pred["external_ad_score"] = prob
    rows = []
    for key, sub in [("all_cells", pred)] + [(ct, pred[pred["cell_type"] == ct]) for ct in sorted(pred["cell_type"].unique())]:
        yy = sub["label"].to_numpy(int)
        ss = sub["external_ad_score"].to_numpy(float)
        if len(sub) < 20 or len(np.unique(yy)) < 2:
            continue
        rows.append(
            {
                "stratum": key,
                "n_cells": int(len(sub)),
                "n_ad": int(yy.sum()),
                "n_control": int((1 - yy).sum()),
                "auroc_oriented": safe_auc(yy, ss),
                "mean_score_ad": float(np.mean(ss[yy == 1])),
                "mean_score_control": float(np.mean(ss[yy == 0])),
                "delta_ad_minus_control": float(np.mean(ss[yy == 1]) - np.mean(ss[yy == 0])),
            }
        )
    return pd.DataFrame(rows), pred


def summarize_modules(features: pd.DataFrame, cov: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    y = cov["label"].to_numpy(int)
    for module in features.columns:
        score = features[module].to_numpy(float)
        auc = safe_auc(y, score)
        lo, hi = bootstrap_auc_ci(y, score if roc_auc_score(y, score) >= 0.5 else -score, n_boot)
        rows.append(
            {
                "module": module,
                "n_genes_found": int(sum(canonical_gene(g) in set(features.attrs.get("genes_found", [])) for g in MODULE_GENES[module])),
                "genes_found": ";".join([g for g in MODULE_GENES[module] if canonical_gene(g) in set(features.attrs.get("genes_found", []))]),
                "auroc_oriented": auc,
                "auroc_ci_low": lo,
                "auroc_ci_high": hi,
                "mean_ad": float(score[y == 1].mean()),
                "mean_control": float(score[y == 0].mean()),
                "delta_ad_minus_control": float(score[y == 1].mean() - score[y == 0].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("auroc_oriented", ascending=False)


def sample_level_summary(pred: pd.DataFrame, features: pd.DataFrame, cov: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(37)
    merged = pred.merge(features.reset_index().rename(columns={"index": "cell_id"}), on="cell_id", how="left")
    rows = []
    celltype_rows = []
    sample_order = []
    for group_prefix in ["Ct", "AD"]:
        suffixes = sorted([s for s in merged["barcode_suffix"].dropna().unique() if str(s).startswith(group_prefix)])
        sample_order.extend(suffixes)
    for sample_id in sample_order:
        sub = merged[merged["barcode_suffix"] == sample_id].copy()
        scores = sub["external_ad_score"].to_numpy(float)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(scores), len(scores))
            boots.append(float(np.mean(scores[idx])))
        rec = {
            "sample_id": sample_id,
            "group": sub["group"].iloc[0],
            "label": int(sub["label"].iloc[0]),
            "n_cells": int(len(sub)),
            "mean_external_ad_score": float(np.mean(scores)),
            "sem_external_ad_score": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
            "bootstrap_ci_low": float(np.percentile(boots, 2.5)),
            "bootstrap_ci_high": float(np.percentile(boots, 97.5)),
        }
        for module in MODULE_GENES:
            rec[f"module_mean__{module}"] = float(sub[module].mean())
        rows.append(rec)
        for cell_type, ct_sub in sub.groupby("cell_type"):
            if len(ct_sub) < 10:
                continue
            celltype_rows.append(
                {
                    "sample_id": sample_id,
                    "group": sub["group"].iloc[0],
                    "label": int(sub["label"].iloc[0]),
                    "cell_type": cell_type,
                    "n_cells": int(len(ct_sub)),
                    "mean_external_ad_score": float(ct_sub["external_ad_score"].mean()),
                }
            )
    sample_df = pd.DataFrame(rows)
    if len(sample_df) and len(sample_df["label"].unique()) == 2:
        sample_df.attrs["sample_level_auroc"] = safe_auc(
            sample_df["label"].to_numpy(int),
            sample_df["mean_external_ad_score"].to_numpy(float),
        )
    return sample_df, pd.DataFrame(celltype_rows)


def make_sample_level_bar_figure(out_dir: Path, sample_df: pd.DataFrame) -> None:
    if sample_df.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    sample_df = sample_df.copy()
    colors = sample_df["group"].map({"Control": "#8E8E8E", "AD": "#B63D3D"}).fillna("#999999").tolist()
    x = np.arange(len(sample_df))
    y = sample_df["mean_external_ad_score"].to_numpy(float)
    yerr = np.vstack(
        [
            y - sample_df["bootstrap_ci_low"].to_numpy(float),
            sample_df["bootstrap_ci_high"].to_numpy(float) - y,
        ]
    )
    ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.5, width=0.72)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    for xi, (_, row) in zip(x, sample_df.iterrows()):
        ax.text(xi, row["mean_external_ad_score"] + yerr[1, xi] + 0.02, f"n={int(row['n_cells'])}", ha="center", va="bottom", fontsize=5.5)
    ctrl_mean = sample_df.loc[sample_df["group"] == "Control", "mean_external_ad_score"].mean()
    ad_mean = sample_df.loc[sample_df["group"] == "AD", "mean_external_ad_score"].mean()
    ax.axhline(ctrl_mean, color="#6E6E6E", lw=0.8, ls="--", label=f"Control mean {ctrl_mean:.2f}")
    ax.axhline(ad_mean, color="#B63D3D", lw=0.8, ls=":", label=f"AD mean {ad_mean:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_df["sample_id"], rotation=35, ha="right")
    ax.set_ylabel("Mean external AD score")
    sample_auc = sample_df.attrs.get("sample_level_auroc", np.nan)
    title = "GSE138852 sample-pool level AD score"
    if np.isfinite(sample_auc):
        title += f" (AUROC={sample_auc:.3f})"
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, max(0.75, float((y + yerr[1]).max() + 0.12))))
    ax.legend(loc="upper left", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out_dir / "GSE138852_sample_level_ad_score_bar.svg", bbox_inches="tight")
    fig.savefig(out_dir / "GSE138852_sample_level_ad_score_bar.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "GSE138852_sample_level_ad_score_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure(out_dir: Path, probe_summary: pd.DataFrame, module_summary: pd.DataFrame, pred: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.0], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    top = probe_summary.head(8).iloc[::-1]
    ax.barh(np.arange(len(top)), top["auroc_oriented"], color="#2A6FBB", edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="#BDBDBD", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["stratum"])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("External AUROC")
    ax.set_title("GSE138852 cell-type support")
    ax.text(-0.14, 1.02, "A", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

    ax = fig.add_subplot(gs[0, 1])
    mod = module_summary.head(6).iloc[::-1]
    ax.barh(np.arange(len(mod)), mod["auroc_oriented"], color="#B9884D", edgecolor="black", linewidth=0.4)
    ax.axvline(0.5, color="#BDBDBD", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(mod)))
    ax.set_yticklabels(mod["module"])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Module AUROC")
    ax.set_title("Internal AD/NVU gene modules")
    ax.text(-0.14, 1.02, "B", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(1)
    sub = pred[pred["cell_type"].isin(["astrocyte", "microglia", "endothelial", "neuron", "oligodendrocyte", "OPC"])].copy()
    order = ["microglia", "astrocyte", "endothelial", "neuron", "oligodendrocyte", "OPC"]
    xpos = {ct: i for i, ct in enumerate(order)}
    colors = {0: "#6E6E6E", 1: "#B63D3D"}
    for label in [0, 1]:
        ss = sub[sub["label"] == label]
        xs = ss["cell_type"].map(xpos).to_numpy(float) + rng.normal(0, 0.08, len(ss)) + (-0.12 if label == 0 else 0.12)
        ax.scatter(xs, ss["external_ad_score"], s=2, color=colors[label], alpha=0.18, linewidths=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("External AD score")
    ax.set_title("Cell-level score distribution")
    ax.text(-0.14, 1.02, "C", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.suptitle("External single-cell support for AD/NVU axes in GSE138852", fontsize=9.2, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.78)
    fig.savefig(out_dir / "GSE138852_external_ad_module_validation.svg", bbox_inches="tight")
    fig.savefig(out_dir / "GSE138852_external_ad_module_validation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "GSE138852_external_ad_module_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bi_root = choose_bi_root(args.bi_root)
    base = bi_root / "external_validation" / "GSE138852"
    raw_dir = base / "raw"
    out_dir = base / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = raw_dir / "GSE138852_counts.csv.gz"
    cov = load_covariates(raw_dir)
    wanted = {canonical_gene(g) for genes in MODULE_GENES.values() for g in genes}
    row_map = find_module_gene_rows(counts_path, wanted)
    genes, counts = read_selected_gene_counts(counts_path, row_map)
    features = module_scores(genes, counts, cov)
    features.attrs["genes_found"] = genes
    features.to_csv(out_dir / "GSE138852_external_module_features.csv")
    probe_summary, pred = run_probe(features, cov, args.n_splits)
    module_summary = summarize_modules(features, cov, args.bootstrap)
    sample_df, sample_celltype_df = sample_level_summary(pred, features, cov, args.bootstrap)
    probe_summary.to_csv(out_dir / "GSE138852_external_probe_summary.csv", index=False)
    module_summary.to_csv(out_dir / "GSE138852_external_module_summary.csv", index=False)
    sample_df.to_csv(out_dir / "GSE138852_external_sample_level_summary.csv", index=False)
    sample_celltype_df.to_csv(out_dir / "GSE138852_external_sample_celltype_summary.csv", index=False)
    pred.to_csv(out_dir / "GSE138852_external_cell_scores.csv", index=False)
    make_figure(out_dir, probe_summary, module_summary, pred)
    make_sample_level_bar_figure(out_dir, sample_df)
    report = {
        "dataset": "GSE138852",
        "note": "External single-cell AD/Control sanity check. GEO covariates lack clean donor-level metadata, so this is not donor-level validation.",
        "n_cells": int(len(cov)),
        "condition_counts": cov["group"].value_counts().to_dict(),
        "cell_type_counts": cov["cell_type"].value_counts().to_dict(),
        "genes_found": genes,
        "sample_pool_counts": sample_df[["sample_id", "group", "n_cells", "mean_external_ad_score"]].to_dict("records"),
        "sample_pool_level_auroc": sample_df.attrs.get("sample_level_auroc", None),
        "outputs": {
            "probe_summary": str(out_dir / "GSE138852_external_probe_summary.csv"),
            "module_summary": str(out_dir / "GSE138852_external_module_summary.csv"),
            "cell_scores": str(out_dir / "GSE138852_external_cell_scores.csv"),
            "sample_level_summary": str(out_dir / "GSE138852_external_sample_level_summary.csv"),
            "sample_level_bar": str(out_dir / "GSE138852_sample_level_ad_score_bar.png"),
            "figure_png": str(out_dir / "GSE138852_external_ad_module_validation.png"),
        },
    }
    (out_dir / "GSE138852_external_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
