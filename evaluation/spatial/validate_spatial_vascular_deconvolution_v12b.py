#!/usr/bin/env python
"""Validation package for v12b spatial vascular deconvolution.

Outputs are written into the existing v12b result/figure directory:
- all-chip broad/fine annotations
- chip-level composition and QC
- label-agreement metrics against harmonized existing labels
- marker-expression support tables and Nature-style validation plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import Normalize
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    }
)


BASE = (
    "${OMNICELL_NVU_ROOT}/projects/nvu_vascular/results/"
    "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
)
FIG_BASE = (
    "${OMNICELL_NVU_ROOT}/projects/nvu_vascular/figures/"
    "vascular_omnicell_cpt_gse256490_adult_control_supcon_backbone_nonzero_hvg_all_data"
)
DEFAULT_H5AD = f"{BASE}/spatial_vascular_deconvolution_v12b/spatial/spatial_vascular_deconv_v12b.h5ad"
DEFAULT_RESULT_DIR = f"{BASE}/spatial_vascular_deconvolution_v12b/spatial"
DEFAULT_FIGURE_DIR = f"{FIG_BASE}/spatial_vascular_deconvolution_v12b/spatial"

CLASS_ORDER = ["Endothelial", "Pericyte", "SMC", "Fibroblast_VLMC"]
CLASS_PALETTE = {
    "Endothelial": "#4E79A7",
    "Pericyte": "#B07AA1",
    "SMC": "#E15759",
    "Fibroblast_VLMC": "#59A14F",
    "Low_confidence": "#BFC5CC",
}
CLUSTER_ORDER = [
    "EC00",
    "EC01",
    "EC02",
    "EC03",
    "EC04",
    "PC00",
    "PC01",
    "PC02",
    "SMC00",
    "VLMC00",
    "VLMC01",
    "VLMC02",
    "VLMC03",
]
MODULES = {
    "Endothelial": ["CLDN5", "PECAM1", "VWF", "CDH5", "FLT1", "KDR", "TIE1", "ESAM", "ERG", "ICAM2", "SLC2A1", "ABCB1", "MFSD2A"],
    "Pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9", "NOTCH3", "CSPG4", "MCAM", "DES"],
    "SMC": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK", "MYOCD", "SMTN", "CALD1", "TPM2"],
    "Fibroblast_VLMC": ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "DCN", "LUM", "APOD", "CFD", "SLC6A13"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default=DEFAULT_H5AD)
    p.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    p.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    p.add_argument("--matrix-key", default="expanded_marker_log1p")
    p.add_argument("--gene-key", default="expanded_marker_genes")
    p.add_argument("--dpi", type=int, default=800)
    return p.parse_args()


def save_all(fig: plt.Figure, prefix: Path, dpi: int) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)


def harmonize_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    out = pd.Series("Unknown", index=s.index, dtype=object)
    low = s.str.lower()
    out[low.str.contains("endo|endothelial|capillary|arterial|venous", regex=True, na=False)] = "Endothelial"
    out[low.str.contains("peri|pericyte", regex=True, na=False)] = "Pericyte"
    out[low.str.contains("smc|vsmc|smooth|mural|contractile", regex=True, na=False)] = "SMC"
    out[low.str.contains("fibro|vlmc|leptomeningeal", regex=True, na=False)] = "Fibroblast_VLMC"
    out[s.isin(CLASS_ORDER)] = s[s.isin(CLASS_ORDER)]
    out[s.isin(["Possible_contaminant", "Mixed_spot", "nan", "None", "Unknown", "vascular_unknown"])] = "Unknown"
    return out


def dominant_from_probs(obs: pd.DataFrame, prefix: str, labels: list[str]) -> pd.Series:
    cols = [f"{prefix}{label}" for label in labels]
    arr = obs[cols].to_numpy(float)
    return pd.Series([labels[i] for i in arr.argmax(axis=1)], index=obs.index)


def export_all_annotations(adata: sc.AnnData, result_dir: Path) -> pd.DataFrame:
    class_cols = [f"deconv_class_{x}" for x in CLASS_ORDER]
    cluster_cols = [f"deconv_cluster_{x}" for x in CLUSTER_ORDER if f"deconv_cluster_{x}" in adata.obs]
    base_cols = [
        "sample_id",
        "cohort",
        "condition_inferred",
        "coord_x",
        "coord_y",
        "bio_macro_label",
        "vascular_class",
        "cell_label_original",
        "cluster_annotation",
        "deconv_dominant_class",
        "deconv_dominant_cluster",
        "deconv_confidence",
        "deconv_entropy",
        "deconv_residual",
        "deconv_is_confident",
    ]
    cols = [c for c in base_cols + class_cols + cluster_cols if c in adata.obs]
    table = adata.obs[cols].copy()
    table["harmonized_bio_macro_label"] = harmonize_label(table["bio_macro_label"]) if "bio_macro_label" in table else "Unknown"
    table["harmonized_vascular_class"] = harmonize_label(table["vascular_class"]) if "vascular_class" in table else "Unknown"
    table["harmonized_cell_label_original"] = harmonize_label(table["cell_label_original"]) if "cell_label_original" in table else "Unknown"
    table["harmonized_cluster_annotation"] = harmonize_label(table["cluster_annotation"]) if "cluster_annotation" in table else "Unknown"
    table.to_csv(result_dir / "spatial_vascular_deconv_v12b_all_chip_annotations.csv.gz", compression="gzip")
    return table


def chip_composition(table: pd.DataFrame, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sample, sub in table.groupby("sample_id", observed=False):
        row = {
            "sample_id": sample,
            "cohort": sub["cohort"].astype(str).mode().iloc[0] if "cohort" in sub else "",
            "condition_inferred": sub["condition_inferred"].astype(str).mode().iloc[0] if "condition_inferred" in sub else "",
            "n_spots": int(len(sub)),
            "mean_confidence": float(sub["deconv_confidence"].mean()),
            "median_confidence": float(sub["deconv_confidence"].median()),
            "mean_entropy": float(sub["deconv_entropy"].mean()),
            "mean_residual": float(sub["deconv_residual"].mean()),
            "confident_fraction": float(sub["deconv_is_confident"].astype(bool).mean()),
        }
        for klass in CLASS_ORDER + ["Low_confidence"]:
            row[f"dominant_class_frac_{klass}"] = float((sub["deconv_dominant_class"].astype(str) == klass).mean())
        for klass in CLASS_ORDER:
            col = f"deconv_class_{klass}"
            if col in sub:
                row[f"mean_prop_{klass}"] = float(sub[col].mean())
        for cl in CLUSTER_ORDER:
            row[f"dominant_cluster_frac_{cl}"] = float((sub["deconv_dominant_cluster"].astype(str) == cl).mean())
            col = f"deconv_cluster_{cl}"
            if col in sub:
                row[f"mean_prop_{cl}"] = float(sub[col].mean())
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(result_dir / "spatial_vascular_deconv_v12b_all_chip_composition.csv", index=False)

    long_rows = []
    for _, row in comp.iterrows():
        for klass in CLASS_ORDER:
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "cohort": row["cohort"],
                    "condition_inferred": row["condition_inferred"],
                    "level": "class",
                    "label": klass,
                    "dominant_fraction": row.get(f"dominant_class_frac_{klass}", np.nan),
                    "mean_proportion": row.get(f"mean_prop_{klass}", np.nan),
                }
            )
        for cl in CLUSTER_ORDER:
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "cohort": row["cohort"],
                    "condition_inferred": row["condition_inferred"],
                    "level": "cluster",
                    "label": cl,
                    "dominant_fraction": row.get(f"dominant_cluster_frac_{cl}", np.nan),
                    "mean_proportion": row.get(f"mean_prop_{cl}", np.nan),
                }
            )
    comp_long = pd.DataFrame(long_rows)
    comp_long.to_csv(result_dir / "spatial_vascular_deconv_v12b_all_chip_composition_long.csv", index=False)
    return comp, comp_long


def metrics_for_truth(pred: pd.Series, truth: pd.Series, name: str) -> dict[str, object] | None:
    truth_h = harmonize_label(truth)
    pred_h = pred.astype(str)
    mask = truth_h.isin(CLASS_ORDER) & pred_h.isin(CLASS_ORDER)
    if int(mask.sum()) < 10:
        return None
    return {
        "truth_column": name,
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(truth_h[mask], pred_h[mask])),
        "balanced_accuracy": float(balanced_accuracy_score(truth_h[mask], pred_h[mask])),
        "macro_f1": float(f1_score(truth_h[mask], pred_h[mask], average="macro")),
        "ari": float(adjusted_rand_score(truth_h[mask], pred_h[mask])),
        "nmi": float(normalized_mutual_info_score(truth_h[mask], pred_h[mask])),
    }


def label_agreement(table: pd.DataFrame, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = table["deconv_dominant_class"].astype(str)
    rows = []
    for col in ["bio_macro_label", "vascular_class", "cell_label_original", "cluster_annotation"]:
        if col in table:
            rec = metrics_for_truth(pred, table[col], col)
            if rec:
                rows.append(rec)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(result_dir / "spatial_vascular_deconv_v12b_harmonized_label_agreement.csv", index=False)

    conf_rows = []
    for col in ["bio_macro_label", "vascular_class", "cell_label_original", "cluster_annotation"]:
        if col not in table:
            continue
        truth = harmonize_label(table[col])
        mask = truth.isin(CLASS_ORDER) & pred.isin(CLASS_ORDER)
        if int(mask.sum()) == 0:
            continue
        cm = confusion_matrix(truth[mask], pred[mask], labels=CLASS_ORDER, normalize="true")
        for i, t in enumerate(CLASS_ORDER):
            for j, p in enumerate(CLASS_ORDER):
                conf_rows.append({"truth_column": col, "truth": t, "prediction": p, "row_fraction": float(cm[i, j])})
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(result_dir / "spatial_vascular_deconv_v12b_harmonized_confusion_long.csv", index=False)
    return metrics, conf


def marker_matrix(adata: sc.AnnData, matrix_key: str, gene_key: str) -> tuple[np.ndarray, list[str], dict[str, int]]:
    genes = list(map(str, adata.uns[gene_key]))
    x = np.asarray(adata.obsm[matrix_key], dtype=np.float32)
    return x, genes, {g: i for i, g in enumerate(genes)}


def marker_support(adata: sc.AnnData, result_dir: Path, matrix_key: str, gene_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, genes, gidx = marker_matrix(adata, matrix_key, gene_key)
    class_rows = []
    cluster_rows = []
    labels_class = adata.obs["deconv_dominant_class"].astype(str).to_numpy()
    labels_cluster = adata.obs["deconv_dominant_cluster"].astype(str).to_numpy()
    for module, module_genes in MODULES.items():
        present = [g for g in module_genes if g in gidx]
        cols = [gidx[g] for g in present]
        if not cols:
            continue
        for label in CLASS_ORDER + ["Low_confidence"]:
            mask = labels_class == label
            if not np.any(mask):
                continue
            vals = x[mask][:, cols]
            class_rows.append(
                {
                    "label": label,
                    "module": module,
                    "n": int(mask.sum()),
                    "module_mean": float(vals.mean()),
                    "module_fraction_positive": float((vals > 0).mean()),
                }
            )
            for gene, col in zip(present, cols):
                gv = x[mask, col]
                class_rows.append(
                    {
                        "label": label,
                        "module": module,
                        "gene": gene,
                        "n": int(mask.sum()),
                        "mean_log1p": float(gv.mean()),
                        "fraction_positive": float((gv > 0).mean()),
                    }
                )
        for label in CLUSTER_ORDER + ["Low_confidence"]:
            mask = labels_cluster == label
            if not np.any(mask):
                continue
            vals = x[mask][:, cols]
            cluster_rows.append(
                {
                    "label": label,
                    "module": module,
                    "n": int(mask.sum()),
                    "module_mean": float(vals.mean()),
                    "module_fraction_positive": float((vals > 0).mean()),
                }
            )
            for gene, col in zip(present, cols):
                gv = x[mask, col]
                cluster_rows.append(
                    {
                        "label": label,
                        "module": module,
                        "gene": gene,
                        "n": int(mask.sum()),
                        "mean_log1p": float(gv.mean()),
                        "fraction_positive": float((gv > 0).mean()),
                    }
                )
    class_df = pd.DataFrame(class_rows)
    cluster_df = pd.DataFrame(cluster_rows)
    class_df.to_csv(result_dir / "spatial_vascular_deconv_v12b_marker_support_by_class.csv", index=False)
    cluster_df.to_csv(result_dir / "spatial_vascular_deconv_v12b_marker_support_by_cluster.csv", index=False)
    return class_df, cluster_df


def deconv_marker_correlation(adata: sc.AnnData, result_dir: Path, matrix_key: str, gene_key: str) -> pd.DataFrame:
    x, genes, gidx = marker_matrix(adata, matrix_key, gene_key)
    rows = []
    for module, module_genes in MODULES.items():
        cols = [gidx[g] for g in module_genes if g in gidx]
        if not cols:
            continue
        marker_score = x[:, cols].mean(axis=1)
        prop = adata.obs[f"deconv_class_{module}"].astype(float).to_numpy()
        if np.std(marker_score) > 0 and np.std(prop) > 0:
            corr = float(np.corrcoef(marker_score, prop)[0, 1])
        else:
            corr = np.nan
        rows.append({"class": module, "marker_score_vs_deconv_prop_r": corr, "mean_marker_score": float(marker_score.mean()), "mean_deconv_prop": float(prop.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(result_dir / "spatial_vascular_deconv_v12b_marker_deconv_correlations.csv", index=False)
    return df


def plot_metrics(metrics: pd.DataFrame, prefix: Path, dpi: int) -> None:
    if metrics.empty:
        return
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "ari", "nmi"]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    x = np.arange(len(metric_cols))
    width = 0.17
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    for i, (_, row) in enumerate(metrics.iterrows()):
        vals = [row[m] for m in metric_cols]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width=width, color=colors[i % len(colors)], label=row["truth_column"], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Balanced\nacc.", "Macro F1", "ARI", "NMI"], fontsize=6.4)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Deconvolution agreement with existing labels", loc="left", fontsize=8.2, fontweight="bold")
    ax.legend(fontsize=5.8, ncol=2, loc="upper right")
    ax.grid(axis="y", lw=0.35, color="#D8DEE8")
    save_all(fig, prefix, dpi)
    plt.close(fig)


def plot_confusion(conf: pd.DataFrame, prefix: Path, dpi: int) -> None:
    if conf.empty:
        return
    cols = conf["truth_column"].unique().tolist()
    fig, axes = plt.subplots(1, len(cols), figsize=(2.35 * len(cols), 2.25), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        sub = conf[conf["truth_column"] == col]
        mat = sub.pivot(index="truth", columns="prediction", values="row_fraction").reindex(index=CLASS_ORDER, columns=CLASS_ORDER).fillna(0)
        im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels([x.replace("_", "\n") for x in CLASS_ORDER], rotation=45, ha="right", fontsize=5.4)
        ax.set_yticklabels([x.replace("_", "\n") for x in CLASS_ORDER], fontsize=5.4)
        ax.set_title(col, fontsize=6.8)
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                v = mat.iloc[i, j]
                if v >= 0.15:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2, color="black")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5)
    fig.suptitle("Row-normalized confusion matrices", x=0.02, y=1.02, ha="left", fontsize=8.2, fontweight="bold")
    save_all(fig, prefix, dpi)
    plt.close(fig)


def module_heatmap(module_df: pd.DataFrame, prefix: Path, dpi: int, level: str) -> None:
    rows = module_df[module_df["module_mean"].notna() & module_df["gene"].isna() if "gene" in module_df else module_df.index == module_df.index].copy()
    if "gene" in rows:
        rows = rows[rows["gene"].isna()]
    if rows.empty:
        return
    labels = CLASS_ORDER + ["Low_confidence"] if level == "class" else CLUSTER_ORDER + ["Low_confidence"]
    mat = rows.pivot(index="label", columns="module", values="module_mean").reindex(index=[x for x in labels if x in rows["label"].unique()], columns=CLASS_ORDER)
    fig_h = max(2.2, 0.22 * mat.shape[0] + 0.8)
    fig, ax = plt.subplots(figsize=(3.2, fig_h))
    vals = mat.to_numpy(float)
    im = ax.imshow(vals, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([x.replace("_", "\n") for x in mat.columns], fontsize=6.0)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=5.8)
    ax.set_title(f"Marker-module support by {level}", loc="left", fontsize=8.2, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("mean log1p", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.5)
    save_all(fig, prefix, dpi)
    plt.close(fig)


def marker_dotplot(gene_df: pd.DataFrame, prefix: Path, dpi: int, level: str) -> None:
    if "gene" not in gene_df:
        return
    df = gene_df[gene_df["gene"].notna()].copy()
    if df.empty:
        return
    gene_order = [g for genes in MODULES.values() for g in genes if g in set(df["gene"])]
    label_order = CLASS_ORDER + ["Low_confidence"] if level == "class" else CLUSTER_ORDER + ["Low_confidence"]
    label_order = [x for x in label_order if x in set(df["label"])]
    df["scaled_mean"] = 0.0
    for gene, idx in df.groupby("gene").groups.items():
        vals = df.loc[idx, "mean_log1p"].to_numpy(float)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        df.loc[idx, "scaled_mean"] = 0 if hi <= lo else (vals - lo) / (hi - lo)
    lookup = df.set_index(["label", "gene"])
    fig_w = max(7.6, 0.18 * len(gene_order) + 2.1)
    fig_h = max(2.3, 0.22 * len(label_order) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for yi, label in enumerate(label_order):
        for xi, gene in enumerate(gene_order):
            if (label, gene) not in lookup.index:
                continue
            row = lookup.loc[(label, gene)]
            ax.scatter(xi, yi, s=2 + 52 * float(row["fraction_positive"]), c=[float(row["scaled_mean"])], cmap="viridis", vmin=0, vmax=1, edgecolors="none")
    ax.set_xticks(range(len(gene_order)))
    ax.set_xticklabels(gene_order, rotation=55, ha="right", fontsize=5.4)
    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(label_order, fontsize=5.8)
    ax.set_ylim(len(label_order) - 0.45, -0.55)
    ax.set_title(f"Marker expression by deconvolved {level}", loc="left", fontsize=8.2, fontweight="bold")
    start = 0
    for module, genes in MODULES.items():
        present = [g for g in genes if g in gene_order]
        if present:
            if start > 0:
                ax.axvline(start - 0.5, color="#D8DEE8", lw=0.7)
            ax.text(start + (len(present) - 1) / 2, -0.95, module.replace("_", " "), ha="center", fontsize=5.7)
            start += len(present)
    cax = fig.add_axes([0.91, 0.25, 0.012, 0.22])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), cax=cax)
    cb.ax.tick_params(labelsize=5.2)
    cb.set_label("scaled mean", fontsize=5.2)
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.26, top=0.85)
    save_all(fig, prefix, dpi)
    plt.close(fig)


def plot_chip_composition(comp_long: pd.DataFrame, prefix: Path, dpi: int) -> None:
    class_df = comp_long[(comp_long["level"] == "class") & (comp_long["label"].isin(CLASS_ORDER))].copy()
    if class_df.empty:
        return
    # Sort chips by cohort then endothelial proportion for a readable all-chip panel.
    sort = class_df[class_df["label"] == "Endothelial"][["sample_id", "cohort", "mean_proportion"]].sort_values(["cohort", "mean_proportion"], ascending=[True, False])
    chips = sort["sample_id"].tolist()
    pivot = class_df.pivot(index="sample_id", columns="label", values="mean_proportion").reindex(chips).fillna(0)
    fig_h = max(7.0, 0.055 * len(chips) + 1.4)
    fig, ax = plt.subplots(figsize=(6.4, fig_h))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for klass in CLASS_ORDER:
        vals = pivot[klass].to_numpy()
        ax.barh(y, vals, left=left, height=0.72, color=CLASS_PALETTE[klass], edgecolor="none", label=klass)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([c.split("/")[-1] for c in pivot.index], fontsize=4.6)
    ax.invert_yaxis()
    ax.set_xlabel("mean deconvolved proportion")
    ax.set_title("All-chip vascular composition", loc="left", fontsize=8.2, fontweight="bold")
    ax.legend(loc="lower right", fontsize=5.8, ncol=2)
    ax.grid(axis="x", lw=0.3, color="#D8DEE8")
    save_all(fig, prefix, dpi)
    plt.close(fig)


def plot_correlation(corr: pd.DataFrame, prefix: Path, dpi: int) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    vals = corr.set_index("class").reindex(CLASS_ORDER)["marker_score_vs_deconv_prop_r"].to_numpy()
    ax.bar(np.arange(len(CLASS_ORDER)), vals, color=[CLASS_PALETTE[x] for x in CLASS_ORDER], edgecolor="none")
    ax.axhline(0, color="#202020", lw=0.6)
    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels([x.replace("_", "\n") for x in CLASS_ORDER], fontsize=6)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(min(-0.05, np.nanmin(vals) - 0.05), min(1.0, max(0.2, np.nanmax(vals) + 0.12)))
    ax.set_title("Marker score supports deconvolved proportions", loc="left", fontsize=8.2, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    save_all(fig, prefix, dpi)
    plt.close(fig)


def update_h5ad_with_validation(adata: sc.AnnData, table: pd.DataFrame, out_path: Path) -> None:
    # Write harmonized labels into the h5ad so downstream work has direct access.
    for col in ["harmonized_bio_macro_label", "harmonized_vascular_class", "harmonized_cell_label_original", "harmonized_cluster_annotation"]:
        if col in table:
            adata.obs[col] = table[col].reindex(adata.obs_names).astype("category")
    adata.write_h5ad(out_path, compression="gzip")


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    table = export_all_annotations(adata, result_dir)
    comp, comp_long = chip_composition(table, result_dir)
    metrics, conf = label_agreement(table, result_dir)
    class_marker, cluster_marker = marker_support(adata, result_dir, args.matrix_key, args.gene_key)
    corr = deconv_marker_correlation(adata, result_dir, args.matrix_key, args.gene_key)

    plot_metrics(metrics, figure_dir / "spatial_vascular_deconv_v12b_validation_metrics", args.dpi)
    plot_confusion(conf, figure_dir / "spatial_vascular_deconv_v12b_harmonized_confusion", args.dpi)
    module_heatmap(class_marker, figure_dir / "spatial_vascular_deconv_v12b_marker_module_heatmap_by_class", args.dpi, "class")
    module_heatmap(cluster_marker, figure_dir / "spatial_vascular_deconv_v12b_marker_module_heatmap_by_cluster", args.dpi, "cluster")
    marker_dotplot(class_marker, figure_dir / "spatial_vascular_deconv_v12b_marker_dotplot_by_class", args.dpi, "class")
    marker_dotplot(cluster_marker, figure_dir / "spatial_vascular_deconv_v12b_marker_dotplot_by_cluster", args.dpi, "cluster")
    plot_chip_composition(comp_long, figure_dir / "spatial_vascular_deconv_v12b_all_chip_composition", args.dpi)
    plot_correlation(corr, figure_dir / "spatial_vascular_deconv_v12b_marker_deconv_correlation", args.dpi)

    validated_h5ad = result_dir / "spatial_vascular_deconv_v12b_validated.h5ad"
    update_h5ad_with_validation(adata, table, validated_h5ad)

    summary = {
        "input_h5ad": args.h5ad,
        "validated_h5ad": str(validated_h5ad),
        "n_spots": int(adata.n_obs),
        "n_chips": int(table["sample_id"].astype(str).nunique()),
        "dominant_class_counts": table["deconv_dominant_class"].astype(str).value_counts().to_dict(),
        "dominant_cluster_counts": table["deconv_dominant_cluster"].astype(str).value_counts().to_dict(),
        "metrics": metrics.to_dict(orient="records"),
        "marker_deconv_correlations": corr.to_dict(orient="records"),
        "result_dir": str(result_dir),
        "figure_dir": str(figure_dir),
    }
    (result_dir / "spatial_vascular_deconv_v12b_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
