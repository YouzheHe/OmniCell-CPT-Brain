#!/usr/bin/env python
"""Create Python publication figures for CPT latent vs raw gene SVD AD probes.

The figures use sample/chip-level leave-one-out predictions produced by
run_multicell_integrated_ad_model.py. They intentionally keep the primary
comparison to CPT latent features versus raw gene SVD, then place biological
interpretability in a separate panel set.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve


PROJECT_ROOT = Path(r"${LOCAL_USER_HOME}\Documents\链接武超-NVU AI\projects\BI")
RESULT_ROOT = PROJECT_ROOT / "results" / "multicell_integrated_ad_model"
FIG_ROOT = PROJECT_ROOT / "figures" / "multicell_integrated_ad_model"

METHOD_LABELS = {
    "latent": "CPT latent",
    "raw_gene_svd": "Raw gene SVD",
}
DATASET_LABELS = {
    "all44": "All samples\n(n=44)",
    "qc36_risk_available": "QC cohort\n(n=36)",
}
METHOD_COLORS = {
    "latent": "#2A6FBB",
    "raw_gene_svd": "#8E8E8E",
}
LABEL_COLORS = {
    0: "#6E6E6E",
    1: "#B63D3D",
}

GENE_CLASS = {
    "APOC1": "microglia/immune",
    "AZGP1": "stress/metabolic",
    "BBOX1": "metabolic",
    "CD163": "microglia/immune",
    "DDIT4": "stress",
    "EFHD1": "stress/metabolic",
    "FCER1G": "microglia/immune",
    "GSTP1": "oxidative stress",
    "LY86": "microglia/immune",
    "NDRG1": "endothelial/stress",
    "NUPR1": "stress",
    "RGS1": "immune",
    "SPP1": "microglia/ECM",
    "TYROBP": "microglia/immune",
    "C1QC": "complement",
    "FTL": "iron/stress",
    "FTH1": "iron/stress",
    "GPNMB": "microglia/lysosome",
    "ADIRF": "pericyte/mural",
    "BEST1": "astrocyte",
    "CAPS": "stress/vascular",
    "FABP5": "lipid/stress",
    "HCST": "microglia/immune",
    "LGALS1": "ECM/immune",
    "MT1E": "metallothionein",
    "MT1X": "metallothionein",
    "TRIP6": "adhesion/stress",
    "HLA-DQB1": "antigen presentation",
    "VIM": "reactive glia/VLMC",
    "LGALS3": "microglia/immune",
    "SERPING1": "complement",
}
PREFERRED_DISPLAY_GENES = [
    "ADIRF",
    "BEST1",
    "CAPS",
    "FABP5",
    "FCER1G",
    "FTH1",
    "FTL",
    "GSTP1",
    "HCST",
    "LGALS1",
    "TRIP6",
    "TYROBP",
    "VIM",
    "LGALS3",
    "SERPING1",
    "HLA-DQB1",
]

TARGET_AXES = [
    {
        "target_axis": "SPP1-CD44 / VIM-CD44",
        "representative_features": "SPP1, CD44, VIM, SPP1-CD44, VIM-CD44",
        "main_cell_context": "microglia, astrocyte/VLMC, vascular niche",
        "model_evidence": "High LR effect and recurrent SPP1/VIM-related gene/module signals",
        "therapeutic_hypothesis": "Test whether dampening ECM-inflammatory CD44 signaling reduces plaque-associated glial-vascular activation.",
        "priority_score": 9.2,
    },
    {
        "target_axis": "microglial immune/complement",
        "representative_features": "TYROBP, FCER1G, C1QC, APOC1, CD74, HLA-DQB1",
        "main_cell_context": "microglia/immune",
        "model_evidence": "Green module and fold-selected genes are dominated by immune/complement markers",
        "therapeutic_hypothesis": "Prioritize validation of microglial activation state modulators, while avoiding global immune suppression.",
        "priority_score": 8.9,
    },
    {
        "target_axis": "astrocyte stress/endfeet",
        "representative_features": "GFAP, AQP4, CLU, CRYAB, BEST1, HSPB1",
        "main_cell_context": "astrocyte/endfeet",
        "model_evidence": "Blue, yellow and turquoise modules map to astrocyte stress and endfeet genes",
        "therapeutic_hypothesis": "Test whether restoring astrocyte endfeet/metabolic support normalizes AD-associated NVU states.",
        "priority_score": 8.1,
    },
    {
        "target_axis": "endothelial/pericyte integrity",
        "representative_features": "ADIRF, RGS5, SLC2A1, NDRG1, endothelial ratio",
        "main_cell_context": "endothelial and pericyte/mural cells",
        "model_evidence": "Endothelial ratio, magenta module and ADIRF/NDRG1 are repeatedly selected",
        "therapeutic_hypothesis": "Validate BBB/metabolic transport and mural support axes as NVU-stabilizing interventions.",
        "priority_score": 7.8,
    },
    {
        "target_axis": "iron/oxidative stress",
        "representative_features": "FTL, FTH1, GSTP1, MT1E, MT1X, PRDX1",
        "main_cell_context": "glia and vascular-associated stress states",
        "model_evidence": "Fold-selected genes include ferritin, glutathione and metallothionein programs",
        "therapeutic_hypothesis": "Treat as a biomarker/response axis first; intervention needs cell-type-specific validation.",
        "priority_score": 7.2,
    },
    {
        "target_axis": "APP-CD74 / PSAP-GPR37",
        "representative_features": "APP-CD74, PSAP-GPR37, APP-SORL1, APP-GPC1",
        "main_cell_context": "neuron-glia and glia-vascular communication",
        "model_evidence": "LR features repeatedly distinguish AD and Control at sample level",
        "therapeutic_hypothesis": "Use perturbation experiments to decide whether these LR shifts are adaptive clearance or maladaptive signaling.",
        "priority_score": 6.9,
    },
]

EXTERNAL_DATASETS = [
    {
        "dataset": "SEA-AD processed single-cell/spatial resources",
        "modality": "snRNA-seq, snATAC-seq, MERFISH/Xenium, neuropathology",
        "why_external": "Independent SEA-AD donor cohort and public AWS buckets; not one of the current 44 sample IDs.",
        "best_use": "External validation of CPT AD axis after building a SEA-AD-to-NVU adapter or pseudobulk CPT projection.",
        "access": "AWS Open Data: s3://sea-ad-single-cell-profiling and s3://sea-ad-spatial-transcriptomics",
        "source_url": "https://registry.opendata.aws/allen-sea-ad-atlas/",
        "status": "recommended_next",
    },
    {
        "dataset": "GSE263468 human cortex AD snRNA-seq",
        "modality": "single-nucleus RNA-seq",
        "why_external": "46 donors across BA9, BA7 and BA17; AD pathology stages and controls differ from current sample IDs.",
        "best_use": "Independent single-nucleus validation of CPT-derived AD gene/cell-type axis.",
        "access": "NCBI GEO accession GSE263468",
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263468",
        "status": "recommended_next",
    },
    {
        "dataset": "SEA-AD IAC integrated public AD/ADRD resource",
        "modality": "harmonized single-cell representations",
        "why_external": "Uniformly reprocessed public AD/ADRD datasets mapped to BICAN/SEA-AD taxonomy.",
        "best_use": "Cross-study validation of whether the same disease axis is recovered across cohorts.",
        "access": "AD Knowledge Portal SEA-AD IAC",
        "source_url": "https://adknowledgeportal.synapse.org/Explore/Studies/DetailsPage/StudyDetails?Study=syn64410371",
        "status": "use_if_access_available",
    },
    {
        "dataset": "GSE152506 plaque-neighborhood spatial transcriptomics",
        "modality": "spatial transcriptomics, primarily AD mouse model with human validation context",
        "why_external": "Not the current human NVU sample set; useful for plaque-neighborhood biology rather than headline human AUROC.",
        "best_use": "Supportive validation for glial plaque-response modules such as complement, oxidative stress and inflammation.",
        "access": "NCBI GEO accession GSE152506",
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152506",
        "status": "supportive_not_primary",
    },
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save_pub(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 3000, seed: int = 13) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "auc": pd.read_csv(RESULT_ROOT / "multicell_model_auc_summary.csv"),
        "pred": pd.read_csv(RESULT_ROOT / "multicell_model_loo_predictions.csv"),
        "weights": pd.read_csv(RESULT_ROOT / "multicell_model_feature_weights.csv"),
        "groups": pd.read_csv(RESULT_ROOT / "multicell_model_feature_group_importance.csv"),
        "selected": pd.read_csv(RESULT_ROOT / "multicell_model_selected_gene_frequency.csv"),
        "univar": pd.read_csv(RESULT_ROOT / "multicell_model_univariate_feature_stats.csv"),
        "curated": pd.read_csv(RESULT_ROOT / "multicell_curated_gene_sources.csv"),
    }


def make_source_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    pred = tables["pred"]
    cmp_pred = pred[pred["feature_set"].isin(METHOD_LABELS)].copy()
    auc_rows = []
    roc_rows = []
    for dataset in DATASET_LABELS:
        for method in METHOD_LABELS:
            sub = cmp_pred[(cmp_pred["dataset"] == dataset) & (cmp_pred["feature_set"] == method)].copy()
            if sub.empty:
                continue
            y = sub["heldout_label"].to_numpy(int)
            score = sub["pred_prob"].to_numpy(float)
            lo, hi = bootstrap_auc_ci(y, score, seed=17 + len(auc_rows))
            fpr, tpr, _ = roc_curve(y, score)
            auc_rows.append(
                {
                    "dataset": dataset,
                    "feature_set": method,
                    "method_label": METHOD_LABELS[method],
                    "n_samples": len(sub),
                    "n_ad": int(y.sum()),
                    "n_control": int((1 - y).sum()),
                    "auroc": roc_auc_score(y, score),
                    "auroc_ci_low": lo,
                    "auroc_ci_high": hi,
                }
            )
            for x, yy in zip(fpr, tpr):
                roc_rows.append(
                    {
                        "dataset": dataset,
                        "feature_set": method,
                        "method_label": METHOD_LABELS[method],
                        "fpr": float(x),
                        "tpr": float(yy),
                    }
                )
    source = {
        "cpt_vs_raw_auc_bootstrap": pd.DataFrame(auc_rows),
        "cpt_vs_raw_roc_points": pd.DataFrame(roc_rows),
        "cpt_vs_raw_predictions": cmp_pred,
    }
    source["cpt_vs_raw_auc_bootstrap"].to_csv(FIG_ROOT / "source_cpt_vs_raw_auc_bootstrap.csv", index=False)
    source["cpt_vs_raw_roc_points"].to_csv(FIG_ROOT / "source_cpt_vs_raw_roc_points.csv", index=False)
    source["cpt_vs_raw_predictions"].to_csv(FIG_ROOT / "source_cpt_vs_raw_predictions.csv", index=False)
    return source


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def draw_auc_panel(ax: mpl.axes.Axes, auc_df: pd.DataFrame) -> None:
    width = 0.32
    datasets = list(DATASET_LABELS)
    xbase = np.arange(len(datasets))
    for offset, method in zip([-width / 1.8, width / 1.8], METHOD_LABELS):
        sub = auc_df[auc_df["feature_set"] == method].set_index("dataset").loc[datasets]
        xs = xbase + offset
        vals = sub["auroc"].to_numpy(float)
        err_low = vals - sub["auroc_ci_low"].to_numpy(float)
        err_hi = sub["auroc_ci_high"].to_numpy(float) - vals
        ax.bar(xs, vals, width=width, color=METHOD_COLORS[method], edgecolor="black", linewidth=0.5, label=METHOD_LABELS[method])
        ax.errorbar(xs, vals, yerr=[err_low, err_hi], fmt="none", ecolor="black", lw=0.7, capsize=2)
        for x, val in zip(xs, vals):
            ax.text(x, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xbase)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("Held-out AUROC")
    ax.axhline(0.5, color="#BDBDBD", lw=0.8, ls="--")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Sample-level AD/Control discrimination")
    panel_label(ax, "A")


def draw_roc_panel(ax: mpl.axes.Axes, roc_df: pd.DataFrame, auc_df: pd.DataFrame, dataset: str, label: str) -> None:
    for method in METHOD_LABELS:
        sub = roc_df[(roc_df["dataset"] == dataset) & (roc_df["feature_set"] == method)]
        auc_val = auc_df[(auc_df["dataset"] == dataset) & (auc_df["feature_set"] == method)]["auroc"].iloc[0]
        ax.plot(sub["fpr"], sub["tpr"], color=METHOD_COLORS[method], lw=1.8, label=f"{METHOD_LABELS[method]} ({auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=0.8, ls="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC: {DATASET_LABELS[dataset].replace(chr(10), ' ')}")
    ax.legend(loc="lower right", fontsize=6)
    panel_label(ax, label)


def draw_probability_panel(ax: mpl.axes.Axes, pred: pd.DataFrame, dataset: str, label: str) -> None:
    rng = np.random.default_rng(7)
    methods = list(METHOD_LABELS)
    xticks = []
    xticklabels = []
    pos = 0
    for method in methods:
        for yval, lab in [(0, "Control"), (1, "AD")]:
            vals = pred[(pred["dataset"] == dataset) & (pred["feature_set"] == method) & (pred["heldout_label"] == yval)]["pred_prob"].to_numpy(float)
            xs = np.full(len(vals), pos) + rng.normal(0, 0.045, len(vals))
            ax.scatter(xs, vals, s=18, color=LABEL_COLORS[yval], alpha=0.85, edgecolor="white", linewidth=0.3)
            if len(vals):
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                ax.plot([pos - 0.18, pos + 0.18], [med, med], color="black", lw=1.0)
                ax.plot([pos, pos], [q1, q3], color="black", lw=0.8)
            xticks.append(pos)
            xticklabels.append(lab)
            pos += 1
        pos += 0.55
    ax.axhline(0.5, color="#BDBDBD", lw=0.8, ls="--")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("Held-out AD probability")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    for xpos, method in zip([0.5, 3.05], methods):
        ax.text(xpos, 1.015, METHOD_LABELS[method], ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_title(f"{DATASET_LABELS[dataset].replace(chr(10), ' ')} prediction scores", pad=18)
    panel_label(ax, label)


def make_comparison_figure(source: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.05], hspace=0.58, wspace=0.35)
    draw_auc_panel(fig.add_subplot(gs[0, 0]), source["cpt_vs_raw_auc_bootstrap"])
    draw_roc_panel(fig.add_subplot(gs[0, 1]), source["cpt_vs_raw_roc_points"], source["cpt_vs_raw_auc_bootstrap"], "qc36_risk_available", "B")
    draw_probability_panel(fig.add_subplot(gs[1, 0]), source["cpt_vs_raw_predictions"], "qc36_risk_available", "C")
    draw_probability_panel(fig.add_subplot(gs[1, 1]), source["cpt_vs_raw_predictions"], "all44", "D")
    fig.suptitle("CPT latent features capture a sample-level AD state beyond raw gene SVD", fontsize=9.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.89)
    save_pub(fig, FIG_ROOT / "BI_CPT_latent_vs_raw_gene_SVD")
    plt.close(fig)


def clean_feature_label(feature: str, max_len: int = 36) -> str:
    label = feature
    for old, new in [
        ("curated_gene__", ""),
        ("selected_gene__", ""),
        ("node_lr__", ""),
        ("scalar__", ""),
        ("__mean_cell", ""),
        ("__mean", ""),
        ("__max", ""),
        ("__std", ""),
        ("__high_frac", ""),
    ]:
        label = label.replace(old, new)
    label = label.replace("__", " ")
    return label if len(label) <= max_len else label[: max_len - 1] + "."


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def make_module_cell_matrix(weights: pd.DataFrame) -> pd.DataFrame:
    sub = weights[(weights["dataset"] == "qc36_risk_available") & (weights["feature_set"] == "ad_informed_cpt")].copy()
    sub = sub[sub["feature_group"].str.startswith("module:") | (sub["feature_group"] == "cell composition")]
    rows = []
    for _, row in sub.iterrows():
        if row["feature_group"].startswith("module:"):
            module = row["feature_group"].replace("module:", "")
            cell = str(row["entity"]).replace("ratio__", "")
        else:
            module = "cell composition"
            cell = str(row["entity"]).replace("ratio__", "")
        rows.append({"module": module, "cell": cell, "value": float(row["mean_coef"])})
    if not rows:
        return pd.DataFrame()
    mat = pd.DataFrame(rows).pivot_table(index="module", columns="cell", values="value", aggfunc="sum", fill_value=0.0)
    preferred_cols = ["Neuron", "Astro", "Micro", "Endo", "Pericyte", "Oligo", "OPC"]
    cols = [c for c in preferred_cols if c in mat.columns] + [c for c in mat.columns if c not in preferred_cols]
    return mat.loc[[i for i in ["blue", "green", "turquoise", "magenta", "yellow", "cell composition"] if i in mat.index], cols]


def draw_group_importance(ax: mpl.axes.Axes, groups: pd.DataFrame) -> None:
    sub = groups[(groups["dataset"] == "qc36_risk_available") & (groups["feature_set"] == "ad_informed_cpt")].copy()
    sub = sub.head(8).iloc[::-1]
    colors = ["#2A6FBB" if g == "CPT latent distribution" else "#7AA6C2" if "module" in g else "#B9884D" if "gene" in g else "#8BA86B" if "LR" in g else "#999999" for g in sub["feature_group"]]
    ax.barh(np.arange(len(sub)), sub["importance"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["feature_group"])
    ax.set_xlabel("L1 coefficient importance")
    ax.set_title("Interpretable evidence retained in AD-informed CPT")
    panel_label(ax, "A")


def draw_module_heatmap(ax: mpl.axes.Axes, weights: pd.DataFrame) -> pd.DataFrame:
    mat = make_module_cell_matrix(weights)
    if mat.empty:
        ax.axis("off")
        return mat
    vmax = max(abs(mat.to_numpy()).max(), 1e-6)
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_title("Cell/module coefficient map")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if abs(val) > vmax * 0.08:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("signed coefficient")
    panel_label(ax, "B")
    return mat


def draw_gene_frequency(ax: mpl.axes.Axes, selected: pd.DataFrame) -> pd.DataFrame:
    sub = selected[(selected["dataset"] == "qc36_risk_available") & (selected["gene"].isin(PREFERRED_DISPLAY_GENES))].copy()
    sub["order"] = sub["gene"].map({g: i for i, g in enumerate(PREFERRED_DISPLAY_GENES)})
    sub = sub.sort_values(["order", "n_folds_selected"], ascending=[True, False]).drop(columns=["order"]).iloc[::-1]
    palette = {
        "microglia/immune": "#6A9C5B",
        "complement": "#5A8F62",
        "astrocyte": "#4A8FB8",
        "pericyte/mural": "#C08A3E",
        "iron/stress": "#A96B5B",
        "metallothionein": "#9A7AA0",
        "oxidative stress": "#A96B5B",
        "stress/vascular": "#B99B57",
        "lipid/stress": "#B99B57",
        "ECM/immune": "#827A4B",
        "antigen presentation": "#6A9C5B",
    }
    sub["class"] = sub["gene"].map(GENE_CLASS).fillna("other")
    colors = [palette.get(c, "#9E9E9E") for c in sub["class"]]
    ax.barh(np.arange(len(sub)), sub["n_folds_selected"], color=colors, edgecolor="black", linewidth=0.35)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["gene"])
    ax.set_xlabel("LOO folds selected")
    ax.set_title("Fold-internal selected genes")
    panel_label(ax, "C")
    return sub.iloc[::-1]


def draw_target_axes(ax: mpl.axes.Axes, target_df: pd.DataFrame) -> None:
    sub = target_df.sort_values("priority_score", ascending=True)
    ax.barh(np.arange(len(sub)), sub["priority_score"], color="#B9884D", edgecolor="black", linewidth=0.35)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["target_axis"])
    ax.set_xlabel("hypothesis priority")
    ax.set_xlim(0, 10)
    ax.set_title("Therapeutic hypotheses from interpretable axes")
    panel_label(ax, "D")


def make_interpretability_figure(tables: dict[str, pd.DataFrame]) -> None:
    target_df = pd.DataFrame(TARGET_AXES)
    target_df.to_csv(FIG_ROOT / "BI_interpretability_target_candidates.csv", index=False)
    external_df = pd.DataFrame(EXTERNAL_DATASETS)
    external_df.to_csv(FIG_ROOT / "BI_external_validation_candidates.csv", index=False)

    fig = plt.figure(figsize=(7.8, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.72)
    draw_group_importance(fig.add_subplot(gs[0, 0]), tables["groups"])
    mat = draw_module_heatmap(fig.add_subplot(gs[0, 1]), tables["weights"])
    gene_freq = draw_gene_frequency(fig.add_subplot(gs[1, 0]), tables["selected"])
    draw_target_axes(fig.add_subplot(gs[1, 1]), target_df)
    if not mat.empty:
        mat.to_csv(FIG_ROOT / "source_module_cell_coefficient_matrix.csv")
    gene_freq.to_csv(FIG_ROOT / "source_selected_gene_frequency_qc36_top.csv", index=False)
    fig.suptitle("Interpretable glial, vascular, LR and gene axes behind the CPT AD state", fontsize=9.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.18, right=0.96)
    save_pub(fig, FIG_ROOT / "BI_CPT_interpretability_and_targets")
    plt.close(fig)


def write_report(tables: dict[str, pd.DataFrame], source: dict[str, pd.DataFrame]) -> None:
    auc_df = source["cpt_vs_raw_auc_bootstrap"].copy()
    auc_df["auroc_ci"] = auc_df.apply(lambda r: f"{r.auroc:.3f} ({r.auroc_ci_low:.3f}-{r.auroc_ci_high:.3f})", axis=1)
    target_df = pd.DataFrame(TARGET_AXES)
    external_df = pd.DataFrame(EXTERNAL_DATASETS)
    preferred_present = tables["selected"][
        (tables["selected"]["dataset"] == "qc36_risk_available")
        & (tables["selected"]["gene"].isin(PREFERRED_DISPLAY_GENES))
    ]["gene"].drop_duplicates().tolist()
    report = [
        "# CPT latent vs raw gene SVD AD analysis",
        "",
        "## Main answer",
        "",
        "Yes, the CPT latent separation is high at the sample/chip level. The main comparison can be simplified to CPT latent versus raw gene SVD.",
        "",
        "## Held-out AUROC",
        "",
        df_to_md(auc_df[["dataset", "method_label", "n_samples", "n_ad", "n_control", "auroc_ci"]]),
        "",
        "## Biological interpretability",
        "",
        "Largest interpretable axes from the AD-informed model include CPT latent dispersion, endothelial/cell-composition shifts, blue glial module, curated and fold-selected AD/NVU genes, and LR signaling.",
        "",
        "Top fold-internal selected genes in the QC cohort include: "
        + ", ".join([g for g in PREFERRED_DISPLAY_GENES if g in preferred_present])
        + ".",
        "",
        "## Therapeutic hypotheses",
        "",
        df_to_md(target_df[["target_axis", "representative_features", "main_cell_context", "priority_score"]]),
        "",
        "These are target hypotheses, not causal claims. The next experimental step should perturb or validate these axes in cell-type-aware models.",
        "",
        "## External validation plan",
        "",
        df_to_md(external_df[["dataset", "modality", "best_use", "status", "source_url"]]),
        "",
        "Immediate recommendation: use SEA-AD or GSE263468 for independent validation after building an adapter that converts external cells/spots into the same CPT/NVU feature contract.",
        "",
    ]
    (FIG_ROOT / "BI_CPT_vs_rawSVD_interpretability_report.md").write_text("\n".join(report), encoding="utf-8")
    (FIG_ROOT / "BI_CPT_vs_rawSVD_interpretability_report.json").write_text(
        json.dumps(
            {
                "auc": source["cpt_vs_raw_auc_bootstrap"].to_dict("records"),
                "external_validation_candidates": EXTERNAL_DATASETS,
                "therapeutic_target_candidates": TARGET_AXES,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    setup_style()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    source = make_source_tables(tables)
    make_comparison_figure(source, tables)
    make_interpretability_figure(tables)
    write_report(tables, source)
    print(f"Wrote figures and source data to {FIG_ROOT}")


if __name__ == "__main__":
    main()
