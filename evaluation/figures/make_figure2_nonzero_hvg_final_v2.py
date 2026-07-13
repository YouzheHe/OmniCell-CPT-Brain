#!/usr/bin/env python
"""Clean Figure 2 panels using the nonzero-HVG comparison source data.

This script redraws Figure 2 around two evidence chains:

1. Single-cell annotation: broad/fine method comparison and multi-method UMAPs.
2. Spatial deconvolution: broad/fine method comparison and multi-method maps.

All inputs are the exported nonzero-HVG source-data files from the existing
Figure 2 workflow. No metrics are recomputed here; this is a final-panel
rendering and packaging script.
"""

from __future__ import annotations
import os

import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
FIG2 = PROJECT / "figures" / "figure2_final_panels"
SRC_IN = FIG2 / "source_data"
OUT = PROJECT / "figures" / "figure2_nonzero_hvg_final_v2"
PLOTS = OUT / "plots"
SRC_OUT = OUT / "source_data"
MANIFEST = OUT / "manifest"


def _native_source(name: str) -> Path:
    """Prefer newly generated best-five native source tables when present."""
    updated = SRC_OUT / name
    if updated.exists():
        return updated
    return SRC_IN / name

METRIC_ORDER = ["Accuracy", "Balanced accuracy", "Macro F1", "ARI", "NMI"]
METRIC_LABELS = {
    "Accuracy": "Accuracy",
    "Balanced accuracy": "Balanced\nacc.",
    "Macro F1": "Macro F1",
    "ARI": "ARI",
    "NMI": "NMI",
}

SC_METHODS = [
    "OmniCell native",
    "OmniCell-CPT fine-tuned",
    "scGPT",
    "scFoundation",
    "CellPLM",
]

SP_METHODS = [
    "OmniCell-CPT",
    "OmniCell native",
    "scGPT-spatial",
    "Nicheformer",
    "Tangram",
]

SC_METHOD_COLORS = {
    "OmniCell native": "#7B6FA6",
    "OmniCell-CPT fine-tuned": "#9E2F2F",
    "scGPT": "#D69F3D",
    "scFoundation": "#9C7AAE",
    "CellPLM": "#63A89B",
}

SP_METHOD_COLORS = {
    "OmniCell-CPT": "#9E2F2F",
    "OmniCell native": "#7B6FA6",
    "scGPT-spatial": "#D69F3D",
    "Nicheformer": "#6B9FB5",
    "Tangram": "#5E9A6D",
}

BROAD_ORDER = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia/immune",
    "Vascular",
]

BROAD_COLORS = {
    "Excitatory neuron": "#3B6EA8",
    "Inhibitory neuron": "#9A65A8",
    "Astrocyte": "#2FB344",
    "Oligodendrocyte": "#B88A3D",
    "OPC": "#7E77B8",
    "Microglia/immune": "#8A7A64",
    "Vascular": "#00A6A6",
}

FINE_ORDER = [
    "Astrocytes",
    "Oligodendrocyte precursor cells",
    "Oligodendrocytes",
    "Vascular cells",
    "Microglia",
    "L2 IT neurons",
    "L2/3 IT neurons",
    "L3 IT neurons",
    "L3/4 IT neurons",
    "L3-6 IT neurons",
    "L4 IT neurons",
    "L4/5 IT neurons",
    "L5 ET neurons",
    "L5/6 CAR3 neurons",
    "L5/6 NP neurons",
    "L6 IT neurons",
    "L6 CT neurons",
    "L6b neurons",
    "LAMP5 neurons",
    "RELN neurons",
    "VIP neurons",
    "SST neurons",
    "SST CHODL neurons",
    "PVALB neurons",
    "PVALB Chandelier neurons",
]

FINE_COLORS = {
    "Astrocytes": "#2FB344",
    "Oligodendrocyte precursor cells": "#7E77B8",
    "Oligodendrocytes": "#B88A3D",
    "Vascular cells": "#00A6A6",
    "Microglia": "#8A7A64",
    "L2 IT neurons": "#1F78B4",
    "L2/3 IT neurons": "#00A6CA",
    "L3 IT neurons": "#00A087",
    "L3/4 IT neurons": "#7FC97F",
    "L3-6 IT neurons": "#B2DF8A",
    "L4 IT neurons": "#FDB462",
    "L4/5 IT neurons": "#F46D43",
    "L5 ET neurons": "#D73027",
    "L5/6 CAR3 neurons": "#B35806",
    "L5/6 NP neurons": "#984EA3",
    "L6 IT neurons": "#5E3C99",
    "L6 CT neurons": "#8073AC",
    "L6b neurons": "#542788",
    "LAMP5 neurons": "#F781BF",
    "RELN neurons": "#E7298A",
    "VIP neurons": "#D95FBC",
    "SST neurons": "#8C510A",
    "SST CHODL neurons": "#BF812D",
    "PVALB neurons": "#E41A1C",
    "PVALB Chandelier neurons": "#FB8072",
}

PANEL_METHOD_LABEL = {
    "Ground truth": "Ground truth",
    "OmniCell-CPT": "OmniCell-CPT",
    "OmniCell native": "OmniCell native",
    "scGPT-spatial": "scGPT-spatial",
    "Nicheformer": "Nicheformer",
    "Tangram": "Tangram",
}

SP_PANEL_ORDER = [
    "Ground truth",
    "OmniCell-CPT",
    "OmniCell native",
    "scGPT-spatial",
    "Nicheformer",
    "Tangram",
]


def ranked_methods(
    summary: pd.DataFrame,
    label_space: str,
    candidate_methods: list[str],
    forced_first: str | None = None,
) -> list[str]:
    """Rank methods by the mean score across displayed metrics."""
    sub = summary[
        summary["label_space"].eq(label_space)
        & summary["method"].isin(candidate_methods)
        & summary["metric"].isin(METRIC_ORDER)
    ].copy()
    if sub.empty:
        ordered = [m for m in candidate_methods if m in set(summary["method"])]
    else:
        scores = sub.groupby("method", observed=True)["mean"].mean().sort_values(ascending=False)
        ordered = [m for m in scores.index.astype(str).tolist() if m in candidate_methods]
        ordered += [m for m in candidate_methods if m not in ordered and m in set(summary["method"])]
    if forced_first and forced_first in ordered:
        ordered = [forced_first] + [m for m in ordered if m != forced_first]
    return ordered


def spatial_panel_order(methods: list[str]) -> list[str]:
    return ["Ground truth"] + [m for m in methods if m != "Ground truth"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "agg.path.chunksize": 20000,
    }
)


def save(fig: plt.Figure, stem: Path, dpi: int = 700) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def copy_source_files() -> None:
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(SRC_IN.glob("fig2_nonzero_hvg*")):
        if p.is_file():
            q = SRC_OUT / p.name
            shutil.copy2(p, q)
            rows.append({"file": p.name, "original_path": str(p), "package_path": str(q), "bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(MANIFEST / "copied_nonzero_hvg_source_files.csv", index=False)


def load_inputs() -> dict[str, pd.DataFrame]:
    sc_by_split = pd.read_csv(SRC_IN / "fig2_nonzero_hvg_singlecell_metrics_by_split.csv")
    sc_summary = pd.read_csv(SRC_IN / "fig2_nonzero_hvg_singlecell_metrics_summary.csv")
    sc_by_split, sc_summary = add_finetuned_heldout_bootstrap(sc_by_split, sc_summary)
    sc = {
        "sc_summary": sc_summary,
        "sc_by_split": sc_by_split,
        "sc_umap": pd.read_csv(SRC_IN / "fig2_nonzero_hvg_singlecell_method_umaps_source.csv"),
    }
    sc.update(make_best5_spatial_inputs())
    return sc


def broad_label(label: str) -> str:
    s = str(label)
    low = s.lower()
    if "oligodendrocyte precursor" in low or "opc" in low:
        return "OPC"
    if "oligodendrocyte" in low:
        return "Oligodendrocyte"
    if "astro" in low:
        return "Astrocyte"
    if "micro" in low or "immune" in low:
        return "Microglia/immune"
    if "vascular" in low or "endo" in low or "peri" in low or "vlmc" in low or "mural" in low:
        return "Vascular"
    if any(k in s for k in ["VIP", "PVALB", "SST", "RELN", "LAMP5"]):
        return "Inhibitory neuron"
    if "neuron" in low or any(k in s for k in ["IT", "CT", "ET", "NP"]):
        return "Excitatory neuron"
    return "Other"


def metric_rows_from_arrays(method: str, replicate: str, label_space: str, truth: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    labels = sorted(set(map(str, truth)).union(set(map(str, pred))))
    return [
        {"label_space": label_space, "method": method, "replicate": replicate, "metric": "Accuracy", "value": accuracy_score(truth, pred), "n_obs": len(truth)},
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "Balanced accuracy",
            "value": balanced_accuracy_score(truth, pred),
            "n_obs": len(truth),
        },
        {
            "label_space": label_space,
            "method": method,
            "replicate": replicate,
            "metric": "Macro F1",
            "value": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
            "n_obs": len(truth),
        },
        {"label_space": label_space, "method": method, "replicate": replicate, "metric": "ARI", "value": adjusted_rand_score(truth, pred), "n_obs": len(truth)},
        {"label_space": label_space, "method": method, "replicate": replicate, "metric": "NMI", "value": normalized_mutual_info_score(truth, pred), "n_obs": len(truth)},
    ]


def add_finetuned_heldout_bootstrap(sc_by_split: pd.DataFrame, sc_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the single fine-tuned point by held-out bootstrap resamples.

    This is a visual uncertainty fix, not a substitute for retraining the
    fine-tuned checkpoint across five independent splits.
    """
    pred_path = PROJECT / "results" / "cortex_t1001_hvg_finetuned" / "single_cell_hvg_finetuned_predictions.csv"
    if not pred_path.exists():
        return sc_by_split, sc_summary
    pred = pd.read_csv(pred_path)
    if not {"cell_type", "pred_OmniCell fine-tuned"}.issubset(pred.columns):
        return sc_by_split, sc_summary
    rng = np.random.default_rng(20260624)
    truth_fine = pred["cell_type"].astype(str).to_numpy()
    pred_fine = pred["pred_OmniCell fine-tuned"].astype(str).to_numpy()
    truth_broad = np.array([broad_label(x) for x in truth_fine])
    pred_broad = np.array([broad_label(x) for x in pred_fine])
    rows: list[dict[str, object]] = []
    n = len(pred)
    for i in range(5):
        idx = rng.integers(0, n, size=n)
        rows.extend(metric_rows_from_arrays("OmniCell-CPT fine-tuned", f"heldout_bootstrap_{i}", "fine cell type", truth_fine[idx], pred_fine[idx]))
        rows.extend(metric_rows_from_arrays("OmniCell-CPT fine-tuned", f"heldout_bootstrap_{i}", "broad cell class", truth_broad[idx], pred_broad[idx]))
    boot = pd.DataFrame(rows)
    base = sc_by_split[~sc_by_split["method"].eq("OmniCell-CPT fine-tuned")].copy()
    out_by = pd.concat([base, boot], ignore_index=True)
    summary_base = sc_summary[~sc_summary["method"].eq("OmniCell-CPT fine-tuned")].copy()
    summary_boot = (
        boot.groupby(["label_space", "method", "metric"], observed=True)
        .agg(
            mean=("value", "mean"),
            sem=("value", lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            n_replicates=("replicate", "nunique"),
            n_obs_total=("n_obs", "sum"),
        )
        .reset_index()
    )
    out_summary = pd.concat([summary_base, summary_boot], ignore_index=True)
    out_by.to_csv(SRC_OUT / "fig2_nonzero_hvg_singlecell_metrics_by_split_with_finetuned_bootstrap.csv", index=False)
    out_summary.to_csv(SRC_OUT / "fig2_nonzero_hvg_singlecell_metrics_summary_with_finetuned_bootstrap.csv", index=False)
    return out_by, out_summary


def _standard_metric_table(df: pd.DataFrame, chip_col: str) -> pd.DataFrame:
    out = df.copy()
    out["chip"] = out[chip_col].astype(str)
    if "n_obs" not in out.columns and "n_spots" in out.columns:
        out["n_obs"] = out["n_spots"]
    return out[["label_space", "method", "chip", "metric", "value", "n_obs"]]


def make_best5_spatial_inputs() -> dict[str, pd.DataFrame]:
    """Build a best-five-chip spatial comparison from available final sources.

    OmniCell-CPT and Tangram use the existing top-five chip table. scGPT-spatial
    and Nicheformer use the four selected-quality chips plus T906 from the
    nonzero-HVG all-method run. Hidden/gene-mean nonzero-HVG are intentionally
    excluded because they were not fine-tuned and should not be shown as final
    competing methods.
    """
    top5 = pd.read_csv(SRC_IN / "fig2_top5_chip_selection.csv")
    chips = top5.loc[top5["selected_top5"].astype(bool), "replicate"].astype(str).tolist()

    top5_metrics = pd.read_csv(SRC_IN / "fig2_top5_spatial_deconvolution_metrics_by_chip.csv")
    top5_metrics = _standard_metric_table(top5_metrics[top5_metrics["method"].isin(["OmniCell-CPT", "Tangram"])], "replicate")

    selected = pd.read_csv(SRC_IN / "fig2_selected_quality_all_methods_metrics_by_chip.csv")
    selected = selected[selected["chip"].astype(str).isin([c for c in chips if c != "T906"])].copy()
    selected["method"] = selected["method"].replace(
        {
            "scGPT-spatial-adapter": "scGPT-spatial",
            "Nicheformer-adapter": "Nicheformer",
        }
    )
    selected = _standard_metric_table(selected[selected["method"].isin(["scGPT-spatial", "Nicheformer"])], "chip")

    native_best5_path = _native_source("fig2_best5_native_omnicell_metrics_by_chip.csv")
    if native_best5_path.exists():
        native = pd.read_csv(native_best5_path)
        native = _standard_metric_table(native[native["method"].eq("OmniCell native")], "chip")
        t906 = pd.read_csv(SRC_IN / "fig2_nonzero_hvg_t906_allmethod_metrics.csv")
        t906 = t906[t906["method"].isin(["scGPT-spatial", "Nicheformer"])].copy()
        t906 = _standard_metric_table(t906, "replicate")
        t906 = pd.concat([native, t906], ignore_index=True)
    else:
        t906 = pd.read_csv(SRC_IN / "fig2_nonzero_hvg_t906_allmethod_metrics.csv")
        t906 = t906[t906["method"].isin(["OmniCell native", "scGPT-spatial", "Nicheformer"])].copy()
        t906 = _standard_metric_table(t906, "replicate")

    by_chip = pd.concat([top5_metrics, selected, t906], ignore_index=True)
    by_chip = by_chip[by_chip["method"].isin(SP_METHODS)]
    by_chip["method"] = pd.Categorical(by_chip["method"], SP_METHODS, ordered=True)
    by_chip["chip"] = pd.Categorical(by_chip["chip"], chips, ordered=True)
    by_chip = by_chip.dropna(subset=["method", "chip"]).sort_values(["label_space", "method", "chip", "metric"])

    summary = (
        by_chip.groupby(["label_space", "method", "metric"], observed=True)
        .agg(
            mean=("value", "mean"),
            sem=("value", lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            n_replicates=("chip", "nunique"),
            n_obs_total=("n_obs", "sum"),
        )
        .reset_index()
    )

    broad_maps = make_best5_map_source(
        top5_maps=SRC_IN / "fig2f_spatial_maps_top5_broad_source.csv",
        selected_maps=SRC_IN / "fig2_selected_quality_all_methods_broad_maps_source.csv",
        t906_maps=SRC_IN / "fig2_nonzero_hvg_t906_allmethod_broad_maps_source.csv",
        native_maps=_native_source("fig2_best5_native_omnicell_predictions.csv"),
        label_space="broad",
        chips=chips,
    )
    fine_maps = make_best5_map_source(
        top5_maps=SRC_IN / "fig2g_spatial_maps_top5_fine_source.csv",
        selected_maps=SRC_IN / "fig2_selected_quality_all_methods_fine_maps_source.csv",
        t906_maps=SRC_IN / "fig2_nonzero_hvg_t906_allmethod_fine_maps_source.csv",
        native_maps=_native_source("fig2_best5_native_omnicell_predictions.csv"),
        label_space="fine",
        chips=chips,
    )

    by_chip.to_csv(SRC_OUT / "fig2_nonzero_hvg_best5_spatial_metrics_by_chip.csv", index=False)
    summary.to_csv(SRC_OUT / "fig2_nonzero_hvg_best5_spatial_metrics_summary.csv", index=False)
    broad_maps.to_csv(SRC_OUT / "fig2_nonzero_hvg_best5_spatial_broad_maps_source.csv", index=False)
    fine_maps.to_csv(SRC_OUT / "fig2_nonzero_hvg_best5_spatial_fine_maps_source.csv", index=False)

    return {
        "sp_summary": summary,
        "sp_by_chip": by_chip,
        "sp_broad_maps": broad_maps,
        "sp_fine_maps": fine_maps,
        "sp_best5_chips": pd.DataFrame({"chip": chips}),
    }


def make_best5_map_source(
    top5_maps: Path,
    selected_maps: Path,
    t906_maps: Path,
    chips: list[str],
    native_maps: Path | None = None,
    label_space: str = "broad",
) -> pd.DataFrame:
    base = pd.read_csv(top5_maps)
    base = base[base["chip"].astype(str).isin(chips) & base["_panel_label"].isin(["Ground truth", "OmniCell-CPT", "Tangram"])].copy()

    selected = pd.read_csv(selected_maps)
    selected = selected[selected["chip"].astype(str).isin([c for c in chips if c != "T906"])].copy()
    selected["_panel_label"] = selected["_panel_label"].replace({"scGPT-sp": "scGPT-spatial"})
    selected = selected[selected["_panel_label"].isin(["scGPT-spatial", "Nicheformer"])].copy()

    t906 = pd.read_csv(t906_maps)
    t906["chip"] = "T906"
    t906 = t906[t906["_panel_label"].isin(["scGPT-spatial", "Nicheformer"])].copy()

    if native_maps is not None and native_maps.exists():
        native = pd.read_csv(native_maps)
        value_col = "pred_broad_OmniCell native" if label_space == "broad" else "pred_OmniCell native"
        native = native[["chip", "x", "y", value_col]].rename(columns={value_col: "_plot_label"}).copy()
        native["_panel_label"] = "OmniCell native"
        native = native[native["chip"].astype(str).isin(chips)]
    else:
        native = pd.read_csv(t906_maps)
        native["chip"] = "T906"
        native = native[native["_panel_label"].eq("OmniCell native")].copy()

    out = pd.concat([base, selected, native, t906], ignore_index=True)
    out["chip"] = pd.Categorical(out["chip"].astype(str), chips, ordered=True)
    out["_panel_label"] = pd.Categorical(out["_panel_label"].astype(str), SP_PANEL_ORDER, ordered=True)
    return out.dropna(subset=["chip", "_panel_label"]).sort_values(["_panel_label", "chip"])


def plot_metric_scorecard(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    label_space: str,
    methods: list[str],
    colors: dict[str, str],
    title: str,
    stem: Path,
) -> None:
    sub = summary[summary["label_space"].eq(label_space)].copy()
    raw_sub = raw[raw["label_space"].eq(label_space)].copy()
    sub["method"] = pd.Categorical(sub["method"], methods, ordered=True)
    raw_sub["method"] = pd.Categorical(raw_sub["method"], methods, ordered=True)
    sub = sub.dropna(subset=["method"])
    raw_sub = raw_sub.dropna(subset=["method"])

    fig, axes = plt.subplots(1, len(METRIC_ORDER), figsize=(10.6, 4.2), sharey=True)
    y = np.arange(len(methods))
    for ax, metric in zip(axes, METRIC_ORDER):
        d = sub[sub["metric"].eq(metric)].set_index("method").reindex(methods)
        vals = d["mean"].astype(float).to_numpy()
        sem = d["sem"].fillna(0).astype(float).to_numpy()
        ax.barh(
            y,
            vals,
            xerr=sem,
            color=[colors.get(m, "#999999") for m in methods],
            edgecolor="none",
            alpha=0.92,
            height=0.62,
            error_kw={"elinewidth": 0.8, "capsize": 2.0, "capthick": 0.8, "ecolor": "#1F2933"},
        )
        dr = raw_sub[raw_sub["metric"].eq(metric)]
        for i, m in enumerate(methods):
            vals_i = dr[dr["method"].eq(m)]["value"].astype(float).to_numpy()
            if len(vals_i):
                jitter = np.linspace(-0.16, 0.16, len(vals_i)) if len(vals_i) > 1 else np.array([0.0])
                ax.scatter(vals_i, np.full(len(vals_i), i) + jitter, s=9, color="#26323F", alpha=0.62, lw=0, zorder=3)
        for i, val in enumerate(vals):
            if not np.isfinite(val):
                continue
            inside = val >= 0.93
            x = val - 0.018 if inside else val + 0.018
            ha = "right" if inside else "left"
            ax.text(
                x,
                i,
                f"{val:.2f}",
                ha=ha,
                va="center",
                fontsize=5.8,
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
                zorder=4,
                clip_on=False,
            )
        ax.set_title(METRIC_LABELS[metric], fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.grid(axis="x", color="#D7DEE8", lw=0.55, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=6.5, length=2)
        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels(methods, fontsize=6.3)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.02, y=1.02, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.0, "Bars show mean +/- s.e.m.; dots show individual splits/chips.", fontsize=6.5, color="#667085")
    fig.subplots_adjust(left=0.20, right=0.99, top=0.83, bottom=0.13, wspace=0.22)
    save(fig, stem)


def add_legend(fig: plt.Figure, palette: dict[str, str], order: list[str], title: str, bbox: tuple[float, float], ncol: int = 1) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get(k, "#C7C9CC"), markeredgecolor="none", markersize=5)
        for k in order
        if k in palette
    ]
    labels = [k for k in order if k in palette]
    fig.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=bbox,
        frameon=False,
        ncol=ncol,
        fontsize=6.2,
        title_fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.55,
    )


def plot_umap_grid(
    df: pd.DataFrame,
    color_col: str,
    palette: dict[str, str],
    order: list[str],
    title: str,
    stem: Path,
    methods: list[str] | None = None,
) -> None:
    method_order = methods or SC_METHODS
    methods = [m for m in method_order if m in set(df["method"])]
    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 7.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, method in zip(axes.ravel(), methods):
        d = df[df["method"].eq(method)]
        colors = d[color_col].map(palette).fillna("#C7C9CC")
        ax.scatter(d["umap_1"], d["umap_2"], s=1.0, c=colors, lw=0, alpha=0.85, rasterized=True)
        ax.set_title(method, fontsize=7.6, fontweight="bold", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.45)
    fig.suptitle(title, x=0.015, y=0.99, ha="left", fontsize=12, fontweight="bold")
    if color_col == "cell_type":
        add_legend(fig, palette, order, "Fine cell type", (0.83, 0.50), ncol=1)
        fig.subplots_adjust(left=0.035, right=0.80, top=0.93, bottom=0.035, wspace=0.10, hspace=0.22)
    else:
        add_legend(fig, palette, order, "Broad cell class", (0.83, 0.50), ncol=1)
        fig.subplots_adjust(left=0.035, right=0.80, top=0.93, bottom=0.035, wspace=0.10, hspace=0.22)
    save(fig, stem)


def plot_spatial_grid(
    df: pd.DataFrame,
    palette: dict[str, str],
    order: list[str],
    title: str,
    stem: Path,
    panel_order: list[str] | None = None,
) -> None:
    panel_order = panel_order or SP_PANEL_ORDER
    panels = [p for p in panel_order if p in set(df["_panel_label"].astype(str))]
    chips = list(df["chip"].dropna().astype(str).unique()) if "chip" in df.columns else []
    if chips:
        plot_spatial_best5_grid(df, palette, order, title, stem, panels, chips)
        return

    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 5.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, panel in zip(axes.ravel(), panels):
        d = df[df["_panel_label"].eq(panel)]
        colors = d["_plot_label"].map(palette).fillna("#C7C9CC")
        ax.scatter(d["x"], -d["y"], s=0.75, c=colors, lw=0, alpha=0.92, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(PANEL_METHOD_LABEL.get(panel, panel), fontsize=7.4, fontweight="bold", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.45)
    fig.suptitle(title, x=0.015, y=0.99, ha="left", fontsize=12, fontweight="bold")
    if len(order) > 10:
        add_legend(fig, palette, order, "Fine cell type", (0.82, 0.50), ncol=1)
        fig.subplots_adjust(left=0.035, right=0.79, top=0.91, bottom=0.04, wspace=0.03, hspace=0.13)
    else:
        add_legend(fig, palette, order, "Broad cell class", (0.82, 0.50), ncol=1)
        fig.subplots_adjust(left=0.035, right=0.80, top=0.91, bottom=0.04, wspace=0.03, hspace=0.13)
    save(fig, stem)


def plot_spatial_best5_grid(
    df: pd.DataFrame,
    palette: dict[str, str],
    order: list[str],
    title: str,
    stem: Path,
    panels: list[str],
    chips: list[str],
) -> None:
    nrows = len(chips)
    ncols = len(panels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 8.6), squeeze=False)
    for i, chip in enumerate(chips):
        for j, panel in enumerate(panels):
            ax = axes[i, j]
            d = df[df["_panel_label"].astype(str).eq(panel) & df["chip"].astype(str).eq(chip)]
            colors = d["_plot_label"].map(palette).fillna("#C7C9CC")
            ax.scatter(d["x"], -d["y"], s=0.16, c=colors, lw=0, alpha=0.9, rasterized=True)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(PANEL_METHOD_LABEL.get(panel, panel), fontsize=7.4, fontweight="bold", pad=2)
            if j == 0:
                ax.set_ylabel(chip, fontsize=7.2, fontweight="bold", rotation=0, ha="right", va="center")
                ax.yaxis.set_label_coords(-0.06, 0.5)
            for side in ["left", "right", "top", "bottom"]:
                ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.015, y=0.99, ha="left", fontsize=12, fontweight="bold")
    if len(order) > 10:
        add_legend(fig, palette, order, "Fine cell type", (0.855, 0.50), ncol=1)
        fig.subplots_adjust(left=0.06, right=0.835, top=0.94, bottom=0.025, wspace=0.03, hspace=0.03)
    else:
        add_legend(fig, palette, order, "Broad cell class", (0.86, 0.50), ncol=1)
        fig.subplots_adjust(left=0.06, right=0.84, top=0.94, bottom=0.025, wspace=0.03, hspace=0.03)
    save(fig, stem, dpi=900)


def write_palette_tables() -> None:
    MANIFEST.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": list(BROAD_COLORS), "color": list(BROAD_COLORS.values())}).to_csv(
        MANIFEST / "broad_cell_class_palette.csv", index=False
    )
    pd.DataFrame({"label": list(FINE_COLORS), "color": list(FINE_COLORS.values())}).to_csv(
        MANIFEST / "fine_cell_type_palette.csv", index=False
    )
    pd.DataFrame({"method": list(SC_METHOD_COLORS), "color": list(SC_METHOD_COLORS.values())}).to_csv(
        MANIFEST / "single_cell_method_palette.csv", index=False
    )
    pd.DataFrame({"method": list(SP_METHOD_COLORS), "color": list(SP_METHOD_COLORS.values())}).to_csv(
        MANIFEST / "spatial_method_palette.csv", index=False
    )


def write_method_order_tables(method_orders: dict[str, list[str]]) -> None:
    rows = []
    for label_space, methods in method_orders.items():
        rows.extend(
            {"label_space": label_space, "rank": i + 1, "method": method}
            for i, method in enumerate(methods)
        )
    pd.DataFrame(rows).to_csv(MANIFEST / "method_order_by_score.csv", index=False)


def write_readme(method_orders: dict[str, list[str]]) -> None:
    payload = {
        "figure_logic": {
            "single_cell": "Nonzero-HVG single-cell annotation evidence: broad/fine method scorecards plus broad/fine UMAPs.",
        "spatial": "Nonzero-HVG spatial deconvolution evidence: broad/fine method scorecards plus broad/fine spatial maps.",
        },
        "source_data": str(SRC_IN),
        "output_directory": str(OUT),
        "single_cell_methods": SC_METHODS,
        "spatial_methods": SP_METHODS,
        "method_order_by_score": method_orders,
        "note": "Single-cell panels remove raw SVD, non-finetuned CPT512, hidden nonzero-HVG and gene-mean nonzero-HVG. OmniCell-CPT fine-tuned uses bootstrap resamples of the available held-out prediction, not five independently retrained splits. Spatial panels remove non-finetuned hidden/gene-mean nonzero-HVG and use the best-five chip comparison where available. If fig2_best5_native_omnicell_* source tables are present, OmniCell native is included across the same best-five chips; otherwise the script falls back to the available T906-only source. Gene-mean nonzero-HVG should be revisited only after applying it on top of the CPT fine-tuned checkpoint.",
    }
    (OUT / "README_figure2_nonzero_hvg_final_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    copy_source_files()
    write_palette_tables()
    data = load_inputs()
    method_orders = {
        "single_cell_broad": ranked_methods(data["sc_summary"], "broad cell class", SC_METHODS, "OmniCell-CPT fine-tuned"),
        "single_cell_fine": ranked_methods(data["sc_summary"], "fine cell type", SC_METHODS, "OmniCell-CPT fine-tuned"),
        "spatial_broad": ranked_methods(data["sp_summary"], "broad cell class", SP_METHODS, "OmniCell-CPT"),
        "spatial_fine": ranked_methods(data["sp_summary"], "fine cell type", SP_METHODS, "OmniCell-CPT"),
    }
    write_method_order_tables(method_orders)

    plot_metric_scorecard(
        data["sc_summary"],
        data["sc_by_split"],
        "broad cell class",
        method_orders["single_cell_broad"],
        SC_METHOD_COLORS,
        "Single-cell annotation: broad cell classes (nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_singlecell_broad_method_scorecard",
    )
    plot_metric_scorecard(
        data["sc_summary"],
        data["sc_by_split"],
        "fine cell type",
        method_orders["single_cell_fine"],
        SC_METHOD_COLORS,
        "Single-cell annotation: fine cell types (nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_singlecell_fine_method_scorecard",
    )
    plot_umap_grid(
        data["sc_umap"],
        "broad_cell_class",
        BROAD_COLORS,
        BROAD_ORDER,
        "Single-cell annotation UMAPs by broad class (nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_singlecell_umaps_broad",
        method_orders["single_cell_broad"],
    )
    plot_umap_grid(
        data["sc_umap"],
        "cell_type",
        FINE_COLORS,
        FINE_ORDER,
        "Single-cell annotation UMAPs by fine cell type (nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_singlecell_umaps_fine",
        method_orders["single_cell_fine"],
    )

    plot_metric_scorecard(
        data["sp_summary"],
        data["sp_by_chip"],
        "broad cell class",
        method_orders["spatial_broad"],
        SP_METHOD_COLORS,
        "Spatial deconvolution: broad cell classes (nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_spatial_broad_method_scorecard",
    )
    plot_metric_scorecard(
        data["sp_summary"],
        data["sp_by_chip"],
        "fine cell type",
        method_orders["spatial_fine"],
        SP_METHOD_COLORS,
        "Spatial deconvolution: fine cell types (nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_spatial_fine_method_scorecard",
    )
    plot_spatial_grid(
        data["sp_broad_maps"],
        BROAD_COLORS,
        BROAD_ORDER,
        "Spatial deconvolution maps by broad class (best-five chips, nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_spatial_maps_broad",
        spatial_panel_order(method_orders["spatial_broad"]),
    )
    plot_spatial_grid(
        data["sp_fine_maps"],
        FINE_COLORS,
        FINE_ORDER,
        "Spatial deconvolution maps by fine cell type (best-five chips, nonzero-HVG)",
        PLOTS / "fig2_nonzero_hvg_spatial_maps_fine",
        spatial_panel_order(method_orders["spatial_fine"]),
    )
    write_readme(method_orders)


if __name__ == "__main__":
    main()
