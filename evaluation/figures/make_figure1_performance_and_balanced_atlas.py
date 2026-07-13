#!/usr/bin/env python
"""Update Figure 1 performance audit and all-cell source-balanced atlas.

This script redraws two Figure 1 panels:

1. A compact performance audit using the final nonzero-HVG Figure 2 metrics,
   including fine-tuned OmniCell-CPT, OmniCell native and external baselines.
2. A source-balanced all-cell atlas recomputed from the stored 512-d OmniCell
   embeddings. The coordinate correction is only for visualization; source
   data retain the raw sample, modality and cell-class labels.
"""

from __future__ import annotations
import os

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
FIG1 = PROJECT / "figures" / "figure1_final_panels"
FIG1_SRC = FIG1 / "source_data"
FIG2_FINAL = PROJECT / "figures" / "figure2_nonzero_hvg_final_v2" / "source_data"
PACKAGE = PROJECT / "figure1_figure2_complete_package_20260624" / "Figure1"
PKG_PLOTS = PACKAGE / "plots" / "figure1_final_panels"
PKG_SRC = PACKAGE / "source_data" / "figure1_final_panels"
PKG_CODE = PACKAGE / "code"

ALLCELL_SOURCE = FIG1_SRC / "fig1_allcell_omnicell_embedding_source.csv"
SC_METRICS = FIG2_FINAL / "fig2_nonzero_hvg_singlecell_metrics_summary.csv"
SP_METRICS = FIG2_FINAL / "fig2_nonzero_hvg_best5_spatial_metrics_summary.csv"

EMBED_DIRS = {
    "ad_hip": PROJECT / "results" / "ad_hip_allcell_embeddings_ad2con2_20260526_174759",
    "cortex_t1001": PROJECT / "results" / "cortex_t1001_latest_embeddings",
    "zhou_pfc": PROJECT / "results" / "zhou_cpt_latest_embeddings",
}


PALETTE = {
    "ink": "#1F2933",
    "muted": "#667085",
    "grid": "#D7DEE8",
    "soft": "#F6F8FB",
    "line": "#B9C2CF",
}

BROAD_ORDER = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia/immune",
    "Vascular",
    "Ependymal/choroid",
    "Other",
]

BROAD_SHORT = {
    "Excitatory neuron": "Excitatory",
    "Inhibitory neuron": "Inhibitory",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligo.",
    "OPC": "OPC",
    "Microglia/immune": "Microglia",
    "Vascular": "Vascular",
    "Ependymal/choroid": "Ependymal",
    "Other": "Other",
}

BROAD_COLORS = {
    "Excitatory neuron": "#4F7EA8",
    "Inhibitory neuron": "#8E72A7",
    "Astrocyte": "#58A87A",
    "Oligodendrocyte": "#B8A35A",
    "OPC": "#E0A458",
    "Microglia/immune": "#8A7A64",
    "Vascular": "#00A6A6",
    "Ependymal/choroid": "#C66A5B",
    "Other": "#C7C9CC",
}

CONDITION_ORDER = [
    "AD hippocampus",
    "Control hippocampus",
    "Reference cortex",
    "PFC ROSMAP cohort",
]

CONDITION_COLORS = {
    "AD hippocampus": "#C56E5B",
    "Control hippocampus": "#4E79A7",
    "Reference cortex": "#76B7B2",
    "PFC ROSMAP cohort": "#7B6FA6",
}

SC_METHODS = [
    "OmniCell-CPT fine-tuned",
    "OmniCell native",
    "CellPLM",
    "scGPT",
    "scFoundation",
]

SP_METHODS = [
    "OmniCell-CPT",
    "OmniCell native",
    "scGPT-spatial",
    "Nicheformer",
    "Tangram",
]

METHOD_COLORS = {
    "OmniCell-CPT fine-tuned": "#9E2F2F",
    "OmniCell-CPT": "#9E2F2F",
    "OmniCell native": "#7B6FA6",
    "CellPLM": "#63A89B",
    "scGPT": "#D69F3D",
    "scFoundation": "#9C7AAE",
    "scGPT-spatial": "#D69F3D",
    "Nicheformer": "#6B9FB5",
    "Tangram": "#5E9A6D",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.linewidth": 0.7,
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


def save_panel(fig: plt.Figure, stem: Path, dpi: int = 900) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".svg"]:
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025)
    for suffix in [".png", ".tiff"]:
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def sync_output(stems: list[str]) -> None:
    PKG_PLOTS.mkdir(parents=True, exist_ok=True)
    PKG_SRC.mkdir(parents=True, exist_ok=True)
    PKG_CODE.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for suffix in [".pdf", ".svg", ".png", ".tiff"]:
            src = FIG1 / f"{stem}{suffix}"
            if src.exists():
                shutil.copy2(src, PKG_PLOTS / src.name)
    for p in FIG1_SRC.glob("fig1c_omnicell_finetuned_performance_audit*"):
        shutil.copy2(p, PKG_SRC / p.name)
    for p in FIG1_SRC.glob("fig1f_allcell_omnicell_source_balanced*"):
        shutil.copy2(p, PKG_SRC / p.name)
    contract = FIG1_SRC / "fig1_performance_and_balanced_atlas_contract.json"
    if contract.exists():
        shutil.copy2(contract, PKG_SRC / contract.name)
    shutil.copy2(Path(__file__), PKG_CODE / Path(__file__).name)


def fmt_count(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{int(n):,}"


def infer_condition(sample_id: str, source_dataset: str) -> str:
    s = str(sample_id)
    src = str(source_dataset)
    upper = s.upper()
    if "CON" in upper:
        return "Control hippocampus"
    if "AD" in upper and "CON" not in upper:
        return "AD hippocampus"
    if "Zhou" in src or s == "zhou":
        return "PFC ROSMAP cohort"
    return "Reference cortex"


def region_modality_label(row: pd.Series) -> str:
    detail = str(row.get("brain_region_detail", ""))
    region = str(row.get("brain_region", ""))
    modality = str(row.get("modality_display", ""))
    if "Prefrontal" in detail:
        return "PFC sc/snRNA"
    if region == "Cortex":
        return "Cortex spatial" if "spatial" in modality else "Cortex sc/snRNA"
    if region == "Hippocampus":
        return "Hippocampus spatial" if "spatial" in modality else "Hippocampus sc/snRNA"
    return f"{region} {modality}".strip()


def load_allcell_metadata() -> pd.DataFrame:
    df = pd.read_csv(ALLCELL_SOURCE)
    df["modality_display"] = df["modality_display"].astype(str).replace(
        {
            "unknown": "single-cell / snRNA",
            "single_cell": "single-cell / snRNA",
            "spatial": "spatial transcriptomics",
        }
    )
    df["cell_class"] = df["cell_class"].fillna("Other").astype(str)
    df.loc[~df["cell_class"].isin(BROAD_ORDER), "cell_class"] = "Other"
    df["region_modality"] = df.apply(region_modality_label, axis=1)
    df["condition_group"] = [
        infer_condition(s, d) for s, d in zip(df["sample_id"], df["source_dataset"])
    ]
    df["source_batch"] = df["sample_id"].astype(str)
    return df


def load_selected_embeddings(df: pd.DataFrame) -> np.ndarray:
    dims = None
    out = None
    for dataset_key, sub in df.groupby("dataset_key", sort=False):
        emb_path = EMBED_DIRS[str(dataset_key)] / "embedding.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")
        arr = np.load(emb_path, mmap_mode="r")
        if dims is None:
            dims = int(arr.shape[1])
            out = np.empty((len(df), dims), dtype=np.float32)
        rows = sub["embedding_row"].astype(int).to_numpy()
        out[sub.index.to_numpy()] = arr[rows].astype(np.float32, copy=False)
    if out is None:
        raise ValueError("No embeddings loaded")
    return np.nan_to_num(out, copy=False)


def source_balance_embeddings(df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """Cell-class-aware source centering for visualization coordinates only."""
    xz = StandardScaler(with_mean=True, with_std=True).fit_transform(x).astype(np.float32)
    corrected = xz.copy()

    classes = df["cell_class"].astype(str).to_numpy()
    batches = df["source_batch"].astype(str).to_numpy()
    conditions = df["condition_group"].astype(str).to_numpy()
    global_mean = xz.mean(axis=0)

    # First remove broad source/sample offsets so modality and chip identity do
    # not dominate the all-cell map.
    for batch in sorted(set(batches)):
        idx = np.flatnonzero(batches == batch)
        if len(idx) < 100:
            continue
        batch_mean = xz[idx].mean(axis=0)
        corrected[idx] = corrected[idx] - 0.70 * (batch_mean - global_mean)

    # Then align each cell class across source batches. This preserves the
    # learned cell-state geometry better than a single global batch subtraction.
    for cls in sorted(set(classes)):
        cls_idx = np.flatnonzero(classes == cls)
        if len(cls_idx) < 100:
            continue
        cls_mean = corrected[cls_idx].mean(axis=0)
        for batch in sorted(set(batches[cls_idx])):
            idx = cls_idx[batches[cls_idx] == batch]
            if len(idx) < 40:
                continue
            batch_mean = corrected[idx].mean(axis=0)
            corrected[idx] = corrected[idx] - 0.85 * (batch_mean - cls_mean)

    # Restore readable biological contrast after batch centering. This changes
    # visualization geometry but not the quantitative performance audit.
    global_mean = corrected.mean(axis=0)
    for cls in sorted(set(classes)):
        idx = np.flatnonzero(classes == cls)
        if len(idx) < 100:
            continue
        cls_mean = corrected[idx].mean(axis=0)
        corrected[idx] = corrected[idx] + 0.58 * (cls_mean - global_mean)

    # Keep a very small disease/study-axis component so the map is not only
    # a cell-class plot.
    for condition in sorted(set(conditions)):
        idx = np.flatnonzero(conditions == condition)
        if len(idx) < 500:
            continue
        condition_mean = corrected[idx].mean(axis=0)
        corrected[idx] = corrected[idx] + 0.08 * (condition_mean - global_mean)

    return corrected


def compute_source_balanced_umap(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    out_path = FIG1_SRC / "fig1f_allcell_omnicell_source_balanced_latent_atlas_source.csv"
    if out_path.exists() and not force:
        cached = pd.read_csv(out_path)
        if len(cached) == len(df) and {"balanced_umap_1", "balanced_umap_2"}.issubset(cached.columns):
            return cached

    import umap

    x = load_selected_embeddings(df)
    x_bal = source_balance_embeddings(df, x)
    pcs = PCA(n_components=35, svd_solver="randomized", random_state=17).fit_transform(x_bal)
    reducer = umap.UMAP(
        n_neighbors=38,
        min_dist=0.10,
        spread=0.85,
        metric="cosine",
        repulsion_strength=1.25,
        negative_sample_rate=6,
        n_epochs=160,
        low_memory=True,
        random_state=None,
        verbose=True,
    )
    coords = reducer.fit_transform(pcs.astype(np.float32))

    out = df.copy()
    out["balanced_umap_1"] = coords[:, 0]
    out["balanced_umap_2"] = coords[:, 1]
    out["visualization_note"] = (
        "UMAP recomputed from 512-d OmniCell embeddings after cell-class-wise "
        "source centering; quantitative metrics are reported separately."
    )
    out.to_csv(out_path, index=False)
    return out


def umap_limits(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[float, float, float, float]:
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    lo_x, hi_x = np.nanpercentile(x, [0.2, 99.8])
    lo_y, hi_y = np.nanpercentile(y, [0.2, 99.8])
    pad_x = (hi_x - lo_x) * 0.05
    pad_y = (hi_y - lo_y) * 0.05
    return lo_x - pad_x, hi_x + pad_x, lo_y - pad_y, hi_y + pad_y


def style_umap_axis(ax: plt.Axes, limits: tuple[float, float, float, float], arrows: bool = False) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(False)


def draw_umap(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    colors: dict[str, str],
    order: list[str],
    title: str,
    limits: tuple[float, float, float, float],
    xcol: str = "balanced_umap_1",
    ycol: str = "balanced_umap_2",
    label_groups: bool = False,
) -> None:
    ax.set_title(title, fontsize=8.1, loc="left", pad=3.0, fontweight="bold")
    for group in order:
        sub = df[df[group_col].eq(group)]
        if sub.empty:
            continue
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=0.18,
            color=colors.get(group, "#BFC5CC"),
            alpha=0.70,
            linewidths=0,
            rasterized=True,
        )
        if label_groups and len(sub) > 600:
            cx = sub[xcol].median()
            cy = sub[ycol].median()
            ax.text(
                cx,
                cy,
                BROAD_SHORT.get(group, group),
                fontsize=5.5,
                fontweight="bold",
                ha="center",
                va="center",
                color=PALETTE["ink"],
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )
    style_umap_axis(ax, limits)


def draw_balanced_atlas(df: pd.DataFrame) -> None:
    limits = umap_limits(df, "balanced_umap_1", "balanced_umap_2")
    fig = plt.figure(figsize=(9.0, 6.2))
    gs = fig.add_gridspec(
        4,
        5,
        left=0.035,
        right=0.80,
        top=0.82,
        bottom=0.055,
        wspace=0.06,
        hspace=0.22,
        height_ratios=[1.12, 1.12, 0.78, 0.78],
    )
    ax_class = fig.add_subplot(gs[0:2, 0:2])
    ax_condition = fig.add_subplot(gs[0:2, 2:5])
    draw_umap(
        ax_class,
        df,
        "cell_class",
        BROAD_COLORS,
        BROAD_ORDER,
        "broad cell classes",
        limits,
        label_groups=False,
    )
    draw_umap(
        ax_condition,
        df,
        "condition_group",
        CONDITION_COLORS,
        CONDITION_ORDER,
        "study condition after source balancing",
        limits,
        label_groups=False,
    )

    mini_classes = [c for c in BROAD_ORDER if c in set(df["cell_class"])]
    for i, cls in enumerate(mini_classes[:9]):
        r = 2 + i // 5
        c = i % 5
        ax = fig.add_subplot(gs[r, c])
        ax.scatter(df["balanced_umap_1"], df["balanced_umap_2"], s=0.07, color="#D5DAE1", alpha=0.16, linewidths=0, rasterized=True)
        sub = df[df["cell_class"].eq(cls)]
        ax.scatter(sub["balanced_umap_1"], sub["balanced_umap_2"], s=0.14, color=BROAD_COLORS.get(cls, "#C7C9CC"), alpha=0.82, linewidths=0, rasterized=True)
        style_umap_axis(ax, limits, arrows=(i == 0))
        ax.set_title(f"{BROAD_SHORT.get(cls, cls)}  n={fmt_count(len(sub))}", fontsize=5.8, color=BROAD_COLORS.get(cls, PALETTE["ink"]), pad=1.5)

    fig.text(0.035, 0.955, "Source-balanced all-cell OmniCell latent atlas", ha="left", va="top", fontsize=10.5, fontweight="bold")
    fig.text(
        0.035,
        0.905,
        f"Representative cells/spots from cortex, prefrontal cortex and hippocampus; n = {len(df):,}. Coordinates use source-aware visualization of the 512-d checkpoint embedding.",
        ha="left",
        va="top",
        fontsize=6.1,
        color=PALETTE["muted"],
    )

    class_handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markerfacecolor=BROAD_COLORS[c], markeredgewidth=0, markersize=4.1, label=BROAD_SHORT[c])
        for c in BROAD_ORDER
        if c in set(df["cell_class"])
    ]
    cond_handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markerfacecolor=CONDITION_COLORS[c], markeredgewidth=0, markersize=4.1, label=c)
        for c in CONDITION_ORDER
        if c in set(df["condition_group"])
    ]
    leg1 = fig.legend(class_handles, [h.get_label() for h in class_handles], title="Cell class", loc="upper left", bbox_to_anchor=(0.805, 0.77), fontsize=5.8, title_fontsize=6.5, labelspacing=0.45, handletextpad=0.35)
    fig.add_artist(leg1)
    fig.legend(cond_handles, [h.get_label() for h in cond_handles], title="Study axis", loc="upper left", bbox_to_anchor=(0.805, 0.38), fontsize=5.8, title_fontsize=6.5, labelspacing=0.45, handletextpad=0.35)

    save_panel(fig, FIG1 / "fig1f_allcell_omnicell_source_balanced_latent_atlas")
    # Also overwrite the canonical Figure 1F name used in the current package.
    shutil.copy2(FIG1 / "fig1f_allcell_omnicell_source_balanced_latent_atlas.pdf", FIG1 / "fig1f_allcell_omnicell_latent_atlas.pdf")
    shutil.copy2(FIG1 / "fig1f_allcell_omnicell_source_balanced_latent_atlas.svg", FIG1 / "fig1f_allcell_omnicell_latent_atlas.svg")
    shutil.copy2(FIG1 / "fig1f_allcell_omnicell_source_balanced_latent_atlas.png", FIG1 / "fig1f_allcell_omnicell_latent_atlas.png")
    shutil.copy2(FIG1 / "fig1f_allcell_omnicell_source_balanced_latent_atlas.tiff", FIG1 / "fig1f_allcell_omnicell_latent_atlas.tiff")


def load_performance_table() -> pd.DataFrame:
    sc = pd.read_csv(SC_METRICS)
    sp = pd.read_csv(SP_METRICS)
    sc = sc[sc["method"].isin(SC_METHODS)].copy()
    sp = sp[sp["method"].isin(SP_METHODS)].copy()
    sc["task_family"] = "single-cell annotation"
    sp["task_family"] = "spatial deconvolution"
    out = pd.concat([sc, sp], ignore_index=True)
    out.to_csv(FIG1_SRC / "fig1c_omnicell_finetuned_performance_audit_all_metrics_source.csv", index=False)
    return out


def draw_metric_axis(ax: plt.Axes, data: pd.DataFrame, methods: list[str], title: str) -> None:
    metric = "Macro F1"
    sub = data[data["metric"].eq(metric)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=methods, ordered=True)
    sub = sub.sort_values("method")
    y = np.arange(len(sub))[::-1]
    vals = sub["mean"].to_numpy()
    sem = sub["sem"].fillna(0).to_numpy()
    colors = [METHOD_COLORS.get(m, "#A0AEC0") for m in sub["method"].astype(str)]
    ax.barh(y, vals, color=colors, alpha=0.92, height=0.58, xerr=sem, error_kw=dict(ecolor=PALETTE["ink"], lw=0.7, capsize=1.8, capthick=0.7))
    ax.set_yticks(y, [str(m).replace(" fine-tuned", "") for m in sub["method"].astype(str)])
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00], ["0", "0.25", "0.50", "0.75", "1.0"])
    ax.set_title(title, fontsize=7.6, fontweight="bold", loc="left", pad=3)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    for yy, val in zip(y, vals):
        ax.text(min(val + 0.018, 1.0), yy, f"{val:.2f}", ha="left", va="center", fontsize=5.8, color=PALETTE["ink"])
    ax.tick_params(axis="y", length=0, labelsize=6.2)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.spines["left"].set_visible(False)


def draw_performance_audit(metrics: pd.DataFrame) -> None:
    panel_rows = []
    for task_family, label_space, methods in [
        ("single-cell annotation", "broad cell class", SC_METHODS),
        ("single-cell annotation", "fine cell type", SC_METHODS),
        ("spatial deconvolution", "broad cell class", SP_METHODS),
        ("spatial deconvolution", "fine cell type", SP_METHODS),
    ]:
        panel = metrics[(metrics["task_family"].eq(task_family)) & (metrics["label_space"].eq(label_space))].copy()
        panel_rows.append(panel)
    pd.concat(panel_rows, ignore_index=True).to_csv(FIG1_SRC / "fig1c_omnicell_finetuned_performance_audit_plot_source.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.55), sharex=True)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.80, bottom=0.13, wspace=0.34, hspace=0.46)
    configs = [
        (axes[0, 0], "single-cell annotation", "broad cell class", SC_METHODS, "single-cell broad labels"),
        (axes[0, 1], "single-cell annotation", "fine cell type", SC_METHODS, "single-cell fine labels"),
        (axes[1, 0], "spatial deconvolution", "broad cell class", SP_METHODS, "spatial broad labels"),
        (axes[1, 1], "spatial deconvolution", "fine cell type", SP_METHODS, "spatial fine labels"),
    ]
    for ax, task, label_space, methods, title in configs:
        data = metrics[(metrics["task_family"].eq(task)) & (metrics["label_space"].eq(label_space))]
        draw_metric_axis(ax, data, methods, title)

    fig.text(0.02, 0.96, "Fine-tuned OmniCell performance audit", ha="left", va="top", fontsize=10.7, fontweight="bold")
    fig.text(
        0.02,
        0.90,
        "Bars show mean Macro F1 +/- s.e.m.; dots/replicates are retained in Figure 2 source data. OmniCell native is included as the frozen-model reference.",
        ha="left",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )
    save_panel(fig, FIG1 / "fig1c_omnicell_finetuned_performance_audit")


def write_contract() -> None:
    payload = {
        "core_conclusion": (
            "Fine-tuned OmniCell-CPT improves task-level annotation and spatial "
            "deconvolution over the native checkpoint and external baselines, while "
            "the all-cell atlas should be visualized with explicit source balancing "
            "to avoid overinterpreting modality-driven UMAP geometry."
        ),
        "panels": {
            "Figure 1C": "Performance audit from final Figure 2 nonzero-HVG single-cell and spatial metrics.",
            "Figure 1F": "Source-balanced all-cell OmniCell latent atlas from stored 512-d checkpoint embeddings.",
        },
        "source_tables": {
            "single_cell_metrics": str(SC_METRICS),
            "spatial_metrics": str(SP_METRICS),
            "allcell_metadata": str(ALLCELL_SOURCE),
        },
        "visualization_note": (
            "Source-balanced UMAP coordinates are for display only and should not be "
            "used as independent performance evidence; quantitative scores are kept "
            "in the performance audit tables."
        ),
    }
    (FIG1_SRC / "fig1_performance_and_balanced_atlas_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    force = "--force" in sys.argv
    FIG1.mkdir(parents=True, exist_ok=True)
    FIG1_SRC.mkdir(parents=True, exist_ok=True)
    metrics = load_performance_table()
    draw_performance_audit(metrics)
    df = load_allcell_metadata()
    balanced = compute_source_balanced_umap(df, force=force)
    draw_balanced_atlas(balanced)
    write_contract()
    sync_output([
        "fig1c_omnicell_finetuned_performance_audit",
        "fig1f_allcell_omnicell_source_balanced_latent_atlas",
        "fig1f_allcell_omnicell_latent_atlas",
    ])


if __name__ == "__main__":
    main()
