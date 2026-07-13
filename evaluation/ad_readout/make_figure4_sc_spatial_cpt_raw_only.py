#!/usr/bin/env python
"""Generate corrected CPT-vs-raw-SVD single-cell/spatial separability panels.

This script intentionally keeps the statistical unit at sample/chip level for
both AUROC bars and ROC curves. It avoids mixing fold-mean cell/spot AUROC with
pooled cell/spot ROC curves, which can change the apparent CPT-vs-raw ranking.
"""
from __future__ import annotations
import os

import argparse
import math
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


DEFAULT_BI = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/BI"))
FEATURES = [
    ("CPT embedding", "mean_cpt_oof_ad_probability", "#B64342"),
    ("Raw SVD", "mean_raw_svd_oof_ad_probability", "#8C83B8"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bi-root", type=Path, default=DEFAULT_BI)
    parser.add_argument("--n-bootstrap", type=int, default=4000)
    return parser.parse_args()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 160,
    })


def add_strip(ax, title: str) -> None:
    ax.add_patch(mpl.patches.Rectangle((0, 1.025), 1, 0.12, transform=ax.transAxes,
                                       color="#E8E8EA", clip_on=False, zorder=-1))
    ax.text(0.5, 1.085, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=7.2, fontweight="bold")


def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="left", va="bottom")


def save_fig(fig, stem: Path, png_dpi: int = 450) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def load_sample_scores(root: Path) -> pd.DataFrame:
    src = root / "figures" / "figure4_nature_style" / "sc_spatial_addendum" / "source_data" / "Figure4G_sample_chip_axis_scores_source.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing sample/chip source table: {src}")
    df = pd.read_csv(src)
    keep = [
        "modality", "batch_id", "condition_inferred", "n_cells",
        "mean_cpt_oof_ad_probability", "mean_raw_svd_oof_ad_probability",
        "mean_ad_axis_score",
    ]
    df = df[keep].copy()
    df["label"] = (df["condition_inferred"] == "AD").astype(int)
    df["metric_definition"] = (
        "Sample/chip-level mean OOF probability. Used for corrected CPT-vs-raw-SVD "
        "separability panels so AUROC bars and ROC curves share the same statistical unit."
    )
    return df


def bootstrap_auc_sem(y: np.ndarray, score: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    auc = float(roc_auc_score(y, score))
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return auc, 0.0, 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = np.r_[rng.choice(pos, size=len(pos), replace=True),
                    rng.choice(neg, size=len(neg), replace=True)]
        vals.append(roc_auc_score(y[idx], score[idx]))
    vals = np.asarray(vals, float)
    return auc, float(vals.std(ddof=1)), float(np.quantile(vals, 0.975) - np.quantile(vals, 0.025))


def make_sources(sample_df: pd.DataFrame, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    auc_rows = []
    roc_rows = []
    seed_base = 9101
    for m_i, (modality, sub) in enumerate(sample_df.groupby("modality", sort=False)):
        y = sub["label"].to_numpy(int)
        for f_i, (feature, score_col, color) in enumerate(FEATURES):
            raw_score = sub[score_col].to_numpy(float)
            raw_auc = float(roc_auc_score(y, raw_score))
            flip = raw_auc < 0.5
            score = 1.0 - raw_score if flip else raw_score
            auc, sem, ci_width = bootstrap_auc_sem(y, score, n_bootstrap, seed_base + 17 * m_i + f_i)
            fpr, tpr, _ = roc_curve(y, score)
            auc_rows.append({
                "modality": modality,
                "feature_space": feature,
                "score_column": score_col,
                "n_samples_or_chips": int(len(sub)),
                "n_ad": int(y.sum()),
                "n_control": int((y == 0).sum()),
                "raw_auroc_before_orientation": raw_auc,
                "orientation_flipped": bool(flip),
                "sample_chip_oriented_auroc": auc,
                "bootstrap_sem": sem,
                "bootstrap_95pct_width": ci_width,
                "metric_definition": (
                    "AUROC after fixing score orientation at sample/chip level; "
                    "error bar is stratified bootstrap SEM over samples/chips."
                ),
            })
            for x, yy in zip(fpr, tpr):
                roc_rows.append({
                    "modality": modality,
                    "feature_space": feature,
                    "fpr": float(x),
                    "tpr": float(yy),
                    "sample_chip_oriented_auroc": auc,
                    "orientation_flipped": bool(flip),
                    "metric_definition": "Sample/chip-level ROC using the same oriented scores as the AUROC bars.",
                })
    return pd.DataFrame(auc_rows), pd.DataFrame(roc_rows)


def plot_auc_panel(ax, auc_df: pd.DataFrame, modality: str, panel: str) -> None:
    sub = auc_df[auc_df["modality"].eq(modality)].copy()
    x = np.arange(len(sub))
    vals = sub["sample_chip_oriented_auroc"].to_numpy(float)
    errs = sub["bootstrap_sem"].to_numpy(float)
    colors = ["#B64342" if fs == "CPT embedding" else "#8C83B8" for fs in sub["feature_space"]]
    ax.bar(x, vals, yerr=errs, color=colors, edgecolor="#333333", linewidth=0.5,
           width=0.58, capsize=2)
    for xx, val in zip(x, vals):
        ax.text(xx, min(1.03, val + 0.035), f"{val:.2f}", ha="center", va="bottom", fontsize=6.2)
    n_ad = int(sub["n_ad"].iloc[0])
    n_ctl = int(sub["n_control"].iloc[0])
    ax.axhline(0.5, color="#AFAFAF", lw=0.9, ls=":")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["feature_space"].tolist())
    ax.set_ylabel("AUROC")
    ax.grid(axis="y", color="#DCE2EC", lw=0.8)
    ax.set_axisbelow(True)
    add_strip(ax, f"{modality}: CPT vs raw SVD (n={n_ad} AD/{n_ctl} Control)")
    add_panel_label(ax, panel)


def plot_roc_panel(ax, roc_df: pd.DataFrame, modality: str, panel: str) -> None:
    sub = roc_df[roc_df["modality"].eq(modality)].copy()
    for feature, _, color in FEATURES:
        df = sub[sub["feature_space"].eq(feature)]
        if df.empty:
            continue
        auc = float(df["sample_chip_oriented_auroc"].iloc[0])
        ax.step(df["fpr"], df["tpr"], where="post", color=color, lw=1.5,
                label=f"{feature} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#B8B8B8", lw=0.9, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(color="#DCE2EC", lw=0.8)
    ax.legend(loc="lower right", fontsize=6, handlelength=2.0)
    add_strip(ax, f"{modality}: sample/chip-level ROC")
    add_panel_label(ax, panel)


def make_composite(out_dir: Path, auc_df: pd.DataFrame, roc_df: pd.DataFrame) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.4))
    plot_auc_panel(axes[0, 0], auc_df, "Single-cell", "A")
    plot_roc_panel(axes[0, 1], roc_df, "Single-cell", "B")
    plot_auc_panel(axes[1, 0], auc_df, "Spatial", "C")
    plot_roc_panel(axes[1, 1], roc_df, "Spatial", "D")
    fig.subplots_adjust(wspace=0.44, hspace=0.58)
    save_fig(fig, out_dir / "Figure4G_CPT_raw_only_SC_ST_sample_chip")
    plt.close(fig)


def make_individual(out_dir: Path, auc_df: pd.DataFrame, roc_df: pd.DataFrame) -> None:
    setup_style()
    jobs = [
        ("Figure4G_A_singlecell_CPT_vs_raw_AUROC", (2.75, 2.25), lambda ax: plot_auc_panel(ax, auc_df, "Single-cell", "A")),
        ("Figure4G_B_singlecell_CPT_vs_raw_ROC", (2.85, 2.25), lambda ax: plot_roc_panel(ax, roc_df, "Single-cell", "B")),
        ("Figure4G_C_spatial_CPT_vs_raw_AUROC", (2.75, 2.25), lambda ax: plot_auc_panel(ax, auc_df, "Spatial", "C")),
        ("Figure4G_D_spatial_CPT_vs_raw_ROC", (2.85, 2.25), lambda ax: plot_roc_panel(ax, roc_df, "Spatial", "D")),
    ]
    for name, figsize, draw in jobs:
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax)
        save_fig(fig, out_dir / name)
        plt.close(fig)


def write_readme(out_dir: Path, auc_df: pd.DataFrame) -> None:
    lines = [
        "# Corrected Figure 4G: CPT embedding versus raw SVD only",
        "",
        f"Output root: `{out_dir}`",
        "",
        "Purpose: replace the earlier four-feature single-cell/spatial separability display with a stricter CPT-vs-raw-SVD comparison.",
        "",
        "Important correction: AUROC bars and ROC curves here both use sample/chip-level aggregated OOF probabilities. The earlier addendum mixed fold-mean cell/spot AUROC in the bar plot with pooled cell/spot ROC curves, so CPT-vs-raw differences could look inconsistent.",
        "",
        "Panels:",
        "- A: single-cell sample-level AUROC, CPT embedding versus raw SVD.",
        "- B: single-cell sample-level ROC, CPT embedding versus raw SVD.",
        "- C: spatial chip-level AUROC, CPT embedding versus raw SVD.",
        "- D: spatial chip-level ROC, CPT embedding versus raw SVD.",
        "",
        "Caveat: spatial smoke analysis has only two AD chips and two Control chips; these panels show separability of the current smoke subset, not definitive external validation.",
        "",
        "AUROC source:",
        auc_df.to_csv(index=False),
    ]
    (out_dir / "README_CPT_raw_only.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.bi_root
    out_dir = root / "figures" / "figure4_nature_style" / "sc_spatial_cpt_raw_only"
    src_dir = out_dir / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    sample_df = load_sample_scores(root)
    auc_df, roc_df = make_sources(sample_df, args.n_bootstrap)

    sample_df.to_csv(src_dir / "Figure4G_CPT_raw_only_sample_chip_scores_source.csv", index=False)
    auc_df.to_csv(src_dir / "Figure4G_CPT_raw_only_AUROC_source.csv", index=False)
    roc_df.to_csv(src_dir / "Figure4G_CPT_raw_only_ROC_source.csv", index=False)

    make_composite(out_dir, auc_df, roc_df)
    make_individual(out_dir, auc_df, roc_df)
    write_readme(out_dir, auc_df)
    print(out_dir)


if __name__ == "__main__":
    main()
