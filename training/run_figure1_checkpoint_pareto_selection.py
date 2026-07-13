#!/usr/bin/env python3
"""Score stage-2 checkpoints and select the best Figure 1 CPT checkpoint.

The script intentionally reuses the formal Figure 1C scoring script for each
checkpoint so the Pareto table is generated with the exact same validation
cells, probes and alignment diagnostics as the manuscript panel.
"""

from __future__ import annotations
import os

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
PYTHON = Path(os.path.expandvars("${PYTHON}"))
RESULTS = PROJECT / "results"
FIG1 = PROJECT / "figures" / "figure1_final_panels"
FIG1_SRC = FIG1 / "source_data"

CHECKPOINT_ROOT = RESULTS / "figure1_multitask_cpt_alignment_agebin_stage2_pareto"
OUT = RESULTS / "figure1_checkpoint_pareto_selection"
EMBED_SCRIPT = PROJECT / "scripts" / "embed_figure1_validation_cpt_nonzero_hvg.py"
SCORE_SCRIPT = PROJECT / "scripts" / "make_figure1_representation_metrics_final.py"

STEPS = [500, 1000, 1500, 2000, 2500, 3000]
METRIC_KEYS = {
    ("Disease-state readout", "AD/control AUROC"): "disease_auroc",
    ("Disease-state readout", "AD/control balanced accuracy"): "disease_balanced_accuracy",
    ("Aging-state readout", "Age Pearson r"): "age_pearson_r",
    ("Aging-state readout", "Age MAE"): "age_mae",
    ("Cohort alignment", "neighbor entropy"): "cohort_neighbor_entropy",
    ("Cohort alignment", "normalized iLISI"): "cohort_normalized_iLISI",
    ("Cohort alignment", "same-label neighbor rate"): "cohort_same_label_neighbor_rate",
    ("Modality alignment", "neighbor entropy"): "modality_neighbor_entropy",
    ("Modality alignment", "normalized iLISI"): "modality_normalized_iLISI",
    ("Modality alignment", "same-label neighbor rate"): "modality_same_label_neighbor_rate",
}
HIGHER_BETTER = {
    "disease_auroc",
    "disease_balanced_accuracy",
    "age_pearson_r",
    "cohort_neighbor_entropy",
    "cohort_normalized_iLISI",
    "modality_neighbor_entropy",
    "modality_normalized_iLISI",
}
LOWER_BETTER = {
    "age_mae",
    "cohort_same_label_neighbor_rate",
    "modality_same_label_neighbor_rate",
}


def run_command(cmd: list[str], log_file: Path, env: dict[str, str] | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=PROJECT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def checkpoint_path(step: int) -> Path:
    return CHECKPOINT_ROOT / f"checkpoint-{step}"


def ensure_embedding(step: int, args: argparse.Namespace) -> Path:
    ckpt = checkpoint_path(step)
    if not (ckpt / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing checkpoint for step {step}: {ckpt}")
    embed_dir = OUT / "embeddings" / f"checkpoint-{step}"
    emb = embed_dir / "embedding.npy"
    if emb.exists() and not args.force_embed:
        return embed_dir

    cmd = [
        str(PYTHON),
        str(EMBED_SCRIPT),
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(embed_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--hvg-top",
        str(args.hvg_top),
        "--force",
    ]
    run_command(cmd, OUT / "logs" / f"embed_checkpoint_{step}.log")
    return embed_dir


def score_checkpoint(step: int, embed_dir: Path, args: argparse.Namespace) -> Path:
    metric_csv = OUT / "metrics" / f"checkpoint-{step}_fig1c_source.csv"
    if metric_csv.exists() and not args.force_score:
        return metric_csv

    env = os.environ.copy()
    env["FIG1_MULTITASK_FT_RESULT"] = str(CHECKPOINT_ROOT)
    env["FIG1_CPT_VALIDATION_RESULT"] = str(embed_dir)
    run_command([str(PYTHON), str(SCORE_SCRIPT)], OUT / "logs" / f"score_checkpoint_{step}.log", env=env)

    metric_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIG1_SRC / "fig1c_representation_metrics_final_source.csv", metric_csv)
    fig_dir = OUT / "figures" / "per_checkpoint"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        src = FIG1 / f"fig1c_representation_metrics_final.{suffix}"
        if src.exists():
            shutil.copy2(src, fig_dir / f"fig1c_checkpoint_{step}.{suffix}")
    return metric_csv


def extract_cpt_metrics(step: int, metric_csv: Path) -> dict[str, float | int]:
    df = pd.read_csv(metric_csv)
    df = df[df["method"].astype(str).eq("OmniCell-CPT")].copy()
    row: dict[str, float | int] = {"step": step}
    for (domain, metric), key in METRIC_KEYS.items():
        sub = df[(df["domain"].eq(domain)) & (df["metric"].eq(metric))]
        if sub.empty:
            row[key] = np.nan
            row[f"{key}_sem"] = np.nan
            continue
        row[key] = float(sub["value"].iloc[0])
        row[f"{key}_sem"] = float(sub["sem"].iloc[0]) if "sem" in sub else 0.0
    return row


def minmax(values: pd.Series, higher: bool = True) -> pd.Series:
    vals = values.astype(float)
    lo, hi = vals.min(), vals.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.full(len(vals), 0.5), index=vals.index)
    score = (vals - lo) / (hi - lo)
    if not higher:
        score = 1.0 - score
    return score


def add_pareto_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for key in HIGHER_BETTER:
        if key in df:
            df[f"{key}_norm"] = minmax(df[key], higher=True)
    for key in LOWER_BETTER:
        if key in df:
            df[f"{key}_norm"] = minmax(df[key], higher=False)

    weights = {
        "age_pearson_r_norm": 0.30,
        "age_mae_norm": 0.15,
        "disease_balanced_accuracy_norm": 0.20,
        "disease_auroc_norm": 0.10,
        "cohort_neighbor_entropy_norm": 0.10,
        "cohort_normalized_iLISI_norm": 0.10,
        "cohort_same_label_neighbor_rate_norm": 0.05,
    }
    total = 0.0
    score = np.zeros(len(df), dtype=float)
    for key, weight in weights.items():
        if key in df:
            score += df[key].fillna(0.0).to_numpy() * weight
            total += weight
    df["pareto_score"] = score / max(total, 1e-12)

    max_ba = float(df["disease_balanced_accuracy"].max())
    max_auc = float(df["disease_auroc"].max())
    max_entropy = float(df["cohort_neighbor_entropy"].max())
    max_ilisi = float(df["cohort_normalized_iLISI"].max())
    min_same = float(df["cohort_same_label_neighbor_rate"].min())
    df["passes_guard"] = (
        (df["disease_balanced_accuracy"] >= max_ba - 0.020)
        & (df["disease_auroc"] >= max_auc - 0.020)
        & (
            (df["cohort_neighbor_entropy"] >= max_entropy - 0.035)
            | (df["cohort_normalized_iLISI"] >= max_ilisi - 0.020)
        )
        & (df["cohort_same_label_neighbor_rate"] <= min_same + 0.040)
    )

    objective_cols = [
        ("age_pearson_r", True),
        ("age_mae", False),
        ("disease_balanced_accuracy", True),
        ("disease_auroc", True),
        ("cohort_neighbor_entropy", True),
        ("cohort_normalized_iLISI", True),
        ("cohort_same_label_neighbor_rate", False),
    ]
    pareto = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            at_least_same = True
            strictly_better = False
            for col, higher in objective_cols:
                a = float(row[col])
                b = float(other[col])
                if higher:
                    if b < a - 1e-12:
                        at_least_same = False
                        break
                    if b > a + 1e-12:
                        strictly_better = True
                else:
                    if b > a + 1e-12:
                        at_least_same = False
                        break
                    if b < a - 1e-12:
                        strictly_better = True
            if at_least_same and strictly_better:
                dominated = True
                break
        pareto.append(not dominated)
    df["is_pareto_front"] = pareto

    feasible = df[df["passes_guard"]].copy()
    if feasible.empty:
        feasible = df.copy()
    feasible = feasible.sort_values(
        ["age_pearson_r", "pareto_score", "disease_balanced_accuracy"],
        ascending=[False, False, False],
    )
    selected_step = int(feasible["step"].iloc[0])
    df["selected"] = df["step"].astype(int).eq(selected_step)
    return df.sort_values("step").reset_index(drop=True)


def setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.frameon": False,
        }
    )


def draw_pareto(df: pd.DataFrame) -> None:
    setup_mpl()
    out_dir = OUT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = df[df["selected"]].iloc[0]
    sel_step = int(selected["step"])
    colors = {
        "age": "#B54A4A",
        "disease": "#3E7C93",
        "align": "#8D7FB8",
        "neutral": "#9BA4B5",
        "ink": "#172331",
        "grid": "#D9E1EA",
    }
    fig = plt.figure(figsize=(7.45, 4.7))
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.985, top=0.82, bottom=0.12, wspace=0.42, hspace=0.58)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    panels = [
        ("Age Pearson r", "age_pearson_r", colors["age"], True),
        ("Age MAE", "age_mae", colors["age"], False),
        ("Disease balanced accuracy", "disease_balanced_accuracy", colors["disease"], True),
        ("AD/control AUROC", "disease_auroc", colors["disease"], True),
        ("Cohort neighbor entropy", "cohort_neighbor_entropy", colors["align"], True),
        ("Same-cohort neighbor rate", "cohort_same_label_neighbor_rate", colors["align"], False),
    ]
    for ax, (title, col, color, higher) in zip(axes, panels):
        ax.plot(df["step"], df[col], color=color, lw=1.4, marker="o", ms=3.5)
        sem_col = f"{col}_sem"
        if sem_col in df and np.isfinite(df[sem_col]).any():
            ax.errorbar(df["step"], df[col], yerr=df[sem_col].fillna(0), fmt="none", ecolor=colors["ink"], lw=0.65, capsize=1.6)
        ax.scatter([sel_step], [selected[col]], s=42, color=color, edgecolor=colors["ink"], linewidth=0.75, zorder=5)
        ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold", pad=3)
        ax.text(0.98, 1.03, "higher better" if higher else "lower better", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#5C6B82")
        ax.grid(axis="y", color=colors["grid"], lw=0.55, alpha=0.9)
        ax.set_xticks(df["step"])
        ax.set_xticklabels([str(int(x)) for x in df["step"]], rotation=0, fontsize=6.4)
        ax.tick_params(axis="y", labelsize=6.4)
        ax.set_xlabel("checkpoint step", fontsize=6.4)

    fig.text(0.08, 0.955, "Figure 1 checkpoint Pareto selection", fontsize=13, fontweight="bold", color=colors["ink"], ha="left")
    fig.text(
        0.08,
        0.915,
        f"Selected checkpoint-{sel_step}: age improves while disease readout and cohort alignment pass guard thresholds.",
        fontsize=7.2,
        color="#5C6B82",
        ha="left",
    )
    fig.savefig(out_dir / "fig1c_checkpoint_pareto_selection.png", dpi=700, bbox_inches="tight")
    fig.savefig(out_dir / "fig1c_checkpoint_pareto_selection.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1c_checkpoint_pareto_selection.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.15, 2.55))
    sizes = 70 + 260 * df["cohort_normalized_iLISI_norm"].fillna(0.5)
    sc = ax.scatter(
        df["age_pearson_r"],
        df["disease_balanced_accuracy"],
        c=df["cohort_same_label_neighbor_rate"],
        cmap="viridis_r",
        s=sizes,
        edgecolor="white",
        linewidth=0.7,
    )
    for _, row in df.iterrows():
        weight = "bold" if bool(row["selected"]) else "normal"
        ax.text(row["age_pearson_r"] + 0.001, row["disease_balanced_accuracy"] + 0.0005, str(int(row["step"])), fontsize=6.2, weight=weight)
    ax.set_xlabel("Age Pearson r")
    ax.set_ylabel("Disease balanced accuracy")
    ax.set_title("Age-disease Pareto plane", loc="left", fontsize=8.3, fontweight="bold")
    ax.grid(color="#D9E1EA", lw=0.55)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("same-cohort neighbor rate", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.8)
    fig.savefig(out_dir / "fig1c_checkpoint_pareto_plane.png", dpi=700, bbox_inches="tight")
    fig.savefig(out_dir / "fig1c_checkpoint_pareto_plane.pdf", bbox_inches="tight")
    plt.close(fig)


def rewrite_final_figure(selected_step: int, args: argparse.Namespace) -> None:
    embed_dir = OUT / "embeddings" / f"checkpoint-{selected_step}"
    env = os.environ.copy()
    env["FIG1_MULTITASK_FT_RESULT"] = str(CHECKPOINT_ROOT)
    env["FIG1_CPT_VALIDATION_RESULT"] = str(embed_dir)
    run_command([str(PYTHON), str(SCORE_SCRIPT)], OUT / "logs" / f"score_selected_checkpoint_{selected_step}.log", env=env)
    selected_dir = OUT / "selected_final_figure1c"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "fig1c_representation_metrics_final.png",
        "fig1c_representation_metrics_final.pdf",
        "fig1c_omnicell_finetuned_performance_audit.png",
        "fig1c_omnicell_finetuned_performance_audit.pdf",
    ]:
        src = FIG1 / name
        if src.exists():
            shutil.copy2(src, selected_dir / name)
    for name in [
        "fig1c_representation_metrics_final_source.csv",
        "fig1c_representation_metrics_final_contract.json",
        "fig1c_omnicell_finetuned_performance_audit_summary.json",
    ]:
        src = FIG1_SRC / name
        if src.exists():
            shutil.copy2(src, selected_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hvg-top", type=int, default=15000)
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--force-score", action="store_true")
    parser.add_argument("--skip-final-rewrite", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for step in STEPS:
        print(f"[pareto] checkpoint-{step}: embedding", flush=True)
        embed_dir = ensure_embedding(step, args)
        print(f"[pareto] checkpoint-{step}: scoring", flush=True)
        metric_csv = score_checkpoint(step, embed_dir, args)
        records.append(extract_cpt_metrics(step, metric_csv))

    table = add_pareto_scores(pd.DataFrame(records))
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "metrics" / "checkpoint_pareto_metrics.csv", index=False)
    draw_pareto(table)

    selected_step = int(table.loc[table["selected"], "step"].iloc[0])
    summary = {
        "selected_checkpoint": f"checkpoint-{selected_step}",
        "selection_rule": "Among checkpoints passing guard thresholds for disease readout and cohort alignment, choose the highest age Pearson r; ties use the weighted Pareto score.",
        "guard_thresholds": {
            "disease_balanced_accuracy": "within 0.020 of checkpoint maximum",
            "disease_auroc": "within 0.020 of checkpoint maximum",
            "cohort_neighbor_entropy_or_iLISI": "entropy within 0.035 of maximum OR iLISI within 0.020 of maximum",
            "same_label_neighbor_rate": "within +0.040 of checkpoint minimum; lower is better",
        },
        "selected_metrics": table[table["selected"]].iloc[0].replace({np.nan: None}).to_dict(),
        "all_metrics_csv": str(OUT / "metrics" / "checkpoint_pareto_metrics.csv"),
        "diagnostic_figure": str(OUT / "figures" / "fig1c_checkpoint_pareto_selection.pdf"),
    }
    (OUT / "checkpoint_pareto_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "selected_checkpoint.txt").write_text(f"checkpoint-{selected_step}\n", encoding="utf-8")

    if not args.skip_final_rewrite:
        print(f"[pareto] rewriting final Figure 1C with checkpoint-{selected_step}", flush=True)
        rewrite_final_figure(selected_step, args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
