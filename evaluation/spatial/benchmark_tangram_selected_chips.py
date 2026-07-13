#!/usr/bin/env python
"""Tangram selected-chip spatial deconvolution benchmark for formal Figure 2."""

from __future__ import annotations
import os

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg
import torch
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)

PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
SC_H5AD = PROJECT / "results/cortex_t1001_task_inputs/cortex_sc_subset.h5ad"
SELECTED = PROJECT / "results/cortex_spatial_chip_screen/selected_10_cortex_spatial_chips.csv"
OUT = PROJECT / "results/figure2_formal_tangram_selected10"
FIG = PROJECT / "figures/figure2_formal_tangram_selected10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chips", default="")
    p.add_argument("--n-genes", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-spots", type=int, default=0, help="0 means use all spots")
    p.add_argument("--seed", type=int, default=20260529)
    return p.parse_args()


def broad(label: str) -> str:
    s = str(label)
    if "Oligodendrocyte precursor" in s:
        return "OPC"
    if "Oligodendrocyte" in s:
        return "Oligodendrocyte"
    if "Astro" in s:
        return "Astrocyte"
    if any(k in s for k in ["Microglia", "Macrophage", "Monocyte", "T cell"]):
        return "Microglia/immune"
    if any(k in s for k in ["Endothelial", "Pericyte", "Vascular", "VLMC", "SMC", "Mural"]):
        return "Vascular"
    if any(k in s for k in ["GABA", "RELN", "VIP", "PVALB", "SST", "LAMP5"]):
        return "Inhibitory neuron"
    if "neuron" in s or "IT" in s or "CT" in s or "ET" in s or "NP" in s:
        return "Excitatory neuron"
    return "Other"


def selected_chips(args: argparse.Namespace) -> list[str]:
    if args.chips.strip():
        return [x.strip() for x in args.chips.split(",") if x.strip()]
    return pd.read_csv(SELECTED)["chip"].astype(str).tolist()


def ensure_spatial_h5ad(chip: str) -> Path:
    import sys

    script_dir = PROJECT / "scripts"
    sys.path.insert(0, str(script_dir))
    from benchmark_random_stereo_hvg_scan import spatial_h5ad_for_chip

    return spatial_h5ad_for_chip(chip, force=False)


def sparse_mean_var(x: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x.mean(axis=0)).ravel()
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    return mean, np.maximum(mean_sq - mean * mean, 0)


def choose_genes(adata_sc: ad.AnnData, adata_sp: ad.AnnData, n_genes: int) -> list[str]:
    common = pd.Index(adata_sc.var_names).intersection(pd.Index(adata_sp.var_names))
    sc_small = adata_sc[:, common]
    sp_small = adata_sp[:, common]
    x_sc = sc_small.X.tocsr() if sparse.issparse(sc_small.X) else sparse.csr_matrix(sc_small.X)
    x_sp = sp_small.X.tocsr() if sparse.issparse(sp_small.X) else sparse.csr_matrix(sp_small.X)
    _, var_sc = sparse_mean_var(x_sc)
    _, var_sp = sparse_mean_var(x_sp)
    score = var_sc + var_sp
    valid = np.where(score > 0)[0]
    order = valid[np.argsort(score[valid])[::-1]]
    return common[order[: min(n_genes, len(order))]].astype(str).tolist()


def metric_rows(chip: str, true: np.ndarray, pred: np.ndarray, n_spots: int, n_genes: int, elapsed: float) -> list[dict]:
    rows = []
    labels = sorted(set(true) | set(pred))
    values = {
        "Accuracy": accuracy_score(true, pred),
        "Balanced accuracy": balanced_accuracy_score(true, pred),
        "Macro F1": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
        "ARI": adjusted_rand_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
    }
    for metric, value in values.items():
        rows.append(
            {
                "task": "selected10 spatial deconvolution",
                "method": "Tangram",
                "chip": chip,
                "metric": metric,
                "value": float(value),
                "n_spots": int(n_spots),
                "n_genes": int(n_genes),
                "elapsed_sec": round(elapsed, 2),
            }
        )
    return rows


def run_chip(chip: str, adata_sc_full: ad.AnnData, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed + sum(ord(c) for c in chip))
    sp_path = ensure_spatial_h5ad(chip)
    adata_sp_full = sc.read_h5ad(sp_path)
    if args.max_spots and adata_sp_full.n_obs > args.max_spots:
        idx = rng.choice(np.arange(adata_sp_full.n_obs), size=args.max_spots, replace=False)
        adata_sp_full = adata_sp_full[idx].copy()

    start = time.time()
    genes = choose_genes(adata_sc_full, adata_sp_full, args.n_genes)
    adata_sc = adata_sc_full[:, genes].copy()
    adata_sp = adata_sp_full[:, genes].copy()
    tg.pp_adatas(adata_sc, adata_sp, genes=genes, gene_to_lowercase=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ad_map = tg.map_cells_to_space(
        adata_sc,
        adata_sp,
        mode="clusters",
        cluster_label="cell_type",
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        random_state=args.seed,
        verbose=True,
    )
    tg.project_cell_annotations(ad_map, adata_sp, annotation="cell_type")
    pred_scores = pd.DataFrame(adata_sp.obsm["tangram_ct_pred"], index=adata_sp.obs_names)
    pred_scores.columns = [str(c) for c in pred_scores.columns]
    pred_label = pred_scores.idxmax(axis=1).astype(str).to_numpy()
    true_label = adata_sp.obs["cell_type"].astype(str).to_numpy()
    true_broad = np.array([broad(x) for x in true_label])
    pred_broad = np.array([broad(x) for x in pred_label])
    elapsed = time.time() - start
    metrics = pd.DataFrame(metric_rows(chip, true_broad, pred_broad, adata_sp.n_obs, len(genes), elapsed))
    pred = pd.DataFrame(
        {
            "chip": chip,
            "spot": adata_sp.obs_names.astype(str),
            "x": np.asarray(adata_sp.obsm["spatial"])[:, 0],
            "y": np.asarray(adata_sp.obsm["spatial"])[:, 1],
            "ground_truth_celltype": true_label,
            "predicted_celltype": pred_label,
            "ground_truth_broad": true_broad,
            "predicted_broad": pred_broad,
        }
    )
    chip_dir = OUT / "per_chip"
    chip_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(chip_dir / f"{chip}_tangram_predictions.csv", index=False)
    pred_scores.to_csv(chip_dir / f"{chip}_tangram_scores.csv")
    (chip_dir / f"{chip}_tangram_summary.json").write_text(
        json.dumps(
            {
                "chip": chip,
                "input_spatial_h5ad": str(sp_path),
                "n_spots": int(adata_sp.n_obs),
                "n_genes": int(len(genes)),
                "epochs": int(args.epochs),
                "device": device,
                "elapsed_sec": round(elapsed, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, pred


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    chips = selected_chips(args)
    adata_sc = sc.read_h5ad(SC_H5AD)
    all_metrics = []
    combined_preds = []
    for chip in chips:
        print(f"[Tangram] start {chip}", flush=True)
        metrics, pred = run_chip(chip, adata_sc, args)
        all_metrics.append(metrics)
        combined_preds.append(pred)
        pd.concat(all_metrics, ignore_index=True).to_csv(OUT / "tangram_selected10_metrics.csv", index=False)
        pd.concat(combined_preds, ignore_index=True).to_csv(OUT / "tangram_selected10_predictions.csv", index=False)
        print(metrics.to_string(index=False), flush=True)
    contract = {
        "chips": chips,
        "n_genes": args.n_genes,
        "epochs": args.epochs,
        "metrics": str(OUT / "tangram_selected10_metrics.csv"),
        "predictions": str(OUT / "tangram_selected10_predictions.csv"),
    }
    (OUT / "tangram_selected10_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
