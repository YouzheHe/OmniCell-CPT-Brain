#!/usr/bin/env python
"""Generate and evaluate native OmniCell embeddings on best-five Cortex chips.

This script fills the missing fair-comparison baseline for Figure 2 spatial
deconvolution. It keeps the same single-cell reference, calibration split, and
held-out spot intersection used by the current best-five benchmark.
"""

from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
RESULTS = PROJECT / "results"
SP_BASE = RESULTS / "random_stereo_hvg_scan" / "spatial_h5ad"
SC_H5AD = RESULTS / "cortex_t906_task_inputs" / "cortex_sc_subset.h5ad"
OMNI_PRED = RESULTS / "random_stereo_hvg_scan" / "random_stereo_hvg_scan_predictions.csv"
SELECTED_PRED = PROJECT / "figures" / "figure2_final_panels" / "source_data" / "fig2_selected_quality_all_methods_predictions.csv"
T906_PRED = PROJECT / "figures" / "figure2_final_panels" / "source_data" / "fig2_t906_available_method_matched_predictions.csv"
OUT = RESULTS / "best5_native_omnicell_embeddings"
SRC = PROJECT / "figures" / "figure2_nonzero_hvg_final_v2" / "source_data"
DEFAULT_OMNICELL = Path(os.path.expandvars("${OMNICELL_LEGACY_ROOT}"))
CHIPS = ["T917", "T991", "T989", "T988", "T906"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips", default=",".join(CHIPS))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--source-dir", type=Path, default=SRC)
    parser.add_argument("--omnicell-root", type=Path, default=DEFAULT_OMNICELL)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--force-embedding", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def hvg_genes(adata: sc.AnnData, n_genes: int) -> list[str]:
    tmp = adata.copy()
    flavor = "seurat_v3" if tmp.X.max() > 10 else "seurat"
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_genes, flavor=flavor)
    selected = tmp.var_names[tmp.var["highly_variable"]].astype(str).tolist()
    return selected or tmp.var_names.astype(str).tolist()[:n_genes]


def infer_one(path: Path, former_cls, checkpoint_dir: Path, vocab_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    print(f"[native] reading {path}", flush=True)
    adata = sc.read_h5ad(path)
    selected = hvg_genes(adata, args.n_genes)
    print(f"[native] {path.name}: {adata.n_obs} obs, {len(selected)} HVGs", flush=True)
    former = former_cls(
        checkpoint_dir=str(checkpoint_dir),
        dtype=dtype_from_name(args.dtype),
        batch_size=args.batch_size,
        vocab_path=str(vocab_path),
        n_genes=args.n_genes,
        mode="sc",
        threshold=0.9,
        selected_genes=selected,
    )
    emb, _ = former.infer(adata=adata)
    emb = np.asarray(emb, dtype=np.float32)
    meta = adata.obs.copy()
    if "obs_name" not in meta.columns:
        meta.insert(0, "obs_name", adata.obs_names.astype(str))
    meta = meta.reset_index(drop=True)
    return meta, emb


def sc_reference_meta() -> pd.DataFrame:
    adata = ad.read_h5ad(SC_H5AD, backed="r")
    meta = adata.obs.copy()
    if "obs_name" not in meta.columns:
        meta.insert(0, "obs_name", adata.obs_names.astype(str))
    adata.file.close()
    meta = meta.reset_index(drop=True)
    meta["native_input_file"] = SC_H5AD.name
    return meta


def load_or_create_sc_cache(args: argparse.Namespace, former_cls) -> tuple[pd.DataFrame, np.ndarray]:
    cache_dir = args.output_dir / "_sc_reference_native"
    emb_path = cache_dir / "embedding.npy"
    meta_path = cache_dir / "embedding_meta.parquet"
    if emb_path.exists() and meta_path.exists() and not args.force_embedding:
        return pd.read_parquet(meta_path), np.asarray(np.load(emb_path, mmap_mode="r"), dtype=np.float32)

    sc_meta = sc_reference_meta()
    sc_n = len(sc_meta)
    for chip in CHIPS:
        candidate = args.output_dir / chip / "embedding.npy"
        if candidate.exists():
            arr = np.load(candidate, mmap_mode="r")
            if arr.shape[0] > sc_n:
                print(f"[native] reusing single-cell cache from {candidate}", flush=True)
                sc_emb = np.asarray(arr[:sc_n], dtype=np.float32)
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(emb_path, sc_emb)
                sc_meta.to_parquet(meta_path, index=False)
                return sc_meta, sc_emb

    omnicell_root = args.omnicell_root.expanduser().resolve()
    checkpoint_dir = omnicell_root / "OmniCell" / "checkpoint"
    vocab_path = omnicell_root / "OmniCell" / "vocab" / "Vocabulary.json"
    sc_meta, sc_emb = infer_one(SC_H5AD, former_cls, checkpoint_dir, vocab_path, args)
    sc_meta["native_input_file"] = SC_H5AD.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, sc_emb.astype(np.float32))
    for col in sc_meta.columns:
        if str(sc_meta[col].dtype) in {"object", "category"}:
            sc_meta[col] = sc_meta[col].astype("object").where(pd.notna(sc_meta[col]), "").astype(str)
    sc_meta.to_parquet(meta_path, index=False)
    return sc_meta, sc_emb.astype(np.float32)


def generate_chip_embedding(chip: str, args: argparse.Namespace, former_cls, sc_cache: tuple[pd.DataFrame, np.ndarray]) -> Path:
    chip_out = args.output_dir / chip
    emb_path = chip_out / "embedding.npy"
    meta_path = chip_out / "embedding_meta.parquet"
    if emb_path.exists() and meta_path.exists() and not args.force_embedding:
        print(f"[native] {chip}: embedding exists", flush=True)
        return emb_path
    chip_out.mkdir(parents=True, exist_ok=True)
    omnicell_root = args.omnicell_root.expanduser().resolve()
    checkpoint_dir = omnicell_root / "OmniCell" / "checkpoint"
    vocab_path = omnicell_root / "OmniCell" / "vocab" / "Vocabulary.json"
    sc_meta, sc_emb = sc_cache
    sp_meta, sp_emb = infer_one(SP_BASE / f"{chip}.h5ad", former_cls, checkpoint_dir, vocab_path, args)
    sp_meta["native_input_file"] = f"{chip}.h5ad"
    metas = [sc_meta.copy(), sp_meta]
    embs = [sc_emb, sp_emb]
    meta_all = pd.concat(metas, ignore_index=True)
    emb_all = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(emb_path, emb_all)
    for col in meta_all.columns:
        if str(meta_all[col].dtype) in {"object", "category"}:
            meta_all[col] = meta_all[col].astype("object").where(pd.notna(meta_all[col]), "").astype(str)
    meta_all.to_parquet(meta_path, index=False)
    config = {
        "chip": chip,
        "sc_h5ad": str(SC_H5AD),
        "spatial_h5ad": str(SP_BASE / f"{chip}.h5ad"),
        "embedding": str(emb_path),
        "n_rows": int(emb_all.shape[0]),
        "embedding_dim": int(emb_all.shape[1]),
        "n_genes": args.n_genes,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
    }
    (chip_out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    return emb_path


def broad(label: str) -> str:
    s = str(label)
    low = s.lower()
    if "oligodendrocyte precursor" in low or "opc" in low:
        return "OPC"
    if "oligodendrocyte" in low or low == "oligo":
        return "Oligodendrocyte"
    if "astro" in low:
        return "Astrocyte"
    if any(k in s for k in ["Microglia", "Macrophage", "Monocyte", "T cell"]) or any(k in low for k in ["micro", "immune"]):
        return "Microglia/immune"
    if any(k in s for k in ["Endothelial", "Pericyte", "Vascular", "VLMC", "SMC", "Mural"]) or any(k in low for k in ["endo", "peri", "vascular", "mural"]):
        return "Vascular"
    if any(k in s for k in ["GABA", "RELN", "VIP", "PVALB", "SST", "LAMP5", "Inhibitory"]):
        return "Inhibitory neuron"
    if "neuron" in low or any(k in s for k in ["IT", "CT", "ET", "NP"]):
        return "Excitatory neuron"
    return "Other"


def metric_rows(method: str, chip: str, label_space: str, truth: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    labels = sorted(set(map(str, truth)).union(set(map(str, pred))))
    return [
        {"label_space": label_space, "method": method, "chip": chip, "metric": "Accuracy", "value": float(accuracy_score(truth, pred)), "n_obs": int(len(truth))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "Balanced accuracy", "value": float(balanced_accuracy_score(truth, pred)), "n_obs": int(len(truth))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "Macro F1", "value": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)), "n_obs": int(len(truth))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "ARI", "value": float(adjusted_rand_score(truth, pred)), "n_obs": int(len(truth))},
        {"label_space": label_space, "method": method, "chip": chip, "metric": "NMI", "value": float(normalized_mutual_info_score(truth, pred)), "n_obs": int(len(truth))},
    ]


def train_embedding_head(features: np.ndarray, sc_labels: np.ndarray, sp_labels: np.ndarray, calibration_mask: np.ndarray, seed: int) -> np.ndarray:
    enc = LabelEncoder().fit(np.r_[sc_labels, sp_labels])
    y_sc = enc.transform(sc_labels)
    y_sp = enc.transform(sp_labels)
    sc_n = len(sc_labels)
    sp_n = len(sp_labels)
    train_idx = np.r_[np.arange(sc_n), sc_n + np.flatnonzero(calibration_mask)]
    y = np.r_[y_sc, y_sp]
    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-4,
            l1_ratio=0.02,
            class_weight="balanced",
            max_iter=220,
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=7,
            average=True,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    clf.fit(features[train_idx], y[train_idx])
    pred = clf.predict(features[sc_n : sc_n + sp_n])
    return enc.inverse_transform(pred)


def load_sc_labels() -> np.ndarray:
    adata = ad.read_h5ad(SC_H5AD, backed="r")
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    adata.file.close()
    return labels


def load_spatial_obs(chip: str) -> pd.DataFrame:
    adata = ad.read_h5ad(SP_BASE / f"{chip}.h5ad", backed="r")
    obs = adata.obs[["x", "y", "cell_type"]].copy()
    obs["spot_index"] = np.arange(adata.n_obs)
    obs["chip"] = chip
    obs["x_key"] = obs["x"].round().astype(int)
    obs["y_key"] = obs["y"].round().astype(int)
    adata.file.close()
    return obs


def evaluate_native(chips: list[str], args: argparse.Namespace) -> None:
    args.source_dir.mkdir(parents=True, exist_ok=True)
    sc_labels = load_sc_labels()
    sc_n = len(sc_labels)
    omni = pd.read_csv(OMNI_PRED)
    selected = pd.read_csv(SELECTED_PRED)
    t906_existing = pd.read_csv(T906_PRED) if T906_PRED.exists() else None
    all_pred: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for chip in chips:
        obs = load_spatial_obs(chip)
        sp_n = len(obs)
        emb_path = args.output_dir / chip / "embedding.npy"
        if chip == "T906" and not emb_path.exists():
            emb_path = RESULTS / "cortex_t906_native_omnicell_embeddings" / "embedding.npy"
        features = np.asarray(np.load(emb_path, mmap_mode="r"), dtype=np.float32)
        if features.shape[0] != sc_n + sp_n:
            raise RuntimeError(f"{chip}: expected {sc_n + sp_n} native rows, got {features.shape[0]}")
        cpt = omni[omni["chip"].astype(str).eq(chip)].copy()
        cpt["x_key"] = cpt["x"].round().astype(int)
        cpt["y_key"] = cpt["y"].round().astype(int)
        cpt = cpt.rename(columns={"ground_truth_celltype": "truth_omni"})
        base = obs.merge(cpt[["x_key", "y_key", "split", "truth_omni"]], on=["x_key", "y_key"], how="left", validate="one_to_one")
        if base["split"].isna().any():
            raise RuntimeError(f"{chip}: failed to align {int(base['split'].isna().sum())} split rows")
        pred_all = train_embedding_head(
            features,
            sc_labels=sc_labels,
            sp_labels=base["truth_omni"].astype(str).to_numpy(),
            calibration_mask=base["split"].astype(str).eq("calibration").to_numpy(),
            seed=20260624 + sum(ord(x) for x in chip),
        )
        mapper = pd.DataFrame({"spot_index": np.arange(sp_n), "pred_OmniCell native": pred_all})
        if chip == "T906" and t906_existing is not None:
            hold = t906_existing[t906_existing.get("chip", "T906").astype(str).eq("T906")].copy() if "chip" in t906_existing.columns else t906_existing.copy()
            if "spot_index" not in hold.columns:
                hold["x_key"] = hold["x"].round().astype(int)
                hold["y_key"] = hold["y"].round().astype(int)
                hold = obs[["spot_index", "x_key", "y_key"]].merge(hold, on=["x_key", "y_key"], how="inner")
            if "x" not in hold.columns and "x_y" in hold.columns:
                hold["x"] = hold["x_y"]
            if "y" not in hold.columns and "y_y" in hold.columns:
                hold["y"] = hold["y_y"]
            keep = hold[["spot_index", "x", "y", "truth_celltype"]].merge(mapper, on="spot_index", how="left")
            keep["chip"] = chip
        else:
            keep = selected[selected["chip"].astype(str).eq(chip)][["chip", "spot_index", "x", "y", "truth_celltype"]].copy()
            keep = keep.merge(mapper, on="spot_index", how="left", validate="many_to_one")
        keep = keep.dropna(subset=["pred_OmniCell native"]).copy()
        keep["truth_broad"] = keep["truth_celltype"].astype(str).map(broad)
        keep["pred_broad_OmniCell native"] = keep["pred_OmniCell native"].astype(str).map(broad)
        all_pred.append(keep)
        rows.extend(metric_rows("OmniCell native", chip, "fine cell type", keep["truth_celltype"].astype(str).to_numpy(), keep["pred_OmniCell native"].astype(str).to_numpy()))
        rows.extend(metric_rows("OmniCell native", chip, "broad cell class", keep["truth_broad"].astype(str).to_numpy(), keep["pred_broad_OmniCell native"].astype(str).to_numpy()))
        print(f"[native-eval] {chip}: n={len(keep)}", flush=True)
    pred_df = pd.concat(all_pred, ignore_index=True)
    metrics = pd.DataFrame(rows)
    pred_df.to_csv(args.source_dir / "fig2_best5_native_omnicell_predictions.csv", index=False)
    metrics.to_csv(args.source_dir / "fig2_best5_native_omnicell_metrics_by_chip.csv", index=False)
    summary = (
        metrics.groupby(["label_space", "method", "metric"], as_index=False)
        .agg(mean=("value", "mean"), sem=("value", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0), n_chips=("chip", "nunique"), n_obs_total=("n_obs", "sum"))
        .sort_values(["label_space", "method", "metric"])
    )
    summary.to_csv(args.source_dir / "fig2_best5_native_omnicell_metrics_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    chips = [x.strip() for x in args.chips.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_embedding:
        omnicell_root = args.omnicell_root.expanduser().resolve()
        sys.path.insert(0, str(omnicell_root))
        from OmniCell.loader.CellEmbedding import CellFormer

        sc_cache = load_or_create_sc_cache(args, CellFormer)
        for chip in chips:
            if chip == "T906" and not (args.output_dir / chip / "embedding.npy").exists() and (RESULTS / "cortex_t906_native_omnicell_embeddings" / "embedding.npy").exists():
                print("[native] T906: using existing native embedding from cortex_t906_native_omnicell_embeddings", flush=True)
                continue
            generate_chip_embedding(chip, args, CellFormer, sc_cache)
    if args.embedding_only:
        print(json.dumps({"status": "embedding_only_done", "chips": chips, "embedding_dir": str(args.output_dir)}, indent=2), flush=True)
        return
    evaluate_native(chips, args)
    run_config = {
        "chips": chips,
        "embedding_dir": str(args.output_dir),
        "source_dir": str(args.source_dir),
        "sc_h5ad": str(SC_H5AD),
        "spatial_h5ad_base": str(SP_BASE),
        "note": "Native OmniCell CellFormer evaluated on the same best-five calibration/held-out benchmark as Figure 2 spatial deconvolution.",
    }
    (args.source_dir / "fig2_best5_native_omnicell_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps(run_config, indent=2), flush=True)


if __name__ == "__main__":
    main()
