#!/usr/bin/env python3
"""Write selected Figure 1 representation candidates as embedding directories."""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}/projects/nvu_vascular"))
RESULTS = PROJECT / "results"
VALIDATION = RESULTS / "atlas_validation_full_ridge"
CPT_DIR = RESULTS / "figure1_multitask_cpt_alignment_agebin_stage2_validation_embedding"
NATIVE_DIR = RESULTS / "figure1_validation_native_omnicell"
OUT_ROOT = RESULTS / "figure1_representation_candidate_optimization" / "candidate_embeddings"


def zscore(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x.astype(np.float32))


def pca_embed(x: np.ndarray, n_components: int, seed: int = 13) -> np.ndarray:
    n_components = min(n_components, x.shape[1], x.shape[0] - 1)
    xz = zscore(x)
    xp = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit_transform(xz)
    return StandardScaler().fit_transform(xp).astype(np.float32)


def concat_pca(parts: list[np.ndarray], n_components: int, seed: int = 13) -> np.ndarray:
    scaled = [zscore(p) for p in parts]
    return pca_embed(np.concatenate(scaled, axis=1), n_components=n_components, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=["cpt_pca128", "cpt_native_pca512", "cpt_rawsvd_pca256_support"])
    args = parser.parse_args()

    cpt = np.load(CPT_DIR / "embedding.npy").astype(np.float32)
    if args.candidate == "cpt_pca128":
        emb = pca_embed(cpt, 128)
        description = "Strict OmniCell-CPT view: z-scored CPT embedding followed by PCA128 and z-score scaling."
    elif args.candidate == "cpt_native_pca512":
        native = np.load(NATIVE_DIR / "embedding.npy").astype(np.float32)
        emb = concat_pca([cpt, native], 512)
        description = "OmniCell ensemble view: fine-tuned CPT plus native OmniCell embedding, PCA512."
    else:
        raw = np.load(VALIDATION / "raw_svd_features.npy").astype(np.float32)
        emb = concat_pca([cpt, raw], 256)
        description = "Support view: fine-tuned CPT plus raw-expression SVD support, PCA256. Use only if explicitly labelled."

    out = OUT_ROOT / args.candidate
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embedding.npy", emb.astype(np.float32))
    meta_src = CPT_DIR / "embedding_meta.csv"
    if meta_src.exists():
        pd.read_csv(meta_src, low_memory=False).to_csv(out / "embedding_meta.csv", index=False)
    config = {
        "candidate": args.candidate,
        "description": description,
        "input_cpt_embedding": str(CPT_DIR / "embedding.npy"),
        "validation_csv": str(VALIDATION / "validation_cells.csv"),
        "shape": list(emb.shape),
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
