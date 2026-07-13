#!/usr/bin/env python
"""Build native OmniCell embeddings for Figure 1 validation cells.

The native OmniCell baseline must be computed on the exact same 50,000
validation cells used for Figure 1 representation probes. This script creates
a sparse validation h5ad from the memmap dataset, then runs the original
OmniCell CellFormer model on that h5ad.
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
import torch
from scipy import sparse


WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
PROJECT = WORK_ROOT / "projects" / "nvu_vascular"
DEFAULT_VALIDATION = PROJECT / "results" / "atlas_validation_full_ridge" / "validation_cells.csv"
DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_OUT = PROJECT / "results" / "figure1_validation_native_omnicell"
DEFAULT_OMNICELL = Path(os.path.expandvars("${OMNICELL_LEGACY_ROOT}"))

sys.path.insert(0, str(WORK_ROOT / "cellfm-datasets" / "src"))
from cellfm_dataset.memmap import MemmapDataset  # noqa: E402

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
for stream in [sys.stdout, sys.stderr]:
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-csv", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--omnicell-root", type=Path, default=DEFAULT_OMNICELL)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--force-h5ad", action="store_true")
    parser.add_argument("--force-embedding", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def sanitize_obs(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or str(out[col].dtype) == "category":
            out[col] = out[col].astype("object").where(pd.notna(out[col]), "").astype(str)
    return out


def build_sparse_h5ad(validation_csv: Path, dataset_root: Path, h5ad_path: Path) -> None:
    meta = pd.read_csv(validation_csv, low_memory=False)
    dataset = MemmapDataset(dataset_root)
    nnz = 0
    cached: list[tuple[np.ndarray, np.ndarray]] = []
    for i, row in enumerate(meta.itertuples(index=False), start=1):
        sample_id = str(getattr(row, "sample_id"))
        cell_index = int(getattr(row, "cell_index"))
        sample = dataset.samples[sample_id]
        start = int(sample.indptr[cell_index])
        end = int(sample.indptr[cell_index + 1])
        gene_ids = np.asarray(sample.indices[start:end], dtype=np.int64)
        values = np.asarray(sample.values[start:end], dtype=np.float32)
        good = np.isfinite(values) & (values > 0) & (gene_ids >= 0) & (gene_ids < len(dataset.gene_vocab))
        gene_ids = gene_ids[good]
        values = values[good]
        cached.append((gene_ids, values))
        nnz += len(gene_ids)
        if i % 5000 == 0:
            print(f"[h5ad] cached {i}/{len(meta)} cells; nnz={nnz}", flush=True)

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    offset = 0
    for i, (gene_ids, values) in enumerate(cached):
        end = offset + len(gene_ids)
        rows[offset:end] = i
        cols[offset:end] = gene_ids.astype(np.int32, copy=False)
        data[offset:end] = values
        offset = end

    x = sparse.csr_matrix((data, (rows, cols)), shape=(len(meta), len(dataset.gene_vocab)), dtype=np.float32)
    obs = sanitize_obs(meta)
    obs_names = obs["sample_id"].astype(str) + "|" + obs["cell_index"].astype(str) + "|" + np.arange(len(obs)).astype(str)
    obs.index = obs_names
    var = pd.DataFrame(index=pd.Index(dataset.gene_vocab, name="gene_id"))
    adata = ad.AnnData(X=x, obs=obs, var=var)
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(h5ad_path, compression="gzip")
    print(f"[h5ad] wrote {h5ad_path} shape={adata.shape}", flush=True)


def select_hvgs(adata: ad.AnnData, n_genes: int) -> list[str]:
    x = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(np.asarray(adata.X))
    n_obs = max(1, adata.n_obs)
    sums = np.asarray(x.sum(axis=0)).ravel()
    sum_sq = np.asarray(x.multiply(x).sum(axis=0)).ravel()
    mean = sums / n_obs
    var = sum_sq / n_obs - mean * mean
    detected = np.asarray((x > 0).sum(axis=0)).ravel()
    var[detected < 10] = -np.inf
    order = np.argsort(-var, kind="mergesort")
    order = order[np.isfinite(var[order])][: min(n_genes, np.isfinite(var).sum())]
    genes = adata.var_names[order].astype(str).tolist()
    print(f"[native] selected {len(genes)} HVGs", flush=True)
    return genes


def run_native_embedding(h5ad_path: Path, out_dir: Path, omnicell_root: Path, n_genes: int, batch_size: int, dtype: str) -> None:
    sys.path.insert(0, str(omnicell_root))
    from OmniCell.loader.CellEmbedding import CellFormer  # noqa: E402

    print(f"[native] reading {h5ad_path}", flush=True)
    adata = ad.read_h5ad(h5ad_path)
    selected = select_hvgs(adata, n_genes)
    checkpoint_dir = omnicell_root / "OmniCell" / "checkpoint"
    vocab_path = omnicell_root / "OmniCell" / "vocab" / "Vocabulary.json"
    former = CellFormer(
        checkpoint_dir=str(checkpoint_dir),
        dtype=dtype_from_name(dtype),
        batch_size=batch_size,
        vocab_path=str(vocab_path),
        n_genes=n_genes,
        mode="sc",
        threshold=0.9,
        selected_genes=selected,
    )
    emb, _ = former.infer(adata=adata)
    emb = np.asarray(emb, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embedding.npy", emb)
    adata.obs.reset_index(drop=True).to_parquet(out_dir / "embedding_meta.parquet", index=False)
    pd.DataFrame({"rank": np.arange(1, len(selected) + 1), "gene": selected}).to_csv(out_dir / "native_hvg_genes.csv", index=False)
    print(f"[native] wrote embedding {emb.shape}", flush=True)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = out_dir / "figure1_validation_cells_50k.h5ad"
    emb_path = out_dir / "embedding.npy"
    if args.force_h5ad or not h5ad_path.exists():
        build_sparse_h5ad(args.validation_csv.expanduser().resolve(), args.dataset_root.expanduser().resolve(), h5ad_path)
    else:
        print(f"[h5ad] reuse {h5ad_path}", flush=True)
    if args.force_embedding or not emb_path.exists():
        run_native_embedding(
            h5ad_path,
            out_dir,
            args.omnicell_root.expanduser().resolve(),
            args.n_genes,
            args.batch_size,
            args.dtype,
        )
    else:
        print(f"[native] reuse {emb_path}", flush=True)
    config = {
        "validation_csv": str(args.validation_csv),
        "dataset_root": str(args.dataset_root),
        "output_dir": str(out_dir),
        "h5ad": str(h5ad_path),
        "embedding": str(emb_path),
        "n_genes": args.n_genes,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
