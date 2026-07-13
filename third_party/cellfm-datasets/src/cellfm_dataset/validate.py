"""Validation utilities for memmap dataset artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (
    align_matrix_to_vocab,
    apply_expression_transform,
    ensure_csr_matrix,
    extract_coords,
)
from ._optional import require_dependency
from .memmap import MemmapDataset


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_dataset(
    dataset_dir: str | Path,
    *,
    skip_source_compare: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, int]:
    """Validate one memmap dataset root."""

    dataset = MemmapDataset(dataset_dir)
    vocab_to_index = {gene: idx for idx, gene in enumerate(dataset.gene_vocab)}
    n_checked = 0

    for sample_id, sample in dataset.samples.items():
        meta = sample.metadata
        _assert(sample.indptr.shape == (meta.n_cells + 1,), f"{sample_id}: indptr shape mismatch.")
        _assert(sample.indices.shape == (meta.nnz,), f"{sample_id}: indices shape mismatch.")
        _assert(sample.values.shape == (meta.nnz,), f"{sample_id}: values shape mismatch.")
        _assert(sample.coords.shape[0] == meta.n_cells, f"{sample_id}: coords row mismatch.")
        _assert(sample.raw_cell_sums.shape == (meta.n_cells,), f"{sample_id}: raw_cell_sums shape mismatch.")
        _assert(sample.obs_names.shape == (meta.n_cells,), f"{sample_id}: obs_names shape mismatch.")
        _assert(int(sample.indptr[0]) == 0, f"{sample_id}: indptr must start at 0.")
        _assert(int(sample.indptr[-1]) == meta.nnz, f"{sample_id}: indptr last value must equal nnz.")
        _assert(np.all(np.diff(sample.indptr) >= 0), f"{sample_id}: indptr must be non-decreasing.")
        _assert(np.all(sample.indices >= 0), f"{sample_id}: negative gene indices found.")
        _assert(np.all(sample.indices < meta.n_genes), f"{sample_id}: gene index out of range.")
        _assert(np.all(np.isfinite(sample.values)), f"{sample_id}: non-finite expression values found.")
        if meta.expression_transform == "normalize_total_log1p":
            _assert(np.all(sample.values >= 0), f"{sample_id}: log1p-normalized values should be >= 0.")
        _assert(
            np.array_equal(np.unique(sample.indices), sample.present_gene_ids),
            f"{sample_id}: present_gene_ids mismatch.",
        )
        if meta.obs_columns:
            _assert(sample.has_obs_table, f"{sample_id}: obs_columns declared but obs.parquet missing.")
            obs_frame = sample.load_obs_frame()
            _assert(len(obs_frame) == meta.n_cells, f"{sample_id}: obs.parquet row count mismatch.")
            _assert(
                list(obs_frame.columns) == ["obs_name", *meta.obs_columns],
                f"{sample_id}: obs.parquet columns mismatch.",
            )

        if not skip_source_compare:
            anndata = require_dependency("anndata", "pip install 'cellfm-datasets[convert]'")
            adata = anndata.read_h5ad(meta.source_h5ad)
            source_coords = extract_coords(adata, z_column=meta.z_column)
            source_expr = ensure_csr_matrix(adata.X)
            aligned, _ = align_matrix_to_vocab(
                source_expr,
                adata.var_names,
                vocab_to_index=vocab_to_index,
                vocab_size=meta.n_genes,
            )
            transformed, source_raw_cell_sums = apply_expression_transform(
                aligned,
                expression_transform=meta.expression_transform,
                normalize_target_sum=meta.normalize_target_sum,
            )
            source_coord_finite = np.isfinite(source_coords)
            stored_coord_finite = np.isfinite(sample.coords)

            _assert(source_coords.shape == sample.coords.shape, f"{sample_id}: coord shape mismatch.")
            _assert(
                np.array_equal(source_coord_finite, stored_coord_finite),
                f"{sample_id}: non-finite coord pattern mismatch.",
            )
            _assert(
                np.allclose(
                    source_coords[source_coord_finite],
                    sample.coords[stored_coord_finite],
                    rtol=rtol,
                    atol=atol,
                ),
                f"{sample_id}: finite coord values mismatch.",
            )
            _assert(
                np.allclose(source_raw_cell_sums, sample.raw_cell_sums, rtol=rtol, atol=atol),
                f"{sample_id}: raw cell sums mismatch.",
            )
            _assert(
                np.array_equal(transformed.indptr.astype(np.int64, copy=False), np.asarray(sample.indptr)),
                f"{sample_id}: indptr mismatch against source.",
            )
            _assert(
                np.array_equal(transformed.indices.astype(np.int32, copy=False), np.asarray(sample.indices)),
                f"{sample_id}: indices mismatch against source.",
            )
            _assert(
                np.allclose(transformed.data.astype(np.float32, copy=False), np.asarray(sample.values), rtol=rtol, atol=atol),
                f"{sample_id}: values mismatch against source.",
            )
        n_checked += 1
    return {"n_samples": n_checked, "n_genes": dataset.metadata.n_genes}


def validate_region_manifest_examples(
    dataset_dir: str | Path,
    region_manifest_path: str | Path,
    *,
    max_regions_to_check: int = 8,
) -> dict[str, int]:
    """Validate that a region manifest can be reconstructed from the memmap store."""

    dataset = MemmapDataset(dataset_dir)
    checked = 0
    for item in dataset.iter_regions(region_manifest_path, limit=max_regions_to_check):
        n_cells = int(item["coords"].shape[0])
        nnz = int(item["gene_indices"].shape[0])
        _assert(int(item["cell_ptr"][0]) == 0, "cell_ptr must start at 0.")
        _assert(int(item["cell_ptr"][-1]) == nnz, "final cell_ptr must equal nnz.")
        _assert(item["raw_coords"].shape[0] == n_cells, "raw_coords length mismatch.")
        _assert(item["gene_values"].shape[0] == item["gene_indices"].shape[0], "gene value/index mismatch.")
        _assert(item["source_cell_indices"].shape[0] == n_cells, "source cell count mismatch.")
        checked += 1
    return {"n_regions": checked}
