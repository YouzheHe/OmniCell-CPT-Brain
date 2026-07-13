"""Shared helpers for memmap conversion and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse


DEFAULT_EXPRESSION_TRANSFORM = "normalize_total_log1p"
SUPPORTED_EXPRESSION_TRANSFORMS = frozenset({"normalize_total_log1p", "none"})


def default_coord_axis_names(coord_dim: int) -> tuple[str, ...]:
    """Return canonical axis names for a coordinate tensor."""

    if coord_dim < 0:
        raise ValueError(f"coord_dim must be non-negative, got {coord_dim}.")
    if coord_dim == 0:
        return tuple()

    canonical = ("x", "y", "z")
    if coord_dim <= len(canonical):
        return canonical[:coord_dim]
    extras = tuple(f"axis_{idx}" for idx in range(len(canonical), coord_dim))
    return canonical + extras


def normalize_coord_axis_names(value: Any, coord_dim: int) -> tuple[str, ...]:
    """Normalize coordinate-axis metadata to a validated tuple."""

    if value is None:
        return default_coord_axis_names(coord_dim)

    if isinstance(value, np.ndarray):
        parsed = value.tolist()
    elif isinstance(value, (list, tuple)):
        parsed = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("coord_axis_names cannot be empty.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split(",")]
    else:
        raise TypeError(f"Unsupported coord_axis_names type: {type(value)!r}")

    if isinstance(parsed, str):
        parsed = [parsed]

    axis_names = tuple(str(part).strip() for part in parsed)
    if coord_dim == 0 and len(axis_names) == 0:
        return tuple()
    if len(axis_names) != coord_dim:
        raise ValueError(
            f"coord_axis_names length must match coord_dim={coord_dim}, got {axis_names!r}."
        )
    if any(not axis_name for axis_name in axis_names):
        raise ValueError(f"coord_axis_names cannot contain empty entries: {axis_names!r}.")
    return axis_names


def normalize_column_names(value: Any) -> tuple[str, ...] | None:
    """Normalize an optional column selection payload."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        parsed = value.tolist()
    elif isinstance(value, (list, tuple)):
        parsed = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        if text == "*":
            return ("*",)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split(",")]
    else:
        raise TypeError(f"Unsupported column-name payload: {type(value)!r}")

    if isinstance(parsed, str):
        parsed = [parsed]

    normalized = tuple(str(item).strip() for item in parsed if str(item).strip())
    return normalized


def normalize_expression_transform(value: Any) -> str:
    """Normalize an expression-transform configuration value."""

    if value is None:
        return DEFAULT_EXPRESSION_TRANSFORM
    transform = str(value).strip()
    if not transform:
        return DEFAULT_EXPRESSION_TRANSFORM
    if transform not in SUPPORTED_EXPRESSION_TRANSFORMS:
        raise ValueError(
            f"Unsupported expression_transform '{transform}'. "
            f"Supported values: {sorted(SUPPORTED_EXPRESSION_TRANSFORMS)}"
        )
    return transform


def read_gene_list(path: Path) -> list[str]:
    """Read a one-gene-per-line vocabulary file."""

    genes: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        gene = raw_line.strip()
        if not gene or gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


def build_gene_union(*paths: Path) -> list[str]:
    """Build a deterministic gene union while preserving source order."""

    union: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for gene in read_gene_list(path):
            if gene in seen:
                continue
            seen.add(gene)
            union.append(gene)
    return union


def load_manifest_items(manifest_path: Path) -> list[dict[str, Any]]:
    """Load a flexible H5AD conversion manifest."""

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        for key in ("items", "samples", "h5ads", "files"):
            if key in data:
                raw_items = data[key]
                break
        else:
            raw_items = [data]
    else:
        raise ValueError("Manifest JSON must be a list or dict.")

    items: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for idx, raw_item in enumerate(raw_items):
        if isinstance(raw_item, str):
            item = {"h5ad": raw_item}
        elif isinstance(raw_item, dict):
            item = dict(raw_item)
        else:
            raise ValueError(f"Unsupported manifest entry at position {idx}: {type(raw_item)!r}")

        h5ad_path = item.get("h5ad") or item.get("path") or item.get("file")
        if h5ad_path is None:
            raise ValueError(f"Manifest entry {idx} must include 'h5ad' or 'path'.")

        resolved_h5ad = Path(h5ad_path).expanduser().resolve()
        sample_id = str(item.get("sample_id") or resolved_h5ad.stem)
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id '{sample_id}' found in manifest.")
        seen_sample_ids.add(sample_id)

        items.append(
            {
                "h5ad": str(resolved_h5ad),
                "sample_id": sample_id,
                "z_column": item.get("z_column"),
                "coord_axis_names": item.get("coord_axis_names"),
                "obs_columns": normalize_column_names(item.get("obs_columns")),
                "expression_transform": (
                    None
                    if item.get("expression_transform") is None
                    else normalize_expression_transform(item.get("expression_transform"))
                ),
            }
        )
    return items


def extract_coords(
    adata,
    z_column: str | None = None,
    *,
    require_coords: bool = False,
) -> np.ndarray:
    """Extract 2D or 3D coordinates from an AnnData object."""

    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    elif {"x", "y"}.issubset(set(adata.obs.columns)):
        coords = adata.obs[["x", "y"]].to_numpy(dtype=np.float32, copy=True)
    else:
        if require_coords:
            raise ValueError("No usable coordinates found. Need obsm['spatial'] or obs[['x','y']].")
        return np.zeros((adata.n_obs, 0), dtype=np.float32)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected coordinate shape (N, 2+), got {coords.shape}.")

    xy_coords = coords[:, :2].astype(np.float32, copy=False)
    if z_column is not None:
        if z_column not in adata.obs.columns:
            raise ValueError(f"Requested z column '{z_column}' not found in adata.obs.")
        z_values = adata.obs[z_column].to_numpy(dtype=np.float32, copy=True)
        return np.column_stack([xy_coords, z_values]).astype(np.float32, copy=False)

    if coords.shape[1] >= 3:
        return coords[:, :3].astype(np.float32, copy=False)

    if "z" in adata.obs.columns:
        z_values = adata.obs["z"].to_numpy(dtype=np.float32, copy=True)
        return np.column_stack([xy_coords, z_values]).astype(np.float32, copy=False)

    return xy_coords


def ensure_csr_matrix(matrix) -> csr_matrix:
    """Convert dense or sparse matrix-like input to CSR float32."""

    if issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return csr_matrix(np.asarray(matrix, dtype=np.float32))


def align_matrix_to_vocab(
    matrix: csr_matrix,
    var_names,
    vocab_to_index: dict[str, int],
    vocab_size: int,
) -> tuple[csr_matrix, dict[str, int]]:
    """Align a source expression matrix to a shared gene vocabulary."""

    source_genes = np.asarray(var_names).astype(str, copy=False)
    keep_columns = np.array([gene in vocab_to_index for gene in source_genes], dtype=bool)
    kept_source_genes = int(np.count_nonzero(keep_columns))
    dropped_source_genes = int(source_genes.shape[0] - kept_source_genes)

    if kept_source_genes == 0:
        empty = csr_matrix((matrix.shape[0], vocab_size), dtype=np.float32)
        return empty, {
            "kept_source_genes": 0,
            "dropped_source_genes": dropped_source_genes,
        }

    filtered = matrix[:, keep_columns].tocoo(copy=False)
    kept_gene_names = source_genes[keep_columns]
    mapped_cols = np.asarray([vocab_to_index[gene] for gene in kept_gene_names], dtype=np.int32)
    aligned = csr_matrix(
        (filtered.data.astype(np.float32, copy=False), (filtered.row, mapped_cols[filtered.col])),
        shape=(matrix.shape[0], vocab_size),
        dtype=np.float32,
    )
    aligned.sum_duplicates()
    aligned.eliminate_zeros()
    return aligned, {
        "kept_source_genes": kept_source_genes,
        "dropped_source_genes": dropped_source_genes,
    }


def normalize_log1p_csr(matrix: csr_matrix, target_sum: float) -> tuple[csr_matrix, np.ndarray]:
    """Apply per-cell normalize_total followed by log1p on CSR data."""

    normalized = matrix.copy().astype(np.float32)
    raw_cell_sums = np.asarray(normalized.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    if normalized.nnz == 0:
        return normalized, raw_cell_sums

    row_nnz = np.diff(normalized.indptr)
    row_scales = np.ones(normalized.shape[0], dtype=np.float32)
    valid_rows = raw_cell_sums > 0
    row_scales[valid_rows] = np.float32(target_sum) / raw_cell_sums[valid_rows]
    normalized.data *= np.repeat(row_scales, row_nnz).astype(np.float32, copy=False)
    np.log1p(normalized.data, out=normalized.data)
    normalized.eliminate_zeros()
    return normalized, raw_cell_sums


def apply_expression_transform(
    matrix: csr_matrix,
    *,
    expression_transform: str,
    normalize_target_sum: float | None,
) -> tuple[csr_matrix, np.ndarray]:
    """Apply a configured expression transform and return transformed matrix plus raw cell sums."""

    expression_transform = normalize_expression_transform(expression_transform)
    aligned = matrix.copy().astype(np.float32)
    raw_cell_sums = np.asarray(aligned.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)

    if expression_transform == "none":
        aligned.eliminate_zeros()
        return aligned, raw_cell_sums

    if expression_transform == "normalize_total_log1p":
        if normalize_target_sum is None:
            raise ValueError("normalize_target_sum is required for 'normalize_total_log1p'.")
        normalized, _ = normalize_log1p_csr(aligned, target_sum=float(normalize_target_sum))
        return normalized, raw_cell_sums

    raise AssertionError(f"Unhandled expression_transform '{expression_transform}'.")


def write_memmap_npy(path: Path, array: np.ndarray) -> None:
    """Write a numeric array as a memmap-friendly .npy file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(
        filename=str(path),
        mode="w+",
        dtype=array.dtype,
        shape=array.shape,
    )
    memmap[...] = array
    memmap.flush()
    del memmap
