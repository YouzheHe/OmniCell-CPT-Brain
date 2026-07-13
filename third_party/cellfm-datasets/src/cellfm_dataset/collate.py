"""Optional Torch collation utilities for sparse region batches."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._optional import require_dependency
from .common import default_coord_axis_names

_NON_MODEL_FIELDS = frozenset(
    {
        "block_bounds",
        "group_id",
        "group_index",
        "group_type",
        "region_id",
        "sample_id",
        "sample_group_index",
        "region_index",
        "sample_region_index",
        "anchor_coord",
        "raw_coords",
        "sampling_metadata",
        "source_cell_indices",
    }
)


def _torch():
    return require_dependency("torch", "pip install 'cellfm-datasets[torch]'")


def _as_array(value: Any, dtype=None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if dtype is not None else value
    return np.asarray(value, dtype=dtype)


def csr_region_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate sparse region items while keeping variable nnz flattened."""

    torch = _torch()

    batch_size = len(batch)
    if batch_size == 0:
        raise ValueError("csr_region_collate requires a non-empty batch.")

    n_cells = int(_as_array(batch[0]["coords"]).shape[0])
    cell_counts = [int(_as_array(item["coords"]).shape[0]) for item in batch]
    if any(count != n_cells for count in cell_counts[1:]):
        raise ValueError(
            "All regions in a batch must have the same number of cells. "
            f"Found batch cell counts: {cell_counts}"
        )

    coord_dim = max(int(_as_array(item["coords"]).shape[1]) for item in batch)
    anchor_dim = max(int(_as_array(item["anchor_coord"], dtype=np.float32).shape[0]) for item in batch)
    coord_axis_names = tuple(batch[0].get("coord_axis_names", default_coord_axis_names(coord_dim)))
    if any(tuple(item.get("coord_axis_names", coord_axis_names)) != coord_axis_names for item in batch[1:]):
        raise ValueError("All regions in a batch must share the same coord_axis_names ordering.")

    coords = torch.zeros((batch_size, n_cells, coord_dim), dtype=torch.float32)
    raw_coords = torch.zeros((batch_size, n_cells, coord_dim), dtype=torch.float32)
    source_cell_indices = torch.full((batch_size, n_cells), -1, dtype=torch.long)
    cell_mask = torch.ones((batch_size, n_cells), dtype=torch.bool)
    cell_ptr = torch.full((batch_size, n_cells + 1), -1, dtype=torch.long)
    anchor_coords = torch.zeros((batch_size, anchor_dim), dtype=torch.float32)

    region_ptr = torch.zeros((batch_size + 1,), dtype=torch.long)
    nnz_counts = [int(_as_array(item["gene_indices"]).shape[0]) for item in batch]
    if nnz_counts:
        region_ptr[1:] = torch.tensor(nnz_counts, dtype=torch.long).cumsum(dim=0)

    gene_indices = torch.cat(
        [torch.from_numpy(_as_array(item["gene_indices"], dtype=np.int32)) for item in batch],
        dim=0,
    )
    gene_values = torch.cat(
        [torch.from_numpy(_as_array(item["gene_values"], dtype=np.float32)) for item in batch],
        dim=0,
    )

    result: dict[str, Any] = {
        "group_index": torch.tensor(
            [int(item.get("group_index", item.get("region_index", 0))) for item in batch],
            dtype=torch.long,
        ),
        "sample_group_index": torch.tensor(
            [int(item.get("sample_group_index", item.get("sample_region_index", 0))) for item in batch],
            dtype=torch.long,
        ),
        "group_id": [str(item.get("group_id", item.get("region_id", ""))) for item in batch],
        "group_type": [str(item.get("group_type", "group")) for item in batch],
        "sample_id": [str(item["sample_id"]) for item in batch],
        "coords": coords,
        "raw_coords": raw_coords,
        "cell_mask": cell_mask,
        "cell_ptr": cell_ptr,
        "source_cell_indices": source_cell_indices,
        "anchor_coord": anchor_coords,
        "coord_axis_names": coord_axis_names,
        "gene_indices": gene_indices,
        "gene_values": gene_values,
        "region_ptr": region_ptr,
    }
    if all("region_index" in item for item in batch):
        result["region_index"] = torch.tensor([int(item["region_index"]) for item in batch], dtype=torch.long)
    if all("sample_region_index" in item for item in batch):
        result["sample_region_index"] = torch.tensor(
            [int(item["sample_region_index"]) for item in batch],
            dtype=torch.long,
        )
    if all("region_id" in item for item in batch):
        result["region_id"] = [str(item["region_id"]) for item in batch]

    for batch_idx, item in enumerate(batch):
        item_coords = _as_array(item["coords"], dtype=np.float32)
        item_raw_coords = _as_array(item["raw_coords"], dtype=np.float32)
        item_source_cell_indices = _as_array(item["source_cell_indices"], dtype=np.int64)
        item_cell_ptr = _as_array(item["cell_ptr"], dtype=np.int64)
        item_anchor = _as_array(item["anchor_coord"], dtype=np.float32)

        local_coord_dim = int(item_coords.shape[1])
        local_anchor_dim = int(item_anchor.shape[0])

        coords[batch_idx, :n_cells, :local_coord_dim] = torch.from_numpy(item_coords)
        raw_coords[batch_idx, :n_cells, :local_coord_dim] = torch.from_numpy(item_raw_coords)
        source_cell_indices[batch_idx, :n_cells] = torch.from_numpy(item_source_cell_indices)
        cell_ptr[batch_idx, : n_cells + 1] = torch.from_numpy(item_cell_ptr)
        if local_anchor_dim > 0:
            anchor_coords[batch_idx, :local_anchor_dim] = torch.from_numpy(item_anchor)

    return result


class CellFMRegionDataCollator:
    """Collate sparse region items and optionally strip non-model fields."""

    def __init__(self, keep_extra: bool = False) -> None:
        self.keep_extra = keep_extra

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = csr_region_collate(features)
        if self.keep_extra:
            return batch
        return {key: value for key, value in batch.items() if key not in _NON_MODEL_FIELDS}
