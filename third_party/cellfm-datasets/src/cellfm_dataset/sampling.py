"""Group samplers for cell foundation model pretraining."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from .distributed import DistributedContext, shard_iterable


@dataclass(frozen=True)
class GroupSpec:
    """A modality-agnostic pretraining group definition."""

    sample_id: str
    group_id: str
    source_cell_indices: np.ndarray
    group_type: str
    group_index: int
    sample_group_index: int = 0
    anchor_coord: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    block_bounds: tuple[tuple[float, float], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported block manifest format: {path}")


class ManifestRegionSampler:
    """Wrap an existing region manifest as group specifications."""

    def __init__(self, region_manifest_path: str | Path | Sequence[str | Path]) -> None:
        self.region_manifest_path = region_manifest_path

    def iter_specs(
        self,
        dataset,
        distributed_context: DistributedContext | None = None,
    ) -> Iterator[GroupSpec]:
        manifest = dataset.load_region_manifest(self.region_manifest_path)

        def _iter() -> Iterator[GroupSpec]:
            for idx, row in enumerate(manifest.itertuples(index=False)):
                row_series = pd.Series(row._asdict())
                anchor_coord = dataset.extract_anchor_vector(row_series)
                metadata = {
                    key: value
                    for key, value in row_series.items()
                    if key not in {"sample_id", "region_id", "source_cell_indices"}
                    and not key.startswith("anchor_dim")
                }
                yield GroupSpec(
                    sample_id=str(row_series["sample_id"]),
                    group_id=str(row_series["region_id"]),
                    source_cell_indices=dataset.coerce_cell_index_array(row_series["source_cell_indices"]),
                    group_type="manifest_region",
                    group_index=int(row_series["region_index"]) if "region_index" in row_series else idx,
                    sample_group_index=(
                        int(row_series["sample_region_index"])
                        if "sample_region_index" in row_series
                        else idx
                    ),
                    anchor_coord=anchor_coord,
                    metadata=metadata,
                )

        yield from shard_iterable(_iter(), distributed_context, index_fn=lambda spec: spec.group_index)


class RandomCellSampler:
    """Sample fixed-size random cell groups for single-cell or spatial data."""

    def __init__(
        self,
        *,
        cells_per_group: int,
        num_groups: int,
        seed: int = 0,
        with_replacement: bool = False,
        sample_ids: Sequence[str] | None = None,
        sample_weight_mode: str = "n_cells",
        stratify_obs_column: str | None = None,
    ) -> None:
        if cells_per_group <= 0:
            raise ValueError("cells_per_group must be positive.")
        if num_groups <= 0:
            raise ValueError("num_groups must be positive.")
        self.cells_per_group = int(cells_per_group)
        self.num_groups = int(num_groups)
        self.seed = int(seed)
        self.with_replacement = bool(with_replacement)
        self.sample_ids = None if sample_ids is None else tuple(str(item) for item in sample_ids)
        self.sample_weight_mode = str(sample_weight_mode)
        self.stratify_obs_column = None if stratify_obs_column is None else str(stratify_obs_column)

    def _resolve_sample_weights(self, dataset, sample_ids: list[str]) -> np.ndarray:
        if self.sample_weight_mode == "uniform":
            weights = np.ones(len(sample_ids), dtype=np.float64)
        elif self.sample_weight_mode == "n_cells":
            weights = np.asarray(
                [dataset.samples[sample_id].metadata.n_cells for sample_id in sample_ids],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"Unsupported sample_weight_mode: {self.sample_weight_mode}")
        return weights / weights.sum()

    def _sample_pool(
        self,
        *,
        dataset,
        sample_id: str,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        sample = dataset.samples[sample_id]
        pool = np.arange(sample.metadata.n_cells, dtype=np.int64)
        metadata: dict[str, Any] = {}
        if self.stratify_obs_column is None:
            return pool, metadata

        if not sample.has_obs_table:
            raise ValueError(
                f"stratify_obs_column='{self.stratify_obs_column}' requires obs.parquet for sample '{sample_id}'."
            )
        obs_frame = sample.load_obs_frame()
        if self.stratify_obs_column not in obs_frame.columns:
            raise ValueError(
                f"Column '{self.stratify_obs_column}' not found in obs.parquet for sample '{sample_id}'."
            )
        labels = obs_frame[self.stratify_obs_column]
        counts = labels.value_counts(dropna=False)
        selected_value = counts.index[rng.integers(len(counts))]
        selected_mask = labels.eq(selected_value).to_numpy()
        metadata["stratify_obs_column"] = self.stratify_obs_column
        metadata["stratify_value"] = selected_value if selected_value == selected_value else None
        return pool[selected_mask], metadata

    def iter_specs(
        self,
        dataset,
        distributed_context: DistributedContext | None = None,
    ) -> Iterator[GroupSpec]:
        rng = np.random.default_rng(
            self.seed if distributed_context is None else distributed_context.seed_offset(self.seed)
        )
        sample_ids = list(self.sample_ids) if self.sample_ids is not None else list(dataset.samples.keys())
        if not sample_ids:
            return
        sample_weights = self._resolve_sample_weights(dataset, sample_ids)
        sample_group_counts = {sample_id: 0 for sample_id in sample_ids}

        def _iter() -> Iterator[GroupSpec]:
            for group_index in range(self.num_groups):
                sample_id = str(rng.choice(sample_ids, p=sample_weights))
                sample_pool, metadata = self._sample_pool(dataset=dataset, sample_id=sample_id, rng=rng)
                if sample_pool.shape[0] < self.cells_per_group and not self.with_replacement:
                    raise ValueError(
                        f"Sample '{sample_id}' has only {sample_pool.shape[0]} eligible cells, "
                        f"but cells_per_group={self.cells_per_group}. "
                        "Use with_replacement=True, reduce cells_per_group, or change the stratification."
                    )

                selected = rng.choice(
                    sample_pool,
                    size=self.cells_per_group,
                    replace=self.with_replacement,
                )
                selected = np.sort(selected.astype(np.int64, copy=False))
                sample_group_index = sample_group_counts[sample_id]
                sample_group_counts[sample_id] += 1
                metadata["with_replacement"] = self.with_replacement
                metadata["epoch"] = 0 if distributed_context is None else distributed_context.epoch
                yield GroupSpec(
                    sample_id=sample_id,
                    group_id=f"{sample_id}_random_group{group_index}",
                    source_cell_indices=selected,
                    group_type="random_cells",
                    group_index=group_index,
                    sample_group_index=sample_group_index,
                    metadata=metadata,
                )

        yield from shard_iterable(_iter(), distributed_context, index_fn=lambda spec: spec.group_index)


class SpatialBlockSampler:
    """Sample spatial groups from user-defined or grid-generated blocks."""

    def __init__(
        self,
        *,
        block_shape: Sequence[float] | None = None,
        stride: Sequence[float] | None = None,
        block_manifest_path: str | Path | None = None,
        sample_ids: Sequence[str] | None = None,
        min_cells: int = 1,
        max_cells: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if block_shape is None and block_manifest_path is None:
            raise ValueError("Provide either block_shape or block_manifest_path.")
        self.block_shape = None if block_shape is None else tuple(float(item) for item in block_shape)
        self.stride = None if stride is None else tuple(float(item) for item in stride)
        self.block_manifest_path = (
            None if block_manifest_path is None else Path(block_manifest_path).expanduser().resolve()
        )
        self.sample_ids = None if sample_ids is None else tuple(str(item) for item in sample_ids)
        self.min_cells = int(min_cells)
        self.max_cells = None if max_cells is None else int(max_cells)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def _select_cells_from_bounds(self, coords: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
        mask = np.ones(coords.shape[0], dtype=bool)
        for axis_index, (low, high) in enumerate(bounds):
            mask &= coords[:, axis_index] >= low
            mask &= coords[:, axis_index] < high
        return np.flatnonzero(mask).astype(np.int64, copy=False)

    def _manifest_specs(self, dataset, distributed_context: DistributedContext | None) -> Iterator[GroupSpec]:
        assert self.block_manifest_path is not None
        frame = _load_table(self.block_manifest_path)
        epoch_seed = self.seed if distributed_context is None else distributed_context.seed_offset(self.seed)

        def _iter() -> Iterator[GroupSpec]:
            for idx, row in enumerate(frame.itertuples(index=False)):
                row_series = pd.Series(row._asdict())
                sample_id = str(row_series["sample_id"])
                sample = dataset.samples[sample_id]
                if "source_cell_indices" in row_series:
                    cell_indices = dataset.coerce_cell_index_array(row_series["source_cell_indices"])
                    bounds: tuple[tuple[float, float], ...] = ()
                else:
                    bounds_list: list[tuple[float, float]] = []
                    for axis_name in sample.coord_axis_names:
                        min_key = f"min_{axis_name}"
                        max_key = f"max_{axis_name}"
                        if min_key not in row_series or max_key not in row_series:
                            raise ValueError(
                                f"Block manifest row for sample '{sample_id}' is missing '{min_key}'/'{max_key}'."
                            )
                        bounds_list.append((float(row_series[min_key]), float(row_series[max_key])))
                    bounds = tuple(bounds_list)
                    cell_indices = self._select_cells_from_bounds(np.asarray(sample.coords), bounds)
                if cell_indices.shape[0] < self.min_cells:
                    continue
                if self.max_cells is not None and cell_indices.shape[0] > self.max_cells:
                    rng = np.random.default_rng(epoch_seed + idx)
                    cell_indices = np.sort(rng.choice(cell_indices, size=self.max_cells, replace=False))
                metadata = {
                    key: value
                    for key, value in row_series.items()
                    if key not in {"sample_id", "block_id", "source_cell_indices"}
                    and not key.startswith("min_")
                    and not key.startswith("max_")
                }
                metadata["epoch"] = 0 if distributed_context is None else distributed_context.epoch
                yield GroupSpec(
                    sample_id=sample_id,
                    group_id=str(row_series.get("block_id", f"{sample_id}_block{idx}")),
                    source_cell_indices=cell_indices,
                    group_type="custom_block",
                    group_index=idx,
                    sample_group_index=int(row_series.get("sample_block_index", idx)),
                    block_bounds=bounds,
                    metadata=metadata,
                )

        yield from shard_iterable(_iter(), distributed_context, index_fn=lambda spec: spec.group_index)

    def _grid_specs(
        self,
        dataset,
        distributed_context: DistributedContext | None,
    ) -> Iterator[GroupSpec]:
        assert self.block_shape is not None
        rng = np.random.default_rng(
            self.seed if distributed_context is None else distributed_context.seed_offset(self.seed)
        )
        sample_ids = list(self.sample_ids) if self.sample_ids is not None else list(dataset.samples.keys())
        epoch_value = 0 if distributed_context is None else distributed_context.epoch

        def _iter() -> Iterator[GroupSpec]:
            group_index = 0
            for sample_id in sample_ids:
                sample = dataset.samples[sample_id]
                coords = np.asarray(sample.coords, dtype=np.float32)
                coord_dim = coords.shape[1]
                if coord_dim == 0:
                    raise ValueError(f"Sample '{sample_id}' has no coordinates; cannot use SpatialBlockSampler.")
                if len(self.block_shape) != coord_dim:
                    raise ValueError(
                        f"block_shape length {len(self.block_shape)} does not match coord_dim={coord_dim} for sample '{sample_id}'."
                    )
                stride = self.block_shape if self.stride is None else self.stride
                if len(stride) != coord_dim:
                    raise ValueError(
                        f"stride length {len(stride)} does not match coord_dim={coord_dim} for sample '{sample_id}'."
                    )

                mins = np.nanmin(coords, axis=0)
                maxs = np.nanmax(coords, axis=0)
                axis_starts = []
                for axis_index in range(coord_dim):
                    start = float(mins[axis_index])
                    stop = float(maxs[axis_index])
                    step = float(stride[axis_index])
                    starts = np.arange(start, stop + step, step, dtype=np.float32)
                    axis_starts.append(starts.tolist())

                sample_specs: list[GroupSpec] = []
                sample_group_index = 0
                for starts in itertools.product(*axis_starts):
                    bounds = tuple(
                        (float(starts[axis_index]), float(starts[axis_index]) + float(self.block_shape[axis_index]))
                        for axis_index in range(coord_dim)
                    )
                    cell_indices = self._select_cells_from_bounds(coords, bounds)
                    if cell_indices.shape[0] < self.min_cells:
                        continue
                    if self.max_cells is not None and cell_indices.shape[0] > self.max_cells:
                        cell_indices = np.sort(rng.choice(cell_indices, size=self.max_cells, replace=False))
                    sample_specs.append(
                        GroupSpec(
                            sample_id=sample_id,
                            group_id=f"{sample_id}_block{sample_group_index}",
                            source_cell_indices=cell_indices,
                            group_type="spatial_block",
                            group_index=group_index,
                            sample_group_index=sample_group_index,
                            block_bounds=bounds,
                            metadata={
                                "block_shape": list(self.block_shape),
                                "stride": list(stride),
                                "epoch": epoch_value,
                            },
                        )
                    )
                    group_index += 1
                    sample_group_index += 1
                if self.shuffle:
                    rng.shuffle(sample_specs)
                yield from sample_specs

        yield from shard_iterable(_iter(), distributed_context, index_fn=lambda spec: spec.group_index)

    def iter_specs(
        self,
        dataset,
        distributed_context: DistributedContext | None = None,
    ) -> Iterator[GroupSpec]:
        if self.block_manifest_path is not None:
            yield from self._manifest_specs(dataset, distributed_context)
            return
        yield from self._grid_specs(dataset, distributed_context)
