"""Random-access readers over CSR memmap transcriptomics datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from .common import default_coord_axis_names, normalize_coord_axis_names
from .schema import DatasetMetadata, SampleMetadata


def _coerce_int_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.empty((0,), dtype=np.int64)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [int(part) for part in text.split(",") if part.strip()]
        return np.asarray(parsed, dtype=np.int64)
    raise TypeError(f"Unsupported index payload type: {type(value)!r}")


def _coerce_path_list(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value).expanduser().resolve()]
    if isinstance(value, Sequence):
        return [Path(item).expanduser().resolve() for item in value]
    raise TypeError(f"Unsupported manifest path type: {type(value)!r}")


def _extract_vector(row: pd.Series, prefix: str) -> np.ndarray:
    matches: list[tuple[int, str]] = []
    for key in row.index:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            matches.append((int(suffix), key))
    if not matches:
        return np.empty((0,), dtype=np.float32)
    matches.sort(key=lambda item: item[0])
    return np.asarray([row[key] for _, key in matches], dtype=np.float32)


def normalize_region_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize region coordinates independently per axis."""

    if coords.ndim != 2:
        raise ValueError(f"Expected 2D coords array, got shape {coords.shape}.")
    if coords.shape[0] == 0:
        return coords.astype(np.float32, copy=False)
    if coords.shape[1] == 0:
        return coords.astype(np.float32, copy=False)
    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    spans = np.maximum(maxs - mins, 1e-6)
    return ((coords - mins) / spans).astype(np.float32, copy=False)


class CSRMemmapSample:
    """Read one sample stored as CSR memmap arrays."""

    def __init__(self, sample_dir: str | Path, metadata: SampleMetadata) -> None:
        self.sample_dir = Path(sample_dir).expanduser().resolve()
        self.metadata = metadata
        self.sample_id = metadata.sample_id
        self.indptr = np.load(self.sample_dir / "indptr.npy", mmap_mode="r")
        self.indices = np.load(self.sample_dir / "indices.npy", mmap_mode="r")
        self.values = np.load(self.sample_dir / "values.npy", mmap_mode="r")
        self.coords = np.load(self.sample_dir / "coords.npy", mmap_mode="r")
        self.raw_cell_sums = np.load(self.sample_dir / "raw_cell_sums.npy", mmap_mode="r")
        self.present_gene_ids = np.load(self.sample_dir / "present_gene_ids.npy", mmap_mode="r")
        self.obs_names = np.load(self.sample_dir / "obs_names.npy", allow_pickle=False)
        self._obs_frame: pd.DataFrame | None = None

    @property
    def coord_axis_names(self) -> tuple[str, ...]:
        return normalize_coord_axis_names(self.metadata.coord_axis_names, self.metadata.coord_dim)

    @property
    def obs_table_path(self) -> Path:
        return self.sample_dir / "obs.parquet"

    @property
    def has_obs_table(self) -> bool:
        return self.obs_table_path.exists()

    def load_obs_frame(self) -> pd.DataFrame:
        """Load persisted obs metadata lazily."""

        if self._obs_frame is None:
            if not self.has_obs_table:
                raise FileNotFoundError(f"No obs table found for sample '{self.sample_id}'.")
            self._obs_frame = pd.read_parquet(self.obs_table_path)
        return self._obs_frame

    def fetch_rows(
        self,
        cell_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return local region CSR arrays for selected cells."""

        cell_indices = np.asarray(cell_indices, dtype=np.int64)
        starts = self.indptr[cell_indices]
        ends = self.indptr[cell_indices + 1]
        lengths = (ends - starts).astype(np.int64, copy=False)

        local_indptr = np.zeros(cell_indices.shape[0] + 1, dtype=np.int64)
        np.cumsum(lengths, out=local_indptr[1:])

        total_nnz = int(local_indptr[-1])
        local_indices = np.empty(total_nnz, dtype=np.int32)
        local_values = np.empty(total_nnz, dtype=np.float32)
        cursor = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            row_nnz = int(end - start)
            if row_nnz > 0:
                local_indices[cursor : cursor + row_nnz] = self.indices[start:end]
                local_values[cursor : cursor + row_nnz] = self.values[start:end]
                cursor += row_nnz

        local_coords = np.asarray(self.coords[cell_indices], dtype=np.float32)
        return local_coords, local_indptr, local_indices, local_values

    def fetch_cell(self, cell_index: int, *, include_obs: bool = False) -> dict[str, Any]:
        """Return one cell in sparse local form."""

        cell_index = int(cell_index)
        start = int(self.indptr[cell_index])
        end = int(self.indptr[cell_index + 1])
        item = {
            "sample_id": self.sample_id,
            "cell_index": cell_index,
            "obs_name": str(self.obs_names[cell_index]),
            "coords": np.asarray(self.coords[cell_index], dtype=np.float32),
            "coord_axis_names": self.coord_axis_names,
            "raw_cell_sum": float(self.raw_cell_sums[cell_index]),
            "gene_indices": np.asarray(self.indices[start:end], dtype=np.int32),
            "gene_values": np.asarray(self.values[start:end], dtype=np.float32),
        }
        if include_obs and self.has_obs_table:
            row = self.load_obs_frame().iloc[cell_index].to_dict()
            row.pop("obs_name", None)
            item["obs"] = row
        return item

    def iter_cells(self, *, include_obs: bool = False) -> Iterator[dict[str, Any]]:
        """Yield all cells in the sample."""

        for cell_index in range(self.metadata.n_cells):
            yield self.fetch_cell(cell_index, include_obs=include_obs)


class MemmapDataset:
    """Open and iterate a memmap dataset rooted at one directory."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.metadata = DatasetMetadata.load(self.dataset_root)
        self.gene_vocab = [
            line.strip()
            for line in (self.dataset_root / "gene_vocab.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.samples: dict[str, CSRMemmapSample] = {
            sample.sample_id: CSRMemmapSample(self.dataset_root / sample.sample_id, sample)
            for sample in self.metadata.samples
        }
        self.sample_coord_dims = {
            sample_id: sample.metadata.coord_dim for sample_id, sample in self.samples.items()
        }
        self.sample_coord_axis_names = {
            sample_id: sample.coord_axis_names for sample_id, sample in self.samples.items()
        }

    @staticmethod
    def coerce_cell_index_array(value: Any) -> np.ndarray:
        return _coerce_int_array(value)

    @staticmethod
    def extract_anchor_vector(row: pd.Series) -> np.ndarray:
        return _extract_vector(row, "anchor_dim")

    def __len__(self) -> int:
        return sum(sample.metadata.n_cells for sample in self.samples.values())

    def iter_cells(
        self,
        sample_ids: Sequence[str] | None = None,
        *,
        include_obs: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield cell-level sparse records."""

        selected_ids = list(sample_ids) if sample_ids is not None else list(self.samples.keys())
        for sample_id in selected_ids:
            if sample_id not in self.samples:
                raise KeyError(f"Unknown sample_id '{sample_id}'.")
            yield from self.samples[sample_id].iter_cells(include_obs=include_obs)

    def infer_coord_axis_names(
        self,
        region_manifest_path: str | Path | Sequence[str | Path] | None = None,
    ) -> tuple[str, ...] | None:
        if region_manifest_path is None:
            axis_names = sorted(set(self.sample_coord_axis_names.values()))
        else:
            manifest = self.load_region_manifest(region_manifest_path)
            sample_ids = manifest["sample_id"].astype(str).drop_duplicates().tolist()
            axis_names = sorted(
                {
                    self.sample_coord_axis_names[sample_id]
                    for sample_id in sample_ids
                    if sample_id in self.sample_coord_axis_names
                }
            )
        if not axis_names:
            return None
        if len(axis_names) != 1:
            raise ValueError(f"Mixed coord_axis_names detected: {axis_names}")
        return axis_names[0]

    def load_region_manifest(
        self,
        region_manifest_path: str | Path | Sequence[str | Path],
    ) -> pd.DataFrame:
        """Load one or more region manifests into one DataFrame."""

        frames: list[pd.DataFrame] = []
        for manifest_id, path in enumerate(_coerce_path_list(region_manifest_path)):
            frame = pd.read_parquet(path).copy()
            required = {"sample_id", "region_id", "source_cell_indices"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Region manifest '{path}' missing required columns: {sorted(missing)}"
                )
            frame["_manifest_id"] = manifest_id
            frame["_manifest_path"] = str(path)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def iter_regions(
        self,
        region_manifest_path: str | Path | Sequence[str | Path],
        *,
        normalize_coords: bool = True,
        include_obs: bool = False,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield region-level sparse records reconstructed from a region manifest."""

        manifest = self.load_region_manifest(region_manifest_path)
        for idx, row in enumerate(manifest.itertuples(index=False)):
            if limit is not None and idx >= limit:
                break

            row_series = pd.Series(row._asdict())
            sample_id = str(row_series["sample_id"])
            if sample_id not in self.samples:
                raise KeyError(f"sample_id '{sample_id}' not found in dataset manifest.")

            sample = self.samples[sample_id]
            source_cell_indices = _coerce_int_array(row_series["source_cell_indices"])
            item = self.build_group(
                sample_id=sample_id,
                source_cell_indices=source_cell_indices,
                group_id=str(row_series["region_id"]),
                group_type="manifest_region",
                group_index=int(row_series["region_index"]) if "region_index" in row_series else idx,
                sample_group_index=(
                    int(row_series["sample_region_index"])
                    if "sample_region_index" in row_series
                    else idx
                ),
                normalize_coords=normalize_coords,
                include_obs=include_obs,
                anchor_coord=_extract_vector(row_series, "anchor_dim"),
                metadata={
                    key: value
                    for key, value in row_series.items()
                    if key not in {"sample_id", "region_id", "source_cell_indices"}
                    and not key.startswith("anchor_dim")
                },
            )
            item["region_index"] = item["group_index"]
            item["sample_region_index"] = item["sample_group_index"]
            item["region_id"] = item["group_id"]
            yield item

    def build_group(
        self,
        *,
        sample_id: str,
        source_cell_indices: np.ndarray,
        group_id: str,
        group_type: str,
        group_index: int,
        sample_group_index: int = 0,
        normalize_coords: bool = True,
        include_obs: bool = False,
        anchor_coord: np.ndarray | None = None,
        block_bounds: Sequence[Sequence[float]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a pretraining group from one sample and a cell-index subset."""

        if sample_id not in self.samples:
            raise KeyError(f"sample_id '{sample_id}' not found in dataset manifest.")
        sample = self.samples[sample_id]
        source_cell_indices = _coerce_int_array(source_cell_indices)
        raw_coords, cell_ptr, gene_indices, gene_values = sample.fetch_rows(source_cell_indices)
        coords = normalize_region_coords(raw_coords) if normalize_coords else raw_coords.copy()
        coord_axis_names = self.sample_coord_axis_names.get(
            sample_id,
            default_coord_axis_names(coords.shape[1]),
        )

        item: dict[str, Any] = {
            "group_index": int(group_index),
            "sample_group_index": int(sample_group_index),
            "group_id": str(group_id),
            "group_type": str(group_type),
            "sample_id": sample_id,
            "coords": coords.astype(np.float32, copy=False),
            "raw_coords": raw_coords.astype(np.float32, copy=False),
            "cell_ptr": cell_ptr.astype(np.int64, copy=False),
            "gene_indices": gene_indices.astype(np.int32, copy=False),
            "gene_values": gene_values.astype(np.float32, copy=False),
            "source_cell_indices": source_cell_indices.astype(np.int64, copy=False),
            "anchor_coord": (
                np.empty((0,), dtype=np.float32)
                if anchor_coord is None
                else np.asarray(anchor_coord, dtype=np.float32)
            ),
            "coord_axis_names": coord_axis_names,
            "block_bounds": []
            if block_bounds is None
            else [[float(low), float(high)] for low, high in block_bounds],
            "sampling_metadata": {} if metadata is None else dict(metadata),
        }
        if include_obs and sample.has_obs_table:
            item["obs"] = (
                sample.load_obs_frame()
                .iloc[source_cell_indices]
                .drop(columns=["obs_name"], errors="ignore")
                .to_dict(orient="records")
            )
        return item
