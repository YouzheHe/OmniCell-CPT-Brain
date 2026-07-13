"""Hugging Face datasets adapters."""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from ._optional import require_dependency
from .distributed import DistributedContext, resolve_distributed_context, shard_iterable
from .memmap import MemmapDataset
from .sampling import ManifestRegionSampler


def _datasets():
    return require_dependency("datasets", "pip install 'cellfm-datasets[hf]'")


def _cell_features(*, include_obs: bool):
    datasets = _datasets()
    features = {
        "sample_id": datasets.Value("string"),
        "cell_index": datasets.Value("int64"),
        "obs_name": datasets.Value("string"),
        "coords": datasets.Sequence(datasets.Value("float32")),
        "coord_axis_names": datasets.Sequence(datasets.Value("string")),
        "raw_cell_sum": datasets.Value("float32"),
        "gene_indices": datasets.Sequence(datasets.Value("int32")),
        "gene_values": datasets.Sequence(datasets.Value("float32")),
    }
    if include_obs:
        features["obs_json"] = datasets.Value("string")
    return datasets.Features(features)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _region_features(*, include_obs: bool):
    datasets = _datasets()
    features = {
        "group_index": datasets.Value("int64"),
        "sample_group_index": datasets.Value("int64"),
        "group_id": datasets.Value("string"),
        "group_type": datasets.Value("string"),
        "sample_id": datasets.Value("string"),
        "coords": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
        "raw_coords": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
        "cell_ptr": datasets.Sequence(datasets.Value("int64")),
        "gene_indices": datasets.Sequence(datasets.Value("int32")),
        "gene_values": datasets.Sequence(datasets.Value("float32")),
        "source_cell_indices": datasets.Sequence(datasets.Value("int64")),
        "anchor_coord": datasets.Sequence(datasets.Value("float32")),
        "coord_axis_names": datasets.Sequence(datasets.Value("string")),
        "block_bounds": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
        "sampling_metadata_json": datasets.Value("string"),
    }
    if include_obs:
        features["obs_json"] = datasets.Value("string")
    return datasets.Features(features)


def _manifest_region_features(*, include_obs: bool):
    datasets = _datasets()
    base_features = _region_features(include_obs=include_obs)
    features = {key: base_features[key] for key in base_features}
    features["region_index"] = datasets.Value("int64")
    features["sample_region_index"] = datasets.Value("int64")
    features["region_id"] = datasets.Value("string")
    return datasets.Features(features)


def _to_python_record(record: dict[str, Any], *, include_obs: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.items():
        if key == "obs":
            if include_obs:
                result["obs_json"] = json.dumps(_json_ready(value), ensure_ascii=False)
            continue
        if key == "sampling_metadata":
            result["sampling_metadata_json"] = json.dumps(_json_ready(value), ensure_ascii=False)
            continue
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def load_hf_cell_dataset(
    *,
    dataset_root: str,
    sample_ids: Sequence[str] | None = None,
    streaming: bool = True,
    include_obs: bool = False,
    distributed_context: DistributedContext | None = None,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    shard_mode: str = "stride",
    infer_worker: bool = True,
):
    """Expose cell-level sparse records as a Hugging Face dataset."""

    datasets = _datasets()
    memmap_dataset = MemmapDataset(dataset_root)
    features = _cell_features(include_obs=include_obs)

    def generator():
        context = resolve_distributed_context(
            distributed_context,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            shard_mode=shard_mode,
            infer_worker=infer_worker,
        )
        cell_iterable = memmap_dataset.iter_cells(sample_ids=sample_ids, include_obs=include_obs)
        for item in shard_iterable(cell_iterable, context):
            yield _to_python_record(item, include_obs=include_obs)

    if streaming:
        return datasets.IterableDataset.from_generator(generator, features=features)
    return datasets.Dataset.from_generator(generator, features=features)


def load_hf_region_dataset(
    *,
    dataset_root: str,
    region_manifest_path,
    normalize_coords: bool = True,
    streaming: bool = True,
    include_obs: bool = False,
    limit: int | None = None,
    distributed_context: DistributedContext | None = None,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    shard_mode: str = "stride",
    infer_worker: bool = True,
):
    """Expose region-level sparse records as a Hugging Face dataset."""
    datasets = _datasets()
    memmap_dataset = MemmapDataset(dataset_root)
    sampler = ManifestRegionSampler(region_manifest_path)
    features = _manifest_region_features(include_obs=include_obs)

    def generator():
        context = resolve_distributed_context(
            distributed_context,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            shard_mode=shard_mode,
            infer_worker=infer_worker,
        )
        yielded = 0
        for spec in sampler.iter_specs(memmap_dataset, distributed_context=context):
            if limit is not None and yielded >= limit:
                break
            item = memmap_dataset.build_group(
                sample_id=spec.sample_id,
                source_cell_indices=spec.source_cell_indices,
                group_id=spec.group_id,
                group_type=spec.group_type,
                group_index=spec.group_index,
                sample_group_index=spec.sample_group_index,
                normalize_coords=normalize_coords,
                include_obs=include_obs,
                anchor_coord=spec.anchor_coord,
                block_bounds=spec.block_bounds,
                metadata=spec.metadata,
            )
            item["region_index"] = item["group_index"]
            item["sample_region_index"] = item["sample_group_index"]
            item["region_id"] = item["group_id"]
            yield _to_python_record(item, include_obs=include_obs)
            yielded += 1

    if streaming:
        return datasets.IterableDataset.from_generator(generator, features=features)
    return datasets.Dataset.from_generator(generator, features=features)


def load_hf_group_dataset(
    *,
    dataset_root: str,
    sampler,
    normalize_coords: bool = True,
    streaming: bool = True,
    include_obs: bool = False,
    limit: int | None = None,
    distributed_context: DistributedContext | None = None,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    shard_mode: str = "stride",
    infer_worker: bool = True,
):
    """Expose generic pretraining groups as a Hugging Face dataset."""

    datasets = _datasets()
    memmap_dataset = MemmapDataset(dataset_root)
    features = _region_features(include_obs=include_obs)

    def generator():
        context = resolve_distributed_context(
            distributed_context,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            shard_mode=shard_mode,
            infer_worker=infer_worker,
        )
        yielded = 0
        for spec in sampler.iter_specs(memmap_dataset, distributed_context=context):
            if limit is not None and yielded >= limit:
                break
            item = memmap_dataset.build_group(
                sample_id=spec.sample_id,
                source_cell_indices=spec.source_cell_indices,
                group_id=spec.group_id,
                group_type=spec.group_type,
                group_index=spec.group_index,
                sample_group_index=spec.sample_group_index,
                normalize_coords=normalize_coords,
                include_obs=include_obs,
                anchor_coord=spec.anchor_coord,
                block_bounds=spec.block_bounds,
                metadata=spec.metadata,
            )
            yield _to_python_record(item, include_obs=include_obs)
            yielded += 1

    if streaming:
        return datasets.IterableDataset.from_generator(generator, features=features)
    return datasets.Dataset.from_generator(generator, features=features)
