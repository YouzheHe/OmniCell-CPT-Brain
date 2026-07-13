"""Utilities for exporting benchmark splits to local disk or the Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ._optional import require_dependency
from .checksum import write_checksum_manifest
from .hf import load_hf_cell_dataset, load_hf_group_dataset, load_hf_region_dataset
from .sampling import RandomCellSampler, SpatialBlockSampler


def _datasets():
    return require_dependency("datasets", "pip install 'cellfm-datasets[hf]'")


def _huggingface_hub():
    return require_dependency("huggingface_hub", "pip install 'cellfm-datasets[hf]'")


def materialize_benchmark_split(
    *,
    dataset_root: str,
    split_kind: str,
    include_obs: bool = False,
    sample_ids: Sequence[str] | None = None,
    limit: int | None = None,
    region_manifest_path: str | Path | None = None,
    cells_per_group: int = 256,
    num_groups: int = 10_000,
    with_replacement: bool = False,
    sample_weight_mode: str = "n_cells",
    stratify_obs_column: str | None = None,
    block_shape: Sequence[float] | None = None,
    stride: Sequence[float] | None = None,
    min_cells: int = 64,
    max_cells: int = 512,
    seed: int = 7,
):
    """Materialize one benchmark split as a finite Hugging Face dataset."""

    normalized_kind = str(split_kind).strip().lower()
    if normalized_kind == "cells":
        dataset = load_hf_cell_dataset(
            dataset_root=dataset_root,
            sample_ids=sample_ids,
            streaming=False,
            include_obs=include_obs,
        )
        if limit is not None:
            dataset = dataset.select(range(min(int(limit), len(dataset))))
        return dataset

    if normalized_kind == "regions":
        if region_manifest_path is None:
            raise ValueError("--region-manifest is required when split_kind=regions")
        return load_hf_region_dataset(
            dataset_root=dataset_root,
            region_manifest_path=region_manifest_path,
            streaming=False,
            include_obs=include_obs,
            limit=limit,
        )

    if normalized_kind == "random_groups":
        sampler = RandomCellSampler(
            cells_per_group=cells_per_group,
            num_groups=num_groups,
            seed=seed,
            with_replacement=with_replacement,
            sample_ids=sample_ids,
            sample_weight_mode=sample_weight_mode,
            stratify_obs_column=stratify_obs_column,
        )
        return load_hf_group_dataset(
            dataset_root=dataset_root,
            sampler=sampler,
            streaming=False,
            include_obs=include_obs,
            limit=limit,
        )

    if normalized_kind == "spatial_blocks":
        if block_shape is None:
            raise ValueError("--block-shape is required when split_kind=spatial_blocks")
        sampler = SpatialBlockSampler(
            block_shape=tuple(float(item) for item in block_shape),
            stride=None if stride is None else tuple(float(item) for item in stride),
            sample_ids=sample_ids,
            min_cells=min_cells,
            max_cells=max_cells,
            seed=seed,
        )
        return load_hf_group_dataset(
            dataset_root=dataset_root,
            sampler=sampler,
            streaming=False,
            include_obs=include_obs,
            limit=limit,
        )

    raise ValueError(
        "split_kind must be one of: cells, random_groups, spatial_blocks, regions"
    )


def export_benchmark_split(
    *,
    dataset_root: str,
    split_kind: str,
    split_name: str = "train",
    output_dir: str | Path | None = None,
    repo_id: str | None = None,
    config_name: str = "default",
    include_obs: bool = False,
    sample_ids: Sequence[str] | None = None,
    limit: int | None = None,
    region_manifest_path: str | Path | None = None,
    cells_per_group: int = 256,
    num_groups: int = 10_000,
    with_replacement: bool = False,
    sample_weight_mode: str = "n_cells",
    stratify_obs_column: str | None = None,
    block_shape: Sequence[float] | None = None,
    stride: Sequence[float] | None = None,
    min_cells: int = 64,
    max_cells: int = 512,
    seed: int = 7,
    token: str | None = None,
    private: bool | None = None,
    create_pr: bool = False,
    commit_message: str | None = None,
    max_shard_size: str | int | None = None,
    num_shards: int | None = None,
    num_proc: int | None = None,
) -> dict[str, Any]:
    """Export a selected benchmark split to local disk and/or the Hugging Face Hub."""

    if output_dir is None and repo_id is None:
        raise ValueError("Provide at least one export target: output_dir or repo_id")

    datasets = _datasets()
    dataset = materialize_benchmark_split(
        dataset_root=dataset_root,
        split_kind=split_kind,
        include_obs=include_obs,
        sample_ids=sample_ids,
        limit=limit,
        region_manifest_path=region_manifest_path,
        cells_per_group=cells_per_group,
        num_groups=num_groups,
        with_replacement=with_replacement,
        sample_weight_mode=sample_weight_mode,
        stratify_obs_column=stratify_obs_column,
        block_shape=block_shape,
        stride=stride,
        min_cells=min_cells,
        max_cells=max_cells,
        seed=seed,
    )
    dataset_dict = datasets.DatasetDict({str(split_name): dataset})

    manifest: dict[str, Any] = {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "split_kind": str(split_kind),
        "split_name": str(split_name),
        "config_name": str(config_name),
        "include_obs": bool(include_obs),
        "sample_ids": None if sample_ids is None else [str(item) for item in sample_ids],
        "limit": limit,
        "region_manifest_path": None
        if region_manifest_path is None
        else str(Path(region_manifest_path).expanduser().resolve()),
        "cells_per_group": int(cells_per_group),
        "num_groups": int(num_groups),
        "with_replacement": bool(with_replacement),
        "sample_weight_mode": str(sample_weight_mode),
        "stratify_obs_column": stratify_obs_column,
        "block_shape": None if block_shape is None else [float(item) for item in block_shape],
        "stride": None if stride is None else [float(item) for item in stride],
        "min_cells": int(min_cells),
        "max_cells": int(max_cells),
        "seed": int(seed),
        "num_rows": len(dataset),
    }

    if output_dir is not None:
        export_root = Path(output_dir).expanduser().resolve()
        export_root.mkdir(parents=True, exist_ok=True)
        dataset_path = export_root / "hf_dataset"
        dataset_dict.save_to_disk(str(dataset_path))
        spec_path = export_root / "export_spec.json"
        spec_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksum_manifest = write_checksum_manifest(export_root)
        manifest["output_dir"] = str(export_root)
        manifest["save_to_disk_path"] = str(dataset_path)
        manifest["checksum_manifest_json"] = str(export_root / "checksum_manifest.json")
        manifest["checksum_n_files"] = checksum_manifest["n_files"]

    if repo_id is not None:
        huggingface_hub = _huggingface_hub()
        api = huggingface_hub.HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True,
        )
        commit_info = dataset.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            split=split_name,
            commit_message=commit_message,
            private=private,
            token=token,
            create_pr=create_pr,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
        )
        manifest["repo_id"] = repo_id
        manifest["repo_url"] = f"https://huggingface.co/datasets/{repo_id}"
        manifest["hub_commit_url"] = getattr(commit_info, "commit_url", None)
        manifest["hub_commit_oid"] = getattr(commit_info, "oid", None)

    return manifest
