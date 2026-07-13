# cellfm-datasets

English | [简体中文](README.zh-CN.md)

`cellfm-datasets` is a pip-installable data infrastructure package for cell foundation model pretraining. It supports both single-cell and spatial transcriptomics, converts H5AD cohorts into a compact CSR-memmap format, and exposes the result through Hugging Face `Dataset` / `IterableDataset` interfaces.

The package is designed as a reusable data layer rather than a project-specific loader. The core abstraction is a sampled cell group, which allows the same storage backend to serve:

- non-spatial single-cell pretraining
- spatial block or region pretraining
- mixed-modality cell-group pretraining
- Hugging Face, PyTorch, or custom training loops

## Highlights

- Supports both single-cell and spatial transcriptomics in one storage protocol.
- Handles samples with or without coordinates.
- Preserves sparse expression in CSR form end to end.
- Supports `obs` metadata persistence and runtime retrieval.
- Supports random sampling and custom block-based sampling.
- Supports distributed sharding for cell, group, and region datasets.
- Includes public synthetic benchmark generators, checksum manifests, and Hugging Face Hub export utilities.
- Ships as an installable package with CLI, tests, and wheel/sdist build support.

## Package Scope

`cellfm-datasets` separates the data system into four layers:

1. Canonical storage:
   `gene_vocab.txt`, `dataset_manifest.json`, per-sample CSR arrays, optional `obs.parquet`
2. Random-access runtime:
   zero-copy memmap readers for cells and sampled groups
3. Group samplers:
   random-cell groups, manifest-backed regions, and spatial blocks
4. Framework adapters:
   Hugging Face datasets and optional Torch collators

This separation is what makes the package suitable for research artifacts and for long-lived pretraining infrastructure.

## Installation

Core package:

```bash
pip install cellfm-datasets
```

This installs the base runtime only: `numpy`, `pandas`, and `scipy`. It is enough for core memmap readers, metadata inspection, and checksum tooling, but it does not install optional conversion or Hugging Face dependencies.

With H5AD conversion support:

```bash
pip install "cellfm-datasets[convert]"
```

Use `convert` when you need to:

- convert `.h5ad` cohorts into the memmap dataset layout
- persist `obs.parquet` tables during conversion
- run H5AD-based synthetic benchmark generators

If your dataset includes `obs.parquet`, installing `convert` is also the simplest way to ensure a parquet engine is available.

With Hugging Face support:

```bash
pip install "cellfm-datasets[hf]"
```

Use `hf` when you need to:

- expose datasets through `load_hf_cell_dataset`, `load_hf_group_dataset`, or `load_hf_region_dataset`
- export selected benchmark splits with `cellfm-dataset export-hf-benchmark-split`
- push materialized splits to the Hugging Face Hub

`hf` is not required for the core memmap storage layer itself. It only adds the Hugging Face `datasets` and Hub integration stack.

With public benchmark and Hub-export support:

```bash
pip install "cellfm-datasets[benchmark]"
```

`benchmark` is the most complete end-user install. It bundles the optional dependencies needed for public synthetic benchmark generation, Hugging Face export, and benchmark scripts.

With conversion, HF support, and local development tools:

```bash
pip install "cellfm-datasets[convert,hf,dev]"
```

## Canonical Dataset Layout

```text
dataset_root/
  gene_vocab.txt
  dataset_manifest.json
  sample_a/
    coords.npy
    indptr.npy
    indices.npy
    values.npy
    raw_cell_sums.npy
    present_gene_ids.npy
    obs_names.npy
    obs.parquet                 # optional
    metadata.json
  sample_b/
    ...
```

Notes:

- `coords.npy` may have shape `(n_cells, 0)` for non-spatial single-cell samples.
- `obs.parquet` is optional and stores selected `adata.obs` columns.
- Group sampling is computed at runtime from stored cells plus a sampler.

## Core Concepts

### 1. Cell Store

Each sample is stored as sparse CSR arrays:

- `indptr.npy`
- `indices.npy`
- `values.npy`

This keeps storage compact and avoids dense expansion for high-dimensional transcriptomics matrices.

### 2. Optional Coordinates

Spatial samples can store 2D or 3D coordinates. Single-cell samples without spatial context are also supported.

- spatial sample: `coord_dim = 2` or `3`
- single-cell sample: `coord_dim = 0`

### 3. `obs` Metadata

Selected columns from `adata.obs` can be persisted and returned at runtime.

Typical examples:

- `donor_id`
- `batch`
- `cell_type`
- `condition`
- `assay`
- `slice_id`

### 4. Pretraining Group

A pretraining group is a subset of cells drawn from one sample. This is the main training unit in the package.

Different modalities use different samplers, but they produce the same group-level output structure:

- sparse expression for the selected cells
- optional coordinates
- optional `obs`
- sampling metadata

## Conversion

### Manifest Format

The converter accepts a JSON manifest. Both list and dict styles are supported.

Example:

```json
[
  {
    "h5ad": "/abs/path/sample_a.h5ad",
    "sample_id": "sample_a",
    "z_column": "section_id",
    "obs_columns": ["donor_id", "cell_type", "batch"]
  },
  {
    "h5ad": "/abs/path/sample_sc.h5ad",
    "sample_id": "sample_sc",
    "obs_columns": ["donor_id", "cell_type"]
  }
]
```

Important fields:

- `h5ad`: input file path
- `sample_id`: stable sample identifier
- `z_column`: optional coordinate column to append as `z`
- `obs_columns`: optional `obs` columns to persist
- `expression_transform`: optional expression preprocessing strategy

### CLI Example

```bash
cellfm-dataset convert \
  --manifest-json /path/to/h5ad_manifest.json \
  --output-dir /path/to/memmap_dataset \
  --gene-vocab /path/to/gene_vocab.txt \
  --obs-column donor_id cell_type batch \
  --expression-transform normalize_total_log1p \
  --normalize-target-sum 10000 \
  --overwrite
```

If you want conversion to fail when coordinates are missing:

```bash
cellfm-dataset convert \
  --manifest-json /path/to/h5ad_manifest.json \
  --output-dir /path/to/memmap_dataset \
  --gene-vocab /path/to/gene_vocab.txt \
  --require-coords
```

## Expression Preprocessing

Expression preprocessing is configurable at conversion time.

Currently supported:

- `normalize_total_log1p`
  align to the shared gene vocabulary, apply per-cell `normalize_total`, then `log1p`
- `none`
  align to the shared gene vocabulary and write values without additional expression normalization

CLI example without expression normalization:

```bash
cellfm-dataset convert \
  --manifest-json /path/to/h5ad_manifest.json \
  --output-dir /path/to/memmap_dataset \
  --gene-vocab /path/to/gene_vocab.txt \
  --expression-transform none
```

Per-sample manifest override:

```json
[
  {
    "h5ad": "/abs/path/sample_a.h5ad",
    "sample_id": "sample_a",
    "expression_transform": "normalize_total_log1p"
  },
  {
    "h5ad": "/abs/path/sample_b.h5ad",
    "sample_id": "sample_b",
    "expression_transform": "none"
  }
]
```

## Validation and Inspection

Inspect dataset metadata:

```bash
cellfm-dataset inspect --dataset-dir /path/to/memmap_dataset
```

Validate dataset structure:

```bash
cellfm-dataset validate --dataset-dir /path/to/memmap_dataset
```

Validate dataset structure plus a region manifest:

```bash
cellfm-dataset validate \
  --dataset-dir /path/to/memmap_dataset \
  --region-manifest /path/to/region_manifest.parquet
```

## Sampling Modes

### RandomCellSampler

Use this for single-cell pretraining or generic random groups.

```python
from cellfm_dataset import RandomCellSampler, load_hf_group_dataset

sampler = RandomCellSampler(
    cells_per_group=256,
    num_groups=10000,
    seed=7,
    sample_weight_mode="n_cells",
)

dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
    include_obs=True,
)
```

Capabilities:

- fixed-size random cell groups
- optional replacement
- sample weighting
- optional `obs`-based stratification

### ManifestRegionSampler

Use this when you already have a region manifest generated by another pipeline.

```python
from cellfm_dataset import load_hf_region_dataset

dataset = load_hf_region_dataset(
    dataset_root="/path/to/memmap_dataset",
    region_manifest_path="/path/to/region_manifest.parquet",
    streaming=True,
    include_obs=True,
)
```

### SpatialBlockSampler

Use this for spatial transcriptomics when training should operate on blocks instead of predefined regions.

Grid-based blocks:

```python
from cellfm_dataset import SpatialBlockSampler, load_hf_group_dataset

sampler = SpatialBlockSampler(
    block_shape=(128.0, 128.0, 1.0),
    stride=(64.0, 64.0, 1.0),
    min_cells=64,
    max_cells=512,
    seed=7,
)

dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
)
```

Custom block manifest:

```python
from cellfm_dataset import SpatialBlockSampler, load_hf_group_dataset

sampler = SpatialBlockSampler(
    block_manifest_path="/path/to/block_manifest.parquet",
    min_cells=64,
    max_cells=512,
)

dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
)
```

## Hugging Face Adapters

### Cell-Level Dataset

```python
from cellfm_dataset import load_hf_cell_dataset

dataset = load_hf_cell_dataset(
    dataset_root="/path/to/memmap_dataset",
    streaming=False,
    include_obs=True,
)

print(dataset[0]["sample_id"])
print(dataset[0]["gene_indices"])
print(dataset[0]["obs_json"])
```

### Group-Level Dataset

```python
from cellfm_dataset import RandomCellSampler, load_hf_group_dataset

sampler = RandomCellSampler(cells_per_group=128, num_groups=1000, seed=0)

dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
    include_obs=True,
)

first = next(iter(dataset))
print(first["group_type"])
print(first["source_cell_indices"])
print(first["sampling_metadata_json"])
```

### Region-Level Dataset

This is a convenience wrapper around `ManifestRegionSampler`.

```python
from cellfm_dataset import load_hf_region_dataset

dataset = load_hf_region_dataset(
    dataset_root="/path/to/memmap_dataset",
    region_manifest_path="/path/to/region_manifest.parquet",
    streaming=True,
)
```

## Torch Collation

If you are training with a PyTorch dataloader and want batched sparse groups:

```python
from torch.utils.data import DataLoader
from cellfm_dataset import CellFMRegionDataCollator, RandomCellSampler, load_hf_group_dataset

sampler = RandomCellSampler(cells_per_group=256, num_groups=10000, seed=7)
dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
)
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    collate_fn=CellFMRegionDataCollator(),
)
```

The collator works on group-style records and preserves flattened sparse arrays.

Important constraint:

- `CellFMRegionDataCollator` expects every item in one batch to have the same number of cells.
- It is a good default for `RandomCellSampler` and other fixed-size group samplers.
- It is not a safe default for variable-size spatial blocks. Use a custom collator or enforce fixed group size before batching.

## Public API Summary

Main conversion and validation functions:

- `convert_h5ad_manifest`
- `convert_one_sample`
- `validate_dataset`
- `validate_region_manifest_examples`

Main runtime classes:

- `MemmapDataset`
- `CSRMemmapSample`

Main samplers:

- `RandomCellSampler`
- `ManifestRegionSampler`
- `SpatialBlockSampler`

Distributed runtime helper:

- `DistributedContext`
- `DistributedRuntime`

Main HF adapters:

- `load_hf_cell_dataset`
- `load_hf_group_dataset`
- `load_hf_region_dataset`

## Design Principles

- Deterministic:
  vocabulary order, metadata serialization, and conversion semantics are explicit.
- Sparse-native:
  expression stays sparse until the final batch assembly step.
- Modality-agnostic:
  one storage backend supports both single-cell and spatial data.
- Research-ready:
  package boundaries, manifests, tests, and build artifacts are explicit.

## Distributed Training

Distributed support is implemented through `DistributedContext`. It applies to:

- `load_hf_cell_dataset`
- `load_hf_group_dataset`
- `load_hf_region_dataset`

It shards the stream by `rank`, `world_size`, and dataloader worker id. Random samplers are also epoch-aware.

Example:

```python
from cellfm_dataset import DistributedContext, RandomCellSampler, load_hf_group_dataset

sampler = RandomCellSampler(
    cells_per_group=256,
    num_groups=100000,
    seed=7,
)

dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
    distributed_context=DistributedContext(
        rank=0,
        world_size=8,
        epoch=0,
    ),
)
```

Semantics:

- each rank receives a disjoint shard
- dataloader workers are also sharded
- `epoch` changes the sampling stream for random samplers while remaining reproducible

If you want `set_epoch()` style integration for multi-epoch training:

```python
from cellfm_dataset import DistributedRuntime, RandomCellSampler, load_hf_group_dataset

runtime = DistributedRuntime.from_environment(epoch=0)
sampler = RandomCellSampler(cells_per_group=256, num_groups=100000, seed=7)

dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
    distributed_context=runtime,
)

for epoch in range(num_epochs):
    runtime.set_epoch(epoch)
    for item in dataset:
        ...
```

### Dataloader + DDP

For actual PyTorch training, combine the streaming dataset with `DataLoader`.

```python
from torch.utils.data import DataLoader
from cellfm_dataset import (
    CellFMRegionDataCollator,
    DistributedRuntime,
    RandomCellSampler,
    load_hf_group_dataset,
)

runtime = DistributedRuntime.from_environment(epoch=0)
sampler = RandomCellSampler(cells_per_group=256, num_groups=100000, seed=7)
dataset = load_hf_group_dataset(
    dataset_root="/path/to/memmap_dataset",
    sampler=sampler,
    streaming=True,
    distributed_context=runtime,
)
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    collate_fn=CellFMRegionDataCollator(),
)

for epoch in range(num_epochs):
    runtime.set_epoch(epoch)
    for batch in loader:
        ...
```

Semantics:

- `num_workers > 0` uses PyTorch worker processes
- worker ids are inferred automatically and sharded into disjoint substreams
- different DDP ranks are also sharded when you pass `distributed_context`
- if you omit `distributed_context` in DDP, different ranks can read overlapping groups

## Examples and Benchmarks

Distributed examples:

- `examples/dataloader_random_groups.py`
- `examples/distributed_random_groups_torchrun.py`
- `examples/distributed_random_groups_dataloader_torchrun.py`
- `examples/distributed_spatial_blocks_torchrun.py`

Benchmark script:

- `benchmarks/benchmark_sampling.py`
- `benchmarks/generate_public_synthetic_data.py`
- `benchmarks/run_public_synthetic_suite.py`

## Documentation

- Chinese README: [README.zh-CN.md](README.zh-CN.md)
- English benchmarks guide: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- Chinese benchmarks guide: [docs/BENCHMARKS.zh-CN.md](docs/BENCHMARKS.zh-CN.md)
- English user guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- Chinese user guide: [docs/USER_GUIDE.zh-CN.md](docs/USER_GUIDE.zh-CN.md)
- English format spec: [docs/FORMAT_SPEC.md](docs/FORMAT_SPEC.md)
- Chinese format spec: [docs/FORMAT_SPEC.zh-CN.md](docs/FORMAT_SPEC.zh-CN.md)
- English tutorial: [docs/TUTORIAL.md](docs/TUTORIAL.md)
- Chinese tutorial: [docs/TUTORIAL.zh-CN.md](docs/TUTORIAL.zh-CN.md)
