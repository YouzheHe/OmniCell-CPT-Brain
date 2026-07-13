# User Guide

English | [简体中文](USER_GUIDE.zh-CN.md)

This document is the practical usage manual for `cellfm-datasets`. It focuses on:

- input data requirements
- conversion manifests
- output dataset layout
- CLI parameter reference
- Python API usage
- sampling modes
- distributed usage
- common pitfalls

## 1. What This Package Expects

`cellfm-datasets` starts from one or more H5AD files and converts them into a reusable memmap-backed dataset.

Each input H5AD is expected to contain:

- `adata.X`
  expression matrix, dense or sparse
- `adata.var_names`
  gene names
- `adata.obs_names`
  cell identifiers

Optional fields:

- `adata.obsm["spatial"]`
  2D or 3D coordinates
- `adata.obs["x"]`, `adata.obs["y"]`
  fallback coordinate columns
- `adata.obs["z"]`
  optional third coordinate
- additional `adata.obs[...]`
  metadata columns such as donor, batch, cell type, condition, assay

### Non-spatial single-cell input

If no coordinates are present and `--require-coords` is not set, the package will still accept the sample and store it as a non-spatial sample with `coord_dim = 0`.

### Spatial input

If coordinates are present, they are stored and can later be used by:

- `ManifestRegionSampler`
- `SpatialBlockSampler`
- any custom spatial sampling logic you build on top of `MemmapDataset`

## 2. Input Manifest Format

The converter accepts a JSON manifest describing the H5AD files to convert.

### Supported styles

List style:

```json
[
  {
    "h5ad": "/abs/path/sample_a.h5ad",
    "sample_id": "sample_a"
  }
]
```

Wrapped dict style:

```json
{
  "items": [
    {
      "h5ad": "/abs/path/sample_a.h5ad",
      "sample_id": "sample_a"
    }
  ]
}
```

### Per-sample fields

- `h5ad`
  absolute or user-resolvable path to the input H5AD
- `sample_id`
  stable sample identifier
  if omitted, the converter uses the H5AD filename stem
- `z_column`
  optional column in `adata.obs` appended as the third coordinate
- `coord_axis_names`
  optional axis-name override, for example `["x", "y", "z"]`
- `obs_columns`
  optional list of `adata.obs` columns to persist
- `expression_transform`
  optional per-sample expression preprocessing strategy
  supported values:
  - `normalize_total_log1p`
  - `none`

### Example

```json
[
  {
    "h5ad": "/data/sample_spatial_a.h5ad",
    "sample_id": "sample_spatial_a",
    "z_column": "slice_id",
    "obs_columns": ["donor_id", "cell_type", "batch"],
    "expression_transform": "normalize_total_log1p"
  },
  {
    "h5ad": "/data/sample_sc_b.h5ad",
    "sample_id": "sample_sc_b",
    "obs_columns": ["donor_id", "cell_type"],
    "expression_transform": "none"
  }
]
```

## 3. Output Dataset Layout

After conversion, the dataset directory looks like this:

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
    obs.parquet
    metadata.json
  sample_b/
    ...
```

### Root-level files

- `gene_vocab.txt`
  final ordered shared gene vocabulary
- `dataset_manifest.json`
  dataset-level metadata

### Per-sample files

- `coords.npy`
  shape `(n_cells, coord_dim)`
  for non-spatial samples this may be `(n_cells, 0)`
- `indptr.npy`, `indices.npy`, `values.npy`
  CSR sparse expression arrays
- `raw_cell_sums.npy`
  raw per-cell library size before expression transform
- `present_gene_ids.npy`
  unique gene indices present in that sample
- `obs_names.npy`
  cell IDs
- `obs.parquet`
  optional persisted `adata.obs` subset
- `metadata.json`
  sample-level metadata

## 4. Expression Preprocessing

Expression preprocessing happens at conversion time.

### Supported transforms

- `normalize_total_log1p`
  align to the shared vocabulary, apply per-cell `normalize_total`, then apply `log1p`
- `none`
  align to the shared vocabulary and keep expression values as-is

### Important note

Coordinate normalization is not the same thing as expression preprocessing.

- expression preprocessing happens during conversion and affects `values.npy`
- coordinate normalization happens at read time when loading a group or region

## 5. CLI Reference

The CLI entry point is:

```bash
cellfm-dataset <command> ...
```

### 5.1 `convert`

Convert H5AD files into the canonical memmap dataset.

```bash
cellfm-dataset convert \
  --manifest-json /path/to/manifest.json \
  --output-dir /path/to/output \
  --gene-vocab /path/to/gene_vocab.txt
```

Parameters:

- `--manifest-json`
  required
  JSON manifest describing H5AD inputs
- `--output-dir`
  required
  output dataset directory
- `--gene-vocab`
  optional
  path to one vocabulary file
- `--union-gene-list`
  optional
  one or more gene-list files whose ordered union becomes the final vocabulary
- `--obs-column`
  optional
  global `obs` columns to persist for every sample
  use `*` to store all `obs` columns
- `--expression-transform`
  optional
  default: `normalize_total_log1p`
  choices:
  - `normalize_total_log1p`
  - `none`
- `--normalize-target-sum`
  optional
  default: `10000`
  only used when `expression_transform=normalize_total_log1p`
- `--require-coords`
  optional flag
  if set, conversion fails when a sample has no usable coordinates
- `--overwrite`
  optional flag
  overwrite existing per-sample outputs

### 5.2 `inspect`

Print dataset metadata as JSON.

```bash
cellfm-dataset inspect --dataset-dir /path/to/output
```

Parameters:

- `--dataset-dir`
  required
  converted dataset root

### 5.3 `validate`

Validate the converted dataset structure and optionally replay values from source H5AD.

```bash
cellfm-dataset validate --dataset-dir /path/to/output
```

Parameters:

- `--dataset-dir`
  required
  converted dataset root
- `--region-manifest`
  optional
  region manifest to test region reconstruction
- `--skip-source-compare`
  optional flag
  skip reloading source H5AD and replaying values
- `--max-regions-to-check`
  optional
  default: `8`
- `--rtol`
  optional
  default: `1e-5`
- `--atol`
  optional
  default: `1e-6`

## 6. Python API Reference

This section focuses on the main entry points users call directly.

### 6.1 Conversion

#### `convert_h5ad_manifest(...)`

Convert one manifest into one memmap dataset.

Key parameters:

- `manifest_json`
  manifest path
- `output_dir`
  output dataset root
- `gene_vocab_path`
  optional path to one vocabulary file
- `union_gene_lists`
  optional tuple of gene-list files
- `default_obs_columns`
  optional default `obs` columns
- `expression_transform`
  default expression transform used when a sample does not override it
- `normalize_target_sum`
  target library size for `normalize_total_log1p`
- `require_coords`
  whether coordinates are mandatory
- `overwrite`
  whether existing files may be overwritten

#### `convert_one_sample(...)`

Convert a single sample into one subdirectory under the dataset root.

This function is mostly useful when you want to build a custom pipeline or batch scheduler.

### 6.2 Runtime Store

#### `MemmapDataset(dataset_root=...)`

Open one converted dataset.

Useful attributes:

- `metadata`
- `gene_vocab`
- `samples`

Useful methods:

- `iter_cells(...)`
- `iter_regions(...)`
- `build_group(...)`
- `load_region_manifest(...)`

#### `CSRMemmapSample(...)`

Open one sample inside the dataset.

Useful methods:

- `fetch_cell(...)`
- `fetch_rows(...)`
- `iter_cells(...)`
- `load_obs_frame(...)`

### 6.3 Sampling

#### `RandomCellSampler`

Used for random fixed-size cell groups.

Parameters:

- `cells_per_group`
  number of cells in one group
- `num_groups`
  number of groups to generate
- `seed`
  base seed
- `with_replacement`
  whether cell indices can repeat
- `sample_ids`
  optional sample subset
- `sample_weight_mode`
  choices:
  - `uniform`
  - `n_cells`
- `stratify_obs_column`
  optional `obs` column used for pool restriction

#### `ManifestRegionSampler`

Used when you already have a `region_manifest.parquet`.

Parameters:

- `region_manifest_path`
  one path or multiple manifest paths

#### `SpatialBlockSampler`

Used for spatial block-based sampling.

Parameters:

- `block_shape`
  spatial block size
- `stride`
  stride between neighboring blocks
- `block_manifest_path`
  optional external block manifest
- `sample_ids`
  optional sample subset
- `min_cells`
  minimum cells required in a block
- `max_cells`
  cap and downsample if a block is too dense
- `shuffle`
  whether to shuffle generated blocks
- `seed`
  seed used for block-level downsampling and shuffle

### 6.4 Hugging Face Adapters

#### `load_hf_cell_dataset(...)`

Return a Hugging Face dataset where each example is one cell.

Important parameters:

- `dataset_root`
- `sample_ids`
- `streaming`
- `include_obs`
- distributed arguments:
  - `distributed_context`
  - `rank`
  - `world_size`
  - `epoch`

#### `load_hf_group_dataset(...)`

Return a Hugging Face dataset where each example is one sampled group.

Important parameters:

- `dataset_root`
- `sampler`
- `normalize_coords`
- `streaming`
- `include_obs`
- `limit`
- distributed arguments:
  - `distributed_context`
  - `rank`
  - `world_size`
  - `epoch`
  - `infer_worker`

#### `load_hf_region_dataset(...)`

Convenience wrapper around `ManifestRegionSampler`.

Important parameters:

- `dataset_root`
- `region_manifest_path`
- `normalize_coords`
- `streaming`
- `include_obs`
- `limit`
- distributed arguments:
  - `distributed_context`
  - `rank`
  - `world_size`
  - `epoch`
  - `infer_worker`

### 6.5 Distributed Helpers

#### `DistributedContext`

Immutable one-shot sharding state.

Parameters:

- `rank`
- `world_size`
- `worker_id`
- `num_workers`
- `epoch`
- `shard_mode`

#### `DistributedRuntime`

Mutable training-loop helper.

Useful methods:

- `from_environment(...)`
- `set_epoch(...)`
- `to_context()`

Use `DistributedRuntime` if you want `set_epoch()` style training.

### 6.6 PyTorch DataLoader Integration

The Hugging Face streaming adapters can be passed directly into `torch.utils.data.DataLoader`.

Example:

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
    pin_memory=True,
    collate_fn=CellFMRegionDataCollator(),
)
```

Semantics:

- `streaming=True` returns a Hugging Face `IterableDataset`
- `num_workers > 0` starts PyTorch worker processes
- worker ids are inferred automatically when `infer_worker=True`
- each worker gets a disjoint shard of the stream

Important constraint:

- `CellFMRegionDataCollator` requires the same number of cells in every item within one batch
- this fits `RandomCellSampler` and other fixed-size group samplers
- this does not fit variable-size spatial blocks unless you write a custom collator

### 6.7 DDP Training Pattern

For multi-GPU training with `torchrun`, use `DistributedRuntime`.

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

Notes:

- `DistributedRuntime.from_environment(...)` reads `RANK/WORLD_SIZE` from `torchrun`
- each rank receives a disjoint shard
- dataloader workers further shard within each rank
- if `distributed_context` is omitted, different ranks can read overlapping records

## 7. Typical Workflows

### Workflow A: non-spatial single-cell pretraining

1. Convert H5AD files
2. Use `RandomCellSampler`
3. Call `load_hf_group_dataset(...)`
4. Train on random groups

### Workflow B: spatial region pretraining with an external region manifest

1. Convert H5AD files
2. Generate `region_manifest.parquet` externally
3. Call `load_hf_region_dataset(...)`
4. Train on region groups

### Workflow C: spatial block pretraining

1. Convert H5AD files
2. Define block size and stride
3. Use `SpatialBlockSampler`
4. Call `load_hf_group_dataset(...)`

### Workflow D: distributed multi-epoch training

1. Build `DistributedRuntime.from_environment(...)`
2. Build sampler once
3. Build streaming HF dataset once
4. Wrap the dataset with `DataLoader`
5. Call `runtime.set_epoch(epoch)` at the start of each epoch

## 8. Common Pitfalls

- Mixed coordinate layouts across samples
  group loading may fail if your downstream code assumes one fixed coordinate dimension
- Missing `obs` columns
  requesting an unavailable `obs` column at conversion time raises an error
- Very small samples with large `cells_per_group`
  `RandomCellSampler` may fail unless `with_replacement=True`
- `expression_transform=none`
  downstream models may still expect normalized or log-transformed values
- Distributed training without context
  multiple ranks may read overlapping samples unless you pass distributed sharding metadata
- Variable-size groups with the default collator
  `CellFMRegionDataCollator` assumes a fixed number of cells per item and will fail on mixed-size batches

## 9. Related Documents

- Main overview: [README.md](../README.md)
- Benchmarks and Hub export: [BENCHMARKS.md](BENCHMARKS.md)
- Format specification: [FORMAT_SPEC.md](FORMAT_SPEC.md)
- Tutorial: [TUTORIAL.md](TUTORIAL.md)
- Chinese user guide: [USER_GUIDE.zh-CN.md](USER_GUIDE.zh-CN.md)
- Research-only design notes now live outside the publishable package tree.
