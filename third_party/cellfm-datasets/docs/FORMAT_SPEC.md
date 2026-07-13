# Dataset Format Specification

English | [简体中文](FORMAT_SPEC.zh-CN.md)

This document defines the on-disk protocol used by `cellfm-datasets`.

It is intended for:

- artifact evaluation
- interoperability with external preprocessing tools
- long-term storage compatibility
- custom loaders outside the reference Python package

## 1. Scope

The format stores a cohort of samples for cell foundation model pretraining.

Each sample contains:

- sparse expression matrix in CSR form
- optional spatial coordinates
- optional `obs` sidecar metadata
- sample-level metadata

The root directory contains:

- one shared gene vocabulary
- one dataset manifest
- one subdirectory per sample

## 2. Root Directory Layout

```text
dataset_root/
  gene_vocab.txt
  dataset_manifest.json
  sample_1/
    ...
  sample_2/
    ...
```

### Required root files

#### `gene_vocab.txt`

One gene symbol per line.

Requirements:

- ordered
- deduplicated
- no empty lines in the final effective vocabulary

The integer gene index used in sample CSR arrays is the zero-based line number in this file.

#### `dataset_manifest.json`

JSON object containing dataset-level metadata.

Required fields:

- `dataset_dir`
- `input_manifest_json`
- `n_genes`
- `expression_transform`
- `normalize_target_sum`
- `samples`

Recommended fields:

- `gene_vocab_paths`

Example:

```json
{
  "dataset_dir": "/abs/path/to/dataset_root",
  "input_manifest_json": "/abs/path/to/input_manifest.json",
  "n_genes": 32768,
  "expression_transform": "normalize_total_log1p",
  "normalize_target_sum": 10000.0,
  "samples": [
    {
      "sample_id": "sample_a",
      "source_h5ad": "/abs/path/sample_a.h5ad",
      "z_column": "slice_id",
      "n_cells": 120000,
      "n_genes": 32768,
      "nnz": 54321000,
      "coord_dim": 3,
      "coord_axis_names": ["x", "y", "z"],
      "n_nonfinite_coord_rows": 0,
      "expression_transform": "normalize_total_log1p",
      "normalize_target_sum": 10000.0,
      "kept_source_genes": 20543,
      "dropped_source_genes": 123,
      "n_present_genes": 18002,
      "obs_columns": ["donor_id", "cell_type", "batch"]
    }
  ]
}
```

## 3. Sample Directory Layout

Each sample lives in one subdirectory:

```text
dataset_root/
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
```

### Required sample files

#### `coords.npy`

NumPy array with shape:

- spatial sample: `(n_cells, 2)` or `(n_cells, 3)`
- non-spatial sample: `(n_cells, 0)`

Type:

- `float32`

#### `indptr.npy`

CSR row pointer array.

Shape:

- `(n_cells + 1,)`

Type:

- `int64`

Constraints:

- `indptr[0] == 0`
- `indptr[-1] == nnz`
- non-decreasing

#### `indices.npy`

CSR column index array.

Shape:

- `(nnz,)`

Type:

- `int32`

Constraints:

- all values in `[0, n_genes)`

#### `values.npy`

CSR data array.

Shape:

- `(nnz,)`

Type:

- `float32`

Semantics:

- transformed expression values after vocabulary alignment and optional expression preprocessing

#### `raw_cell_sums.npy`

Per-cell sum before expression transformation.

Shape:

- `(n_cells,)`

Type:

- `float32`

#### `present_gene_ids.npy`

Sorted unique gene IDs present in the sample.

Shape:

- `(n_present_genes,)`

Type:

- `int32`

#### `obs_names.npy`

Per-cell IDs aligned with row order.

Shape:

- `(n_cells,)`

Type:

- string array saved with `numpy.save`

#### `metadata.json`

JSON object describing the sample.

Required fields:

- `sample_id`
- `source_h5ad`
- `n_cells`
- `n_genes`
- `nnz`
- `coord_dim`
- `coord_axis_names`
- `expression_transform`
- `normalize_target_sum`

### Optional sample files

#### `obs.parquet`

Optional columnar sidecar for selected `adata.obs` columns.

Requirements:

- row count must equal `n_cells`
- row order must match the CSR row order
- first column should be `obs_name`
- remaining columns must match `obs_columns` recorded in metadata

## 4. Coordinate Semantics

Coordinates are stored in raw sample space.

The package does not globally normalize coordinates during conversion.

At runtime:

- `raw_coords` are returned as stored
- `coords` may be locally normalized within each sampled group if `normalize_coords=True`

## 5. Expression Semantics

All expression arrays are aligned to the shared root vocabulary.

Supported stored transform labels:

- `normalize_total_log1p`
- `none`
- `mixed`
  allowed only at dataset level when different samples use different transforms

Important rule:

- sample-level metadata is authoritative
- dataset-level metadata is a summary only

## 6. Group Reconstruction Contract

The on-disk protocol stores cells, not groups.

Group-level training examples are reconstructed at runtime from:

- one sample ID
- one sequence of source cell indices
- optional auxiliary manifest metadata

This is why region manifests and block manifests are not embedded into the canonical sample store.

## 7. Region Manifest Contract

For `load_hf_region_dataset(...)` or `ManifestRegionSampler`, the external manifest must contain:

- `sample_id`
- `region_id`
- `source_cell_indices`

Optional fields:

- `region_index`
- `sample_region_index`
- `anchor_dim0`, `anchor_dim1`, ...
- any extra metadata columns

`source_cell_indices` may be:

- a list/array
- a JSON-encoded string list
- a comma-separated string

## 8. Block Manifest Contract

For `SpatialBlockSampler(block_manifest_path=...)`, the external block manifest may describe groups in two ways:

1. direct cell membership
   via `source_cell_indices`
2. coordinate bounds
   via `min_<axis>` / `max_<axis>`

Required:

- `sample_id`

Optional:

- `block_id`
- `sample_block_index`
- `source_cell_indices`
- `min_x`, `max_x`, `min_y`, `max_y`, ...

If coordinate bounds are used, the corresponding sample must have coordinates.

## 9. Compatibility Rules

A compatible reader must:

- use `gene_vocab.txt` as the global gene index mapping
- treat sample metadata as authoritative for sample-specific settings
- preserve row order across:
  - CSR arrays
  - `coords.npy`
  - `raw_cell_sums.npy`
  - `obs_names.npy`
  - `obs.parquet`

A compatible writer must:

- emit all required root and sample files
- satisfy array shape and dtype constraints
- keep manifests and arrays consistent

## 10. Validation Checklist

A dataset is considered structurally valid if:

- all required files exist
- all array shapes match metadata
- CSR invariants hold
- `present_gene_ids == unique(indices)`
- `obs.parquet`, if present, matches recorded columns and row count

For source-level replay validation, a validator may:

- reopen the source H5AD
- realign to the shared vocabulary
- reapply the recorded expression transform
- compare against stored `indptr`, `indices`, `values`, and `raw_cell_sums`
