# Data and Checkpoint Layout

The public repository is path-agnostic. Set environment variables or pass explicit command-line arguments rather than editing scripts.

## Recommended Environment Variables

| Variable | Meaning |
| --- | --- |
| `OMNICELL_NVU_ROOT` | Project working directory containing local data, checkpoints, outputs, and this code checkout. |
| `DATA_ROOT` | Root directory for raw H5AD files or converted CSR memmap datasets. |
| `CHECKPOINT_ROOT` | Root directory containing legacy OmniCell checkpoints or converted Hugging Face checkpoints. |
| `OUTPUT_ROOT` | Destination for embeddings, scorecards, source tables, and generated figures. |
| `PYTHON` | Python executable used by scheduler wrappers; defaults to `python` when unset. |
| `CONDA_SH` | Optional conda activation script used by cluster job wrappers. |

## CSR Memmap Dataset Contract

Continual pretraining uses the `cellfm-datasets` CSR memmap format. A dataset directory should provide a manifest similar to:

```json
{
  "format": "cellfm-csr-memmap",
  "samples": [
    {
      "sample_id": "sample_001",
      "matrix_dir": "sample_001/matrix",
      "obs_path": "sample_001/obs.parquet",
      "var_path": "sample_001/var.parquet"
    }
  ]
}
```

See `third_party/cellfm-datasets/docs/FORMAT_SPEC.md` for the full schema.

## Checkpoints

Two checkpoint layouts are supported:

1. Legacy OmniCell checkpoints loaded through `--legacy-checkpoint-dir`.
2. Hugging Face-style checkpoints created by `scripts/convert_legacy_checkpoint.py` and loaded with `from_pretrained`.

For full model execution, make sure the vocabulary asset in `assets/vocab/Vocabulary.json` matches the tokenizer/gene vocabulary used during checkpoint training.

## Outputs Not Tracked by Git

The following output classes are intentionally ignored by `.gitignore`:

- raw H5AD files and converted memmaps;
- model checkpoints and safetensor files;
- embedding arrays and parquet metadata;
- benchmark scorecards and generated figure panels;
- cluster scheduler logs.
