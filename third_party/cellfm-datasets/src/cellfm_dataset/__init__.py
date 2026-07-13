"""Memmap-backed dataset adapters for cell foundation model pretraining."""

from .benchmarking import benchmark_group_sampling, benchmark_streaming_group_dataset
from .checksum import build_checksum_manifest, compute_file_checksum, verify_checksum_manifest, write_checksum_manifest
from .collate import CellFMRegionDataCollator, csr_region_collate
from .convert import convert_h5ad_manifest, convert_one_sample, resolve_gene_vocab
from .distributed import DistributedContext, DistributedRuntime
from .hf import load_hf_cell_dataset, load_hf_group_dataset, load_hf_region_dataset
from .hub import export_benchmark_split, materialize_benchmark_split
from .memmap import CSRMemmapSample, MemmapDataset
from .sampling import GroupSpec, ManifestRegionSampler, RandomCellSampler, SpatialBlockSampler
from .schema import DatasetMetadata, SampleMetadata
from .synthetic import (
    PUBLIC_BENCHMARK_PRESETS,
    generate_public_synthetic_benchmark_data,
    generate_synthetic_singlecell_cohort,
    generate_synthetic_spatial_cohort,
    resolve_public_benchmark_preset,
)
from .validate import validate_dataset, validate_region_manifest_examples
from .common import DEFAULT_EXPRESSION_TRANSFORM, SUPPORTED_EXPRESSION_TRANSFORMS, apply_expression_transform

__all__ = [
    "CellFMRegionDataCollator",
    "CSRMemmapSample",
    "DEFAULT_EXPRESSION_TRANSFORM",
    "DatasetMetadata",
    "DistributedContext",
    "DistributedRuntime",
    "GroupSpec",
    "ManifestRegionSampler",
    "MemmapDataset",
    "PUBLIC_BENCHMARK_PRESETS",
    "RandomCellSampler",
    "SampleMetadata",
    "SpatialBlockSampler",
    "SUPPORTED_EXPRESSION_TRANSFORMS",
    "apply_expression_transform",
    "benchmark_group_sampling",
    "benchmark_streaming_group_dataset",
    "build_checksum_manifest",
    "compute_file_checksum",
    "convert_h5ad_manifest",
    "convert_one_sample",
    "csr_region_collate",
    "export_benchmark_split",
    "generate_public_synthetic_benchmark_data",
    "generate_synthetic_singlecell_cohort",
    "generate_synthetic_spatial_cohort",
    "load_hf_cell_dataset",
    "load_hf_group_dataset",
    "load_hf_region_dataset",
    "materialize_benchmark_split",
    "resolve_public_benchmark_preset",
    "resolve_gene_vocab",
    "validate_dataset",
    "validate_region_manifest_examples",
    "verify_checksum_manifest",
    "write_checksum_manifest",
]
