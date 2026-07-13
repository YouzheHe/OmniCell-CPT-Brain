"""Command line interface for cellfm-datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import normalize_column_names
from .convert import convert_h5ad_manifest
from .checksum import verify_checksum_manifest, write_checksum_manifest
from .hub import export_benchmark_split
from .schema import DatasetMetadata
from .synthetic import generate_public_synthetic_benchmark_data
from .validate import validate_dataset, validate_region_manifest_examples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cellfm-datasets CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert H5AD manifest to CSR memmap.")
    convert_parser.add_argument("--manifest-json", required=True, type=Path)
    convert_parser.add_argument("--output-dir", required=True, type=Path)
    convert_parser.add_argument("--gene-vocab", type=Path, default=None)
    convert_parser.add_argument(
        "--union-gene-list",
        type=Path,
        nargs="*",
        default=None,
        help="One or more gene-list files whose ordered union becomes the shared vocabulary.",
    )
    convert_parser.add_argument(
        "--obs-column",
        type=str,
        nargs="*",
        default=None,
        help="Optional obs columns to persist for every sample. Use '*' to store all obs columns.",
    )
    convert_parser.add_argument(
        "--expression-transform",
        type=str,
        choices=("normalize_total_log1p", "none"),
        default="normalize_total_log1p",
        help="Expression preprocessing applied before writing values.npy.",
    )
    convert_parser.add_argument("--normalize-target-sum", type=float, default=1e4)
    convert_parser.add_argument(
        "--require-coords",
        action="store_true",
        help="Fail conversion when a sample has no usable coordinates.",
    )
    convert_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate a memmap dataset.")
    validate_parser.add_argument("--dataset-dir", required=True, type=Path)
    validate_parser.add_argument("--region-manifest", type=Path, default=None)
    validate_parser.add_argument("--skip-source-compare", action="store_true")
    validate_parser.add_argument("--max-regions-to-check", type=int, default=8)
    validate_parser.add_argument("--rtol", type=float, default=1e-5)
    validate_parser.add_argument("--atol", type=float, default=1e-6)

    inspect_parser = subparsers.add_parser("inspect", help="Print dataset metadata as JSON.")
    inspect_parser.add_argument("--dataset-dir", required=True, type=Path)

    checksum_parser = subparsers.add_parser(
        "checksum-manifest",
        help="Write a checksum manifest for one release bundle or dataset directory.",
    )
    checksum_parser.add_argument("--input-dir", required=True, type=Path)
    checksum_parser.add_argument("--output-json", type=Path, default=None)
    checksum_parser.add_argument("--algorithm", type=str, default="sha256")

    verify_checksum_parser = subparsers.add_parser(
        "verify-checksum-manifest",
        help="Verify one checksum manifest against local files.",
    )
    verify_checksum_parser.add_argument("--manifest-json", required=True, type=Path)
    verify_checksum_parser.add_argument("--root-dir", type=Path, default=None)

    synthetic_parser = subparsers.add_parser(
        "generate-synthetic-benchmark",
        help="Generate public synthetic benchmark H5AD cohorts and manifests.",
    )
    synthetic_parser.add_argument("--output-dir", required=True, type=Path)
    synthetic_parser.add_argument(
        "--mode",
        choices=("singlecell", "spatial", "both"),
        default="both",
    )
    synthetic_parser.add_argument(
        "--scale-preset",
        choices=("smoke", "medium"),
        default="smoke",
    )
    synthetic_parser.add_argument("--seed", type=int, default=7)

    export_parser = subparsers.add_parser(
        "export-hf-benchmark-split",
        help="Materialize and optionally publish a selected benchmark split to the Hugging Face Hub.",
    )
    export_parser.add_argument("--dataset-root", required=True, type=Path)
    export_parser.add_argument(
        "--split-kind",
        required=True,
        choices=("cells", "random_groups", "spatial_blocks", "regions"),
    )
    export_parser.add_argument("--split-name", type=str, default="train")
    export_parser.add_argument("--config-name", type=str, default="default")
    export_parser.add_argument("--output-dir", type=Path, default=None)
    export_parser.add_argument("--repo-id", type=str, default=None)
    export_parser.add_argument("--sample-id", nargs="*", default=None)
    export_parser.add_argument("--include-obs", action="store_true")
    export_parser.add_argument("--limit", type=int, default=None)
    export_parser.add_argument("--region-manifest", type=Path, default=None)
    export_parser.add_argument("--cells-per-group", type=int, default=256)
    export_parser.add_argument("--num-groups", type=int, default=10000)
    export_parser.add_argument("--with-replacement", action="store_true")
    export_parser.add_argument(
        "--sample-weight-mode",
        choices=("uniform", "n_cells"),
        default="n_cells",
    )
    export_parser.add_argument("--stratify-obs-column", type=str, default=None)
    export_parser.add_argument("--block-shape", nargs="+", type=float, default=None)
    export_parser.add_argument("--stride", nargs="+", type=float, default=None)
    export_parser.add_argument("--min-cells", type=int, default=64)
    export_parser.add_argument("--max-cells", type=int, default=512)
    export_parser.add_argument("--seed", type=int, default=7)
    export_parser.add_argument("--token", type=str, default=None)
    export_parser.add_argument("--private", action="store_true")
    export_parser.add_argument("--create-pr", action="store_true")
    export_parser.add_argument("--commit-message", type=str, default=None)
    export_parser.add_argument("--max-shard-size", type=str, default=None)
    export_parser.add_argument("--num-shards", type=int, default=None)
    export_parser.add_argument("--num-proc", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "convert":
        metadata = convert_h5ad_manifest(
            manifest_json=args.manifest_json,
            output_dir=args.output_dir,
            gene_vocab_path=args.gene_vocab,
            union_gene_lists=None
            if args.union_gene_list is None
            else tuple(args.union_gene_list),
            default_obs_columns=normalize_column_names(args.obs_column),
            expression_transform=args.expression_transform,
            normalize_target_sum=args.normalize_target_sum,
            require_coords=args.require_coords,
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "validate":
        result = {
            "dataset": validate_dataset(
                args.dataset_dir,
                skip_source_compare=args.skip_source_compare,
                rtol=args.rtol,
                atol=args.atol,
            )
        }
        if args.region_manifest is not None:
            result["regions"] = validate_region_manifest_examples(
                args.dataset_dir,
                args.region_manifest,
                max_regions_to_check=args.max_regions_to_check,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "inspect":
        metadata = DatasetMetadata.load(args.dataset_dir)
        print(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.command == "checksum-manifest":
        payload = write_checksum_manifest(
            args.input_dir,
            output_path=args.output_json,
            algorithm=args.algorithm,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "verify-checksum-manifest":
        payload = verify_checksum_manifest(
            args.manifest_json,
            root_dir=args.root_dir,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "generate-synthetic-benchmark":
        payload = generate_public_synthetic_benchmark_data(
            output_dir=args.output_dir,
            mode=args.mode,
            scale_preset=args.scale_preset,
            seed=args.seed,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "export-hf-benchmark-split":
        payload = export_benchmark_split(
            dataset_root=str(args.dataset_root),
            split_kind=args.split_kind,
            split_name=args.split_name,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            config_name=args.config_name,
            include_obs=args.include_obs,
            sample_ids=args.sample_id,
            limit=args.limit,
            region_manifest_path=args.region_manifest,
            cells_per_group=args.cells_per_group,
            num_groups=args.num_groups,
            with_replacement=args.with_replacement,
            sample_weight_mode=args.sample_weight_mode,
            stratify_obs_column=args.stratify_obs_column,
            block_shape=args.block_shape,
            stride=args.stride,
            min_cells=args.min_cells,
            max_cells=args.max_cells,
            seed=args.seed,
            token=args.token,
            private=True if args.private else None,
            create_pr=args.create_pr,
            commit_message=args.commit_message,
            max_shard_size=args.max_shard_size,
            num_shards=args.num_shards,
            num_proc=args.num_proc,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
