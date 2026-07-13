#!/usr/bin/env python
"""Run OmniCell inference on an H5AD file and write obsm['OmniCell_embedding']."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnicell_hf import embed_h5ad  # noqa: E402


DEFAULT_OMNICELL_ROOT = REPO_ROOT.parent / "OmniCell"
DEFAULT_VOCAB = DEFAULT_OMNICELL_ROOT / "vocab" / "Vocabulary.json"
DEFAULT_ALIAS = DEFAULT_OMNICELL_ROOT / "vocab" / "new_genes_homo_sapiens.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument(
        "--output-h5ad",
        type=Path,
        default=None,
        help="Default: write <input_stem>_OmniCell.h5ad next to the input.",
    )
    parser.add_argument("--inplace", action="store_true", help="Write back to --h5ad.")
    parser.add_argument("--obsm-key", type=str, default="OmniCell_embedding")

    parser.add_argument("--model-name-or-path", type=Path, default=None)
    parser.add_argument(
        "--legacy-checkpoint-dir",
        type=Path,
        default=None,
        help="Default: ../OmniCell/checkpoint when no model path is provided.",
    )
    parser.add_argument(
        "--model-type",
        choices=["backbone", "unsupervised", "supervised"],
        default="backbone",
    )
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--alias-csv", type=Path, default=DEFAULT_ALIAS)

    parser.add_argument("--token-per-cell", type=int, default=500)
    parser.add_argument("--n-cells-per-sample", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["single_cell", "spatial", "region"],
        default="single_cell",
    )
    parser.add_argument(
        "--gene-strategy",
        choices=["selected", "hvg", "nonzero_hvg", "all_nonzero_hvg"],
        default="nonzero_hvg",
    )
    parser.add_argument("--selected-genes", type=str, default=None)
    parser.add_argument("--hvg-top-n", type=int, default=None)
    parser.add_argument("--region-manifest-path", type=Path, default=None)
    parser.add_argument("--center-cells", type=str, default=None)
    parser.add_argument("--neighbor-cells", type=str, default=None)
    parser.add_argument("--cell-id-obs-key", type=str, default=None)
    parser.add_argument("--allow-short-groups", action="store_true")
    parser.add_argument("--use-smooth-rank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--project",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the legacy OmniCell OrthogonalProjector before writing obsm.",
    )
    parser.add_argument("--projection-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = embed_h5ad(
        args.h5ad,
        output_h5ad=args.output_h5ad,
        inplace=args.inplace,
        obsm_key=args.obsm_key,
        model_name_or_path=args.model_name_or_path,
        legacy_checkpoint_dir=args.legacy_checkpoint_dir,
        model_type=args.model_type,
        vocab_path=args.vocab_path,
        alias_csv=args.alias_csv,
        token_per_cell=args.token_per_cell,
        n_cells_per_sample=args.n_cells_per_sample,
        mode=args.mode,
        gene_strategy=args.gene_strategy,
        selected_genes=args.selected_genes,
        hvg_top_n=args.hvg_top_n,
        region_manifest_path=args.region_manifest_path,
        center_cells=args.center_cells,
        neighbor_cells=args.neighbor_cells,
        cell_id_obs_key=args.cell_id_obs_key,
        allow_short_groups=args.allow_short_groups,
        use_smooth_rank=args.use_smooth_rank,
        backed=args.backed,
        batch_size=args.batch_size,
        device=args.device,
        fp16=args.fp16,
        project=args.project,
        projection_threshold=args.projection_threshold,
    )
    print(f"Saved OmniCell embeddings to: {output_path}")


if __name__ == "__main__":
    main()
