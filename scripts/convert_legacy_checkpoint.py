#!/usr/bin/env python
"""Convert an original OmniCell checkpoint into Hugging Face format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnicell_hf import OmniCellProcessor
from omnicell_hf.legacy import build_model_from_legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=["backbone", "unsupervised", "supervised"],
        default="backbone",
    )
    parser.add_argument("--token-per-cell", type=int, default=None)
    parser.add_argument("--n-cells-per-sample", type=int, default=None)
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--problem-type", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, report = build_model_from_legacy(
        args.legacy_checkpoint_dir,
        model_type=args.model_type,
        token_per_cell=args.token_per_cell,
        n_cells_per_sample=args.n_cells_per_sample,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    OmniCellProcessor(
        token_per_cell=args.token_per_cell or model.config.token_per_cell,
        n_cells_per_sample=args.n_cells_per_sample or model.config.n_cells_per_sample,
    ).save_pretrained(args.output_dir)
    (args.output_dir / "legacy_load_report.json").write_text(
        json.dumps(
            {
                "legacy_checkpoint_dir": str(args.legacy_checkpoint_dir),
                "model_type": args.model_type,
                "missing_keys": list(report["missing_keys"]),
                "unexpected_keys": list(report["unexpected_keys"]),
            },
            indent=2,
        )
    )
    print(f"Saved Hugging Face model to: {args.output_dir}")
    if report["missing_keys"]:
        print(f"Missing keys: {len(report['missing_keys'])}")
    if report["unexpected_keys"]:
        print(f"Unexpected keys: {len(report['unexpected_keys'])}")


if __name__ == "__main__":
    main()
