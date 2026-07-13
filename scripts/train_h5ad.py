#!/usr/bin/env python
"""Fine-tune Hugging Face OmniCell models from H5AD inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omnicell_hf import (  # noqa: E402
    OmniCellConfig,
    OmniCellDataCollator,
    OmniCellForSupervisedFineTuning,
    OmniCellForUnsupervisedFineTuning,
    OmniCellH5ADDataset,
    OmniCellProcessor,
)
from omnicell_hf.legacy import build_model_from_legacy  # noqa: E402


DEFAULT_OMNICELL_ROOT = REPO_ROOT.parent / "OmniCell"
DEFAULT_VOCAB = DEFAULT_OMNICELL_ROOT / "vocab" / "Vocabulary.json"
DEFAULT_ALIAS = DEFAULT_OMNICELL_ROOT / "vocab" / "new_genes_homo_sapiens.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["unsupervised", "supervised"], required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--alias-csv", type=Path, default=DEFAULT_ALIAS)

    parser.add_argument("--model-name-or-path", type=Path, default=None)
    parser.add_argument("--legacy-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--save-legacy-load-report", action="store_true")

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
    parser.add_argument("--target-obs-key", type=str, default=None)
    parser.add_argument("--cell-id-obs-key", type=str, default=None)

    parser.add_argument("--region-manifest-path", type=Path, default=None)
    parser.add_argument("--center-cells", type=str, default=None)
    parser.add_argument("--neighbor-cells", type=str, default=None)
    parser.add_argument("--allow-short-groups", action="store_true")

    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument(
        "--problem-type",
        choices=["regression", "single_label_classification", "multi_label_classification"],
        default=None,
    )
    parser.add_argument("--use-smooth-rank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backed", action="store_true")

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--train-val-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> OmniCellH5ADDataset:
    return OmniCellH5ADDataset(
        h5ad_path=args.h5ad,
        vocab_path=args.vocab_path,
        alias_csv=args.alias_csv,
        token_per_cell=args.token_per_cell,
        mode=args.mode,
        n_cells_per_sample=args.n_cells_per_sample,
        gene_strategy=args.gene_strategy,
        selected_genes=args.selected_genes,
        hvg_top_n=args.hvg_top_n,
        target_obs_key=args.target_obs_key,
        backed=args.backed,
        use_smooth_rank=args.use_smooth_rank,
        region_manifest_path=args.region_manifest_path,
        center_cells=args.center_cells,
        neighbor_cells=args.neighbor_cells,
        cell_id_obs_key=args.cell_id_obs_key,
        allow_short_groups=args.allow_short_groups,
    )


def build_model(args: argparse.Namespace, dataset: OmniCellH5ADDataset):
    model_type = args.task
    num_labels = args.num_labels
    problem_type = args.problem_type
    if args.task == "supervised":
        if args.target_obs_key is None:
            raise ValueError("--target-obs-key is required for supervised fine-tuning.")
        num_labels = num_labels or dataset.num_labels or 1
        problem_type = problem_type or dataset.problem_type

    common_overrides = {
        "token_per_cell": args.token_per_cell,
        "n_cells_per_sample": args.n_cells_per_sample,
        "num_labels": num_labels,
        "problem_type": problem_type,
    }

    if args.legacy_checkpoint_dir is not None:
        model, report = build_model_from_legacy(
            args.legacy_checkpoint_dir,
            model_type=model_type,
            **common_overrides,
        )
        if args.save_legacy_load_report:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "legacy_load_report.json").write_text(
                json.dumps(
                    {
                        "missing_keys": list(report["missing_keys"]),
                        "unexpected_keys": list(report["unexpected_keys"]),
                    },
                    indent=2,
                )
            )
        return model

    model_cls = (
        OmniCellForUnsupervisedFineTuning
        if args.task == "unsupervised"
        else OmniCellForSupervisedFineTuning
    )
    if args.model_name_or_path is not None:
        config = OmniCellConfig.from_pretrained(args.model_name_or_path)
        for key, value in common_overrides.items():
            if value is not None:
                setattr(config, key, value)
        return model_cls.from_pretrained(
            args.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )

    config = OmniCellConfig(
        token_per_cell=args.token_per_cell,
        n_cells_per_sample=args.n_cells_per_sample,
        num_labels=num_labels or 1,
        problem_type=problem_type,
    )
    return model_cls(config)


def split_dataset(dataset, split: float, seed: int):
    if split <= 0:
        return dataset, None
    if not 0 < split < 1:
        raise ValueError("--train-val-split must be in [0, 1).")
    eval_len = max(1, int(len(dataset) * split))
    train_len = len(dataset) - eval_len
    if train_len <= 0:
        raise ValueError("Validation split leaves no training samples.")
    return random_split(
        dataset,
        [train_len, eval_len],
        generator=__import__("torch").Generator().manual_seed(seed),
    )


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    model = build_model(args, dataset)
    train_dataset, eval_dataset = split_dataset(dataset, args.train_val_split, args.seed)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=OmniCellDataCollator(),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    OmniCellProcessor(
        vocab_path=args.vocab_path,
        alias_csv=args.alias_csv,
        token_per_cell=args.token_per_cell,
        n_cells_per_sample=args.n_cells_per_sample,
        mode=args.mode,
        gene_strategy=args.gene_strategy,
        selected_genes=args.selected_genes,
        use_smooth_rank=args.use_smooth_rank,
        backed=args.backed,
        cell_id_obs_key=args.cell_id_obs_key,
    ).save_pretrained(args.output_dir)
    if getattr(dataset, "label_names", None) is not None:
        (args.output_dir / "label_names.json").write_text(
            json.dumps(dataset.label_names, indent=2)
        )


if __name__ == "__main__":
    main()
