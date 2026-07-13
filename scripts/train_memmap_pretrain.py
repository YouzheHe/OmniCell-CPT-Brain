#!/usr/bin/env python
"""Continual pretraining for OmniCell from cellfm-datasets CSR memmaps."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT.parent
CELLFM_SRC = WORK_ROOT / "cellfm-datasets" / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CELLFM_SRC))

from cellfm_dataset.distributed import DistributedRuntime, resolve_distributed_context  # noqa: E402
from cellfm_dataset.memmap import MemmapDataset  # noqa: E402
from cellfm_dataset.sampling import RandomCellSampler  # noqa: E402
from omnicell_hf.configuration_omnicell import OmniCellConfig  # noqa: E402
from omnicell_hf.legacy import build_model_from_legacy  # noqa: E402
from omnicell_hf.modeling_omnicell import OmniCellForUnsupervisedFineTuning  # noqa: E402


DEFAULT_DATASET_ROOT = WORK_ROOT / "NVU_hyz"
DEFAULT_MODEL = REPO_ROOT / "outputs" / "OmniCell_backbone_hf"
DEFAULT_VOCAB_JSON = REPO_ROOT / "assets" / "vocab" / "Vocabulary.json"
REQUIRED_SAMPLE_FILES = {
    "coords.npy",
    "indices.npy",
    "indptr.npy",
    "metadata.json",
    "obs_names.npy",
    "present_gene_ids.npy",
    "raw_cell_sums.npy",
    "values.npy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB_JSON)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--legacy-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    parser.add_argument("--sequence-length", type=int, default=1500)
    parser.add_argument("--token-per-cell", type=int, default=None)
    parser.add_argument("--n-cells-per-sample", type=int, default=1)
    parser.add_argument("--selection-strategy", choices=["top_expression", "input_order"], default="top_expression")
    parser.add_argument("--sample-ids", type=str, default=None, help="Comma-separated sample ids, e.g. 31435019,Cortex_Spatial/T1001")
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument("--coord-dims", type=str, default=None, help="Optional comma-separated coord_dim filter, e.g. 0 or 2 or 0,2")
    parser.add_argument("--sample-weight-mode", choices=["n_cells", "uniform"], default="n_cells")
    parser.add_argument("--stratify-obs-column", type=str, default=None)
    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--num-groups", type=int, default=1_000_000_000)
    parser.add_argument("--normalize-coords", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--use-smooth-rank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-rank-min", type=float, default=0.0)
    parser.add_argument("--smooth-rank-max", type=float, default=5.0)
    parser.add_argument("--unsupervised-loss", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--unsupervised-loss-on", choices=["all_gene_tokens", "nonzero"], default="nonzero")
    parser.add_argument("--load-balance-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-flash", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-to", type=str, default="")

    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-dataset-index", action="store_true")
    parser.add_argument("--speed-test-only", action="store_true")
    parser.add_argument("--speed-test-steps", type=int, default=0)
    parser.add_argument("--speed-test-warmup-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-backbone-subdir", type=str, default="backbone")
    return parser.parse_args()


def resolve_sequence_shape(args: argparse.Namespace) -> None:
    if args.sequence_length is not None:
        sequence_length = int(args.sequence_length)
        n_cells = int(args.n_cells_per_sample)
        if sequence_length <= 0:
            raise ValueError("--sequence-length must be positive.")
        if sequence_length % n_cells != 0:
            raise ValueError(
                f"--sequence-length={sequence_length} must be divisible by "
                f"--n-cells-per-sample={n_cells}."
            )
        cell_token_len = sequence_length // n_cells
        if cell_token_len <= 2:
            raise ValueError(
                "Each cell needs at least one gene token plus RNA start/end tokens; "
                f"got cell token length {cell_token_len}."
            )
        derived_token_per_cell = cell_token_len - 2
        if args.token_per_cell is not None and int(args.token_per_cell) != derived_token_per_cell:
            raise ValueError(
                f"--sequence-length={sequence_length} and --n-cells-per-sample={n_cells} "
                f"imply --token-per-cell={derived_token_per_cell}, "
                f"but got --token-per-cell={args.token_per_cell}."
            )
        args.token_per_cell = derived_token_per_cell
        return

    if args.token_per_cell is None:
        args.token_per_cell = 500


def split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [item.strip() for item in value.split(",") if item.strip()]
    return out or None


def parse_int_filter(value: str | None) -> set[int] | None:
    parts = split_csv(value)
    if not parts:
        return None
    return {int(item) for item in parts}


def load_requested_sample_ids(args: argparse.Namespace) -> list[str] | None:
    sample_ids = split_csv(args.sample_ids) or []
    if args.sample_id_file is not None:
        sample_ids.extend(
            line.strip()
            for line in args.sample_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return sample_ids or None


def is_sample_dir(path: Path) -> bool:
    return all((path / filename).exists() for filename in REQUIRED_SAMPLE_FILES)


def scan_sample_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for metadata_path in sorted(dataset_root.rglob("metadata.json")):
        sample_dir = metadata_path.parent
        if not is_sample_dir(sample_dir):
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["sample_id"] = sample_dir.relative_to(dataset_root).as_posix()
        samples.append(payload)
    if not samples:
        raise FileNotFoundError(f"No memmap samples found under {dataset_root}")
    return samples


def write_gene_vocab_txt(vocab_json: Path, dataset_root: Path, *, force: bool = False) -> None:
    output_path = dataset_root / "gene_vocab.txt"
    if output_path.exists() and not force:
        return
    vocab_map = json.loads(vocab_json.read_text(encoding="utf-8"))
    max_id = max(int(idx) for idx in vocab_map.values())
    vocab = [""] * (max_id + 1)
    for gene, idx in vocab_map.items():
        vocab[int(idx)] = str(gene)
    missing = [idx for idx, gene in enumerate(vocab) if not gene]
    if missing:
        raise ValueError(f"Vocabulary JSON has missing token ids, first missing id={missing[0]}")
    output_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")


def write_dataset_manifest(dataset_root: Path, samples: list[dict[str, Any]], *, force: bool = False) -> None:
    output_path = dataset_root / "dataset_manifest.json"
    if output_path.exists() and not force:
        return
    n_genes_values = sorted({int(sample["n_genes"]) for sample in samples})
    transforms = sorted({str(sample.get("expression_transform", "normalize_total_log1p")) for sample in samples})
    target_sums = {sample.get("normalize_target_sum") for sample in samples}
    payload = {
        "dataset_dir": str(dataset_root),
        "input_manifest_json": "",
        "n_genes": n_genes_values[0] if len(n_genes_values) == 1 else max(n_genes_values),
        "expression_transform": transforms[0] if len(transforms) == 1 else "mixed",
        "normalize_target_sum": target_sums.pop() if len(target_sums) == 1 else None,
        "samples": samples,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_cellfm_dataset_root(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    samples = scan_sample_metadata(dataset_root)
    write_gene_vocab_txt(args.vocab_json.expanduser().resolve(), dataset_root, force=args.rebuild_dataset_index)
    write_dataset_manifest(dataset_root, samples, force=args.rebuild_dataset_index)
    return samples


def filter_sample_ids(
    samples: list[dict[str, Any]],
    requested: list[str] | None,
    coord_dims: set[int] | None,
) -> list[str] | None:
    allowed = {str(sample["sample_id"]) for sample in samples}
    if coord_dims is not None:
        allowed = {str(sample["sample_id"]) for sample in samples if int(sample["coord_dim"]) in coord_dims}
    if requested is not None:
        unknown = sorted(set(requested) - {str(sample["sample_id"]) for sample in samples})
        if unknown:
            raise KeyError(f"Unknown sample ids: {unknown[:10]}")
        allowed &= set(requested)
    if not allowed:
        raise ValueError("No samples left after applying sample filters.")
    if requested is None and coord_dims is None:
        return None
    return sorted(allowed)


class MemmapRandomGroupDataset(IterableDataset):
    """Stream RandomCellSampler groups from a cellfm-datasets memmap root."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        cells_per_group: int,
        num_groups: int,
        seed: int,
        sample_ids: list[str] | None,
        sample_weight_mode: str,
        with_replacement: bool,
        stratify_obs_column: str | None,
        normalize_coords: bool,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.cells_per_group = int(cells_per_group)
        self.num_groups = int(num_groups)
        self.seed = int(seed)
        self.sample_ids = sample_ids
        self.sample_weight_mode = sample_weight_mode
        self.with_replacement = bool(with_replacement)
        self.stratify_obs_column = stratify_obs_column
        self.normalize_coords = bool(normalize_coords)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        dataset = MemmapDataset(self.dataset_root)
        runtime = DistributedRuntime.from_environment(epoch=self.epoch)
        context = resolve_distributed_context(runtime)
        sampler = RandomCellSampler(
            cells_per_group=self.cells_per_group,
            num_groups=self.num_groups,
            seed=self.seed,
            with_replacement=self.with_replacement,
            sample_ids=self.sample_ids,
            sample_weight_mode=self.sample_weight_mode,
            stratify_obs_column=self.stratify_obs_column,
        )
        for spec in sampler.iter_specs(dataset, distributed_context=context):
            yield dataset.build_group(
                sample_id=spec.sample_id,
                source_cell_indices=spec.source_cell_indices,
                group_id=spec.group_id,
                group_type=spec.group_type,
                group_index=spec.group_index,
                sample_group_index=spec.sample_group_index,
                normalize_coords=self.normalize_coords,
                include_obs=False,
                anchor_coord=spec.anchor_coord,
                block_bounds=spec.block_bounds,
                metadata=spec.metadata,
            )


class OmniCellMemmapCollator:
    """Turn sparse memmap groups into fixed-length OmniCell training tensors."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        token_per_cell: int,
        vocab_size: int,
        start_token_id: int,
        end_token_id: int,
        max_expression_token_id: int,
        use_smooth_rank: bool,
        smooth_rank_range: tuple[float, float],
        selection_strategy: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.token_per_cell = int(token_per_cell)
        self.vocab_size = int(vocab_size)
        self.start_token_id = int(start_token_id)
        self.end_token_id = int(end_token_id)
        self.max_expression_token_id = int(max_expression_token_id)
        self.use_smooth_rank = bool(use_smooth_rank)
        self.smooth_rank_range = smooth_rank_range
        self.selection_strategy = selection_strategy
        self._fallback_cache: dict[str, np.ndarray] = {}
        self._global_fallback = np.arange(self.max_expression_token_id + 1, dtype=np.int64)

    def _fallback_gene_ids(self, sample_id: str) -> np.ndarray:
        if sample_id in self._fallback_cache:
            return self._fallback_cache[sample_id]
        path = self.dataset_root / sample_id / "present_gene_ids.npy"
        if path.exists():
            values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
            values = values[(values >= 0) & (values <= self.max_expression_token_id)]
            if values.size > 0:
                self._fallback_cache[sample_id] = values
                return values
        self._fallback_cache[sample_id] = self._global_fallback
        return self._global_fallback

    def _smooth_rank(self, values: np.ndarray) -> np.ndarray:
        unique_values = np.unique(values.astype(np.float32, copy=False))
        if not np.any(unique_values == 0):
            unique_values = np.append(unique_values, 0.0)
        unique_sorted = np.sort(unique_values)
        left, right = self.smooth_rank_range
        if unique_sorted.size == 1:
            ranks = np.full(unique_sorted.size, left, dtype=np.float32)
        else:
            ranks = left + (np.arange(unique_sorted.size) / (unique_sorted.size - 1)) * (right - left)
            ranks = ranks.astype(np.float32)
        return ranks[np.searchsorted(unique_sorted, values, side="left")]

    def _select_cell_tokens(
        self,
        *,
        sample_id: str,
        row_indices: np.ndarray,
        row_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        row_values = np.asarray(row_values, dtype=np.float32)
        valid = (
            np.isfinite(row_values)
            & (row_indices >= 0)
            & (row_indices <= self.max_expression_token_id)
            & (row_indices < self.vocab_size)
        )
        row_indices = row_indices[valid]
        row_values = row_values[valid]

        if row_indices.size > self.token_per_cell:
            if self.selection_strategy == "top_expression":
                order = np.argsort(-row_values, kind="mergesort")[: self.token_per_cell]
            else:
                order = np.arange(self.token_per_cell)
            row_indices = row_indices[order]
            row_values = row_values[order]

        selected_ids = row_indices[: self.token_per_cell].astype(np.int64, copy=False)
        selected_values = row_values[: self.token_per_cell].astype(np.float32, copy=False)
        nonzero_mask = (selected_values > 0).astype(np.float32, copy=False)

        if selected_ids.size < self.token_per_cell:
            needed = self.token_per_cell - selected_ids.size
            used = set(int(item) for item in selected_ids.tolist())
            pad_ids = []
            for token_id in self._fallback_gene_ids(sample_id).tolist():
                token_id = int(token_id)
                if token_id in used:
                    continue
                pad_ids.append(token_id)
                used.add(token_id)
                if len(pad_ids) >= needed:
                    break
            if len(pad_ids) < needed:
                for token_id in self._global_fallback.tolist():
                    token_id = int(token_id)
                    if token_id in used:
                        continue
                    pad_ids.append(token_id)
                    used.add(token_id)
                    if len(pad_ids) >= needed:
                        break
            if len(pad_ids) < needed:
                raise ValueError(f"Unable to pad cell to {self.token_per_cell} gene tokens.")
            selected_ids = np.concatenate([selected_ids, np.asarray(pad_ids, dtype=np.int64)])
            selected_values = np.concatenate([selected_values, np.zeros(needed, dtype=np.float32)])
            nonzero_mask = np.concatenate([nonzero_mask, np.zeros(needed, dtype=np.float32)])

        expression_values = selected_values.astype(np.float32, copy=True)
        if self.use_smooth_rank:
            expression_values = self._smooth_rank(expression_values)
        return selected_ids, expression_values, nonzero_mask

    def _tokenize_group(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        sample_id = str(item["sample_id"])
        coords = np.asarray(item["coords"], dtype=np.float32)
        if coords.ndim != 2:
            raise ValueError(f"coords must be 2D, got shape={coords.shape}")
        if coords.shape[1] == 0:
            coords = np.zeros((coords.shape[0], 2), dtype=np.float32)
        elif coords.shape[1] == 1:
            coords = np.pad(coords, ((0, 0), (0, 1)), mode="constant")

        cell_ptr = np.asarray(item["cell_ptr"], dtype=np.int64)
        gene_indices = np.asarray(item["gene_indices"], dtype=np.int64)
        gene_values = np.asarray(item["gene_values"], dtype=np.float32)
        input_chunks: list[np.ndarray] = []
        value_chunks: list[np.ndarray] = []
        mask_chunks: list[np.ndarray] = []
        position_chunks: list[np.ndarray] = []

        for cell_idx in range(coords.shape[0]):
            start = int(cell_ptr[cell_idx])
            end = int(cell_ptr[cell_idx + 1])
            selected_ids, expression_values, nonzero_mask = self._select_cell_tokens(
                sample_id=sample_id,
                row_indices=gene_indices[start:end],
                row_values=gene_values[start:end],
            )
            input_chunks.append(
                np.concatenate(
                    [
                        np.asarray([self.start_token_id], dtype=np.int64),
                        selected_ids,
                        np.asarray([self.end_token_id], dtype=np.int64),
                    ]
                )
            )
            value_chunks.append(
                np.concatenate(
                    [
                        np.asarray([0.0], dtype=np.float32),
                        expression_values.astype(np.float32, copy=False),
                        np.asarray([0.0], dtype=np.float32),
                    ]
                )
            )
            mask_chunks.append(
                np.concatenate(
                    [
                        np.asarray([0.0], dtype=np.float32),
                        nonzero_mask.astype(np.float32, copy=False),
                        np.asarray([0.0], dtype=np.float32),
                    ]
                )
            )
            position_chunks.append(
                np.repeat(coords[cell_idx][None, :], self.token_per_cell + 2, axis=0).astype(np.float32)
            )

        return {
            "input_ids": torch.from_numpy(np.concatenate(input_chunks).astype(np.int64, copy=False)),
            "expression_values": torch.from_numpy(np.concatenate(value_chunks).astype(np.float32, copy=False)),
            "positions": torch.from_numpy(np.concatenate(position_chunks).astype(np.float32, copy=False)),
            "nonzero_mask": torch.from_numpy(np.concatenate(mask_chunks).astype(np.float32, copy=False)),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        tokenized = [self._tokenize_group(feature) for feature in features]
        return {
            key: torch.stack([feature[key] for feature in tokenized], dim=0)
            for key in ("input_ids", "expression_values", "positions", "nonzero_mask")
        }


def apply_config_overrides(config: OmniCellConfig, args: argparse.Namespace) -> OmniCellConfig:
    config.token_per_cell = int(args.token_per_cell)
    config.n_cells_per_sample = int(args.n_cells_per_sample)
    config.unsupervised_loss = args.unsupervised_loss
    config.unsupervised_loss_on = args.unsupervised_loss_on
    config.load_balance_loss_weight = float(args.load_balance_loss_weight)
    config.use_flash = bool(args.use_flash)
    return config


def load_state_dict_from_pretrained(path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = path / "model.safetensors"
    torch_path = path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")
    if torch_path.exists():
        return torch.load(torch_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {path}")


def build_unsupervised_model(args: argparse.Namespace) -> OmniCellForUnsupervisedFineTuning:
    if args.legacy_checkpoint_dir is not None:
        model, report = build_model_from_legacy(
            args.legacy_checkpoint_dir,
            model_type="unsupervised",
            token_per_cell=args.token_per_cell,
            n_cells_per_sample=args.n_cells_per_sample,
        )
        model.config.unsupervised_loss = args.unsupervised_loss
        model.config.unsupervised_loss_on = args.unsupervised_loss_on
        model.config.load_balance_loss_weight = args.load_balance_loss_weight
        model.config.use_flash = args.use_flash
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoint_load_report.json").write_text(
            json.dumps({"source": str(args.legacy_checkpoint_dir), **report}, indent=2),
            encoding="utf-8",
        )
        return model

    config = OmniCellConfig.from_pretrained(args.model_name_or_path)
    config = apply_config_overrides(config, args)
    model = OmniCellForUnsupervisedFineTuning(config)
    state_dict = load_state_dict_from_pretrained(args.model_name_or_path)
    if any(key.startswith("omnicell.") for key in state_dict):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        load_target = "unsupervised"
    else:
        missing, unexpected = model.omnicell.load_state_dict(state_dict, strict=False)
        load_target = "backbone"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_load_report.json").write_text(
        json.dumps(
            {
                "source": str(args.model_name_or_path),
                "loaded_as": load_target,
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def build_dataset_and_collator(
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    config: OmniCellConfig,
) -> tuple[MemmapRandomGroupDataset, OmniCellMemmapCollator, list[str] | None]:
    requested_sample_ids = load_requested_sample_ids(args)
    sample_ids = filter_sample_ids(samples, requested_sample_ids, parse_int_filter(args.coord_dims))
    dataset = MemmapRandomGroupDataset(
        dataset_root=args.dataset_root,
        cells_per_group=args.n_cells_per_sample,
        num_groups=args.num_groups,
        seed=args.seed,
        sample_ids=sample_ids,
        sample_weight_mode=args.sample_weight_mode,
        with_replacement=args.with_replacement,
        stratify_obs_column=args.stratify_obs_column,
        normalize_coords=args.normalize_coords,
    )
    max_expression_token_id = min(config.vocab_size - 1, config.start_token_id - 1)
    collator = OmniCellMemmapCollator(
        dataset_root=args.dataset_root,
        token_per_cell=args.token_per_cell,
        vocab_size=config.vocab_size,
        start_token_id=config.start_token_id,
        end_token_id=config.end_token_id,
        max_expression_token_id=max_expression_token_id,
        use_smooth_rank=args.use_smooth_rank,
        smooth_rank_range=(args.smooth_rank_min, args.smooth_rank_max),
        selection_strategy=args.selection_strategy,
    )
    return dataset, collator, sample_ids


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def autocast_context(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or (not args.bf16 and not args.fp16):
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_speed_test(
    *,
    args: argparse.Namespace,
    model: OmniCellForUnsupervisedFineTuning,
    dataset: MemmapRandomGroupDataset,
    collator: OmniCellMemmapCollator,
) -> dict[str, float]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.per_device_train_batch_size,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collator,
    }
    if args.dataloader_num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    total_steps = args.speed_test_warmup_steps + args.speed_test_steps
    if total_steps <= 0:
        raise ValueError("--speed-test-steps must be positive for speed testing.")
    measured_steps = 0
    measured_samples = 0
    measured_cells = 0
    measured_tokens = 0
    total_loss = 0.0
    start_time = time.perf_counter() if args.speed_test_warmup_steps == 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))

    for step, batch in enumerate(loader):
        if step >= total_steps:
            break
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args, device):
            outputs = model(**batch)
            loss = outputs.loss
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if args.speed_test_warmup_steps > 0 and step + 1 == args.speed_test_warmup_steps:
            start_time = time.perf_counter()
            measured_steps = 0
            measured_samples = 0
            measured_cells = 0
            measured_tokens = 0
            total_loss = 0.0
            continue
        if step + 1 > args.speed_test_warmup_steps:
            measured_steps += 1
            measured_samples += int(batch["input_ids"].shape[0])
            measured_cells += int(batch["input_ids"].shape[0] * args.n_cells_per_sample)
            measured_tokens += int(batch["input_ids"].numel())
            total_loss += float(loss.detach().cpu())

    if start_time is None:
        start_time = time.perf_counter()
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    result = {
        "steps": float(measured_steps),
        "elapsed_sec": elapsed,
        "steps_per_sec": measured_steps / elapsed,
        "samples_per_sec": measured_samples / elapsed,
        "cells_per_sec": measured_cells / elapsed,
        "tokens_per_sec": measured_tokens / elapsed,
        "mean_loss": total_loss / max(measured_steps, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "speed_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def save_run_config(args: argparse.Namespace, samples: list[dict[str, Any]], sample_ids: list[str] | None) -> None:
    selected_count = len(sample_ids) if sample_ids is not None else len(samples)
    total_cells = 0
    allowed = None if sample_ids is None else set(sample_ids)
    for sample in samples:
        if allowed is None or str(sample["sample_id"]) in allowed:
            total_cells += int(sample["n_cells"])
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_indexed_samples": len(samples),
        "num_selected_samples": selected_count,
        "selected_cells": total_cells,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def report_to_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    resolve_sequence_shape(args)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    samples = prepare_cellfm_dataset_root(args)
    if args.prepare_only:
        print(f"Prepared {len(samples)} memmap samples under {args.dataset_root}", flush=True)
        return

    model = build_unsupervised_model(args)
    train_dataset, data_collator, sample_ids = build_dataset_and_collator(args, samples, model.config)
    save_run_config(args, samples, sample_ids)

    if args.speed_test_only or args.speed_test_steps > 0:
        run_speed_test(args=args, model=model, dataset=train_dataset, collator=data_collator)
        if args.speed_test_only:
            return

    if args.max_steps <= 0:
        raise ValueError("Streaming memmap training requires --max-steps > 0.")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        remove_unused_columns=False,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        seed=args.seed,
        report_to=report_to_list(args.report_to),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if args.save_backbone_subdir:
        model.omnicell.save_pretrained(args.output_dir / args.save_backbone_subdir)


if __name__ == "__main__":
    main()
