#!/usr/bin/env python
"""Multi-task OmniCell fine-tuning for Figure 1 representation metrics.

The objective is deliberately broader than the vascular subtype fine-tune:
reconstruction keeps the CPT language-model objective, disease/age/cell-class
heads preserve biological state, and gradient-reversal domain heads reduce
cohort/modality separability in the backbone representation.
"""

from __future__ import annotations
import os

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.utils.data import IterableDataset, get_worker_info

WORK_ROOT = Path(os.path.expandvars("${OMNICELL_NVU_ROOT}"))
REPO_ROOT = WORK_ROOT / "OmniCell-HF"
CELLFM_SRC = WORK_ROOT / "cellfm-datasets" / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CELLFM_SRC))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cellfm_dataset.distributed import DistributedRuntime, resolve_distributed_context  # noqa: E402
from cellfm_dataset.memmap import MemmapDataset  # noqa: E402
from cellfm_dataset.sampling import RandomCellSampler  # noqa: E402
from omnicell_hf.configuration_omnicell import OmniCellConfig  # noqa: E402
from omnicell_hf.modeling_omnicell import OmniCellForUnsupervisedFineTuning  # noqa: E402
from train_memmap_pretrain import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_VOCAB_JSON,
    OmniCellMemmapCollator,
    filter_sample_ids,
    load_requested_sample_ids,
    parse_int_filter,
    prepare_cellfm_dataset_root,
    report_to_list,
    resolve_sequence_shape,
)


DEFAULT_CELL_CLASSES = [
    "Excitatory neuron",
    "Inhibitory neuron",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia/immune",
    "Vascular",
    "Ependymal/choroid",
    "Other",
]
DEFAULT_COHORTS = [
    "AD_Hip_Saptial",
    "AD_Hip_sc",
    "AD_sc",
    "Cortex_Spatial",
    "Cortex_sc",
    "AD_Cortex_Spatial",
    "39402379",
    "Public_snRNA_PMID",
    "Other",
]
DISEASE_LABELS = ["Control", "AD"]
MODALITY_LABELS = ["single_cell", "spatial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-json", type=Path, default=DEFAULT_VOCAB_JSON)
    parser.add_argument("--model-name-or-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=1500)
    parser.add_argument("--token-per-cell", type=int, default=None)
    parser.add_argument("--n-cells-per-sample", type=int, default=1)
    parser.add_argument("--selection-strategy", choices=["top_expression", "input_order"], default="top_expression")
    parser.add_argument("--sample-ids", type=str, default=None)
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument("--coord-dims", type=str, default=None)
    parser.add_argument("--sample-weight-mode", choices=["n_cells", "uniform"], default="uniform")
    parser.add_argument("--stratify-obs-column", type=str, default=None)
    parser.add_argument("--anchor-csv", type=Path, default=None)
    parser.add_argument("--anchor-strata-column", type=str, default="age_condition_modality_cell_class_stratum")
    parser.add_argument("--anchor-weight-mode", choices=["stratified", "uniform"], default="stratified")
    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--num-groups", type=int, default=1_000_000_000)
    parser.add_argument("--normalize-coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-smooth-rank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth-rank-min", type=float, default=0.0)
    parser.add_argument("--smooth-rank-max", type=float, default=5.0)
    parser.add_argument("--unsupervised-loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--unsupervised-loss-on", choices=["all_gene_tokens", "nonzero"], default="nonzero")
    parser.add_argument("--load-balance-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-flash", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cell-class-labels", default=",".join(DEFAULT_CELL_CLASSES))
    parser.add_argument("--cohort-labels", default=",".join(DEFAULT_COHORTS))
    parser.add_argument("--age-scale", type=float, default=100.0)
    parser.add_argument("--num-age-bins", type=int, default=8)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=1.0)
    parser.add_argument("--disease-loss-weight", type=float, default=0.35)
    parser.add_argument("--age-loss-weight", type=float, default=0.25)
    parser.add_argument("--age-bin-loss-weight", type=float, default=0.0)
    parser.add_argument("--age-supcon-loss-weight", type=float, default=0.0)
    parser.add_argument("--cell-class-loss-weight", type=float, default=0.65)
    parser.add_argument("--cell-supcon-loss-weight", type=float, default=0.10)
    parser.add_argument("--cohort-adversarial-loss-weight", type=float, default=0.20)
    parser.add_argument("--modality-adversarial-loss-weight", type=float, default=0.20)
    parser.add_argument("--domain-grl-lambda", type=float, default=1.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-to", type=str, default="")
    parser.add_argument("--rebuild-dataset-index", action="store_true")
    parser.add_argument("--save-backbone-subdir", type=str, default="backbone")
    return parser.parse_args()


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def first_present(row: dict[str, Any], columns: list[str]) -> str:
    for col in columns:
        if col in row and row[col] is not None:
            value = str(row[col]).strip()
            if value and value.lower() not in {"nan", "na", "none", "<na>", "unknown"}:
                return value
    return ""


def parse_age(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    age = float(match.group(0))
    if not np.isfinite(age) or age < 0:
        return None
    return age


def infer_condition(row: dict[str, Any], sample_id: str) -> str:
    value = first_present(
        row,
        [
            "condition_inferred",
            "condition",
            "disease",
            "ROIGroupFine",
            "Diagnosis",
            "diagnosis",
            "sample",
            "chip",
            "chipID",
            "chipId",
            "batch_id",
        ],
    )
    text = f"{value} {sample_id}".lower()
    if re.search(r"\bad\b|alzheimer|dement|(?:^|[/_\-\s])ad\d", text):
        return "AD"
    if re.search(r"\bcon\b|control|normal|ctrl|healthy|(?:^|[/_\-\s])con\d", text):
        return "Control"
    return ""


def infer_modality(row: dict[str, Any], sample_id: str) -> str:
    value = first_present(row, ["modality", "modality_display", "dataset_source"])
    text = f"{value} {sample_id}".lower()
    if "spatial" in text or "saptial" in text or "stereo" in text:
        return "spatial"
    return "single_cell"


def infer_cohort(row: dict[str, Any], sample_id: str) -> str:
    value = first_present(row, ["cohort", "dataset_source", "source_dataset"])
    text = f"{value} {sample_id}"
    if "AD_Hip_Saptial" in text:
        return "AD_Hip_Saptial"
    if "AD_Hip_sc" in text:
        return "AD_Hip_sc"
    if sample_id == "AD_sc" or "AD_sc" in text:
        return "AD_sc"
    if "Cortex_Spatial" in text:
        return "Cortex_Spatial"
    if "Cortex_sc" in text:
        return "Cortex_sc"
    if "AD_Cortex_Spatial" in text:
        return "AD_Cortex_Spatial"
    if "39402379" in text:
        return "39402379"
    if "Public_snRNA" in text or sample_id.isdigit():
        return "Public_snRNA_PMID"
    return "Other"


def infer_cell_class(row: dict[str, Any]) -> str:
    value = first_present(
        row,
        [
            "ground_truth_celltype",
            "cell_class",
            "CellType_m",
            "celltype",
            "cell_type",
            "CellType",
            "subclass.v4",
            "ground_truth_label",
            "cell_label_original",
        ],
    )
    text = value.lower()
    if not text:
        return ""
    if any(x in text for x in ["excit", "ex_", "glut", "ca1", "ca2", "ca3", "dg", "sub", "it neuron", "et neuron", "ct neuron"]):
        return "Excitatory neuron"
    if any(x in text for x in ["inhib", "in_", "gaba", "pvalb", "vip", "sst", "lamp5"]):
        return "Inhibitory neuron"
    if "astro" in text:
        return "Astrocyte"
    if any(x in text for x in ["oligo", "oligodendrocyte"]):
        return "Oligodendrocyte"
    if "opc" in text:
        return "OPC"
    if any(x in text for x in ["micro", "immune", "macrophage", "myeloid", "monocyte"]):
        return "Microglia/immune"
    if any(x in text for x in ["endo", "vascular", "pericyte", "mural", "smc", "vsmc", "fibro", "vlmc"]):
        return "Vascular"
    if any(x in text for x in ["ependymal", "choroid"]):
        return "Ependymal/choroid"
    return "Other"


def resolve_training_sample_ids(args: argparse.Namespace, samples: list[dict[str, Any]]) -> list[str] | None:
    requested = load_requested_sample_ids(args)
    coord_dims = parse_int_filter(args.coord_dims)
    if requested is None:
        return filter_sample_ids(samples, None, coord_dims)

    dataset = MemmapDataset(args.dataset_root)
    available = set(dataset.samples.keys())
    alias = {
        "AD_Hip_sc": "AD_sc",
        "AD_sc": "AD_Hip_sc",
    }
    resolved: list[str] = []
    unknown: list[str] = []
    for sample_id in requested:
        if sample_id in available:
            resolved.append(sample_id)
            continue
        alt = alias.get(sample_id)
        if alt and alt in available:
            resolved.append(alt)
            continue
        unknown.append(sample_id)
    if unknown:
        raise KeyError(f"Unknown sample ids for MemmapDataset: {unknown[:10]}")

    if coord_dims is not None:
        filtered: list[str] = []
        for sample_id in resolved:
            sample = dataset.samples[sample_id]
            coords = getattr(sample, "coords", None)
            coord_dim = int(coords.shape[1]) if coords is not None and getattr(coords, "ndim", 0) == 2 else 0
            if coord_dim in coord_dims:
                filtered.append(sample_id)
        resolved = filtered
    if not resolved:
        raise ValueError("No samples left after resolving sample ids and coordinate filters.")
    return sorted(set(resolved))


class MultiTaskMemmapRandomGroupDataset(IterableDataset):
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
            item = dataset.build_group(
                sample_id=spec.sample_id,
                source_cell_indices=spec.source_cell_indices,
                group_id=spec.group_id,
                group_type=spec.group_type,
                group_index=spec.group_index,
                sample_group_index=spec.sample_group_index,
                normalize_coords=self.normalize_coords,
                include_obs=True,
                anchor_coord=spec.anchor_coord,
                block_bounds=spec.block_bounds,
                metadata=spec.metadata,
            )
            item["sample_id"] = spec.sample_id
            yield item


def resolve_dataset_sample_id(dataset: MemmapDataset, sample_id: str) -> str:
    if sample_id in dataset.samples:
        return sample_id
    alias = {
        "AD_Hip_sc": "AD_sc",
        "AD_sc": "AD_Hip_sc",
    }
    alt = alias.get(sample_id)
    if alt and alt in dataset.samples:
        return alt
    raise KeyError(f"Unknown sample_id in anchor table: {sample_id}")


class MultiTaskAnchorCsvDataset(IterableDataset):
    """Stratified single-cell groups from a precomputed anchor table."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        anchor_csv: Path,
        num_groups: int,
        seed: int,
        strata_column: str,
        weight_mode: str,
        normalize_coords: bool,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.anchor_csv = Path(anchor_csv)
        self.num_groups = int(num_groups)
        self.seed = int(seed)
        self.strata_column = strata_column
        self.weight_mode = weight_mode
        self.normalize_coords = bool(normalize_coords)
        self.epoch = 0

        frame = pd.read_csv(self.anchor_csv)
        if "sample_id" not in frame.columns:
            raise ValueError(f"{self.anchor_csv} must contain a sample_id column.")
        index_col = ""
        for candidate in ["source_cell_index", "cell_index", "row_index", "obs_index"]:
            if candidate in frame.columns:
                index_col = candidate
                break
        if not index_col:
            raise ValueError(f"{self.anchor_csv} must contain source_cell_index/cell_index/row_index/obs_index.")
        frame = frame.dropna(subset=["sample_id", index_col]).copy()
        frame[index_col] = frame[index_col].astype(int)
        if self.strata_column not in frame.columns:
            frame[self.strata_column] = "all"
        frame[self.strata_column] = frame[self.strata_column].fillna("missing").astype(str)
        self.frame = frame.reset_index(drop=True)
        self.index_col = index_col
        self.strata = sorted(self.frame[self.strata_column].unique().tolist())
        self.indices_by_stratum = {
            stratum: self.frame.index[self.frame[self.strata_column] == stratum].to_numpy(dtype=np.int64)
            for stratum in self.strata
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        dataset = MemmapDataset(self.dataset_root)
        runtime = DistributedRuntime.from_environment(epoch=self.epoch)
        context = resolve_distributed_context(runtime)
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng(self.seed + 1009 * self.epoch + 9173 * context.rank + 137 * worker_id)
        local_counter = 0
        for group_index in range(self.num_groups):
            if group_index % context.world_size != context.rank:
                continue
            if self.weight_mode == "stratified" and self.strata:
                stratum = self.strata[int(rng.integers(0, len(self.strata)))]
                candidates = self.indices_by_stratum[stratum]
                row_idx = int(candidates[int(rng.integers(0, len(candidates)))])
            else:
                row_idx = int(rng.integers(0, len(self.frame)))
            row = self.frame.iloc[row_idx].to_dict()
            sample_id = resolve_dataset_sample_id(dataset, str(row["sample_id"]))
            source_cell_index = int(row[self.index_col])
            item = dataset.build_group(
                sample_id=sample_id,
                source_cell_indices=[source_cell_index],
                group_id=f"anchor:{row_idx}",
                group_type="anchor_csv",
                group_index=int(group_index),
                sample_group_index=int(local_counter),
                normalize_coords=self.normalize_coords,
                include_obs=True,
                anchor_coord=None,
                block_bounds=None,
                metadata={
                    "anchor_row_index": row_idx,
                    "anchor_stratum": row.get(self.strata_column, "all"),
                },
            )
            if not item.get("obs"):
                item["obs"] = [row]
            else:
                item["obs"][0].update({key: value for key, value in row.items() if key not in item["obs"][0]})
            item["sample_id"] = sample_id
            local_counter += 1
            yield item


class MultiTaskOmniCellMemmapCollator(OmniCellMemmapCollator):
    def __init__(
        self,
        *,
        disease_to_id: dict[str, int],
        cell_class_to_id: dict[str, int],
        cohort_to_id: dict[str, int],
        modality_to_id: dict[str, int],
        age_scale: float,
        num_age_bins: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.disease_to_id = disease_to_id
        self.cell_class_to_id = cell_class_to_id
        self.cohort_to_id = cohort_to_id
        self.modality_to_id = modality_to_id
        self.age_scale = float(age_scale)
        self.num_age_bins = int(num_age_bins)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = super().__call__(features)
        disease, cell_class, cohort, modality, age_values, age_mask = [], [], [], [], [], []
        age_bins = []
        for feature in features:
            obs = feature.get("obs") or []
            row = dict(obs[0]) if obs else {}
            sample_id = str(feature.get("sample_id") or row.get("sample_id") or row.get("sample") or "")

            disease_name = infer_condition(row, sample_id)
            disease.append(self.disease_to_id.get(disease_name, -100))

            cell_name = infer_cell_class(row)
            cell_class.append(self.cell_class_to_id.get(cell_name, -100))

            cohort_name = infer_cohort(row, sample_id)
            cohort.append(self.cohort_to_id.get(cohort_name, self.cohort_to_id.get("Other", -100)))

            modality_name = infer_modality(row, sample_id)
            modality.append(self.modality_to_id.get(modality_name, -100))

            age = parse_age(first_present(row, ["age_years", "ageNum", "age", "Age"]))
            if age is None:
                age_values.append(0.0)
                age_mask.append(0.0)
                age_bins.append(-100)
            else:
                age_values.append(float(age) / self.age_scale)
                age_mask.append(1.0)
                bin_value = first_present(row, ["age_bin"])
                if bin_value:
                    try:
                        parsed_bin = int(float(bin_value))
                        age_bins.append(parsed_bin if 0 <= parsed_bin < self.num_age_bins else -100)
                    except ValueError:
                        age_bins.append(-100)
                else:
                    scaled = max(0.0, min(0.999999, float(age) / self.age_scale))
                    age_bins.append(int(scaled * self.num_age_bins))

        batch["disease_labels"] = torch.tensor(disease, dtype=torch.long)
        batch["cell_class_labels"] = torch.tensor(cell_class, dtype=torch.long)
        batch["cohort_labels"] = torch.tensor(cohort, dtype=torch.long)
        batch["modality_labels"] = torch.tensor(modality, dtype=torch.long)
        batch["age_values"] = torch.tensor(age_values, dtype=torch.float32)
        batch["age_mask"] = torch.tensor(age_mask, dtype=torch.float32)
        batch["age_bin_labels"] = torch.tensor(age_bins, dtype=torch.long)
        return batch


def apply_config_overrides(config: OmniCellConfig, args: argparse.Namespace) -> OmniCellConfig:
    config.token_per_cell = int(args.token_per_cell)
    config.n_cells_per_sample = int(args.n_cells_per_sample)
    config.unsupervised_loss = args.unsupervised_loss
    config.unsupervised_loss_on = args.unsupervised_loss_on
    config.load_balance_loss_weight = float(args.load_balance_loss_weight)
    config.use_flash = bool(args.use_flash)
    return config


class GradientReverse(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ge(0)
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid])


def masked_age_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.gt(0)
    if not valid.any():
        return pred.sum() * 0.0
    return F.mse_loss(pred[valid], target[valid])


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    valid = labels.ge(0)
    if valid.sum() <= 1:
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings[valid], dim=-1, eps=1e-7)
    labels = labels[valid]
    logits = torch.matmul(z, z.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive = same & ~eye
    if not positive.any():
        return z.sum() * 0.0
    exp_logits = torch.exp(logits) * (~eye).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    per_anchor = (log_prob * positive.to(logits.dtype)).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -per_anchor[positive.sum(dim=1) > 0].mean()


class OmniCellFigure1MultiTaskModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: OmniCellForUnsupervisedFineTuning,
        num_cell_classes: int,
        num_cohorts: int,
        num_age_bins: int,
        weights: dict[str, float],
        domain_grl_lambda: float,
        supcon_temperature: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        d = backbone.config.d_model
        self.disease_head = nn.Linear(d, len(DISEASE_LABELS))
        self.age_head = nn.Linear(d, 1)
        self.age_bin_head = nn.Linear(d, int(num_age_bins))
        self.cell_class_head = nn.Linear(d, num_cell_classes)
        self.cohort_head = nn.Linear(d, num_cohorts)
        self.modality_head = nn.Linear(d, len(MODALITY_LABELS))
        for module in [self.disease_head, self.age_head, self.age_bin_head, self.cell_class_head, self.cohort_head, self.modality_head]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
        self.weights = dict(weights)
        self.domain_grl_lambda = float(domain_grl_lambda)
        self.supcon_temperature = float(supcon_temperature)

    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        return self.backbone.gradient_checkpointing_enable(**kwargs)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(save_directory, **kwargs)
        torch.save(
            {
                "disease_head": self.disease_head.state_dict(),
                "age_head": self.age_head.state_dict(),
                "age_bin_head": self.age_bin_head.state_dict(),
                "cell_class_head": self.cell_class_head.state_dict(),
                "cohort_head": self.cohort_head.state_dict(),
                "modality_head": self.modality_head.state_dict(),
                "weights": self.weights,
            },
            save_directory / "figure1_multitask_heads.pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        expression_values: torch.Tensor,
        positions: torch.Tensor | None = None,
        nonzero_mask: torch.Tensor | None = None,
        disease_labels: torch.Tensor | None = None,
        age_values: torch.Tensor | None = None,
        age_mask: torch.Tensor | None = None,
        age_bin_labels: torch.Tensor | None = None,
        cell_class_labels: torch.Tensor | None = None,
        cohort_labels: torch.Tensor | None = None,
        modality_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        use_cuda_amp = bool(input_ids.is_cuda and (torch.cuda.is_available()))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
            out = self.backbone(
                input_ids=input_ids,
                expression_values=expression_values,
                positions=positions,
                nonzero_mask=nonzero_mask,
                return_dict=True,
            )
        emb = out.cell_embeddings[:, 0, :]
        head_emb = emb.float()
        disease_logits = self.disease_head(head_emb)
        age_pred = self.age_head(head_emb).squeeze(-1)
        age_bin_logits = self.age_bin_head(head_emb)
        cell_logits = self.cell_class_head(head_emb)
        domain_emb = grad_reverse(head_emb, self.domain_grl_lambda)
        cohort_logits = self.cohort_head(domain_emb)
        modality_logits = self.modality_head(domain_emb)

        recon_loss = out.loss
        disease_loss = masked_cross_entropy(disease_logits, disease_labels) if disease_labels is not None else recon_loss * 0.0
        cell_loss = masked_cross_entropy(cell_logits, cell_class_labels) if cell_class_labels is not None else recon_loss * 0.0
        con_loss = supcon_loss(head_emb, cell_class_labels, self.supcon_temperature) if cell_class_labels is not None else recon_loss * 0.0
        age_loss = masked_age_mse(age_pred, age_values, age_mask) if age_values is not None and age_mask is not None else recon_loss * 0.0
        age_bin_loss = masked_cross_entropy(age_bin_logits, age_bin_labels) if age_bin_labels is not None else recon_loss * 0.0
        age_con_loss = supcon_loss(head_emb, age_bin_labels, self.supcon_temperature) if age_bin_labels is not None else recon_loss * 0.0
        cohort_loss = masked_cross_entropy(cohort_logits, cohort_labels) if cohort_labels is not None else recon_loss * 0.0
        modality_loss = masked_cross_entropy(modality_logits, modality_labels) if modality_labels is not None else recon_loss * 0.0

        loss = (
            self.weights["reconstruction"] * recon_loss
            + self.weights["disease"] * disease_loss
            + self.weights["age"] * age_loss
            + self.weights["age_bin"] * age_bin_loss
            + self.weights["age_supcon"] * age_con_loss
            + self.weights["cell_class"] * cell_loss
            + self.weights["cell_supcon"] * con_loss
            + self.weights["cohort_adversarial"] * cohort_loss
            + self.weights["modality_adversarial"] * modality_loss
        )
        return {
            "loss": loss,
            "disease_logits": disease_logits,
            "age_prediction": age_pred,
            "age_bin_logits": age_bin_logits,
            "cell_class_logits": cell_logits,
            "cohort_logits": cohort_logits,
            "modality_logits": modality_logits,
            "cell_embeddings": head_emb,
            "reconstruction_loss": recon_loss.detach(),
            "disease_loss": disease_loss.detach(),
            "age_loss": age_loss.detach(),
            "age_bin_loss": age_bin_loss.detach(),
            "age_supcon_loss": age_con_loss.detach(),
            "cell_class_loss": cell_loss.detach(),
            "cell_supcon_loss": con_loss.detach(),
            "cohort_adversarial_loss": cohort_loss.detach(),
            "modality_adversarial_loss": modality_loss.detach(),
        }


def build_backbone(args: argparse.Namespace) -> OmniCellForUnsupervisedFineTuning:
    config = apply_config_overrides(OmniCellConfig.from_pretrained(args.model_name_or_path), args)
    model = OmniCellForUnsupervisedFineTuning(config)
    path = Path(args.model_name_or_path)
    if (path / "model.safetensors").exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(path / "model.safetensors"), device="cpu")
    elif (path / "pytorch_model.bin").exists():
        state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    if any(key.startswith("omnicell.") for key in state_dict):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded_as = "unsupervised"
    else:
        missing, unexpected = model.omnicell.load_state_dict(state_dict, strict=False)
        loaded_as = "backbone"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoint_load_report.json").write_text(
        json.dumps(
            {
                "source": str(path),
                "loaded_as": loaded_as,
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def build_dataset_and_collator(args: argparse.Namespace, samples: list[dict[str, Any]], config: OmniCellConfig):
    sample_ids = resolve_training_sample_ids(args, samples)
    cell_classes = split_csv(args.cell_class_labels)
    cohorts = split_csv(args.cohort_labels)
    if "Other" not in cohorts:
        cohorts.append("Other")
    if args.anchor_csv is not None:
        dataset = MultiTaskAnchorCsvDataset(
            dataset_root=args.dataset_root,
            anchor_csv=args.anchor_csv,
            num_groups=args.num_groups,
            seed=args.seed,
            strata_column=args.anchor_strata_column,
            weight_mode=args.anchor_weight_mode,
            normalize_coords=args.normalize_coords,
        )
    else:
        dataset = MultiTaskMemmapRandomGroupDataset(
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
    collator = MultiTaskOmniCellMemmapCollator(
        dataset_root=args.dataset_root,
        token_per_cell=args.token_per_cell,
        vocab_size=config.vocab_size,
        start_token_id=config.start_token_id,
        end_token_id=config.end_token_id,
        max_expression_token_id=max_expression_token_id,
        use_smooth_rank=args.use_smooth_rank,
        smooth_rank_range=(args.smooth_rank_min, args.smooth_rank_max),
        selection_strategy=args.selection_strategy,
        disease_to_id={label: i for i, label in enumerate(DISEASE_LABELS)},
        cell_class_to_id={label: i for i, label in enumerate(cell_classes)},
        cohort_to_id={label: i for i, label in enumerate(cohorts)},
        modality_to_id={label: i for i, label in enumerate(MODALITY_LABELS)},
        age_scale=args.age_scale,
        num_age_bins=args.num_age_bins,
    )
    return dataset, collator, sample_ids, cell_classes, cohorts


def save_run_config(args: argparse.Namespace, samples: list[dict[str, Any]], sample_ids: list[str] | None, cell_classes: list[str], cohorts: list[str]) -> None:
    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_indexed_samples": len(samples),
        "num_selected_samples": len(sample_ids) if sample_ids is not None else len(samples),
        "selected_sample_ids": sample_ids,
        "disease_labels": DISEASE_LABELS,
        "cell_class_labels": cell_classes,
        "cohort_labels": cohorts,
        "modality_labels": MODALITY_LABELS,
        "objective": "reconstruction + disease classification + age regression + cell-class classification/SupCon + cohort/modality gradient-reversal alignment",
        "loss_weights": {
            "reconstruction": args.reconstruction_loss_weight,
            "disease": args.disease_loss_weight,
            "age": args.age_loss_weight,
            "age_bin": args.age_bin_loss_weight,
            "age_supcon": args.age_supcon_loss_weight,
            "cell_class": args.cell_class_loss_weight,
            "cell_supcon": args.cell_supcon_loss_weight,
            "cohort_adversarial": args.cohort_adversarial_loss_weight,
            "modality_adversarial": args.modality_adversarial_loss_weight,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_sequence_shape(args)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    samples = prepare_cellfm_dataset_root(args)
    backbone = build_backbone(args)
    dataset, collator, sample_ids, cell_classes, cohorts = build_dataset_and_collator(args, samples, backbone.config)
    weights = {
            "reconstruction": args.reconstruction_loss_weight,
            "disease": args.disease_loss_weight,
            "age": args.age_loss_weight,
            "age_bin": args.age_bin_loss_weight,
            "age_supcon": args.age_supcon_loss_weight,
            "cell_class": args.cell_class_loss_weight,
            "cell_supcon": args.cell_supcon_loss_weight,
        "cohort_adversarial": args.cohort_adversarial_loss_weight,
        "modality_adversarial": args.modality_adversarial_loss_weight,
    }
    model = OmniCellFigure1MultiTaskModel(
        backbone=backbone,
        num_cell_classes=len(cell_classes),
        num_cohorts=len(cohorts),
        num_age_bins=args.num_age_bins,
        weights=weights,
        domain_grl_lambda=args.domain_grl_lambda,
        supcon_temperature=args.supcon_temperature,
    )
    save_run_config(args, samples, sample_ids, cell_classes, cohorts)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

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
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collator)
    start = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if args.save_backbone_subdir:
        model.backbone.omnicell.save_pretrained(args.output_dir / args.save_backbone_subdir)
    (args.output_dir / "train_done.json").write_text(json.dumps({"elapsed_sec": time.time() - start}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
