"""Helpers for loading original OmniCell checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .configuration_omnicell import OmniCellConfig
from .modeling_omnicell import (
    OmniCellForSupervisedFineTuning,
    OmniCellForUnsupervisedFineTuning,
    OmniCellModel,
)


def load_legacy_config(
    checkpoint_dir: str | Path,
    *,
    token_per_cell: int | None = None,
    n_cells_per_sample: int | None = None,
    num_labels: int | None = None,
    problem_type: str | None = None,
) -> OmniCellConfig:
    checkpoint_dir = Path(checkpoint_dir)
    config_data = json.loads((checkpoint_dir / "LMConfig.json").read_text())
    config = OmniCellConfig.from_legacy_dict(config_data)
    if token_per_cell is not None:
        config.token_per_cell = int(token_per_cell)
    if n_cells_per_sample is not None:
        config.n_cells_per_sample = int(n_cells_per_sample)
    if num_labels is not None:
        config.num_labels = int(num_labels)
    if problem_type is not None:
        config.problem_type = problem_type
    return config


def load_legacy_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "backbone.pth"
    if not state_path.exists():
        raise FileNotFoundError(f"Legacy checkpoint not found: {state_path}")
    return torch.load(state_path, map_location="cpu", weights_only=True)


def build_model_from_legacy(
    checkpoint_dir: str | Path,
    model_type: str = "backbone",
    **config_overrides,
):
    config = load_legacy_config(checkpoint_dir, **config_overrides)
    if model_type == "backbone":
        model = OmniCellModel(config)
        target = model
    elif model_type == "unsupervised":
        model = OmniCellForUnsupervisedFineTuning(config)
        target = model.omnicell
    elif model_type == "supervised":
        model = OmniCellForSupervisedFineTuning(config)
        target = model.omnicell
    else:
        raise ValueError("model_type must be one of: backbone, unsupervised, supervised.")

    state_dict = load_legacy_state_dict(checkpoint_dir)
    missing, unexpected = target.load_state_dict(state_dict, strict=False)
    return model, {"missing_keys": missing, "unexpected_keys": unexpected}
