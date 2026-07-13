"""Typed metadata containers for the memmap dataset protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import normalize_coord_axis_names


@dataclass(frozen=True)
class SampleMetadata:
    """Per-sample metadata persisted alongside CSR memmap arrays."""

    sample_id: str
    source_h5ad: str
    z_column: str | None
    n_cells: int
    n_genes: int
    nnz: int
    coord_dim: int
    coord_axis_names: tuple[str, ...]
    n_nonfinite_coord_rows: int
    expression_transform: str
    normalize_target_sum: float | None
    kept_source_genes: int
    dropped_source_genes: int
    n_present_genes: int
    obs_columns: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleMetadata":
        coord_dim = int(data["coord_dim"])
        return cls(
            sample_id=str(data["sample_id"]),
            source_h5ad=str(data.get("source_h5ad", "")),
            z_column=None if data.get("z_column") is None else str(data.get("z_column")),
            n_cells=int(data["n_cells"]),
            n_genes=int(data["n_genes"]),
            nnz=int(data["nnz"]),
            coord_dim=coord_dim,
            coord_axis_names=normalize_coord_axis_names(data.get("coord_axis_names"), coord_dim),
            n_nonfinite_coord_rows=int(data.get("n_nonfinite_coord_rows", 0)),
            expression_transform=str(data.get("expression_transform", "normalize_total_log1p")),
            normalize_target_sum=None
            if data.get("normalize_target_sum") is None
            else float(data["normalize_target_sum"]),
            kept_source_genes=int(data.get("kept_source_genes", 0)),
            dropped_source_genes=int(data.get("dropped_source_genes", 0)),
            n_present_genes=int(data.get("n_present_genes", 0)),
            obs_columns=tuple(str(item) for item in data.get("obs_columns", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_h5ad": self.source_h5ad,
            "z_column": self.z_column,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "nnz": self.nnz,
            "coord_dim": self.coord_dim,
            "coord_axis_names": list(self.coord_axis_names),
            "n_nonfinite_coord_rows": self.n_nonfinite_coord_rows,
            "expression_transform": self.expression_transform,
            "normalize_target_sum": self.normalize_target_sum,
            "kept_source_genes": self.kept_source_genes,
            "dropped_source_genes": self.dropped_source_genes,
            "n_present_genes": self.n_present_genes,
            "obs_columns": list(self.obs_columns),
        }


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset-level metadata persisted at the root of the memmap dataset."""

    dataset_dir: str
    input_manifest_json: str
    n_genes: int
    expression_transform: str
    normalize_target_sum: float | None
    samples: tuple[SampleMetadata, ...]
    gene_vocab_paths: dict[str, str] | None = None

    @classmethod
    def load(cls, dataset_root: str | Path) -> "DatasetMetadata":
        dataset_root = Path(dataset_root).expanduser().resolve()
        payload = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetMetadata":
        return cls(
            dataset_dir=str(data["dataset_dir"]),
            input_manifest_json=str(data.get("input_manifest_json", "")),
            n_genes=int(data["n_genes"]),
            expression_transform=str(data.get("expression_transform", "normalize_total_log1p")),
            normalize_target_sum=None
            if data.get("normalize_target_sum") is None
            else float(data["normalize_target_sum"]),
            samples=tuple(SampleMetadata.from_dict(item) for item in data.get("samples", [])),
            gene_vocab_paths=None
            if data.get("gene_vocab_paths") is None
            else {str(k): str(v) for k, v in dict(data["gene_vocab_paths"]).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset_dir": self.dataset_dir,
            "input_manifest_json": self.input_manifest_json,
            "n_genes": self.n_genes,
            "expression_transform": self.expression_transform,
            "normalize_target_sum": self.normalize_target_sum,
            "samples": [sample.to_dict() for sample in self.samples],
        }
        if self.gene_vocab_paths is not None:
            payload["gene_vocab_paths"] = dict(self.gene_vocab_paths)
        return payload

    def save(self, dataset_root: str | Path) -> None:
        dataset_root = Path(dataset_root).expanduser().resolve()
        (dataset_root / "dataset_manifest.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def sample_map(self) -> dict[str, SampleMetadata]:
        return {sample.sample_id: sample for sample in self.samples}
