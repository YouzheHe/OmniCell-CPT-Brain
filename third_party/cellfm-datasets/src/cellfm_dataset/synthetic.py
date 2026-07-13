"""Public synthetic benchmark data generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ._optional import require_dependency


PUBLIC_BENCHMARK_PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "n_genes": 512,
        "n_cell_types": 6,
        "n_drugs": 8,
        "n_moa_broad": 3,
        "n_moa_fine": 4,
        "sc_n_samples": 3,
        "sc_cells_per_sample": 2048,
        "st_n_samples": 2,
        "st_cells_per_sample": 1024,
        "n_spatial_domains": 4,
    },
    "medium": {
        "n_genes": 1024,
        "n_cell_types": 8,
        "n_drugs": 12,
        "n_moa_broad": 4,
        "n_moa_fine": 6,
        "sc_n_samples": 4,
        "sc_cells_per_sample": 8192,
        "st_n_samples": 3,
        "st_cells_per_sample": 4096,
        "n_spatial_domains": 6,
    },
}


def _anndata():
    return require_dependency("anndata", "pip install 'cellfm-datasets[convert]'")


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _sample_programs(
    rng: np.random.Generator,
    *,
    n_programs: int,
    n_genes: int,
    program_size: int,
) -> list[np.ndarray]:
    return [
        np.sort(rng.choice(n_genes, size=min(program_size, n_genes), replace=False).astype(np.int32))
        for _ in range(n_programs)
    ]


def _weighted_choice(rng: np.random.Generator, weights: np.ndarray, size: int) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(np.arange(probabilities.shape[0]), size=size, replace=True, p=probabilities)


def _sample_sparse_rows(
    rng: np.random.Generator,
    *,
    n_rows: int,
    n_genes: int,
    primary_labels: np.ndarray,
    primary_programs: list[np.ndarray],
    secondary_labels: np.ndarray,
    secondary_programs: list[np.ndarray],
    tertiary_labels: np.ndarray,
    tertiary_programs: list[np.ndarray],
    extra_programs: list[np.ndarray] | None = None,
    extra_labels: np.ndarray | None = None,
    primary_take: int = 24,
    secondary_take: int = 12,
    tertiary_take: int = 8,
    extra_take: int = 12,
    noise_take: int = 12,
) -> sparse.csr_matrix:
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    indices_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    cursor = 0

    for row_idx in range(n_rows):
        picked_cols: list[np.ndarray] = []
        picked_vals: list[np.ndarray] = []

        primary_cols = rng.choice(
            primary_programs[int(primary_labels[row_idx])],
            size=min(primary_take, primary_programs[int(primary_labels[row_idx])].shape[0]),
            replace=False,
        )
        picked_cols.append(primary_cols)
        picked_vals.append(rng.lognormal(mean=1.8, sigma=0.35, size=primary_cols.shape[0]).astype(np.float32))

        secondary_cols = rng.choice(
            secondary_programs[int(secondary_labels[row_idx])],
            size=min(secondary_take, secondary_programs[int(secondary_labels[row_idx])].shape[0]),
            replace=False,
        )
        picked_cols.append(secondary_cols)
        picked_vals.append(rng.lognormal(mean=1.4, sigma=0.30, size=secondary_cols.shape[0]).astype(np.float32))

        tertiary_cols = rng.choice(
            tertiary_programs[int(tertiary_labels[row_idx])],
            size=min(tertiary_take, tertiary_programs[int(tertiary_labels[row_idx])].shape[0]),
            replace=False,
        )
        picked_cols.append(tertiary_cols)
        picked_vals.append(rng.lognormal(mean=1.0, sigma=0.25, size=tertiary_cols.shape[0]).astype(np.float32))

        if extra_programs is not None and extra_labels is not None:
            extra_cols = rng.choice(
                extra_programs[int(extra_labels[row_idx])],
                size=min(extra_take, extra_programs[int(extra_labels[row_idx])].shape[0]),
                replace=False,
            )
            picked_cols.append(extra_cols)
            picked_vals.append(rng.lognormal(mean=1.5, sigma=0.30, size=extra_cols.shape[0]).astype(np.float32))

        noise_cols = rng.choice(n_genes, size=min(noise_take, n_genes), replace=False).astype(np.int32)
        picked_cols.append(noise_cols)
        picked_vals.append(rng.lognormal(mean=0.2, sigma=0.40, size=noise_cols.shape[0]).astype(np.float32))

        row_cols = np.concatenate(picked_cols, axis=0)
        row_vals = np.concatenate(picked_vals, axis=0)
        order = np.argsort(row_cols, kind="mergesort")
        row_cols = row_cols[order]
        row_vals = row_vals[order]

        unique_cols, inverse = np.unique(row_cols, return_inverse=True)
        merged_vals = np.zeros(unique_cols.shape[0], dtype=np.float32)
        np.add.at(merged_vals, inverse, row_vals)

        indices_parts.append(unique_cols.astype(np.int32, copy=False))
        data_parts.append(merged_vals)
        cursor += unique_cols.shape[0]
        indptr[row_idx + 1] = cursor

    indices = np.concatenate(indices_parts, axis=0) if indices_parts else np.empty((0,), dtype=np.int32)
    data = np.concatenate(data_parts, axis=0) if data_parts else np.empty((0,), dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_genes), dtype=np.float32)


def _write_manifest(output_dir: Path, items: list[dict[str, Any]], filename: str) -> Path:
    path = output_dir / filename
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def resolve_public_benchmark_preset(name: str) -> dict[str, int]:
    """Resolve a named synthetic benchmark preset."""

    preset_name = str(name).strip().lower()
    if preset_name not in PUBLIC_BENCHMARK_PRESETS:
        raise ValueError(
            f"Unsupported scale preset '{name}'. Supported values: {sorted(PUBLIC_BENCHMARK_PRESETS)}"
        )
    return dict(PUBLIC_BENCHMARK_PRESETS[preset_name])


def generate_synthetic_singlecell_cohort(
    *,
    output_dir: str | Path,
    n_samples: int = 4,
    cells_per_sample: int = 4096,
    n_genes: int = 1024,
    n_cell_types: int = 8,
    n_drugs: int = 12,
    n_moa_broad: int = 4,
    n_moa_fine: int = 6,
    seed: int = 7,
) -> dict[str, str]:
    """Generate a public single-cell benchmark cohort as H5AD files plus manifest."""

    ad = _anndata()

    output_root = _ensure_dir(output_dir)
    genes = [f"gene_{idx:05d}" for idx in range(n_genes)]
    gene_vocab_path = output_root / "gene_vocab.txt"
    gene_vocab_path.write_text("\n".join(genes) + "\n", encoding="utf-8")

    rng = np.random.default_rng(seed)
    cell_type_programs = _sample_programs(rng, n_programs=n_cell_types, n_genes=n_genes, program_size=80)
    drug_programs = _sample_programs(rng, n_programs=n_drugs, n_genes=n_genes, program_size=48)
    plate_programs = _sample_programs(rng, n_programs=n_samples, n_genes=n_genes, program_size=32)

    manifest_items: list[dict[str, Any]] = []
    moa_fine_by_drug = np.arange(n_drugs, dtype=np.int32) % n_moa_fine
    moa_broad_by_drug = moa_fine_by_drug % n_moa_broad

    for sample_idx in range(n_samples):
        sample_id = f"plate_{sample_idx:02d}"
        sample_path = output_root / f"{sample_id}.h5ad"

        preferred_cell_types = rng.choice(n_cell_types, size=max(2, n_cell_types // 2), replace=False)
        preferred_drugs = rng.choice(n_drugs, size=max(3, n_drugs // 3), replace=False)

        cell_type_weights = np.ones(n_cell_types, dtype=np.float64)
        cell_type_weights[preferred_cell_types] *= 5.0
        drug_weights = np.ones(n_drugs, dtype=np.float64)
        drug_weights[preferred_drugs] *= 6.0

        donor_labels = _weighted_choice(rng, np.array([3.0, 2.0, 1.0]), size=cells_per_sample)
        batch_labels = np.full(cells_per_sample, sample_idx % 2, dtype=np.int32)
        cell_type_labels = _weighted_choice(rng, cell_type_weights, size=cells_per_sample)
        drug_labels = _weighted_choice(rng, drug_weights, size=cells_per_sample)
        moa_fine_labels = moa_fine_by_drug[drug_labels]
        moa_broad_labels = moa_broad_by_drug[drug_labels]
        plate_labels = np.full(cells_per_sample, sample_idx, dtype=np.int32)

        order = np.lexsort((cell_type_labels, drug_labels, batch_labels))
        donor_labels = donor_labels[order]
        batch_labels = batch_labels[order]
        cell_type_labels = cell_type_labels[order]
        drug_labels = drug_labels[order]
        moa_fine_labels = moa_fine_labels[order]
        moa_broad_labels = moa_broad_labels[order]
        plate_labels = plate_labels[order]

        expression = _sample_sparse_rows(
            rng,
            n_rows=cells_per_sample,
            n_genes=n_genes,
            primary_labels=cell_type_labels,
            primary_programs=cell_type_programs,
            secondary_labels=drug_labels,
            secondary_programs=drug_programs,
            tertiary_labels=plate_labels,
            tertiary_programs=plate_programs,
            primary_take=26,
            secondary_take=14,
            tertiary_take=10,
            noise_take=14,
        )

        obs = pd.DataFrame(
            {
                "plate": [sample_id] * cells_per_sample,
                "plate_index": plate_labels,
                "cell_type": [f"cell_type_{idx:02d}" for idx in cell_type_labels],
                "cell_type_index": cell_type_labels,
                "drug": [f"drug_{idx:02d}" for idx in drug_labels],
                "drug_index": drug_labels,
                "moa_broad": [f"moa_broad_{idx:02d}" for idx in moa_broad_labels],
                "moa_fine": [f"moa_fine_{idx:02d}" for idx in moa_fine_labels],
                "donor_id": [f"donor_{idx:02d}" for idx in donor_labels],
                "batch": [f"batch_{idx:02d}" for idx in batch_labels],
            },
            index=[f"{sample_id}_cell_{idx:07d}" for idx in range(cells_per_sample)],
        )
        var = pd.DataFrame(index=genes)
        adata = ad.AnnData(X=expression, obs=obs, var=var)
        adata.write_h5ad(sample_path)
        manifest_items.append({"h5ad": str(sample_path), "sample_id": sample_id})

    manifest_path = _write_manifest(output_root, manifest_items, "synthetic_singlecell_manifest.json")
    summary_path = output_root / "synthetic_singlecell_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task": "singlecell",
                "n_samples": n_samples,
                "cells_per_sample": cells_per_sample,
                "n_genes": n_genes,
                "n_cell_types": n_cell_types,
                "n_drugs": n_drugs,
                "n_moa_broad": n_moa_broad,
                "n_moa_fine": n_moa_fine,
                "seed": seed,
                "manifest_json": str(manifest_path),
                "gene_vocab": str(gene_vocab_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "manifest_json": str(manifest_path),
        "gene_vocab": str(gene_vocab_path),
        "summary_json": str(summary_path),
    }


def generate_synthetic_spatial_cohort(
    *,
    output_dir: str | Path,
    n_samples: int = 2,
    cells_per_sample: int = 4096,
    n_genes: int = 1024,
    n_cell_types: int = 6,
    n_domains: int = 4,
    seed: int = 17,
) -> dict[str, str]:
    """Generate a public spatial benchmark cohort as H5AD files plus manifest."""

    ad = _anndata()

    output_root = _ensure_dir(output_dir)
    genes = [f"gene_{idx:05d}" for idx in range(n_genes)]
    gene_vocab_path = output_root / "gene_vocab.txt"
    gene_vocab_path.write_text("\n".join(genes) + "\n", encoding="utf-8")

    rng = np.random.default_rng(seed)
    cell_type_programs = _sample_programs(rng, n_programs=n_cell_types, n_genes=n_genes, program_size=72)
    domain_programs = _sample_programs(rng, n_programs=n_domains, n_genes=n_genes, program_size=72)
    plate_programs = _sample_programs(rng, n_programs=n_samples, n_genes=n_genes, program_size=24)

    manifest_items: list[dict[str, Any]] = []
    for sample_idx in range(n_samples):
        sample_id = f"section_{sample_idx:02d}"
        sample_path = output_root / f"{sample_id}.h5ad"

        side = int(np.ceil(np.sqrt(cells_per_sample)))
        xs = np.tile(np.arange(side, dtype=np.float32), side)[:cells_per_sample]
        ys = np.repeat(np.arange(side, dtype=np.float32), side)[:cells_per_sample]
        coords = np.stack([xs, ys], axis=1)

        grid_side = int(np.ceil(np.sqrt(n_domains)))
        x_bins = np.minimum((xs * grid_side / max(side, 1)).astype(np.int32), grid_side - 1)
        y_bins = np.minimum((ys * grid_side / max(side, 1)).astype(np.int32), grid_side - 1)
        domain_labels = np.minimum(y_bins * grid_side + x_bins, n_domains - 1)

        cell_type_weights = np.ones((n_domains, n_cell_types), dtype=np.float64)
        for domain_idx in range(n_domains):
            favored = domain_idx % n_cell_types
            cell_type_weights[domain_idx, favored] *= 5.0
            cell_type_weights[domain_idx, (favored + 1) % n_cell_types] *= 3.0

        cell_type_labels = np.empty(cells_per_sample, dtype=np.int32)
        for idx in range(cells_per_sample):
            cell_type_labels[idx] = _weighted_choice(
                rng,
                cell_type_weights[int(domain_labels[idx])],
                size=1,
            )[0]

        plate_labels = np.full(cells_per_sample, sample_idx, dtype=np.int32)
        order = np.lexsort((ys, xs))
        coords = coords[order]
        domain_labels = domain_labels[order]
        cell_type_labels = cell_type_labels[order]
        plate_labels = plate_labels[order]

        expression = _sample_sparse_rows(
            rng,
            n_rows=cells_per_sample,
            n_genes=n_genes,
            primary_labels=domain_labels,
            primary_programs=domain_programs,
            secondary_labels=cell_type_labels,
            secondary_programs=cell_type_programs,
            tertiary_labels=plate_labels,
            tertiary_programs=plate_programs,
            primary_take=22,
            secondary_take=14,
            tertiary_take=8,
            noise_take=14,
        )

        obs = pd.DataFrame(
            {
                "section": [sample_id] * cells_per_sample,
                "section_index": plate_labels,
                "spatial_domain": [f"domain_{idx:02d}" for idx in domain_labels],
                "spatial_domain_index": domain_labels,
                "cell_type": [f"cell_type_{idx:02d}" for idx in cell_type_labels],
                "cell_type_index": cell_type_labels,
            },
            index=[f"{sample_id}_spot_{idx:07d}" for idx in range(cells_per_sample)],
        )
        var = pd.DataFrame(index=genes)
        adata = ad.AnnData(X=expression, obs=obs, var=var)
        adata.obsm["spatial"] = coords.astype(np.float32, copy=False)
        adata.write_h5ad(sample_path)
        manifest_items.append({"h5ad": str(sample_path), "sample_id": sample_id})

    manifest_path = _write_manifest(output_root, manifest_items, "synthetic_spatial_manifest.json")
    summary_path = output_root / "synthetic_spatial_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task": "spatial",
                "n_samples": n_samples,
                "cells_per_sample": cells_per_sample,
                "n_genes": n_genes,
                "n_cell_types": n_cell_types,
                "n_domains": n_domains,
                "seed": seed,
                "manifest_json": str(manifest_path),
                "gene_vocab": str(gene_vocab_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "manifest_json": str(manifest_path),
        "gene_vocab": str(gene_vocab_path),
        "summary_json": str(summary_path),
    }


def generate_public_synthetic_benchmark_data(
    *,
    output_dir: str | Path,
    mode: str = "both",
    scale_preset: str = "smoke",
    seed: int = 7,
) -> dict[str, Any]:
    """Generate public synthetic benchmark inputs for the open-source repo."""

    selected_mode = str(mode).strip().lower()
    if selected_mode not in {"singlecell", "spatial", "both"}:
        raise ValueError("mode must be one of: singlecell, spatial, both")

    output_root = _ensure_dir(output_dir)
    preset = resolve_public_benchmark_preset(scale_preset)

    payload: dict[str, Any] = {
        "mode": selected_mode,
        "scale_preset": str(scale_preset).strip().lower(),
        "seed": int(seed),
        "artifacts": {},
    }

    if selected_mode in {"singlecell", "both"}:
        payload["artifacts"]["singlecell"] = generate_synthetic_singlecell_cohort(
            output_dir=output_root / "singlecell",
            n_samples=preset["sc_n_samples"],
            cells_per_sample=preset["sc_cells_per_sample"],
            n_genes=preset["n_genes"],
            n_cell_types=preset["n_cell_types"],
            n_drugs=preset["n_drugs"],
            n_moa_broad=preset["n_moa_broad"],
            n_moa_fine=preset["n_moa_fine"],
            seed=seed,
        )

    if selected_mode in {"spatial", "both"}:
        payload["artifacts"]["spatial"] = generate_synthetic_spatial_cohort(
            output_dir=output_root / "spatial",
            n_samples=preset["st_n_samples"],
            cells_per_sample=preset["st_cells_per_sample"],
            n_genes=preset["n_genes"],
            n_cell_types=preset["n_cell_types"],
            n_domains=preset["n_spatial_domains"],
            seed=seed + 10_000,
        )

    summary_path = output_root / "public_synthetic_benchmark_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_path)
    return payload
