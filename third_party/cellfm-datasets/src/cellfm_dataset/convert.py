"""Offline conversion from H5AD manifests to CSR memmap datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._optional import require_dependency
from .common import (
    DEFAULT_EXPRESSION_TRANSFORM,
    apply_expression_transform,
    align_matrix_to_vocab,
    build_gene_union,
    ensure_csr_matrix,
    extract_coords,
    load_manifest_items,
    normalize_column_names,
    normalize_coord_axis_names,
    normalize_expression_transform,
    read_gene_list,
    write_memmap_npy,
)
from .schema import DatasetMetadata, SampleMetadata


def resolve_gene_vocab(
    *,
    gene_vocab_path: str | Path | None = None,
    union_gene_lists: tuple[str | Path, ...] | None = None,
) -> list[str]:
    """Resolve a final ordered gene vocabulary."""

    if gene_vocab_path is not None:
        return read_gene_list(Path(gene_vocab_path).expanduser().resolve())

    if union_gene_lists:
        return build_gene_union(*(Path(item).expanduser().resolve() for item in union_gene_lists))

    raise ValueError("Provide either gene_vocab_path or union_gene_lists.")


def convert_one_sample(
    *,
    item: dict,
    output_dir: str | Path,
    vocab: list[str],
    vocab_to_index: dict[str, int],
    normalize_target_sum: float,
    expression_transform: str = DEFAULT_EXPRESSION_TRANSFORM,
    require_coords: bool = False,
    overwrite: bool = False,
) -> SampleMetadata:
    """Convert one H5AD file into per-sample CSR memmap arrays."""

    ad = require_dependency("anndata", "pip install 'cellfm-datasets[convert]'")

    output_dir = Path(output_dir).expanduser().resolve()
    sample_id = item["sample_id"]
    h5ad_path = Path(item["h5ad"]).expanduser().resolve()
    sample_dir = output_dir / sample_id

    if sample_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for filename in (
            "indptr.npy",
            "indices.npy",
            "values.npy",
            "coords.npy",
            "raw_cell_sums.npy",
            "present_gene_ids.npy",
            "obs_names.npy",
            "obs.parquet",
            "metadata.json",
        ):
            target = sample_dir / filename
            if target.exists():
                target.unlink()

    adata = ad.read_h5ad(h5ad_path)
    coords = extract_coords(
        adata,
        z_column=item.get("z_column"),
        require_coords=bool(item.get("require_coords", require_coords)),
    )
    expr = ensure_csr_matrix(adata.X)
    if expr.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{sample_id}: matrix rows ({expr.shape[0]}) and coords rows ({coords.shape[0]}) mismatch."
        )
    finite_coord_mask = np.all(np.isfinite(coords), axis=1)

    aligned, align_stats = align_matrix_to_vocab(
        expr,
        adata.var_names,
        vocab_to_index=vocab_to_index,
        vocab_size=len(vocab),
    )
    resolved_expression_transform = normalize_expression_transform(
        expression_transform if item.get("expression_transform") is None else item.get("expression_transform")
    )
    transformed, raw_cell_sums = apply_expression_transform(
        aligned,
        expression_transform=resolved_expression_transform,
        normalize_target_sum=normalize_target_sum,
    )

    write_memmap_npy(sample_dir / "indptr.npy", transformed.indptr.astype(np.int64, copy=False))
    write_memmap_npy(sample_dir / "indices.npy", transformed.indices.astype(np.int32, copy=False))
    write_memmap_npy(sample_dir / "values.npy", transformed.data.astype(np.float32, copy=False))
    write_memmap_npy(sample_dir / "coords.npy", coords.astype(np.float32, copy=False))
    write_memmap_npy(sample_dir / "raw_cell_sums.npy", raw_cell_sums.astype(np.float32, copy=False))

    present_gene_ids = np.unique(transformed.indices).astype(np.int32, copy=False)
    write_memmap_npy(sample_dir / "present_gene_ids.npy", present_gene_ids)

    obs_names = adata.obs_names.to_numpy(dtype=str, copy=True)
    np.save(sample_dir / "obs_names.npy", obs_names)

    requested_obs_columns = normalize_column_names(item.get("obs_columns"))
    stored_obs_columns: tuple[str, ...] = ()
    if requested_obs_columns is not None:
        require_dependency("pyarrow", "pip install 'cellfm-datasets[convert]'")
        if requested_obs_columns == ("*",):
            obs_frame = adata.obs.copy()
        else:
            missing = [column for column in requested_obs_columns if column not in adata.obs.columns]
            if missing:
                raise ValueError(f"{sample_id}: obs columns not found in adata.obs: {missing}")
            obs_frame = adata.obs.loc[:, list(requested_obs_columns)].copy()
        obs_frame.insert(0, "obs_name", obs_names)
        obs_frame.to_parquet(sample_dir / "obs.parquet", index=False)
        stored_obs_columns = tuple(str(column) for column in obs_frame.columns if column != "obs_name")

    coord_axis_names = normalize_coord_axis_names(item.get("coord_axis_names"), int(coords.shape[1]))
    metadata = SampleMetadata(
        sample_id=sample_id,
        source_h5ad=str(h5ad_path),
        z_column=item.get("z_column"),
        n_cells=int(transformed.shape[0]),
        n_genes=int(transformed.shape[1]),
        nnz=int(transformed.nnz),
        coord_dim=int(coords.shape[1]),
        coord_axis_names=coord_axis_names,
        n_nonfinite_coord_rows=int((~finite_coord_mask).sum()),
        expression_transform=resolved_expression_transform,
        normalize_target_sum=(
            float(normalize_target_sum)
            if resolved_expression_transform == "normalize_total_log1p"
            else None
        ),
        kept_source_genes=int(align_stats["kept_source_genes"]),
        dropped_source_genes=int(align_stats["dropped_source_genes"]),
        n_present_genes=int(present_gene_ids.shape[0]),
        obs_columns=stored_obs_columns,
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def convert_h5ad_manifest(
    *,
    manifest_json: str | Path,
    output_dir: str | Path,
    gene_vocab_path: str | Path | None = None,
    union_gene_lists: tuple[str | Path, ...] | None = None,
    default_obs_columns: tuple[str, ...] | None = None,
    expression_transform: str = DEFAULT_EXPRESSION_TRANSFORM,
    normalize_target_sum: float = 1e4,
    require_coords: bool = False,
    overwrite: bool = False,
) -> DatasetMetadata:
    """Convert a JSON manifest of H5AD files into a memmap dataset."""

    manifest_json = Path(manifest_json).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_manifest_items(manifest_json)
    if default_obs_columns is not None:
        items = [
            {
                **item,
                "obs_columns": item["obs_columns"]
                if item.get("obs_columns") is not None
                else default_obs_columns,
            }
            for item in items
        ]
    items = [
        {
            **item,
            "expression_transform": (
                expression_transform
                if item.get("expression_transform") is None
                else item.get("expression_transform")
            ),
        }
        for item in items
    ]
    vocab = resolve_gene_vocab(gene_vocab_path=gene_vocab_path, union_gene_lists=union_gene_lists)
    vocab_to_index = {gene: idx for idx, gene in enumerate(vocab)}
    (output_dir / "gene_vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    samples = tuple(
        convert_one_sample(
            item=item,
            output_dir=output_dir,
            vocab=vocab,
            vocab_to_index=vocab_to_index,
            normalize_target_sum=normalize_target_sum,
            expression_transform=expression_transform,
            require_coords=require_coords,
            overwrite=overwrite,
        )
        for item in items
    )
    sample_transforms = sorted({sample.expression_transform for sample in samples})
    sample_target_sums = sorted(
        {
            float(sample.normalize_target_sum)
            for sample in samples
            if sample.normalize_target_sum is not None
        }
    )
    metadata = DatasetMetadata(
        dataset_dir=str(output_dir),
        input_manifest_json=str(manifest_json),
        n_genes=len(vocab),
        expression_transform=sample_transforms[0] if len(sample_transforms) == 1 else "mixed",
        normalize_target_sum=sample_target_sums[0] if len(sample_target_sums) == 1 else None,
        samples=samples,
        gene_vocab_paths=None
        if gene_vocab_path is not None
        else {
            f"union_{idx}": str(Path(path).expanduser().resolve())
            for idx, path in enumerate(union_gene_lists or ())
        },
    )
    metadata.save(output_dir)
    return metadata
