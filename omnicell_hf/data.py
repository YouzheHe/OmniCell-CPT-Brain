"""H5AD tokenization and datasets for Hugging Face OmniCell training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset


def _load_anndata(path: str | Path, backed: bool = False):
    import anndata as ad

    return ad.read_h5ad(path, backed="r" if backed else None)


def _read_gene_list(value: str | Path | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    text = str(value).strip()
    if not text:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def _coerce_int_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.empty((0,), dtype=np.int64)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [int(part) for part in text.split(",") if part.strip()]
        return np.asarray(parsed, dtype=np.int64)
    return np.asarray(value, dtype=np.int64)


def _coerce_ref_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, pd.Series)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [part.strip() for part in text.split(",") if part.strip()]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


class GeneVocab:
    """Gene vocabulary with optional gene-symbol to Ensembl alias mapping."""

    def __init__(self, vocab_path: str | Path, alias_csv: str | Path | None = None) -> None:
        self.vocab_path = Path(vocab_path)
        self.vocab: dict[str, int] = json.loads(self.vocab_path.read_text())
        self.id_to_gene = {idx: gene for gene, idx in self.vocab.items()}
        self.alias_to_gene: dict[str, str] = {}
        if alias_csv is not None:
            self._load_alias_csv(alias_csv)

    def _load_alias_csv(self, alias_csv: str | Path) -> None:
        frame = pd.read_csv(alias_csv, header=None)
        if frame.shape[1] < 2:
            raise ValueError(f"Alias CSV must have at least two columns: {alias_csv}")
        for ensembl, symbol in frame.iloc[:, :2].itertuples(index=False):
            ensembl = str(ensembl).strip()
            symbol = str(symbol).strip()
            if ensembl in self.vocab and symbol:
                self.alias_to_gene.setdefault(symbol, ensembl)

    def canonical_gene(self, name: str) -> str | None:
        name = str(name)
        if name in self.vocab:
            return name
        return self.alias_to_gene.get(name)

    def token_id(self, name: str) -> int | None:
        gene = self.canonical_gene(name)
        if gene is None:
            return None
        return self.vocab.get(gene)


@dataclass(slots=True)
class TokenizedCell:
    input_ids: np.ndarray
    expression_values: np.ndarray
    nonzero_mask: np.ndarray


class OmniCellH5ADTokenizer:
    """Convert H5AD expression rows into fixed-length OmniCell gene tokens."""

    def __init__(
        self,
        adata,
        gene_vocab: GeneVocab,
        token_per_cell: int,
        gene_strategy: str = "nonzero_hvg",
        selected_genes: str | Path | Sequence[str] | None = None,
        hvg_top_n: int | None = None,
        start_token: str = "<RNA_START>",
        end_token: str = "<RNA_END>",
        use_smooth_rank: bool = True,
        smooth_rank_range: tuple[float, float] = (0.0, 5.0),
        overflow_strategy: str = "top_expression",
    ) -> None:
        self.adata = adata
        self.X = adata.X
        self.gene_vocab = gene_vocab
        self.token_per_cell = int(token_per_cell)
        self.gene_strategy = gene_strategy
        self.use_smooth_rank = bool(use_smooth_rank)
        self.smooth_rank_range = tuple(smooth_rank_range)
        self.overflow_strategy = overflow_strategy
        self.start_token_id = self._require_token(start_token)
        self.end_token_id = self._require_token(end_token)

        self.var_names = np.asarray(adata.var_names).astype(str)
        self.var_token_ids = np.full(self.var_names.shape[0], -1, dtype=np.int64)
        for idx, name in enumerate(self.var_names):
            token_id = self.gene_vocab.token_id(name)
            if token_id is not None:
                self.var_token_ids[idx] = int(token_id)
        self.valid_var_indices = np.flatnonzero(self.var_token_ids >= 0)
        if self.valid_var_indices.size == 0:
            raise ValueError("No adata.var_names matched the OmniCell vocabulary.")

        self.hvg_var_indices = self._select_hvg_indices(hvg_top_n or self.token_per_cell)
        self.fixed_var_indices: np.ndarray | None = None
        if self.gene_strategy in {"selected", "hvg"}:
            if self.gene_strategy == "selected":
                genes = _read_gene_list(selected_genes)
                if not genes:
                    raise ValueError("gene_strategy='selected' requires selected_genes.")
                self.fixed_var_indices = self._resolve_selected_var_indices(genes)
            else:
                self.fixed_var_indices = self.hvg_var_indices[: self.token_per_cell]
            if self.fixed_var_indices.shape[0] < self.token_per_cell:
                self.fixed_var_indices = self._pad_var_indices_with_hvg(
                    self.fixed_var_indices,
                    forbidden=set(self.fixed_var_indices.tolist()),
                )
            self.fixed_var_indices = self.fixed_var_indices[: self.token_per_cell]

    def _require_token(self, token: str) -> int:
        token_id = self.gene_vocab.vocab.get(token)
        if token_id is None:
            raise ValueError(f"Special token {token!r} not found in vocabulary.")
        return int(token_id)

    def _as_csr_matrix(self):
        if sparse.issparse(self.X):
            return self.X.tocsr()
        arr = np.asarray(self.X)
        if arr.dtype == object:
            raise TypeError(
                "Backed sparse H5AD matrices cannot be converted through np.asarray; "
                "use chunked matrix access instead."
            )
        return sparse.csr_matrix(arr)

    def _slice_rows(self, start: int, end: int):
        block = self.X[start:end]
        if sparse.issparse(block):
            return block.tocsr()
        arr = np.asarray(block)
        if arr.dtype == object:
            raise TypeError(
                f"Unable to materialize expression rows {start}:{end}; got dtype object."
            )
        return arr

    def _select_hvg_indices(self, n_top: int) -> np.ndarray:
        n_obs = int(self.adata.n_obs)
        n_valid = int(self.valid_var_indices.shape[0])
        sums = np.zeros(n_valid, dtype=np.float64)
        sum_squares = np.zeros(n_valid, dtype=np.float64)
        chunk_size = 4096

        for start in range(0, n_obs, chunk_size):
            end = min(start + chunk_size, n_obs)
            block = self._slice_rows(start, end)
            if sparse.issparse(block):
                valid_block = block[:, self.valid_var_indices]
                sums += np.asarray(valid_block.sum(axis=0)).ravel()
                sum_squares += np.asarray(valid_block.power(2).sum(axis=0)).ravel()
            else:
                valid_block = np.asarray(block[:, self.valid_var_indices], dtype=np.float32)
                sums += valid_block.sum(axis=0, dtype=np.float64)
                sum_squares += np.square(valid_block, dtype=np.float64).sum(axis=0)

        mean = sums / max(n_obs, 1)
        mean_sq = sum_squares / max(n_obs, 1)
        variance = mean_sq - mean * mean
        order = np.argsort(-variance, kind="mergesort")
        selected = self.valid_var_indices[order[: max(n_top, self.token_per_cell)]]
        return selected.astype(np.int64, copy=False)

    def _resolve_selected_var_indices(self, genes: Sequence[str]) -> np.ndarray:
        canonical_to_var: dict[str, int] = {}
        for idx, name in enumerate(self.var_names):
            canonical = self.gene_vocab.canonical_gene(name)
            if canonical is not None and self.var_token_ids[idx] >= 0:
                canonical_to_var.setdefault(canonical, idx)

        selected: list[int] = []
        missing: list[str] = []
        for gene in genes:
            canonical = self.gene_vocab.canonical_gene(gene)
            if canonical is None or canonical not in canonical_to_var:
                missing.append(gene)
                continue
            selected.append(canonical_to_var[canonical])
        if not selected:
            raise ValueError("None of selected_genes were found in both H5AD and vocabulary.")
        if missing:
            print(f"[omnicell_hf] skipped {len(missing)} selected genes not present in data/vocab")
        return np.asarray(selected, dtype=np.int64)

    def _pad_var_indices_with_hvg(
        self,
        selected: np.ndarray,
        forbidden: set[int],
        row_nonzero: set[int] | None = None,
    ) -> np.ndarray:
        out = selected.astype(np.int64, copy=True).tolist()
        for var_idx in self.hvg_var_indices.tolist():
            if var_idx in forbidden:
                continue
            if row_nonzero is not None and var_idx in row_nonzero:
                continue
            out.append(int(var_idx))
            forbidden.add(int(var_idx))
            if len(out) >= self.token_per_cell:
                break
        if len(out) < self.token_per_cell:
            raise ValueError(
                f"Unable to build {self.token_per_cell} gene tokens. "
                f"Only found {len(out)} matched genes."
            )
        return np.asarray(out[: self.token_per_cell], dtype=np.int64)

    def _row_nonzero(self, cell_idx: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.X[cell_idx]
        if sparse.issparse(row):
            row = row.tocsr()
            indices = row.indices.astype(np.int64, copy=False)
            values = row.data.astype(np.float32, copy=False)
        else:
            arr = np.asarray(row).reshape(-1)
            indices = np.flatnonzero(arr).astype(np.int64, copy=False)
            values = arr[indices].astype(np.float32, copy=False)
        valid = self.var_token_ids[indices] >= 0
        return indices[valid], values[valid]

    def _row_values_for_indices(self, cell_idx: int, var_indices: np.ndarray) -> np.ndarray:
        row = self.X[cell_idx, var_indices]
        if sparse.issparse(row):
            return np.asarray(row.toarray()).reshape(-1).astype(np.float32, copy=False)
        return np.asarray(row).reshape(-1).astype(np.float32, copy=False)

    def _smooth_rank(self, values: np.ndarray) -> np.ndarray:
        unique_values = np.unique(values)
        if 0 not in unique_values:
            unique_values = np.append(unique_values, 0)
        unique_sorted = np.sort(unique_values)
        left, right = self.smooth_rank_range
        if unique_sorted.size == 1:
            ranks = np.full(unique_sorted.size, left, dtype=np.float32)
        else:
            ranks = left + (np.arange(unique_sorted.size) / (unique_sorted.size - 1)) * (
                right - left
            )
            ranks = ranks.astype(np.float32)
        return ranks[np.searchsorted(unique_sorted, values, side="left")]

    def _tokenize_fixed(self, cell_idx: int) -> TokenizedCell:
        assert self.fixed_var_indices is not None
        var_indices = self.fixed_var_indices
        gene_values = self._row_values_for_indices(cell_idx, var_indices)
        gene_token_ids = self.var_token_ids[var_indices]
        return self._wrap_cell(gene_token_ids, gene_values)

    def _tokenize_nonzero_hvg(self, cell_idx: int) -> TokenizedCell:
        row_indices, row_values = self._row_nonzero(cell_idx)
        if row_indices.size > 0:
            if row_indices.size > self.token_per_cell and self.overflow_strategy == "top_expression":
                order = np.argsort(-row_values, kind="mergesort")[: self.token_per_cell]
                row_indices = row_indices[order]
                row_values = row_values[order]
        selected = row_indices[: self.token_per_cell].astype(np.int64, copy=False)
        values = row_values[: self.token_per_cell].astype(np.float32, copy=False)

        if selected.shape[0] < self.token_per_cell:
            row_nonzero = set(row_indices.tolist())
            padded = self._pad_var_indices_with_hvg(
                selected,
                forbidden=set(selected.tolist()),
                row_nonzero=row_nonzero,
            )
            pad_count = padded.shape[0] - selected.shape[0]
            selected = padded
            values = np.concatenate([values, np.zeros(pad_count, dtype=np.float32)])

        gene_token_ids = self.var_token_ids[selected]
        return self._wrap_cell(gene_token_ids, values)

    def _wrap_cell(self, gene_token_ids: np.ndarray, gene_values: np.ndarray) -> TokenizedCell:
        if gene_token_ids.shape[0] != self.token_per_cell:
            raise ValueError(
                f"Expected {self.token_per_cell} gene tokens, got {gene_token_ids.shape[0]}."
            )
        values = gene_values.astype(np.float32, copy=True)
        if self.use_smooth_rank:
            values = self._smooth_rank(values)
        input_ids = np.concatenate(
            [
                np.asarray([self.start_token_id], dtype=np.int64),
                gene_token_ids.astype(np.int64, copy=False),
                np.asarray([self.end_token_id], dtype=np.int64),
            ]
        )
        expression_values = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float32),
                values.astype(np.float32, copy=False),
                np.asarray([0.0], dtype=np.float32),
            ]
        )
        nonzero_mask = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float32),
                (gene_values > 0).astype(np.float32, copy=False),
                np.asarray([0.0], dtype=np.float32),
            ]
        )
        return TokenizedCell(input_ids, expression_values, nonzero_mask)

    def tokenize_cell(self, cell_idx: int) -> TokenizedCell:
        if self.gene_strategy in {"selected", "hvg"}:
            return self._tokenize_fixed(cell_idx)
        if self.gene_strategy in {"nonzero_hvg", "all_nonzero_hvg"}:
            return self._tokenize_nonzero_hvg(cell_idx)
        raise ValueError(f"Unsupported gene_strategy: {self.gene_strategy}")


class OmniCellH5ADDataset(Dataset):
    """Trainer-ready H5AD dataset for single-cell, spatial, or region inputs."""

    def __init__(
        self,
        h5ad_path: str | Path,
        vocab_path: str | Path,
        alias_csv: str | Path | None = None,
        token_per_cell: int = 500,
        mode: str = "single_cell",
        n_cells_per_sample: int = 1,
        gene_strategy: str = "nonzero_hvg",
        selected_genes: str | Path | Sequence[str] | None = None,
        hvg_top_n: int | None = None,
        target_obs_key: str | None = None,
        backed: bool = False,
        use_smooth_rank: bool = True,
        region_manifest_path: str | Path | None = None,
        center_cells: str | Path | Sequence[Any] | None = None,
        neighbor_cells: str | Path | dict[Any, Sequence[Any]] | None = None,
        cell_id_obs_key: str | None = None,
        allow_short_groups: bool = False,
        x_key: str = "x",
        y_key: str = "y",
    ) -> None:
        self.h5ad_path = Path(h5ad_path)
        self.adata = _load_anndata(self.h5ad_path, backed=backed)
        self.mode = mode
        self.n_cells_per_sample = int(n_cells_per_sample)
        self.target_obs_key = target_obs_key
        self.cell_id_obs_key = cell_id_obs_key
        self.allow_short_groups = bool(allow_short_groups)
        self.gene_vocab = GeneVocab(vocab_path, alias_csv=alias_csv)
        self.tokenizer = OmniCellH5ADTokenizer(
            self.adata,
            self.gene_vocab,
            token_per_cell=token_per_cell,
            gene_strategy=gene_strategy,
            selected_genes=selected_genes,
            hvg_top_n=hvg_top_n,
            use_smooth_rank=use_smooth_rank,
        )
        self.coords = self._extract_coords(x_key=x_key, y_key=y_key)
        self._cell_lookup = self._build_cell_lookup()
        self.labels = self._build_labels()
        self.groups = self._build_groups(
            region_manifest_path=region_manifest_path,
            center_cells=center_cells,
            neighbor_cells=neighbor_cells,
        )

    def _extract_coords(self, x_key: str, y_key: str) -> np.ndarray:
        if "spatial" in self.adata.obsm:
            coords = np.asarray(self.adata.obsm["spatial"], dtype=np.float32)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                return coords[:, :2]
        if {x_key, y_key}.issubset(set(self.adata.obs.columns)):
            return self.adata.obs[[x_key, y_key]].to_numpy(dtype=np.float32, copy=True)
        return np.zeros((self.adata.n_obs, 2), dtype=np.float32)

    def _build_labels(self) -> np.ndarray | None:
        self.label_names: list[str] | None = None
        self.num_labels = 0
        self.problem_type: str | None = None
        if self.target_obs_key is None:
            return None
        if self.target_obs_key not in self.adata.obs.columns:
            raise ValueError(f"target_obs_key={self.target_obs_key!r} not in adata.obs.")

        series = self.adata.obs[self.target_obs_key]
        if pd.api.types.is_numeric_dtype(series):
            self.num_labels = 1
            self.problem_type = "regression"
            return series.to_numpy(dtype=np.float32, copy=True)

        categorical = pd.Categorical(series.astype(str))
        self.label_names = [str(item) for item in categorical.categories.tolist()]
        self.num_labels = len(self.label_names)
        self.problem_type = "single_label_classification"
        return categorical.codes.astype(np.int64, copy=False)

    def _build_cell_lookup(self) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for idx, obs_name in enumerate(self.adata.obs_names.astype(str)):
            lookup.setdefault(str(obs_name), idx)
        if self.cell_id_obs_key is not None:
            if self.cell_id_obs_key not in self.adata.obs.columns:
                raise ValueError(f"cell_id_obs_key={self.cell_id_obs_key!r} not in adata.obs.")
            for idx, value in enumerate(self.adata.obs[self.cell_id_obs_key].astype(str)):
                lookup.setdefault(str(value), idx)
        return lookup

    def _resolve_cell_ref(self, ref: Any) -> int:
        if isinstance(ref, str):
            key = ref.strip()
            if key in self._cell_lookup:
                return self._cell_lookup[key]
            try:
                idx = int(key)
            except ValueError as exc:
                raise KeyError(f"Cell reference {ref!r} not found.") from exc
            if 0 <= idx < self.adata.n_obs:
                return idx
            raise KeyError(f"Cell row index out of range: {idx}")

        key = str(ref)
        if self.cell_id_obs_key is not None and key in self._cell_lookup:
            return self._cell_lookup[key]
        try:
            idx = int(ref)
        except Exception as exc:
            raise KeyError(f"Cell reference {ref!r} not found.") from exc
        if 0 <= idx < self.adata.n_obs:
            return idx
        if key in self._cell_lookup:
            return self._cell_lookup[key]
        raise KeyError(f"Cell row index out of range or unknown cell id: {ref!r}")

    def _resolve_cell_refs(self, refs: Iterable[Any]) -> np.ndarray:
        return np.asarray([self._resolve_cell_ref(ref) for ref in refs], dtype=np.int64)

    def _load_center_cells(self, center_cells: str | Path | Sequence[Any] | None) -> np.ndarray | None:
        if center_cells is None:
            return None
        if isinstance(center_cells, (str, Path)):
            path = Path(center_cells)
            if path.exists():
                suffix = path.suffix.lower()
                if suffix == ".json":
                    data = json.loads(path.read_text())
                    if isinstance(data, dict):
                        for key in ("center_cells", "centers", "cells"):
                            if key in data:
                                data = data[key]
                                break
                    refs = _coerce_ref_list(data)
                elif suffix == ".parquet":
                    frame = pd.read_parquet(path)
                    column = next(
                        (c for c in ("center_cell", "center", "cell", "obs_name") if c in frame.columns),
                        frame.columns[0],
                    )
                    refs = frame[column].tolist()
                elif suffix in {".csv", ".tsv"}:
                    frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
                    column = next(
                        (c for c in ("center_cell", "center", "cell", "obs_name") if c in frame.columns),
                        frame.columns[0],
                    )
                    refs = frame[column].tolist()
                else:
                    refs = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            else:
                refs = _coerce_ref_list(center_cells)
        else:
            refs = list(center_cells)
        return self._resolve_cell_refs(refs)

    def _normalize_group(self, center: int, neighbors: Sequence[int] | np.ndarray | None = None) -> np.ndarray:
        out: list[int] = [int(center)]
        if neighbors is not None:
            for item in neighbors:
                idx = int(item)
                if idx not in out:
                    out.append(idx)
                if len(out) >= self.n_cells_per_sample:
                    break
        if len(out) < self.n_cells_per_sample:
            if self.allow_short_groups:
                out.extend([int(center)] * (self.n_cells_per_sample - len(out)))
                return np.asarray(out, dtype=np.int64)
            raise ValueError(
                f"Explicit group for center={center} has {len(out)} cells, "
                f"but n_cells_per_sample={self.n_cells_per_sample}."
            )
        return np.asarray(out[: self.n_cells_per_sample], dtype=np.int64)

    def _read_neighbor_rows(self, neighbor_cells: str | Path | dict[Any, Sequence[Any]]) -> Any:
        if isinstance(neighbor_cells, dict):
            return neighbor_cells
        path = Path(neighbor_cells)
        if not path.exists():
            raise FileNotFoundError(f"neighbor_cells path not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(path.read_text())
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".tsv"}:
            return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            rows.append({"center_cell": parts[0], "neighbor_cells": parts[1:]})
        return rows

    def _build_neighbor_groups(
        self,
        neighbor_cells: str | Path | dict[Any, Sequence[Any]],
        center_cells: str | Path | Sequence[Any] | None,
    ) -> list[np.ndarray]:
        raw = self._read_neighbor_rows(neighbor_cells)
        explicit_groups: list[np.ndarray] = []
        mapping: dict[int, list[int]] = {}

        def add_mapping(center_ref: Any, neighbor_refs: Any) -> None:
            center = self._resolve_cell_ref(center_ref)
            neighbors = self._resolve_cell_refs(_coerce_ref_list(neighbor_refs)).tolist()
            mapping[center] = neighbors

        if isinstance(raw, dict):
            for center_ref, neighbor_refs in raw.items():
                add_mapping(center_ref, neighbor_refs)
        elif isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    refs = _coerce_ref_list(row)
                    if not refs:
                        continue
                    center = self._resolve_cell_ref(refs[0])
                    neighbors = self._resolve_cell_refs(refs[1:]).tolist()
                    explicit_groups.append(self._normalize_group(center, neighbors))
                    continue
                if "source_cell_indices" in row and not any(
                    key in row for key in ("center_cell", "center", "cell")
                ):
                    refs = _coerce_ref_list(row["source_cell_indices"])
                    center = self._resolve_cell_ref(refs[0])
                    neighbors = self._resolve_cell_refs(refs[1:]).tolist()
                    explicit_groups.append(self._normalize_group(center, neighbors))
                else:
                    center_key = next(
                        key for key in ("center_cell", "center", "cell", "obs_name") if key in row
                    )
                    neighbor_key = next(
                        key
                        for key in ("neighbor_cells", "neighbors", "source_cell_indices")
                        if key in row
                    )
                    add_mapping(row[center_key], row[neighbor_key])
        elif isinstance(raw, pd.DataFrame):
            frame = raw
            if "source_cell_indices" in frame.columns and not any(
                c in frame.columns for c in ("center_cell", "center", "cell")
            ):
                for refs_value in frame["source_cell_indices"]:
                    refs = _coerce_ref_list(refs_value)
                    if not refs:
                        continue
                    center = self._resolve_cell_ref(refs[0])
                    neighbors = self._resolve_cell_refs(refs[1:]).tolist()
                    explicit_groups.append(self._normalize_group(center, neighbors))
            elif {"center_cell", "neighbor_cell"}.issubset(frame.columns):
                for center_ref, group in frame.groupby("center_cell"):
                    add_mapping(center_ref, group["neighbor_cell"].tolist())
            else:
                center_col = next(
                    (c for c in ("center_cell", "center", "cell", "obs_name") if c in frame.columns),
                    frame.columns[0],
                )
                neighbor_col = next(
                    (c for c in ("neighbor_cells", "neighbors", "source_cell_indices") if c in frame.columns),
                    frame.columns[1] if len(frame.columns) > 1 else None,
                )
                if neighbor_col is None:
                    raise ValueError("neighbor_cells table needs a neighbor list column.")
                for _, row in frame.iterrows():
                    add_mapping(row[center_col], row[neighbor_col])
        else:
            raise TypeError(f"Unsupported neighbor_cells type: {type(raw)!r}")

        if explicit_groups:
            return explicit_groups

        centers = self._load_center_cells(center_cells)
        if centers is None:
            centers = np.asarray(list(mapping.keys()), dtype=np.int64)
        groups = []
        for center in centers.tolist():
            if int(center) not in mapping:
                raise KeyError(f"No explicit neighbor list found for center cell {center}.")
            groups.append(self._normalize_group(int(center), mapping[int(center)]))
        return groups

    def _build_spatial_groups(self, centers: np.ndarray | None = None) -> list[np.ndarray]:
        if centers is None:
            centers = np.arange(self.adata.n_obs, dtype=np.int64)
        if self.n_cells_per_sample == 1:
            return [np.asarray([idx], dtype=np.int64) for idx in centers.tolist()]
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.n_cells_per_sample, metric="euclidean")
        nn.fit(self.coords)
        _, indices = nn.kneighbors(self.coords[centers], return_distance=True)
        return [row.astype(np.int64, copy=False) for row in indices]

    def _build_region_groups(self, region_manifest_path: str | Path) -> list[np.ndarray]:
        frame = pd.read_parquet(region_manifest_path)
        if "source_cell_indices" not in frame.columns:
            raise ValueError("region_manifest_path must contain source_cell_indices.")
        groups = []
        for _, row in frame.iterrows():
            value = row["source_cell_indices"]
            arr = _coerce_int_array(value)[: self.n_cells_per_sample]
            if "anchor_source_cell_index" in row.index and len(arr) > 0:
                center = int(row["anchor_source_cell_index"])
                arr_list = [center] + [int(idx) for idx in arr.tolist() if int(idx) != center]
                arr = np.asarray(arr_list[: self.n_cells_per_sample], dtype=np.int64)
            if arr.shape[0] < self.n_cells_per_sample:
                if self.allow_short_groups:
                    groups.append(arr.astype(np.int64, copy=False))
                    continue
                raise ValueError(
                    f"Region manifest row has {arr.shape[0]} cells, "
                    f"but n_cells_per_sample={self.n_cells_per_sample}."
                )
            groups.append(arr.astype(np.int64, copy=False))
        if not groups:
            raise ValueError("No usable groups found in region manifest.")
        return groups

    def _build_groups(
        self,
        region_manifest_path: str | Path | None,
        center_cells: str | Path | Sequence[Any] | None,
        neighbor_cells: str | Path | dict[Any, Sequence[Any]] | None,
    ) -> list[np.ndarray]:
        if region_manifest_path is not None:
            return self._build_region_groups(region_manifest_path)
        if neighbor_cells is not None:
            return self._build_neighbor_groups(neighbor_cells, center_cells)
        centers = self._load_center_cells(center_cells)
        if self.mode in {"spatial", "region"}:
            return self._build_spatial_groups(centers)
        if centers is not None:
            if self.n_cells_per_sample == 1:
                return [np.asarray([idx], dtype=np.int64) for idx in centers.tolist()]
            return [self._normalize_group(int(center), []) for center in centers.tolist()]
        if self.n_cells_per_sample == 1:
            return [np.asarray([idx], dtype=np.int64) for idx in range(self.adata.n_obs)]
        groups = []
        for start in range(0, self.adata.n_obs - self.n_cells_per_sample + 1, self.n_cells_per_sample):
            groups.append(np.arange(start, start + self.n_cells_per_sample, dtype=np.int64))
        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def _label_for_group(self, group: np.ndarray) -> np.ndarray | None:
        if self.labels is None:
            return None
        return np.asarray(self.labels[int(group[0])])

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        tokenized = [self.tokenizer.tokenize_cell(int(cell_idx)) for cell_idx in group]
        input_ids = np.concatenate([item.input_ids for item in tokenized])
        expression_values = np.concatenate([item.expression_values for item in tokenized])
        nonzero_mask = np.concatenate([item.nonzero_mask for item in tokenized])
        positions = np.concatenate(
            [
                np.repeat(
                    self.coords[int(cell_idx)][None, :],
                    self.tokenizer.token_per_cell + 2,
                    axis=0,
                )
                for cell_idx in group
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        item: dict[str, Any] = {
            "input_ids": torch.from_numpy(input_ids.astype(np.int64, copy=False)),
            "expression_values": torch.from_numpy(expression_values.astype(np.float32, copy=False)),
            "positions": torch.from_numpy(positions),
            "nonzero_mask": torch.from_numpy(nonzero_mask.astype(np.float32, copy=False)),
            "cell_indices": torch.from_numpy(group.astype(np.int64, copy=False)),
        }
        label = self._label_for_group(group)
        if label is not None:
            item["labels"] = torch.as_tensor(label)
        return item


class OmniCellDataCollator:
    """Stack fixed-length OmniCell H5AD samples for HF Trainer."""

    def __init__(self, keep_cell_indices: bool = False) -> None:
        self.keep_cell_indices = keep_cell_indices

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        keys = ["input_ids", "expression_values", "positions", "nonzero_mask"]
        batch = {key: torch.stack([feature[key] for feature in features], dim=0) for key in keys}
        if "labels" in features[0]:
            labels = [feature["labels"] for feature in features]
            batch["labels"] = torch.stack(labels, dim=0)
        if self.keep_cell_indices and "cell_indices" in features[0]:
            batch["cell_indices"] = torch.stack(
                [feature["cell_indices"] for feature in features], dim=0
            )
        return batch
