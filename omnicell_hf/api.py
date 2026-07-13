"""High-level OmniCell convenience API."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import OmniCellDataCollator, OmniCellH5ADDataset
from .legacy import build_model_from_legacy
from .modeling_omnicell import (
    OmniCellForSupervisedFineTuning,
    OmniCellForUnsupervisedFineTuning,
    OmniCellModel,
)
from .processing_omnicell import OmniCellProcessor


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_OMNICELL_ROOT = PACKAGE_ROOT.parent / "OmniCell"
DEFAULT_CHECKPOINT_DIR = WORKSPACE_OMNICELL_ROOT / "checkpoint"
DEFAULT_VOCAB_PATH = WORKSPACE_OMNICELL_ROOT / "vocab" / "Vocabulary.json"
DEFAULT_ALIAS_CSV = WORKSPACE_OMNICELL_ROOT / "vocab" / "new_genes_homo_sapiens.csv"


def default_output_h5ad_path(h5ad_path: str | Path) -> Path:
    path = Path(h5ad_path)
    return path.with_name(f"{path.stem}_OmniCell.h5ad")


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_hf_model(model_name_or_path: str | Path, model_type: str):
    if model_type == "backbone":
        return OmniCellModel.from_pretrained(model_name_or_path)
    if model_type == "unsupervised":
        return OmniCellForUnsupervisedFineTuning.from_pretrained(model_name_or_path)
    if model_type == "supervised":
        return OmniCellForSupervisedFineTuning.from_pretrained(model_name_or_path)
    raise ValueError("model_type must be one of: backbone, unsupervised, supervised.")


def _backbone(model):
    return getattr(model, "omnicell", model)


def _projection_basis(model, threshold: float) -> torch.Tensor:
    backbone = _backbone(model)
    if not hasattr(backbone, "output") or not hasattr(backbone.output, "W"):
        raise ValueError("Projection requires a model with output.W.")
    w = backbone.output.W.detach().float()
    sym_w = (w + w.t()) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(sym_w)
    order = torch.argsort(torch.abs(eigenvalues), descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    variances = torch.abs(eigenvalues)
    ratios = variances / torch.sum(variances)
    cumulative = torch.cumsum(ratios, dim=0)
    n_components = max(1, int(torch.sum(cumulative < threshold).item()) + 1)
    basis = eigenvectors[:, :n_components]
    return basis / torch.norm(basis, dim=0, keepdim=True).clamp(min=1e-12)


class OmniCellPipeline:
    """Pipeline-style OmniCell H5AD embedder.

    Examples
    --------
    One-line default embedding:

    ``OmniCellPipeline.from_pretrained()("sample.h5ad")``
    """

    def __init__(
        self,
        model,
        *,
        vocab_path: str | Path = DEFAULT_VOCAB_PATH,
        alias_csv: str | Path | None = DEFAULT_ALIAS_CSV,
        device: str | torch.device | None = "auto",
        fp16: bool = False,
    ) -> None:
        self.model = model
        self.vocab_path = Path(vocab_path)
        self.alias_csv = None if alias_csv is None else Path(alias_csv)
        self.device = _resolve_device(device)
        self.fp16 = bool(fp16)

        self.backbone = _backbone(self.model).to(self.device)
        if self.fp16:
            self.backbone = self.backbone.half()
        self.backbone.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path | None = None,
        *,
        legacy_checkpoint_dir: str | Path | None = None,
        model_type: str = "backbone",
        vocab_path: str | Path = DEFAULT_VOCAB_PATH,
        alias_csv: str | Path | None = DEFAULT_ALIAS_CSV,
        token_per_cell: int = 500,
        n_cells_per_sample: int = 1,
        num_labels: int | None = None,
        problem_type: str | None = None,
        device: str | torch.device | None = "auto",
        fp16: bool = False,
    ) -> "OmniCellPipeline":
        """Load a Hugging Face OmniCell model or the bundled legacy checkpoint.

        If no path is provided, this method uses
        ``../OmniCell/checkpoint`` relative to this package.
        """

        if model_name_or_path is None and legacy_checkpoint_dir is None:
            if not DEFAULT_CHECKPOINT_DIR.exists():
                raise FileNotFoundError(
                    "No model path was provided and the default legacy checkpoint "
                    f"does not exist: {DEFAULT_CHECKPOINT_DIR}"
                )
            legacy_checkpoint_dir = DEFAULT_CHECKPOINT_DIR

        if legacy_checkpoint_dir is not None:
            model, _ = build_model_from_legacy(
                legacy_checkpoint_dir,
                model_type=model_type,
                token_per_cell=token_per_cell,
                n_cells_per_sample=n_cells_per_sample,
                num_labels=num_labels,
                problem_type=problem_type,
            )
        else:
            model = _load_hf_model(model_name_or_path, model_type=model_type)

        return cls(
            model,
            vocab_path=vocab_path,
            alias_csv=alias_csv,
            device=device,
            fp16=fp16,
        )

    def __call__(self, h5ad: str | Path, **kwargs) -> Path:
        return self.embed_h5ad(h5ad, **kwargs)

    def embed_h5ad(
        self,
        h5ad: str | Path,
        output_h5ad: str | Path | None = None,
        *,
        inplace: bool = False,
        obsm_key: str = "OmniCell_embedding",
        token_per_cell: int | None = None,
        n_cells_per_sample: int | None = None,
        mode: str = "single_cell",
        gene_strategy: str = "nonzero_hvg",
        selected_genes: str | Path | list[str] | None = None,
        hvg_top_n: int | None = None,
        region_manifest_path: str | Path | None = None,
        center_cells: str | Path | list[Any] | None = None,
        neighbor_cells: str | Path | dict[Any, list[Any]] | None = None,
        cell_id_obs_key: str | None = None,
        allow_short_groups: bool = False,
        use_smooth_rank: bool = True,
        backed: bool = True,
        batch_size: int = 1,
        show_progress: bool = True,
        project: bool = True,
        projection_threshold: float = 0.9,
    ) -> Path:
        """Embed an H5AD file and write ``obsm[obsm_key]`` into an H5AD.

        Defaults are intentionally runnable:

        ``OmniCell.from_pretrained().embed_h5ad("sample.h5ad")``

        writes ``sample_OmniCell.h5ad`` next to the input.
        """

        h5ad = Path(h5ad)
        if output_h5ad is None:
            output_h5ad = h5ad if inplace else default_output_h5ad_path(h5ad)
        output_h5ad = Path(output_h5ad)
        if inplace and output_h5ad.resolve() != h5ad.resolve():
            raise ValueError("inplace=True cannot be combined with a different output_h5ad.")

        config = self.backbone.config
        token_per_cell = int(token_per_cell or config.token_per_cell)
        n_cells_per_sample = int(n_cells_per_sample or config.n_cells_per_sample)

        dataset = OmniCellH5ADDataset(
            h5ad_path=h5ad,
            vocab_path=self.vocab_path,
            alias_csv=self.alias_csv,
            token_per_cell=token_per_cell,
            mode=mode,
            n_cells_per_sample=n_cells_per_sample,
            gene_strategy=gene_strategy,
            selected_genes=selected_genes,
            hvg_top_n=hvg_top_n,
            backed=backed,
            use_smooth_rank=use_smooth_rank,
            region_manifest_path=region_manifest_path,
            center_cells=center_cells,
            neighbor_cells=neighbor_cells,
            cell_id_obs_key=cell_id_obs_key,
            allow_short_groups=allow_short_groups,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=OmniCellDataCollator(keep_cell_indices=True),
        )

        embedding_sum: np.ndarray | None = None
        embedding_count = np.zeros(dataset.adata.n_obs, dtype=np.int32)
        iterator = tqdm(loader, desc="OmniCell", unit="batch") if show_progress else loader

        with torch.no_grad():
            for batch in iterator:
                cell_indices = batch.pop("cell_indices")
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                if self.fp16:
                    inputs["expression_values"] = inputs["expression_values"].half()
                    inputs["positions"] = inputs["positions"].half()
                outputs = self.backbone(**inputs, return_dict=True)
                emb = outputs.cell_embeddings.float().cpu().numpy()
                idx = cell_indices.numpy()
                if embedding_sum is None:
                    embedding_sum = np.zeros(
                        (dataset.adata.n_obs, emb.shape[-1]),
                        dtype=np.float32,
                    )
                flat_idx = idx.reshape(-1)
                flat_emb = emb.reshape(-1, emb.shape[-1])
                np.add.at(embedding_sum, flat_idx, flat_emb)
                np.add.at(embedding_count, flat_idx, 1)

        if embedding_sum is None:
            raise RuntimeError("No embeddings were produced.")

        embedding = np.zeros_like(embedding_sum)
        covered = embedding_count > 0
        embedding[covered] = embedding_sum[covered] / embedding_count[covered, None]
        if project:
            basis = _projection_basis(self.backbone, projection_threshold).cpu().numpy()
            embedding = embedding @ basis
        if getattr(dataset.adata, "isbacked", False):
            dataset.adata.file.close()

        if output_h5ad.resolve() != h5ad.resolve():
            output_h5ad.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(h5ad, output_h5ad)

        adata = ad.read_h5ad(output_h5ad)
        adata.obsm[obsm_key] = embedding
        adata.obs[f"{obsm_key}_count"] = embedding_count
        adata.uns[f"{obsm_key}_params"] = {
            "token_per_cell": token_per_cell,
            "n_cells_per_sample": n_cells_per_sample,
            "mode": mode,
            "gene_strategy": gene_strategy,
            "region_manifest_path": None
            if region_manifest_path is None
            else str(region_manifest_path),
            "center_cells": None if center_cells is None else str(center_cells),
            "neighbor_cells": None if neighbor_cells is None else str(neighbor_cells),
            "cell_id_obs_key": cell_id_obs_key,
            "covered_cells": int(covered.sum()),
            "project": bool(project),
            "projection_threshold": float(projection_threshold),
        }
        adata.write_h5ad(output_h5ad)
        return output_h5ad


def embed_h5ad(h5ad: str | Path, **kwargs) -> Path:
    """One-call default OmniCell embedding helper.

    Example
    -------
    ``embed_h5ad("sample.h5ad")``
    """

    load_keys = {
        "model_name_or_path",
        "legacy_checkpoint_dir",
        "model_type",
        "vocab_path",
        "alias_csv",
        "token_per_cell",
        "n_cells_per_sample",
        "num_labels",
        "problem_type",
        "device",
        "fp16",
    }
    load_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in load_keys}
    model_name_or_path = load_kwargs.pop("model_name_or_path", None)
    runner = OmniCell.from_pretrained(model_name_or_path, **load_kwargs)
    return runner.embed_h5ad(h5ad, **kwargs)


# Backward-friendly short alias. The HF-style model class remains OmniCellModel.
OmniCell = OmniCellPipeline
