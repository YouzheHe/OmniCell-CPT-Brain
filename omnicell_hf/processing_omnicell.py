"""Processor for OmniCell H5AD inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .data import OmniCellDataCollator, OmniCellH5ADDataset


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_OMNICELL_ROOT = PACKAGE_ROOT.parent / "OmniCell"
DEFAULT_VOCAB_PATH = WORKSPACE_OMNICELL_ROOT / "vocab" / "Vocabulary.json"
DEFAULT_ALIAS_CSV = WORKSPACE_OMNICELL_ROOT / "vocab" / "new_genes_homo_sapiens.csv"


class OmniCellProcessor:
    """Hugging Face-style processor for H5AD to OmniCell tensors.

    The processor stores data-tokenization defaults and can be saved next to
    a model with ``save_pretrained``.
    """

    config_name = "preprocessor_config.json"
    model_input_names = ["input_ids", "expression_values", "positions", "nonzero_mask"]

    def __init__(
        self,
        *,
        vocab_path: str | Path = DEFAULT_VOCAB_PATH,
        alias_csv: str | Path | None = DEFAULT_ALIAS_CSV,
        token_per_cell: int = 500,
        n_cells_per_sample: int = 1,
        mode: str = "single_cell",
        gene_strategy: str = "nonzero_hvg",
        selected_genes: str | Path | Sequence[str] | None = None,
        use_smooth_rank: bool = True,
        backed: bool = True,
        cell_id_obs_key: str | None = None,
        obsm_key: str = "OmniCell_embedding",
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.alias_csv = None if alias_csv is None else Path(alias_csv)
        self.token_per_cell = int(token_per_cell)
        self.n_cells_per_sample = int(n_cells_per_sample)
        self.mode = mode
        self.gene_strategy = gene_strategy
        self.selected_genes = selected_genes
        self.use_smooth_rank = bool(use_smooth_rank)
        self.backed = bool(backed)
        self.cell_id_obs_key = cell_id_obs_key
        self.obsm_key = obsm_key

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path | None = None,
        **kwargs,
    ) -> "OmniCellProcessor":
        """Load processor defaults from a HF directory, or use bundled defaults."""

        data: dict[str, Any] = {}
        if pretrained_model_name_or_path is not None:
            config_path = Path(pretrained_model_name_or_path) / cls.config_name
            if config_path.exists():
                data = json.loads(config_path.read_text())
        data.update(kwargs)
        return cls(**data)

    def save_pretrained(self, save_directory: str | Path) -> tuple[str]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        config_path = save_directory / self.config_name
        config_path.write_text(json.dumps(self.to_dict(), indent=2))
        return (str(config_path),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_path": str(self.vocab_path),
            "alias_csv": None if self.alias_csv is None else str(self.alias_csv),
            "token_per_cell": self.token_per_cell,
            "n_cells_per_sample": self.n_cells_per_sample,
            "mode": self.mode,
            "gene_strategy": self.gene_strategy,
            "selected_genes": None if self.selected_genes is None else str(self.selected_genes),
            "use_smooth_rank": self.use_smooth_rank,
            "backed": self.backed,
            "cell_id_obs_key": self.cell_id_obs_key,
            "obsm_key": self.obsm_key,
        }

    def to_dataset(
        self,
        h5ad: str | Path,
        *,
        token_per_cell: int | None = None,
        n_cells_per_sample: int | None = None,
        mode: str | None = None,
        gene_strategy: str | None = None,
        selected_genes: str | Path | Sequence[str] | None = None,
        hvg_top_n: int | None = None,
        target_obs_key: str | None = None,
        region_manifest_path: str | Path | None = None,
        center_cells: str | Path | Sequence[Any] | None = None,
        neighbor_cells: str | Path | dict[Any, Sequence[Any]] | None = None,
        cell_id_obs_key: str | None = None,
        allow_short_groups: bool = False,
        use_smooth_rank: bool | None = None,
        backed: bool | None = None,
    ) -> OmniCellH5ADDataset:
        return OmniCellH5ADDataset(
            h5ad_path=h5ad,
            vocab_path=self.vocab_path,
            alias_csv=self.alias_csv,
            token_per_cell=token_per_cell or self.token_per_cell,
            mode=mode or self.mode,
            n_cells_per_sample=n_cells_per_sample or self.n_cells_per_sample,
            gene_strategy=gene_strategy or self.gene_strategy,
            selected_genes=selected_genes or self.selected_genes,
            hvg_top_n=hvg_top_n,
            target_obs_key=target_obs_key,
            backed=self.backed if backed is None else backed,
            use_smooth_rank=self.use_smooth_rank if use_smooth_rank is None else use_smooth_rank,
            region_manifest_path=region_manifest_path,
            center_cells=center_cells,
            neighbor_cells=neighbor_cells,
            cell_id_obs_key=cell_id_obs_key or self.cell_id_obs_key,
            allow_short_groups=allow_short_groups,
        )

    def __call__(
        self,
        h5ad: str | Path,
        *,
        indices: int | Sequence[int] | None = 0,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """Tokenize H5AD rows/groups into a model batch.

        By default this returns the first sample, matching normal processor
        behavior for quick model calls. Use ``to_dataset`` for full-file
        streaming.
        """

        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is currently supported.")
        dataset = self.to_dataset(h5ad, **kwargs)
        if indices is None:
            indices = list(range(len(dataset)))
        elif isinstance(indices, int):
            indices = [indices]
        features = [dataset[int(index)] for index in indices]
        return OmniCellDataCollator()(features)
