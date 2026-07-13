"""Hugging Face configuration for OmniCell."""

from __future__ import annotations

from transformers import PretrainedConfig


class OmniCellConfig(PretrainedConfig):
    """Configuration for OmniCell Transformer models.

    The defaults mirror the public OmniCell checkpoint layout, while
    ``token_per_cell`` can be changed for downstream runs that use shorter
    per-cell gene sequences.
    """

    model_type = "OmniCell"

    def __init__(
        self,
        vocab_size: int = 60607,
        num_gene: int | None = None,
        d_model: int = 512,
        token_per_cell: int = 500,
        n_cells_per_sample: int = 1,
        num_shared: int = 5,
        num_routing: int = 10,
        topk: int = 5,
        heads: int = 8,
        dropout: float = 0.1,
        rotary_dim: int = 64,
        dim: int | None = None,
        rope_base: float = 10000.0,
        base: float | None = None,
        eps: float = 1e-6,
        multiple_of: int = 512,
        num_layers: int = 10,
        rotary: bool = True,
        hidden_dim: int | None = 2048,
        use_flash: bool = True,
        start_token_id: int = 60603,
        end_token_id: int = 60604,
        pad_token_id: int | None = None,
        normalize_cell_embeddings: bool = True,
        pooling: str = "nonzero_mean",
        center_cell_index: int = 0,
        embedding_pooling: str = "center",
        num_labels: int = 1,
        problem_type: str | None = None,
        supervised_loss: str = "mse",
        unsupervised_loss: str = "mse",
        unsupervised_loss_on: str = "all_gene_tokens",
        load_balance_loss_weight: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        if num_gene is not None:
            vocab_size = int(num_gene)
        if dim is not None:
            rotary_dim = int(dim)
        if base is not None:
            rope_base = float(base)
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=start_token_id,
            eos_token_id=end_token_id,
            **kwargs,
        )

        self.vocab_size = int(vocab_size)
        self.num_gene = int(vocab_size)
        self.d_model = int(d_model)
        self.token_per_cell = int(token_per_cell)
        self.n_cells_per_sample = int(n_cells_per_sample)
        self.num_shared = int(num_shared)
        self.num_routing = int(num_routing)
        self.topk = int(topk)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.rotary_dim = int(rotary_dim)
        self.dim = int(rotary_dim)
        self.rope_base = float(rope_base)
        self.base = float(rope_base)
        self.eps = float(eps)
        self.multiple_of = int(multiple_of)
        self.num_layers = int(num_layers)
        self.rotary = bool(rotary)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.use_flash = bool(use_flash)
        self.start_token_id = int(start_token_id)
        self.end_token_id = int(end_token_id)
        self.normalize_cell_embeddings = bool(normalize_cell_embeddings)
        self.pooling = str(pooling)
        self.center_cell_index = int(center_cell_index)
        self.embedding_pooling = str(embedding_pooling)
        self.num_labels = int(num_labels)
        self.problem_type = problem_type
        self.supervised_loss = str(supervised_loss)
        self.unsupervised_loss = str(unsupervised_loss)
        self.unsupervised_loss_on = str(unsupervised_loss_on)
        self.load_balance_loss_weight = float(load_balance_loss_weight)
        self.initializer_range = float(initializer_range)

        if self.d_model % self.heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by heads={self.heads}."
            )
        if self.rotary_dim > self.head_dim:
            raise ValueError(
                f"rotary_dim={self.rotary_dim} cannot exceed head_dim={self.head_dim}."
            )
        if self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}.")
        if self.token_per_cell <= 0:
            raise ValueError("token_per_cell must be positive.")
        if self.n_cells_per_sample <= 0:
            raise ValueError("n_cells_per_sample must be positive.")
        if self.embedding_pooling not in {"center", "mean", "flatten"}:
            raise ValueError(
                "embedding_pooling must be one of: center, mean, flatten."
            )
        if self.unsupervised_loss_on not in {"all_gene_tokens", "nonzero"}:
            raise ValueError(
                "unsupervised_loss_on must be 'all_gene_tokens' or 'nonzero'."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    @property
    def cell_token_len(self) -> int:
        return self.token_per_cell + 2

    @property
    def sequence_length(self) -> int:
        return self.n_cells_per_sample * self.cell_token_len

    @classmethod
    def from_legacy_dict(cls, data: dict) -> "OmniCellConfig":
        """Build a config from the original ``LMConfig.json`` dictionary."""

        return cls(
            vocab_size=data.get("num_gene", data.get("vocab_size", 60607)),
            d_model=data.get("d_model", 512),
            token_per_cell=data.get("token_per_cell", 500),
            topk=data.get("topk", 5),
            num_shared=data.get("num_shared", 5),
            num_routing=data.get("num_routing", 10),
            heads=data.get("heads", 8),
            dropout=data.get("dropout", 0.1),
            rotary_dim=data.get("dim", data.get("rotary_dim", 64)),
            rope_base=data.get("base", data.get("rope_base", 10000.0)),
            eps=data.get("eps", 1e-6),
            multiple_of=data.get("multiple_of", 512),
            num_layers=data.get("num_layers", 10),
            rotary=bool(data.get("rotary", True)),
            hidden_dim=data.get("hidden_dim", 2048),
            use_flash=bool(data.get("use_flash", True)),
        )
