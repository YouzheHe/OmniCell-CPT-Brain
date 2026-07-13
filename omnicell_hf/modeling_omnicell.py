"""Hugging Face model classes for OmniCell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_omnicell import OmniCellConfig

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - optional acceleration path
    flash_attn_func = None


@dataclass
class OmniCellModelOutput(ModelOutput):
    last_hidden_state: Tensor | None = None
    cell_embeddings: Tensor | None = None
    load_balance_loss: Tensor | None = None


@dataclass
class OmniCellUnsupervisedOutput(ModelOutput):
    loss: Tensor | None = None
    logits: Tensor | None = None
    last_hidden_state: Tensor | None = None
    cell_embeddings: Tensor | None = None
    load_balance_loss: Tensor | None = None


@dataclass
class OmniCellSupervisedOutput(ModelOutput):
    loss: Tensor | None = None
    logits: Tensor | None = None
    last_hidden_state: Tensor | None = None
    cell_embeddings: Tensor | None = None
    load_balance_loss: Tensor | None = None


class MoE4Embedder(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_shared = config.num_shared
        self.num_routing = config.num_routing
        self.topk = config.topk
        self.load_balance_loss: Tensor | None = None

        self.shared_experts = nn.ModuleList(
            [nn.Linear(1, self.d_model, bias=False) for _ in range(self.num_shared)]
        )
        self.routing_experts = nn.ModuleList(
            [nn.Linear(1, self.d_model, bias=False) for _ in range(self.num_routing)]
        )
        self.router = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_model, self.num_routing, bias=False),
        )

    def forward(self, gene_embedded: Tensor, value: Tensor) -> Tensor:
        value = value.to(dtype=gene_embedded.dtype)
        shared_input = value.unsqueeze(-1)
        shared_output = sum(expert(shared_input) for expert in self.shared_experts)

        routing_logits = self.router(gene_embedded)
        routing_weights = F.softmax(routing_logits, dim=-1)
        topk_weights, topk_idx = torch.topk(routing_weights, self.topk, dim=-1)
        sparse_weights = torch.zeros_like(routing_weights).scatter(
            -1, topk_idx, topk_weights
        )
        expert_outputs = torch.stack(
            [expert(shared_input) for expert in self.routing_experts], dim=2
        )
        routing_output = (expert_outputs * sparse_weights.unsqueeze(-1)).sum(dim=2)

        self.load_balance_loss = self._calc_balance_loss(routing_weights, sparse_weights)
        return shared_output + routing_output

    def _calc_balance_loss(self, routing_weights: Tensor, sparse_weights: Tensor) -> Tensor:
        routing_mask = (sparse_weights > 0).float()
        total_tokens = sparse_weights.size(0) * sparse_weights.size(1)
        expert_count = routing_mask.sum(dim=(0, 1))
        f_i = (expert_count / (self.topk * total_tokens)) * self.num_routing
        p_i = routing_weights.mean(dim=(0, 1))
        return (f_i * p_i).sum()


class Embedder(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.gene_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.value_embedding = MoE4Embedder(config)
        self.load_balance_loss: Tensor | None = None

    def forward(self, gene: Tensor, value: Tensor) -> Tensor:
        gene_embedded = self.gene_embedding(gene)
        value_embedded = self.value_embedding(gene_embedded, value)
        self.load_balance_loss = self.value_embedding.load_balance_loss
        return gene_embedded + value_embedded


class RMSNorm(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.eps = config.eps
        self.gamma = nn.Parameter(torch.ones(config.d_model))

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.gamma


class FeedForward(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        if config.hidden_dim is None:
            expanded_dim = 4 * config.d_model
            hidden_dim = int(2 * expanded_dim / 3)
            hidden_dim = config.multiple_of * (
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )
        else:
            hidden_dim = config.hidden_dim

        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.base = config.rope_base
        self.half_dim = self.rotary_dim // 2
        inv_freq = self._build_inv_freq(torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self, device: torch.device) -> Tensor:
        steps = torch.arange(0, self.rotary_dim, 2, device=device)[: self.half_dim].float()
        return 1.0 / (self.base ** (steps / self.rotary_dim))

    def _compute_rotation_matrix(self, positions: Tensor) -> Tensor:
        coord_dim = positions.shape[-1]
        if coord_dim <= 0:
            raise ValueError("positions must have at least one coordinate dimension.")
        repeats = (self.half_dim + coord_dim - 1) // coord_dim
        expanded_positions = positions.repeat(1, 1, repeats)[..., : self.half_dim]
        inv_freq = self._build_inv_freq(positions.device).view(1, 1, -1)
        freqs = expanded_positions.float() * inv_freq
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        batch, seq_len, heads, head_dim = x.shape
        if self.rotary_dim == 0:
            return x
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]

        x_rot = x_rot.view(batch, seq_len, heads, -1, 2)
        x_complex = torch.view_as_complex(x_rot.float().contiguous())
        pos_cis = self._compute_rotation_matrix(positions).unsqueeze(2)

        rotated = torch.view_as_real(x_complex * pos_cis)
        rotated = rotated.view(batch, seq_len, heads, -1).to(dtype=x_pass.dtype)
        return torch.cat([rotated, x_pass], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.use_flash = config.use_flash
        self.rotary = config.rotary

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary_embedding = RotaryPositionalEncoding(config) if self.rotary else None

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.heads, self.head_dim)

        if self.rotary_embedding is not None and positions is not None:
            q = self.rotary_embedding(q, positions)
            k = self.rotary_embedding(k, positions)

        if not self.use_flash:
            raise RuntimeError("OmniCell HF is configured to use only flash-attn.")
        if flash_attn_func is None:
            raise ImportError("flash_attn is required for OmniCell HF attention.")
        if not q.is_cuda:
            raise RuntimeError("flash-attn requires CUDA tensors.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "flash-attn requires fp16 or bf16 q/k/v tensors; run the forward pass "
                "under CUDA autocast or convert the model to half/bfloat16."
            )

        attn_output = flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
            return_attn_probs=False,
        )

        return self.out_proj(attn_output.reshape(batch_size, seq_len, self.d_model))


class TransformerBlock(nn.Module):
    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config)
        self.self_attn = SelfAttention(config)
        self.norm_ffn = RMSNorm(config)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        x = x + self.dropout(self.self_attn(self.norm_attn(x), positions))
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x


class Output(nn.Module):
    """Legacy OmniCell token-value output head.

    The class name and parameter name ``output.W`` intentionally match the
    original checkpoint for direct state-dict conversion.
    """

    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.token_per_cell = config.token_per_cell
        self.W = nn.Parameter(torch.empty(config.d_model, config.d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)

    def _extract_middle_tokens(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        cell_length = self.token_per_cell + 2
        if seq_len % cell_length != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by cell length {cell_length}."
            )
        num_cells = seq_len // cell_length
        x_reshaped = x.view(batch_size, num_cells, cell_length, d_model)
        selected = x_reshaped[:, :, 1:-1, :]
        return selected.reshape(batch_size, num_cells * self.token_per_cell, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self._extract_middle_tokens(x)
        batch_size, n_gene_tokens, d_model = x.shape
        if n_gene_tokens % self.token_per_cell != 0:
            raise ValueError("Gene-token length must be divisible by token_per_cell.")
        num_cells = n_gene_tokens // self.token_per_cell
        x_blocks = x.view(batch_size, num_cells, self.token_per_cell, d_model)
        pooled = x_blocks.mean(dim=2)
        w_sym = (self.W + self.W.T) / 2
        transformed_pooled = torch.matmul(pooled, w_sym).unsqueeze(2)
        scores = torch.einsum("bnid,bnjd->bnj", transformed_pooled, x_blocks)
        return scores.reshape(batch_size, n_gene_tokens).unsqueeze(-1)


class OmniCellPreTrainedModel(PreTrainedModel):
    config_class = OmniCellConfig
    base_model_prefix = "omnicell"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gamma)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class OmniCellModel(OmniCellPreTrainedModel):
    """HF backbone that returns token states and per-cell embeddings."""

    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__(config)
        self.embedder = Embedder(config)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.output = Output(config)
        self.post_init()

    def _default_nonzero_mask(self, input_ids: Tensor) -> Tensor:
        batch, seq_len = input_ids.shape
        cell_len = self.config.cell_token_len
        if seq_len % cell_len != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by cell length {cell_len}."
            )
        mask = torch.zeros_like(input_ids, dtype=torch.float32)
        mask = mask.view(batch, seq_len // cell_len, cell_len)
        mask[:, :, 1:-1] = 1.0
        return mask.view(batch, seq_len)

    def _pool_cells(self, hidden_states: Tensor, nonzero_mask: Tensor | None) -> Tensor:
        batch, seq_len, dim = hidden_states.shape
        cell_len = self.config.cell_token_len
        if seq_len % cell_len != 0:
            raise ValueError(
                f"Sequence length {seq_len} is not divisible by cell length {cell_len}."
            )
        n_cells = seq_len // cell_len
        blocks = hidden_states.view(batch, n_cells, cell_len, dim)

        if nonzero_mask is None:
            gene_mask = torch.ones(
                (batch, n_cells, self.config.token_per_cell),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        else:
            gene_mask = nonzero_mask.view(batch, n_cells, cell_len)[:, :, 1:-1].to(
                dtype=hidden_states.dtype
            )
        gene_states = blocks[:, :, 1:-1, :]
        denom = gene_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        cell_embeddings = (gene_states * gene_mask.unsqueeze(-1)).sum(dim=2) / denom
        if self.config.normalize_cell_embeddings:
            cell_embeddings = F.normalize(cell_embeddings, dim=-1, eps=1e-7)
        return cell_embeddings

    def forward(
        self,
        input_ids: Tensor,
        expression_values: Tensor,
        positions: Tensor | None = None,
        nonzero_mask: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> OmniCellModelOutput | tuple[Tensor, Tensor, Tensor | None]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if positions is None:
            positions = torch.zeros(
                (*input_ids.shape, 2),
                dtype=expression_values.dtype,
                device=input_ids.device,
            )
        if nonzero_mask is None:
            nonzero_mask = self._default_nonzero_mask(input_ids)

        hidden_states = self.embedder(input_ids, expression_values)
        load_balance_loss = self.embedder.load_balance_loss
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)

        cell_embeddings = self._pool_cells(hidden_states, nonzero_mask)
        if not return_dict:
            return hidden_states, cell_embeddings, load_balance_loss
        return OmniCellModelOutput(
            last_hidden_state=hidden_states,
            cell_embeddings=cell_embeddings,
            load_balance_loss=load_balance_loss,
        )


class OmniCellForUnsupervisedFineTuning(OmniCellPreTrainedModel):
    """OmniCell backbone plus the legacy value-reconstruction head."""

    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__(config)
        self.omnicell = OmniCellModel(config)
        self.post_init()

    def _middle_values(self, values: Tensor) -> Tensor:
        batch, seq_len = values.shape
        return values.view(batch, -1, self.config.cell_token_len)[:, :, 1:-1].reshape(
            batch, -1
        )

    def forward(
        self,
        input_ids: Tensor,
        expression_values: Tensor,
        positions: Tensor | None = None,
        nonzero_mask: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> OmniCellUnsupervisedOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.omnicell(
            input_ids=input_ids,
            expression_values=expression_values,
            positions=positions,
            nonzero_mask=nonzero_mask,
            return_dict=True,
        )
        logits = self.omnicell.output(outputs.last_hidden_state).squeeze(-1)
        if labels is None:
            labels = self._middle_values(expression_values)
        if label_mask is None:
            if self.config.unsupervised_loss_on == "nonzero":
                source_mask = (
                    nonzero_mask
                    if nonzero_mask is not None
                    else self.omnicell._default_nonzero_mask(input_ids)
                )
                label_mask = self._middle_values(source_mask).bool()
            else:
                label_mask = torch.ones_like(labels, dtype=torch.bool)

        diff = logits - labels.to(dtype=logits.dtype)
        if self.config.unsupervised_loss == "smooth_l1":
            token_loss = F.smooth_l1_loss(logits, labels.to(dtype=logits.dtype), reduction="none")
        else:
            token_loss = diff.pow(2)
        loss = token_loss[label_mask].mean() if label_mask.any() else token_loss.mean() * 0.0
        if (
            outputs.load_balance_loss is not None
            and self.config.load_balance_loss_weight > 0.0
        ):
            loss = loss + self.config.load_balance_loss_weight * outputs.load_balance_loss

        if not return_dict:
            return loss, logits, outputs.last_hidden_state, outputs.cell_embeddings
        return OmniCellUnsupervisedOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            cell_embeddings=outputs.cell_embeddings,
            load_balance_loss=outputs.load_balance_loss,
        )


class OmniCellForSupervisedFineTuning(OmniCellPreTrainedModel):
    """OmniCell model with a regression/classification head."""

    def __init__(self, config: OmniCellConfig) -> None:
        super().__init__(config)
        self.omnicell = OmniCellModel(config)
        head_dim = config.d_model
        if config.embedding_pooling == "flatten":
            head_dim = config.d_model * config.n_cells_per_sample
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(head_dim, config.num_labels)
        self.post_init()

    def _pool_sample(self, cell_embeddings: Tensor) -> Tensor:
        if self.config.embedding_pooling == "mean":
            return cell_embeddings.mean(dim=1)
        if self.config.embedding_pooling == "flatten":
            return cell_embeddings.reshape(cell_embeddings.shape[0], -1)
        center = min(max(self.config.center_cell_index, 0), cell_embeddings.shape[1] - 1)
        return cell_embeddings[:, center, :]

    def forward(
        self,
        input_ids: Tensor,
        expression_values: Tensor,
        positions: Tensor | None = None,
        nonzero_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> OmniCellSupervisedOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.omnicell(
            input_ids=input_ids,
            expression_values=expression_values,
            positions=positions,
            nonzero_mask=nonzero_mask,
            return_dict=True,
        )
        pooled = self.dropout(self._pool_sample(outputs.cell_embeddings))
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    problem_type = "regression"
                elif labels.dtype in (torch.long, torch.int, torch.int64):
                    problem_type = "single_label_classification"
                else:
                    problem_type = "multi_label_classification"
            else:
                problem_type = self.config.problem_type

            if problem_type == "regression":
                loss = F.mse_loss(logits.squeeze(-1), labels.to(dtype=logits.dtype).view(-1))
            elif problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.to(dtype=logits.dtype)
                )
            else:
                raise ValueError(f"Unsupported problem_type: {problem_type}")

            if (
                outputs.load_balance_loss is not None
                and self.config.load_balance_loss_weight > 0.0
            ):
                loss = loss + self.config.load_balance_loss_weight * outputs.load_balance_loss

        if not return_dict:
            result = (logits, outputs.last_hidden_state, outputs.cell_embeddings)
            return ((loss,) + result) if loss is not None else result
        return OmniCellSupervisedOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            cell_embeddings=outputs.cell_embeddings,
            load_balance_loss=outputs.load_balance_loss,
        )
