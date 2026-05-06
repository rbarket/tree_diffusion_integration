from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tree_diffusion.tokenizer import TreeDiffusionTokenizer


@dataclass
class TreeDiffusionModelConfig:
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    max_input_length: int = 512
    max_target_length: int = 128

    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True

    tie_embeddings: bool = True
    zero_pad_queries: bool = True

    # Edit-target generation is autoregressive: POS token, subtree tokens, EOS.
    causal_decoder: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.max_input_length < 1:
            raise ValueError("max_input_length must be >= 1.")
        if self.max_target_length < 1:
            raise ValueError("max_target_length must be >= 1.")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_encoder_layers < 1:
            raise ValueError("n_encoder_layers must be >= 1.")
        if self.n_decoder_layers < 1:
            raise ValueError("n_decoder_layers must be >= 1.")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be > 0.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        for name, token_id in (
            ("pad_token_id", self.pad_token_id),
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
        ):
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"{name} must be a valid vocabulary id.")


@dataclass
class TreeDiffusionModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    position_accuracy: torch.Tensor | None = None
    token_accuracy: torch.Tensor | None = None


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        if max_len < 1:
            raise ValueError("max_len must be >= 1.")
        if dim <= 0:
            raise ValueError("dim must be > 0.")
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")
        if seq_len > self.embedding.num_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_len={self.embedding.num_embeddings}."
            )
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        return self.embedding(positions)


class TreeDiffusionPolicyModel(nn.Module):
    def __init__(self, config: TreeDiffusionModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.input_position_embedding = LearnedPositionalEmbedding(
            config.max_input_length,
            config.d_model,
        )
        self.target_position_embedding = LearnedPositionalEmbedding(
            config.max_target_length,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_decoder_layers,
        )

        self.output_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def make_padding_mask(
        self,
        ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ids.ndim != 2:
            raise ValueError("ids must have shape (B, L).")
        if attention_mask is not None:
            if attention_mask.shape != ids.shape:
                raise ValueError("attention_mask shape must match ids shape.")
            return attention_mask.eq(0)
        return ids.eq(self.config.pad_token_id)

    def causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if length < 1:
            raise ValueError("length must be >= 1.")
        return torch.triu(
            torch.ones((length, length), device=device, dtype=torch.bool),
            diagonal=1,
        )

    def shift_right(self, target_ids: torch.Tensor) -> torch.Tensor:
        _validate_token_ids("target_ids", target_ids)
        decoder_input_ids = target_ids.new_full(target_ids.shape, self.config.bos_token_id)
        decoder_input_ids[:, 1:] = target_ids[:, :-1]
        return decoder_input_ids

    def encode(
        self,
        input_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_token_ids("input_ids", input_ids)
        _, input_length = input_ids.shape
        if input_length > self.config.max_input_length:
            raise ValueError(
                f"input_ids length {input_length} exceeds "
                f"max_input_length={self.config.max_input_length}."
            )

        memory_padding_mask = self.make_padding_mask(
            input_ids,
            attention_mask=input_attention_mask,
        )
        source = self.token_embedding(input_ids)
        source = source + self.input_position_embedding(input_length, input_ids.device).unsqueeze(0)
        source = self.dropout(source)

        memory = self.encoder(
            source,
            src_key_padding_mask=memory_padding_mask,
        )
        if self.config.zero_pad_queries:
            memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
        return memory, memory_padding_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        *,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_token_ids("decoder_input_ids", decoder_input_ids)
        batch_size, target_length = decoder_input_ids.shape
        if target_length > self.config.max_target_length:
            raise ValueError(
                f"decoder_input_ids length {target_length} exceeds "
                f"max_target_length={self.config.max_target_length}."
            )
        if memory.ndim != 3:
            raise ValueError("memory must have shape (B, L_in, d_model).")
        if memory.shape[0] != batch_size:
            raise ValueError("memory batch size must match decoder_input_ids batch size.")
        if memory.shape[2] != self.config.d_model:
            raise ValueError("memory hidden size must match config.d_model.")
        if memory_padding_mask.shape != memory.shape[:2]:
            raise ValueError("memory_padding_mask shape must match memory batch/length dims.")
        if memory_padding_mask.dtype != torch.bool:
            raise TypeError("memory_padding_mask must be a bool tensor.")

        target_padding_mask = self.make_padding_mask(
            decoder_input_ids,
            attention_mask=target_attention_mask,
        )
        target = self.token_embedding(decoder_input_ids)
        target = target + self.target_position_embedding(target_length, decoder_input_ids.device).unsqueeze(0)
        target = self.dropout(target)

        target_mask = None
        if self.config.causal_decoder:
            target_mask = self.causal_mask(target_length, decoder_input_ids.device)

        hidden = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        hidden = self.output_norm(hidden)
        if self.config.zero_pad_queries:
            hidden = hidden.masked_fill(target_padding_mask.unsqueeze(-1), 0.0)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        input_attention_mask: torch.Tensor | None = None,
        target_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TreeDiffusionModelOutput:
        _validate_token_ids("input_ids", input_ids)
        _validate_token_ids("target_ids", target_ids)
        if input_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("input_ids and target_ids must have the same batch size.")
        if labels is not None:
            if labels.dtype != torch.long:
                raise TypeError("labels must be torch.long.")
            if labels.shape != target_ids.shape:
                raise ValueError("labels shape must match target_ids shape.")

        memory, memory_padding_mask = self.encode(
            input_ids,
            input_attention_mask=input_attention_mask,
        )
        decoder_input_ids = self.shift_right(target_ids)
        hidden = self.decode(
            decoder_input_ids,
            memory=memory,
            memory_padding_mask=memory_padding_mask,
            target_attention_mask=target_attention_mask,
        )
        logits = self.lm_head(hidden)

        if labels is None:
            labels_for_loss = target_ids.clone()
            labels_for_loss[target_ids.eq(self.config.pad_token_id)] = -100
        else:
            labels_for_loss = labels

        valid_labels = labels_for_loss.ne(-100)
        if valid_labels.any():
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels_for_loss.reshape(-1),
                ignore_index=-100,
            )
        else:
            loss = logits.sum() * 0.0

        predictions = logits.argmax(dim=-1)
        position_accuracy = None
        if labels_for_loss.size(1) > 0:
            valid_position = labels_for_loss[:, 0].ne(-100)
            if valid_position.any():
                position_accuracy = (
                    predictions[:, 0][valid_position]
                    .eq(labels_for_loss[:, 0][valid_position])
                    .float()
                    .mean()
                )

        if valid_labels.any():
            token_accuracy = (
                predictions[valid_labels]
                .eq(labels_for_loss[valid_labels])
                .float()
                .mean()
            )
        else:
            token_accuracy = logits.sum() * 0.0

        return TreeDiffusionModelOutput(
            logits=logits,
            loss=loss,
            position_accuracy=position_accuracy,
            token_accuracy=token_accuracy,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        input_ids: torch.Tensor,
        *,
        input_attention_mask: torch.Tensor | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        _validate_token_ids("input_ids", input_ids)
        decode_length = self.config.max_target_length if max_length is None else max_length
        if decode_length < 1:
            raise ValueError("max_length must be >= 1.")
        if decode_length > self.config.max_target_length:
            raise ValueError(
                f"max_length={decode_length} exceeds "
                f"max_target_length={self.config.max_target_length}."
            )

        memory, memory_padding_mask = self.encode(
            input_ids,
            input_attention_mask=input_attention_mask,
        )
        batch_size = input_ids.size(0)
        decoder_input_ids = input_ids.new_full(
            (batch_size, 1),
            self.config.bos_token_id,
        )
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, device=input_ids.device, dtype=torch.bool)

        for _ in range(decode_length):
            target_attention_mask = torch.ones_like(decoder_input_ids)
            hidden = self.decode(
                decoder_input_ids,
                memory=memory,
                memory_padding_mask=memory_padding_mask,
                target_attention_mask=target_attention_mask,
            )
            logits = self.lm_head(hidden)
            next_token = logits[:, -1, :].argmax(dim=-1)
            eos_tokens = next_token.new_full(next_token.shape, self.config.eos_token_id)
            next_token = torch.where(finished, eos_tokens, next_token)
            generated.append(next_token.unsqueeze(1))

            finished = finished | next_token.eq(self.config.eos_token_id)
            if finished.all():
                break
            decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(1)], dim=1)

        return torch.cat(generated, dim=1)


def build_tree_diffusion_policy_model(
    tokenizer: TreeDiffusionTokenizer,
    *,
    max_input_length: int = 512,
    max_target_length: int = 128,
    d_model: int = 256,
    n_heads: int = 8,
    d_ff: int = 1024,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 4,
    dropout: float = 0.1,
) -> TreeDiffusionPolicyModel:
    config = TreeDiffusionModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        dropout=dropout,
    )
    return TreeDiffusionPolicyModel(config)


def _validate_token_ids(name: str, value: torch.Tensor) -> None:
    if value.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long token ID tensors.")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (B, L).")
    if value.size(1) < 1:
        raise ValueError(f"{name} sequence length must be >= 1.")


__all__ = [
    "LearnedPositionalEmbedding",
    "TreeDiffusionModelConfig",
    "TreeDiffusionModelOutput",
    "TreeDiffusionPolicyModel",
    "build_tree_diffusion_policy_model",
]
