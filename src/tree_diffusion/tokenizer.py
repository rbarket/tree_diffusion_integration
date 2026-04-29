from __future__ import annotations

import math
import operator
from typing import Sequence

from src.mathlang.ast import Expr
from src.mathlang.grammar import ALL_VALID_TOKENS
from src.mathlang.serializer import serialize_prefix_tokens
from src.tree_diffusion.edit_path import EditTarget
from src.tree_diffusion.observation import Observation


BASE_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

FIELD_CONTROL_TOKENS: tuple[str, ...] = (
    "<F>",
    "</F>",
    "<CUR>",
    "</CUR>",
    "<DER>",
    "</DER>",
    "<RES>",
    "</RES>",
    "<NUM>",
    "</NUM>",
    "<EDIT>",
    "<NO_DER>",
    "<NO_RES>",
    "<NO_NUM>",
    *(f"<NUM_VALUE_{index}>" for index in range(32)),
    "<NUM_MEAN_ABS>",
    "<NUM_MSE>",
    "<NUM_MAX_ABS>",
)

MAX_NUMERIC_PROBE_VALUES = 32
NUMERIC_ZERO_EPSILON = 1e-12


def numeric_bucket_token(
    value: float | None,
    *,
    numeric_log_min: int = -12,
    numeric_log_max: int = 12,
) -> str:
    if numeric_log_min > numeric_log_max:
        raise ValueError("numeric_log_min must be less than or equal to numeric_log_max.")

    if value is None:
        return "<NUM_NAN>"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "<NUM_NAN>"

    if not math.isfinite(numeric_value):
        return "<NUM_NAN>"

    absolute_value = abs(numeric_value)
    if absolute_value <= NUMERIC_ZERO_EPSILON:
        return "<NUM_ZERO>"

    sign = "POS" if numeric_value > 0.0 else "NEG"
    exponent = math.floor(math.log10(absolute_value))
    exponent = min(max(exponent, numeric_log_min), numeric_log_max)
    return f"<NUM_{sign}_LOG_{exponent}>"


class TreeDiffusionTokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(
        self,
        *,
        max_positions: int = 512,
        numeric_log_min: int = -12,
        numeric_log_max: int = 12,
    ) -> None:
        if max_positions <= 0:
            raise ValueError("max_positions must be positive.")
        if numeric_log_min > numeric_log_max:
            raise ValueError("numeric_log_min must be less than or equal to numeric_log_max.")

        self.max_positions = max_positions
        self.numeric_log_min = numeric_log_min
        self.numeric_log_max = numeric_log_max

        vocabulary = self._build_vocabulary()
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens.")

        self.token_to_id: dict[str, int] = {token: index for index, token in enumerate(vocabulary)}
        self.id_to_token: dict[int, str] = {index: token for token, index in self.token_to_id.items()}
        self.vocab_size = len(vocabulary)

        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.unk_id = self.token_to_id[self.unk_token]

    @property
    def vocabulary_size(self) -> int:
        return self.vocab_size

    def encode_tokens(
        self,
        tokens: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        pad_to_length: int | None = None,
        allow_unk: bool = False,
    ) -> list[int]:
        output_tokens: list[str] = []
        if add_bos:
            output_tokens.append(self.bos_token)
        output_tokens.extend(tokens)
        if add_eos:
            output_tokens.append(self.eos_token)

        ids: list[int] = []
        for token in output_tokens:
            token_id = self.token_to_id.get(token)
            if token_id is None:
                if not allow_unk:
                    raise ValueError(f"Unknown token: {token!r}")
                token_id = self.unk_id
            ids.append(token_id)

        if pad_to_length is not None:
            if len(ids) > pad_to_length:
                raise ValueError(
                    f"Encoded sequence length {len(ids)} exceeds pad_to_length {pad_to_length}."
                )
            ids.extend([self.pad_id] * (pad_to_length - len(ids)))

        return ids

    def decode_ids(self, ids: Sequence[int], *, strip_pad: bool = False) -> list[str]:
        tokens: list[str] = []
        for token_id in ids:
            normalized_id = int(token_id)
            token = self.id_to_token.get(normalized_id)
            if token is None:
                raise ValueError(f"Unknown token id: {token_id!r}")
            tokens.append(token)

        if strip_pad:
            while tokens and tokens[-1] == self.pad_token:
                tokens.pop()

        return tokens

    def position_token(self, position: int) -> str:
        try:
            position = operator.index(position)
        except TypeError as exc:
            raise ValueError(f"Invalid position: {position!r}") from exc

        if position < 0 or position >= self.max_positions:
            raise ValueError(
                f"Invalid position {position}; expected 0 <= position < {self.max_positions}."
            )
        return f"<POS_{position}>"

    def position_id(self, position: int) -> int:
        return self.token_to_id[self.position_token(position)]

    def token_to_position(self, token_or_id: str | int) -> int:
        if isinstance(token_or_id, str):
            token = token_or_id
        else:
            try:
                token_id = operator.index(token_or_id)
            except TypeError as exc:
                raise ValueError(f"Unknown token id: {token_or_id!r}") from exc
            token = self.id_to_token.get(token_id)
            if token is None:
                raise ValueError(f"Unknown token id: {token_or_id!r}")

        if not token.startswith("<POS_") or not token.endswith(">"):
            raise ValueError(f"Not a position token: {token!r}")

        position_text = token[len("<POS_") : -1]
        if not position_text.isdigit():
            raise ValueError(f"Invalid position token: {token!r}")

        position = int(position_text)
        if position < 0 or position >= self.max_positions:
            raise ValueError(
                f"Invalid position {position}; expected 0 <= position < {self.max_positions}."
            )
        if token != self.position_token(position):
            raise ValueError(f"Invalid position token: {token!r}")
        return position

    def serialize_expr(self, expr: Expr) -> list[str]:
        return serialize_prefix_tokens(expr)

    def numeric_bucket_token(self, value: float | None) -> str:
        return numeric_bucket_token(
            value,
            numeric_log_min=self.numeric_log_min,
            numeric_log_max=self.numeric_log_max,
        )

    def serialize_observation(
        self,
        observation: Observation,
        *,
        include_numeric: bool = True,
    ) -> list[str]:
        tokens: list[str] = []
        tokens.extend(("<F>", *self.serialize_expr(observation.target_integrand), "</F>"))
        tokens.extend(("<CUR>", *self.serialize_expr(observation.current_antiderivative), "</CUR>"))

        if observation.current_derivative is None:
            derivative_tokens = ["<NO_DER>"]
        else:
            derivative_tokens = self.serialize_expr(observation.current_derivative)
        tokens.extend(("<DER>", *derivative_tokens, "</DER>"))

        if observation.symbolic_residual is None:
            residual_tokens = ["<NO_RES>"]
        else:
            residual_tokens = self.serialize_expr(observation.symbolic_residual)
        tokens.extend(("<RES>", *residual_tokens, "</RES>"))

        tokens.extend(self._serialize_numeric_probes(observation, include_numeric=include_numeric))
        return tokens

    def serialize_edit_target(
        self,
        edit_target: EditTarget,
        *,
        add_eos: bool = True,
    ) -> list[str]:
        tokens = [
            self.position_token(edit_target.selected_node_id),
            *self.serialize_expr(edit_target.replacement_subtree),
        ]
        if add_eos:
            tokens.append(self.eos_token)
        return tokens

    def serialize_training_pair(
        self,
        observation: Observation,
        edit_target: EditTarget,
    ) -> tuple[list[str], list[str]]:
        input_tokens = self.serialize_observation(observation) + ["<EDIT>"]
        target_tokens = self.serialize_edit_target(edit_target, add_eos=True)
        return input_tokens, target_tokens

    def encode_observation(
        self,
        observation: Observation,
        *,
        include_numeric: bool = True,
        add_bos: bool = False,
        pad_to_length: int | None = None,
    ) -> list[int]:
        return self.encode_tokens(
            self.serialize_observation(observation, include_numeric=include_numeric),
            add_bos=add_bos,
            pad_to_length=pad_to_length,
        )

    def encode_edit_target(
        self,
        edit_target: EditTarget,
        *,
        add_eos: bool = True,
        pad_to_length: int | None = None,
    ) -> list[int]:
        return self.encode_tokens(
            self.serialize_edit_target(edit_target, add_eos=add_eos),
            pad_to_length=pad_to_length,
        )

    def _build_vocabulary(self) -> list[str]:
        return [
            *BASE_SPECIAL_TOKENS,
            *FIELD_CONTROL_TOKENS,
            *sorted(ALL_VALID_TOKENS),
            *self._numeric_bucket_tokens(),
            *(self.position_token(position) for position in range(self.max_positions)),
        ]

    def _numeric_bucket_tokens(self) -> list[str]:
        tokens = ["<NUM_NAN>", "<NUM_ZERO>"]
        for exponent in range(self.numeric_log_min, self.numeric_log_max + 1):
            tokens.append(f"<NUM_POS_LOG_{exponent}>")
            tokens.append(f"<NUM_NEG_LOG_{exponent}>")
        return tokens

    def _serialize_numeric_probes(
        self,
        observation: Observation,
        *,
        include_numeric: bool,
    ) -> list[str]:
        if not include_numeric or observation.numeric_probes is None:
            return ["<NUM>", "<NO_NUM>", "</NUM>"]

        numeric_probes = observation.numeric_probes
        residual_values = numeric_probes.residual_values
        if len(residual_values) > MAX_NUMERIC_PROBE_VALUES:
            raise ValueError(
                f"Numeric probe residual value count {len(residual_values)} exceeds "
                f"the supported maximum of {MAX_NUMERIC_PROBE_VALUES}."
            )

        tokens = ["<NUM>"]
        for index, value in enumerate(residual_values):
            tokens.extend((f"<NUM_VALUE_{index}>", self.numeric_bucket_token(value)))

        tokens.extend(
            (
                "<NUM_MEAN_ABS>",
                self.numeric_bucket_token(numeric_probes.mean_abs_residual),
                "<NUM_MSE>",
                self.numeric_bucket_token(numeric_probes.mean_squared_residual),
                "<NUM_MAX_ABS>",
                self.numeric_bucket_token(numeric_probes.max_abs_residual),
                "</NUM>",
            )
        )
        return tokens


__all__ = [
    "TreeDiffusionTokenizer",
    "numeric_bucket_token",
]
