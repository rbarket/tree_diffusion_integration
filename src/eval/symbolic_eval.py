from __future__ import annotations

import contextlib
from dataclasses import dataclass
import signal
import threading
from typing import Iterable, List, Optional, Sequence

import sympy as sp
import torch

from src.data.vocab import Vocab
from src.mathlang.conversions import prefix_tokens_to_sympy


class SymbolicEvaluationTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def _time_limit(seconds: Optional[float]):
    if seconds is None:
        yield
        return
    if seconds <= 0.0:
        raise ValueError("timeout_seconds must be > 0 when provided.")

    # SIGALRM-based timeout only works in main thread on Unix-like systems.
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def _handler(signum, frame):
        del signum, frame
        raise SymbolicEvaluationTimeout(f"symbolic evaluation timed out after {seconds:.3f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        signal.signal(signal.SIGALRM, old_handler)


@dataclass(frozen=True)
class SymbolicEvalResult:
    is_correct: bool
    reason: str
    prediction_has_eos: bool
    integrand_has_eos: bool
    prediction_parse_ok: bool
    integrand_parse_ok: bool
    prediction_error: Optional[str] = None
    integrand_error: Optional[str] = None


@dataclass
class SymbolicEvalSummary:
    total: int = 0
    correct: int = 0
    derivative_mismatch: int = 0
    evaluation_timeout: int = 0
    prediction_parse_fail: int = 0
    integrand_parse_fail: int = 0
    prediction_missing_eos: int = 0
    integrand_missing_eos: int = 0
    exceptions: int = 0
    last_predicted_answer: Optional[str] = None
    last_true_answer: Optional[str] = None

    @property
    def incorrect(self) -> int:
        return self.total - self.correct

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return float(self.correct) / float(self.total)

    def update(
        self,
        result: SymbolicEvalResult,
        *,
        predicted_answer: Optional[str] = None,
        true_answer: Optional[str] = None,
    ) -> None:
        if predicted_answer is not None:
            self.last_predicted_answer = predicted_answer
        if true_answer is not None:
            self.last_true_answer = true_answer

        self.total += 1
        if result.is_correct:
            self.correct += 1
            return
        if result.reason == "prediction_missing_eos":
            self.prediction_missing_eos += 1
        elif result.reason == "integrand_missing_eos":
            self.integrand_missing_eos += 1
        elif result.reason == "prediction_parse_error":
            self.prediction_parse_fail += 1
        elif result.reason == "integrand_parse_error":
            self.integrand_parse_fail += 1
        elif result.reason == "evaluation_timeout":
            self.evaluation_timeout += 1
        elif result.reason == "derivative_mismatch":
            self.derivative_mismatch += 1
        else:
            self.exceptions += 1

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy,
            "derivative_mismatch": self.derivative_mismatch,
            "evaluation_timeout": self.evaluation_timeout,
            "prediction_parse_fail": self.prediction_parse_fail,
            "integrand_parse_fail": self.integrand_parse_fail,
            "prediction_missing_eos": self.prediction_missing_eos,
            "integrand_missing_eos": self.integrand_missing_eos,
            "exceptions": self.exceptions,
            "last_predicted_answer": self.last_predicted_answer,
            "last_true_answer": self.last_true_answer,
        }


def _extract_expression_tokens(
    sequence_tokens: Sequence[str],
    *,
    bos_token: str,
    eos_token: str,
) -> tuple[List[str], bool]:
    """
    Convert a fixed-length sequence with BOS/EOS/PAD convention into pure expression tokens.

    We enforce "first EOS terminates the sequence". If EOS is missing, the sequence is
    considered invalid for this harness and we still return all tokens after BOS so the caller
    can inspect/debug if needed.
    """
    tokens = list(sequence_tokens)
    start = 1 if tokens and tokens[0] == bos_token else 0
    eos_idx = None
    for i in range(start, len(tokens)):
        if tokens[i] == eos_token:
            eos_idx = i
            break
    if eos_idx is None:
        return tokens[start:], False
    return tokens[start:eos_idx], True


def _safe_prefix_to_sympy(tokens: Sequence[str]) -> tuple[Optional[sp.Expr], bool, Optional[str]]:
    if len(tokens) == 0:
        return None, False, "empty_expression"
    try:
        return prefix_tokens_to_sympy(list(tokens)), True, None
    except Exception as exc:
        return None, False, f"{type(exc).__name__}: {exc}"


def _is_derivative_equivalent(pred_integral: sp.Expr, integrand: sp.Expr, *, variable: str = "x") -> bool:
    x = sp.Symbol(variable, real=True)
    residual = sp.simplify(sp.simplify(sp.diff(pred_integral, x)) - sp.simplify(integrand))
    if residual == 0:
        return True
    eq = residual.equals(0)
    return bool(eq) if eq is not None else False


def evaluate_prefix_pair(
    *,
    integrand_tokens: Sequence[str],
    prediction_tokens: Sequence[str],
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
    variable: str = "x",
    timeout_seconds: Optional[float] = None,
) -> SymbolicEvalResult:
    """
    Evaluate one (integrand, predicted_antiderivative) pair.

    Success criterion:
      d/d(variable) [prediction] == integrand   (symbolically)

    Any parsing failure is counted as incorrect.
    """
    x_expr_tokens, x_has_eos = _extract_expression_tokens(
        integrand_tokens,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    y_expr_tokens, y_has_eos = _extract_expression_tokens(
        prediction_tokens,
        bos_token=bos_token,
        eos_token=eos_token,
    )

    if not x_has_eos:
        return SymbolicEvalResult(
            is_correct=False,
            reason="integrand_missing_eos",
            prediction_has_eos=y_has_eos,
            integrand_has_eos=False,
            prediction_parse_ok=False,
            integrand_parse_ok=False,
            integrand_error="missing_eos",
        )
    if not y_has_eos:
        return SymbolicEvalResult(
            is_correct=False,
            reason="prediction_missing_eos",
            prediction_has_eos=False,
            integrand_has_eos=True,
            prediction_parse_ok=False,
            integrand_parse_ok=False,
            prediction_error="missing_eos",
        )

    x_expr, x_ok, x_err = _safe_prefix_to_sympy(x_expr_tokens)
    if not x_ok or x_expr is None:
        return SymbolicEvalResult(
            is_correct=False,
            reason="integrand_parse_error",
            prediction_has_eos=True,
            integrand_has_eos=True,
            prediction_parse_ok=False,
            integrand_parse_ok=False,
            integrand_error=x_err,
        )

    y_expr, y_ok, y_err = _safe_prefix_to_sympy(y_expr_tokens)
    if not y_ok or y_expr is None:
        return SymbolicEvalResult(
            is_correct=False,
            reason="prediction_parse_error",
            prediction_has_eos=True,
            integrand_has_eos=True,
            prediction_parse_ok=False,
            integrand_parse_ok=True,
            prediction_error=y_err,
        )

    try:
        with _time_limit(timeout_seconds):
            is_eq = _is_derivative_equivalent(y_expr, x_expr, variable=variable)
    except SymbolicEvaluationTimeout as exc:
        return SymbolicEvalResult(
            is_correct=False,
            reason="evaluation_timeout",
            prediction_has_eos=True,
            integrand_has_eos=True,
            prediction_parse_ok=True,
            integrand_parse_ok=True,
            prediction_error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        return SymbolicEvalResult(
            is_correct=False,
            reason="evaluation_exception",
            prediction_has_eos=True,
            integrand_has_eos=True,
            prediction_parse_ok=True,
            integrand_parse_ok=True,
            prediction_error=f"{type(exc).__name__}: {exc}",
        )

    if is_eq:
        return SymbolicEvalResult(
            is_correct=True,
            reason="ok",
            prediction_has_eos=True,
            integrand_has_eos=True,
            prediction_parse_ok=True,
            integrand_parse_ok=True,
        )
    return SymbolicEvalResult(
        is_correct=False,
        reason="derivative_mismatch",
        prediction_has_eos=True,
        integrand_has_eos=True,
        prediction_parse_ok=True,
        integrand_parse_ok=True,
    )


def evaluate_prefix_pairs(
    *,
    integrands: Iterable[Sequence[str]],
    predictions: Iterable[Sequence[str]],
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
    variable: str = "x",
    timeout_seconds: Optional[float] = None,
) -> SymbolicEvalSummary:
    summary = SymbolicEvalSummary()
    for x_toks, y_toks in zip(integrands, predictions):
        result = evaluate_prefix_pair(
            integrand_tokens=x_toks,
            prediction_tokens=y_toks,
            bos_token=bos_token,
            eos_token=eos_token,
            variable=variable,
            timeout_seconds=timeout_seconds,
        )
        summary.update(result)
    return summary


def _decode_row_ids(row_ids: Sequence[int], vocab: Vocab) -> List[str]:
    return vocab.decode([int(v) for v in row_ids])


def evaluate_id_pair(
    *,
    integrand_ids: Sequence[int] | torch.Tensor,
    prediction_ids: Sequence[int] | torch.Tensor,
    vocab: Vocab,
    variable: str = "x",
    timeout_seconds: Optional[float] = None,
) -> SymbolicEvalResult:
    if isinstance(integrand_ids, torch.Tensor):
        x_ids = integrand_ids.detach().cpu().tolist()
    else:
        x_ids = list(integrand_ids)

    if isinstance(prediction_ids, torch.Tensor):
        y_ids = prediction_ids.detach().cpu().tolist()
    else:
        y_ids = list(prediction_ids)

    specials = vocab.specials
    return evaluate_prefix_pair(
        integrand_tokens=_decode_row_ids(x_ids, vocab),
        prediction_tokens=_decode_row_ids(y_ids, vocab),
        bos_token=specials.bos,
        eos_token=specials.eos,
        variable=variable,
        timeout_seconds=timeout_seconds,
    )


def evaluate_id_batch(
    *,
    integrand_ids: torch.Tensor,
    prediction_ids: torch.Tensor,
    vocab: Vocab,
    variable: str = "x",
    timeout_seconds: Optional[float] = None,
) -> SymbolicEvalSummary:
    if integrand_ids.ndim != 2 or prediction_ids.ndim != 2:
        raise ValueError("integrand_ids and prediction_ids must both be rank-2 tensors (B, L).")
    if integrand_ids.shape[0] != prediction_ids.shape[0]:
        raise ValueError("Batch sizes do not match between integrand_ids and prediction_ids.")

    summary = SymbolicEvalSummary()
    for i in range(integrand_ids.shape[0]):
        result = evaluate_id_pair(
            integrand_ids=integrand_ids[i],
            prediction_ids=prediction_ids[i],
            vocab=vocab,
            variable=variable,
            timeout_seconds=timeout_seconds,
        )
        summary.update(result)
    return summary
