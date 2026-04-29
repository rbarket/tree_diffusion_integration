from __future__ import annotations

from dataclasses import fields
import math
import unittest
from unittest.mock import patch

import sympy as sp

from src.mathlang.ast import Expr, UnaryOp, Var
from src.mathlang.conversions import ast_to_sympy
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.observation import (
    DEFAULT_PROBE_POINTS,
    NumericProbeFeatures,
    Observation,
    build_observation,
    compute_current_derivative,
    compute_numeric_probes,
    compute_symbolic_residual,
)


EXPECTED_DEFAULT_PROBE_POINTS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)


class ObservationTests(unittest.TestCase):
    def assert_ast_equivalent(self, actual: Expr, expected_prefix: str) -> None:
        expected = parse_prefix_string(expected_prefix)
        residual = sp.simplify(ast_to_sympy(actual) - ast_to_sympy(expected))
        if residual == 0:
            return
        is_zero = residual.equals(0)
        self.assertTrue(
            bool(is_zero) if is_zero is not None else False,
            msg=(
                f"ASTs are not symbolically equivalent:\n"
                f"  actual={serialize_prefix_string(actual)!r}\n"
                f"  expected={expected_prefix!r}\n"
                f"  residual={residual!r}"
            ),
        )

    def assert_is_math_ast(self, expr: Expr | None) -> None:
        self.assertIsNotNone(expr)
        assert expr is not None
        self.assertIsInstance(expr, Expr)
        self.assertNotIsInstance(expr, sp.Basic)

    def test_observation_dataclass_fields_are_inference_only(self) -> None:
        field_names = {field.name for field in fields(Observation)}
        self.assertEqual(
            field_names,
            {
                "target_integrand",
                "current_antiderivative",
                "current_derivative",
                "symbolic_residual",
                "numeric_probes",
                "residual_mode",
                "status",
                "warnings",
            },
        )
        self.assertNotIn("reverse_edit_path", field_names)
        self.assertNotIn("gold_target_antiderivative", field_names)
        self.assertNotIn("target_antiderivative", field_names)

    def test_default_probe_points_are_positive_nonzero_and_deterministic(self) -> None:
        self.assertEqual(DEFAULT_PROBE_POINTS, EXPECTED_DEFAULT_PROBE_POINTS)
        self.assertTrue(all(point > 0.0 for point in DEFAULT_PROBE_POINTS))
        self.assertTrue(all(point != 0.0 for point in DEFAULT_PROBE_POINTS))

    def test_compute_current_derivative_examples(self) -> None:
        cases = (
            ("div pow x INT+ 3 INT+ 3", "pow x INT+ 2"),
            ("sin x", "cos x"),
            ("cos x", "mul INT- 1 sin x"),
            ("exp x", "exp x"),
            ("ln x", "pow x INT- 1"),
            ("INT+ 5", "INT+ 0"),
            ("x", "INT+ 1"),
            ("abs x", "sign x"),
        )

        for antiderivative_prefix, expected_derivative_prefix in cases:
            with self.subTest(antiderivative=antiderivative_prefix):
                derivative = compute_current_derivative(parse_prefix_string(antiderivative_prefix))
                self.assert_is_math_ast(derivative)
                self.assert_ast_equivalent(derivative, expected_derivative_prefix)

        constant_derivative = compute_current_derivative(parse_prefix_string("INT+ 5"))
        self.assertEqual(serialize_prefix_string(constant_derivative), "INT+ 0")

        variable_derivative = compute_current_derivative(parse_prefix_string("x"))
        self.assertEqual(serialize_prefix_string(variable_derivative), "INT+ 1")

    def test_compute_symbolic_residual_examples(self) -> None:
        zero_residual = compute_symbolic_residual(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("pow x INT+ 2"),
        )
        self.assert_is_math_ast(zero_residual)
        self.assert_ast_equivalent(zero_residual, "INT+ 0")

        coefficient_mismatch = compute_symbolic_residual(
            current_derivative=compute_current_derivative(
                parse_prefix_string("add div pow x INT+ 3 INT+ 5 sin x")
            ),
            target_integrand=parse_prefix_string("add pow x INT+ 2 cos x"),
        )
        self.assert_is_math_ast(coefficient_mismatch)
        self.assert_ast_equivalent(
            coefficient_mismatch,
            "mul div INT- 2 INT+ 5 pow x INT+ 2",
        )

        extra_term = compute_symbolic_residual(
            current_derivative=compute_current_derivative(
                parse_prefix_string("add div pow x INT+ 3 INT+ 3 sin x")
            ),
            target_integrand=parse_prefix_string("pow x INT+ 2"),
        )
        self.assert_is_math_ast(extra_term)
        self.assert_ast_equivalent(extra_term, "cos x")

        missing_term = compute_symbolic_residual(
            current_derivative=compute_current_derivative(
                parse_prefix_string("div pow x INT+ 3 INT+ 3")
            ),
            target_integrand=parse_prefix_string("add pow x INT+ 2 cos x"),
        )
        self.assert_is_math_ast(missing_term)
        self.assert_ast_equivalent(missing_term, "mul INT- 1 cos x")

        constant_residual = compute_symbolic_residual(
            current_derivative=compute_current_derivative(
                parse_prefix_string("add div pow x INT+ 2 INT+ 2 mul INT+ 5 x")
            ),
            target_integrand=parse_prefix_string("x"),
        )
        self.assert_is_math_ast(constant_residual)
        self.assertEqual(serialize_prefix_string(constant_residual), "INT+ 5")
        self.assert_ast_equivalent(constant_residual, "INT+ 5")

    def test_build_observation_residual_modes(self) -> None:
        target = parse_prefix_string("pow x INT+ 2")
        current = parse_prefix_string("div pow x INT+ 3 INT+ 3")

        none_observation = build_observation(target, current, residual_mode="none")
        self.assertEqual(none_observation.status, "ok")
        self.assert_is_math_ast(none_observation.current_derivative)
        self.assertIsNone(none_observation.symbolic_residual)
        self.assertIsNone(none_observation.numeric_probes)

        symbolic_observation = build_observation(target, current, residual_mode="symbolic")
        self.assertEqual(symbolic_observation.status, "ok")
        self.assert_is_math_ast(symbolic_observation.current_derivative)
        self.assert_is_math_ast(symbolic_observation.symbolic_residual)
        assert symbolic_observation.symbolic_residual is not None
        self.assert_ast_equivalent(symbolic_observation.symbolic_residual, "INT+ 0")
        self.assertIsNone(symbolic_observation.numeric_probes)

        numeric_observation = build_observation(target, current, residual_mode="numeric")
        self.assertEqual(numeric_observation.status, "ok")
        self.assert_is_math_ast(numeric_observation.current_derivative)
        self.assertIsNone(numeric_observation.symbolic_residual)
        self.assertIsNotNone(numeric_observation.numeric_probes)

        both_observation = build_observation(target, current, residual_mode="both")
        self.assertEqual(both_observation.status, "ok")
        self.assert_is_math_ast(both_observation.current_derivative)
        self.assert_is_math_ast(both_observation.symbolic_residual)
        self.assertIsNotNone(both_observation.numeric_probes)

    def test_invalid_residual_mode_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported residual_mode"):
            build_observation(
                parse_prefix_string("pow x INT+ 2"),
                parse_prefix_string("div pow x INT+ 3 INT+ 3"),
                residual_mode="invalid",
            )

    def test_numeric_probes_zero_residual(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("pow x INT+ 2"),
        )

        self.assertEqual(probes.probe_points, EXPECTED_DEFAULT_PROBE_POINTS)
        self.assertEqual(probes.finite_mask, tuple(True for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.complex_mask, tuple(False for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.residual_real, tuple(0.0 for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.residual_imag, tuple(0.0 for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.residual_abs, tuple(0.0 for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(
            probes.residual_abs_squared,
            tuple(0.0 for _ in EXPECTED_DEFAULT_PROBE_POINTS),
        )
        self.assertEqual(probes.mean_abs_residual, 0.0)
        self.assertEqual(probes.mean_squared_abs_residual, 0.0)
        self.assertEqual(probes.mean_squared_residual, 0.0)
        self.assertEqual(probes.max_abs_residual, 0.0)
        self.assertEqual(probes.fraction_finite, 1.0)
        self.assertEqual(probes.fraction_complex, 0.0)

    def test_numeric_probes_nonzero_real_residual_matches_expected_values(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("INT+ 0"),
        )

        self.assertEqual(probes.fraction_complex, 0.0)
        for point, real_part, imag_part, abs_value, abs_squared in zip(
            probes.probe_points,
            probes.residual_real,
            probes.residual_imag,
            probes.residual_abs,
            probes.residual_abs_squared,
        ):
            expected = point * point
            assert real_part is not None
            assert imag_part is not None
            assert abs_value is not None
            assert abs_squared is not None
            self.assertAlmostEqual(real_part, expected)
            self.assertAlmostEqual(imag_part, 0.0)
            self.assertAlmostEqual(abs_value, abs(expected))
            self.assertAlmostEqual(abs_squared, expected * expected)

    def test_numeric_probe_aggregates_use_complex_magnitude(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("add INT+ 3 I"),
            target_integrand=parse_prefix_string("INT+ 0"),
            probe_points=(1.0,),
        )

        self.assertEqual(probes.residual_real, (3.0,))
        self.assertEqual(probes.residual_imag, (1.0,))
        self.assertEqual(probes.complex_mask, (True,))
        self.assertAlmostEqual(probes.residual_abs[0] or 0.0, math.sqrt(10.0))
        self.assertAlmostEqual(probes.residual_abs_squared[0] or 0.0, 10.0)
        self.assertAlmostEqual(probes.mean_abs_residual or 0.0, math.sqrt(10.0))
        self.assertAlmostEqual(probes.mean_squared_abs_residual or 0.0, 10.0)

    def test_numeric_probes_pure_imaginary_residual(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("I"),
            target_integrand=parse_prefix_string("INT+ 0"),
        )

        self.assertEqual(probes.finite_mask, tuple(True for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.complex_mask, tuple(True for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.fraction_complex, 1.0)
        for real_part, imag_part, abs_value, abs_squared in zip(
            probes.residual_real,
            probes.residual_imag,
            probes.residual_abs,
            probes.residual_abs_squared,
        ):
            assert real_part is not None
            assert imag_part is not None
            assert abs_value is not None
            assert abs_squared is not None
            self.assertAlmostEqual(real_part, 0.0)
            self.assertAlmostEqual(imag_part, 1.0)
            self.assertAlmostEqual(abs_value, 1.0)
            self.assertAlmostEqual(abs_squared, 1.0)

    def test_numeric_probes_mixed_real_imaginary_residual(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("add x I"),
            target_integrand=parse_prefix_string("x"),
        )

        self.assertEqual(probes.fraction_complex, 1.0)
        self.assertEqual(probes.complex_mask, tuple(True for _ in EXPECTED_DEFAULT_PROBE_POINTS))
        for real_part, imag_part, abs_value in zip(
            probes.residual_real,
            probes.residual_imag,
            probes.residual_abs,
        ):
            assert real_part is not None
            assert imag_part is not None
            assert abs_value is not None
            self.assertAlmostEqual(real_part, 0.0)
            self.assertAlmostEqual(imag_part, 1.0)
            self.assertAlmostEqual(abs_value, 1.0)

    def test_numeric_probes_log_negative_point_is_complex_but_finite(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("ln x"),
            target_integrand=parse_prefix_string("INT+ 0"),
            probe_points=(-1.0, 1.0),
        )

        self.assertEqual(probes.finite_mask, (True, True))
        self.assertEqual(probes.complex_mask, (True, False))
        self.assertEqual(probes.fraction_finite, 1.0)
        self.assertEqual(probes.fraction_complex, 0.5)
        assert probes.residual_real[0] is not None
        assert probes.residual_imag[0] is not None
        assert probes.residual_abs[0] is not None
        self.assertAlmostEqual(probes.residual_real[0], 0.0)
        self.assertAlmostEqual(probes.residual_imag[0], math.pi)
        self.assertAlmostEqual(probes.residual_abs[0], math.pi)
        self.assertAlmostEqual(probes.residual_real[1] or 0.0, 0.0)
        self.assertAlmostEqual(probes.residual_imag[1] or 0.0, 0.0)

    def test_numeric_probes_complex_plus_singular_is_safe(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("mul I pow x INT- 1"),
            target_integrand=parse_prefix_string("INT+ 0"),
            probe_points=(0.0, 1.0),
        )

        self.assertEqual(probes.finite_mask, (False, True))
        self.assertEqual(probes.complex_mask, (False, True))
        self.assertEqual(probes.fraction_finite, 0.5)
        self.assertEqual(probes.fraction_complex, 1.0)
        self.assertEqual(probes.residual_real, (None, 0.0))
        self.assertEqual(probes.residual_imag, (None, 1.0))
        self.assertEqual(probes.residual_abs, (None, 1.0))
        self.assertEqual(probes.residual_abs_squared, (None, 1.0))

    def test_numeric_probes_singularities_are_partial_and_safe(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT- 1"),
            target_integrand=parse_prefix_string("INT+ 0"),
            probe_points=(0.0, 1.0),
        )

        self.assertEqual(probes.finite_mask, (False, True))
        self.assertEqual(probes.complex_mask, (False, False))
        self.assertEqual(probes.residual_real, (None, 1.0))
        self.assertEqual(probes.residual_imag, (None, 0.0))
        self.assertEqual(probes.residual_abs, (None, 1.0))
        self.assertEqual(probes.residual_abs_squared, (None, 1.0))
        self.assertAlmostEqual(probes.mean_abs_residual or 0.0, 1.0)
        self.assertAlmostEqual(probes.mean_squared_abs_residual or 0.0, 1.0)
        self.assertAlmostEqual(probes.max_abs_residual or 0.0, 1.0)
        self.assertAlmostEqual(probes.fraction_finite, 0.5)
        self.assertAlmostEqual(probes.fraction_complex, 0.0)

    def test_numeric_probe_features_type_is_returned(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("pow x INT+ 2"),
        )
        self.assertIsInstance(probes, NumericProbeFeatures)

    def test_current_derivative_and_symbolic_residual_are_asts_when_present(self) -> None:
        observation = build_observation(
            target_integrand=parse_prefix_string("pow x INT+ 2"),
            current_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            residual_mode="both",
        )

        self.assert_is_math_ast(observation.current_derivative)
        self.assert_is_math_ast(observation.symbolic_residual)

    def test_symbolic_residual_failure_does_not_kill_numeric_both_mode(self) -> None:
        with patch(
            "src.tree_diffusion.observation._compute_symbolic_residual_sympy",
            side_effect=RuntimeError("boom"),
        ):
            observation = build_observation(
                target_integrand=parse_prefix_string("pow x INT+ 2"),
                current_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
                residual_mode="both",
            )

        self.assertEqual(observation.status, "partial")
        self.assert_is_math_ast(observation.current_derivative)
        self.assertIsNone(observation.symbolic_residual)
        self.assertIsNotNone(observation.numeric_probes)
        self.assertIn("symbolic_residual_failed:RuntimeError", observation.warnings)
        self.assertFalse(any(warning.startswith("numeric_probe_failed:") for warning in observation.warnings))

    def test_partial_numeric_failure_in_build_observation_stays_ok(self) -> None:
        observation = build_observation(
            target_integrand=parse_prefix_string("INT+ 0"),
            current_antiderivative=parse_prefix_string("ln x"),
            residual_mode="numeric",
            probe_points=(0.0, 1.0),
        )

        self.assertEqual(observation.status, "ok")
        self.assert_is_math_ast(observation.current_derivative)
        self.assertIsNone(observation.symbolic_residual)
        self.assertIsNotNone(observation.numeric_probes)
        assert observation.numeric_probes is not None
        self.assertEqual(observation.numeric_probes.finite_mask, (False, True))
        self.assertIn("numeric_probe_nonfinite:1/2", observation.warnings)

    def test_unsupported_current_expression_returns_derivative_failed_observation(self) -> None:
        observation = build_observation(
            target_integrand=Var(name="x"),
            current_antiderivative=UnaryOp(op="asec", operand=Var(name="x")),
            residual_mode="both",
        )

        self.assertEqual(observation.status, "derivative_failed")
        self.assertIsNone(observation.current_derivative)
        self.assertIsNone(observation.symbolic_residual)
        self.assertIsNone(observation.numeric_probes)
        self.assertIn("derivative_failed:KeyError", observation.warnings)

    def test_numeric_mode_observation_accepts_complex_probes(self) -> None:
        observation = build_observation(
            target_integrand=parse_prefix_string("INT+ 0"),
            current_antiderivative=parse_prefix_string("mul I x"),
            residual_mode="numeric",
        )

        self.assertEqual(observation.status, "ok")
        self.assert_is_math_ast(observation.current_derivative)
        self.assertIsNone(observation.symbolic_residual)
        self.assertIsNotNone(observation.numeric_probes)
        assert observation.numeric_probes is not None
        self.assertEqual(observation.numeric_probes.fraction_complex, 1.0)
        self.assertTrue(all(observation.numeric_probes.finite_mask))
        self.assertTrue(any(warning.startswith("numeric_probe_complex:") for warning in observation.warnings))
        self.assertFalse(any(warning.startswith("numeric_probe_failed:") for warning in observation.warnings))

    def test_unsupported_target_expression_returns_partial_observation(self) -> None:
        observation = build_observation(
            target_integrand=UnaryOp(op="asec", operand=Var(name="x")),
            current_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            residual_mode="both",
        )

        self.assertEqual(observation.status, "partial")
        self.assert_is_math_ast(observation.current_derivative)
        self.assertIsNone(observation.symbolic_residual)
        self.assertIsNone(observation.numeric_probes)
        self.assertIn("target_integrand_sympy_failed:KeyError", observation.warnings)

    def test_token_caps_downgrade_components_without_crashing(self) -> None:
        target = parse_prefix_string("pow x INT+ 2")
        current = parse_prefix_string("div pow x INT+ 3 INT+ 3")

        derivative_capped = build_observation(
            target,
            current,
            residual_mode="both",
            max_derivative_tokens=3,
        )
        self.assertEqual(derivative_capped.status, "partial")
        self.assertIsNone(derivative_capped.current_derivative)
        self.assertIsNotNone(derivative_capped.symbolic_residual)
        self.assertIsNotNone(derivative_capped.numeric_probes)
        self.assertIn("current_derivative_token_cap_exceeded:4>3", derivative_capped.warnings)

        residual_capped = build_observation(
            target,
            current,
            residual_mode="both",
            max_residual_tokens=1,
        )
        self.assertEqual(residual_capped.status, "partial")
        self.assertIsNotNone(residual_capped.current_derivative)
        self.assertIsNone(residual_capped.symbolic_residual)
        self.assertIsNotNone(residual_capped.numeric_probes)
        self.assertIn("symbolic_residual_token_cap_exceeded:2>1", residual_capped.warnings)

    def test_build_observation_preserves_additive_constants(self) -> None:
        observation = build_observation(
            target_integrand=parse_prefix_string("add x INT+ 2"),
            current_antiderivative=parse_prefix_string("add div pow x INT+ 2 INT+ 2 mul INT+ 2 x"),
            residual_mode="symbolic",
        )

        self.assert_is_math_ast(observation.current_derivative)
        self.assert_is_math_ast(observation.symbolic_residual)
        assert observation.current_derivative is not None
        assert observation.symbolic_residual is not None
        self.assert_ast_equivalent(observation.current_derivative, "add INT+ 2 x")
        self.assert_ast_equivalent(observation.symbolic_residual, "INT+ 0")

    def test_compute_symbolic_residual_preserves_top_level_constants(self) -> None:
        residual = compute_symbolic_residual(
            current_derivative=parse_prefix_string("add x INT+ 2"),
            target_integrand=parse_prefix_string("x"),
        )
        self.assertEqual(serialize_prefix_string(residual), "INT+ 2")


if __name__ == "__main__":
    unittest.main()
