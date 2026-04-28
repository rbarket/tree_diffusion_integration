from __future__ import annotations

import unittest

from src.mathlang.ast import UnaryOp, Var
from src.mathlang.parser import parse_prefix_string
from src.mathlang.serializer import serialize_prefix_string
from src.tree_diffusion.observation import (
    DEFAULT_PROBE_POINTS,
    NumericProbeFeatures,
    build_observation,
    compute_current_derivative,
    compute_numeric_probes,
    compute_symbolic_residual,
)


class ObservationTests(unittest.TestCase):
    def test_compute_current_derivative_examples(self) -> None:
        cases = {
            "div pow x INT+ 3 INT+ 3": "pow x INT+ 2",
            "sin x": "cos x",
            "cos x": "mul INT- 1 sin x",
            "exp x": "exp x",
            "ln x": "pow x INT- 1",
            "abs x": "sign x",
        }

        for antiderivative, expected in cases.items():
            with self.subTest(antiderivative=antiderivative):
                derivative = compute_current_derivative(parse_prefix_string(antiderivative))
                self.assertEqual(serialize_prefix_string(derivative), expected)

        sec_derivative = compute_current_derivative(parse_prefix_string("sec x"))
        self.assertEqual(serialize_prefix_string(sec_derivative), "mul sec x tan x")

        sech_derivative = compute_current_derivative(parse_prefix_string("sech x"))
        self.assertEqual(
            serialize_prefix_string(sech_derivative),
            "mul INT- 1 mul sech x tanh x",
        )

    def test_symbolic_residual_examples(self) -> None:
        zero_residual = compute_symbolic_residual(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("pow x INT+ 2"),
        )
        self.assertEqual(serialize_prefix_string(zero_residual), "INT+ 0")

        derivative = compute_current_derivative(
            parse_prefix_string("add div pow x INT+ 3 INT+ 5 sin x")
        )
        residual = compute_symbolic_residual(
            current_derivative=derivative,
            target_integrand=parse_prefix_string("add pow x INT+ 2 cos x"),
        )
        self.assertEqual(
            serialize_prefix_string(residual),
            "mul div INT- 2 INT+ 5 pow x INT+ 2",
        )

    def test_build_observation_residual_modes(self) -> None:
        target = parse_prefix_string("pow x INT+ 2")
        current = parse_prefix_string("div pow x INT+ 3 INT+ 3")

        none_observation = build_observation(target, current, residual_mode="none")
        self.assertEqual(none_observation.status, "ok")
        self.assertIsNotNone(none_observation.current_derivative)
        self.assertIsNone(none_observation.symbolic_residual)
        self.assertIsNone(none_observation.numeric_probes)

        symbolic_observation = build_observation(target, current, residual_mode="symbolic")
        self.assertEqual(symbolic_observation.status, "ok")
        self.assertIsNotNone(symbolic_observation.current_derivative)
        self.assertIsNotNone(symbolic_observation.symbolic_residual)
        self.assertEqual(
            serialize_prefix_string(symbolic_observation.symbolic_residual),
            "INT+ 0",
        )
        self.assertIsNone(symbolic_observation.numeric_probes)

        numeric_observation = build_observation(target, current, residual_mode="numeric")
        self.assertEqual(numeric_observation.status, "ok")
        self.assertIsNotNone(numeric_observation.current_derivative)
        self.assertIsNone(numeric_observation.symbolic_residual)
        self.assertIsNotNone(numeric_observation.numeric_probes)

        both_observation = build_observation(target, current, residual_mode="both")
        self.assertEqual(both_observation.status, "ok")
        self.assertIsNotNone(both_observation.current_derivative)
        self.assertIsNotNone(both_observation.symbolic_residual)
        self.assertIsNotNone(both_observation.numeric_probes)

    def test_invalid_residual_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
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

        self.assertEqual(probes.probe_points, DEFAULT_PROBE_POINTS)
        self.assertEqual(probes.finite_mask, tuple(True for _ in DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.residual_values, tuple(0.0 for _ in DEFAULT_PROBE_POINTS))
        self.assertEqual(probes.mean_abs_residual, 0.0)
        self.assertEqual(probes.mean_squared_residual, 0.0)
        self.assertEqual(probes.max_abs_residual, 0.0)
        self.assertEqual(probes.fraction_finite, 1.0)

    def test_numeric_probes_nonzero_and_singularities(self) -> None:
        nonzero = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("add x pow x INT+ 2"),
        )
        self.assertGreater(nonzero.mean_abs_residual or 0.0, 0.0)
        self.assertGreater(nonzero.max_abs_residual or 0.0, 0.0)

        singular = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT- 1"),
            target_integrand=parse_prefix_string("INT+ 0"),
            probe_points=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(singular.finite_mask, (True, False, True))
        self.assertEqual(singular.residual_values, (-1.0, None, 1.0))
        self.assertAlmostEqual(singular.mean_abs_residual or 0.0, 1.0)
        self.assertAlmostEqual(singular.mean_squared_residual or 0.0, 1.0)
        self.assertAlmostEqual(singular.max_abs_residual or 0.0, 1.0)
        self.assertAlmostEqual(singular.fraction_finite, 2.0 / 3.0)

    def test_numeric_probe_features_type_is_returned(self) -> None:
        probes = compute_numeric_probes(
            current_derivative=parse_prefix_string("pow x INT+ 2"),
            target_integrand=parse_prefix_string("pow x INT+ 2"),
        )
        self.assertIsInstance(probes, NumericProbeFeatures)

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

    def test_unsupported_target_expression_returns_partial_observation(self) -> None:
        observation = build_observation(
            target_integrand=UnaryOp(op="asec", operand=Var(name="x")),
            current_antiderivative=parse_prefix_string("div pow x INT+ 3 INT+ 3"),
            residual_mode="both",
        )

        self.assertEqual(observation.status, "partial")
        self.assertIsNotNone(observation.current_derivative)
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

        self.assertIsNotNone(observation.current_derivative)
        assert observation.current_derivative is not None
        self.assertEqual(
            serialize_prefix_string(observation.current_derivative),
            "add INT+ 2 x",
        )
        self.assertIsNotNone(observation.symbolic_residual)
        assert observation.symbolic_residual is not None
        self.assertEqual(
            serialize_prefix_string(observation.symbolic_residual),
            "INT+ 0",
        )

    def test_compute_symbolic_residual_preserves_top_level_constants(self) -> None:
        residual = compute_symbolic_residual(
            current_derivative=parse_prefix_string("add x INT+ 2"),
            target_integrand=parse_prefix_string("x"),
        )
        self.assertEqual(serialize_prefix_string(residual), "INT+ 2")


if __name__ == "__main__":
    unittest.main()
