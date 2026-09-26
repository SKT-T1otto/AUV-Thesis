"""Analytic tests only: no simulator, policy source, trainer, or rollout."""
import itertools
import json

import numpy as np
import pytest

from scripts.hgr_decision_statistics import summarize_pair


def costs(repeats=2):
    return dict(hgr_steps=np.full(repeats, 10), direct_steps=np.full(repeats, 5))


def test_known_cross_product_risks_and_costs():
    result = summarize_pair(
        [[1, 0], [3, 0]], [[2, 0], [4, 0]], [[4, -1], [4, 1]],
        dict(hgr_steps=[10, 20], direct_steps=[5, 5]),
    )
    assert result["repeat_count"] == 2
    assert result["gradient_dimension"] == 2
    assert result["signal"]["debiased_signal_energy"] == 8
    assert result["signal"]["mean_standard_error_scale"] == 1
    assert result["signal"]["relative_mean_delta_norm"] == pytest.approx(3 / np.sqrt(5))
    risks = result["risk_estimates"]
    assert risks["direct_target_squared_norm_u"] == 15
    assert risks["methods"]["g0"]["single_macro_mse"] == 4
    assert risks["methods"]["hgr"]["single_macro_mse"] == 4
    assert risks["methods"]["hgr"]["mean_of_repeats_mse"] == 0
    assert risks["methods"]["direct"]["single_macro_mse"] == 2
    assert risks["per_repeat_hgr_minus_g0_risk"] == [-8, 8]
    assert risks["matched_hgr_minus_g0_single_macro_mse"] == 0
    assert risks["matched_hgr_minus_g0_mean_of_repeats_mse"] == -3
    assert result["cost_proxy"]["hgr"] == 120
    assert result["cost_proxy"]["direct"] == 10
    assert result["cost_proxy"]["hgr_over_direct"] == 12
    assert result["cost_proxy"]["actual_equal_budget_comparison"] is False
    json.dumps(result, allow_nan=False)


def test_exact_enumeration_of_additive_noise_matches_analytic_expectations():
    # Six independent Rademacher variables: two g0, two correction and two
    # Direct draws. Enumerating the complete distribution tests expectations,
    # rather than checking a reimplementation against itself.
    summaries = []
    for signs in itertools.product((-1, 1), repeat=6):
        old = np.array([[0, -2], [0, -2]], dtype=float)
        old[:, 0] += .5 * np.array(signs[:2])
        delta = np.array([[1, 0], [1, 0]], dtype=float)
        delta[:, 1] += .25 * np.array(signs[2:4])
        direct = np.array([[1, -2], [1, -2]], dtype=float)
        direct[:, 1] += np.array(signs[4:])
        summaries.append(summarize_pair(old, delta, direct, costs()))
    assert np.mean([s["signal"]["debiased_signal_energy"] for s in summaries]) == pytest.approx(1)
    assert np.mean([s["signal"]["trace_sample_variance"] for s in summaries]) == pytest.approx(.0625)
    for method, single, averaged in (("g0", 1.25, 1.125), ("hgr", .3125, .15625), ("direct", 1, .5)):
        assert np.mean([s["risk_estimates"]["methods"][method]["single_macro_mse"] for s in summaries]) == pytest.approx(single)
        assert np.mean([s["risk_estimates"]["methods"][method]["mean_of_repeats_mse"] for s in summaries]) == pytest.approx(averaged)
    assert np.mean([s["risk_estimates"]["matched_hgr_minus_g0_single_macro_mse"] for s in summaries]) == pytest.approx(-.9375)


def test_null_correction_and_negative_noise_corrected_risk_are_preserved():
    result = summarize_pair([[0], [0]], [[0], [0]], [[-1], [1]], costs())
    assert result["signal"]["debiased_signal_energy"] == 0
    assert result["signal"]["relative_mean_delta_norm"] is None
    assert result["signal"]["signal_to_noise_ratio"] is None
    assert result["risk_estimates"]["methods"]["g0"]["single_macro_mse"] == -1
    assert result["risk_estimates"]["matched_hgr_minus_g0_single_macro_mse"] == 0
    json.dumps(result, allow_nan=False)


def test_noisy_zero_mean_can_have_negative_signal_energy():
    result = summarize_pair([[2], [2]], [[-1], [1]], [[2], [2]], costs())
    assert result["signal"]["debiased_signal_energy"] == -1
    assert result["signal"]["debiased_relative_signal_energy"] == -.25
    assert result["cost_proxy"]["hgr_over_direct"] is None


def test_matched_contrast_preserves_tiny_correction():
    result = summarize_pair([[1e8], [1e8]], [[1e-12], [1e-12]], [[0], [0]], costs())
    assert result["risk_estimates"]["matched_hgr_minus_g0_single_macro_mse"] == pytest.approx(2e-4)


def test_explicit_direct_pairwise_target_energy_for_more_than_two_repeats():
    direct = np.array([[1, 2], [3, -1], [-2, 4], [5, 0]], dtype=float)
    target = sum(direct[i] @ direct[j] for i in range(4) for j in range(4) if i != j) / 12
    result = summarize_pair(np.zeros((4, 2)), np.zeros((4, 2)), direct, costs(4))
    assert result["risk_estimates"]["direct_target_squared_norm_u"] == pytest.approx(target)


@pytest.mark.parametrize("bad", [np.ones((1, 2)), np.ones((2, 0)), np.ones((2, 2, 3)), [1, 2]])
def test_invalid_shape_or_pooled_policy_pairs_rejected(bad):
    with pytest.raises(ValueError, match="one frozen policy pair"):
        summarize_pair(bad, [[1], [1]], [[1], [1]], costs())


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError, match="identical"):
        summarize_pair(np.ones((2, 2)), np.ones((2, 1)), np.ones((2, 2)), costs())


@pytest.mark.parametrize("field", ["g0", "delta", "direct"])
def test_nonfinite_vectors_rejected(field):
    values = {key: [[1], [1]] for key in ("g0", "delta", "direct")}
    values[field] = [[float("nan")], [1]]
    with pytest.raises(ValueError, match="nonfinite"):
        summarize_pair(**values, costs=costs())


@pytest.mark.parametrize("bad_cost", [[0, 1], [-1, 1], [1], [1, float("inf")]])
def test_invalid_costs_rejected(bad_cost):
    with pytest.raises(ValueError, match="positive finite"):
        summarize_pair([[1], [1]], [[0], [0]], [[1], [1]], dict(hgr_steps=bad_cost, direct_steps=[1, 1]))


def test_missing_cost_key_rejected():
    with pytest.raises(ValueError, match="hgr_steps and direct_steps"):
        summarize_pair([[1], [1]], [[0], [0]], [[1], [1]], {})


def test_finite_inputs_whose_statistics_overflow_are_rejected():
    with pytest.raises(ValueError, match="overflowed"):
        summarize_pair([[1e308], [-1e308]], [[0], [0]], [[1], [1]], costs())


def test_inputs_are_not_mutated():
    old = np.arange(6., dtype=float).reshape(3, 2)
    delta = old / 100
    direct = old + 1
    before = [value.copy() for value in (old, delta, direct)]
    summarize_pair(old, delta, direct, costs(3))
    for value, expected in zip((old, delta, direct), before):
        np.testing.assert_array_equal(value, expected)
