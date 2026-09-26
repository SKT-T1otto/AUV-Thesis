"""Small-budget diagnostics for ONE frozen HGR policy pair, without rollouts.

Rows are independent, identically designed macro repeats for the same theta,
phi0, phi1 and scenario target. Different natural policy updates must be passed
in separate calls. All Direct repeats must be independent of all HGR repeats;
g0 and delta, in contrast, are intentionally matched within a row.

These are noisy point estimates, not confidence intervals or signal bounds.
Negative noise-corrected energy and risk estimates are retained. This module
does not select policy pairs, bypass a correction, or decide whether to train.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SCHEMA = "hgr.decision.pair_statistics.v1"


def _vectors(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric (R, D) array") from exc
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape (R >= 2, D >= 1) for one frozen policy pair; "
            "do not pool different policy pairs"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _costs(value, repeats, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain positive finite repeat costs") from exc
    if array.shape != (repeats,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must have shape ({repeats},) and positive finite costs")
    return array


def _ratio(numerator, denominator):
    # Undefined ratios stay JSON null, never infinity and never a silent zero.
    return None if denominator == 0 else float(numerator / denominator)


def _moments(array):
    mean = array.mean(axis=0)
    centered = array - mean
    trace = float(np.einsum("ij,ij->", centered, centered) / (len(array) - 1))
    mean_square = float(np.einsum("ij,ij->", array, array) / len(array))
    return mean, dict(
        mean_norm=float(np.linalg.norm(mean)),
        trace_sample_variance=trace,
        mean_standard_error_scale=float(np.sqrt(trace / len(array))),
        rms_norm=float(np.sqrt(mean_square)),
        mean_squared_norm=mean_square,
    )


def summarize_pair(g0, delta, direct, costs):
    """Return JSON-safe statistics for a single frozen policy pair.

    ``g0``, ``delta``, and ``direct`` have shape (R, D), R >= 2. ``costs``
    supplies positive (R,) arrays ``hgr_steps`` and ``direct_steps``. HGR and
    Direct can use different fixed trajectory counts, but each row must target
    the same equally weighted scenario gradient. Costs are actual complete
    macro-repeat costs; shared source/training costs are accounted separately.

    Debiased signal energy is ||mean(delta)||^2 - tr(S_delta)/R. For the
    independent Direct mean d, U = ||d||^2 - tr(S_direct)/R estimates the
    unknown new-target squared norm. Thus ||X||^2 - 2 X.d + U estimates risk
    for X independent of Direct, even if X (such as g0) is biased. Direct's
    own risk uses its sample variance, not self-reference in that formula.
    These expectation statements require unbiased Direct gradients and the
    independence/identical-target assumptions; they do not establish them.
    """
    old = _vectors(g0, "g0")
    correction = _vectors(delta, "delta")
    new = _vectors(direct, "direct")
    if old.shape != correction.shape or old.shape != new.shape:
        raise ValueError("g0, delta, and direct must have identical (R, D) shapes")
    if not isinstance(costs, Mapping) or not {"hgr_steps", "direct_steps"} <= costs.keys():
        raise ValueError("costs must provide hgr_steps and direct_steps")
    repeats, dimension = old.shape
    hgr_cost = _costs(costs["hgr_steps"], repeats, "hgr_steps")
    direct_cost = _costs(costs["direct_steps"], repeats, "direct_steps")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            full = old + correction
            old_mean, old_stats = _moments(old)
            delta_mean, delta_stats = _moments(correction)
            full_mean, full_stats = _moments(full)
            direct_mean, direct_stats = _moments(new)
            signal_energy = float(delta_mean @ delta_mean - delta_stats["trace_sample_variance"] / repeats)
            direct_target_energy = float(direct_mean @ direct_mean - direct_stats["trace_sample_variance"] / repeats)

            risks = {}
            for name, mean, stats in (("g0", old_mean, old_stats), ("hgr", full_mean, full_stats)):
                risks[name] = dict(
                    single_macro_mse=float(stats["mean_squared_norm"] - 2 * mean @ direct_mean + direct_target_energy),
                    mean_of_repeats_mse=float(mean @ mean - 2 * mean @ direct_mean + direct_target_energy),
                )
            risks["direct"] = dict(
                single_macro_mse=direct_stats["trace_sample_variance"],
                mean_of_repeats_mse=direct_stats["trace_sample_variance"] / repeats,
            )
            # Compute the matched contrast directly: subtracting two large MSEs
            # can erase the tiny correction that this experiment is measuring.
            matched_rows = np.einsum("ij,ij->i", correction, 2 * old + correction - 2 * direct_mean)
            matched_mean = float(delta_mean @ (2 * old_mean + delta_mean - 2 * direct_mean))
            hgr_proxy = float(full_stats["trace_sample_variance"] * hgr_cost.mean())
            direct_proxy = float(direct_stats["trace_sample_variance"] * direct_cost.mean())

            result = dict(
                schema=SCHEMA,
                repeat_count=repeats,
                gradient_dimension=dimension,
                method_statistics=dict(g0=old_stats, hgr=full_stats, direct=direct_stats),
                signal=dict(
                    mean_delta_norm=delta_stats["mean_norm"],
                    rms_delta_norm=delta_stats["rms_norm"],
                    trace_sample_variance=delta_stats["trace_sample_variance"],
                    mean_standard_error_scale=delta_stats["mean_standard_error_scale"],
                    debiased_signal_energy=signal_energy,
                    g0_rms_norm=old_stats["rms_norm"],
                    relative_mean_delta_norm=_ratio(delta_stats["mean_norm"], old_stats["rms_norm"]),
                    relative_mean_standard_error_scale=_ratio(delta_stats["mean_standard_error_scale"], old_stats["rms_norm"]),
                    debiased_relative_signal_energy=_ratio(signal_energy, old_stats["mean_squared_norm"]),
                    signal_to_noise_ratio=_ratio(delta_stats["mean_norm"], delta_stats["mean_standard_error_scale"]),
                    zero_correction_squared_bias_estimate=signal_energy,
                    correction_mean_noise_energy_estimate=delta_stats["trace_sample_variance"] / repeats,
                    uncertainty_kind="estimated vector mean-square standard-error scale; not a CI or upper bound",
                ),
                risk_estimates=dict(
                    methods=risks,
                    direct_target_squared_norm_u=direct_target_energy,
                    matched_hgr_minus_g0_single_macro_mse=float(matched_rows.mean()),
                    matched_hgr_minus_g0_mean_of_repeats_mse=matched_mean,
                    per_repeat_hgr_minus_g0_risk=matched_rows.tolist(),
                    negative_estimates_retained=True,
                    interpretation="negative matched contrast favors HGR over its own g0; noisy point estimate only",
                    dependence="matched rows share the Direct mean and must not be treated as independent CI samples",
                ),
                cost_proxy=dict(
                    definition="trace_sample_variance * actual_mean_macro_steps",
                    interpretation="asymptotic variance-cost proxy for unbiased estimators, not an actual equal-cost comparison",
                    actual_equal_budget_comparison=False,
                    includes_source_or_suffix_training_cost=False,
                    hgr=hgr_proxy,
                    direct=direct_proxy,
                    hgr_over_direct=_ratio(hgr_proxy, direct_proxy),
                    hgr_mean_steps=float(hgr_cost.mean()),
                    direct_mean_steps=float(direct_cost.mean()),
                    hgr_total_steps=float(hgr_cost.sum()),
                    direct_total_steps=float(direct_cost.sum()),
                    limitation="random cost and gradient may be dependent; small-sample products are descriptive",
                ),
                assumptions=[
                    "one frozen theta/phi0/phi1 pair and identical scenario target within this call",
                    "independent, identically designed macro repeats within each method",
                    "every Direct repeat independent of every HGR repeat; g0 and delta intentionally matched",
                    "Direct estimates the new gradient without bias; variance-cost interpretation also assumes unbiased HGR",
                    "finite second moments; parameter dimensions are not independent experimental replicates",
                    "no hypothesis test, confidence interval, finite-sample signal upper bound, or success claim",
                ],
            )
        except FloatingPointError as exc:
            raise ValueError("statistics overflowed; input magnitudes cannot be summarized safely") from exc

    # Python scalar operations can overflow without NumPy raising. Never emit
    # non-standard JSON Infinity/NaN or turn such failures into decision inputs.
    def check_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, list):
            for item in value:
                check_finite(item)
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError("statistics overflowed; a result is nonfinite")

    check_finite(result)
    return result
