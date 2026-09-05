"""Offline labels and scenario-cluster statistics. No runtime state dependencies."""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from . import HORIZON, MAX_STEPS, THRESHOLD


def window_label(t, found_step, observed_until, *, already_found=False,
                 horizon=HORIZON, max_steps=MAX_STEPS):
    """t is before the next action. An event at t is not a future discovery."""
    t, observed_until = int(t), int(observed_until)
    if not 0 <= t <= observed_until <= max_steps:
        raise ValueError("invalid state/observation/cutoff times")
    if found_step is not None and not 0 <= int(found_step) <= observed_until:
        raise ValueError("found step outside observed interval")
    main = max_steps - t >= horizon
    if already_found or (found_step is not None and found_step <= t):
        return dict(label=None, main_eligible=False, censored=False, window="already_found")
    event = found_step is not None and 0 < found_step - t <= horizon
    complete = observed_until >= t + horizon
    label = 1 if event else 0 if complete else None
    return dict(label=label, main_eligible=main, censored=label is None,
                window="full" if main else "administrative_short_tail")


def binary_metrics(labels, predictions):
    if not np.isin(np.asarray(labels), (0, 1)).all():
        raise ValueError("labels must be binary, not rounded to integers")
    y, p = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError("invalid binary metric inputs")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid prediction probabilities")
    n = len(y)
    reasons = {}
    def ratio(name, numerator, denominator):
        if not denominator:
            reasons[name] = "zero_denominator"
            return None
        return float(numerator / denominator)
    positive = p >= THRESHOLD
    tp, tn = int(np.sum(positive & (y == 1))), int(np.sum(~positive & (y == 0)))
    fp, fn = int(np.sum(positive & (y == 0))), int(np.sum(~positive & (y == 1)))
    precision = ratio("precision", tp, tp + fp)
    recall = ratio("recall", tp, tp + fn)
    specificity = ratio("specificity", tn, tn + fp)
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2
    if balanced is None:
        reasons["balanced_accuracy"] = "missing_positive_or_negative_class"
    ap = None
    if tp + fn and tn + fp:
        # Non-interpolated AP: sum over distinct descending score thresholds
        # of precision_at_threshold * increase_in_recall. Tied scores form a group.
        order = np.argsort(-p, kind="stable")
        scores, truth = p[order], y[order]
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), n - 1]
        hits = np.cumsum(truth)[ends]
        ap = float(np.sum((hits / (ends + 1)) * np.diff(np.r_[0, hits / y.sum()])))
    else:
        reasons["average_precision"] = "missing_positive_or_negative_class"
    return {
        "count": n, "positive_count": int(y.sum()),
        "positive_fraction": ratio("positive_fraction", y.sum(), n),
        "brier": ratio("brier", np.sum((p - y) ** 2), n),
        "average_precision": ap, "precision": precision, "recall": recall,
        "specificity": specificity, "balanced_accuracy": balanced,
        "accuracy": ratio("accuracy", tp + tn, n), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "probability_quantiles": dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                                          np.quantile(p, (0, .1, .25, .5, .75, .9, 1)).tolist())) if n else None,
        "probability_mean": float(p.mean()) if n else None,
        "probability_std": float(p.std()) if n else None, "undefined_reasons": reasons,
    }


def calibration(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    bins = []
    for index in range(10):
        mask = (p >= index / 10) & ((p < (index + 1) / 10) if index < 9 else (p <= 1))
        count = int(mask.sum())
        bins.append(dict(bin=index, lower=index / 10, upper=(index + 1) / 10, count=count,
                         mean_prediction=float(p[mask].mean()) if count else None,
                         observed_fraction=float(y[mask].mean()) if count else None,
                         reason=None if count else "empty_bin"))
    return bins


def scenario_key(row):
    return (str(row["scenario_id"]), int(row["scenario_seed"]))


def prediction_summary(rows, training_baseline, *, bootstrap_seed=61729, replicates=2000):
    valid = [r for r in rows if r["main_eligible"] and r["label"] is not None]
    prior = training_baseline.get("positive_fraction")
    predictors = {"head": lambda r: r["prediction"], "always_not_found": lambda r: 0.0}
    if prior is not None:
        predictors["training_constant"] = lambda r: prior
    groups = {"all": valid}
    groups.update({f"searcher_{i}": [r for r in valid if r["agent_id"] == i] for i in range(3)})
    report, bins = {}, []
    for group, records in groups.items():
        by_scenario = defaultdict(list)
        for r in records:
            by_scenario[scenario_key(r)].append(r)
        result = {"scenario_count": len(by_scenario), "unique_state_count": len({(*scenario_key(r), r["step"]) for r in records}),
                  "agent_state_count": len(records), "predictors": {}}
        for name, predictor in predictors.items():
            micro = binary_metrics([r["label"] for r in records], [predictor(r) for r in records])
            per = [dict(scenario_id=k[0], scenario_seed=k[1], **binary_metrics([r["label"] for r in v], [predictor(r) for r in v]))
                   for k, v in sorted(by_scenario.items())]
            equal, defined = {}, {}
            for metric in ("brier", "average_precision", "precision", "recall", "specificity", "balanced_accuracy", "accuracy", "positive_fraction"):
                values = [v[metric] for v in per if v[metric] is not None]
                equal[metric] = float(np.mean(values)) if values else None
                defined[metric] = len(values)
            result["predictors"][name] = dict(micro=micro, scenario_equal=equal, defined_scenarios=defined, per_scenario=per)
            bins.extend(dict(group=group, predictor=name, **b) for b in calibration([r["label"] for r in records], [predictor(r) for r in records]))
        report[group] = result
    ci = dict(estimate=None, lower=None, upper=None, seed=bootstrap_seed, replicates=replicates,
              unit="scenario; all time steps and all three searchers kept together", reason="training_prior_unavailable")
    if prior is not None:
        clusters = defaultdict(list)
        for r in valid:
            clusters[scenario_key(r)].append((r["prediction"] - r["label"])**2 - (prior - r["label"])**2)
        sums = np.array([sum(v) for _, v in sorted(clusters.items())])
        counts = np.array([len(v) for _, v in sorted(clusters.items())])
        if len(sums) >= 2:
            rng = np.random.default_rng(bootstrap_seed)
            draws = rng.integers(len(sums), size=(replicates, len(sums)))
            values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            lo, hi = np.quantile(values, (.025, .975))
            ci.update(estimate=float(sums.sum()/counts.sum()), lower=float(lo), upper=float(hi), reason=None)
        else:
            ci["reason"] = "fewer_than_two_scenarios"
    return dict(diagnostic_only=True, threshold=THRESHOLD, horizon=HORIZON, groups=report,
                total_scenarios=len({scenario_key(r) for r in rows}), total_unique_states=len({(*scenario_key(r), r["step"]) for r in rows}),
                total_agent_state_records=len(rows), censored_count=sum(r["censored"] for r in rows),
                short_tail_count=sum(not r["main_eligible"] for r in rows),
                short_tail_metrics=binary_metrics([r["label"] for r in rows if not r["main_eligible"] and r["label"] is not None],
                                                [r["prediction"] for r in rows if not r["main_eligible"] and r["label"] is not None]),
                constant_baseline_status="available" if prior is not None else "training_prior_unavailable",
                brier_minus_training_constant_cluster_bootstrap=ci,
                ap_definition="sum precision at each distinct descending probability threshold times recall increment; tied probabilities grouped; non-interpolated",
                scientific_conclusion="evidence_requires_interpretation; no accuracy performance gate"), bins


def paired_outcomes(rows):
    by_root = defaultdict(dict)
    for row in rows:
        if row["branch"] in ("A", "B", "C"):
            by_root[scenario_key(row)][row["branch"]] = row
    pairs, binary, continuous = [], {}, {}
    for metric in ("found_50", "found", "success"):
        counts = dict(both=0, A_only=0, B_only=0, neither=0, unavailable=0)
        for key, branches in sorted(by_root.items()):
            a, b = branches.get("A"), branches.get("B")
            valid = a and b and a.get("valid_pair") and b.get("valid_pair")
            av, bv = (a.get(metric), b.get(metric)) if valid else (None, None)
            outcome = "unavailable" if av is None or bv is None else "both" if av and bv else "A_only" if av else "B_only" if bv else "neither"
            counts[outcome] += 1
            pairs.append(dict(scenario_id=key[0], scenario_seed=key[1], metric=metric, A=av, B=bv,
                              outcome=outcome, tie=av == bv if av is not None and bv is not None else None,
                              B_treatment_delivered=b.get("treatment_delivered") if b else None))
        binary[metric] = counts
    for metric in ("searcher_collisions", "max_collision_streak", "known_ratio_gain", "travel_distance"):
        deltas = []
        for key, branches in sorted(by_root.items()):
            a, b = branches.get("A"), branches.get("B")
            if not (a and b and a.get("valid_pair") and b.get("valid_pair")):
                continue
            delta = float(b[metric] - a[metric])
            deltas.append(delta)
            pairs.append(dict(scenario_id=key[0], scenario_seed=key[1], metric=metric, A=a[metric], B=b[metric],
                              B_minus_A=delta, tie=delta == 0, B_treatment_delivered=b.get("treatment_delivered")))
        continuous[metric] = dict(count=len(deltas), mean=float(np.mean(deltas)) if deltas else None,
                                  quantiles=np.quantile(deltas, (0, .25, .5, .75, 1)).tolist() if deltas else None,
                                  ties=sum(v == 0 for v in deltas))
    posthoc = []
    for key, branches in sorted(by_root.items()):
        for metric in ("found_50", "found", "success"):
            observed = {b: r.get(metric) for b, r in branches.items() if r.get("valid_pair") and r.get(metric) is not None}
            best = max(observed.values()) if observed else None
            posthoc.append(dict(scenario_id=key[0], scenario_seed=key[1], metric=metric,
                                tested_values=observed, best_observed=best,
                                tied_best_branches=[b for b in sorted(observed) if observed[b] == best]))
    return pairs, dict(diagnostic_only=True, root_count=len(by_root), binary=binary, continuous=continuous,
                       posthoc_tested_branch_reference=posthoc,
                       population="historically intervention-selected development scenarios; not M20 population performance",
                       probability_ranking_consistency="not_applicable: per-agent predictions of shared team label are not a summed team allocation probability",
                       posthoc_best_tested_branch="diagnostic reference only; not an online oracle or theoretical upper bound")
