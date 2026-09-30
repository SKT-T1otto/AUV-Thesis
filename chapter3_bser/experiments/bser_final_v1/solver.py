"""D2-referenced guards and deterministic bounded candidate selection."""
from collections import Counter
from itertools import product
from .options import SETTINGS


def guard(candidate, reference, arm):
    eps = SETTINGS["numerical_tolerance"]
    if arm["search_guard"] and candidate.full+eps < reference.full:
        return "full_search_regression"
    if arm["search_guard"] and candidate.near+eps < reference.near:
        return "near_search_regression"
    if arm["risk"]:
        if set(candidate.risks) != set(reference.risks):
            return "agent_set_changed"
        for i, risk in candidate.risks.items():
            # No extra known-obstacle exposure, and no cross-agent compensation.
            if any(a > b+eps for a, b in zip(risk, reference.risks[i])):
                return "risk_regression_agent_"+str(i)
    search_gain = (candidate.near-reference.near)/reference.near
    if arm["response"] == "sequential":
        return None if search_gain+eps >= SETTINGS["minimum_search_gain"] else "below_minimum_gain"
    if candidate.response+eps < reference.response:
        return "response_regression"
    response_gain = (candidate.response-reference.response)/reference.response
    if search_gain+eps < SETTINGS["minimum_search_gain"] and response_gain+eps < SETTINGS["minimum_response_gain"]:
        return "below_minimum_gain"
    return None


def rank(score, arm):
    return (score.near, score.full) if arm["response"] == "sequential" else (score.response, score.near, score.full)


def choose(groups, standbys, reference, evaluate, arm, *, seed_items=None):
    """evaluate returns (installable allocation, score), or a cheap rejection.

    F6 is one deterministic coordinate-greedy sweep per standby, seeded with
    the same-pool D2 search selection. Same model and final guards as F1.
    It does not import the old D3 objective or controller.
    """
    rejections, evaluated, seen = Counter(), 0, {}
    best = None

    def attempt(items, y):
        nonlocal best, evaluated
        key = (tuple(c.candidate_id for c in items), y.candidate_id)
        if key in seen:
            return seen[key]
        if evaluated >= SETTINGS["max_combinations"]:
            from .forecast import Incomparable
            raise Incomparable("combination_budget")
        evaluated += 1
        value, reason = evaluate(items, y)
        seen[key] = value
        if reason:
            rejections[reason] += 1
            return None
        allocation, score = value
        reason = guard(score, reference, arm)
        if reason:
            rejections[reason] += 1
        else:
            ranking = rank(score, arm)
            if best is None or ranking > best[0] or ranking == best[0] and key < best[1]:
                best = (ranking, key, allocation, score)
        return value

    if arm["solver"] == "enumerate":
        for items in product(*groups):
            for y in standbys:
                attempt(items, y)
    else:
        seed = tuple(g[0] for g in groups) if seed_items is None else tuple(seed_items)
        if len(seed) != len(groups) or any(c.key not in {v.key for v in group} for c, group in zip(seed, groups)):
            raise ValueError("greedy seed must belong to the same candidate groups")
        for y in standbys:
            current = seed
            for i, group in enumerate(groups):
                choices = []
                for candidate in group:
                    proposed = (*current[:i], candidate, *current[i+1:])
                    value = attempt(proposed, y)
                    if value is not None:
                        choices.append((rank(value[1], arm), tuple(c.key for c in proposed), proposed))
                if choices:
                    current = sorted(choices, key=lambda x: (tuple(-v for v in x[0]), x[1]))[0][2]
    return (None if best is None else (best[2], best[3])), dict(rejections), evaluated
