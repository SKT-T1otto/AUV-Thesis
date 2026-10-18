"""Matched-detector scores over common public-model trajectory hypotheses.

No environment, scenario, target truth, checkpoint or simulator snapshot input.
The velocity lift is an explicit approximation, not a filtered velocity posterior.
"""
from itertools import product
import hashlib

import numpy as np

from core.env.target_motion import EPS, TargetState, simulate_target_trajectory

METHODS = ('static', 'arrival_discount', 'time_marginal', 'correlated_first')


def swept_events(agent_positions, target_positions, radius):
    """Vectorized exact algebra of the shared swept_relative_min_distance.

    Targets have shape (M,H+1,3); output is (H,M). EPS and threshold match core.
    Contract equivalence is checked against the shared scalar function in tests.
    """
    a = np.asarray(agent_positions, dtype=float)
    x = np.asarray(target_positions, dtype=float)
    if x.ndim != 3 or x.shape[1:] != a.shape or a.shape[1] != 3 or radius <= 0:
        raise ValueError('incompatible trajectory shapes or radius')
    r0 = a[None, :-1] - x[:, :-1]
    delta = np.diff(a, axis=0)[None] - np.diff(x, axis=1)
    denominator = np.sum(delta * delta, axis=2)
    fraction = np.zeros_like(denominator)
    np.divide(-np.sum(r0 * delta, axis=2), denominator, out=fraction, where=denominator > EPS)
    fraction = np.clip(fraction, 0., 1.)
    distance = np.linalg.norm(r0 + fraction[:, :, None] * delta, axis=2)
    return (distance <= radius).T


def first_values(events, dt, discount):
    e = np.asarray(events, dtype=bool)
    hit = np.any(e, axis=0)
    first = np.argmax(e, axis=0) + 1
    return np.where(hit, np.exp(-discount * dt * first), 0.)


def marginal_utility(events, dt, discount):
    q = np.mean(events, axis=1)
    survival_before = np.r_[1., np.cumprod(1. - q)[:-1]]
    value = float(np.sum(np.exp(-discount * dt * np.arange(1, len(q)+1)) * q * survival_before))
    return value, float(1. - np.prod(1. - q))


def known_target_boxes(state):
    # Raw occupied voxel bounds, not the Searcher clearance expansion.
    half = np.asarray(state.grid.spacing) / 2
    return [dict(center=p.tolist(), size=(2*half).tolist())
            for p in np.asarray(state.grid.cell_centers)[state.occupancy.occupied_mask]]


def hypotheses(state, settings, public, seed):
    rng = np.random.default_rng(seed)
    support = np.asarray(state.grid.cell_centers)
    mass = np.asarray(state.target_belief.probabilities, dtype=float).copy()
    boxes = known_target_boxes(state)
    allowed = ~np.asarray(state.occupancy.occupied_mask)
    for box in boxes:
        center, half = np.asarray(box['center']), np.asarray(box['size'])/2 + public['target_clearance']
        allowed &= ~np.all(np.abs(support-center) < half, axis=1)
    for agent in state.agents[:3]:
        allowed &= np.linalg.norm(support - agent.position, axis=1) > agent.sensor_radius + public['detect_eps_bias']
    retained_mass = float(np.sum(mass[allowed]))
    if retained_mass <= 1e-12:
        raise ValueError('no mass after public current-nondetection conditioning')
    mass *= allowed; mass /= retained_mass
    n, h = settings['particles'], min(settings['horizon_steps'], public['remaining_steps'])
    indices = rng.choice(len(mass), size=n, p=mass)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    speed = rng.uniform(*settings['target_speed_prior'], size=n)
    result = []
    for index, velocity in zip(indices, direction * speed[:, None]):
        initial = TargetState(support[index], velocity, int(state.step), 'constant_velocity_reflect_v1')
        sequence = simulate_target_trajectory(initial, h, public['dt'], public['bounds'], boxes,
            clearance=public['target_clearance'], max_reflections=public['max_reflections'])
        result.append([item.position for item in sequence])
    result = np.asarray(result)
    return result, dict(seed=int(seed), particles=n, horizon=h, retained_current_mass=retained_mass,
        trajectories_sha256=hashlib.sha256(result.tobytes()).hexdigest(), known_voxel_boxes=len(boxes),
        posterior_claim=False)


def route_scores(moving_events, static_events, arrival, dt, discount):
    stationary = np.any(static_events, axis=0).astype(float)
    first = first_values(moving_events, dt, discount)
    marginal, marginal_p = marginal_utility(moving_events, dt, discount)
    return dict(static=float(stationary.mean()),
        arrival_discount=float(stationary.mean()*np.exp(-discount*arrival)),
        time_marginal=marginal, correlated_first=float(first.mean()),
        dynamic_hit_probability=float(np.mean(np.any(moving_events, axis=0))),
        marginal_hit_probability=marginal_p,
        stationary_first=float(np.mean(first_values(static_events, dt, discount))))


def team_scores(moving, stationary, arrivals, dt, discount):
    union = np.logical_or.reduce(moving)
    static_hit = np.array([np.any(s, axis=0) for s in stationary])
    discounted = static_hit * np.exp(-discount*np.asarray(arrivals))[:, None]
    marginal, marginal_p = marginal_utility(union, dt, discount)
    return dict(static=float(np.any(static_hit, axis=0).mean()),
        arrival_discount=float(np.max(discounted, axis=0).mean()),
        time_marginal=marginal,
        correlated_first=float(first_values(union, dt, discount).mean()),
        dynamic_hit_probability=float(np.mean(np.any(union, axis=0))),
        marginal_hit_probability=marginal_p)


def choose(rows, field):
    # Numerical ties retain the shortest-query candidate (index zero).
    best = max(row[field] for row in rows)
    return next(i for i, row in enumerate(rows) if row[field] >= best - 1e-12)


def compare_case(groups, selection_events, selection_static, validation_events,
                 validation_static, dt, settings, seed):
    """Enumerate a fixed team pool; select once, evaluate on independent draws."""
    combinations = list(product(*(range(len(group)) for group in groups)))
    scores, validation, values = [], [], []
    for combo in combinations:
        arrivals = [groups[i][j].physical_time for i,j in enumerate(combo)]
        args = ([selection_events[i][j] for i,j in enumerate(combo)],
                [selection_static[i][j] for i,j in enumerate(combo)], arrivals, dt, settings['discount_per_second'])
        scores.append(team_scores(*args))
        union = np.logical_or.reduce([validation_events[i][j] for i,j in enumerate(combo)])
        values.append(first_values(union, dt, settings['discount_per_second']))
        validation.append(team_scores([validation_events[i][j] for i,j in enumerate(combo)],
            [validation_static[i][j] for i,j in enumerate(combo)], arrivals, dt, settings['discount_per_second']))
    choices = {method: choose(scores, method) for method in METHODS}
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, settings['particles'], (settings['bootstrap_repetitions'], settings['particles']))
    selected_first = values[choices['correlated_first']]
    comparisons = {}
    for method in METHODS[:-1]:
        delta = selected_first - values[choices[method]]
        comparisons[method] = dict(validation_delta=float(delta.mean()),
            particle_bootstrap95=np.quantile(delta[sample].mean(axis=1), [.025,.975]).tolist(),
            changed=combinations[choices[method]] != combinations[choices['correlated_first']])
    return dict(combinations=len(combinations), choices={m:list(combinations[j]) for m,j in choices.items()},
        selected_scores={m:scores[j] for m,j in choices.items()},
        validation_scores={m:validation[j] for m,j in choices.items()}, comparisons=comparisons,
        reference_validation=validation[0],
        all_scores=[dict(routes=list(c),selection=s,validation=v) for c,s,v in zip(combinations,scores,validation)])
