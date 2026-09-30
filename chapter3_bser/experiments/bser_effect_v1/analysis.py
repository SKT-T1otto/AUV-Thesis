"""Predeclared paired summaries. Development evidence never establishes efficacy."""
import math
import random
from statistics import mean

COMPARISONS = (("D1", "D0"), ("D2", "D0"), ("D3", "D0"), ("D3", "D2"))


def mean_or_none(values):
    values = [x for x in values if x is not None]
    return mean(values) if values else None


def paired_interval(values, *, seed=2929, samples=5000):
    """Resample whole paired scenarios, never individual transitions or arms."""
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    return [draws[int(.025*(samples-1))], draws[int(.975*(samples-1))]]


def penalized(row):
    return row["found_step"] if row["found_step"] is not None else 400


def summarize(rows, plan):
    groups, by_arm = {}, {}
    for arm in plan["arms"]:
        values = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {r["original_episode_index"]: r for r in values}
        if len(by_arm[arm]) != len(values):
            raise ValueError("duplicate paired scenario")
        complete = len(values) == 20
        exposure = sum(r["pre_found_exposure_steps"] for r in values)
        found_times = [r["found_step"] for r in values if r["found_step"] is not None]
        effect = [r["controller"]["bser_effect_v1"] for r in values]
        groups[arm] = dict(n_recorded=len(values), n_expected=20, complete=complete,
            found_count=len(found_times), found_rate=len(found_times)/20 if complete else None,
            pre_found_collision_count=sum(r["pre_found_collision"] for r in values),
            pre_found_collision_rate=sum(r["pre_found_collision"] for r in values)/20 if complete else None,
            penalized_found_steps_mean_400=mean([penalized(r) for r in values]) if complete else None,
            found_steps_mean_conditional=mean_or_none(found_times),
            pre_found_exposure_steps=exposure,
            stall_plus_hold_fraction=sum(r["searcher_motion_stall_proxy_agent_steps"]+r["searcher_hold_agent_steps"]
                for r in values)/(3*exposure) if exposure else None,
            effective_observation_fraction=sum(r["effective_search_steps"] for r in values)/exposure if exposure else None,
            no_found_timeout_count=sum(r["found_step"] is None and r["stop_reason"] == "timeout" for r in values),
            success_count_secondary=sum(r["episode_result"]["success"] for r in values),
            physical_steps=sum(r["physical_steps"] for r in values),
            executor_pre_found_distance_mean=mean_or_none([v["executor_pre_found_distance"] for v in effect]),
            response_weight_search_change_proposals=sum(v["response_weight_search_change_proposals"] for v in effect),
            proposal_count=sum(v["proposal_count"] for v in effect))
    paired = {}
    for arm, reference in COMPARISONS:
        if arm not in by_arm or reference not in by_arm:
            continue
        indices = sorted(by_arm[arm].keys() & by_arm[reference].keys())
        pairs = []
        for index in indices:
            r, b = by_arm[arm][index], by_arm[reference][index]
            common_found = r["found_step"] is not None and b["found_step"] is not None
            pairs.append(dict(original_episode_index=index,
                found_delta=int(r["found_within_budget"])-int(b["found_within_budget"]),
                pre_found_collision_delta=int(r["pre_found_collision"])-int(b["pre_found_collision"]),
                penalized_found_steps_delta=penalized(r)-penalized(b),
                common_found=common_found,
                found_steps_delta_common_found=r["found_step"]-b["found_step"] if common_found else None))
        complete = len(pairs) == 20
        wins = sum(p["found_delta"] > 0 for p in pairs)
        losses = sum(p["found_delta"] < 0 for p in pairs)
        n = wins+losses
        stats = {}
        if complete:
            for key in ("found_delta", "pre_found_collision_delta", "penalized_found_steps_delta"):
                values = [p[key] for p in pairs]
                stats[key] = dict(mean=mean(values), paired_bootstrap_95_interval=paired_interval(values))
        response = {}
        for key in ("executor_distance_to_target_at_found", "executor_distance_at_handoff", "handoff_delay", "found_to_success_steps"):
            differences = []
            for i in indices:
                a, b = by_arm[arm][i]["complete_episode_row"], by_arm[reference][i]["complete_episode_row"]
                if a.get(key) is not None and b.get(key) is not None:
                    differences.append(a[key]-b[key])
            response[key] = dict(n_common_observed=len(differences), mean_delta=mean_or_none(differences))
        paired[arm+"_vs_"+reference] = dict(n_pairs=len(pairs), complete=complete,
            found_wins=wins, found_losses=losses, found_ties=len(pairs)-wins-losses,
            found_mcnemar_exact_two_sided=(min(1., 2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n))
                if n else 1.) if complete else None,
            statistics=stats, pairs=pairs, secondary_common_observed=response,
            found_steps_delta_common_found_mean=mean_or_none([p["found_steps_delta_common_found"] for p in pairs]))
    return dict(schema="ch3.bser_effect.summary.v1", groups=groups, paired=paired,
        experiment_complete=len(rows) == plan["planned_episode_runs"] and all(g["complete"] for g in groups.values()),
        primary_comparison="D3_vs_D0", joint_added_value_comparison="D3_vs_D2",
        training=False, checkpoint_loaded=False, performance_passed=None, formal_thesis_evaluation=False,
        caveats=["Same 20 reused development scenes; 80 runs are 20 paired scenario units, not 80 independent samples.",
            "Missing jobs/program errors do not count as physical failures; incomplete full-group rates are null.",
            "No Found is assigned 400 only for the comparison score, not as an observed detection time.",
            "Conditional Found/response/success metrics have outcome selection bias and are secondary.",
            "Bootstrap intervals and exact McNemar p values are exploratory; other comparisons are unadjusted.",
            "Surrogate objective increases, extra lifetime and shorter standby distance alone do not prove BSER efficacy.",
            "No gain on 20 scenes cannot prove absence of useful effect; confirmation requires a separate fixed evaluation."])
