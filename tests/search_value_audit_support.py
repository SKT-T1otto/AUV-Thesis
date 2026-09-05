"""Small synthetic fixtures only; never substitutes for trained Linux evidence."""

import copy
from pathlib import Path
import torch
from tests.prrac_evaluation_support import checkpoint_payload
from tests.test_search_value_guided_ranking import scorer
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.experiments.phase1c_prrac.search_value_audit.runner import make_job, NATIVE


def synthetic_checkpoint():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61729)
        payload = checkpoint_payload()
        head = scorer().head
    payload["metadata"].update(execution_runtime_revision=NATIVE, execution_variant="B1_ATOMIC_LAST_VALID", runtime_integration_mode="native")
    payload["prrac_training_state"]["search_value"] = dict(enabled=True, hidden_dim=8, horizon=50, threshold=.5, loss_weight=.05)
    payload["prrac_training_state"]["search_value_head"] = copy.deepcopy(head.state_dict())
    return payload


def scene(seed=100, identifier="synthetic_audit"):
    return dict(scenario_id=identifier, scenario_seed=seed, scenario_profile="M20_MOVING_UNKNOWN_MULTI",
                max_steps=400, initial_agent_positions=[[2, 2, 1], [2, 6, 1], [6, 2, 1], [6, 6, 1]],
                initial_executor_wait_point=[6, 6, 1], target_position=[17, 17, 2], target_initial_position=[17, 17, 2],
                target_initial_velocity=[.02, .01, 0], target_motion_mode="constant_velocity_reflect_v1",
                target_motion_known=True, target_motion_seed=seed, use_obstacles=False, obstacles=[],
                obstacle_knowledge_mode="online_unknown", planner_grid_size=[6, 6, 2],
                planner_seed=seed, flow_phase_x=0., flow_phase_y=0.)


def small_job(*, enabled=False, max_steps=2):
    payload = synthetic_checkpoint()
    state = payload["prrac_training_state"]
    model = dict(architecture=payload["metadata"]["architecture"], loss=payload["metadata"]["loss"],
                 gamma=state["gamma"], tau=state["tau"], reward=payload["metadata"]["reward"],
                 policy_snapshot=tuple({k: v.numpy().copy() for k, v in a["actor"].items()} for a in state["agents"]),
                 search_value_snapshot={k: v.numpy().copy() for k, v in state["search_value_head"].items()},
                 search_value_config=state["search_value"])
    config = evaluator._load_config(evaluator.ROOT/"configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json")
    config["max_steps"] = max_steps  # Unit fixture only; production CLI never exposes this.
    config["search_value_guidance"] = dict(enabled=enabled, weight=.1, clip_min=0., clip_max=1.)
    return make_job(model, dict(checkpoint_path="synthetic.pt"), payload, config, scene(), 0, {})


def arithmetic_worker(item):
    return dict(unit=item["unit"], value=item["value"] ** 2)


class TinyAuditEnv:
    """Test-only deterministic physics. Production uses only the original evaluator."""
    def __init__(self, state):
        from types import SimpleNamespace
        import numpy as np
        self._env = SimpleNamespace(step=0, positions=np.array([a.position for a in state.agents]),
                                    collision_flags=np.zeros(4, dtype=bool), hidden_target=np.array([2.7, .5, 1.]))
        self.installed_context = None

    @property
    def unwrapped(self):
        return self._env

    def get_task_state(self):
        from types import SimpleNamespace
        return SimpleNamespace(step=self._env.step)


def synthetic_branch(branch="A", root=0, *, cutoff=4, third=True):
    """Real controller/allocator/bridge/head over a nine-cell synthetic plant.

    A predeclared finite candidate fixture replaces generation ONLY in this test.
    The production observer, scoring, acceptance, prefix and OFF switch are real.
    """
    import numpy as np
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch
    from tests.test_search_value_guided_ranking import fixture
    from chapter3_bser.online.controller import OnlineBSERController
    from chapter3_bser.online.config import load_phase1b2_config
    from chapter3_bser.online.mission_context import OnlineMissionContext
    from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
    from chapter3_bser.experiments.phase1c_prrac.search_value_guidance import SearchValueGuidedBSERAllocator
    from chapter3_bser.experiments.phase1c_prrac.search_value_audit.runtime_audit import RuntimeAudit, CUnavailable
    from chapter3_bser.experiments.phase1c_prrac.search_value_audit.provenance import digest
    from chapter3_bser.experiments.phase1c_prrac.search_value_audit.state_fingerprint import component_fingerprint, rng_state
    from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
    from core.mapping.planning_graph import EndpointConnectorSet, PlanningConnectorView
    torch.set_num_threads(1)
    evaluator._seed_all(123)
    job = small_job(enabled=True)
    base_state, candidates, standby, objective = fixture()
    standby = replace(standby, planning_cost=1., physical_travel_time=1.)
    objective = replace(objective, standby_candidates=(standby,))
    # Complete legal assignments for all three searchers prevent missing-agent
    # repair events from pre-empting the intended synthetic periodic boundary.
    companions = tuple(replace(candidates[0], agent_id=agent_id, candidate_id=f"agent{agent_id}", waypoint=target,
                               path_points=np.array([base_state.agents[agent_id].position, target]))
                       for agent_id, target in ((1, (.5, 2.5, 1.)), (2, (2.5, .5, 1.))))
    candidates = (*candidates, *companions)
    objective = replace(objective, candidates=candidates, detection_by_id={**objective.detection_by_id, "agent1": np.array([.03]), "agent2": np.array([.02])})
    if third:
        c = replace(candidates[0], candidate_id="c", waypoint=(2.5, 1.5, 1.), path_points=np.array([base_state.agents[0].position, [2.5, 1.5, 1.]]))
        candidates = (*candidates, c)
        objective = replace(objective, candidates=candidates, detection_by_id={**objective.detection_by_id, "c": np.array([.18])})
    value = scorer()
    controller = OnlineBSERController(load_phase1b2_config(), allocator=SearchValueGuidedBSERAllocator(value))
    actor = PRRACMADDPG(architecture=job["architecture"], loss=job["loss"])
    actor.load_policy_snapshot(job["policy_snapshot"])
    actor.prep_rollouts("cpu")
    env = TinyAuditEnv(base_state)
    provider = SimpleNamespace(env=env, cached=base_state, _last_full_refresh_step=0)
    audit = RuntimeAudit(job, intervention=branch, root_step=root)
    audit.bind(actor=actor, env=env, provider=provider, controller=controller, scorer=value)
    bridge = None
    observations = np.zeros((4, 28), np.float32)
    state = base_state
    status = "completed"
    def context_builder(state, current_candidates, current_standbys, config):
        return replace(objective, state=state, candidates=tuple(current_candidates), standby_candidates=tuple(current_standbys),
                       response_weight_by_id={s.candidate_id: np.array([1.]) for s in current_standbys},
                       response_time_by_id={s.candidate_id: np.array([1.]) for s in current_standbys})
    generated = SimpleNamespace(search_candidates=candidates, standby_candidates=(standby,))
    with patch("chapter3_bser.online.allocator.generate_candidates", return_value=generated), \
         patch("chapter3_bser.online.allocator.build_objective_context", side_effect=context_builder), \
         patch("chapter3_bser.online.allocator.BSEROnlineAllocator._partial_search_candidates", side_effect=lambda state, affected: tuple(c for c in candidates if c.agent_id in affected)):
        try:
            for t in range(cutoff+1):
                agents = tuple(replace(a, position=tuple(env.unwrapped.positions[a.agent_id]),
                                       current_navigation_target=(env.installed_context.assignment_for(a.agent_id).tracking_waypoint if env.installed_context else a.current_navigation_target)) for a in base_state.agents)
                endpoints = []
                for a in agents:
                    targets = [a.position, a.current_navigation_target, *(c.waypoint for c in candidates if c.agent_id == a.agent_id)]
                    if a.agent_id == 3:
                        targets.append(standby.waypoint)
                    for i, target in enumerate(targets):
                        cell = int(np.argmin(np.linalg.norm(base_state.grid.cell_centers-np.array(target), axis=1)))
                        distance = float(np.linalg.norm(base_state.grid.cell_centers[cell]-np.array(target)))
                        endpoints.append(EndpointConnectorSet(f"{a.agent_id}_{i}", a.role, tuple(target), (PlanningConnectorView(cell, 0, distance, distance),)))
                state = replace(base_state, step=t, agents=agents, planning_graph=replace(base_state.planning_graph, endpoint_connectors=tuple(endpoints)))
                env.unwrapped.step = t
                observations[:, :3] = env.unwrapped.positions
                context = OnlineMissionContext.from_planning_view(state)
                if t:
                    audit.transition(state, observations, None)
                value.observe_state(observations, state, collision_flags=env.unwrapped.collision_flags if t else None)
                audit.before_decision(state, observations, context, initialize=t == 0)
                result = controller.initialize(state, context) if t == 0 else controller.step(state, context)
                if t == root and t:
                    root_reason = (result.decision_reason, [e.value for e in result.events])
                value.record_installed(result.allocation, accepted=t == 0 or result.replanned)
                audit.after_decision(result, initialize=t == 0)
                if bridge is None:
                    bridge = RMADDPGGuidanceBridge()
                    audit.attach_guidance_runtime(bridge, SimpleNamespace(failure_memory=[], mode="normal"), None)
                guidance = bridge.compile_guidance(result.allocation, state, context)
                env.installed_context = guidance
                for a in base_state.agents:
                    delta = np.array(guidance.assignment_for(a.agent_id).tracking_waypoint)-env.unwrapped.positions[a.agent_id]
                    observations[a.agent_id, 6:9] = delta
                audit.after_install(state, observations, guidance, guidance)
                if t == cutoff:
                    break
                audit.before_action(state, observations, guidance)
                with torch.no_grad():
                    outputs = actor.step(torch.tensor(observations), explore=False)
                actions = (torch.stack(outputs) if isinstance(outputs, (list, tuple)) else outputs).reshape(4, 3)
                audit.action(state, actions)
                env.unwrapped.positions += .001*actions.numpy() + .0001*observations[:, 6:9]
        except CUnavailable:
            status = "C_UNAVAILABLE"
    audit.finish(dict(episode=dict(found=False, contact_episode=False, success=False, episode_length=env.unwrapped.step,
                                   reward=0., collision_episode=False)) if status == "completed" else None)
    return dict(status=status, audit=audit.export(), root_reason=locals().get("root_reason"))
