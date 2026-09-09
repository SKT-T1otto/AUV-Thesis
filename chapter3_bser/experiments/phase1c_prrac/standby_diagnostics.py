"""Read-only transition diagnostics; never supplies values to control."""
import numpy as np

from chapter3_bser.online.safe_executor_standby import min_searcher_distance


STANDBY_FIELDS = """scenario_id episode step transition_step executor_position executor_velocity executor_speed
standby_target raw_standby_target safe_standby_target target_update_event target_update_reason target_shift
route_available route_length route_point_count tracking_target standby_distance tracking_target_distance
executor_action executor_action_norm slowdown_active hold_active min_distance_to_searchers
three_searcher_mean_distance executor_collision_event executor_pre_found_collision_count
executor_post_found_collision_count searcher_collision_flags searcher_collision_count_by_agent
c2_active_if_available target_found target_found_after standby_active weights_source residual_factor
update_interval target_shift_threshold path_tracking_threshold safe_dist prior_slow_radius_xy
prior_slow_radius_z hold_radius""".split()
FOUND_FIELDS = """scenario_id episode step executor_distance_to_target_at_found executor_speed_at_found
executor_velocity_at_found executor_action_norm_at_found standby_distance_at_found
min_executor_searcher_distance_at_found standby_active_before_found""".split()


def vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


class StandbyDiagnostics:
    def __init__(self, scenario_id, episode, enabled):
        self.identity = dict(scenario_id=str(scenario_id), episode=int(episode))
        self.enabled = bool(enabled)
        self.rows, self.found_rows = [], []
        self.pre_count = self.post_count = 0
        self.searcher_counts = [0, 0, 0]
        self.pre_complete = self.post_complete = self.searcher_complete = True
        self.pending = None

    def before_action(self, state, actions, navigation, c2_active=None):
        if not self.enabled:
            return
        agents = {a.agent_id: a for a in state.agents}
        executor = agents[state.executor_id]
        position, velocity = vector(executor.position), vector(executor.velocity)
        action = vector(actions[state.executor_id])
        details = dict(navigation.details) if navigation is not None else {}
        row = dict(self.identity, step=int(state.step), executor_position=position.tolist(),
                   executor_velocity=velocity.tolist(), executor_speed=float(np.linalg.norm(velocity)),
                   executor_action=action.tolist(), executor_action_norm=float(np.linalg.norm(action)),
                   min_distance_to_searchers=min_searcher_distance(position, state),
                   three_searcher_mean_distance=float(np.mean([np.linalg.norm(position-agents[i].position) for i in state.searcher_ids])),
                   c2_active_if_available=c2_active, target_found=bool(state.target_found),
                   standby_active=bool(navigation is not None and navigation.active and not state.target_found),
                   weights_source="uniform_no_audited_searcher_values", **{k: v for k, v in details.items() if k != "standby_active"})
        row["standby_target"] = row.get("safe_standby_target")
        self.pending = row
        self.rows.append(row)

    def observe_transition(self, before, after, actions, collision_flags, distance_at_found=None):
        if not self.enabled:
            return
        # Each true per-agent flag is one collision-bearing physical transition,
        # not a new collision onset. The first Found transition belongs to Search.
        flags = None if collision_flags is None else tuple(bool(v) for v in collision_flags)
        if flags is None:
            self.searcher_complete = False
            if before.target_found:
                self.post_complete = False
            else:
                self.pre_complete = False
        if flags is not None:
            if len(flags) != len(after.agents):
                raise ValueError("per-agent collision flags must cover all agents")
            if flags[after.executor_id]:
                if before.target_found:
                    self.post_count += 1
                else:
                    self.pre_count += 1
            for index, agent_id in enumerate(after.searcher_ids):
                self.searcher_counts[index] += int(flags[agent_id])
        if self.pending is not None:
            self.pending.update(transition_step=int(after.step), target_found_after=bool(after.target_found),
                                executor_collision_event=None if flags is None else flags[after.executor_id],
                                executor_pre_found_collision_count=self.pre_count if self.pre_complete else None,
                                executor_post_found_collision_count=self.post_count if self.post_complete else None,
                                searcher_collision_flags=None if flags is None else [flags[i] for i in after.searcher_ids],
                                searcher_collision_count_by_agent=list(self.searcher_counts) if self.searcher_complete else None)
        if not before.target_found and after.target_found and not self.found_rows:
            executor = {a.agent_id: a for a in after.agents}[after.executor_id]
            velocity = vector(executor.velocity)
            safe_target = None if self.pending is None else self.pending.get("safe_standby_target")
            self.found_rows.append(dict(self.identity, step=int(after.step),
                executor_distance_to_target_at_found=distance_at_found,
                executor_velocity_at_found=velocity.tolist(), executor_speed_at_found=float(np.linalg.norm(velocity)),
                executor_action_norm_at_found=float(np.linalg.norm(vector(actions[after.executor_id]))),
                standby_distance_at_found=None if safe_target is None else float(np.linalg.norm(vector(executor.position)-safe_target)),
                min_executor_searcher_distance_at_found=min_searcher_distance(executor.position, after),
                standby_active_before_found=None if self.pending is None else self.pending["standby_active"]))
        self.pending = None
