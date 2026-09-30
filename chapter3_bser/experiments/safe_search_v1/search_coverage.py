"""Read-only target-evidence footprint from the planner's actual update stamps.

The online-unknown planner writes ``target_last_observed_step`` for every valid
cell whose centre is within a searcher's target sensor radius.  An instance tap
observes those writes after the original negative-evidence update.  This is not
obstacle-map coverage, swept-volume coverage, or a target detection probability.
No hidden target/obstacle state is read and no value is returned to a controller.
"""
from __future__ import annotations

from collections import Counter
from contextlib import AbstractContextManager
import math


def _flat(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        return value.reshape(-1).tolist()
    result = []
    for item in value:
        result.extend(_flat(item)) if isinstance(item, (list, tuple)) else result.append(item)
    return result


class CoverageObserver(AbstractContextManager):
    """Serial per-runtime observer; attach after construction, drain each step.

    ``effective_observation_steps`` is a proxy: a physical pre-Found step with
    at least one first observation or a repeat whose previous observation is at
    least one configured target-belief revisit half-life old.  Continuous
    observation of the same cells counts as real repeated evidence, but not as
    new/aged evidence.  Initialization coverage is recorded separately and does
    not contribute physical steps.  Missing/inconsistent footprint data produces
    null metrics rather than an inferred footprint.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.env = runtime.env.unwrapped
        self.planner = self.env.map_module
        self.available = False
        self.errors = []
        self.counts = Counter()
        self._pending = {}
        self._last_drain_step = int(self.env.step_count)
        self._last_found = bool(self.env.task_found)
        self._entered = False
        self._wrapper = None
        self._original = None
        self._seen = set()
        self._last_seen = {}
        self.initial_observed_cell_indices = []
        self.grid_cell_count = None
        self.revisit_age_steps = None
        self._attribution_available = True
        self._attribution_seen = False
        self._attribution_errors = []
        self._agent_seen = {i: set() for i in range(3)}
        self._agent_counts = {i: Counter() for i in range(3)}
        self._overlap_counts = Counter()
        try:
            if not callable(getattr(self.planner, "update_belief_negative", None)):
                raise ValueError("planner has no target negative-evidence update")
            values = [int(v) for v in _flat(self.planner.target_last_observed_step)]
            if not values:
                raise ValueError("empty target-observation stamp grid")
            if len(_flat(self.planner.flat_valid_mask)) != len(values):
                raise ValueError("target stamp and valid-mask dimensions differ")
            self.grid_cell_count = len(values)
            self.revisit_age_steps = float(self.planner.target_revisit_half_life_steps)
            if not math.isfinite(self.revisit_age_steps) or self.revisit_age_steps <= 0:
                raise ValueError("invalid target revisit half-life")
            self._last_seen = {i: step for i, step in enumerate(values) if step >= 0}
            self._seen = set(self._last_seen)
            self.initial_observed_cell_indices = sorted(self._seen)
            self.available = True
        except Exception as exc:
            self._unavailable(exc)

    def _unavailable(self, error):
        self.available = False
        self.errors.append(type(error).__name__ + ": " + str(error))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("coverage observer is already active")
        self._entered = True
        if not self.available:
            return self
        self._original = self.planner.update_belief_negative
        if getattr(self._original, "_safe_search_coverage_tap", False):
            raise RuntimeError("target coverage tap is already active on this planner")
        self._had_instance_method = "update_belief_negative" in vars(self.planner)
        self._instance_method = vars(self.planner).get("update_belief_negative")
        owner = self

        def observed(*args, **kwargs):
            pre_found = not bool(owner.env.task_found)
            # Preserve the original arguments, object identity, exception and RNG.
            result = owner._original(*args, **kwargs)
            if pre_found and owner.available:
                try:
                    owner._capture_actual_update(args, kwargs)
                except Exception as exc:
                    # Instrumentation failure cannot change simulator behavior.
                    owner._unavailable(exc)
            return result

        observed._safe_search_coverage_tap = True
        self._wrapper = observed
        self.planner.update_belief_negative = observed
        return self

    def _capture_actual_update(self, args, kwargs):
        if not bool(self.planner.belief_enabled):
            return
        step = int(self.planner.runtime_step)
        if step != int(self.env.step_count):
            raise ValueError("target update stamp differs from current physical step")
        if step <= self._last_drain_step:
            raise ValueError("target update occurred outside the next physical transition")
        values = [int(v) for v in _flat(self.planner.target_last_observed_step)]
        if len(values) != self.grid_cell_count:
            raise ValueError("target observation grid changed shape")
        # Static re-observation is retained: equality to the current stamp does
        # not require a value change from the last snapshot or a new map revision.
        cells = {i for i, last in enumerate(values) if last == step}
        entry = self._pending.setdefault(step, {"cells": set(), "calls": 0,
                                                "agents": {i: set() for i in range(3)}})
        entry["cells"].update(cells)
        entry["calls"] += 1
        if self._attribution_available:
            try:
                footprints = self._agent_footprints(args, kwargs)
                combined = {i: entry["agents"][i] | footprints[i] for i in range(3)}
                if set().union(*combined.values()) != cells:
                    raise ValueError("per-agent reconstructed union differs from actual update stamps")
                entry["agents"] = combined
                self._attribution_seen = True
            except Exception as exc:
                self._attribution_available = False
                self._attribution_errors.append(type(exc).__name__ + ": " + str(exc))

    def _agent_footprints(self, args, kwargs):
        """Exact active-planner geometry, accepted only after stamped-union audit.

        This duplicates read-only tensor arithmetic, never a belief update.  The
        original method's timestamp union remains the authority.  Agent labels
        require the canonical ordered three-searcher positions in the call.
        """
        import torch
        planner = self.planner
        positions_value = args[0] if args else kwargs["search_positions"]
        ranges_value = args[1] if len(args) > 1 else kwargs.get("sensor_ranges")

        def points(value):
            tensor = value if torch.is_tensor(value) else torch.as_tensor(
                value, dtype=planner.dtype, device=planner.device)
            return tensor.to(device=planner.device, dtype=planner.dtype)

        positions = points(positions_value).reshape(-1, 3)
        canonical = self.env._agent_pos[:3]
        if positions.shape != canonical.shape or not torch.equal(positions, canonical):
            raise ValueError("target-update positions do not identify canonical searcher IDs")
        if ranges_value is None:
            ranges = torch.full((positions.shape[0],), 1.2 * min(planner.cell_dx, planner.cell_dy),
                                dtype=planner.dtype, device=planner.device)
        else:
            ranges = points(ranges_value).reshape(-1)
            if ranges.numel() == 1:
                ranges = ranges.repeat(positions.shape[0])
            elif ranges.numel() < positions.shape[0]:
                ranges = torch.cat([ranges, ranges[-1:].repeat(positions.shape[0] - ranges.numel())])
        footprints = {}
        for index, position in enumerate(positions):
            radius = float(max(ranges[index].item(), planner.eps))
            distance = torch.linalg.vector_norm(planner.flat_xyz_centers - position.unsqueeze(0), dim=1)
            visible = (distance <= radius) & planner.flat_valid_mask
            footprints[index] = set(torch.nonzero(visible, as_tuple=False).flatten().cpu().tolist())
        return footprints

    def drain(self):
        """Commit the just-finished transition, including a terminal transition."""
        step = int(self.env.step_count)
        delta = step - self._last_drain_step
        if delta < 0:
            self._unavailable(ValueError("runtime reset during coverage observation"))
            delta = 0
        exposure = delta if not self._last_found else 0
        if exposure and bool(self.env.task_found):
            found_step = getattr(self.env, "found_step", step)
            if found_step is not None:
                exposure = max(0, min(step, int(found_step)) - self._last_drain_step)
        self.counts["pre_found_exposure_steps"] += exposure
        batch = Counter()
        footprint = set()
        step_agents = None
        step_overlap = None
        if self.available:
            for update_step, entry in sorted(self._pending.items()):
                cells = entry["cells"]
                new = cells - self._seen
                repeated = cells & self._seen
                revisits = {i for i in repeated
                            if update_step - self._last_seen[i] >= self.revisit_age_steps}
                batch["observation_calls"] += entry["calls"]
                batch["target_update_call_steps"] += 1
                batch["observation_update_steps"] += bool(cells)
                batch["new_cell_observations"] += len(new)
                batch["repeated_cell_observations"] += len(repeated)
                batch["aged_revisit_cell_observations"] += len(revisits)
                batch["effective_observation_steps"] += bool(new or revisits)
                self._seen.update(cells)
                self._last_seen.update((i, update_step) for i in cells)
                footprint.update(cells)
                if self._attribution_available and self._attribution_seen:
                    counts = Counter(i for indices in entry["agents"].values() for i in indices)
                    agent_total = sum(counts.values())
                    overlap = dict(total_agent_cell_observations=agent_total,
                                   simultaneous_union_cell_observations=len(counts),
                                   redundant_agent_cell_observations=agent_total - len(counts),
                                   overlapped_cell_observations=sum(n > 1 for n in counts.values()))
                    self._overlap_counts.update(overlap)
                    self._overlap_counts["observation_steps"] += bool(cells)
                    for agent, indices in entry["agents"].items():
                        self._agent_seen[agent].update(indices)
                        self._agent_counts[agent]["cell_observations"] += len(indices)
                        self._agent_counts[agent]["observation_steps"] += bool(indices)
                    step_agents = {str(i): sorted(indices) for i, indices in entry["agents"].items()}
                    step_overlap = overlap
            self.counts.update(batch)
        self._pending.clear()
        self._last_drain_step = step
        self._last_found = bool(self.env.task_found)
        return dict(step=step, available=self.available,
                    pre_found_exposure_steps=exposure,
                    observed_cell_indices=sorted(footprint) if self.available else None,
                    unique_observed_cells=len(self._seen) if self.available else None,
                    per_agent_observed_cell_indices=step_agents,
                    inter_agent_overlap=step_overlap,
                    **{name: int(batch[name]) if self.available else None
                       for name in self._metric_names()})

    @staticmethod
    def _metric_names():
        return ("observation_calls", "target_update_call_steps", "observation_update_steps",
                "new_cell_observations", "repeated_cell_observations",
                "aged_revisit_cell_observations", "effective_observation_steps")

    def result(self):
        if self._pending or int(self.env.step_count) != self._last_drain_step:
            self.drain()
        attributed = self.available and self._attribution_available and self._attribution_seen
        overlap = dict(self._overlap_counts) if attributed else None
        if overlap is not None:
            total = overlap.get("total_agent_cell_observations", 0)
            overlap["redundant_agent_cell_fraction"] = (
                overlap.get("redundant_agent_cell_observations", 0) / total if total else None)
        return dict(
            schema="ch3.safe_search.target_evidence_coverage.v1", available=self.available,
            source="actual update_belief_negative target_last_observed_step writes",
            diagnostic_only=True, privileged_truth_read=False,
            initial_observed_cells=len(self.initial_observed_cell_indices) if self.available else None,
            initial_observed_cell_indices=self.initial_observed_cell_indices if self.available else None,
            unique_observed_cells=len(self._seen) if self.available else None,
            observed_cell_indices=sorted(self._seen) if self.available else None,
            grid_cell_count=self.grid_cell_count,
            unique_observed_grid_fraction=(len(self._seen) / self.grid_cell_count
                                           if self.available else None),
            revisit_age_steps=self.revisit_age_steps,
            pre_found_exposure_steps=self.counts["pre_found_exposure_steps"],
            per_agent_coverage=({str(i): dict(self._agent_counts[i],
                unique_observed_cells=len(self._agent_seen[i])) for i in range(3)} if attributed else None),
            inter_agent_overlap=overlap,
            attribution_definition=("canonical searcher positions and original sensor arguments; "
                "same online-unknown tensor geometry; published only if reconstructed union equals "
                "actual update stamps; initialization excluded from attribution"),
            attribution_errors=list(self._attribution_errors),
            effective_observation_steps_definition=("pre-Found physical step with at least one "
                "newly observed cell or repeat after the configured target revisit half-life; "
                "initialization excluded; this is an evidence-renewal proxy"),
            footprint_definition=("union of valid grid cell centres actually stamped by target "
                "negative-evidence updates; denominator is all grid cells, not physically "
                "reachable cells; no swept footprint, visibility/occlusion or detection probability inferred"),
            errors=list(self.errors),
            **{name: int(self.counts[name]) if self.available else None
               for name in self._metric_names()})

    def __exit__(self, *exception):
        if self._wrapper is not None:
            if self.planner.update_belief_negative is self._wrapper:
                if self._had_instance_method:
                    self.planner.update_belief_negative = self._instance_method
                else:
                    del self.planner.update_belief_negative
            else:
                self._unavailable(RuntimeError("another observer replaced the active target tap"))
        self._entered = False
        return False
