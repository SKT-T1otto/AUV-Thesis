import ast
import json
from pathlib import Path
import unittest

import numpy as np

from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.mapping.planning_state import extract_planning_state


class NoTruthLeakageTest(unittest.TestCase):
    def test_algorithm_ast_has_no_privileged_attributes(self):
        root = Path(__file__).resolve().parents[1]
        files = [root / "core/mapping/planning_state.py"] + list((root / "chapter3_bser").glob("*.py")) + list((root / "chapter3_bser/baselines").glob("*.py"))
        forbidden = {"_task_target", "target_state", "ground_truth_obstacles", "_truth_occupancy_mask", "target_future"}
        accesses = set()
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            accesses.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        self.assertFalse(accesses & forbidden)

    def test_runtime_guard_rejects_privileged_reads(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/scenarios/e0_equivalence/M10_MOVING_UNKNOWN_SINGLE.json").read_text(encoding="utf-8"))
        env = MissionCoreEnv(**environment_kwargs_from_config(build_ch3_config("ch3_v3_full_reference", "M10_MOVING_UNKNOWN_SINGLE"), device="cpu", max_steps=1))
        env.reset(scenario=manifest["scenarios"][0])
        blocked = {"_task_target", "target_state", "obstacles", "default_obstacles", "ground_truth_obstacles", "_truth_occupancy_mask"}

        class RuntimeGuard:
            def __init__(self, wrapped): object.__setattr__(self, "_wrapped", wrapped)
            def __getattr__(self, name):
                if name in blocked: raise AssertionError(f"privileged runtime read: {name}")
                return getattr(object.__getattribute__(self, "_wrapped"), name)

        class EnvGuard:
            def __init__(self, wrapped): self.wrapped = wrapped
            @property
            def unwrapped(self): return RuntimeGuard(self.wrapped.unwrapped)
            def get_task_state(self): return self.wrapped.get_task_state()
            def get_agent_state(self): return self.wrapped.get_agent_state()
            def get_scenario_identity(self): return self.wrapped.get_scenario_identity()

        runtime = env.unwrapped
        before = (runtime.step_count, runtime.map_module.map_revision, runtime._agent_pos.detach().cpu().numpy().tobytes(), runtime.map_module.belief_map.detach().cpu().numpy().tobytes())
        state = extract_planning_state(EnvGuard(env))
        after = (runtime.step_count, runtime.map_module.map_revision, runtime._agent_pos.detach().cpu().numpy().tobytes(), runtime.map_module.belief_map.detach().cpu().numpy().tobytes())
        self.assertEqual(before, after); self.assertEqual(state.occupancy.knowledge_mode, "online_unknown"); self.assertTrue(np.any(state.occupancy.unknown_mask))
        env.close()


if __name__ == "__main__": unittest.main()
