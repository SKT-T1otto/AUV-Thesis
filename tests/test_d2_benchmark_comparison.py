import copy
import unittest

from tools.compare_d2_performance import compare


class BenchmarkComparisonTests(unittest.TestCase):
    def fixture(self):
        return dict(schema="ch3.d2.performance.benchmark.v1", formal_experiment=False,
                    arm="B2", seed=2729, threads=1, device="cpu", manifest_sha256="manifest", torch="torch",
                    global_step=400, optimizer_updates=74, cases=[dict(name="training_episode", episode=1,
                        steps=400, wall_seconds=10, rewards=[[.1] * 4], transition_sha256="trace", model_sha256="model",
                        summary=dict(found=False, terminal_step=400, wall_seconds=9,
                                     planner=dict(safe_counts=dict(forced_planning_refreshes=5, safety_route_queries=8))),
                        timings=dict(H_route_query=dict(calls=12, seconds=3)))])

    def test_timing_and_source_changes_are_allowed(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b.update(source={"sha256": "different"}, cache_diagnostics={"hits": 2})
        b["cases"][0]["wall_seconds"] = 5
        b["cases"][0]["summary"]["wall_seconds"] = 4
        b["cases"][0]["timings"]["H_route_query"]["seconds"] = 1
        self.assertTrue(compare(a, b)["exact_passed"])
        self.assertEqual(compare(a, b)["training_episode"]["speedup"], 2)

    def test_model_reward_transition_safety_and_query_changes_fail(self):
        a = self.fixture()
        for field in ("model_sha256", "transition_sha256", "rewards", "summary", "timings"):
            b = copy.deepcopy(a)
            row = b["cases"][0]
            if field == "summary":
                row[field]["planner"]["safe_counts"]["forced_planning_refreshes"] += 1
            elif field == "timings":
                row[field]["H_route_query"]["calls"] += 1
            else:
                row[field] = "changed"
            with self.subTest(field=field):
                self.assertFalse(compare(a, b)["exact_passed"])

    def test_missing_signatures_and_duplicate_cases_fail_even_on_both_sides(self):
        a = self.fixture()
        del a["cases"][0]["model_sha256"]
        self.assertFalse(compare(a, a)["exact_passed"])
        a = self.fixture()
        a["cases"].append(copy.deepcopy(a["cases"][0]))
        self.assertFalse(compare(a, a)["exact_passed"])
        a = self.fixture()
        a["cases"] = []
        self.assertFalse(compare(a, a)["exact_passed"])
