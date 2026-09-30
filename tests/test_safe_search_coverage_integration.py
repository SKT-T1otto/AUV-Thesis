"""Runner integration for optional read-only coverage, without simulator steps."""
from pathlib import Path
import json
import tempfile
import unittest

from chapter3_bser.experiments.safe_search_v1 import run_paired as runner
from tests.test_safe_search_runner import FakeRuntime, fake_dependencies


class FakeCoverage:
    instances = []

    def __init__(self, runtime):
        self.runtime = runtime
        self.closed = False
        self.drained = 0
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False

    def drain(self):
        self.drained += 1
        return dict(step=self.runtime.step, new_observed_cells=2)

    def result(self):
        return dict(effective_observation_steps=self.drained)


class CoverageIntegrationTests(unittest.TestCase):
    def setUp(self):
        FakeCoverage.instances.clear()

    def test_opt_in_coverage_preserves_signatures_and_records_terminal_step(self):
        deps = fake_dependencies()
        deps['CoverageObserver'] = FakeCoverage
        with tempfile.TemporaryDirectory() as temporary:
            plain, _ = runner.execute_episode(lambda: FakeRuntime(end=2), Path(temporary),
                steps=3, observed=False, deps=deps)
            self.assertEqual(FakeCoverage.instances, [])
            observed, metrics = runner.execute_episode(lambda: FakeRuntime(end=2), Path(temporary),
                steps=3, observed=True, deps=deps)
            rows = [json.loads(line) for line in
                    (Path(temporary) / 'step_trace.jsonl').read_text().splitlines()]
        self.assertEqual(plain, observed)
        self.assertEqual([r['search_coverage']['step'] for r in rows], [1, 2])
        self.assertEqual(metrics['effective_search_steps'], 2)
        self.assertEqual(metrics['search_coverage'], dict(effective_observation_steps=2))
        self.assertTrue(FakeCoverage.instances[0].closed)

    def test_coverage_context_restores_on_observation_failure(self):
        deps = fake_dependencies()
        deps['CoverageObserver'] = FakeCoverage
        deps['capture_runtime'] = lambda runtime: (_ for _ in ()).throw(RuntimeError('read failed'))
        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(RuntimeError, 'read failed'):
            runner.execute_episode(lambda: runtime, Path(temporary), steps=3, observed=True, deps=deps)
        self.assertTrue(FakeCoverage.instances[0].closed)
        self.assertTrue(runtime.closed)

    def test_coverage_is_opt_in_and_unavailable_remains_null(self):
        args = runner.parser().parse_args(['--manifest', 'scene.json', '--output-dir', 'runs/new'])
        self.assertFalse(args.search_coverage)
        with tempfile.TemporaryDirectory() as temporary:
            _, metrics = runner.execute_episode(FakeRuntime, Path(temporary), steps=1,
                observed=True, deps=fake_dependencies())
        self.assertIsNone(metrics['search_coverage'])
        self.assertIsNone(metrics['effective_search_steps'])


if __name__ == '__main__':
    unittest.main()
