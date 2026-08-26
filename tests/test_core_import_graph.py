import ast
import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CoreImportGraphTests(unittest.TestCase):
    def test_all_absolute_core_imports_resolve(self):
        modules = set()
        for path in sorted((ROOT / "core").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names if alias.name.startswith("core"))
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("core"):
                    modules.add(node.module)
        failures = []
        for module in sorted(modules):
            try:
                importlib.import_module(module)
            except Exception as exc:  # pragma: no cover - reported as one actionable graph failure
                failures.append((module, repr(exc)))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
