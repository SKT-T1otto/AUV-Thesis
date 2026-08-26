import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NoHistoricalRuntimeImportsTests(unittest.TestCase):
    def test_core_ast_has_no_historical_runtime_imports_or_path_injection(self):
        forbidden_modules = ("legacy_adapters", "ch3_snapshot")
        violations = []
        injections = []
        for path in sorted((ROOT / "core").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(forbidden_modules):
                            violations.append((path, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.startswith(forbidden_modules):
                        violations.append((path, module))
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"insert", "append"} and isinstance(node.func.value, ast.Attribute):
                        if node.func.value.attr == "path":
                            injections.append((path, node.lineno))
        self.assertEqual(violations, [])
        self.assertEqual(injections, [])


if __name__ == "__main__":
    unittest.main()
