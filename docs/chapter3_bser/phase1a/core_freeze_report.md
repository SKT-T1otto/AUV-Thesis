# Core freeze report

The preflight manifest recorded all 40 Python files that existed under `core/`
at base commit `cf9b6e1de8a2ddf5bd992ff6dad20f8de13971a5`, including byte size, SHA-256 and
an attribute-inclusive AST dump SHA-256. The final verifier recomputed all four
fields.

- Existing core Python files: 40.
- Existing core Python files changed: 0.
- New core Python files: 2 (`planning_state.py`, `travel_cost_service.py`).
- Freeze status: PASS.

The new files are independent services and no existing `__init__.py` was edited.
