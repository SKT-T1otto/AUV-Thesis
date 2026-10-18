"""Exact reviewed shared-core exception, not a directory/hash exemption."""
import json
from pathlib import Path

PATH = "core/mapping/path_planner.py"
HISTORICAL = "0c86f2a701c931179d6609ea755f82fe27d1e8df35672ff5696f5471bcad1925"
CURRENT = "26fd9e2686ef738b3e34269b5e967788926843dad33b7bbe6d8334bf5f488634"
AST = "16d12e16dd2d9e05794ec87f5c1496b43fb08607815019d6082e0d0039688e71"
RECORD = dict(path=PATH, phase0b2_sha256=HISTORICAL, current_sha256=CURRENT, current_ast_dump_sha256=AST)
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "docs/provenance/d2_performance_v1_evolution.json").read_text())
TARGET_RECORD = dict(path="core/env/target_motion.py",
    phase0b2_sha256="8e2cd1fc585bd3b148a1f0ff87ab8eb758bfe3db9142792d40da987b7ec5fde0",
    current_sha256="e35ae02447723dfdc6be23d31fa4ed73a992ddcf12998947bd2af0b6084844af",
    current_ast_dump_sha256="e57c1e5486cdbc51299ff09cbd8de1dea0657b561019962dc07dce7d4d58d9dd")
RECORDS = [RECORD, TARGET_RECORD]
assert manifest["core_records"] == RECORDS
assert manifest["historical_manifests_modified"] is False

PERMITTED = dict(historical_sha256=HISTORICAL, current_sha256=CURRENT, current_ast_dump_sha256=AST)

PERMITTED_BY_PATH = {record["path"]: dict(
    historical_sha256=record["phase0b2_sha256"], current_sha256=record["current_sha256"],
    current_ast_dump_sha256=record["current_ast_dump_sha256"]) for record in RECORDS}

# The first-round manifest and assertions above remain frozen. The new user
# request explicitly reviews only this small successor to its planner record.
ROUND2_CURRENT = "032cc1f5f1488f668ac7c1b4e8f7bb0a01003f9546f04312008dd464ffcb269c"
ROUND2_AST = "e2020c73db9b4181bdf35a98371beb12fa12e50bc52f270395266f922088db83"
ROUND2_RECORD = dict(path=PATH, phase0b2_sha256=HISTORICAL, previous_reviewed_sha256=CURRENT,
                     current_sha256=ROUND2_CURRENT, current_ast_dump_sha256=ROUND2_AST)
round2 = json.loads((ROOT / "docs/provenance/d2_performance_v2_evolution.json").read_text())
assert round2["core_records"] == [ROUND2_RECORD]
assert round2["historical_manifests_modified"] is False
PERMITTED_BY_PATH[PATH] = dict(historical_sha256=HISTORICAL, current_sha256=ROUND2_CURRENT,
                              current_ast_dump_sha256=ROUND2_AST)
RECORDS = [ROUND2_RECORD, TARGET_RECORD]
