"""External Windows measurement driver; does not modify either source checkout.

Run with the same Python used by D2, from the candidate repository root:
  python docs/chapter3/d2_suite_v1/performance_round2/revalidate_windows.py --probe
  python docs/chapter3/d2_suite_v1/performance_round2/revalidate_windows.py --run

Fresh child processes share one external benchmark loader and one logical CPU.
Only this driver's children have their affinity changed. No power setting,
priority, user application, production algorithm, or RNG is changed.
"""
from __future__ import annotations

import argparse
import ctypes as C
from ctypes import wintypes as W
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASELINE = ROOT / "runs/d2_round2_20261010/baseline_source"
MANIFEST = BASELINE / "configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json"
HARNESS = ROOT / "tools/benchmark_d2_performance.py"
OUT = ROOT / "runs/d2_round2_revalidation_20261010"
CPU = 8
BLOCKS = [("B2", "AB"), ("B3", "BA"), ("B2", "BA"), ("B3", "AB")]
K = C.WinDLL("kernel32", use_last_error=True)
K.GetCurrentProcess.restype = W.HANDLE
K.SetProcessAffinityMask.argtypes = [W.HANDLE, C.c_size_t]
K.GetProcessAffinityMask.argtypes = [W.HANDLE, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
K.GetSystemCpuSetInformation.argtypes = [C.c_void_p, W.ULONG, C.POINTER(W.ULONG), W.HANDLE, W.ULONG]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topology():
    size = W.ULONG()
    K.GetSystemCpuSetInformation(None, 0, C.byref(size), None, 0)
    buffer = C.create_string_buffer(size.value)
    if not K.GetSystemCpuSetInformation(buffer, size, C.byref(size), None, 0):
        raise C.WinError(C.get_last_error())
    offset, rows = 0, []
    while offset < size.value:
        length, kind = struct.unpack_from("<II", buffer.raw, offset)
        if length < 8:
            raise RuntimeError("invalid CPU set record")
        if kind == 0:
            rows.append(dict(zip(("id", "group", "logical", "core", "cache", "numa", "efficiency", "flags"),
                                 struct.unpack_from("<IHBBBBBB", buffer.raw, offset + 8))))
        offset += length
    return rows


class PowerStatus(C.Structure):
    _fields_ = [("ac", W.BYTE), ("battery_flag", W.BYTE), ("battery_percent", W.BYTE),
                ("system_status", W.BYTE), ("life_seconds", W.DWORD), ("full_life_seconds", W.DWORD)]


def power():
    status = PowerStatus()
    if not K.GetSystemPowerStatus(C.byref(status)):
        raise C.WinError(C.get_last_error())
    return {key: getattr(status, key) for key, _ in status._fields_}


def plan():
    return subprocess.check_output(["powercfg", "/getactivescheme"]).decode("utf-8", errors="replace").strip()


def affinity():
    process, system = C.c_size_t(), C.c_size_t()
    if not K.GetProcessAffinityMask(K.GetCurrentProcess(), C.byref(process), C.byref(system)):
        raise C.WinError(C.get_last_error())
    return process.value


class CounterValue(C.Structure):
    _fields_ = [("status", W.DWORD), ("value", C.c_double)]


class Counters:
    """OS counter observations. Performance percent is nominal-relative, not a thermometer."""
    def __init__(self):
        self.dll = C.WinDLL("pdh")
        self.dll.PdhOpenQueryW.argtypes = [W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE)]
        self.dll.PdhAddEnglishCounterW.argtypes = [W.HANDLE, W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE)]
        self.dll.PdhCollectQueryData.argtypes = [W.HANDLE]
        self.dll.PdhGetFormattedCounterValue.argtypes = [W.HANDLE, W.DWORD, C.POINTER(W.DWORD), C.POINTER(CounterValue)]
        self.dll.PdhCloseQuery.argtypes = [W.HANDLE]
        self.query, self.handles = W.HANDLE(), {}
        assert self.dll.PdhOpenQueryW(None, 0, C.byref(self.query)) == 0
        paths = {
            "host_busy_pct": r"\Processor Information(_Total)\% Processor Time",
            "core_busy_pct": rf"\Processor Information(0,{CPU})\% Processor Time",
            "sibling_busy_pct": rf"\Processor Information(0,{CPU + 1})\% Processor Time",
            "core_performance_pct": rf"\Processor Information(0,{CPU})\% Processor Performance",
            "core_frequency_mhz": rf"\Processor Information(0,{CPU})\Processor Frequency",
        }
        for name, path in paths.items():
            handle = W.HANDLE()
            code = self.dll.PdhAddEnglishCounterW(self.query, path, 0, C.byref(handle))
            if code:
                raise RuntimeError((name, hex(code & 0xffffffff)))
            self.handles[name] = handle
        self.dll.PdhCollectQueryData(self.query)

    def sample(self):
        code = self.dll.PdhCollectQueryData(self.query)
        result = {"time_unix": time.time(), "power": power(), "collect_status": code}
        for name, handle in self.handles.items():
            value = CounterValue()
            code = self.dll.PdhGetFormattedCounterValue(handle, 0x200 | 0x8000, None, C.byref(value))
            result[name] = value.value if code == 0 and value.status in (0, 1) else None
            if code or value.status not in (0, 1):
                result[name + "_status"] = [code, value.status]
        return result

    def close(self):
        self.dll.PdhCloseQuery(self.query)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sentinel():
    """Identical stdlib-only CPU workload; no simulation imports, state, or RNG."""
    payload = bytes(range(256)) * 256
    rows = []
    for repeat in range(4):
        wall, cpu = time.perf_counter(), time.process_time()
        value = 7
        for i in range(1200000):
            value = (value * 1664525 + i + 1013904223) & 0xffffffff
        for _ in range(1200):
            digest = hashlib.sha256(payload).hexdigest()
        row = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                   result=[value, digest])
        if repeat:
            rows.append(row)
    return dict(rows=rows, median_wall_seconds=statistics.median(r["wall_seconds"] for r in rows),
                median_cpu_seconds=statistics.median(r["cpu_seconds"] for r in rows))


def child(args):
    if not K.SetProcessAffinityMask(K.GetCurrentProcess(), 1 << CPU):
        raise C.WinError(C.get_last_error())
    assert affinity() == 1 << CPU
    label, arm, output = args[:3]
    metadata = dict(cpu=CPU, affinity_before=affinity(), power_before=power(),
                    sentinel_before=sentinel(), start_unix=time.time(), cwd=str(Path.cwd()),
                    harness_sha256=sha(HARNESS), driver_sha256=sha(Path(__file__)))
    harness = load(HARNESS, "d2_identical_external_harness")
    harness.ROOT = Path.cwd()

    # Add process-CPU time only at existing measurement boundaries. Snapshot
    # contents and every underlying call remain unchanged; no per-step hooks.
    original_snapshot, original_record = harness.Timings.snapshot, harness.record
    class Snapshot(dict):
        pass
    def snapshot(self):
        value = Snapshot(original_snapshot(self))
        value.cpu_timestamp = time.process_time()
        return value
    def record(timing, before, started, steps, **extra):
        cpu = time.process_time() - before.cpu_timestamp
        row = original_record(timing, before, started, steps, **extra)
        row["process_cpu_seconds"] = cpu
        row["cpu_wall_ratio"] = cpu / row["wall_seconds"]
        return row
    harness.Timings.snapshot, harness.record = snapshot, record
    sys.argv = [str(HARNESS), "--manifest", str(MANIFEST), "--scenes", "0", "1", "--arm", arm,
                "--seed", "2729", "--device", "cpu", "--threads", "1", "--training-episodes", "2",
                "--label", label, "--output", output]
    harness.main()
    metadata.update(end_unix=time.time(), affinity_after=affinity(), power_after=power(),
                    sentinel_after=sentinel())
    write(Path(output) / "process_metadata.json", metadata)


def run():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = topology()
    chosen = next(r for r in sources if r["logical"] == CPU and r["group"] == 0)
    assert chosen["efficiency"] == max(r["efficiency"] for r in sources)
    assert power()["ac"] == 1
    protocol = dict(schema="d2.windows_revalidation.v1", created_unix=time.time(),
                    baseline=str(BASELINE), candidate=str(ROOT), manifest=str(MANIFEST),
                    manifest_sha256=sha(MANIFEST), harness_sha256=sha(HARNESS),
                    driver_sha256=sha(Path(__file__)), cpu_sets=sources, logical_cpu=CPU,
                    power_plan=plan(), power=power(), blocks=BLOCKS, total_processes=8,
                    episodes_per_process=dict(rollout=2, training=2),
                    order_design="four adjacent pairs, both orders once per arm; all fresh processes",
                    sentinel="same stdlib integer/hash workload before and after each process",
                    exclusion_policy="No timing outlier deletion. All runs retained. Exact/source/AC/affinity violations invalidate batch.",
                    stability_targets=dict(within_variant_rollout_cv_max_pct=5,
                        unchanged_control_median_ratio_max_deviation_pct=5,
                        process_cpu_wall_min=.95, paired_speedup_min=1.0),
                    inference="Repeated local workload only. Report all pairs and ranges; no formal-training extrapolation.")
    write(OUT / "protocol.json", protocol)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0")
    loader = "import importlib.util,sys; p=sys.argv.pop(1); s=importlib.util.spec_from_file_location('d2_measure_driver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.child(sys.argv[1:])"
    comparison = load(ROOT / "tools/compare_d2_performance.py", "d2_revalidation_compare")
    counters = Counters()
    try:
        for block, (arm, order) in enumerate(BLOCKS, 1):
            members = {}
            for variant in order:
                label = f"block{block}_{arm}_{variant}"
                output = OUT / label
                cwd = BASELINE if variant == "A" else ROOT
                assert sha(HARNESS) == protocol["harness_sha256"]
                assert sha(Path(__file__)) == protocol["driver_sha256"]
                assert sha(MANIFEST) == protocol["manifest_sha256"]
                print("START", label, time.strftime("%H:%M:%S"), flush=True)
                start = dict(time_unix=time.time(), power_plan=plan(), power=power())
                command = [sys.executable, "-c", loader, str(Path(__file__)), label, arm, str(output)]
                with (OUT / (label + ".log")).open("w", encoding="utf-8") as log, (OUT / (label + "_telemetry.jsonl")).open("w", encoding="utf-8") as stream:
                    process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        try:
                            code = process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            stream.write(json.dumps(counters.sample()) + "\n")
                            stream.flush()
                            continue
                        break
                start.update(end_unix=time.time(), exit_code=code, power_after=power(), power_plan_after=plan())
                write(OUT / (label + "_host.json"), start)
                if code:
                    raise RuntimeError(f"{label} exited {code}; inspect its retained log")
                value = json.loads((output / "benchmark.json").read_text())
                metadata = json.loads((output / "process_metadata.json").read_text())
                assert metadata["affinity_before"] == metadata["affinity_after"] == 1 << CPU
                assert start["power_plan"] == start["power_plan_after"] == protocol["power_plan"]
                assert metadata["power_before"]["ac"] == metadata["power_after"]["ac"] == 1
                members[variant] = value
                totals = {kind: sum(r["wall_seconds"] for r in value["cases"] if r["name"] == kind)
                          for kind in ("full_episode", "training_episode")}
                print("DONE", label, totals, flush=True)
            exact = comparison.compare(members["A"], members["B"])
            assert exact["bounded_training_verified"]
            write(OUT / f"block{block}_exact.json", exact)
            assert exact["exact_passed"], exact["errors"]
            print("EXACT PASS", block, flush=True)
    finally:
        counters.close()
    write(OUT / "completed.json", dict(completed_unix=time.time(), blocks=4, processes=8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(dict(root=str(ROOT), topology=topology(), power=power(), plan=plan()), indent=2))
        counters = Counters()
        time.sleep(2)
        print(json.dumps(counters.sample(), indent=2))
        counters.close()
    elif args.run:
        run()
    else:
        parser.error("choose --probe or --run")
