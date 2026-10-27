"""Real hardware benchmarks and validation-only inference variant assignment."""

import copy, time, warnings, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import (
    load_config,
    read_json,
    write_json,
    environment,
    digest,
    fingerprint,
    calibration_workload,
    validate_calibration,
)
from .data import load_dataset
from .models import setup, fit, evaluate
from .engine import model_for, committed, load_checkpoint


def peak_cpu_mb():
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in [
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                ]
            ]

        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        x = Counters()
        x.cb = ctypes.sizeof(x)
        if psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(x), x.cb
        ):
            return x.PeakWorkingSetSize / 2**20
        return None
    import resource

    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (2**20 if platform.system() == "Darwin" else 1024)


def benchmark(config, out, steps=20, repeats=1):
    if steps < 1 or repeats < 1:
        raise ValueError("Profiling steps and repeats must be positive")
    from datetime import datetime, timezone

    c = load_config(config)
    m, hs = load_dataset(c["data"])
    t = c["train"]
    device = setup(c["seed"], t["device"], t["threads"])
    trials = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for trial in range(repeats):
        model = model_for(c, m).to(device)
        fit(model, [hs[0]], 2, t["batch_size"], t["lr"], trial + 1, device)
        stats = fit(
            model, [hs[0]], steps, t["batch_size"], t["lr"], trial + 100, device
        )
        trials.append(stats["measured_train_s"] / steps)
    timing = float(np.median(trials))
    if not np.isfinite(timing) or timing <= 0:
        raise ValueError("Invalid profiling measurement")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "config": c,
        "dataset_fingerprint": m["fingerprint"],
        "workload_fingerprint": fingerprint(calibration_workload(c, m)),
        "measured_train_step_s": timing,
        "trial_step_seconds": trials,
        "steps_per_trial": steps,
        "warmup_steps_per_trial": 2,
        "batch_size": t["batch_size"],
        "peak_process_rss_mb": peak_cpu_mb(),
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else 0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "note": "Measurement for this hardware/workload only. Use calibrate to generate a linked configuration automatically.",
    }
    write_json(out, report)
    return report


def calibrate(config, out_config, report_path, steps=20, repeats=3):
    """Keep the template unchanged; link the generated config to immutable evidence."""
    from .download import exclusive_lock

    config = Path(config)
    out_config = Path(out_config)
    report_path = Path(report_path)
    if len({config.resolve(), out_config.resolve(), report_path.resolve()}) != 3:
        raise ValueError(
            "Template, generated configuration and report must have different paths"
        )
    base = load_config(config)
    with exclusive_lock(out_config.with_suffix(out_config.suffix + ".lock")):
        if out_config.exists():
            cached = load_config(out_config)
            cal = cached["simulation"].get("calibration", {})
            if cal.get("base_config_fingerprint") != fingerprint(base):
                raise ValueError(
                    "Base configuration changed; use new calibration output paths"
                )
            if Path(cal.get("report_path", "")).resolve() != report_path.resolve():
                raise ValueError(
                    "Calibration report path differs; choose a new output configuration"
                )
            m, _ = load_dataset(base["data"])
            validate_calibration(cached, m)
            return {
                "config": str(out_config),
                "report": str(report_path),
                "reused": True,
            }
        if report_path.exists():
            # Resume an interruption between report creation and config creation.
            report = read_json(report_path)
            m, _ = load_dataset(base["data"])
            if fingerprint(report["config"]) != fingerprint(base) or report[
                "workload_fingerprint"
            ] != fingerprint(calibration_workload(base, m)):
                raise ValueError(
                    "Existing report does not match this workload; choose a new report path"
                )
        else:
            report = benchmark(config, report_path, steps, repeats)
        updated = copy.deepcopy(base)
        updated["simulation"]["reference_step_s"] = report["measured_train_step_s"]
        updated["simulation"]["calibration_source"] = str(report_path.resolve())
        updated["simulation"]["calibration"] = {
            "report_path": str(report_path.resolve()),
            "report_sha256": digest(report_path),
            "workload_fingerprint": report["workload_fingerprint"],
            "base_config_fingerprint": fingerprint(base),
            "statistic": "median_of_measured_local_step_times",
        }
        write_json(out_config, updated)
        return {
            "config": str(out_config),
            "report": str(report_path),
            "reference_step_s": report["measured_train_step_s"],
            "reused": False,
        }


@torch.inference_mode()
def compression(run_dir, out, repeats=100, tolerance=0.05):
    if repeats < 10 or tolerance < 0:
        raise ValueError("Use >=10 benchmark repeats and nonnegative tolerance")
    run = Path(run_dir)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new compression output directory")
    c = load_config(run / "config.json")
    if c["train"]["method"] in ["local", "seasonal"]:
        raise ValueError("Compression needs a shared neural model")
    m, hs = load_dataset(read_json(run / "metadata.json")["dataset_path"])
    setup(c["seed"], "cpu", c["train"]["threads"])
    rs = committed(run)
    if not rs:
        raise ValueError("No checkpoints")
    latest = load_checkpoint(rs[-1] / "checkpoint.pt")
    r = int(rs[-1].name) if c["privacy"]["enabled"] else latest["best_round"]
    if not r:
        raise ValueError("No validated checkpoint")
    model = model_for(c, m)
    model.load_state_dict(
        load_checkpoint(run / "rounds" / f"{r:04}" / "checkpoint.pt")["model"]
    )
    model.eval()
    # Dynamic quantization is CPU-only here. Never feed quantized parameters back
    # into FL aggregation, and select deployment variants using validation only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        quant = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model), {nn.GRU, nn.Linear}, dtype=torch.qint8
        )
    eligible_hs = [h for h in hs if len(h.arr["val"])]
    if not eligible_hs:
        raise ValueError("No validation windows for compression benchmark")
    x = torch.from_numpy(eligible_hs[0].batch("val", np.array([0]))[0])
    variants = {}
    out.mkdir(parents=True, exist_ok=True)
    for name, net in [("fp32", model), ("int8", quant)]:
        for _ in range(20):
            net(x)
        times = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            net(x)
            times.append((time.perf_counter_ns() - start) / 1e6)
        artifact = out / (name + ".pt")
        torch.save(net.state_dict(), artifact)
        val = evaluate(net, hs, "val", torch.device("cpu"))
        variants[name] = {
            "serialized_bytes": artifact.stat().st_size,
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "validation_mae_kwh": val["macro_mae_kwh"],
        }
    assignments = []
    base = variants["fp32"]["validation_mae_kwh"]
    for p in c["simulation"]["profiles"]:
        eligible = [
            k
            for k, v in variants.items()
            if v["validation_mae_kwh"] <= base * (1 + tolerance)
            and v["serialized_bytes"] / 2**20 <= p["ram_mb"]
            and v["p95_ms"] / p["speed"] <= p["inference_deadline_ms"]
        ]
        chosen = (
            min(eligible, key=lambda k: variants[k]["serialized_bytes"])
            if eligible
            else None
        )
        assignments.append(
            {
                "profile": p["name"],
                "variant": chosen,
                "modeled_p95_ms": variants[chosen]["p95_ms"] / p["speed"]
                if chosen
                else None,
                "note": "Serialized-size feasibility is a lower bound; runtime/framework memory must also be profiled before real deployment.",
            }
        )
    result = {
        "environment": environment(),
        "checkpoint_round": r,
        "variants": variants,
        "assignment": assignments,
        "batch_size": 1,
        "repeats": repeats,
        "relative_error_tolerance": tolerance,
    }
    write_json(out / "compression.json", result)
    return result
