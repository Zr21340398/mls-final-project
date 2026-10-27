"""Atomic artifacts and configuration validation. No machine-specific paths."""

import hashlib, json, os, platform, sys, zipfile
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(tmp, p)


def load_config(path):
    c = read_json(path)
    required = {"seed", "data", "model", "train", "simulation", "privacy"}
    if required - set(c):
        raise ValueError(f"Missing config sections: {required - set(c)}")
    t, s, d = c["train"], c["simulation"], c["privacy"]
    if t["method"] not in ["federated", "centralized", "local", "seasonal"]:
        raise ValueError("Unknown method")
    if t["scheduler"] not in ["random", "fastest", "coverage"]:
        raise ValueError("Unknown scheduler")
    if t["weighting"] not in ["equal", "samples"]:
        raise ValueError("Unknown weighting")
    for k in [
        "rounds",
        "local_steps",
        "batch_size",
        "threads",
        "eval_every",
        "clients_per_round",
    ]:
        if not isinstance(t[k], int) or t[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if t["lr"] <= 0 or not np.isfinite(t["lr"]):
        raise ValueError("lr must be finite and positive")
    for k in ["dropout", "availability"]:
        if not 0 <= s[k] <= 1:
            raise ValueError(k + " must be in [0,1]")
    if not 0 <= s["coverage_fraction"] <= 1:
        raise ValueError("coverage_fraction must be in [0,1]")
    if not 1 <= s["min_survivors"] <= t["clients_per_round"]:
        raise ValueError("Invalid minimum cohort")
    for k in ["deadline_s", "reference_step_s"]:
        if s[k] <= 0:
            raise ValueError(k + " must be positive")
    if s["aggregation"] not in ["plain", "masking_sim"]:
        raise ValueError("Supported aggregation: plain/masking_sim")
    for profile in s["profiles"]:
        if any(
            profile[k] <= 0
            for k in ["speed", "ram_mb", "mbps", "energy_budget", "latency_ms"]
        ):
            raise ValueError("Profile values must be positive")
    if c["model"]["kind"] not in ["gru", "mlp"]:
        raise ValueError("Unknown model")
    if c["model"]["hidden"] < 1:
        raise ValueError("hidden must be positive")
    if (
        not 0 < d["delta"] < 1
        or d["clip_norm"] <= 0
        or d["noise_multiplier"] <= 0
        or d["epsilon_cap"] <= 0
    ):
        raise ValueError("Invalid privacy parameters")
    if d["enabled"] and (t["method"] != "federated" or t["weighting"] != "equal"):
        raise ValueError("DP requires equal-weight federated learning")
    if s["aggregation"] == "masking_sim" and t["weighting"] != "equal":
        raise ValueError("Masking simulator supports equal weights only")
    return c


def snapshot(run):
    root = Path(__file__).resolve().parents[1]
    files = []
    for pattern in [
        "energyfl/**/*.py",
        "tests/**/*.py",
        "configs/*.json",
        "scripts/*",
        "docs/*.md",
        "README.md",
        "pyproject.toml",
        "requirements*.txt",
    ]:
        files.extend(root.glob(pattern))
    with zipfile.ZipFile(Path(run) / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            if p.is_file():
                z.write(p, p.relative_to(root))
    return digest(Path(run) / "source.zip")


def environment():
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": sys.argv,
        "pid": os.getpid(),
    }


def seal_run(run):
    """Create a content manifest before copying a completed experiment off a node."""
    run = Path(run)
    if (run / "RUNNING.lock").exists():
        raise ValueError("Cannot seal a running experiment")
    if read_json(run / "status.json")["status"] not in [
        "complete",
        "privacy_budget_exhausted",
    ]:
        raise ValueError("Only completed experiments can be sealed")
    files = {
        p.relative_to(run).as_posix(): digest(p)
        for p in run.rglob("*")
        if p.is_file()
        and p.name != "integrity.json"
        and not any(part.startswith(".") for part in p.relative_to(run).parts)
    }
    write_json(run / "integrity.json", {"algorithm": "SHA256", "files": files})
    return {"files": len(files), "status": "sealed"}


def verify_run(run):
    run = Path(run).resolve()
    saved = read_json(run / "integrity.json")["files"]
    bad = []
    for name, expected in saved.items():
        p = (run / name).resolve()
        if not p.is_relative_to(run) or not p.is_file() or digest(p) != expected:
            bad.append(name)
    if bad:
        raise ValueError("Missing or modified artifacts: " + ", ".join(bad))
    return {"files": len(saved), "status": "verified"}


def calibration_workload(config, manifest):
    """Only fields affecting one reference local step; seeds/schedulers may vary."""
    return {
        "model": config["model"],
        "batch_size": config["train"]["batch_size"],
        "device": config["train"]["device"],
        "threads": config["train"]["threads"],
        "lookback": manifest["lookback"],
        "features": manifest["features"],
        "dataset_fingerprint": manifest["fingerprint"],
    }


def validate_calibration(config, manifest, archived_report=None):
    cal = config["simulation"].get("calibration")
    if not cal:
        return None  # Legacy/manual configurations remain explicit assumptions.
    report = Path(archived_report or cal["report_path"])
    if not report.is_file() or digest(report) != cal["report_sha256"]:
        raise ValueError(
            "Calibration report is missing or changed; recalibrate into a new configuration"
        )
    measured = read_json(report)
    if (
        fingerprint(calibration_workload(config, manifest))
        != cal["workload_fingerprint"]
    ):
        raise ValueError(
            "Model, data, batch size, device or threads changed; recalibrate before training"
        )
    if measured["workload_fingerprint"] != cal["workload_fingerprint"]:
        raise ValueError("Calibration workload mismatch")
    if config["simulation"]["reference_step_s"] != measured["measured_train_step_s"]:
        raise ValueError("Calibration timing was edited independently of its report")
    return report
