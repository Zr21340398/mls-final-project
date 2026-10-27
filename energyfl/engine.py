"""Sequential virtual-client training with transactionally committed round folders."""

import csv, gzip, os, time, importlib.metadata, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from .common import (
    load_config,
    read_json,
    write_json,
    fingerprint,
    snapshot,
    environment,
    digest,
    validate_calibration,
)
from .data import load_dataset
from .models import Forecaster, setup, state, flat, assign, fit, evaluate
from .privacy import aggregate, epsilon
from .simulation import make_trace, select, outcome


def model_for(c, m):
    return Forecaster(
        c["model"]["kind"], c["model"]["hidden"], m["lookback"], m["features"]
    )


def committed(run):
    return sorted(
        p
        for p in (Path(run) / "rounds").glob("[0-9]*")
        if p.is_dir() and (p / "checkpoint.pt").is_file()
    )


def load_checkpoint(path):
    # Only checkpoints produced by this code are accepted. Never load an untrusted
    # pickle; weights_only constrains PyTorch's deserializer.
    return torch.load(path, map_location="cpu", weights_only=True)


def run_training(config_path, run_dir, resume=False, stop_after=None):
    c = load_config(config_path)
    run = Path(run_dir)
    dataset = Path(c["data"]).resolve()
    if (run / "test").exists():
        raise ValueError(
            "Run has been tested and is frozen; create a new run for further training"
        )
    manifest, households = load_dataset(dataset)
    calibration_report = validate_calibration(
        c,
        manifest,
        run / "calibration_report.json"
        if resume and (run / "calibration_report.json").exists()
        else None,
    )
    ids = [h.id for h in households]
    by_id = {h.id: h for h in households}
    t = c["train"]
    p = c["privacy"]
    s = c["simulation"]
    if t["method"] == "federated" and t["clients_per_round"] > len(ids):
        raise ValueError("clients_per_round exceeds processed cohort")
    device = setup(c["seed"], t["device"], t["threads"])
    model = model_for(c, manifest).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    last_seen = {cid: 0 for cid in ids}
    participation = {cid: 0 for cid in ids}
    local_states = (
        {cid: state(model) for cid in ids} if t["method"] == "local" else None
    )
    releases = 0
    virtual_s = 0.0
    best = float("inf")
    best_round = 0
    start_round = 1
    wall_previous = 0.0
    config_fp = fingerprint(c)
    if resume:
        saved = read_json(run / "metadata.json")
        if (
            saved["config_fingerprint"] != config_fp
            or saved["dataset_fingerprint"] != manifest["fingerprint"]
        ):
            raise ValueError(
                "Resume requires identical config and dataset; create a new run for changes"
            )
        # Detect edits instead of silently mixing two implementations in one run.
        for rel, expected in saved["source_files"].items():
            if digest(Path(__file__).parents[1] / rel) != expected:
                raise ValueError("Source changed; start a new run: " + rel)
        rounds = committed(run)
        if rounds:
            ck = load_checkpoint(rounds[-1] / "checkpoint.pt")
            model.load_state_dict(ck["model"])
            local_states = ck["local_states"]
            last_seen = ck["last_seen"]
            participation = ck["participation"]
            releases = ck["releases"]
            virtual_s = ck["virtual_s"]
            best = ck["best"]
            best_round = ck["best_round"]
            start_round = ck["round"] + 1
            wall_previous = ck["wall_s"]
        trace = read_json(run / "trace.json")
    else:
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(
                "Run directory is not empty; use --resume or a new run ID"
            )
        run.mkdir(parents=True, exist_ok=True)
        (run / "rounds").mkdir(exist_ok=True)
        write_json(run / "config.json", c)
        write_json(run / "dataset_manifest.json", manifest)
        if calibration_report:
            shutil.copyfile(calibration_report, run / "calibration_report.json")
        source_hash = snapshot(run)
        root = Path(__file__).parents[1]
        meta = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": config_fp,
            "dataset_fingerprint": manifest["fingerprint"],
            "dataset_path": str(dataset),
            "source_zip_sha256": source_hash,
            "source_files": {
                str(q.relative_to(root)).replace("\\", "/"): digest(q)
                for q in root.glob("energyfl/*.py")
            },
            "environment": environment(),
            "synthetic_data": manifest["synthetic"],
            "privacy_scope": "Research artifacts include non-private validation metrics. Do not publish this directory as DP output.",
            "aggregation_backend": s["aggregation"],
            "security_claim": "No real secure aggregation; masking_sim is a correctness simulator only.",
        }
        write_json(run / "metadata.json", meta)
        packages = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        )
        (run / "environment.freeze.txt").write_text(
            "\n".join(packages), encoding="utf-8"
        )
        trace_config = dict(c, lookback=manifest["lookback"])
        trace = make_trace(trace_config, ids, len(flat(model)))
        write_json(run / "trace.json", trace)
    # Lock prevents concurrent jobs from corrupting one run; a crash can leave the
    # lock behind. Removing it requires the explicit unlock CLI after checking PID.
    lock = run / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            "Run already locked; verify old process has stopped before unlock"
        )
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    from .profile import peak_cpu_mb

    started = time.perf_counter()
    status = "complete"
    try:
        total_rounds = 1 if t["method"] == "seasonal" else t["rounds"]
        for r in range(start_round, total_rounds + 1):
            if stop_after is not None and r > stop_after:
                status = "paused"
                break
            if (
                p["enabled"]
                and epsilon(releases + 1, p["noise_multiplier"], p["delta"])
                > p["epsilon_cap"]
            ):
                status = "privacy_budget_exhausted"
                break
            tick = time.perf_counter()
            events = []
            committed_update = False
            noise_std = 0.0
            losses = []
            if t["method"] == "federated":
                chosen = select(trace[r - 1], last_seen, r, c)
                survivors, dt, ok = outcome(chosen, c)
                virtual_s += dt
                chosen_ids = {v["client"] for v in chosen}
                survivor_ids = {v["client"] for v in survivors}
                for row in trace[r - 1]:
                    e = dict(
                        row,
                        round=r,
                        selected=row["client"] in chosen_ids,
                        survived=row["client"] in survivor_ids,
                    )
                    events.append(e)
                for row in chosen:
                    last_seen[row["client"]] = r
                if ok:
                    base = flat(model)
                    updates = []
                    weights = []
                    for row in survivors:
                        cid = row["client"]
                        assign(model, base.copy())
                        stats = fit(
                            model,
                            [by_id[cid]],
                            t["local_steps"],
                            t["batch_size"],
                            t["lr"],
                            c["seed"] + r * 10000 + ids.index(cid),
                            device,
                        )
                        updates.append(flat(model) - base)
                        weights.append(len(by_id[cid]))
                        losses.append(stats["train_loss_normalized"])
                        for e in events:
                            if e["client"] == cid:
                                e.update(stats)
                        participation[cid] += 1
                    delta, noise_std = aggregate(
                        updates, weights, len(chosen), c, c["seed"] + r * 65537
                    )
                    assign(model, base + delta)
                    committed_update = True
                    if p["enabled"]:
                        releases += 1
                modeled_bytes = sum(
                    e["modeled_upload_bytes"] + e["modeled_download_bytes"]
                    for e in chosen
                )
            elif t["method"] in ["centralized", "local"]:
                steps = (
                    t["local_steps"] * t["clients_per_round"]
                    if t["method"] == "centralized"
                    else t["local_steps"]
                )
                if t["method"] == "centralized":
                    stats = fit(
                        model,
                        households,
                        steps,
                        t["batch_size"],
                        t["lr"],
                        c["seed"] + r * 10000,
                        device,
                    )
                    losses.append(stats["train_loss_normalized"])
                else:
                    for i, h in enumerate(households):
                        model.load_state_dict(local_states[h.id])
                        stats = fit(
                            model,
                            [h],
                            steps,
                            t["batch_size"],
                            t["lr"],
                            c["seed"] + r * 10000 + i,
                            device,
                        )
                        local_states[h.id] = state(model)
                        losses.append(stats["train_loss_normalized"])
                committed_update = True
                modeled_bytes = 0
            else:
                modeled_bytes = 0
            validation = None
            if (
                r % t["eval_every"] == 0
                or r == total_rounds
                or t["method"] == "seasonal"
            ):
                validation = evaluate(
                    model,
                    households,
                    "val",
                    device,
                    local_states=local_states,
                    seasonal=t["method"] == "seasonal",
                )
                if validation["macro_mae_kwh"] < best:
                    best = validation["macro_mae_kwh"]
                    best_round = r
            metrics = {
                "round": r,
                "update_applied": committed_update,
                "validation": validation,
                "train_loss_normalized": float(np.mean(losses)) if losses else None,
                "peak_process_rss_mb": peak_cpu_mb(),
                "measured_round_s": time.perf_counter() - tick,
                "measured_wall_s": wall_previous + time.perf_counter() - started,
                "virtual_elapsed_s": virtual_s,
                "modeled_round_bytes": modeled_bytes,
                "participation_coverage": sum(v > 0 for v in participation.values())
                / len(ids),
                "epsilon": epsilon(releases, p["noise_multiplier"], p["delta"])
                if p["enabled"]
                else None,
                "dp_releases": releases,
                "noise_std": noise_std,
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda"
                else 0.0,
            }
            # The directory rename is the commit point. A crash before this point
            # leaves .pending data ignored; resume never double-counts committed rounds.
            tmp = run / "rounds" / f".pending-{r:04}"
            tmp.mkdir(exist_ok=True)
            write_json(tmp / "metrics.json", metrics)
            write_json(tmp / "clients.json", events)
            ck = {
                "round": r,
                "model": state(model),
                "local_states": local_states,
                "last_seen": last_seen,
                "participation": participation,
                "releases": releases,
                "virtual_s": virtual_s,
                "best": best,
                "best_round": best_round,
                "wall_s": metrics["measured_wall_s"],
            }
            torch.save(ck, tmp / "checkpoint.pt")
            os.replace(tmp, run / "rounds" / f"{r:04}")
            write_json(
                run / "status.json",
                {
                    "status": "running",
                    "last_round": r,
                    "best_validation_round": best_round,
                    "epsilon": metrics["epsilon"],
                },
            )
            print(
                f"round={r} val_mae={validation['macro_mae_kwh'] if validation else None} virtual_s={virtual_s:.2f} epsilon={metrics['epsilon']}",
                flush=True,
            )
        rounds = committed(run)
        write_json(
            run / "status.json",
            {
                "status": status,
                "last_round": int(rounds[-1].name) if rounds else 0,
                "best_validation_round": best_round,
            },
        )
    except BaseException as exc:
        write_json(run / "status.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)
    return run


def test_run(run_dir, checkpoint="best", data_path=None):
    run = Path(run_dir)
    c = load_config(run / "config.json")
    if read_json(run / "status.json")["status"] not in [
        "complete",
        "privacy_budget_exhausted",
    ]:
        raise ValueError(
            "Finish training or exhaust the declared privacy budget before testing"
        )
    m, hs = load_dataset(data_path or read_json(run / "metadata.json")["dataset_path"])
    if m["fingerprint"] != read_json(run / "metadata.json")["dataset_fingerprint"]:
        raise ValueError("Dataset mismatch")
    if (run / "test").exists():
        raise FileExistsError(
            "Test already evaluated; preserved to discourage repeated test-based tuning"
        )
    rounds = committed(run)
    if not rounds:
        raise ValueError("No committed model")
    latest = load_checkpoint(rounds[-1] / "checkpoint.pt")
    if c["privacy"]["enabled"] and checkpoint != "last":
        raise ValueError(
            "DP experiments use last checkpoint; raw validation-based selection is not a DP release mechanism"
        )
    r = int(rounds[-1].name) if checkpoint == "last" else latest["best_round"]
    if not r:
        raise ValueError("No validated checkpoint; use last")
    ck = load_checkpoint(run / "rounds" / f"{r:04}" / "checkpoint.pt")
    device = setup(c["seed"], c["train"]["device"], c["train"]["threads"])
    model = model_for(c, m).to(device)
    model.load_state_dict(ck["model"])
    tmp = run / ".test-pending"
    tmp.mkdir(exist_ok=True)
    with gzip.open(tmp / "predictions.csv.gz", "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["client", "target_timestamp_ns", "actual_kwh", "predicted_kwh"]
        )
        result = evaluate(
            model,
            hs,
            "test",
            device,
            local_states=ck["local_states"],
            seasonal=c["train"]["method"] == "seasonal",
            prediction_sink=writer.writerow,
        )
    result.update(
        checkpoint_round=r,
        checkpoint_sha256=digest(run / "rounds" / f"{r:04}" / "checkpoint.pt"),
        dataset_fingerprint=m["fingerprint"],
    )
    write_json(tmp / "metrics.json", result)
    os.replace(tmp, run / "test")
    return result
