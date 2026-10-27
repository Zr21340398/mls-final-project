"""Synthetic public traces, independent of private loads and scheduler decisions."""

import math
import numpy as np


def make_trace(config, ids, parameter_count):
    s = config["simulation"]
    t = config["train"]
    rng = np.random.default_rng(config["seed"] + 104729)
    profiles = s["profiles"]
    assignment = {cid: profiles[i % len(profiles)] for i, cid in enumerate(ids)}
    payload = parameter_count * 4
    trace = []
    for r in range(1, t["rounds"] + 1):
        rows = []
        for cid in ids:
            p = assignment[cid]
            compute = t["local_steps"] * s["reference_step_s"] / p["speed"]
            # Protocol bytes are MODELED assumptions. They are not captured
            # network traffic and must not be reported as measured SecAgg overhead.
            overhead = (
                (t["clients_per_round"] - 1) * 64
                if s["aggregation"] == "masking_sim"
                else 0
            )
            upload = payload + overhead
            download = payload
            network = (upload + download) * 8 / (p["mbps"] * 1e6) + 2 * p[
                "latency_ms"
            ] / 1000
            estimate = compute + network
            actual = estimate * float(rng.uniform(0.9, 1.1))
            memory_mb = (
                parameter_count * 20
                + t["batch_size"]
                * config.get("lookback", 96)
                * config["model"]["hidden"]
                * 16
            ) / 2**20
            work_units = t["local_steps"] / p["speed"] + (upload + download) / 1e6
            rows.append(
                {
                    "client": cid,
                    "profile": p["name"],
                    "available": bool(rng.random() < s["availability"]),
                    "dropout": bool(rng.random() < s["dropout"]),
                    "predicted_s": estimate,
                    "actual_s": actual,
                    "fits_memory": bool(memory_mb <= p["ram_mb"]),
                    "fits_energy": bool(work_units <= p["energy_budget"]),
                    "estimated_model_working_mb": memory_mb,
                    "modeled_energy_units": work_units,
                    "modeled_upload_bytes": upload,
                    "modeled_download_bytes": download,
                }
            )
        trace.append(rows)
    return trace


def select(rows, last_seen, round_no, config):
    t = config["train"]
    s = config["simulation"]
    m = t["clients_per_round"]
    eligible = [
        r for r in rows if r["available"] and r["fits_memory"] and r["fits_energy"]
    ]
    if t["scheduler"] == "random":
        rng = np.random.default_rng(config["seed"] + round_no * 7919)
        rng.shuffle(eligible)
        return eligible[:m]
    fastest = sorted(eligible, key=lambda r: (r["predicted_s"], r["client"]))
    if t["scheduler"] == "fastest":
        return fastest[:m]
    reserve = math.ceil(m * s["coverage_fraction"])
    old = sorted(
        eligible,
        key=lambda r: (last_seen.get(r["client"], 0), r["predicted_s"], r["client"]),
    )[:reserve]
    chosen = {r["client"] for r in old}
    return (old + [r for r in fastest if r["client"] not in chosen])[:m]


def outcome(selected, config):
    s = config["simulation"]
    survivors = [
        r for r in selected if not r["dropout"] and r["actual_s"] <= s["deadline_s"]
    ]
    # Failure paths consume virtual time but do not release a new model or DP budget.
    failed = any(r["dropout"] or r["actual_s"] > s["deadline_s"] for r in selected)
    elapsed = (
        s["deadline_s"]
        if failed
        else max([r["actual_s"] for r in selected], default=0.0)
    )
    return survivors, elapsed, len(survivors) >= s["min_survivors"]
