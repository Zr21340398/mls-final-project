"""Generate reviewed experiment configs; expensive execution requires --execute."""

import argparse, copy, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energyfl.common import load_config, write_json
from energyfl.privacy import epsilon


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/l20.json")
    p.add_argument("--suite", choices=["core", "privacy", "dropout"], default="core")
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    p.add_argument("--plan", default="reports/plans")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--test-after", action="store_true")
    a = p.parse_args()
    c = load_config(a.base)
    variants = []
    if a.suite == "core":
        for name, method, scheduler, weight in [
            ("seasonal", "seasonal", "random", "equal"),
            ("local", "local", "random", "equal"),
            ("centralized", "centralized", "random", "equal"),
            ("fedavg_sample", "federated", "random", "samples"),
            ("random_equal", "federated", "random", "equal"),
            ("fastest_equal", "federated", "fastest", "equal"),
            ("coverage_equal", "federated", "coverage", "equal"),
        ]:
            d = copy.deepcopy(c)
            d["train"].update(method=method, scheduler=scheduler, weighting=weight)
            d["privacy"]["enabled"] = False
            d["simulation"]["aggregation"] = "plain"
            variants.append((name, d))
    elif a.suite == "privacy":
        for name, enabled, mask, cap in [
            ("plain", False, False, 8),
            ("masking", False, True, 8),
            ("dp4", True, True, 4),
            ("dp8", True, True, 8),
            ("dp16", True, True, 16),
        ]:
            d = copy.deepcopy(c)
            d["train"]["weighting"] = "equal"
            d["privacy"].update(enabled=enabled, epsilon_cap=cap, clip_updates=True)
            d["simulation"]["aggregation"] = "masking_sim" if mask else "plain"
            variants.append((name, d))
    else:
        for rate in [0.0, 0.1, 0.3]:
            for scheduler in ["random", "fastest", "coverage"]:
                d = copy.deepcopy(c)
                d["simulation"]["dropout"] = rate
                d["train"]["scheduler"] = scheduler
                variants.append((f"{scheduler}_drop{int(rate * 100)}", d))
    jobs = []
    for name, d in variants:
        for seed in a.seeds:
            d = copy.deepcopy(d)
            d["seed"] = seed
            label = f"{a.suite}_{name}_s{seed}"
            cfg = Path(a.plan) / (label + ".json")
            run = Path(a.runs_root) / label
            if cfg.exists():
                raise FileExistsError(
                    f"Plan exists: {cfg}; choose a new --plan directory"
                )

            if a.suite == "privacy" and d["privacy"]["enabled"]:
                # Match the requested epsilon at the configured full number of releases;
                # changing only the cap can otherwise yield identical dp8/dp16 runs.
                lo, hi = 0.01, 1024.0
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if (
                        epsilon(d["train"]["rounds"], mid, d["privacy"]["delta"])
                        > d["privacy"]["epsilon_cap"]
                    ):
                        lo = mid
                    else:
                        hi = mid
                d["privacy"]["noise_multiplier"] = hi * 1.000001
            write_json(cfg, d)
            jobs.append(
                {"config": str(cfg), "run": str(run), "method": d["train"]["method"]}
            )
    write_json(Path(a.plan) / f"{a.suite}_jobs.json", jobs)
    print(
        f"{len(jobs)} jobs prepared. Sequential execution only; check dataset and profile calibration before --execute."
    )
    if a.execute:
        for job in jobs:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "energyfl",
                    "train",
                    "--config",
                    job["config"],
                    "--run",
                    job["run"],
                ],
                check=True,
                cwd=ROOT,
            )
            if a.test_after:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "energyfl",
                        "test",
                        "--run",
                        job["run"],
                        "--checkpoint",
                        "last" if a.suite == "privacy" else "best",
                    ],
                    check=True,
                    cwd=ROOT,
                )


if __name__ == "__main__":
    main()
