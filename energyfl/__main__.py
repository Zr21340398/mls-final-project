import argparse, glob, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Reproducible federated household forecasting research"
    )
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("download")
    a.add_argument("--out", default="data/raw/lcl")
    a.add_argument("--archive-only", action="store_true")
    a = sub.add_parser("calibrate")
    a.add_argument("--config", required=True)
    a.add_argument("--out-config", required=True)
    a.add_argument("--report", required=True)
    a.add_argument("--steps", type=int, default=20)
    a.add_argument("--repeats", type=int, default=3)
    a = sub.add_parser("bootstrap-pilot")
    a.add_argument("--config", default="configs/pilot_cpu.json")
    a.add_argument("--raw", default="data/raw/lcl")
    a.add_argument("--out-config", default="configs/pilot_cpu.calibrated.json")
    a.add_argument("--report", default="reports/pilot_setup/profile.json")
    a.add_argument("--start", default="2013-01-01")
    a.add_argument("--end", default="2013-07-01")
    a.add_argument("--clients", type=int, default=12)
    a.add_argument(
        "--run", help="Optional pilot training directory; no test-set evaluation"
    )
    a = sub.add_parser("synthetic")
    a.add_argument("--out", required=True)
    a.add_argument("--clients", type=int, default=8)
    a.add_argument("--days", type=int, default=30)
    a.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("prepare")
    a.add_argument("--csv", nargs="+", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--start", required=True)
    a.add_argument("--end", required=True)
    a.add_argument("--clients", type=int, default=60)
    a.add_argument("--min-coverage", type=float, default=0.9)
    a.add_argument("--lookback", type=int, default=96)
    a.add_argument("--max-fill", type=int, default=2)
    a = sub.add_parser("train")
    a.add_argument("--config", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--stop-after", type=int)
    a = sub.add_parser("test")
    a.add_argument("--run", required=True)
    a.add_argument("--checkpoint", choices=["best", "last"], default="best")
    a.add_argument("--data", help="Relocated processed dataset; fingerprint must match")
    a = sub.add_parser("profile")
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("compress")
    a.add_argument("--run", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--repeats", type=int, default=100)
    a = sub.add_parser("analyze")
    a.add_argument("--runs", default="runs")
    a.add_argument("--out", default="reports/latest")
    a = sub.add_parser("dashboard")
    a.add_argument("--runs", default="runs")
    a.add_argument("--port", type=int, default=8765)
    for command in ["seal", "verify"]:
        a = sub.add_parser(command)
        a.add_argument("--run", required=True)
    a = sub.add_parser("unlock")
    a.add_argument("--run", required=True)
    a.add_argument("--confirmed-process-stopped", action="store_true", required=True)
    args = p.parse_args()
    if args.command == "download":
        from .download import download_lcl

        print(
            json.dumps(download_lcl(args.out, extract=not args.archive_only), indent=2)
        )
    elif args.command == "calibrate":
        from .profile import calibrate

        print(
            json.dumps(
                calibrate(
                    args.config, args.out_config, args.report, args.steps, args.repeats
                ),
                indent=2,
            )
        )
    elif args.command == "bootstrap-pilot":
        from .bootstrap import bootstrap_pilot

        print(
            json.dumps(
                bootstrap_pilot(
                    args.config,
                    args.raw,
                    args.out_config,
                    args.report,
                    args.start,
                    args.end,
                    args.clients,
                    args.run,
                ),
                indent=2,
            )
        )
    elif args.command == "synthetic":
        from .data import synthetic

        synthetic(args.out, args.clients, args.days, args.seed)
    elif args.command == "prepare":
        from .data import prepare

        paths = []
        for pattern in args.csv:
            paths.extend(glob.glob(pattern))
        print(
            json.dumps(
                prepare(
                    paths,
                    args.out,
                    args.start,
                    args.end,
                    args.clients,
                    args.lookback,
                    args.min_coverage,
                    args.max_fill,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        from .engine import run_training

        run_training(args.config, args.run, args.resume, args.stop_after)
    elif args.command == "test":
        from .engine import test_run

        result = test_run(args.run, args.checkpoint, args.data)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "per_household"}, indent=2
            )
        )
    elif args.command == "profile":
        from .profile import benchmark

        print(json.dumps(benchmark(args.config, args.out), indent=2))
    elif args.command == "compress":
        from .profile import compression

        print(json.dumps(compression(args.run, args.out, args.repeats), indent=2))
    elif args.command == "analyze":
        from .reporting import analyze

        print(analyze(args.runs, args.out))
    elif args.command == "dashboard":
        from .reporting import dashboard

        dashboard(args.runs, args.port)
    elif args.command in ["seal", "verify"]:
        from .common import seal_run, verify_run

        print((seal_run if args.command == "seal" else verify_run)(args.run))
    else:
        lock = Path(args.run) / "RUNNING.lock"
        if lock.exists():
            lock.unlink()
            print("Removed stale lock")


if __name__ == "__main__":
    main()
