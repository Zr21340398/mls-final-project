"""Portable CSV/HTML report and a localhost live dashboard; no external service."""

import csv, html
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from .common import read_json, write_json, fingerprint
from .engine import committed


def rows_for(root):
    result = []
    for config in sorted(Path(root).glob("*/config.json")):
        run = config.parent
        if not (run / "metadata.json").exists():
            continue
        c = read_json(config)
        meta = read_json(run / "metadata.json")
        rounds = committed(run)
        if not rounds:
            continue
        metrics = [read_json(r / "metrics.json") for r in rounds]
        last = metrics[-1]
        vals = [m["validation"]["macro_mae_kwh"] for m in metrics if m["validation"]]
        group = dict(c)
        group.pop("seed")
        group.pop("data")
        group["dataset_fingerprint"] = meta["dataset_fingerprint"]
        test = (
            read_json(run / "test/metrics.json")
            if (run / "test/metrics.json").exists()
            else {}
        )
        result.append(
            {
                "run": run.name,
                "seed": c["seed"],
                "method": c["train"]["method"],
                "scheduler": c["train"]["scheduler"],
                "dp": c["privacy"]["enabled"],
                "aggregation": c["simulation"]["aggregation"],
                "synthetic": meta["synthetic_data"],
                "group": fingerprint(group)[:16],
                "rounds": last["round"],
                "best_val_mae_kwh": min(vals) if vals else None,
                "test_mae_kwh": test.get("macro_mae_kwh"),
                "test_rmse_kwh": test.get("macro_rmse_kwh"),
                "test_p90_mae_kwh": test.get("p90_household_mae_kwh"),
                "virtual_s": last["virtual_elapsed_s"],
                "wall_s": last["measured_wall_s"],
                "modeled_bytes": sum(m["modeled_round_bytes"] for m in metrics),
                "failed_rounds": sum(not m["update_applied"] for m in metrics)
                if c["train"]["method"] == "federated"
                else 0,
                "coverage": last["participation_coverage"],
                "epsilon": last["epsilon"],
                "curve": [
                    (m["round"], m["validation"]["macro_mae_kwh"])
                    for m in metrics
                    if m["validation"]
                ],
            }
        )
    return result


def plot(series, width=700, height=220):
    if not series:
        return ""
    xmax = max(p[0] for p in series)
    ys = [p[1] for p in series]
    low = min(ys)
    high = max(ys)
    span = max(high - low, 1e-6)
    points = " ".join(
        f"{40 + (x - 1) / max(xmax - 1, 1) * (width - 60):.1f},{20 + (high - y) / span * (height - 55):.1f}"
        for x, y in series
    )
    return f'<svg viewBox="0 0 {width} {height}"><path d="M40 15 V{height - 30} H{width - 10}" stroke="#aaa" fill="none"/><polyline points="{points}" stroke="#147d86" stroke-width="2" fill="none"/><text x="45" y="15" font-size="12">Validation MAE (kWh): {low:.4g} - {high:.4g}</text><text x="45" y="{height - 5}" font-size="12">Round 1 - {xmax}</text></svg>'


def page(rows, refresh=False):
    out = [
        '<!doctype html><html><meta charset="utf-8"><title>EnergyFL experiments</title>',
        '<meta http-equiv="refresh" content="10">' if refresh else "",
        "<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;color:#17344a}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}article{border:1px solid #ddd;padding:18px;margin:22px 0}svg{max-width:700px}code{color:#147d86}</style>",
        "<h1>Household forecasting experiment monitor</h1><p>Measured wall time and modeled deployment time are separate. Synthetic data and masking simulations do not establish real-world accuracy or security.</p>",
    ]
    for r in rows:
        out.append(f"<article><h2>{html.escape(r['run'])}</h2><table>")
        for k in [
            "method",
            "scheduler",
            "synthetic",
            "dp",
            "aggregation",
            "rounds",
            "best_val_mae_kwh",
            "test_mae_kwh",
            "wall_s",
            "virtual_s",
            "modeled_bytes",
            "failed_rounds",
            "coverage",
            "epsilon",
        ]:
            v = r[k]
            v = f"{v:.6g}" if isinstance(v, float) else str(v)
            out.append(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>")
        out.append("</table>" + plot(r["curve"]) + "</article>")
    if not rows:
        out.append("<p>No committed rounds yet.</p>")
    return "".join(out) + "</html>"


def analyze(root, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_for(root)
    if not rows:
        raise ValueError("No runs with committed rounds")
    keys = [k for k in rows[0] if k != "curve"]
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    grouped = []
    for group in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == group and r["test_mae_kwh"] is not None]
        if not rs:
            continue
        if len({r["seed"] for r in rs}) != len(rs):
            raise ValueError("Duplicate seeds in comparison group " + group)
        values = np.array([r["test_mae_kwh"] for r in rs])
        grouped.append(
            {
                "group": group,
                "n_seeds": len(rs),
                "seeds": [r["seed"] for r in rs],
                "test_mae_mean": float(values.mean()),
                "test_mae_std": float(values.std(ddof=1)) if len(rs) > 1 else None,
            }
        )
    write_json(out / "seed_summary.json", grouped)
    (out / "report.html").write_text(page(rows), encoding="utf-8")
    return out


def dashboard(root, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            data = page(rows_for(root), refresh=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"Dashboard: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
