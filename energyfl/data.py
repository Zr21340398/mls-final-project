"""Public LCL ingestion, train-only normalization, and lazy forecasting windows."""

from pathlib import Path
import numpy as np
import pandas as pd
from .common import digest, fingerprint, read_json, write_json

SOURCE = "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d"


def synthetic(out, clients=8, days=30, seed=7):
    """Smoke-test fixture, deliberately NOT a substitute for empirical LCL results."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2013-01-01", periods=48 * days, freq="30min")
    frames = []
    for i in range(clients):
        phase = rng.uniform(-1, 1)
        x = np.arange(len(times))
        load = np.maximum(
            0,
            0.15
            + 0.02 * i
            + 0.12 * np.sin(x * 2 * np.pi / 48 + phase)
            + rng.normal(0, 0.025, len(x)),
        )
        frames.append(
            pd.DataFrame(
                {
                    "LCLid": f"SYN{i:04}",
                    "stdorToU": "Std",
                    "DateTime": times,
                    "KWH/hh (per half hour) ": load,
                }
            )
        )
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(p, index=False)


def chunks(path, start, end):
    # The full archive is ~10 GB uncompressed. Never read it as one DataFrame.
    for df in pd.read_csv(path, chunksize=200_000, low_memory=False):
        df.columns = df.columns.str.strip()
        need = {"LCLid", "DateTime", "KWH/hh (per half hour)", "stdorToU"}
        if not need.issubset(df.columns):
            raise ValueError(f"{path}: expected original LCL columns {need}")
        df = df[df.stdorToU.astype(str).str.strip().eq("Std")].copy()
        df["time"] = pd.to_datetime(df.DateTime, errors="coerce")
        df["value"] = pd.to_numeric(df["KWH/hh (per half hour)"], errors="coerce")
        df = df[
            (df.time >= start)
            & (df.time < end)
            & df.value.ge(0)
            & np.isfinite(df.value)
        ]
        df = df[df.time.dt.minute.isin([0, 30]) & df.time.dt.second.eq(0)]
        yield df[["LCLid", "time", "value"]]


def prepare(
    paths, out, start, end, clients=60, lookback=96, min_coverage=0.9, max_fill=2
):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Processed directory must be empty; use a new versioned path"
        )
    paths = sorted({Path(p).resolve() for p in paths})
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if (
        not paths
        or end <= start
        or clients < 1
        or lookback < 48
        or max_fill < 0
        or not 0 < min_coverage <= 1
    ):
        raise ValueError("Invalid preparation arguments")
    grid = pd.date_range(start, end, freq="30min", inclusive="left")
    n = len(grid)
    if n < lookback + 30:
        raise ValueError("Time range too short")
    cut1, cut2 = int(n * 0.7), int(n * 0.85)
    # First pass ranks only TRAINING observations. Validation/test completeness
    # must never decide which household enters the cohort.
    counts = {}
    for path in paths:
        for df in chunks(path, start, end):
            for key, num in df[df.time < grid[cut1]].groupby("LCLid").size().items():
                counts[str(key)] = counts.get(str(key), 0) + int(num)
    candidates = sorted(counts, key=lambda k: (-counts[k], k))[
        : max(clients * 3, clients)
    ]
    pieces = []
    for path in paths:
        for df in chunks(path, start, end):
            chosen = df[df.LCLid.astype(str).isin(candidates)]
            if len(chosen):
                pieces.append(chosen)
    if not pieces:
        raise ValueError("No eligible Std tariff records in requested period")
    records = pd.concat(pieces, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for cid in candidates:
        rows = records[records.LCLid.astype(str).eq(cid)]
        # Source timestamps are treated as naive source clock time. Duplicate
        # half-hours (including DST ambiguity) are averaged and counted in audit.
        series = rows.groupby("time").value.mean().reindex(grid)
        observed = series.notna().to_numpy()
        coverage = float(observed[:cut1].mean())
        if coverage < min_coverage:
            continue
        filled = series.ffill(limit=max_fill) if max_fill else series
        values = filled.to_numpy(dtype=np.float32)
        raw = series.to_numpy(dtype=np.float32)
        mean = float(np.nanmean(raw[:cut1]))
        std = max(float(np.nanstd(raw[:cut1])), 1e-3)
        hour = grid.hour.to_numpy() + grid.minute.to_numpy() / 60
        dow = grid.dayofweek.to_numpy()
        x = np.column_stack(
            [
                (values - mean) / std,
                ~observed,
                np.sin(hour * 2 * np.pi / 24),
                np.cos(hour * 2 * np.pi / 24),
                np.sin(dow * 2 * np.pi / 7),
                np.cos(dow * 2 * np.pi / 7),
            ]
        ).astype(np.float32)
        indices = {k: [] for k in ["train", "val", "test"]}
        for t in range(lookback, n - 1):
            # Both prediction intervals must be in one split. Context can come
            # from the previous split because those readings are already observed.
            if (
                not observed[t : t + 2].all()
                or not np.isfinite(x[t - lookback : t]).all()
            ):
                continue
            split = (
                "train"
                if t + 1 < cut1
                else "val"
                if cut1 <= t and t + 1 < cut2
                else "test"
                if t >= cut2
                else None
            )
            if split:
                indices[split].append(t)
        if not indices["train"]:
            continue
        safe = f"client_{len(meta):04}.npz"
        np.savez_compressed(
            out / safe,
            x=x,
            raw=raw,
            timestamps=grid.astype("int64").to_numpy(),
            mean=mean,
            std=std,
            **{k: np.array(v, dtype=np.int64) for k, v in indices.items()},
        )
        meta.append(
            {
                "id": cid,
                "file": safe,
                "sha256": digest(out / safe),
                "train_coverage": coverage,
                "duplicate_rows": int(len(rows) - rows.time.nunique()),
                "counts": {k: len(v) for k, v in indices.items()},
            }
        )
        if len(meta) == clients:
            break
    if not meta:
        raise ValueError(
            "No training-eligible households. Inspect date range and coverage threshold"
        )
    manifest = {
        "schema": 1,
        "source": SOURCE,
        "synthetic": all(m["id"].startswith("SYN") for m in meta),
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in paths],
        "start": str(start),
        "end_exclusive": str(end),
        "train_end_exclusive": str(grid[cut1]),
        "val_end_exclusive": str(grid[cut2]),
        "lookback": lookback,
        "horizon": 2,
        "features": 6,
        "max_forward_fill": max_fill,
        "selection": "training coverage only; no test-based selection",
        "timezone_policy": "naive source clock; duplicate timestamps averaged",
        "clients": meta,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(out / "manifest.json", manifest)
    return manifest


class Household:
    def __init__(self, root, entry, lookback):
        self.id = entry["id"]
        self.lookback = lookback
        with np.load(Path(root) / entry["file"]) as z:
            self.arr = {k: z[k] for k in z.files}

    def __len__(self):
        return len(self.arr["train"])

    def batch(self, split, positions):
        t = self.arr[split][positions]
        x = np.stack([self.arr["x"][i - self.lookback : i] for i in t])
        # The next-hour target is a SUM of two half-hour kWh readings.
        y = (self.arr["raw"][t] + self.arr["raw"][t + 1] - 2 * self.arr["mean"]) / (
            2 * self.arr["std"]
        )
        return x, y.astype(np.float32), t

    def inverse(self, y):
        return np.asarray(y) * 2 * self.arr["std"] + 2 * self.arr["mean"]


def load_dataset(path):
    manifest = read_json(Path(path) / "manifest.json")
    content = dict(manifest)
    fp = content.pop("fingerprint")
    if fingerprint(content) != fp:
        raise ValueError("Manifest fingerprint mismatch")
    for e in manifest["clients"]:
        if digest(Path(path) / e["file"]) != e["sha256"]:
            raise ValueError("Processed data modified: " + e["file"])
    return manifest, [
        Household(path, e, manifest["lookback"]) for e in manifest["clients"]
    ]
