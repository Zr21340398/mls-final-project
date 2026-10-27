"""One command prepares real data and a calibrated pilot, without reading test results."""

from pathlib import Path
import os, uuid
from .common import load_config, read_json, write_json
from .download import download_lcl, exclusive_lock
from .data import prepare, load_dataset
from .profile import calibrate


def bootstrap_pilot(
    config="configs/pilot_cpu.json",
    raw="data/raw/lcl",
    out_config="configs/pilot_cpu.calibrated.json",
    report="reports/pilot_setup/profile.json",
    start="2013-01-01",
    end="2013-07-01",
    clients=12,
    run=None,
):
    import pandas as pd

    c = load_config(config)
    result = download_lcl(raw)
    extraction = read_json(Path(result["csv_dir"]) / "extraction.json")
    paths = sorted(Path(result["csv_dir"]) / f["file"] for f in extraction["files"])
    dataset = Path(c["data"])
    with exclusive_lock(dataset.with_name(dataset.name + ".prepare.lock")):
        if (dataset / "manifest.json").exists():
            manifest, _ = load_dataset(dataset)
            same_inputs = sorted(i["sha256"] for i in manifest["inputs"]) == sorted(
                f["sha256"] for f in extraction["files"]
            )
            if (
                manifest["synthetic"]
                or pd.Timestamp(manifest["start"]) != pd.Timestamp(start)
                or pd.Timestamp(manifest["end_exclusive"]) != pd.Timestamp(end)
                or manifest["lookback"] != 96
                or not same_inputs
            ):
                raise ValueError(
                    "Processed cache does not match the requested real-data setup; use a new data path"
                )
            receipt_path = dataset / "bootstrap.json"
            if (
                not receipt_path.exists()
                or read_json(receipt_path)["requested_clients"] != clients
            ):
                raise ValueError(
                    "Existing cohort lacks matching bootstrap provenance; use a new data path"
                )
        else:
            if dataset.exists() and any(dataset.iterdir()):
                raise FileExistsError(
                    "Unrecognized processed directory; choose a fresh data path"
                )
            # Publish the complete dataset only after manifest and receipt exist.
            # Interrupted staging directories remain for diagnosis; a retry uses
            # a new owned sibling and never mixes incomplete household arrays.
            stage = dataset.with_name(
                dataset.name + ".preparing-" + uuid.uuid4().hex[:8]
            )
            if stage.resolve().parent != dataset.resolve().parent:
                raise ValueError("Unexpected preparation staging path")
            manifest = prepare(paths, stage, start, end, clients=clients)
            write_json(
                stage / "bootstrap.json",
                {"requested_clients": clients, "archive_sha256": result["sha256"]},
            )
            if dataset.exists():
                dataset.rmdir()  # Only the explicitly requested, verified empty directory.
            os.replace(stage, dataset)
    if len(manifest["clients"]) < c["train"]["clients_per_round"]:
        raise ValueError(
            "Too few training-eligible households for configured cohort size; inspect manifest before changing selection"
        )
    if not any(e["counts"]["val"] for e in manifest["clients"]):
        raise ValueError("No validation windows; inspect requested period")
    calibrated = calibrate(config, out_config, report)
    summary = {
        "download": result,
        "data": str(dataset),
        "actual_clients": len(manifest["clients"]),
        "calibration": calibrated,
        "test_evaluated": False,
    }
    write_json(Path(report).parent / "setup.json", summary)
    if run:
        from .engine import run_training

        target = Path(run)
        if (target / "status.json").exists() and read_json(target / "status.json")[
            "status"
        ] in ["complete", "privacy_budget_exhausted"]:
            # A completed run is reused only when its saved configuration matches.
            if read_json(target / "config.json") != load_config(out_config):
                raise ValueError(
                    "Requested run already contains a different configuration"
                )
        else:
            run_training(out_config, run, resume=(target / "metadata.json").exists())
        summary["run"] = str(run)
        write_json(Path(report).parent / "setup.json", summary)
    return summary
