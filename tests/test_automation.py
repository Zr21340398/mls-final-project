"""Bounded HTTP fixtures exercise download behavior without fetching 0.8 GB in tests."""

import copy
import io
import re
import ssl
import urllib.error
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from energyfl.common import read_json, write_json, digest, validate_calibration
from energyfl.download import _transfer, extract_archive, verified_ssl_context
from energyfl.data import synthetic, prepare, load_dataset
from energyfl.profile import calibrate
from energyfl.engine import run_training
from energyfl.bootstrap import bootstrap_pilot

ROOT = Path(__file__).resolve().parents[1]


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("folder/a.csv", "id,value\na,1\n")
            z.writestr("folder/b.csv", "id,value\nb,2\n")
        cls.payload = stream.getvalue()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = cls.payload
                start = 0
                requested = self.headers.get("Range")
                if requested and self.path != "/ignore-range":
                    start = int(re.fullmatch(r"bytes=(\d+)-", requested)[1])
                    self.send_response(206)
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                    )
                else:
                    self.send_response(200)
                self.send_header("ETag", '"fixture-v1"')
                self.send_header("Content-Length", str(len(payload) - start))
                self.end_headers()
                self.wfile.write(payload[start:])

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_extract_cache_and_corruption(self):
        archive = self.root / "archive.zip"
        receipt = _transfer(self.base + "/archive", archive)
        self.assertEqual(receipt["sha256"], digest(archive))
        self.assertEqual(
            _transfer(self.base + "/archive", archive)["sha256"], receipt["sha256"]
        )
        out = self.root / "csv"
        manifest = extract_archive(archive, out)
        self.assertEqual(len(manifest["files"]), 2)
        self.assertTrue((out / "a.csv").exists())
        self.assertEqual(
            extract_archive(archive, out)["archive_sha256"], digest(archive)
        )
        (out / "a.csv").write_text("changed")
        with self.assertRaises(ValueError):
            extract_archive(archive, out)

    def test_tls_verification_stays_enabled(self):
        context = verified_ssl_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_certificate_failure_is_not_retried(self):
        error = urllib.error.URLError(
            ssl.SSLCertVerificationError(1, "fixture untrusted issuer")
        )
        with patch(
            "energyfl.download.urllib.request.urlopen", side_effect=error
        ) as request:
            with self.assertRaisesRegex(RuntimeError, "system trust store"):
                _transfer("https://example.invalid/data.zip", self.root / "tls.zip")
            self.assertEqual(request.call_count, 1)
        self.assertFalse((self.root / "tls.zip").exists())

    def test_range_resume_and_ignored_range_restart(self):
        for path in ["/archive", "/ignore-range"]:
            folder = self.root / path.strip("/")
            folder.mkdir()
            archive = folder / "archive.zip"
            url = self.base + path
            archive.with_suffix(".part").write_bytes(self.payload[:40])
            write_json(
                archive.with_suffix(".partial.json"),
                {"url": url, "etag": '"fixture-v1"'},
            )
            _transfer(url, archive)
            self.assertEqual(archive.read_bytes(), self.payload)

    def test_wrong_hash_and_unsafe_archive_rejected(self):
        with self.assertRaises(ValueError):
            _transfer(
                self.base + "/archive", self.root / "bad.zip", expected_sha256="0" * 64
            )
        for name in ["../escape.csv", "/absolute.csv", "C:/drive.csv"]:
            archive = self.root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(name, "x")
            with self.assertRaises(ValueError):
                extract_archive(archive, self.root / ("out" + str(len(name))))
        self.assertFalse((self.root / "escape.csv").exists())

    def test_insufficient_disk_fails_before_extract(self):
        archive = self.root / "source.zip"
        archive.write_bytes(self.payload)

        class FullDisk:
            free = 1

        with patch("energyfl.download.shutil.disk_usage", return_value=FullDisk()):
            with self.assertRaises(OSError):
                extract_archive(archive, self.root / "csv")


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw.csv"
        synthetic(self.raw, clients=2, days=14)
        # Public test fixture with ordinary IDs exercises the real-data bootstrap
        # branch. This is still a generated fixture, not measured LCL data.
        frame = pd.read_csv(self.raw)
        frame.LCLid = frame.LCLid.str.replace("SYN", "FIX")
        frame.to_csv(self.raw, index=False)
        self.base = read_json(ROOT / "configs/smoke.json")
        self.base["data"] = str(self.root / "processed")
        self.base["model"]["hidden"] = 4
        self.base["train"].update(
            rounds=1, local_steps=1, batch_size=4, threads=1, clients_per_round=2
        )
        self.base["simulation"].update(min_survivors=1, dropout=0.0)
        self.config = self.root / "base.json"
        write_json(self.config, self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def test_calibration_provenance_immutable_template_and_training(self):
        prepare([self.raw], self.base["data"], "2013-01-01", "2013-01-15", 2)
        original = digest(self.config)
        out = self.root / "calibrated.json"
        report = self.root / "profile.json"
        info = calibrate(self.config, out, report, steps=1, repeats=2)
        self.assertFalse(info["reused"])
        self.assertEqual(original, digest(self.config))
        c = read_json(out)
        self.assertGreater(c["simulation"]["reference_step_s"], 0)
        self.assertEqual(
            c["simulation"]["calibration"]["report_sha256"], digest(report)
        )
        self.assertTrue(calibrate(self.config, out, report)["reused"])
        run = self.root / "run"
        run_training(out, run)
        self.assertEqual(digest(run / "calibration_report.json"), digest(report))
        m, _ = load_dataset(self.base["data"])
        changed = copy.deepcopy(c)
        changed["train"]["batch_size"] *= 2
        with self.assertRaises(ValueError):
            validate_calibration(changed, m)
        report.write_text("{}")
        with self.assertRaises(ValueError):
            validate_calibration(c, m)
        # Resume may use the archived calibration evidence even if the source
        # report has since disappeared/changed.
        run_training(out, run, resume=True)

    def test_bootstrap_reuses_verified_data_and_never_tests(self):
        csvdir = self.root / "csv"
        csvdir.mkdir()
        copy_path = csvdir / "fixture.csv"
        copy_path.write_bytes(self.raw.read_bytes())
        write_json(
            csvdir / "extraction.json",
            {"files": [{"file": "fixture.csv", "sha256": digest(copy_path)}]},
        )
        download = {"csv_dir": str(csvdir), "sha256": "fixture-archive"}
        out = self.root / "calibrated.json"
        report = self.root / "reports/profile.json"
        run = self.root / "run"
        with (
            patch("energyfl.bootstrap.download_lcl", return_value=download),
            patch(
                "energyfl.bootstrap.calibrate",
                side_effect=lambda *a: calibrate(*a, steps=1, repeats=1),
            ),
        ):
            first = bootstrap_pilot(
                self.config,
                self.root / "raw",
                out,
                report,
                "2013-01-01",
                "2013-01-15",
                2,
                run,
            )
            second = bootstrap_pilot(
                self.config,
                self.root / "raw",
                out,
                report,
                "2013-01-01",
                "2013-01-15",
                2,
                run,
            )
        self.assertEqual(first["actual_clients"], 2)
        self.assertTrue(second["calibration"]["reused"])
        self.assertFalse((run / "test").exists())
        self.assertEqual(len(list((run / "rounds").glob("[0-9]*"))), 1)


if __name__ == "__main__":
    unittest.main()
