import copy, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from energyfl.common import read_json, write_json, seal_run, verify_run
from energyfl.data import synthetic, prepare, load_dataset
from energyfl.models import Forecaster, flat, assign, setup
from energyfl.privacy import epsilon, clip, masked_sum, aggregate
from energyfl.simulation import make_trace, select, outcome
from energyfl.engine import run_training, test_run, load_checkpoint, committed
from energyfl.profile import compression
from energyfl.reporting import analyze

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / "raw.csv"
        synthetic(cls.raw, clients=4, days=14)
        cls.data = cls.root / "processed"
        cls.manifest = prepare([cls.raw], cls.data, "2013-01-01", "2013-01-15", 4)
        cls.base = read_json(ROOT / "configs/smoke.json")
        cls.base["data"] = str(cls.data)
        cls.base["model"]["hidden"] = 4
        cls.base["train"].update(rounds=3, local_steps=1, batch_size=8, threads=1)
        cls.base["simulation"].update(dropout=0.0, availability=1.0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, name, c):
        p = self.root / (name + ".json")
        write_json(p, c)
        return p

    def test_chronological_no_leakage(self):
        m, hs = load_dataset(self.data)
        a = pd.Timestamp(m["train_end_exclusive"]).value
        b = pd.Timestamp(m["val_end_exclusive"]).value
        for h in hs:
            for split in ["train", "val", "test"]:
                t = h.arr[split]
                ts = h.arr["timestamps"]
                self.assertTrue((t >= 96).all())
                if split == "train":
                    self.assertTrue((ts[t + 1] < a).all())
                if split == "val":
                    self.assertTrue(((ts[t] >= a) & (ts[t + 1] < b)).all())
                if split == "test":
                    self.assertTrue((ts[t] >= b).all())
        modified = pd.read_csv(self.raw)
        mask = pd.to_datetime(modified.DateTime) >= pd.Timestamp(
            m["train_end_exclusive"]
        )
        modified.loc[mask, "KWH/hh (per half hour) "] *= 1000
        p = self.root / "future_changed.csv"
        modified.to_csv(p, index=False)
        out = self.root / "changed"
        prepare([p], out, "2013-01-01", "2013-01-15", 4)
        _, hs2 = load_dataset(out)
        for left, right in zip(hs, hs2):
            self.assertEqual(left.id, right.id)
            self.assertEqual(float(left.arr["mean"]), float(right.arr["mean"]))
            x, y, _ = left.batch("train", np.arange(10))
            xx, yy, _ = right.batch("train", np.arange(10))
            np.testing.assert_array_equal(x, xx)
            np.testing.assert_array_equal(y, yy)

    def test_missing_targets_and_long_input_gaps_excluded(self):
        frame = pd.read_csv(self.raw)
        cid = frame.LCLid.iloc[0]
        ix = frame.index[frame.LCLid == cid][150:160]
        frame.loc[ix, "KWH/hh (per half hour) "] = np.nan
        raw = self.root / "missing.csv"
        frame.to_csv(raw, index=False)
        out = self.root / "missing"
        prepare([raw], out, "2013-01-01", "2013-01-15", 4, min_coverage=0.8)
        _, hs = load_dataset(out)
        h = next(h for h in hs if h.id == cid)
        for split in ["train", "val", "test"]:
            t = h.arr[split]
            self.assertFalse(np.isin(t, np.arange(149, 160)).any())
            if len(t):
                self.assertTrue(np.isfinite(h.batch(split, np.arange(len(t)))[0]).all())

    def test_masking_and_accounting(self):
        rng = np.random.default_rng(3)
        u = [rng.normal(size=30) for _ in range(4)]
        np.testing.assert_allclose(masked_sum(u, 1), np.sum(u, axis=0), atol=1e-12)
        self.assertLessEqual(np.linalg.norm(clip(u[0], 0.2)), 0.2000000001)
        self.assertEqual(epsilon(0, 8, 1e-5), 0.0)
        self.assertGreater(epsilon(4, 8, 1e-5), epsilon(3, 8, 1e-5))
        c = copy.deepcopy(self.base)
        c["privacy"]["clip_updates"] = False
        avg, _ = aggregate([np.ones(3), np.ones(3)], [1, 1], 4, c, 1)
        np.testing.assert_allclose(avg, np.full(3, 0.5))

    def test_backup_integrity(self):
        run = self.root / "integrity_fixture"
        write_json(run / "status.json", {"status": "complete"})
        (run / "model.txt").write_text("original")
        seal_run(run)
        self.assertEqual(verify_run(run)["status"], "verified")
        (run / "model.txt").write_text("modified")
        with self.assertRaises(ValueError):
            verify_run(run)

    def test_assignment_owns_tensor_storage(self):
        setup(1, "cpu", 1)
        model = Forecaster(hidden=4)
        v = flat(model)
        saved = v.copy()
        assign(model, v)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        np.testing.assert_array_equal(v, saved)

    def test_trace_independent_of_scheduler_and_failure(self):
        ids = ["a", "b", "c", "d"]
        c = copy.deepcopy(self.base)
        d = copy.deepcopy(c)
        d["train"]["scheduler"] = "fastest"
        self.assertEqual(make_trace(c, ids, 20), make_trace(d, ids, 20))
        c["simulation"]["dropout"] = 1.0
        rows = make_trace(c, ids, 20)[0]
        survivors, elapsed, ok = outcome(select(rows, {k: 0 for k in ids}, 1, c), c)
        self.assertFalse(ok)
        self.assertFalse(survivors)
        self.assertEqual(elapsed, c["simulation"]["deadline_s"])

    def test_resume_exact_and_test_once(self):
        cfg = self.config("resume", self.base)
        a = self.root / "runs" / "continuous"
        b = self.root / "runs" / "resumed"
        run_training(cfg, a)
        run_training(cfg, b, stop_after=1)
        run_training(cfg, b, resume=True)
        ca = load_checkpoint(a / "rounds/0003/checkpoint.pt")
        cb = load_checkpoint(b / "rounds/0003/checkpoint.pt")
        for key in ca["model"]:
            self.assertTrue(torch.equal(ca["model"][key], cb["model"][key]))
        self.assertEqual(ca["virtual_s"], cb["virtual_s"])
        self.assertEqual(ca["participation"], cb["participation"])
        result = test_run(b)
        self.assertTrue(np.isfinite(result["macro_mae_kwh"]))
        self.assertTrue((b / "test/predictions.csv.gz").exists())
        with self.assertRaises(FileExistsError):
            test_run(b)
        changed = copy.deepcopy(self.base)
        changed["train"]["lr"] = 0.02
        with self.assertRaises(ValueError):
            run_training(self.config("changed_cfg", changed), b, resume=True)
        out = self.root / "summary"
        analyze(self.root / "runs", out)
        self.assertTrue((out / "report.html").exists())

    def test_baselines_and_dp_budget(self):
        for method in ["centralized", "local", "seasonal"]:
            c = copy.deepcopy(self.base)
            c["train"].update(method=method, rounds=1)
            out = self.root / "baseline" / method
            run_training(self.config(method, c), out)
            self.assertTrue(np.isfinite(test_run(out)["macro_mae_kwh"]))
        c = copy.deepcopy(self.base)
        c["privacy"].update(enabled=True, deterministic_noise=True, epsilon_cap=1.0)
        c["simulation"]["aggregation"] = "masking_sim"
        out = self.root / "dp"
        run_training(self.config("dp", c), out)
        self.assertEqual(
            read_json(out / "status.json")["status"], "privacy_budget_exhausted"
        )
        self.assertEqual(len(committed(out)), 2)
        with self.assertRaises(ValueError):
            test_run(out)
        self.assertTrue(np.isfinite(test_run(out, "last")["macro_mae_kwh"]))
        comp = compression(out, self.root / "compression", repeats=10)
        self.assertIn("int8", comp["variants"])

    def test_no_model_update_or_budget_when_all_drop(self):
        c = copy.deepcopy(self.base)
        c["simulation"]["dropout"] = 1.0
        c["privacy"]["enabled"] = True
        c["train"]["rounds"] = 1
        out = self.root / "all_drop"
        run_training(self.config("all_drop", c), out)
        met = read_json(out / "rounds/0001/metrics.json")
        self.assertEqual(met["dp_releases"], 0)
        self.assertFalse(met["update_applied"])


if __name__ == "__main__":
    unittest.main()
