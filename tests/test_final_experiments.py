import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from energyfl.common import read_json, write_json
from energyfl.data import synthetic, prepare
from energyfl.engine import run_training

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('final_experiments', ROOT / 'scripts/final_experiments.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class FinalExperimentsTests(unittest.TestCase):
    def test_frozen_plan_test_once_and_compression(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw, data = root / 'raw.csv', root / 'data'
            synthetic(raw, clients=4, days=14)
            prepare([raw], data, '2013-01-01', '2013-01-15', 4)
            cfg = copy.deepcopy(read_json(ROOT / 'configs/smoke.json'))
            cfg['data'] = str(data)
            cfg['model']['hidden'] = 4
            cfg['train'].update(rounds=2, local_steps=1, batch_size=8, threads=1)
            cfg['simulation'].update(dropout=0.0, availability=1.0)
            config, run = root / 'config.json', root / 'run'
            write_json(config, cfg)
            run_training(config, run)
            jobs, plan = root / 'jobs.json', root / 'plan.json'
            write_json(jobs, [{'config': str(config), 'run': str(run)}])
            workflow.prepare_tests(SimpleNamespace(jobs=jobs, out=plan, runs=None))
            self.assertFalse((run / 'test').exists())
            with self.assertRaisesRegex(ValueError, 'confirm'):
                workflow.run_tests(SimpleNamespace(plan=plan, data=None, confirm_frozen=False))
            # The compression workflow uses validation data and never creates test output.
            out = root / 'compression'
            workflow.compress(SimpleNamespace(jobs=jobs, runs=['run'], out=out, repeats=10, tolerance=0.05))
            self.assertTrue((out / 'variants.csv').is_file())
            self.assertTrue((out / 'assignments.csv').is_file())
            self.assertFalse((run / 'test').exists())
            workflow.run_tests(SimpleNamespace(plan=plan, data=None, confirm_frozen=True))
            first = (run / 'test/metrics.json').read_bytes()
            workflow.run_tests(SimpleNamespace(plan=plan, data=None, confirm_frozen=True))
            self.assertEqual(first, (run / 'test/metrics.json').read_bytes())
            summary = root / 'summary'
            workflow.summarize_tests(SimpleNamespace(plan=plan, out=summary))
            self.assertTrue((summary / 'test_groups.csv').is_file())
            changed = read_json(run / 'config.json')
            changed['seed'] = 99
            write_json(run / 'config.json', changed)
            with self.assertRaisesRegex(ValueError, 'changed'):
                workflow.run_tests(SimpleNamespace(plan=plan, data=None, confirm_frozen=True))

    def test_select_rejects_missing_and_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs = Path(folder) / 'jobs.json'
            jobs.write_text(json.dumps([{'run': 'a'}, {'run': 'a'}]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                workflow.selected_jobs(jobs, None)
            with self.assertRaisesRegex(ValueError, 'missing'):
                workflow.selected_jobs(jobs, ['b'])


if __name__ == '__main__':
    unittest.main()
