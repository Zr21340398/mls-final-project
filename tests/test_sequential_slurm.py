import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sequential_slurm', Path(__file__).resolve().parents[1] / 'scripts/sequential_slurm.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class SequentialSlurmTests(unittest.TestCase):
    def test_failure_stops_and_retry_resumes_before_next_job(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                Path('run.sbatch').write_text('#!/bin/bash\n')
                Path('plan.json').write_text(json.dumps([
                    {'config': 'one.json', 'run': 'runs/one'},
                    {'config': 'two.json', 'run': 'runs/two'},
                ]))
                submitted = []
                def fake_call(args):
                    if args[0] == 'sbatch':
                        submitted.append(args)
                        run = Path(args[args.index('--run') + 1])
                        run.mkdir(parents=True, exist_ok=True)
                        (run / 'metadata.json').write_text('{}')
                        (run / 'config.json').write_text('{}')
                        (run / 'status.json').write_text(json.dumps({'status': 'failed' if len(submitted) == 1 else 'complete'}))
                        return str(100 + len(submitted))
                    if args[0] == 'squeue':
                        return ''
                    jobid = args[args.index('-j') + 1]
                    return f'{jobid}|{"TIMEOUT" if jobid == "101" else "COMPLETED"}|0:0'
                argv = ['controller', 'run', '--plan', 'plan.json', '--state-dir', 'state', '--batch-script', 'run.sbatch']
                with patch.object(controller, 'call', side_effect=fake_call), patch('sys.argv', argv):
                    with self.assertRaisesRegex(RuntimeError, 'Training stopped'):
                        controller.main()
                self.assertEqual(len(submitted), 1)
                state = json.loads(Path('state/state.json').read_text())
                self.assertEqual([j['phase'] for j in state['jobs']], ['failed', 'pending'])
                with patch.object(controller, 'call', side_effect=fake_call), patch('sys.argv', argv + ['--retry-failed']):
                    controller.main()
                self.assertIn('--resume', submitted[1])
                self.assertNotIn('--resume', submitted[2])
                state = json.loads(Path('state/state.json').read_text())
                self.assertEqual([j['phase'] for j in state['jobs']], ['completed', 'completed'])
            finally:
                os.chdir(old)

    def test_uncertain_submission_never_repeats(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                Path('run.sbatch').write_text('')
                Path('plan.json').write_text('[{"config":"x","run":"y"}]')
                argv = ['controller', 'run', '--plan', 'plan.json', '--state-dir', 'state', '--batch-script', 'run.sbatch']
                with patch.object(controller, 'call', side_effect=OSError('connection lost')), patch('sys.argv', argv):
                    with self.assertRaises(OSError):
                        controller.main()
                with patch.object(controller, 'call') as mock_call, patch('sys.argv', argv):
                    with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                        controller.main()
                    mock_call.assert_not_called()
            finally:
                os.chdir(old)


if __name__ == '__main__':
    unittest.main()
