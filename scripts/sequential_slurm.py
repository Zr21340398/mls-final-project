"""Submit one Slurm training job at a time and persist its progress.

Run the controller inside a CPU Slurm allocation.  An interrupted submission
is deliberately marked ambiguous: never guess whether sbatch accepted a job.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def call(args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def slurm_state(jobid):
    # squeue covers active jobs; sacct is authoritative for terminal outcomes.
    try:
        active = call(['squeue', '-h', '-j', str(jobid), '-o', '%T'])
        if active:
            active_state = active.splitlines()[0].strip().upper().split()[0].rstrip('+')
            if active_state not in TERMINAL:
                return active_state
            # Wait for sacct's final exit code before accepting completion.
        records = call(['sacct', '-n', '-X', '-P', '-j', str(jobid),
                        '--format=JobIDRaw,State,ExitCode'])
        for line in records.splitlines():
            cols = line.split('|')
            if len(cols) >= 3 and cols[0] == str(jobid):
                state = cols[1].strip().upper().split()[0].rstrip('+')
                if state == 'COMPLETED' and cols[2].strip() != '0:0':
                    return 'FAILED'
                return state
    except (OSError, subprocess.CalledProcessError):
        pass
    return 'UNKNOWN'


def write_state(path, state):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def event(path, message):
    line = f'{dt.datetime.now(dt.timezone.utc).isoformat()} {message}'
    print(line, flush=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
        stream.flush()


def read_run_status(run):
    path = Path(run) / 'status.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8')).get('status')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'status'])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--batch-script', default='hpc/run.sbatch')
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('--interval must be at least 10 seconds')
    plan_path = Path(args.plan).resolve()
    directory = Path(args.state_dir).resolve()
    state_path, log_path, lock = directory / 'state.json', directory / 'events.log', directory / 'controller.lock'
    plan_bytes = plan_path.read_bytes()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, list) or not plan or any(not isinstance(j, dict) or not j.get('config') or not j.get('run') for j in plan):
        raise ValueError('Plan must contain jobs with config and run paths')
    if len({j['run'] for j in plan}) != len(plan):
        raise ValueError('Plan contains duplicate run paths')
    if args.action == 'status':
        if not state_path.exists():
            print('No controller state yet; no job has been recorded by this controller.')
            return
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['plan_sha256'] != plan_hash:
            raise ValueError('Plan changed since this controller was created')
        for i, job in enumerate(state['jobs'], 1):
            jobid = job.get('jobid')
            live = slurm_state(jobid) if jobid and job['phase'] in ('submitted', 'failed') else '-'
            print(f"{i:2}/{len(plan)} {job['phase']:11} job={str(jobid or '-'):10} slurm={live:18} {job['run']}")
        print('Event log:', log_path)
        return
    if not Path(args.batch_script).is_file():
        raise FileNotFoundError(args.batch_script)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Controller lock exists: {lock}. Verify the previous controller stopped before removing it.')
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state['plan_sha256'] != plan_hash:
                raise ValueError('Plan changed; create a new state directory')
            if state['batch_script'] != str(Path(args.batch_script).resolve()):
                raise ValueError('Batch script path changed; inspect the original state')
        else:
            state = {'plan_sha256': plan_hash, 'batch_script': str(Path(args.batch_script).resolve()),
                     'jobs': [dict(j, phase='pending', jobid=None, attempts=[]) for j in plan]}
            write_state(state_path, state)
        event(log_path, 'Controller started')
        for index, job in enumerate(state['jobs'], 1):
            if job['phase'] == 'completed':
                continue
            if job['phase'] == 'submitting':
                raise RuntimeError('Submission outcome is ambiguous. Inspect Slurm and state.json before recovery; automatic resubmission is disabled.')
            if job['phase'] == 'failed':
                if not args.retry_failed:
                    raise RuntimeError('Previous job failed; inspect it before using --retry-failed')
                if slurm_state(job['jobid']) not in TERMINAL:
                    raise RuntimeError('Previous job is not confirmed terminal by Slurm')
                job['phase'] = 'pending'
                write_state(state_path, state)
            if job['phase'] == 'pending':
                run = Path(job['run'])
                # Never silently reuse a directory that was not initialized by training.
                resume = (run / 'metadata.json').is_file() and (run / 'config.json').is_file()
                if run.exists() and any(run.iterdir()) and not resume:
                    raise RuntimeError(f'Run directory exists without complete metadata: {run}')
                if resume and not job['attempts']:
                    raise RuntimeError(f'Existing run was not submitted by this controller: {run}')
                cmd = ['sbatch', '--parsable', args.batch_script, '-m', 'energyfl', 'train',
                       '--config', job['config'], '--run', job['run']]
                if resume:
                    cmd.append('--resume')
                job['phase'] = 'submitting'
                write_state(state_path, state)
                event(log_path, f'{index}/{len(plan)} submitting {job["run"]}, resume={resume}')
                answer = call(cmd).split(';')[0]
                if not answer.isdigit():
                    raise RuntimeError(f'Unexpected sbatch response: {answer!r}')
                job.update(phase='submitted', jobid=answer)
                job['attempts'].append(answer)
                write_state(state_path, state)
                event(log_path, f'{index}/{len(plan)} submitted job={answer}')
            while True:
                current = slurm_state(job['jobid'])
                if current in TERMINAL:
                    run_status = read_run_status(job['run'])
                    if current == 'COMPLETED' and run_status in ('complete', 'privacy_budget_exhausted'):
                        job['phase'] = 'completed'
                        write_state(state_path, state)
                        event(log_path, f'{index}/{len(plan)} completed job={job["jobid"]} run_status={run_status}')
                        break
                    job['phase'] = 'failed'
                    job['slurm_state'] = current
                    write_state(state_path, state)
                    event(log_path, f'{index}/{len(plan)} stopped job={job["jobid"]} slurm={current} run_status={run_status}')
                    raise RuntimeError('Training stopped; inspect logs and recover before continuing')
                time.sleep(args.interval)
        event(log_path, 'All planned jobs completed')
    finally:
        lock.rmdir()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Controller error: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
