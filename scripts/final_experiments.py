"""Prepare frozen test plans and run reproducible CPU compression experiments.

Run from the repository root. Preparing a plan reads checkpoints and metadata,
but never loads test samples. Executing a test requires an explicit flag.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def selected_jobs(plan, names):
    jobs = load(plan)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('The source jobs file must be a nonempty list')
    result = [j for j in jobs if not names or Path(j['run']).name in names]
    if names and {Path(j['run']).name for j in result} != set(names):
        raise ValueError('Some requested run names are missing from the source plan')
    if len({j['run'] for j in result}) != len(result):
        raise ValueError('Duplicate run paths in source plan')
    return result


def inspect_run(job):
    from energyfl.engine import committed, load_checkpoint
    run = Path(job['run'])
    if (run / 'RUNNING.lock').exists():
        raise ValueError(f'Run is locked: {run}')
    config, meta, status = (load(run / n) for n in ('config.json', 'metadata.json', 'status.json'))
    if status['status'] not in ('complete', 'privacy_budget_exhausted'):
        raise ValueError(f'Run is not finished: {run}')
    rounds = committed(run)
    if not rounds:
        raise ValueError(f'No completed checkpoint: {run}')
    latest = rounds[-1] / 'checkpoint.pt'
    ck = load_checkpoint(latest)
    policy = 'last' if config['privacy']['enabled'] else 'best'
    chosen = int(rounds[-1].name) if policy == 'last' else ck['best_round']
    if not chosen:
        raise ValueError(f'No validated checkpoint in {run}')
    chosen_path = run / 'rounds' / f'{chosen:04d}' / 'checkpoint.pt'
    if not chosen_path.is_file():
        raise ValueError(f'Missing selected checkpoint: {chosen_path}')
    # Only seeds and filesystem location may vary within one reported group.
    grouping = dict(config)
    grouping.pop('seed', None)
    grouping.pop('data', None)
    group = hashlib.sha256(json.dumps(grouping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    return {'run': str(run), 'seed': config['seed'], 'method': config['train']['method'],
            'scheduler': config['train']['scheduler'], 'weighting': config['train']['weighting'],
            'dp': config['privacy']['enabled'], 'group': group, 'policy': policy, 'round': chosen,
            'checkpoint_sha256': digest(chosen_path), 'latest_round': int(rounds[-1].name),
            'latest_checkpoint_sha256': digest(latest), 'config_sha256': digest(run / 'config.json'),
            'dataset_fingerprint': meta['dataset_fingerprint']}


def verify_entry(entry):
    current = inspect_run({'run': entry['run']})
    if current != entry:
        raise ValueError(f'Run changed since the plan was frozen: {entry["run"]}')


def prepare_tests(args):
    entries = []
    for job in selected_jobs(args.jobs, args.runs):
        run = Path(job['run'])
        if (run / 'test').exists() or (run / '.test-pending').exists():
            raise ValueError(f'Existing or incomplete test found, inspect before planning: {run}')
        entries.append(inspect_run(job))
    datasets = {e['dataset_fingerprint'] for e in entries}
    if len(datasets) != 1:
        raise ValueError('Selected runs have different data fingerprints')
    save_new(args.out, {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                        'source_jobs': str(args.jobs), 'source_jobs_sha256': digest(args.jobs),
                        'policy': 'best validation checkpoint for non-DP; last checkpoint for DP',
                        'dataset_fingerprint': next(iter(datasets)), 'runs': entries})
    print(f'Prepared {len(entries)} runs in {args.out}; no test data evaluated.')


def run_tests(args):
    from energyfl.engine import test_run
    if not args.confirm_frozen:
        raise ValueError('Confirm the frozen protocol explicitly with --confirm-frozen')
    plan = load(args.plan)
    if not plan.get('runs'):
        raise ValueError('Frozen plan contains no runs')
    # Validate every planned run before evaluating any test samples.
    for e in plan['runs']:
        verify_entry(e)
        run = Path(e['run'])
        if (run / '.test-pending').exists():
            raise ValueError(f'Incomplete test output requires inspection: {run}')
        if (run / 'test').exists():
            m = load(run / 'test' / 'metrics.json')
            if m['checkpoint_sha256'] != e['checkpoint_sha256'] or m['dataset_fingerprint'] != e['dataset_fingerprint']:
                raise ValueError(f'Existing test does not match frozen plan: {run}')
    for e in plan['runs']:
        run = Path(e['run'])
        if (run / 'test').exists():
            print(f'Skip previously completed matching test: {run}', flush=True)
            continue
        print(f'Testing {run}, {e["policy"]} round {e["round"]}', flush=True)
        result = test_run(run, checkpoint=e['policy'])
        if result['checkpoint_sha256'] != e['checkpoint_sha256']:
            raise RuntimeError(f'Tested checkpoint differs from frozen plan: {run}')
    print('All selected tests completed; use summarize-tests to aggregate them.')


def csv_new(path, fields, rows):
    with Path(path).open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_tests(args):
    plan = load(args.plan)
    rows = []
    for e in plan['runs']:
        verify_entry(e)
        m = load(Path(e['run']) / 'test' / 'metrics.json')
        if m['checkpoint_sha256'] != e['checkpoint_sha256'] or m['dataset_fingerprint'] != e['dataset_fingerprint']:
            raise ValueError(f'Test result mismatch: {e["run"]}')
        rows.append({k: e[k] for k in ('run', 'method', 'scheduler', 'weighting', 'seed', 'dp', 'group', 'policy', 'round')} | {
            'mae_kwh': m['macro_mae_kwh'], 'rmse_kwh': m['macro_rmse_kwh'],
            'p90_household_mae_kwh': m['p90_household_mae_kwh']})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    csv_new(out / 'test_runs.csv', list(rows[0]), rows)
    groups = {}
    for row in rows:
        key = (row['method'], row['scheduler'], row['weighting'], row['dp'], row['group'])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        item = dict(zip(('method', 'scheduler', 'weighting', 'dp', 'group'), key))
        item['n'] = len(group)
        for k in ('mae_kwh', 'rmse_kwh', 'p90_household_mae_kwh'):
            vals = [r[k] for r in group]
            item[k + '_mean'] = statistics.mean(vals)
            item[k + '_sample_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
        summary.append(item)
    csv_new(out / 'test_groups.csv', list(summary[0]), summary)
    save_new(out / 'provenance.json', {'frozen_plan': str(args.plan), 'frozen_plan_sha256': digest(args.plan),
                                       'note': 'Groups assume the frozen plan defines comparable settings; review configurations.'})
    print(f'Saved {len(rows)} test results to {out}')


def compress(args):
    from energyfl.profile import compression
    if args.repeats < 10 or not 0 <= args.tolerance <= 1:
        raise ValueError('Use at least 10 repeats and tolerance in [0,1]')
    jobs = selected_jobs(args.jobs, args.runs)
    entries = [inspect_run(j) for j in jobs]
    if any(e['method'] in ('local', 'seasonal') for e in entries):
        raise ValueError('Compression needs a shared neural model, not local or seasonal')
    if any(e['dp'] for e in entries):
        raise ValueError('Use non-DP runs for this validation-based compression experiment')
    if len({e['dataset_fingerprint'] for e in entries}) != 1:
        raise ValueError('Compression runs use different data fingerprints')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    save_new(out / 'experiment.json', {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_jobs_sha256': digest(args.jobs), 'repeats': args.repeats, 'tolerance': args.tolerance,
        'checkpoint_policy': 'best validation checkpoint', 'runs': entries,
        'note': 'CPU latency measured on this host; edge latency and memory feasibility are modeled.'})
    variants, assignments = [], []
    for e in entries:
        verify_entry(e)
        name = Path(e['run']).name
        print(f'Compressing {name}', flush=True)
        report = compression(e['run'], out / name, repeats=args.repeats, tolerance=args.tolerance)
        if report['checkpoint_round'] != e['round']:
            raise RuntimeError(f'Unexpected checkpoint for {name}')
        for variant, v in report['variants'].items():
            variants.append({'run': name, 'seed': e['seed'], 'variant': variant, **v})
        for a in report['assignment']:
            assignments.append({'run': name, 'seed': e['seed'], **a})
    csv_new(out / 'variants.csv', list(variants[0]), variants)
    csv_new(out / 'assignments.csv', list(assignments[0]), assignments)
    print(f'Compression and simulated assignment saved to {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare-tests')
    a.add_argument('--jobs', required=True); a.add_argument('--out', required=True)
    a.add_argument('--runs', nargs='+'); a.set_defaults(func=prepare_tests)
    a = sub.add_parser('run-tests')
    a.add_argument('--plan', required=True)
    a.add_argument('--confirm-frozen', action='store_true'); a.set_defaults(func=run_tests)
    a = sub.add_parser('summarize-tests')
    a.add_argument('--plan', required=True); a.add_argument('--out', required=True)
    a.set_defaults(func=summarize_tests)
    a = sub.add_parser('compress')
    a.add_argument('--jobs', required=True); a.add_argument('--out', required=True)
    a.add_argument('--runs', nargs='+', required=True)
    a.add_argument('--repeats', type=int, default=100); a.add_argument('--tolerance', type=float, default=0.05)
    a.set_defaults(func=compress)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
