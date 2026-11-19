"""Use activity-onset recovery to calibrate sequential refresh under compact notices."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import run_targeted_refresh_v34 as v34
import run_compact_refresh_v35 as v35
from run_nested_survival import subject_frame
from run_oracle_imitation_v29 import events_for, external_cases, load, original_cases, read, sha, write
from summarize_refresh_v34_v35 import summarize

OUT = v34.ROOT / 'results' / 'recovery_refresh_v36'
CALIBRATION_LIMIT = .0085
METHOD = 'recovery_compact_sel'


def choose(item, config, fallback, grid, codec, size):
    b = load(item, 'b')
    scale = np.asarray(config['scale'], dtype=np.float32)
    desc = v34.descriptors(b, item, v34.dataset_of(item)) / scale
    subjects = []
    for subject in np.unique(b['subjects']):
        f = subject_frame(b, subject)
        subjects.append((int(subject), f, desc[b['subjects'] == subject],
                         v34.baseline(f, codec).sum(), events_for(item, subject)))
    setup = v35.BINDING_BYTES * np.mean([f['session_start'].mean() for _, f, *_ in subjects])
    candidates = v35.adjusted(grid, 'notices_per_window', v35.DELTA, setup)
    candidates = candidates[v35.eligible(candidates, CALIBRATION_LIMIT)].sort_values('bytes_per_window', kind='stable')
    for index in candidates.index:
        params = v34.GRID[int(index)]
        events, positive = [], []
        for subject, f, x, base, event in subjects:
            send, req, pred = v34.replay(f['q'], f['reset'], x, *params)
            costs = v34.accounting(f, send, req, size) - v35.DELTA*np.count_nonzero(req, axis=1)
            positive.append(costs.sum() + v35.BINDING_BYTES*f['session_start'].sum() < base)
            events.extend(v34.onset_metrics(f, pred, np.zeros(len(x)), event, {'subject': subject}))
        if np.mean(positive) < .75:
            continue
        outcomes = pd.DataFrame(events)
        for _, part in outcomes.groupby('criterion'):
            added = part[part.observable].groupby('subject').additional_failure.mean().mean()
            paired = part.dropna(subset=['extra_recovery_windows'])
            within = paired.assign(in_range=paired.extra_recovery_windows <= 1).groupby('subject').in_range.mean().mean()
            if not (added <= .01 and within >= .95):
                break
        else:
            return params, False
    return fallback, True


def run_one(item):
    dest = OUT / 'cases' / item['key']
    done = dest / 'complete.json'
    if done.exists():
        assert all(sha(dest / name) == value for name, value in read(done)['files'].items())
        return
    prev34 = v34.OUT / 'cases' / item['key']
    prev35 = v35.OUT / 'cases' / item['key']
    config = read(prev34 / 'b_selection.json')
    fallback = read(prev35 / 'b_selection.json')['targeted']['compact_reserve_sel']
    scale = np.asarray(config['scale'], dtype=np.float32)
    size = len(v34.package(scale, 1., 0)) + 2
    case, dataset = item['case'], v34.dataset_of(item)
    codec = 'rice3' if item['external'] else v34.fixed_baseline_name(dataset, item['stride'], item['offset'])
    params, used_fallback = choose(item, config, fallback, pd.read_csv(prev34 / 'b_candidates.csv', index_col='index'), codec, size)
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / 'b_selection.json', dict(params=params, used_fallback=used_fallback, A_loaded=False))
    a = load(item, 'a')
    desc = v34.descriptors(a, item, dataset) / scale
    rows, onsets, packets = [], [], 0
    for subject in case['evaluation_subjects']:
        x = desc[a['subjects'] == subject]
        f = subject_frame(a, subject)
        base = v34.baseline(f, codec)
        send, req, pred = v34.replay(f['q'], f['reset'], x, *params)
        total = v34.accounting(f, send, req, size)
        packets += v34.verify(f, x, params, send, req, pred, total, size)
        v35.verify_notices(f, req)
        n = len(x)
        counts = np.count_nonzero(req, axis=1)
        total = total - v35.DELTA*counts + v35.BINDING_BYTES*f['session_start']
        messages = send.sum(1) + counts + 3*f['status'] + 8*f['session_start']
        channel = v34.latency(f['end_time'], np.arange(n), np.ones(n, int), total, messages,
                              np.zeros(n, int), np.zeros(n, int), 0)['latency_250000_1_ms']
        meta = dict(dataset=dataset, outer_fold=case['outer_fold'], subject=subject, group=case.get('group', 0),
                    direction=case.get('direction', 0), stride=item['stride'], offset=item['offset'], method=METHOD)
        row = v34.results(f, pred, total, send, base, v34.reference_f1(f), req)
        rows.append(dict(**meta, **row, used_fallback=used_fallback, windows=n))
        onsets.extend(v34.onset_metrics(f, pred, channel, events_for(item, subject), meta))
    write(dest / 'metrics.json', rows)
    write(dest / 'onsets.json', onsets)
    names = ['b_selection.json', 'metrics.json', 'onsets.json']
    write(done, dict(files={name: sha(dest / name) for name in names}, score_packets_verified=packets))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'frozen.json'
    hashes = {module.__name__: sha(module.__file__) for module in (v34, v35)} | {'v36': sha(__file__)}
    if not lock.exists():
        write(lock, dict(code=hashes, created_utc=datetime.now(timezone.utc).isoformat(),
                         B_disagreement_limit=CALIBRATION_LIMIT, B_recovery_gates=True,
                         fallback='V35 B-only 0.75% reserve', A_screen_changed=False,
                         designed_after_V35_evaluation=True, new_independent_subjects=0))
    assert read(lock)['code'] == hashes
    items = [*original_cases(), *external_cases()]
    for i, item in enumerate(items, 1):
        run_one(item)
        if i % 24 == 0 or i == len(items):
            print(f'{i}/{len(items)} {item["key"]}', flush=True)
    _, summary = summarize(OUT, 'score_packets_verified', 1)
    print(summary[['dataset','stride','offset','savings_fraction','reference_disagreement',
                   'all_checks_passed','failed_checks']].to_string(index=False))


if __name__ == '__main__':
    main()
