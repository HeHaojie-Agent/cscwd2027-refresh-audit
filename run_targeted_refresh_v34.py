"""Replay sequential auxiliary requests under the original CSCWD ledger."""

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'

from argparse import ArgumentParser
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from threadpoolctl import threadpool_limits

from adaptive_lease_oracle import grant_packet
from audit_guided_design_v27 import score_lengths
from continuous_screen_accounting import latency, macro_f1
from run_aux_trigger_v30 import CONFIG_BYTES, MAX_AGE, STATUS_BYTES, dataset_of, descriptors, fit_scale
from run_matched_calibration import fixed_baseline_name
from run_nested_survival import subject_frame
from run_oracle_imitation_v29 import events_for, external_cases, load, original_cases, read, sha, write
from run_twoside_trigger_v32 import OUT as V32, package, twoside_schedule
from run_validity_diagnostic import onset_metrics
from summarize_matched_calibration import KEYS, VIEW, equal_summary, recovery, recovery_mixture
from summarize_validity_diagnostic import checks
from validity_diagnostic import packet_cost

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'targeted_refresh_v34'
V33 = ROOT / 'results' / 'published_baselines_v33'
KAPPAS = (.75, 1., 1.25, 1.5, 2., 2.5, 3., 4.)
DELTAS = (0, 4, 8, 12, 16, 24, 32, 48, 64)
GRID = list(product(KAPPAS, DELTAS, (False, True)))
NOTICE_BYTES = 15


@njit(cache=True)
def vote(q, t, a, b):
    best, second, winner = -1, -1, 0
    for c in range(q.shape[2]):
        v = int(q[t, 0, c]) + int(q[a, 1, c]) + int(q[b, 2, c])
        if v > best:
            best, second, winner = v, best, c
        elif v > second:
            second = v
    return winner, best - second


@njit(cache=True)
def replay(q, reset, d, kappa, delta, scale_by_stale_count):
    n = len(q)
    send = np.ones((n, 3), np.bool_)
    requests = np.zeros((n, 3), np.int8)
    prediction = np.empty(n, np.int16)
    anchor = np.zeros(3, np.int64)
    for t in range(n):
        for node in (1, 2):
            previous = anchor[node]
            fire = reset[t] or t - previous >= MAX_AGE
            if not fire:
                for k in range(d.shape[2]):
                    if abs(d[t, node - 1, k] - d[previous, node - 1, k]) > kappa:
                        fire = True
                        break
            send[t, node] = fire
            if fire:
                anchor[node] = t
        winner, margin = vote(q, t, anchor[1], anchor[2])
        for step in (1, 2):
            stale = int(anchor[1] != t) + int(anchor[2] != t)
            if stale == 0:
                break
            limit = delta * stale // 2 if scale_by_stale_count else delta
            if margin > limit:
                break
            node = 1 if anchor[1] <= anchor[2] and anchor[1] != t else 2
            requests[t, node] = step
            send[t, node] = True
            anchor[node] = t
            winner, margin = vote(q, t, anchor[1], anchor[2])
        prediction[t] = winner
    return send, requests, prediction


def baseline(f, codec):
    return (score_lengths(f, np.ones((len(f['q']), 3), bool), 12, codec)
            + STATUS_BYTES * f['status'] + 116 * f['session_start'])


def accounting(f, send, requests, package_size):
    count = np.count_nonzero(requests, axis=1)
    return (score_lengths(f, send, 12, 'rice2') + NOTICE_BYTES * count
            + STATUS_BYTES * f['status'] + (CONFIG_BYTES + package_size) * f['session_start'])


def results(f, pred, total, send, base, ref_f1, req):
    known = f['endpoint_targets'] != -1
    classes = f['classes']
    f1 = macro_f1(f['endpoint_targets'][known], classes[pred[known]], classes)
    return dict(bytes_per_window=total.mean(), baseline_bytes_per_window=base.mean(),
                deployable_full_models_per_window=send.sum()/len(send),
                reference_disagreement=np.mean(pred != f['reference']),
                known_endpoint_macro_f1=f1, known_endpoint_reference_f1=ref_f1,
                known_endpoint_macro_f1_drop=ref_f1 - f1,
                notices_per_window=np.count_nonzero(req)/len(send),
                positive_savings=bool(total.sum() < base.sum()))


def reference_f1(f):
    known = f['endpoint_targets'] != -1
    classes = f['classes']
    return macro_f1(f['endpoint_targets'][known], classes[f['reference'][known]], classes)


def calibrate(b, d, codec, package_size):
    rows = []
    for subject in np.unique(b['subjects']):
        f = subject_frame(b, subject)
        x = d[b['subjects'] == subject]
        base = baseline(f, codec)
        ref_f1 = reference_f1(f)
        for index, parameters in enumerate(GRID):
            send, req, pred = replay(f['q'], f['reset'], x, *parameters)
            total = accounting(f, send, req, package_size)
            rows.append(dict(index=index, **results(f, pred, total, send, base, ref_f1, req)))
    mean = pd.DataFrame(rows).groupby('index').mean(numeric_only=True)
    mean['saving'] = 1 - mean.bytes_per_window / mean.baseline_bytes_per_window
    eligible = ((mean.positive_savings >= .75) & (mean.deployable_full_models_per_window <= 2.7)
                & (mean.known_endpoint_macro_f1_drop <= .005))
    choices = {}
    for name, limit in [('targeted_sel', .01), ('targeted_reserve_sel', .0075)]:
        candidates = mean[eligible & (mean.reference_disagreement <= limit)]
        if not len(candidates):
            # Minimize the largest relative violation of a calibration constraint.
            violation = np.maximum.reduce([
                np.maximum(mean.reference_disagreement / limit - 1, 0),
                np.maximum(mean.known_endpoint_macro_f1_drop / .005 - 1, 0),
                np.maximum(mean.deployable_full_models_per_window / 2.7 - 1, 0),
                np.maximum(1 - mean.positive_savings / .75, 0),
                np.maximum(1 - mean.saving / .1, 0),
            ])
            candidates = mean.assign(violation=violation).sort_values(['violation', 'bytes_per_window'])
        else:
            candidates = candidates.sort_values('bytes_per_window', kind='stable')
        choices[name] = int(candidates.index[0])
    mean['qualifies'] = eligible & (mean.reference_disagreement <= .01) & (mean.saving >= .1)
    return choices, mean


def verify(f, d, parameters, send, req, pred, total, package_size):
    q, reset = f['q'], f['reset']
    kappa, delta, scaled = parameters
    anchor = np.zeros(2, dtype=int)
    for t in range(len(q)):
        local = reset[t] | (t - anchor >= MAX_AGE) | (np.abs(d[t] - d[anchor, [0, 1]]).max(1) > kappa)
        anchor[local] = t
        expected = np.zeros(3, dtype=np.int8)
        for step in range(1, 4):
            scores = q[t, 0].astype(int) + q[anchor[0], 1].astype(int) + q[anchor[1], 2].astype(int)
            stale = np.where(anchor < t)[0]
            margin = np.diff(np.sort(scores)[-2:])[0]
            if not len(stale) or margin > (delta * len(stale) // 2 if scaled else delta):
                break
            choose = stale[np.argmin(anchor[stale])]
            anchor[choose] = t
            expected[choose + 1] = step
        assert send[t, 0] and np.array_equal(send[t, 1:], anchor == t)
        assert np.array_equal(req[t], expected) and pred[t] == scores.argmax()
    stamp = np.rint(f['end_time']*1e6).astype(np.int64)
    windows = f['continuous_global_indices']
    score, samples = packet_cost(q, send, reset, windows, stamp, 'rice2')
    for t, node in zip(*np.where(req > 0)):
        assert len(grant_packet(node, windows[t], stamp[t], 1)) == NOTICE_BYTES
    reconstructed = (score + NOTICE_BYTES * np.count_nonzero(req, axis=1)
                     + STATUS_BYTES * f['status'] + (CONFIG_BYTES + package_size) * f['session_start'])
    np.testing.assert_array_equal(total, reconstructed)
    return samples


def run_one(item):
    dest = OUT / 'cases' / item['key']
    done = dest / 'complete.json'
    if done.exists():
        assert all(sha(dest / name) == digest for name, digest in read(done)['files'].items())
        return
    dest.mkdir(parents=True, exist_ok=True)
    case, dataset = item['case'], dataset_of(item)
    codec = 'rice3' if item['external'] else fixed_baseline_name(dataset, item['stride'], item['offset'])
    b = load(item, 'b')
    db = descriptors(b, item, dataset)
    scale = fit_scale(b, db).astype('float32')
    # The V32 layout also carries two one-byte guard-mode flags.
    package_size = len(package(scale, 1., 0)) + 2
    choices, grid = calibrate(b, db / scale, codec, package_size)
    old = read(V32 / 'cases' / item['key'] / 'b_selection.json')
    policies = {k: GRID[v] for k, v in choices.items()}
    policies['targeted_v32_parameters'] = (old['chosen_kappa'], old['chosen_delta'], False)
    write(dest / 'b_selection.json', dict(policies=policies, scale=scale, A_loaded=False,
          eligible_count=int(grid.qualifies.sum()), grid=GRID))
    grid.to_csv(dest / 'b_candidates.csv')
    del b, db
    a = load(item, 'a')
    da = descriptors(a, item, dataset) / scale
    rows, onsets, packets = [], [], 0
    for subject in case['evaluation_subjects']:
        f = subject_frame(a, subject)
        x = da[a['subjects'] == subject]
        base = baseline(f, codec)
        ref_f1 = reference_f1(f)
        events = events_for(item, subject)
        for method, parameters in policies.items():
            send, req, pred = replay(f['q'], f['reset'], x, *parameters)
            total = accounting(f, send, req, package_size)
            if method == 'targeted_sel':
                packets += verify(f, x, parameters, send, req, pred, total, package_size)
            notices = np.count_nonzero(req, axis=1)
            messages = send.sum(1) + notices + 3*f['status'] + 5*f['session_start']
            n = len(total)
            channel = latency(f['end_time'], np.arange(n), np.ones(n, int), total, messages,
                              np.zeros(n, int), np.zeros(n, int), 0)['latency_250000_1_ms']
            meta = dict(dataset=dataset, outer_fold=case['outer_fold'], subject=subject,
                        group=case.get('group', 0), direction=case.get('direction', 0),
                        stride=item['stride'], offset=item['offset'], method=method)
            rows.append(dict(**meta, **results(f, pred, total, send, base, ref_f1, req),
                             channel_p95_ms=float(np.quantile(channel, .95)), windows=n))
            onsets.extend(onset_metrics(f, pred, channel, events, meta))
    write(dest / 'metrics.json', rows)
    write(dest / 'onsets.json', onsets)
    files = ['b_selection.json', 'b_candidates.csv', 'metrics.json', 'onsets.json']
    write(done, dict(files={name: sha(dest / name) for name in files}, main_packet_inverses=packets))


def summarize():
    folders = sorted(p.parent for p in (OUT / 'cases').rglob('complete.json'))
    assert len(folders) == 396
    rows = pd.DataFrame([row for p in folders for row in read(p / 'metrics.json')])
    events = pd.DataFrame([event for p in folders for event in read(p / 'onsets.json')])
    rows = rows.merge(recovery(events), on=VIEW+['direction'], validate='one_to_one')
    rows.to_csv(OUT / 'view_metrics.csv', index=False)
    _, subject, summary = equal_summary(rows, KEYS)
    subject['positive_subject_fraction'] = subject.bytes_per_window < subject.baseline_bytes_per_window
    summary = summary.merge(subject.groupby(KEYS, as_index=False).positive_subject_fraction.mean(), on=KEYS)
    mixture = recovery_mixture(events)
    for criterion in ('first_correct', 'two_consecutive_correct'):
        p = mixture[mixture.criterion == criterion][KEYS+['p95']]
        summary = summary.merge(p.rename(columns={'p95': criterion+'_p95_extra_recovery_windows'}), on=KEYS)
    summary['savings_fraction'] = 1 - summary.bytes_per_window / summary.baseline_bytes_per_window
    failures = [[name for name, passed in checks(r).items() if not passed] for r in summary.to_dict('records')]
    summary['all_checks_passed'] = [not f for f in failures]
    summary['failed_checks'] = [','.join(f) for f in failures]
    summary.to_csv(OUT / 'dataset_equal_metrics.csv', index=False)
    print(summary[['dataset','stride','offset','method','savings_fraction','reference_disagreement',
                   'known_endpoint_macro_f1','all_checks_passed','failed_checks']].to_string(index=False))
    write(OUT / 'summary_complete.json', dict(case_grids=len(folders), view_rows=len(rows),
          onsets=len(events), packet_inverses=sum(read(p / 'complete.json')['main_packet_inverses'] for p in folders),
          passed=summary.groupby('method').all_checks_passed.sum().to_dict(), new_independent_subjects=0))


def selftest():
    item = next(external_cases())
    f = load(item, 'b')
    d = descriptors(f, item, dataset_of(item))
    scale = fit_scale(f, d)
    for delta in (-1, 24, 1000):
        parameters = (2., delta, False)
        x = d / scale
        send, req, pred = replay(f['q'], f['reset'], x, *parameters)
        total = accounting(f, send, req, 0)
        verify(f, x, parameters, send, req, pred, total, 0)
        if delta in (-1, 1000):
            old, _, expected = twoside_schedule(f['q'], f['reset'], d, scale, 2., delta, MAX_AGE)
            np.testing.assert_array_equal(send, old)
            np.testing.assert_array_equal(pred, expected)
    print('B-only replay, limiting policies, and packet ledger verified.')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['selftest', 'run', 'summarize'])
    args = parser.parse_args()
    if args.action == 'selftest':
        selftest()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = OUT / 'frozen.json'
    protocol = dict(source_sha256=sha(__file__), case_grids=396, candidate_grid=GRID,
                    selector='B only; minimize bytes under quality and call caps, with constraint violation fallback',
                    primary_method='targeted_sel', reserve_diagnostic='0.75% B disagreement cap',
                    notice_bytes=NOTICE_BYTES, previous_A_results_used_in_design=True, new_independent_subjects=0)
    if not freeze.exists():
        assert args.action == 'run'
        write(freeze, dict(**protocol, created_utc=datetime.now(timezone.utc).isoformat()))
    else:
        assert read(freeze)['source_sha256'] == protocol['source_sha256']
    if args.action == 'summarize':
        summarize()
        return
    items = list(original_cases()) + list(external_cases())
    assert len(items) == 396
    for i, item in enumerate(items, 1):
        run_one(item)
        if i % 12 == 0 or i == len(items):
            print(f'{i}/{len(items)} {item["key"]}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
