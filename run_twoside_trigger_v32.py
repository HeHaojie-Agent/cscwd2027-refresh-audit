"""V32: two-sided trigger = auxiliary self-triggered push (V30 rule) + primary margin guard.

Declared context: designed after V30 and V31 A results were seen (V30 failed the
screen mainly through output agreement); the same subjects are evaluated again.
Frozen before its own A evaluation. V30/V31 remain reported unchanged.

Rationale: residual V30 disagreement needs two facts held by different nodes:
an auxiliary has drifted (visible only to it) and the fused decision is fragile
(visible only to the primary). Each node acts on what it can observe.

Per window t:
  1. each auxiliary i pushes when reset, age >= 16, or
     max_k |d_ik(t)-d_ik(a_i)|/sigma_ik > kappa         (V30 descriptor and sigma)
  2. the primary fuses its current scores with the latest auxiliary scores;
     if some auxiliary is stale at t and the fused margin (top1 - top2 of the
     uint8 score sums) is <= delta, the primary pulls: it sends one 30-byte
     notice pair and every stale auxiliary infers and sends fresh scores.
  3. output = fused argmax after any pull (ties: smallest index).
sigma is fitted on B as in V30. (kappa, delta) is chosen on B from KAPPAS x DELTAS:
lowest B bytes among pairs meeting B saving>=10%, positive>=75%, calls<=2.7,
disagreement<=1%, F1 drop<=0.005; else the most conservative pair
(smallest kappa, largest delta). All pairs are also evaluated on A.
Ledger: V30 ledger plus 30 bytes and two messages per pull; the trigger package
also carries delta.

Usage: python run_twoside_trigger_v32.py selftest|freeze|run --workers 6|summarize
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from run_oracle_imitation_v29 import original_cases, external_cases, load, events_for, write, read, sha
from run_aux_trigger_v30 import (descriptors, check_descriptors, fit_scale, dataset_of, MAX_AGE, CONFIG_BYTES, STATUS_BYTES)
from run_nested_survival import subject_frame
from validity_diagnostic import packet_cost, longest_error
from continuous_screen_accounting import latency, macro_f1
from run_validity_diagnostic import onset_metrics
from run_matched_calibration import accounting, fixed_baseline_name
from causal_score_codec import HEADER

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'twoside_trigger_v32'
KAPPAS = (1.0, 1.5, 2.0, 3.0, 4.0)
DELTAS = (4, 8, 16, 24, 32, 48)
PULL_BYTES = 30


@njit(cache=True)
def twoside_schedule(q, reset, d, scale, kappa, delta, max_age):
    n = len(q)
    send = np.zeros((n, 3), np.bool_)
    send[:, 0] = True
    pull = np.zeros(n, np.bool_)
    anchor = np.zeros(3, np.int64)
    prediction = np.empty(n, np.int16)
    fused = np.empty(12, np.int64)
    for t in range(n):
        for node in (1, 2):
            fire = bool(reset[t]) or t-anchor[node] >= max_age
            if not fire:
                s = 0.
                for k in range(d.shape[2]):
                    v = abs(d[t, node-1, k]-d[anchor[node], node-1, k])/scale[node-1, k]
                    if v > s:
                        s = v
                fire = s > kappa
            if fire:
                send[t, node] = True
                anchor[node] = t
        if not (send[t, 1] and send[t, 2]):
            first, second = -1, -1
            for c in range(12):
                v = int(q[t, 0, c])+int(q[anchor[1], 1, c])+int(q[anchor[2], 2, c])
                if v > first:
                    second = first
                    first = v
                elif v > second:
                    second = v
            if first-second <= delta:
                pull[t] = True
                for node in (1, 2):
                    if not send[t, node]:
                        send[t, node] = True
                        anchor[node] = t
        best, winner = -1, 0
        for c in range(12):
            v = int(q[t, 0, c])+int(q[anchor[1], 1, c])+int(q[anchor[2], 2, c])
            if v > best:
                best, winner = v, c
        prediction[t] = winner
    return send, pull, prediction


def verify_twoside(q, reset, d, scale, kappa, delta, send, pull, prediction):
    anchor = [0, 0, 0]
    assert send[:, 0].all() and send[reset].all() and not pull[reset].any()
    for t in range(len(q)):
        pushed = [False, False, False]
        for node in (1, 2):
            moved = np.max(np.abs(d[t, node-1]-d[anchor[node], node-1])/scale[node-1]) > kappa
            pushed[node] = bool(reset[t]) or t-anchor[node] >= MAX_AGE or bool(moved)
            if pushed[node]:
                anchor[node] = t
        expect_pull = False
        if not (pushed[1] and pushed[2]):
            fused = np.sort(q[t, 0].astype(int)+q[anchor[1], 1].astype(int)+q[anchor[2], 2].astype(int))
            expect_pull = fused[-1]-fused[-2] <= delta
        assert bool(pull[t]) == expect_pull, t
        for node in (1, 2):
            if expect_pull and not pushed[node]:
                pushed[node] = True
                anchor[node] = t
            assert bool(send[t, node]) == pushed[node], (t, node)
        fused = q[t, 0].astype(int)+q[anchor[1], 1].astype(int)+q[anchor[2], 2].astype(int)
        assert int(np.argmax(fused)) == int(prediction[t])


def package(scale, kappa, delta):
    body = b''.join(np.r_[scale[i], kappa, delta].astype('<f4').tobytes() for i in range(2))
    return HEADER.pack(101, 0, 0, 0)+hashlib.sha256(body).digest()[:16]+body


def ledger(f, send, pull, package_bytes):
    stamps = np.rint(f['end_time']*1e6).astype(np.int64)
    score, inverses = packet_cost(f['q'], send, f['reset'], f['continuous_global_indices'], stamps, 'rice2')
    setup = STATUS_BYTES*f['status'].astype(np.int32)+CONFIG_BYTES*f['session_start'].astype(np.int32)
    deployment = package_bytes*f['session_start'].astype(np.int32)
    control = PULL_BYTES*pull.astype(np.int32)
    total = score+setup+deployment+control
    messages = send.sum(1)+2*pull+3*f['status']+5*f['session_start']
    channel = latency(f['end_time'], np.arange(len(send)), np.ones(len(send), int), total, messages,
                      np.zeros(len(send), int), np.zeros(len(send), int), 0)['latency_250000_1_ms']
    return total, channel, dict(score_bytes=int(score.sum()), grant_bytes=int(control.sum()),
        configuration_status_bytes=int(setup.sum()), weight_deployment_bytes=int(deployment.sum()),
        package_bytes_per_session=int(package_bytes), packet_inverse_samples=inverses)


def pairs():
    return [(k, dl) for k in KAPPAS for dl in DELTAS]


def b_screen(b, d, scale, codec):
    rows = []
    for subject in np.unique(b['subjects']):
        sf = subject_frame(b, subject)
        ds = d[b['subjects'] == subject]
        n = len(sf['q'])
        base, _, _ = accounting(sf, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
        known = sf['endpoint_targets'] != -1
        classes = sf['classes']
        ref_f1 = macro_f1(sf['endpoint_targets'][known], classes[sf['reference'][known]], classes)
        for kappa, delta in pairs():
            send, pull, pred = twoside_schedule(sf['q'], sf['reset'], ds, scale, kappa, delta, MAX_AGE)
            total, _, _ = ledger(sf, send, pull, len(package(scale, kappa, delta)))
            f1 = macro_f1(sf['endpoint_targets'][known], classes[pred[known]], classes)
            rows.append(dict(subject=int(subject), kappa=kappa, delta=delta, bytes=total.mean(), base=base.mean(),
                             calls=send.sum()/n, disagreement=float((pred != sf['reference']).mean()), f1_drop=ref_f1-f1))
    r = pd.DataFrame(rows)
    r['positive'] = r.bytes < r.base
    g = r.groupby(['kappa', 'delta'], as_index=False).mean(numeric_only=True)
    g['saving'] = 1-g.bytes/g.base
    g['qualifies'] = (g.saving >= .1) & (g.positive >= .75) & (g.calls <= 2.7) & (g.disagreement <= .01) & (g.f1_drop <= .005)
    ok = g[g.qualifies]
    if len(ok):
        top = ok.sort_values(['bytes', 'kappa', 'delta'], ascending=[True, True, False]).iloc[0]
        chosen = (float(top.kappa), int(top.delta))
    else:
        chosen = (float(min(KAPPAS)), int(max(DELTAS)))
    return chosen, g


def evaluate(f, d, scale, chosen, meta, codec, events, verify=True):
    n = len(f['q'])
    subject = int(f['subjects'][0])
    classes, reference = f['classes'], f['reference']
    known = f['endpoint_targets'] != -1
    ref_f1 = macro_f1(f['endpoint_targets'][known], classes[reference[known]], classes)
    baseline_total, _, _ = accounting(f, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
    common = dict(**meta, subject=subject, windows=n)
    rows, onsets = [], []
    for method, (kappa, delta) in [('twoside_sel', chosen)]+[(f'twoside_k{k:g}_d{dl}', (k, dl)) for k, dl in pairs()]:
        send, pull, pred = twoside_schedule(f['q'], f['reset'], d, scale, kappa, delta, MAX_AGE)
        if verify and method == 'twoside_sel':
            verify_twoside(f['q'], f['reset'], d, scale, kappa, delta, send, pull, pred)
        total, channel, fees = ledger(f, send, pull, len(package(scale, kappa, delta)))
        f1 = macro_f1(f['endpoint_targets'][known], classes[pred[known]], classes)
        row = dict(**common, method=method, kappa=kappa, delta=delta, deployable=True,
            baseline_bytes_per_window=float(baseline_total.mean()), total_bytes=int(total.sum()),
            bytes_per_window=float(total.mean()), **fees, pull_fraction=float(pull.mean()),
            full_model_calls=int(send.sum()), full_models_per_window=float(send.sum()/n),
            deployable_full_models_per_window=float(send.sum()/n), auxiliary_push_fraction=float(send[:, 1:].mean()),
            reference_mismatches=int((pred != reference).sum()), reference_disagreement=float((pred != reference).mean()),
            known_endpoint_macro_f1=f1, known_endpoint_reference_f1=ref_f1, known_endpoint_macro_f1_drop=ref_f1-f1,
            longest_reference_error_windows=longest_error(pred, reference, f['reset']),
            channel_p95_ms=float(np.quantile(channel, .95)), known_windows=int(known.sum()), unknown_windows=int((~known).sum()))
        for node in range(3):
            row[f'node{node}_calls'] = int(send[:, node].sum())
        rows.append(row)
        onsets += onset_metrics(f, pred, channel, events, dict(**common, method=method))
    return rows, onsets


def protocol():
    return dict(version='twoside_trigger_v32', client_date='2026-11-11', post_hoc=True,
        designed_after='V30 and V31 A results were seen; same subjects',
        question='Does adding a primary margin guard to auxiliary self-triggered push pass the unchanged screen?',
        auxiliary_rule='V30 descriptor, B-fitted sigma, push when reset, age >= 16, or max normalized change > kappa',
        primary_rule='if any auxiliary is stale and fused uint8 margin <= delta, pull all stale auxiliaries with one 30-byte notice pair',
        kappas=list(KAPPAS), deltas=list(DELTAS),
        selection='B only: lowest B bytes among (kappa, delta) meeting B saving>=10%, positive>=75%, calls<=2.7, disagreement<=1%, F1 drop<=0.005; else smallest kappa with largest delta',
        accounting='V30 ledger + 30 bytes and two messages per pull; package carries sigma, kappa, delta',
        screen='V20 joint screen unchanged; deployable method twoside_sel must pass all six original scenarios (separately all three external)',
        no_result_dependent_changes=True, new_independent_subjects=0, drill_opened=False,
        original_case_grids=384, external_case_grids=12, source_sha256=sha(__file__))


def assert_frozen():
    frozen = read(OUT/'frozen.json')
    assert frozen['source_sha256'] == sha(__file__), 'Source changed after freezing'
    return frozen


def run_one(item):
    dest = OUT/'cases'/item['key']
    done = dest/'complete.json'
    if done.exists():
        for name, value in read(done)['files_sha256'].items():
            assert sha(dest/name) == value
        return False
    started = time.perf_counter()
    dataset = dataset_of(item)
    case = item['case']
    codec = 'rice3' if item['external'] else fixed_baseline_name(case['dataset'], item['stride'], item['offset'])
    b = load(item, 'b')
    db = descriptors(b, item, dataset)
    checked = check_descriptors(b, item, dataset, db)
    scale = fit_scale(b, db)
    chosen, grid = b_screen(b, db, scale, codec)
    dest.mkdir(parents=True, exist_ok=True)
    write(dest/'b_selection.json', dict(key=item['key'], chosen_kappa=chosen[0], chosen_delta=chosen[1], scale=scale,
          descriptor_checks=checked, grid=grid.to_dict('records'), A_loaded=False, created_utc=datetime.now(timezone.utc).isoformat()))
    a = load(item, 'a')
    da = descriptors(a, item, dataset)
    check_descriptors(a, item, dataset, da, 16)
    meta = dict(dataset=dataset, outer_fold=case['outer_fold'],
                group=case.get('group', 0) if not item['external'] else 0, direction=case.get('direction', 0) if not item['external'] else 0,
                stride=item['stride'], offset=item['offset'])
    rows, onsets = [], []
    for subject in case['evaluation_subjects']:
        at = a['subjects'] == subject
        sf = subject_frame(a, subject) if not item['external'] else a
        r, o = evaluate(sf, da[at] if not item['external'] else da, scale, chosen, meta, codec, events_for(item, subject))
        rows += r
        onsets += o
    write(dest/'metrics.json', rows)
    write(dest/'onsets.json', onsets)
    write(done, dict(files_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file() and p.name != 'complete.json'},
                     elapsed_seconds=time.perf_counter()-started))
    return True


def _worker(item):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return item['key'], run_one(item)


def run(external_only=False, workers=1):
    assert_frozen()
    items = list(external_cases()) if external_only else list(original_cases())+list(external_cases())
    missing = [str(it[r]) for it in items for r in ('a', 'b') if not Path(it[r]).exists()]
    assert not missing, f'{len(missing)} input frames missing, first: {missing[0]}'
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max(1, workers)) as pool:
        for i, (key, new) in enumerate(pool.map(_worker, items, chunksize=1)):
            print(f'V32 {i+1}/{len(items)} {key} {"done" if new else "verified"}', flush=True)
    write(OUT/('external_complete.json' if external_only else 'run_complete.json'), dict(case_grids=len(items),
          completed_utc=datetime.now(timezone.utc).isoformat()))


def summarize():
    from summarize_matched_calibration import equal_summary, recovery, recovery_mixture, KEYS, VIEW
    from summarize_validity_diagnostic import checks
    assert_frozen()
    folders = sorted(p.parent for p in (OUT/'cases').rglob('complete.json'))
    rows = pd.DataFrame([r for p in folders for r in read(p/'metrics.json')])
    events = pd.DataFrame([e for p in folders for e in read(p/'onsets.json')])
    assert not rows.duplicated(VIEW+['direction', 'method']).any()
    rows = rows.merge(recovery(events), on=VIEW+['direction'], validate='one_to_one')
    rows.to_csv(OUT/'view_metrics.csv', index=False)
    view, subject, equal = equal_summary(rows, KEYS)
    subject['positive_savings'] = subject.bytes_per_window < subject.baseline_bytes_per_window
    positive = subject.groupby(KEYS, as_index=False).positive_savings.mean().rename(columns={'positive_savings': 'positive_subject_fraction'})
    equal = equal.merge(positive, on=KEYS, validate='one_to_one')
    equal['savings_fraction'] = 1-equal.bytes_per_window/equal.baseline_bytes_per_window
    mixture = recovery_mixture(events)
    for criterion in ('first_correct', 'two_consecutive_correct'):
        m = mixture[mixture.criterion == criterion][KEYS+['p95']].rename(columns={'p95': criterion+'_p95_extra_recovery_windows'})
        equal = equal.merge(m, on=KEYS, how='left', validate='one_to_one')
    flags = [checks(r) for r in equal.to_dict('records')]
    equal['all_checks_passed'] = [all(c.values()) for c in flags]
    equal['failed_checks'] = [','.join(k for k, v in c.items() if not v) for c in flags]
    equal.to_csv(OUT/'dataset_equal_metrics.csv', index=False)
    original = equal[equal.dataset != 'opportunity']
    external = equal[equal.dataset == 'opportunity']
    decision = dict(all_six_original=[m for m, g in original.groupby('method') if len(g) == 6 and g.all_checks_passed.all()],
                    all_three_external=[m for m, g in external.groupby('method') if len(g) == 3 and g.all_checks_passed.all()])
    sel = [(read(p/'b_selection.json')['chosen_kappa'], read(p/'b_selection.json')['chosen_delta']) for p in folders]
    write(OUT/'research_decision.json', dict(version='twoside_trigger_v32', passing=decision, case_grids=len(folders),
          deployable_method='twoside_sel', chosen_counts={f'k{k:g}_d{d}': int(v) for (k, d), v in pd.Series(sel).value_counts().items()}))
    print(equal[equal.method == 'twoside_sel'][['dataset', 'stride', 'offset', 'savings_fraction', 'reference_disagreement',
                 'deployable_full_models_per_window', 'positive_subject_fraction', 'all_checks_passed', 'failed_checks']].to_string(index=False))
    print(decision)


def selftest():
    """B-frame-only checks; never loads an A frame."""
    item = next(external_cases())
    b = load(item, 'b')
    d = descriptors(b, item, 'opportunity')
    check_descriptors(b, item, 'opportunity', d, 128)
    scale = fit_scale(b, d)
    sf = subject_frame(b, int(b['subjects'][0]))
    ds = d[b['subjects'] == int(b['subjects'][0])]
    for kappa, delta in [(KAPPAS[0], DELTAS[0]), (KAPPAS[-1], DELTAS[-1]), (2.0, 16)]:
        send, pull, pred = twoside_schedule(sf['q'], sf['reset'], ds, scale, kappa, delta, MAX_AGE)
        verify_twoside(sf['q'], sf['reset'], ds, scale, kappa, delta, send, pull, pred)
        ledger(sf, send, pull, len(package(scale, kappa, delta)))
    # A guard that always fires reproduces complete-push fusion; a guard that never fires reproduces V30 exactly.
    always, pull, pred = twoside_schedule(sf['q'], sf['reset'], ds, scale, 1e9, 10**9, MAX_AGE)
    assert always.all() and (pred == sf['reference']).all()
    from run_aux_trigger_v30 import trigger_schedule
    s30, p30 = trigger_schedule(sf['q'], sf['reset'], ds, scale, 2.0, MAX_AGE)
    s32, pull, p32 = twoside_schedule(sf['q'], sf['reset'], ds, scale, 2.0, -1, MAX_AGE)
    assert (s30 == s32).all() and (p30 == p32).all() and not pull.any()
    print('two-sided schedule replay, ledger, and reduction checks passed on', len(sf['q']), 'B windows; no A frame loaded')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'frozen.json').exists():
        assert_frozen()
        print('already frozen')
        return
    write(OUT/'frozen.json', dict(**protocol(), created_utc=datetime.now(timezone.utc).isoformat()))
    print(read(OUT/'frozen.json')['source_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['selftest', 'freeze', 'run', 'summarize'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--external-only', action='store_true')
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        {'freeze': freeze, 'selftest': selftest, 'summarize': summarize}.get(args.action, lambda: run(args.external_only, args.workers))()
