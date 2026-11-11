"""V30: auxiliary self-triggered score push.

Question (fixed before any A evaluation): V28/V29 located the missing information
at the auxiliaries. If each auxiliary watches a cheap descriptor of its own raw
window and pushes fresh class scores only when that descriptor has moved since
its last push (or the cached scores reach age 16, or a reset occurs), does the
system pass the unchanged V20/V23 deployment screen?

Mechanism, per auxiliary node i and window t (all causal, all local to node i):
  d_i(t)  = per-axis mean and per-axis standard deviation of node i's window
            (2m numbers; m = 6 axes on PAMAP2, 3 otherwise; no classifier call)
  s_i(t)  = max_k |d_ik(t) - d_ik(a_i)| / sigma_ik,   a_i = window of last push
  push    = reset(t) or t - a_i >= 16 or s_i(t) > kappa
On push, node i runs its classifier and sends a Rice2 score packet; no notice
or request messages exist. The primary infers every window and fuses its
current scores with the last scores received from each auxiliary.
sigma_ik is fitted on B only: the subject-equal mean of per-subject medians of
|d_ik(t) - d_ik(t-1)| over consecutive windows within a segment.
kappa is chosen on B only from KAPPAS: the lowest-B-byte value among those
meeting the B-measurable screen gates (>=10% saving, >=75% subjects positive,
<=2.7 calls/window, <=1% reference disagreement, <=0.005 known macro-F1 drop);
if none qualifies, the smallest kappa (most conservative) is used.
Every fixed kappa is also evaluated on A for the frontier; the deployable
method is 'auxtrig_sel'.

Accounting is the V20 policy ledger without notices: Rice2 score packets
(14-byte header), 134-byte configuration per session, 45-byte status per 100
windows and at resets, and a trigger package per session (header, 16-byte
identifier, float32 sigma and kappa for both auxiliaries), all charged.

Usage (from the cscwd folder):
    python run_aux_trigger_v30.py selftest
    python run_aux_trigger_v30.py freeze
    python run_aux_trigger_v30.py run --workers 6
    python run_aux_trigger_v30.py summarize
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import gc
import hashlib
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from run_oracle_imitation_v29 import (original_cases, external_cases, load, events_for, write, read, sha,
                                      V20, V23, DATA)
from run_nested_survival import subject_frame
from validity_diagnostic import packet_cost, longest_error
from continuous_screen_accounting import latency, macro_f1
from run_validity_diagnostic import onset_metrics
from run_matched_calibration import accounting, fixed_baseline_name
from causal_score_codec import HEADER

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'aux_trigger_v30'
KAPPAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
MAX_AGE = 16
CONFIG_BYTES, STATUS_BYTES = 134, 45
AXES = {'pamap2': 6, 'mhealth': 3, 'opportunity': 3}
WINDOW = {'pamap2': 200, 'mhealth': 100, 'opportunity': 60}


# ---------------------------------------------------------------- raw descriptors
def signal_path(item, subject):
    if item['external']:
        return V23/'data'/f'subject{subject}'/'signal.npy'
    return DATA/item['case']['dataset']/'records'/f'subject{subject}'/'signal.npy'


def descriptors(f, item, dataset):
    """d[t, node-1, :] for auxiliaries 1 and 2 from the node's own raw window only."""
    m, length = AXES[dataset], WINDOW[dataset]
    out = np.empty((len(f['q']), 2, 2*m))
    for subject in np.unique(f['subjects']):
        at = np.flatnonzero(f['subjects'] == subject)
        signal = np.load(signal_path(item, int(subject)), mmap_mode='r')
        start, stop = f['start_rows'][at], f['end_rows_exclusive'][at]
        assert (stop-start == length).all(), 'window length mismatch'
        for node in (1, 2):
            x = np.asarray(signal[:, node*m:(node+1)*m], np.float64)
            finite = np.isfinite(x)
            bad = np.r_[np.zeros((1, m)), np.cumsum(~finite, 0)]
            x = np.where(finite, x, 0.)
            s1 = np.r_[np.zeros((1, m)), np.cumsum(x, 0)]
            s2 = np.r_[np.zeros((1, m)), np.cumsum(x*x, 0)]
            assert ((bad[stop]-bad[start]) == 0).all(), 'non-finite sample inside a scored window'
            mean = (s1[stop]-s1[start])/length
            var = np.maximum((s2[stop]-s2[start])/length-mean**2, 0.)
            out[at, node-1, :m] = mean
            out[at, node-1, m:] = np.sqrt(var)
    return out


def check_descriptors(f, item, dataset, d, samples=64):
    """Direct per-window recomputation on a deterministic sample."""
    m = AXES[dataset]
    for t in np.linspace(0, len(d)-1, min(samples, len(d)), dtype=int):
        signal = np.load(signal_path(item, int(f['subjects'][t])), mmap_mode='r')
        w = np.asarray(signal[f['start_rows'][t]:f['end_rows_exclusive'][t]], np.float64)
        for node in (1, 2):
            x = w[:, node*m:(node+1)*m]
            np.testing.assert_allclose(d[t, node-1], np.r_[x.mean(0), x.std(0)], rtol=1e-6, atol=1e-6)
    return min(samples, len(d))


def fit_scale(b, d):
    """B-only noise scale: subject-equal mean of per-subject median consecutive change."""
    per_subject = []
    for subject in np.unique(b['subjects']):
        at = np.flatnonzero(b['subjects'] == subject)
        step = at[1:][~b['reset'][at[1:]]]
        per_subject.append(np.median(np.abs(d[step]-d[step-1]), axis=0))
    scale = np.mean(per_subject, axis=0)
    return np.maximum(scale, 1e-6)


# ---------------------------------------------------------------- schedule
@njit(cache=True)
def trigger_schedule(q, reset, d, scale, kappa, max_age):
    n = len(q)
    send = np.zeros((n, 3), np.bool_)
    send[:, 0] = True
    anchor = np.zeros(3, np.int64)
    prediction = np.empty(n, np.int16)
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
        best, winner = -1, 0
        for c in range(12):
            value = int(q[t, 0, c])+int(q[anchor[1], 1, c])+int(q[anchor[2], 2, c])
            if value > best:
                best, winner = value, c
        prediction[t] = winner
    return send, prediction


def verify_trigger(q, reset, d, scale, kappa, send, prediction):
    """Independent pure-Python replay of the push rule and the cached fusion."""
    anchor = [0, 0, 0]
    assert send[:, 0].all() and send[reset].all()
    for t in range(len(q)):
        for node in (1, 2):
            moved = np.max(np.abs(d[t, node-1]-d[anchor[node], node-1])/scale[node-1]) > kappa
            expected = bool(reset[t]) or t-anchor[node] >= MAX_AGE or bool(moved)
            assert bool(send[t, node]) == expected, (t, node)
            if expected:
                anchor[node] = t
        fused = q[t, 0].astype(int)+q[anchor[1], 1].astype(int)+q[anchor[2], 2].astype(int)
        assert int(np.argmax(fused)) == int(prediction[t])


def trigger_package(scale, kappa):
    body = b''.join(np.r_[scale[i], kappa].astype('<f4').tobytes() for i in range(2))
    return HEADER.pack(99, 0, 0, 0)+hashlib.sha256(body).digest()[:16]+body


def trigger_accounting(f, send, package_bytes):
    stamps = np.rint(f['end_time']*1e6).astype(np.int64)
    score, inverses = packet_cost(f['q'], send, f['reset'], f['continuous_global_indices'], stamps, 'rice2')
    setup = STATUS_BYTES*f['status'].astype(np.int32)+CONFIG_BYTES*f['session_start'].astype(np.int32)
    deployment = package_bytes*f['session_start'].astype(np.int32)
    total = score+setup+deployment
    messages = send.sum(1)+3*f['status']+5*f['session_start']
    channel = latency(f['end_time'], np.arange(len(send)), np.ones(len(send), int), total, messages,
                      np.zeros(len(send), int), np.zeros(len(send), int), 0)['latency_250000_1_ms']
    return total, channel, dict(score_bytes=int(score.sum()), grant_bytes=0,
        configuration_status_bytes=int(setup.sum()), weight_deployment_bytes=int(deployment.sum()),
        package_bytes_per_session=int(package_bytes), packet_inverse_samples=inverses)


# ---------------------------------------------------------------- B selection
def b_screen(b, d, scale, codec):
    """Subject-equal B metrics for every kappa; uses B labels only."""
    rows = []
    for subject in np.unique(b['subjects']):
        sf = subject_frame(b, subject)
        ds = d[b['subjects'] == subject]
        n = len(sf['q'])
        base, _, _ = accounting(sf, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
        known = sf['endpoint_targets'] != -1
        classes = sf['classes']
        ref_f1 = macro_f1(sf['endpoint_targets'][known], classes[sf['reference'][known]], classes)
        for kappa in KAPPAS:
            send, pred = trigger_schedule(sf['q'], sf['reset'], ds, scale, kappa, MAX_AGE)
            total, _, _ = trigger_accounting(sf, send, len(trigger_package(scale, kappa)))
            f1 = macro_f1(sf['endpoint_targets'][known], classes[pred[known]], classes)
            rows.append(dict(subject=int(subject), kappa=kappa, bytes=total.mean(), base=base.mean(),
                             calls=send.sum()/n, disagreement=float((pred != sf['reference']).mean()), f1_drop=ref_f1-f1))
    r = pd.DataFrame(rows)
    r['positive'] = r.bytes < r.base
    g = r.groupby('kappa', as_index=False).mean(numeric_only=True)
    g['saving'] = 1-g.bytes/g.base
    g['qualifies'] = (g.saving >= .1) & (g.positive >= .75) & (g.calls <= 2.7) & (g.disagreement <= .01) & (g.f1_drop <= .005)
    ok = g[g.qualifies]
    chosen = float(ok.sort_values(['bytes', 'kappa']).kappa.iloc[0]) if len(ok) else float(min(KAPPAS))
    return chosen, g


# ---------------------------------------------------------------- evaluation
def evaluate(f, d, scale, chosen, meta, codec, events):
    n = len(f['q'])
    subject = int(f['subjects'][0])
    classes, reference = f['classes'], f['reference']
    known = f['endpoint_targets'] != -1
    ref_f1 = macro_f1(f['endpoint_targets'][known], classes[reference[known]], classes)
    baseline_total, _, _ = accounting(f, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
    common = dict(**meta, subject=subject, windows=n)
    rows, onsets = [], []
    for method, kappa in [('auxtrig_sel', chosen)]+[(f'auxtrig_k{k:g}', k) for k in KAPPAS]:
        send, pred = trigger_schedule(f['q'], f['reset'], d, scale, kappa, MAX_AGE)
        verify_trigger(f['q'], f['reset'], d, scale, kappa, send, pred)
        package = trigger_package(scale, kappa)
        total, channel, fees = trigger_accounting(f, send, len(package))
        f1 = macro_f1(f['endpoint_targets'][known], classes[pred[known]], classes)
        row = dict(**common, method=method, kappa=kappa, deployable=True,
            baseline_bytes_per_window=float(baseline_total.mean()), total_bytes=int(total.sum()),
            bytes_per_window=float(total.mean()), **fees,
            full_model_calls=int(send.sum()), full_models_per_window=float(send.sum()/n),
            deployable_full_models_per_window=float(send.sum()/n),
            auxiliary_push_fraction=float(send[:, 1:].mean()),
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
    return dict(version='aux_trigger_v30', client_date='2026-11-11', post_hoc=True,
        question='Do auxiliary self-triggered pushes on a cheap raw-window descriptor pass the unchanged V20/V23 screen?',
        descriptor='per-axis mean and standard deviation of the auxiliary node own window (cleaned signal), no classifier call',
        statistic='max over components of |d(t)-d(last push)| / sigma, sigma = B subject-equal mean of per-subject median consecutive change',
        push_rule='reset or age >= 16 or statistic > kappa; independent per auxiliary; no notice/request messages',
        kappas=list(KAPPAS), selection='B only: lowest B bytes among kappas meeting B saving>=10%, positive>=75%, calls<=2.7, disagreement<=1%, F1 drop<=0.005; else smallest kappa',
        accounting='Rice2 score packets, 134-byte configuration and trigger package per session, 45-byte status, no notices',
        screen='V20 joint screen unchanged; deployable method auxtrig_sel must pass all six original scenarios (separately all three external)',
        fixed_kappas='reported as a frontier; a fixed kappa passing everywhere is post hoc across nine values',
        no_result_dependent_changes=True, new_independent_subjects=0, drill_opened=False,
        original_case_grids=384, external_case_grids=12, source_sha256=sha(__file__))


def assert_frozen():
    frozen = read(OUT/'frozen.json')
    assert frozen['source_sha256'] == sha(__file__), 'Source changed after freezing'
    return frozen


def dataset_of(item):
    return 'opportunity' if item['external'] else item['case']['dataset']


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
    checked_b = check_descriptors(b, item, dataset, db)
    scale = fit_scale(b, db)
    chosen, grid = b_screen(b, db, scale, codec)
    dest.mkdir(parents=True, exist_ok=True)
    write(dest/'b_selection.json', dict(key=item['key'], chosen_kappa=chosen, scale=scale, descriptor_checks=checked_b,
          grid=grid.to_dict('records'), A_loaded=False, created_utc=datetime.now(timezone.utc).isoformat()))
    # A is loaded only after the B choice is durable.
    a = load(item, 'a')
    da = descriptors(a, item, dataset)
    check_descriptors(a, item, dataset, da)
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


def run(external_only=False, limit=None, workers=1):
    assert_frozen()
    items = list(external_cases()) if external_only else list(original_cases())+list(external_cases())
    missing = [str(it[r]) for it in items for r in ('a', 'b') if not Path(it[r]).exists()]
    assert not missing, f'{len(missing)} input frames missing, first: {missing[0]}'
    if limit:
        items = items[:limit]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            for i, (key, new) in enumerate(pool.map(_worker, items, chunksize=1)):
                print(f'V30 {i+1}/{len(items)} {key} {"done" if new else "verified"}', flush=True)
    else:
        for i, item in enumerate(items):
            new = run_one(item)
            print(f'V30 {i+1}/{len(items)} {item["key"]} {"done" if new else "verified"}', flush=True)
            gc.collect()
    if limit:
        return
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
    selections = pd.DataFrame([dict(key=read(p/'b_selection.json')['key'], kappa=read(p/'b_selection.json')['chosen_kappa']) for p in folders])
    write(OUT/'research_decision.json', dict(version='aux_trigger_v30', passing=decision, case_grids=len(folders),
          deployable_method='auxtrig_sel', chosen_kappa_counts={str(k): int(v) for k, v in selections.kappa.value_counts().items()}))
    print(equal[['dataset', 'stride', 'offset', 'method', 'savings_fraction', 'reference_disagreement',
                 'deployable_full_models_per_window', 'positive_subject_fraction', 'all_checks_passed']].to_string(index=False))
    print(decision)


def selftest():
    """B-frame-only checks; never loads an A frame."""
    item = next(external_cases())
    b = load(item, 'b')
    d = descriptors(b, item, 'opportunity')
    print('descriptor windows checked against direct recomputation:', check_descriptors(b, item, 'opportunity', d, 256))
    scale = fit_scale(b, d)
    sf = subject_frame(b, int(b['subjects'][0]))
    ds = d[b['subjects'] == int(b['subjects'][0])]
    for kappa in (KAPPAS[0], KAPPAS[-1]):
        send, pred = trigger_schedule(sf['q'], sf['reset'], ds, scale, kappa, MAX_AGE)
        verify_trigger(sf['q'], sf['reset'], ds, scale, kappa, send, pred)
        trigger_accounting(sf, send, len(trigger_package(scale, kappa)))
    # With every push forced, the schedule is complete push and the score ledger equals the V20 complete ledger.
    always, pred = trigger_schedule(sf['q'], sf['reset'], ds, scale, -1., MAX_AGE)
    assert always.all() and (pred == sf['reference']).all()
    n = len(sf['q'])
    stamps = np.rint(sf['end_time']*1e6).astype(np.int64)
    mine, _ = packet_cost(sf['q'], always, sf['reset'], sf['continuous_global_indices'], stamps, 'rice2')
    theirs, _, fees = accounting(sf, np.ones((n, 3), bool), np.zeros(n, np.int16), None, 'rice2', True)
    assert int(mine.sum()) == fees['score_bytes']
    print('trigger schedule replay, package, and ledger checks passed on', n, 'B windows; no A frame loaded')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'frozen.json').exists():
        assert_frozen()
        print('already frozen')
        return
    write(OUT/'frozen.json', dict(**protocol(), created_utc=datetime.now(timezone.utc).isoformat()))
    print(read(OUT/'frozen.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['selftest', 'freeze', 'run', 'summarize'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--external-only', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        {'freeze': freeze, 'selftest': selftest, 'summarize': summarize}.get(args.action, lambda: run(args.external_only, args.limit, args.workers))()
