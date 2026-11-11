"""V31: learned auxiliary-side change detector for self-triggered push.

Declared context: designed after the V30 A results were seen (V30's raw-descriptor
trigger moved the frontier but exceeded 1% disagreement everywhere). V31 is
frozen before its own A evaluation; the same subjects had been evaluated in V30.

Question: if each auxiliary replaces V30's fixed max-change rule by a tiny
logistic detector, trained on B to predict when its own stale scores would change
the fused decision, does self-triggered push pass the unchanged screen?

Per auxiliary i, window t, last push a (all local to node i, no classifier call):
  e_i(t)  = per-axis mean, standard deviation, and mean absolute first difference
            of node i's window (3m numbers)
  x_i(t,a)= [ (e(t)-mu)/s , (e(t)-e(a))/sigma , |e(t)-e(a)|/sigma ,
              max_k |e_k(t)-e_k(a)|/sigma_k , (t-a)/16 ]
  p_i     = logistic(w_i . x_i + b_i)
  push    = reset(t) or t - a >= 16 or p_i > theta
Training (B only): pairs (t, a=t-h), h in AGES, inside one segment; label 1 when
  argmax(q0[t]+q_i[a]+q_j[t]) != argmax(q0[t]+q_i[t]+q_j[t]),
i.e. node i's staleness alone changes the fused output. At most PAIR_CAP pairs per
subject (deterministic), subject-equal weights, L2 logistic with C=1.
mu, s, sigma are fitted on B. theta is chosen on B from THETAS: the lowest-byte
value meeting B saving>=10%, positive>=75%, calls<=2.7, F1 drop<=0.005, and
B disagreement <= 0.75% (a margin for in-sample selection); otherwise the smallest
theta (most conservative). Every fixed theta is also evaluated on A.
Ledger as V30: Rice2 score packets, 134-byte configuration, 45-byte status, and a
charged per-session package (header, identifier, float32 mu, s, sigma, weights,
bias, theta for both auxiliaries). No notices.

Usage: python run_learned_trigger_v31.py selftest|freeze|run --workers 6|summarize
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import gc
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression

from run_oracle_imitation_v29 import original_cases, external_cases, load, events_for, write, read, sha
from run_aux_trigger_v30 import signal_path, dataset_of, trigger_accounting, AXES, WINDOW, MAX_AGE
from run_nested_survival import subject_frame
from validity_diagnostic import longest_error
from continuous_screen_accounting import macro_f1
from run_validity_diagnostic import onset_metrics
from run_matched_calibration import accounting, fixed_baseline_name
from causal_score_codec import HEADER

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'learned_trigger_v31'
THETAS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
AGES = (1, 2, 3, 4, 6, 8, 12, 15)
PAIR_CAP = 6000
B_DISAGREEMENT_MARGIN = 0.0075


# ---------------------------------------------------------------- descriptors and features
def descriptors3(f, item, dataset):
    """e[t, node-1, :] = per-axis mean, std, mean |first difference| of node's own window."""
    m, length = AXES[dataset], WINDOW[dataset]
    out = np.empty((len(f['q']), 2, 3*m))
    for subject in np.unique(f['subjects']):
        at = np.flatnonzero(f['subjects'] == subject)
        signal = np.load(signal_path(item, int(subject)), mmap_mode='r')
        start, stop = f['start_rows'][at], f['end_rows_exclusive'][at]
        assert (stop-start == length).all()
        for node in (1, 2):
            x = np.asarray(signal[:, node*m:(node+1)*m], np.float64)
            finite = np.isfinite(x)
            bad = np.r_[np.zeros((1, m)), np.cumsum(~finite, 0)]
            assert ((bad[stop]-bad[start]) == 0).all()
            x = np.where(finite, x, 0.)
            s1 = np.r_[np.zeros((1, m)), np.cumsum(x, 0)]
            s2 = np.r_[np.zeros((1, m)), np.cumsum(x*x, 0)]
            dd = np.r_[np.zeros((1, m)), np.cumsum(np.abs(np.diff(x, axis=0)), 0)]
            mean = (s1[stop]-s1[start])/length
            out[at, node-1, :m] = mean
            out[at, node-1, m:2*m] = np.sqrt(np.maximum((s2[stop]-s2[start])/length-mean**2, 0.))
            out[at, node-1, 2*m:] = (dd[stop-1]-dd[start])/(length-1)
    return out


def check_descriptors3(f, item, dataset, e, samples=64):
    m = AXES[dataset]
    for t in np.linspace(0, len(e)-1, min(samples, len(e)), dtype=int):
        signal = np.load(signal_path(item, int(f['subjects'][t])), mmap_mode='r')
        w = np.asarray(signal[f['start_rows'][t]:f['end_rows_exclusive'][t]], np.float64)
        for node in (1, 2):
            x = w[:, node*m:(node+1)*m]
            np.testing.assert_allclose(e[t, node-1], np.r_[x.mean(0), x.std(0), np.abs(np.diff(x, axis=0)).mean(0)],
                                       rtol=1e-6, atol=1e-6)
    return min(samples, len(e))


def pair_features(et, ea, age, mu, sd, sig):
    """Rows of x_i(t,a); et/ea are (k, 3m) for one auxiliary."""
    z = (et-ea)/sig
    return np.column_stack([(et-mu)/sd, z, np.abs(z), np.abs(z).max(1), np.asarray(age, float)/16])


def harm(q, t, a, node):
    other = 3-node
    cur = q[t, 0].astype(np.int32)+q[t, node].astype(np.int32)+q[t, other].astype(np.int32)
    stale = q[t, 0].astype(np.int32)+q[a, node].astype(np.int32)+q[t, other].astype(np.int32)
    return (np.argmax(cur, 1) != np.argmax(stale, 1)).astype(int)


def training_pairs(b, e):
    """Deterministic (t, a) pairs inside segments for every B subject."""
    seg = np.cumsum(b['reset'])
    out = []
    for subject in np.unique(b['subjects']):
        at = np.flatnonzero(b['subjects'] == subject)
        t_all, a_all = [], []
        for h in AGES:
            t = at[at-h >= at[0]]
            t = t[seg[t] == seg[t-h]]
            t_all.append(t)
            a_all.append(t-h)
        t, a = np.concatenate(t_all), np.concatenate(a_all)
        keep = np.linspace(0, len(t)-1, min(PAIR_CAP, len(t)), dtype=int) if len(t) else np.array([], int)
        out.append((subject, t[np.unique(keep)], a[np.unique(keep)]))
    return out


def fit_detectors(b, e):
    mu = np.empty((2, e.shape[2])); sd = np.empty_like(mu); sig = np.empty_like(mu)
    for node in (1, 2):
        mu[node-1] = e[:, node-1].mean(0)
        sd[node-1] = np.maximum(e[:, node-1].std(0), 1e-6)
        per_subject = []
        for subject in np.unique(b['subjects']):
            at = np.flatnonzero(b['subjects'] == subject)
            step = at[1:][~b['reset'][at[1:]]]
            per_subject.append(np.median(np.abs(e[step, node-1]-e[step-1, node-1]), axis=0))
        sig[node-1] = np.maximum(np.mean(per_subject, 0), 1e-6)
    pairs = training_pairs(b, e)
    W, bias, diag = [], [], {}
    for node in (1, 2):
        xs, ys, ws = [], [], []
        for subject, t, a in pairs:
            if not len(t):
                continue
            xs.append(pair_features(e[t, node-1], e[a, node-1], t-a, mu[node-1], sd[node-1], sig[node-1]))
            ys.append(harm(b['q'], t, a, node))
            ws.append(np.full(len(t), 1/len(t)))
        x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
        w = w*len(w)/w.sum()
        if len(np.unique(y)) == 1:
            p = (float((w*y).sum())+.5)/(w.sum()+1)
            W.append(np.zeros(x.shape[1])); bias.append(np.log(p/(1-p)))
        else:
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(x, y, sample_weight=w)
            W.append(clf.coef_[0].astype(np.float64)); bias.append(float(clf.intercept_[0]))
        diag[f'node{node}'] = dict(pairs=int(len(y)), positive_rate=float(y.mean()))
    return dict(mu=mu, sd=sd, sig=sig, W=np.array(W), b=np.array(bias)), diag


def detector_package(det, theta):
    body = b''.join(np.r_[det['mu'][i], det['sd'][i], det['sig'][i], det['W'][i], det['b'][i], theta].astype('<f4').tobytes()
                    for i in range(2))
    return HEADER.pack(100, 0, 0, 0)+hashlib.sha256(body).digest()[:16]+body


def quantized(det):
    """The deployed detector uses the float32 values that are actually transmitted."""
    return {k: np.asarray(v, np.float32).astype(np.float64) for k, v in det.items()}


# ---------------------------------------------------------------- schedule
@njit(cache=True)
def learned_schedule(q, reset, e, mu, sd, sig, W, b, theta, max_age):
    n, k = len(q), e.shape[2]
    send = np.zeros((n, 3), np.bool_)
    send[:, 0] = True
    anchor = np.zeros(3, np.int64)
    prediction = np.empty(n, np.int16)
    logit_threshold = np.log(theta/(1-theta))
    for t in range(n):
        for node in (1, 2):
            i = node-1
            age = t-anchor[node]
            fire = bool(reset[t]) or age >= max_age
            if not fire:
                s = b[i]
                zmax = 0.
                for j in range(k):
                    z = (e[t, i, j]-e[anchor[node], i, j])/sig[i, j]
                    s += W[i, j]*(e[t, i, j]-mu[i, j])/sd[i, j]+W[i, k+j]*z+W[i, 2*k+j]*abs(z)
                    if abs(z) > zmax:
                        zmax = abs(z)
                s += W[i, 3*k]*zmax+W[i, 3*k+1]*age/16.
                fire = s > logit_threshold
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


def verify_learned(q, reset, e, det, theta, send, prediction):
    """Independent replay with pair_features and a logistic probability."""
    anchor = [0, 0, 0]
    assert send[:, 0].all() and send[reset].all()
    for t in range(len(q)):
        for node in (1, 2):
            i = node-1
            age = t-anchor[node]
            if reset[t] or age >= MAX_AGE:
                expected = True
            else:
                x = pair_features(e[t:t+1, i], e[anchor[node]:anchor[node]+1, i], [age], det['mu'][i], det['sd'][i], det['sig'][i])[0]
                p = 1/(1+np.exp(-(x@det['W'][i]+det['b'][i])))
                if abs(p-theta) < 1e-9:
                    expected = bool(send[t, node])      # numerically tied; accept the compiled decision
                else:
                    expected = bool(p > theta)
            assert bool(send[t, node]) == expected, (t, node)
            if expected:
                anchor[node] = t
        fused = q[t, 0].astype(int)+q[anchor[1], 1].astype(int)+q[anchor[2], 2].astype(int)
        assert int(np.argmax(fused)) == int(prediction[t])


# ---------------------------------------------------------------- B selection and A evaluation
def b_screen(b, e, det, codec):
    rows = []
    for subject in np.unique(b['subjects']):
        sf = subject_frame(b, subject)
        es = e[b['subjects'] == subject]
        n = len(sf['q'])
        base, _, _ = accounting(sf, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
        known = sf['endpoint_targets'] != -1
        classes = sf['classes']
        ref_f1 = macro_f1(sf['endpoint_targets'][known], classes[sf['reference'][known]], classes)
        for theta in THETAS:
            send, pred = learned_schedule(sf['q'], sf['reset'], es, det['mu'], det['sd'], det['sig'], det['W'], det['b'], theta, MAX_AGE)
            total, _, _ = trigger_accounting(sf, send, len(detector_package(det, theta)))
            f1 = macro_f1(sf['endpoint_targets'][known], classes[pred[known]], classes)
            rows.append(dict(subject=int(subject), theta=theta, bytes=total.mean(), base=base.mean(), calls=send.sum()/n,
                             disagreement=float((pred != sf['reference']).mean()), f1_drop=ref_f1-f1))
    r = pd.DataFrame(rows)
    r['positive'] = r.bytes < r.base
    g = r.groupby('theta', as_index=False).mean(numeric_only=True)
    g['saving'] = 1-g.bytes/g.base
    g['qualifies'] = ((g.saving >= .1) & (g.positive >= .75) & (g.calls <= 2.7) & (g.disagreement <= B_DISAGREEMENT_MARGIN)
                      & (g.f1_drop <= .005))
    ok = g[g.qualifies]
    chosen = float(ok.sort_values(['bytes', 'theta']).theta.iloc[0]) if len(ok) else float(min(THETAS))
    return chosen, g


def evaluate(f, e, det, chosen, meta, codec, events, verify=True):
    n = len(f['q'])
    subject = int(f['subjects'][0])
    classes, reference = f['classes'], f['reference']
    known = f['endpoint_targets'] != -1
    ref_f1 = macro_f1(f['endpoint_targets'][known], classes[reference[known]], classes)
    baseline_total, _, _ = accounting(f, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
    common = dict(**meta, subject=subject, windows=n)
    rows, onsets = [], []
    for method, theta in [('learntrig_sel', chosen)]+[(f'learntrig_t{t:g}', t) for t in THETAS]:
        send, pred = learned_schedule(f['q'], f['reset'], e, det['mu'], det['sd'], det['sig'], det['W'], det['b'], theta, MAX_AGE)
        if verify and method == 'learntrig_sel':
            verify_learned(f['q'], f['reset'], e, det, theta, send, pred)
        total, channel, fees = trigger_accounting(f, send, len(detector_package(det, theta)))
        f1 = macro_f1(f['endpoint_targets'][known], classes[pred[known]], classes)
        row = dict(**common, method=method, theta=theta, deployable=True,
            baseline_bytes_per_window=float(baseline_total.mean()), total_bytes=int(total.sum()),
            bytes_per_window=float(total.mean()), **fees,
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
    return dict(version='learned_trigger_v31', client_date='2026-11-11', post_hoc=True,
        designed_after='V30 A results were seen; same subjects as V30',
        question='Does a B-trained logistic auxiliary-side change detector make self-triggered push pass the unchanged screen?',
        descriptor='per-axis mean, standard deviation, mean absolute first difference of the auxiliary own window',
        features='standardized level, normalized change since last push, its absolute value, max absolute change, age/16',
        label='staleness of this auxiliary alone changes the fused argmax (other nodes current)',
        training=dict(ages=list(AGES), pair_cap_per_subject=PAIR_CAP, weights='subject-equal', model='L2 logistic, C=1'),
        push_rule='reset or age >= 16 or p > theta; independent per auxiliary; no notice/request messages',
        thetas=list(THETAS), selection=f'B only: lowest B bytes among thetas meeting B saving>=10%, positive>=75%, calls<=2.7, F1 drop<=0.005, disagreement<={B_DISAGREEMENT_MARGIN}; else smallest theta',
        accounting='V30 ledger; float32 detector package charged per session; no notices',
        screen='V20 joint screen unchanged; deployable method learntrig_sel must pass all six original scenarios (separately all three external)',
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
    eb = descriptors3(b, item, dataset)
    checked = check_descriptors3(b, item, dataset, eb)
    det, diag = fit_detectors(b, eb)
    det = quantized(det)
    chosen, grid = b_screen(b, eb, det, codec)
    dest.mkdir(parents=True, exist_ok=True)
    write(dest/'b_selection.json', dict(key=item['key'], chosen_theta=chosen, detector=det, training=diag, descriptor_checks=checked,
          package_bytes=len(detector_package(det, chosen)), grid=grid.to_dict('records'), A_loaded=False,
          created_utc=datetime.now(timezone.utc).isoformat()))
    a = load(item, 'a')
    ea = descriptors3(a, item, dataset)
    check_descriptors3(a, item, dataset, ea, 16)
    meta = dict(dataset=dataset, outer_fold=case['outer_fold'],
                group=case.get('group', 0) if not item['external'] else 0, direction=case.get('direction', 0) if not item['external'] else 0,
                stride=item['stride'], offset=item['offset'])
    rows, onsets = [], []
    for subject in case['evaluation_subjects']:
        at = a['subjects'] == subject
        sf = subject_frame(a, subject) if not item['external'] else a
        r, o = evaluate(sf, ea[at] if not item['external'] else ea, det, chosen, meta, codec, events_for(item, subject))
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
            print(f'V31 {i+1}/{len(items)} {key} {"done" if new else "verified"}', flush=True)
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
    sel = [read(p/'b_selection.json')['chosen_theta'] for p in folders]
    write(OUT/'research_decision.json', dict(version='learned_trigger_v31', passing=decision, case_grids=len(folders),
          deployable_method='learntrig_sel', chosen_theta_counts={str(k): int(v) for k, v in pd.Series(sel).value_counts().items()}))
    print(equal[['dataset', 'stride', 'offset', 'method', 'savings_fraction', 'reference_disagreement',
                 'deployable_full_models_per_window', 'positive_subject_fraction', 'all_checks_passed']].to_string(index=False))
    print(decision)


def selftest():
    """B-frame-only checks; never loads an A frame."""
    item = next(external_cases())
    b = load(item, 'b')
    e = descriptors3(b, item, 'opportunity')
    print('descriptor windows checked:', check_descriptors3(b, item, 'opportunity', e, 256))
    det, diag = fit_detectors(b, e)
    det = quantized(det)
    print('training pairs and positive rates:', diag, 'package bytes:', len(detector_package(det, .05)))
    sf = subject_frame(b, int(b['subjects'][0]))
    es = e[b['subjects'] == int(b['subjects'][0])]
    for theta in (THETAS[0], THETAS[-1]):
        send, pred = learned_schedule(sf['q'], sf['reset'], es, det['mu'], det['sd'], det['sig'], det['W'], det['b'], theta, MAX_AGE)
        verify_learned(sf['q'], sf['reset'], es, det, theta, send, pred)
    # Pair features in training equal the compiled schedule's arithmetic at a sampled push decision.
    t = np.arange(20, 40)
    x = pair_features(es[t, 0], es[t-3, 0], np.full(len(t), 3), det['mu'][0], det['sd'][0], det['sig'][0])
    assert np.isfinite(x).all() and x.shape[1] == 3*es.shape[2]+2
    # theta -> 0 forces every push: complete-push fusion.
    always, pred = learned_schedule(sf['q'], sf['reset'], es, det['mu'], det['sd'], det['sig'], det['W'], det['b'], 1e-12, MAX_AGE)
    assert always.all() and (pred == sf['reference']).all()
    print('learned schedule replay checks passed on', len(sf['q']), 'B windows; no A frame loaded')


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
