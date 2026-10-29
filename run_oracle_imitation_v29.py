"""V29: causal imitation of the matched zero-conflict offline schedule.

Question (fixed before any A evaluation): can a scheduler that sees only the
paid 52-feature state, and is trained on B subjects to reproduce the V28
zero-conflict offline refresh choices, recover the offline communication
opportunity under the unchanged V20/V23 protocol and joint screen?

Usage (from the cscwd folder):
    python run_oracle_imitation_v29.py selftest    # B-only harness checks
    python run_oracle_imitation_v29.py freeze      # writes the frozen protocol
    python run_oracle_imitation_v29.py run --workers 6   # all 384 original + 12 external case-grids
    python run_oracle_imitation_v29.py run --external-only
    python run_oracle_imitation_v29.py summarize
Each case-grid is resumable; finished folders are hash-checked and skipped.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import gc
import hashlib
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier

from run_matched_oracle_v28 import edge, optimal
from validity_survival import frame, paid_features, logistic
from run_nested_survival import subject_frame
from lease_prediction import landmarks, fit_logistic, lease_schedule, choose_interval
from adaptive_lease_oracle import verify_lease
from validity_diagnostic import packet_cost, profile, longest_error
from continuous_screen_accounting import latency, macro_f1
from run_validity_diagnostic import onset_metrics
from run_matched_calibration import accounting, fixed_baseline_name
from causal_score_codec import HEADER

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'oracle_imitation_v29'
V20 = ROOT/'results'/'matched_calibration_v20'
V23 = ROOT/'results'/'opportunity_v23'
DATA = ROOT/'results'/'continuous_training_v13'
SCENARIOS = ((1, 0), (2, 0), (2, 1))
INTERVALS = (1, 2, 4, 8, 16)
CUMULATIVE = (2, 4, 8, 16)
GAMMAS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CHARGE, WIDTH = 30, 12
FAMILIES = {
    # Matched capacity: four cumulative logistic heads, same 512-landmark cap and
    # subject-equal weighting as the V20 forecasters; 1,726-byte package.
    'imit_linear': dict(model='logistic', cap=True, fee=True),
    # Small nonlinear deployable model; every serialized byte is charged.
    'imit_boost': dict(model='boost', cap=True, fee=True,
                       params=dict(max_iter=30, max_leaf_nodes=4, learning_rate=0.1,
                                   min_samples_leaf=20, l2_regularization=1.0)),
    # Diagnostic only: larger model, all B landmarks, package fee reported but
    # not charged. Bounds what the paid state can reveal, not a deployable rule.
    'imit_boost_diag': dict(model='boost', cap=False, fee=False,
                            params=dict(max_iter=100, max_leaf_nodes=16, learning_rate=0.1,
                                        min_samples_leaf=20, l2_regularization=1.0)),
}
POLICY_BASE_ID = 120

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json)+'\n', encoding='utf-8')


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value))


# ---------------------------------------------------------------- oracle labels
@njit(cache=True)
def oracle_choices(q, reset, width, charge):
    """V28 recursion; returns the optimal interval for every refresh state a."""
    n = len(q)
    choice = np.zeros(n, np.int16)
    hs = np.array([1, 2, 4, 8, 16])
    starts = 0
    while starts < n:
        end = starts+1
        while end < n and not reset[end]:
            end += 1
        cost = np.zeros(end-starts+1, np.int64)
        for a in range(end-1, starts-1, -1):
            first = end
            for t in range(a+1, min(a+16, end)):
                cw = 0
                fw = 0
                cb = -1
                fb = -1
                for c in range(12):
                    cv = int(q[t, 0, c])+int(q[a, 1, c])+int(q[a, 2, c])
                    fv = int(q[t, 0, c])+int(q[t, 1, c])+int(q[t, 2, c])
                    if cv > cb:
                        cb = cv
                        cw = c
                    if fv > fb:
                        fb = fv
                        fw = c
                if cw != fw:
                    first = t
                    break
            best = np.int64(1 << 60)
            ch = 1
            for h in hs:
                b = min(a+h, end)
                if first < b:
                    continue
                value = charge if a+1 < end else 0
                if b < end:
                    value += edge(q, a, b, width)+cost[b-starts]
                if value < best or (value == best and h > ch):
                    best = value
                    ch = h
            cost[a-starts] = best
            choice[a] = ch
        starts = end
    return choice


def labelled_landmarks(b, capped):
    """Paid features at age-one landmarks of B and the oracle choice at their anchor."""
    data = landmarks(b, capped)
    labels = np.zeros(len(b['q']), np.int16)
    for subject in np.unique(b['subjects']):
        at = np.flatnonzero(b['subjects'] == subject)
        sf = subject_frame(b, subject)
        labels[at] = oracle_choices(sf['q'], sf['reset'], WIDTH, CHARGE)
    y = labels[data['anchor']]
    assert np.isin(y, INTERVALS).all()
    return data['x'], y, data['subjects']


def check_oracle_labels(f):
    """Following the labels from each segment start must reproduce V28 optimal()."""
    checked = 0
    for subject in np.unique(f['subjects']):
        sf = subject_frame(f, subject)
        choice = oracle_choices(sf['q'], sf['reset'], WIDTH, CHARGE)
        sent, grants, _ = optimal(sf['q'], sf['reset'], WIDTH, CHARGE)
        decisions = np.flatnonzero(grants)
        np.testing.assert_array_equal(grants[decisions], choice[decisions-1])
        checked += len(decisions)
    return checked


# ---------------------------------------------------------------- models
def subject_weights(subjects):
    weights = np.zeros(len(subjects))
    labels, counts = np.unique(subjects, return_counts=True)
    for subject, count in zip(labels, counts):
        weights[subjects == subject] = len(subjects)/(len(labels)*count)
    return weights


def fit_family(name, x, y, subjects):
    spec = FAMILIES[name]
    heads = []
    for h in CUMULATIVE:
        target = (y >= h).astype(int)
        if spec['model'] == 'logistic':
            fit = fit_logistic(x, target, subjects)
            heads.append(dict(kind='logistic', coefficient=np.asarray(fit['coefficient'], float),
                              intercept=float(fit['intercept']), positives=int(target.sum())))
        else:
            w = subject_weights(subjects)
            if len(np.unique(target)) == 1:
                p = float(((w*target).sum()+.5)/(w.sum()+1))
                heads.append(dict(kind='constant', baseline=float(np.log(p/(1-p))), trees=[], positives=int(target.sum())))
                continue
            clf = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **spec['params'])
            clf.fit(x, target, sample_weight=w)
            heads.append(dict(kind='boost', baseline=float(np.asarray(clf._baseline_prediction).ravel()[0]),
                              trees=[_tree(p[0]) for p in clf._predictors], positives=int(target.sum()), clf=clf))
    return dict(name=name, heads=heads)


def _tree(predictor):
    nodes = predictor.nodes
    assert not nodes['is_categorical'].any() and len(nodes) < 256
    leaf = nodes['is_leaf'].astype(bool)
    return dict(feature=nodes['feature_idx'].astype(np.int64), leaf=leaf,
                missing_left=nodes['missing_go_to_left'].astype(bool),
                value=np.where(leaf, nodes['value'], nodes['num_threshold']).astype(np.float64),
                left=nodes['left'].astype(np.int64), right=nodes['right'].astype(np.int64))


def package(model):
    """Deterministic deployment packet; its length is the charged model fee."""
    body = b''
    for head in model['heads']:
        if head['kind'] == 'logistic':
            body += np.r_[head['coefficient'], head['intercept']].astype('<f8').tobytes()
        else:
            body += struct.pack('<dH', head['baseline'], len(head['trees']))
            for tree in head['trees']:
                body += struct.pack('<B', len(tree['value']))
                for k in range(len(tree['value'])):
                    body += struct.pack('<BBdBB', tree['feature'][k], int(tree['leaf'][k]) | (int(tree['missing_left'][k]) << 1),
                                        tree['value'][k], tree['left'][k], tree['right'][k])
    kind = 97 if model['heads'][0]['kind'] == 'logistic' else 98
    identifier = hashlib.sha256(body).digest()[:16]
    return HEADER.pack(kind, 0, 0, 0)+identifier+body


def raw_scores(model, x):
    """Per-head logits computed only from the serialized parameters."""
    out = np.empty((len(x), len(CUMULATIVE)))
    for j, head in enumerate(model['heads']):
        if head['kind'] == 'logistic':
            out[:, j] = x@head['coefficient']+head['intercept']
            continue
        value = np.full(len(x), head['baseline'])
        rows = np.arange(len(x))
        for tree in head['trees']:
            node = np.zeros(len(x), np.int64)
            while True:
                internal = ~tree['leaf'][node]
                if not internal.any():
                    break
                go_left = x[rows, tree['feature'][node]] <= tree['value'][node]
                node = np.where(internal, np.where(go_left, tree['left'][node], tree['right'][node]), node)
            value += tree['value'][node]
        out[:, j] = value
    return out


def cumulative_probabilities(model, x):
    p = logistic(raw_scores(model, x))
    return np.minimum.accumulate(p, axis=1)


def intervals_from(p, gamma):
    result = np.ones(len(p), np.int16)
    for j, h in enumerate(CUMULATIVE):
        result[p[:, j] >= gamma] = h
    return result


def verify_model(model, x):
    """Serialized-parameter predictions must equal the fitted library predictions."""
    for j, head in enumerate(model['heads']):
        if head['kind'] == 'boost':
            expected = head['clf'].decision_function(x)
            np.testing.assert_allclose(raw_scores(dict(heads=[head]*len(CUMULATIVE)), x)[:, 0], expected, atol=1e-9, rtol=0)


# ---------------------------------------------------------------- schedule and accounting
@njit(cache=True)
def table_schedule(q, reset, table):
    """Identical state machine to lease_schedule; the age-one choice is read from table[t]."""
    n = len(q)
    send = np.zeros((n, 3), np.bool_)
    send[:, 0] = True
    grants = np.zeros(n, np.int16)
    prediction = np.empty(n, np.int16)
    anchor, due = 0, -1
    for t in range(n):
        refresh = bool(reset[t])
        if not refresh:
            if due == t:
                refresh = True
            elif t == anchor+1 and due < 0:
                h = table[t]
                grants[t] = h
                due = anchor+h
                refresh = due == t
        if refresh:
            send[t, 1:] = True
            anchor, due = t, -1
        best, winner = -1, 0
        for c in range(12):
            value = int(q[t, 0, c])+int(q[anchor, 1, c])+int(q[anchor, 2, c])
            if value > best:
                best, winner = value, c
        prediction[t] = winner
    return send, grants, prediction


def age_one_table(f, model, gamma):
    n = len(f['q'])
    segment = np.cumsum(f['reset'])
    t = np.arange(1, n)
    t = t[segment[t] == segment[t-1]]
    table = np.ones(n, np.int16)
    if len(t):
        table[t] = intervals_from(cumulative_probabilities(model, paid_features(f['q'], t, t-1)), gamma)
    return table


def policy_accounting(f, sent, grants, weights, charged, policy_id, threshold):
    """Same charges as run_matched_calibration.accounting with an explicit packet."""
    stamps = np.rint(f['end_time']*1e6).astype(np.int64)
    score, inverses = packet_cost(f['q'], sent, f['reset'], f['continuous_global_indices'], stamps, 'rice2')
    assert sum(map(len, profile(policy_id, 16, threshold))) == 134
    setup = 45*f['status'].astype(np.int32)+134*f['session_start'].astype(np.int32)
    package_bytes = len(weights)*f['session_start'].astype(np.int32)
    deployment = package_bytes if charged else 0*package_bytes
    notices = 30*(grants > 0).astype(np.int32)
    total = score+setup+deployment+notices
    messages = sent.sum(1)+2*(grants > 0)+3*f['status']+5*f['session_start']
    channel = latency(f['end_time'], np.arange(len(sent)), np.ones(len(sent), int), total, messages,
                      np.zeros(len(sent), int), np.zeros(len(sent), int), 0)['latency_250000_1_ms']
    return total, channel, dict(score_bytes=int(score.sum()), grant_bytes=int(notices.sum()),
        configuration_status_bytes=int(setup.sum()), weight_deployment_bytes=int(np.sum(deployment)),
        uncharged_package_bytes=int(package_bytes.sum())-int(np.sum(deployment)),
        package_bytes_per_session=len(weights), setup_status_bytes=int((setup+deployment).sum()),
        packet_inverse_samples=inverses)


def evaluate(f, meta, models, codec, events, check_accounting=None):
    n = len(f['q'])
    subject = int(f['subjects'][0])
    classes, reference = f['classes'], f['reference']
    known = f['endpoint_targets'] != -1
    ref_f1 = macro_f1(f['endpoint_targets'][known], classes[reference[known]], classes)
    baseline_total, _, _ = accounting(f, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
    common = dict(**meta, subject=subject, windows=n)
    rows, onsets = [], []
    policy_id = POLICY_BASE_ID
    for name, model in models.items():
        weights = package(model)
        charged = FAMILIES[name]['fee']
        for gamma in GAMMAS:
            threshold = float(np.float32(gamma))
            table = age_one_table(f, model, threshold)
            sent, grants, pred = table_schedule(f['q'], f['reset'], table)
            verify_lease(f['q'], f['reset'], sent, grants, pred)
            decided = np.flatnonzero(grants)
            np.testing.assert_array_equal(grants[decided], table[decided])
            total, channel, fees = policy_accounting(f, sent, grants, weights, charged, policy_id, threshold)
            f1 = macro_f1(f['endpoint_targets'][known], classes[pred[known]], classes)
            method = f'{name}_g{gamma:g}'
            row = dict(**common, method=method, family=name, gamma=gamma, deployable=charged,
                baseline_method='complete_'+codec, baseline_bytes_per_window=float(baseline_total.mean()),
                total_bytes=int(total.sum()), bytes_per_window=float(total.mean()),
                fee_free_bytes_per_window=float((total.sum()-fees['weight_deployment_bytes'])/n), **fees,
                grant_decisions=int(len(decided)), mean_granted_interval=float(grants[decided].mean()) if len(decided) else None,
                full_model_calls=int(sent.sum()), full_models_per_window=float(sent.sum()/n),
                deployable_full_models_per_window=float(sent.sum()/n), controller_calls=int(len(decided)),
                reference_mismatches=int((pred != reference).sum()), reference_disagreement=float((pred != reference).mean()),
                known_endpoint_macro_f1=f1, known_endpoint_reference_f1=ref_f1, known_endpoint_macro_f1_drop=ref_f1-f1,
                longest_reference_error_windows=longest_error(pred, reference, f['reset']),
                channel_p95_ms=float(np.quantile(channel, .95)), known_windows=int(known.sum()), unknown_windows=int((~known).sum()))
            for node in range(3):
                row[f'node{node}_calls'] = int(sent[:, node].sum())
            for kind in ('pure_known', 'mixed', 'unknown_only'):
                at = f['window_kind'] == kind
                row[kind+'_windows'] = int(at.sum())
                row[kind+'_reference_disagreement'] = float((pred != reference)[at].mean()) if at.any() else None
            rows.append(row)
            onsets += onset_metrics(f, pred, channel, events, dict(**common, method=method))
            policy_id += 1
    if check_accounting is not None:
        # Harness check: a frozen V20/V23 direct model through both accounting paths.
        model, threshold = check_accounting
        sent, grants, _ = lease_schedule(f['q'], f['reset'], np.asarray(model['matrix']), np.asarray(model['intercept']), 1, threshold)
        from run_matched_calibration import pack
        mine, _, _ = policy_accounting(f, sent, grants, pack(model), True, 1, threshold)
        theirs, _, _ = accounting(f, sent, grants, model, 'rice2', False, 1, 16, threshold)
        np.testing.assert_array_equal(mine, theirs)
    return rows, onsets


# ---------------------------------------------------------------- cases
def original_cases():
    for case in read(V20/'plan.json')['cases']:
        for stride, offset in SCENARIOS:
            folder = V20/'score_views'/case['id']
            yield dict(case=case, stride=stride, offset=offset, external=False,
                       b=folder/'calibration.npz', a=folder/'evaluation.npz', key=f"{case['id']}/stride{stride}_offset{offset}")


def external_cases():
    for a in range(1, 5):
        for stride, offset in SCENARIOS:
            folder = V23/'controllers'/f'A{a}'/f'stride{stride}_offset{offset}'
            case = read(V23/'controllers'/f'A{a}'/'stride1_offset0'/'lineage.json')['case']
            yield dict(case=case, stride=stride, offset=offset, external=True, b=folder/'B_frame.npz', a=folder/'A_frame.npz',
                       key=f'opportunity/A{a}/stride{stride}_offset{offset}')


def load(item, role):
    path = item[role]
    if item['external']:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return frame(path, item['case']['dataset'], item['stride'], item['offset'])


def events_for(item, subject):
    if item['external']:
        path = V23/'data'/f'subject{subject}'/'annotated_onsets.csv'
    else:
        path = DATA/item['case']['dataset']/'records'/f'subject{subject}'/'annotated_onsets.csv'
    return pd.read_csv(path).to_dict('records')


def protocol():
    return dict(version='oracle_imitation_v29', client_date='2026-10-27', post_hoc=True,
        question='Can a paid-state causal policy trained on B to imitate V28 zero-conflict offline choices recover offline savings under the unchanged protocol?',
        labels='V28 recursion, twelve slots, Rice2 edges, 30-byte notice pair, zero reference conflict, optimal interval at every B refresh state',
        features='identical 52 paid age-one features as V20 forecasters', training_subjects='B only; S node models and A untouched',
        families={k: {kk: vv for kk, vv in v.items()} for k, v in FAMILIES.items()}, gammas=list(GAMMAS),
        decision='largest h in 2/4/8/16 with monotone P(H*>=h) >= gamma, otherwise immediate refresh',
        accounting='V20 accounting: Rice2 score packets, 30-byte notice pair per decision, 134-byte configuration, 45-byte status, serialized model package per session',
        screen='V20 joint screen unchanged; a family passes only if one gamma passes all six original scenarios (separately all three external)',
        diagnostic_family='imit_boost_diag is non-deployable: package bytes reported but not charged; it bounds paid-state predictability',
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
    b = load(item, 'b')
    oracle_checks = check_oracle_labels(b)
    models, diagnostics = {}, {}
    for name, spec in FAMILIES.items():
        x, y, subjects = labelled_landmarks(b, spec['cap'])
        model = fit_family(name, x, y, subjects)
        verify_model(model, x)
        p = cumulative_probabilities(model, x)
        diagnostics[name] = dict(training_landmarks=len(y), package_bytes=len(package(model)),
            label_distribution={str(h): float((y == h).mean()) for h in INTERVALS},
            in_sample={f'g{g:g}': dict(exact=float((intervals_from(p, g) == y).mean()),
                                        too_long=float((intervals_from(p, g) > y).mean())) for g in GAMMAS})
        models[name] = model
    dest.mkdir(parents=True, exist_ok=True)
    write(dest/'b_diagnostics.json', dict(key=item['key'], oracle_decisions_checked=oracle_checks, families=diagnostics,
                                          A_loaded=False, created_utc=datetime.now(timezone.utc).isoformat()))
    (dest/'packages.bin').write_bytes(b''.join(struct.pack('<I', len(package(m)))+package(m) for m in models.values()))
    # A is loaded only after all B fitting is durable.
    a = load(item, 'a')
    case = item['case']
    codec = 'rice3' if item['external'] else fixed_baseline_name(case['dataset'], item['stride'], item['offset'])
    meta = dict(dataset='opportunity' if item['external'] else case['dataset'], outer_fold=case['outer_fold'],
                group=case.get('group', 0) if not item['external'] else 0, direction=case.get('direction', 0) if not item['external'] else 0,
                stride=item['stride'], offset=item['offset'])
    rows, onsets = [], []
    for subject in case['evaluation_subjects']:
        sf = subject_frame(a, subject) if not item['external'] else a
        r, o = evaluate(sf, meta, models, codec, events_for(item, subject))
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
                print(f'V29 {i+1}/{len(items)} {key} {"done" if new else "verified"}', flush=True)
    else:
        for i, item in enumerate(items):
            new = run_one(item)
            print(f'V29 {i+1}/{len(items)} {item["key"]} {"done" if new else "verified"}', flush=True)
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
    tables = []
    for fee_mode in ('charged', 'fee_free'):
        part = rows.copy()
        if fee_mode == 'fee_free':
            part['bytes_per_window'] = part['fee_free_bytes_per_window']
        view, subject, equal = equal_summary(part.drop(columns=['family', 'baseline_method']), KEYS)
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
        equal['fee_mode'] = fee_mode
        tables.append(equal)
    equal = pd.concat(tables, ignore_index=True)
    equal.to_csv(OUT/'dataset_equal_metrics.csv', index=False)
    decision = {}
    for fee_mode, part in equal.groupby('fee_mode'):
        original = part[part.dataset != 'opportunity']
        external = part[part.dataset == 'opportunity']
        decision[fee_mode] = dict(
            all_six_original=[m for m, g in original.groupby('method') if len(g) == 6 and g.all_checks_passed.all()],
            all_three_external=[m for m, g in external.groupby('method') if len(g) == 3 and g.all_checks_passed.all()])
    write(OUT/'research_decision.json', dict(version='oracle_imitation_v29', passing=decision,
          case_grids=len(folders), deployable_decision_uses='charged', diagnostic_family='imit_boost_diag'))
    print(json.dumps(decision, indent=2))


def selftest():
    """B-frame-only harness checks; never loads an A frame."""
    item = next(external_cases())
    b = load(item, 'b')
    sf = subject_frame(b, int(b['subjects'][0]))
    print('oracle decisions matching V28:', check_oracle_labels(b))
    models = read(V23/'controllers'/'A1'/'stride1_offset0'/'models.json')
    model = models['direct']
    for key in ('matrix', 'intercept', 'parameters'):
        model[key] = np.asarray(model[key], float)
    threshold = float(np.float32(.05))
    sent, grants, pred = lease_schedule(sf['q'], sf['reset'], model['matrix'], model['intercept'], 1, threshold)
    table = np.ones(len(sf['q']), np.int16)
    segment = np.cumsum(sf['reset'])
    for t in range(1, len(table)):
        if segment[t] == segment[t-1]:
            table[t] = choose_interval(sf['q'][t, 0], sf['q'][t-1], model['matrix'], model['intercept'], 1, threshold)
    mine = table_schedule(sf['q'], sf['reset'], table)
    for x, y in zip(mine, (sent, grants, pred)):
        np.testing.assert_array_equal(x, y)
    from run_matched_calibration import pack
    ours, _, _ = policy_accounting(sf, sent, grants, pack(model), True, 1, threshold)
    theirs, _, _ = accounting(sf, sent, grants, model, 'rice2', False, 1, 16, threshold)
    np.testing.assert_array_equal(ours, theirs)
    print('table_schedule == lease_schedule and accounting identical on', len(sf['q']), 'B windows')


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
        if args.action == 'freeze':
            freeze()
        elif args.action == 'selftest':
            selftest()
        elif args.action == 'run':
            run(args.external_only, args.limit, args.workers)
        else:
            summarize()
