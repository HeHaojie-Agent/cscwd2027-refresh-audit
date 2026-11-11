"""V30b: post hoc frontier extension of V30 (declared after V30 A results were seen).

V30 failed only the 1% disagreement gate in all nine scenarios at its smallest
frozen kappa (1.0). This run adds lower fixed kappas so that the auxiliary-trigger
frontier can be compared with forecasters and imitators at the screen's own 1%
limit. It changes nothing in V30: same descriptor, B-fitted sigma, push rule,
ledger and screen; the deployable V30 result (auxtrig_sel) stays as frozen.
Points here are a frontier description, not a deployable selection.

Usage: python run_aux_trigger_v30b.py run --workers 6 ; python run_aux_trigger_v30b.py summarize
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import run_aux_trigger_v30 as v30
from run_aux_trigger_v30 import (descriptors, check_descriptors, fit_scale, trigger_schedule, verify_trigger,
                                 trigger_package, trigger_accounting, dataset_of, original_cases, external_cases,
                                 load, events_for, write, read, sha, MAX_AGE, ROOT)
from run_nested_survival import subject_frame
from validity_diagnostic import longest_error
from continuous_screen_accounting import macro_f1
from run_validity_diagnostic import onset_metrics
from run_matched_calibration import accounting, fixed_baseline_name

OUT = ROOT/'results'/'aux_trigger_v30b'
KAPPAS = (0.5, 0.6, 0.75, 0.9)


def run_one(item):
    dest = OUT/'cases'/item['key']
    done = dest/'complete.json'
    if done.exists():
        return False
    started = time.perf_counter()
    dataset = dataset_of(item)
    case = item['case']
    codec = 'rice3' if item['external'] else fixed_baseline_name(case['dataset'], item['stride'], item['offset'])
    b = load(item, 'b')
    scale = fit_scale(b, descriptors(b, item, dataset))
    frozen = read(v30.OUT/'cases'/item['key']/'b_selection.json')
    np.testing.assert_allclose(scale, np.asarray(frozen['scale']), rtol=1e-12)   # identical B fit as V30
    a = load(item, 'a')
    da = descriptors(a, item, dataset)
    check_descriptors(a, item, dataset, da, 16)
    meta = dict(dataset=dataset, outer_fold=case['outer_fold'],
                group=case.get('group', 0) if not item['external'] else 0, direction=case.get('direction', 0) if not item['external'] else 0,
                stride=item['stride'], offset=item['offset'])
    rows, onsets = [], []
    for subject in case['evaluation_subjects']:
        at = a['subjects'] == subject
        f = subject_frame(a, subject) if not item['external'] else a
        d = da[at] if not item['external'] else da
        n = len(f['q'])
        classes, reference = f['classes'], f['reference']
        known = f['endpoint_targets'] != -1
        ref_f1 = macro_f1(f['endpoint_targets'][known], classes[reference[known]], classes)
        base, _, _ = accounting(f, np.ones((n, 3), bool), np.zeros(n, np.int16), None, codec, True)
        common = dict(**meta, subject=int(subject), windows=n)
        events = events_for(item, subject)
        for kappa in KAPPAS:
            send, pred = trigger_schedule(f['q'], f['reset'], d, scale, kappa, MAX_AGE)
            verify_trigger(f['q'], f['reset'], d, scale, kappa, send, pred)
            total, channel, fees = trigger_accounting(f, send, len(trigger_package(scale, kappa)))
            f1 = macro_f1(f['endpoint_targets'][known], classes[pred[known]], classes)
            method = f'auxtrig_k{kappa:g}'
            rows.append(dict(**common, method=method, kappa=kappa, deployable=True, baseline_bytes_per_window=float(base.mean()),
                total_bytes=int(total.sum()), bytes_per_window=float(total.mean()), **fees,
                full_model_calls=int(send.sum()), full_models_per_window=float(send.sum()/n), deployable_full_models_per_window=float(send.sum()/n),
                auxiliary_push_fraction=float(send[:, 1:].mean()), reference_mismatches=int((pred != reference).sum()),
                reference_disagreement=float((pred != reference).mean()), known_endpoint_macro_f1=f1, known_endpoint_reference_f1=ref_f1,
                known_endpoint_macro_f1_drop=ref_f1-f1, longest_reference_error_windows=longest_error(pred, reference, f['reset']),
                channel_p95_ms=float(np.quantile(channel, .95)), known_windows=int(known.sum()), unknown_windows=int((~known).sum()),
                node0_calls=int(send[:, 0].sum()), node1_calls=int(send[:, 1].sum()), node2_calls=int(send[:, 2].sum())))
            onsets += onset_metrics(f, pred, channel, events, dict(**common, method=method))
    dest.mkdir(parents=True, exist_ok=True)
    write(dest/'metrics.json', rows)
    write(dest/'onsets.json', onsets)
    write(done, dict(elapsed_seconds=time.perf_counter()-started, v30_source_sha256=sha(v30.__file__), source_sha256=sha(__file__)))
    return True


def _worker(item):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return item['key'], run_one(item)


def run(workers):
    v30.assert_frozen()
    items = list(original_cases())+list(external_cases())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max(1, workers)) as pool:
        for i, (key, new) in enumerate(pool.map(_worker, items, chunksize=1)):
            print(f'V30b {i+1}/{len(items)} {key} {"done" if new else "skipped"}', flush=True)
    write(OUT/'run_complete.json', dict(case_grids=len(items), completed_utc=datetime.now(timezone.utc).isoformat(), post_hoc=True))


def summarize():
    from summarize_matched_calibration import equal_summary, recovery, recovery_mixture, KEYS, VIEW
    from summarize_validity_diagnostic import checks
    folders = sorted(p.parent for p in (OUT/'cases').rglob('complete.json'))
    assert len(folders) == 396, len(folders)
    rows = pd.DataFrame([r for p in folders for r in read(p/'metrics.json')])
    events = pd.DataFrame([e for p in folders for e in read(p/'onsets.json')])
    rows = rows.merge(recovery(events), on=VIEW+['direction'], validate='one_to_one')
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
    print(equal[['dataset', 'stride', 'offset', 'method', 'savings_fraction', 'reference_disagreement', 'all_checks_passed', 'failed_checks']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'summarize'])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.workers) if args.action == 'run' else summarize()
