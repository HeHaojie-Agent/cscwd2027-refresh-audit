"""Compare targeted refresh and published controls with shared session ticks."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import run_targeted_refresh_v34 as v34
import run_published_baselines_v33 as bt
from protocol_replay_v26 import compact_packet, decode_compact, extension_packets
from run_nested_survival import subject_frame
from run_oracle_imitation_v29 import events_for, external_cases, load, original_cases, read, sha, write

OUT = v34.ROOT / 'results' / 'compact_refresh_v35'
NOTICE = 3
DELTA = v34.NOTICE_BYTES - NOTICE
BINDING_BYTES = 54


def adjusted(table, notice_column, difference, overhead):
    result = table.copy()
    result.bytes_per_window += overhead - difference * result[notice_column]
    result['saving'] = 1 - result.bytes_per_window / result.baseline_bytes_per_window
    return result


def eligible(table, limit):
    return ((table.deployable_full_models_per_window <= 2.7)
            & (table.known_endpoint_macro_f1_drop <= .005) & (table.reference_disagreement <= limit))


def select(item):
    source = v34.OUT / 'cases' / item['key']
    completion = read(source / 'complete.json')
    assert all(sha(source / n) == completion['files'][n] for n in ('b_candidates.csv', 'b_selection.json'))
    config = read(source / 'b_selection.json')
    b = load(item, 'b')
    dataset = v34.dataset_of(item)
    codec = 'rice3' if item['external'] else v34.fixed_baseline_name(dataset, item['stride'], item['offset'])
    scale = np.asarray(config['scale'], np.float32)
    d = v34.descriptors(b, item, dataset) / scale
    size = len(v34.package(scale, 1., 0)) + 2
    subjects = []
    for subject in np.unique(b['subjects']):
        f = subject_frame(b, subject)
        subjects.append((f, d[b['subjects'] == subject], bt.confidence(f), v34.baseline(f, codec).sum()))
    overhead = BINDING_BYTES * np.mean([f['session_start'].mean() for f, _, _, _ in subjects])

    def targeted_positive(parameters):
        flags = []
        for f, x, _, base in subjects:
            send, req, _ = v34.replay(f['q'], f['reset'], x, *parameters)
            total = v34.accounting(f, send, req, size) - DELTA*np.count_nonzero(req, axis=1)
            flags.append(total.sum() + BINDING_BYTES*f['session_start'].sum() < base)
        return np.mean(flags) >= .75

    def published_positive(c):
        flags = []
        for f, _, confidence, base in subjects:
            send, pull, _ = bt.schedule(f['q'], f['reset'], confidence, c['marks'], c['mask'],
                                         c['budget'], c['cache'], c['margin'])
            total = bt.costs(f, send, pull, c['size']) - 2*DELTA*pull
            flags.append(total.sum() + BINDING_BYTES*f['session_start'].sum() < base)
        return np.mean(flags) >= .75

    control = pd.read_csv(source / 'b_candidates.csv', index_col='index')
    control = adjusted(control, 'notices_per_window', DELTA, overhead)
    picked = {}
    for name, limit in [('compact_sel', .01), ('compact_reserve_sel', .0075)]:
        # Retain the V34 fallback if no operating point meets the B quality and call caps.
        original = name.replace('compact', 'targeted')
        admissible = control[eligible(control, limit)].sort_values('bytes_per_window', kind='stable')
        parameters = (v34.GRID[int(i)] for i in admissible.index)
        picked[name] = next((p for p in parameters if targeted_positive(p)), config['policies'][original])

    pub_source = bt.OUT / 'cases' / item['key']
    pub_done = read(pub_source / 'complete.json')['files_sha256']
    assert all(sha(pub_source / p) == pub_done[p] for p in ('b_candidates.csv', 'b_selection.json'))
    pub_config = read(pub_source / 'b_selection.json')
    pub_table = adjusted(pd.read_csv(pub_source / 'b_candidates.csv'), 'pull_fraction', 2*DELTA, overhead)
    pub_candidates = {c['method']: c for c in bt.candidates({k: np.asarray(v) for k, v in pub_config['model'].items()})}
    admissible = pub_table[eligible(pub_table, .01) & (pub_table.saving >= .1)]
    published = {}
    for family in bt.METHODS:
        ranked = admissible[admissible.family == family].sort_values(['bytes_per_window', 'method'])
        candidates = (pub_candidates[name] for name in ranked.method)
        published[family+'_compact_sel'] = next((c for c in candidates if published_positive(c)), None)
    return config['scale'], picked, published


def verify_notices(f, requests):
    session_origin = np.maximum.accumulate(np.where(f['session_start'], np.arange(len(requests)), 0))
    count = 0
    for t, peer in zip(*np.where(requests)):
        assert peer in (1, 2)
        tick = int(t - session_origin[t])
        data = compact_packet(tick, 1)
        assert len(data) == NOTICE and decode_compact(data, tick) == 1
        count += 1
    return count


def run_one(item):
    dest = OUT / 'cases' / item['key']
    done = dest / 'complete.json'
    if done.exists():
        assert all(sha(dest / name) == digest for name, digest in read(done)['files'].items())
        return
    dest.mkdir(parents=True, exist_ok=True)
    scale, picked, published = select(item)
    write(dest / 'b_selection.json', dict(targeted=picked, published=published, A_loaded=False))
    scale = np.asarray(scale, np.float32)
    case, dataset = item['case'], v34.dataset_of(item)
    size = len(v34.package(scale, 1., 0)) + 2
    codec = 'rice3' if item['external'] else v34.fixed_baseline_name(dataset, item['stride'], item['offset'])
    a = load(item, 'a')
    features = v34.descriptors(a, item, dataset) / scale
    rows, onsets, packets = [], [], 0
    for subject in case['evaluation_subjects']:
        f = subject_frame(a, subject)
        x = features[a['subjects'] == subject]
        base = v34.baseline(f, codec)
        ref_f1 = v34.reference_f1(f)
        events = events_for(item, subject)
        conf = bt.confidence(f)
        for at in np.flatnonzero(f['session_start']):
            packet = extension_packets(12, True, f['continuous_global_indices'][at], round(f['end_time'][at]*1e6))
            assert sum(map(len, packet)) == BINDING_BYTES
        for method in [*picked, *published]:
            fallback = False
            if method in picked:
                send, req, pred = v34.replay(f['q'], f['reset'], x, *picked[method])
                notices = np.count_nonzero(req, axis=1)
                total = v34.accounting(f, send, req, size) - DELTA * notices
            else:
                candidate = published[method]
                fallback = candidate is None
                if fallback:
                    send = np.ones_like(f['q'][:, :, 0], dtype=bool)
                    req = np.zeros_like(send)
                    pred, total = f['reference'], base
                else:
                    send, pull, pred = bt.schedule(f['q'], f['reset'], conf, candidate['marks'], candidate['mask'],
                                                   candidate['budget'], candidate['cache'], candidate['margin'])
                    req = np.column_stack([np.zeros_like(pull), pull, pull])
                    total = bt.costs(f, send, pull, candidate['size']) - 2 * DELTA * pull
                notices = np.count_nonzero(req, axis=1)
            packets += verify_notices(f, req)
            messages = send.sum(1) + notices + 3*f['status'] + (4 if fallback else 5)*f['session_start']
            if not fallback:
                total = total + BINDING_BYTES*f['session_start']
                messages = messages + 3*f['session_start']
            n = len(total)
            channel = v34.latency(f['end_time'], np.arange(n), np.ones(n, int), total, messages,
                                  np.zeros(n, int), np.zeros(n, int), 0)['latency_250000_1_ms']
            meta = dict(dataset=dataset, outer_fold=case['outer_fold'], subject=subject,
                        group=case.get('group', 0), direction=case.get('direction', 0),
                        stride=item['stride'], offset=item['offset'], method=method)
            rows.append(dict(**meta, **v34.results(f, pred, total, send, base, ref_f1, req), windows=n,
                             fallback=fallback, channel_p95_ms=float(np.quantile(channel, .95))))
            onsets.extend(v34.onset_metrics(f, pred, channel, events, meta))
    write(dest / 'metrics.json', rows)
    write(dest / 'onsets.json', onsets)
    names = ['b_selection.json', 'metrics.json', 'onsets.json']
    write(done, dict(files={name: sha(dest / name) for name in names}, compact_packets_verified=packets))


def summarize():
    folders = sorted(p.parent for p in (OUT / 'cases').rglob('complete.json'))
    assert len(folders) == 396
    metrics, paired = [], []
    for p in folders:
        table = pd.DataFrame(read(p / 'metrics.json'))
        events = pd.DataFrame(read(p / 'onsets.json'))
        metrics.append(table.merge(v34.recovery(events), on=v34.VIEW+['direction'], validate='one_to_one'))
        keys = v34.VIEW + ['direction', 'criterion', 'extra_recovery_windows']
        paired.append(events.loc[events.extra_recovery_windows.notna(), keys])
    rows = pd.concat(metrics, ignore_index=True)
    mixture = v34.recovery_mixture(pd.concat(paired, ignore_index=True))
    _, subject, summary = v34.equal_summary(rows, v34.KEYS)
    subject['positive_subject_fraction'] = subject.bytes_per_window < subject.baseline_bytes_per_window
    summary = summary.merge(subject.groupby(v34.KEYS, as_index=False).positive_subject_fraction.mean(), on=v34.KEYS)
    for criterion in ('first_correct', 'two_consecutive_correct'):
        p = mixture[mixture.criterion == criterion][v34.KEYS+['p95']]
        summary = summary.merge(p.rename(columns={'p95': criterion+'_p95_extra_recovery_windows'}), on=v34.KEYS)
    summary['savings_fraction'] = 1 - summary.bytes_per_window / summary.baseline_bytes_per_window
    failures = [[name for name, passed in v34.checks(r).items() if not passed] for r in summary.to_dict('records')]
    summary['all_checks_passed'] = [not f for f in failures]
    summary['failed_checks'] = [','.join(f) for f in failures]
    rows.to_csv(OUT / 'view_metrics.csv', index=False)
    summary.to_csv(OUT / 'dataset_equal_metrics.csv', index=False)
    print(summary[['dataset','stride','offset','method','savings_fraction','reference_disagreement',
                   'known_endpoint_macro_f1','all_checks_passed','failed_checks']].to_string(index=False))
    write(OUT / 'summary_complete.json', dict(case_grids=len(folders), view_rows=len(rows),
          compact_packets_verified=sum(read(p / 'complete.json')['compact_packets_verified'] for p in folders),
          passed=summary.groupby('method').all_checks_passed.sum().to_dict(), new_independent_subjects=0))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'summarize'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--ready-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'frozen.json'
    hashes = {p.name: sha(p) for p in map(Path, (__file__, v34.__file__, bt.__file__))}
    if not lock.exists():
        assert args.action == 'run'
        write(lock, dict(source_hashes=hashes, created_utc=datetime.now(timezone.utc).isoformat(),
                         notice_bytes=NOTICE, shared_session_ticks=True, peer_bound_channel=True,
                         binding_bytes_per_session=BINDING_BYTES,
                         existing_development_data=True, new_independent_subjects=0))
    else:
        assert read(lock)['source_hashes'] == hashes
    if args.action == 'summarize':
        summarize()
        return
    items = list(original_cases()) + list(external_cases())
    assert len(items) == 396
    if args.ready_only:
        items = [i for i in items if (v34.OUT / 'cases' / i['key'] / 'complete.json').exists()]
    if args.workers == 1:
        for i, item in enumerate(items, 1):
            run_one(item)
            if i % 12 == 0 or i == len(items):
                print(f'{i}/{len(items)} {item["key"]}', flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(run_one, items, chunksize=1), 1):
            if i % 12 == 0 or i == len(items):
                print(f'{i}/{len(items)} {items[i-1]["key"]}', flush=True)


if __name__ == '__main__':
    main()
