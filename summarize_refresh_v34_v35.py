"""Verify the refresh replays and compare all nine scenarios under each ledger."""

from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np
import pandas as pd

import run_targeted_refresh_v34 as v34
import run_compact_refresh_v35 as v35
import run_published_baselines_v33 as v33
from run_oracle_imitation_v29 import read, sha, write

SCENARIO = ['dataset', 'stride', 'offset']
STRATA = ['subject', 'outer_fold', 'group', 'direction']


def aggregate_recovery(records):
    quantiles = []
    for key, person_views in records.items():
        weights = defaultdict(Fraction)
        for views in person_views.values():
            for directions in views.values():
                for histogram in directions.values():
                    denominator = len(person_views) * len(views) * len(directions) * sum(histogram.values())
                    for window, count in histogram.items():
                        weights[window] += Fraction(count, denominator)
        assert sum(weights.values()) == 1
        accumulated = Fraction(0)
        for window, weight in sorted(weights.items()):
            accumulated += weight
            if accumulated >= Fraction(95, 100):
                quantiles.append(dict(zip(v34.KEYS+['criterion'], key), p95=window))
                break
    return pd.DataFrame(quantiles)


def summarize(out, packet_field, expected_methods):
    folders = sorted(p.parent for p in (out / 'cases').rglob('complete.json'))
    assert len(folders) == 396
    rows, records, packets = [], {}, 0
    for folder in folders:
        completion = read(folder / 'complete.json')
        assert all(sha(folder / name) == digest for name, digest in completion['files'].items())
        packets += completion[packet_field]
        metrics = pd.DataFrame(read(folder / 'metrics.json'))
        events = pd.DataFrame(read(folder / 'onsets.json'))
        rows.append(metrics.merge(v34.recovery(events), on=v34.VIEW+['direction'], validate='one_to_one'))
        paired = events.dropna(subset=['extra_recovery_windows'])
        indices = v34.KEYS + ['criterion'] + STRATA
        for key, group in paired.groupby(indices, sort=False):
            dataset, stride, offset, method, kind, subject, fold, sub_group, direction = key
            per_method = records.setdefault((dataset, stride, offset, method, kind), {})
            per_subject = per_method.setdefault(subject, {})
            per_view = per_subject.setdefault((fold, sub_group), {})
            assert direction not in per_view
            per_view[direction] = Counter(group.extra_recovery_windows.astype(int).tolist())
    views = pd.concat(rows, ignore_index=True)
    assert len(views) == 780 * expected_methods
    assert not views.duplicated(v34.VIEW+['direction']).any()
    _, subject, summary = v34.equal_summary(views, v34.KEYS)
    subject['positive_subject_fraction'] = subject.bytes_per_window < subject.baseline_bytes_per_window
    summary = summary.merge(subject.groupby(v34.KEYS, as_index=False).positive_subject_fraction.mean(), on=v34.KEYS)
    mixture = aggregate_recovery(records)
    for criterion in ('first_correct', 'two_consecutive_correct'):
        p = mixture[mixture.criterion == criterion][v34.KEYS+['p95']]
        summary = summary.merge(p.rename(columns={'p95': criterion+'_p95_extra_recovery_windows'}), on=v34.KEYS)
    summary['savings_fraction'] = 1 - summary.bytes_per_window / summary.baseline_bytes_per_window
    failures = [[name for name, passed in v34.checks(r).items() if not passed] for r in summary.to_dict('records')]
    summary['all_checks_passed'] = [not f for f in failures]
    summary['failed_checks'] = [','.join(f) for f in failures]
    views.to_csv(out / 'view_metrics.csv', index=False)
    summary.to_csv(out / 'dataset_equal_metrics.csv', index=False)
    subject.to_csv(out / 'subject_metrics.csv', index=False)
    mixture.to_csv(out / 'recovery_mixture.csv', index=False)
    counts = summary.groupby('method').all_checks_passed.sum().to_dict()
    write(out / 'summary_complete.json', dict(case_grids=len(folders), view_rows=len(views),
          packets_verified=packets, exact_rational_recovery_weights=True, passed=counts))
    return views, summary


def check_reference(views, prior):
    keys = SCENARIO + STRATA
    match = views.merge(prior[keys+['baseline_bytes_per_window', 'known_endpoint_reference_f1']], on=keys,
                        suffixes=('', '_v33'), validate='many_to_one')
    assert len(match) == len(views)
    np.testing.assert_allclose(match.baseline_bytes_per_window, match.baseline_bytes_per_window_v33, rtol=0, atol=1e-12)
    np.testing.assert_allclose(match.known_endpoint_reference_f1, match.known_endpoint_reference_f1_v33, rtol=0, atol=1e-12)


def main():
    assert read(v34.OUT / 'frozen.json')['source_sha256'] == sha(v34.__file__)
    assert all(sha(v34.ROOT / name) == value for name, value in read(v35.OUT / 'frozen.json')['source_hashes'].items())
    v34_views, targeted = summarize(v34.OUT, 'main_packet_inverses', 3)
    v35_views, compact = summarize(v35.OUT, 'compact_packets_verified', 7)
    prior = pd.read_csv(v33.OUT / 'view_metrics.csv')
    prior = prior[prior.method == 'bt_supervised_cache_sel']
    check_reference(v34_views, prior)
    check_reference(v35_views, prior)
    base = pd.read_csv(v33.OUT / 'dataset_equal_metrics.csv')
    bt_fixed = base[base.method.str.startswith('bt_') & ~base.method.str.endswith('_sel')]
    eligible = bt_fixed[(bt_fixed.reference_disagreement <= .01) & (bt_fixed.known_endpoint_macro_f1_drop <= .005)
                        & (bt_fixed.deployable_full_models_per_window <= 2.7)]
    # A lower request price leaves these three quantities unchanged.
    old = pd.read_csv(v34.V32 / 'dataset_equal_metrics.csv')
    comparison = pd.concat([
        old[old.method == 'twoside_sel'].assign(notice_bytes=15),
        base[base.method.str.endswith('_sel')].assign(notice_bytes=15),
        targeted.assign(notice_bytes=15),
        compact.assign(notice_bytes=3),
    ], ignore_index=True)
    comparison.to_csv(v35.OUT / 'matched_comparison.csv', index=False)
    used = comparison.method.str.startswith(('twoside', 'targeted', 'compact'))
    print(comparison.loc[used, SCENARIO+['method','savings_fraction','reference_disagreement',
          'known_endpoint_macro_f1','all_checks_passed','failed_checks']].to_string(index=False))
    write(v35.OUT / 'comparison_verified.json', dict(all_passed=True, source_sha256=sha(__file__),
          legacy_cases=396, compact_cases=396, legacy_view_rows=len(v34_views), compact_view_rows=len(v35_views),
          all_reference_bytes_and_f1_match_v33=True, bt_fixed_scenario_points=len(bt_fixed),
          bt_fixed_points_meeting_agreement_f1_and_call_caps=len(eligible),
          new_independent_subjects=0, comparison_sha256=sha(v35.OUT / 'matched_comparison.csv')))


if __name__ == '__main__':
    main()
