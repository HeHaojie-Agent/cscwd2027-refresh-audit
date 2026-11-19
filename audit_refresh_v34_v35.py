"""Independently replay analyzed reserve selectors on every dataset and grid."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import run_targeted_refresh_v34 as v34
import run_compact_refresh_v35 as v35
from run_nested_survival import subject_frame
from run_oracle_imitation_v29 import external_cases, load, original_cases, read, sha, write
from validity_diagnostic import packet_cost


def audit():
    results = []
    seen = set()
    for item in [*original_cases(), *external_cases()]:
        dataset = v34.dataset_of(item)
        key = (dataset, item['stride'], item['offset'])
        if key in seen:
            continue
        seen.add(key)
        p34 = v34.OUT / 'cases' / item['key']
        p35 = v35.OUT / 'cases' / item['key']
        selected = read(p34 / 'b_selection.json')
        config = read(p35 / 'b_selection.json')
        scale = np.asarray(selected['scale'], dtype=np.float32)
        a = load(item, 'a')
        x = v34.descriptors(a, item, dataset) / scale
        codec = 'rice3' if item['external'] else v34.fixed_baseline_name(*key)
        for subject in item['case']['evaluation_subjects']:
            f = subject_frame(a, subject)
            d = x[a['subjects'] == subject]
            n = len(d)
            stamp = np.rint(f['end_time'] * 1e6).astype(np.int64)
            base, _ = packet_cost(f['q'], np.ones((n, 3), bool), f['reset'], f['continuous_global_indices'], stamp, codec)
            base = base + 116*f['session_start'] + v34.STATUS_BYTES*f['status']
            known = f['endpoint_targets'] != -1
            parts = [('targeted_reserve_sel', selected['policies'], p34, 0),
                     ('compact_reserve_sel', config['targeted'], p35, v35.BINDING_BYTES)]
            for method, policies, folder, binding in parts:
                params = policies[method]
                body = v34.package(scale, params[0], params[1]) + bytes((int(params[2]), 0))
                send, req, pred = v34.replay(f['q'], f['reset'], d, *params)
                legacy_charge = v34.accounting(f, send, req, len(body))
                packets = v34.verify(f, d, params, send, req, pred, legacy_charge, len(body))
                alternate = legacy_charge.copy()
                requests = np.count_nonzero(req)
                if binding:
                    assert v35.verify_notices(f, req) == requests
                    alternate += binding*f['session_start'] - v35.DELTA*np.count_nonzero(req, axis=1)
                f1 = f1_score(f['endpoint_targets'][known], f['classes'][pred[known]], labels=f['classes'],
                              average='macro', zero_division=0)
                observed = pd.DataFrame(read(folder / 'metrics.json'))
                row, = observed[(observed.method == method) & (observed.subject == subject)].to_dict('records')
                np.testing.assert_allclose([alternate.mean(), base.mean(), f1, (pred != f['reference']).mean()],
                                           [row['bytes_per_window'], row['baseline_bytes_per_window'],
                                            row['known_endpoint_macro_f1'], row['reference_disagreement']], rtol=0, atol=1e-12)
                # Perturb every score that is unavailable to the receiver at runtime.
                hidden = f['q'].copy()
                hidden[~send] ^= np.uint8(255)
                for actual, expected in zip(v34.replay(hidden, f['reset'], d, *params), (send, req, pred)):
                    np.testing.assert_array_equal(actual, expected)
                results.append(dict(method=method, dataset=dataset, stride=item['stride'], offset=item['offset'],
                                    subject=subject, windows=n, requests=requests, score_inverses=packets))
    assert len(seen) == 9
    write(v35.OUT / 'reserve_replay_audit.json', dict(passed=True, scenarios=9, streams=len(results),
          source_sha256=sha(__file__), streams_detail=results))
    print(f'{len(results)} reserve traces passed independent replay, metric and hidden-score checks.')


if __name__ == '__main__':
    audit()
