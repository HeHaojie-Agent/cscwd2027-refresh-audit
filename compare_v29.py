"""Join V29 imitation summaries with the frozen V20/V23 policies and V28 offline points."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'oracle_imitation_v29'


def saving(df):
    return 1-df.bytes_per_window/df.baseline_bytes_per_window


def main():
    v29 = pd.read_csv(OUT/'dataset_equal_metrics.csv')
    v29['source'] = 'V29 '+v29.fee_mode
    old = []
    for path in (ROOT/'results/matched_calibration_v20/dataset_equal_metrics.csv', ROOT/'results/opportunity_v23/dataset_equal_metrics.csv'):
        if path.exists():
            d = pd.read_csv(path)
            d['source'] = 'V20' if 'v20' in str(path) else 'V23'
            old.append(d)
    old = pd.concat(old, ignore_index=True) if old else pd.DataFrame()
    cols = ['source', 'dataset', 'stride', 'offset', 'method', 'savings_fraction', 'reference_disagreement',
            'deployable_full_models_per_window', 'positive_subject_fraction', 'known_endpoint_macro_f1_drop']
    if len(old):
        old['savings_fraction'] = saving(old)
        old = old[old.method.str.match(r'(survival|direct|isotonic|platt)_t|periodic_h|complete_')]
    points = pd.concat([v29.assign(savings_fraction=saving(v29))[cols+['all_checks_passed', 'failed_checks']], old[cols] if len(old) else None],
                       ignore_index=True)
    oracle = ROOT/'results/matched_oracle_v28/fee_matched_dataset_equal_metrics.csv'
    if oracle.exists():
        o = pd.read_csv(oracle)
        o = o[(o.charge == 30) & (o.width == 12)]
        points = pd.concat([points, pd.DataFrame(dict(source='V28 offline', dataset=o.dataset, stride=o.stride, offset=o.offset,
                            method='offline_zero_conflict', savings_fraction=o.saving, reference_disagreement=0.,
                            deployable_full_models_per_window=3.))], ignore_index=True)
    points.to_csv(OUT/'frontier_points.csv', index=False)
    primary = points[points.stride == 1].sort_values(['dataset', 'source', 'method'])
    print(primary[['dataset', 'source', 'method', 'savings_fraction', 'reference_disagreement', 'deployable_full_models_per_window',
                   'positive_subject_fraction']].to_string(index=False))


if __name__ == '__main__':
    main()
