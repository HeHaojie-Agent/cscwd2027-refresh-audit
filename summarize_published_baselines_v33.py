"""Bounded-memory subject-equal summaries with exact recovery-mixture weights."""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np,pandas as pd
from run_oracle_imitation_v29 import read,write,sha
from summarize_matched_calibration import equal_summary,recovery,KEYS,VIEW
from summarize_validity_diagnostic import checks

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/published_baselines_v33'
folders=sorted(p.parent for p in (OUT/'cases').rglob('complete.json'));assert len(folders)==396
summary_lock=OUT/'summary_source.json'
if summary_lock.exists():assert read(summary_lock)['sha256']==sha(__file__)
else:write(summary_lock,dict(file=Path(__file__).name,sha256=sha(__file__),
    reason='Stream cases instead of loading all event dictionaries at once; exact rational weights for paired recovery p95'))
metrics=[];histograms=[];packet_count=0
direction_keys=VIEW+['direction','criterion']
for i,p in enumerate(folders,1):
    complete=read(p/'complete.json')
    assert all(sha(p/n)==h for n,h in complete['files_sha256'].items())
    packet_count+=complete['selected_packets_serialized']
    table=pd.DataFrame(read(p/'metrics.json'))
    events=pd.DataFrame(read(p/'onsets.json'))
    table=table.merge(recovery(events),on=VIEW+['direction'],validate='one_to_one')
    metrics.append(table)
    paired=events.dropna(subset=['extra_recovery_windows'])
    for key,part in paired.groupby(direction_keys,sort=False):
        values,counts=np.unique(part.extra_recovery_windows.to_numpy(),return_counts=True)
        histograms.append(dict(zip(direction_keys,key),pair_count=len(part),values=values.tolist(),counts=counts.tolist()))
    if i%60==0:print('Summary case',i,'/',len(folders),flush=True)
rows=pd.concat(metrics,ignore_index=True)
assert not rows.duplicated(VIEW+['direction']).any()
rows.to_csv(OUT/'view_metrics.csv',index=False)
_,subject,mean=equal_summary(rows,KEYS)
subject['positive_savings']=subject.bytes_per_window<subject.baseline_bytes_per_window
positive=subject.groupby(KEYS,as_index=False).positive_savings.mean().rename(columns={'positive_savings':'positive_subject_fraction'})
mean=mean.merge(positive,on=KEYS);mean['savings_fraction']=1-mean.bytes_per_window/mean.baseline_bytes_per_window

hist=pd.DataFrame(histograms);mixture_keys=KEYS+['criterion'];view_keys=mixture_keys+['subject','outer_fold','group']
hist['available_directions']=hist.groupby(view_keys).direction.transform('nunique')
views=hist[view_keys].drop_duplicates().groupby(mixture_keys+['subject'],as_index=False).size().rename(columns={'size':'available_views'})
hist=hist.merge(views,on=mixture_keys+['subject'],validate='many_to_one')
hist['available_subjects']=hist.groupby(mixture_keys).subject.transform('nunique')
mixture=[]
for key,part in hist.groupby(mixture_keys,sort=False):
    distribution={}
    for record in part.itertuples():
        denominator=int(record.pair_count*record.available_directions*record.available_views*record.available_subjects)
        for value,count in zip(record.values,record.counts):
            distribution[value]=distribution.get(value,Fraction(0))+Fraction(int(count),denominator)
    assert sum(distribution.values())==1
    cumulative=Fraction(0);p95=None
    for value,weight in sorted(distribution.items()):
        cumulative+=weight
        if cumulative>=Fraction(95,100):p95=value;break
    mixture.append(dict(zip(mixture_keys,key),p95=p95))
mixture=pd.DataFrame(mixture);mixture.to_csv(OUT/'recovery_mixture.csv',index=False)
for kind in ('first_correct','two_consecutive_correct'):
    part=mixture[mixture.criterion==kind][KEYS+['p95']].rename(columns={'p95':kind+'_p95_extra_recovery_windows'})
    mean=mean.merge(part,on=KEYS,how='left',validate='one_to_one')
flags=[checks(r) for r in mean.to_dict('records')]
mean['all_checks_passed']=[all(g.values()) for g in flags]
mean['failed_checks']=[','.join(k for k,v in g.items() if not v) for g in flags]
mean.to_csv(OUT/'dataset_equal_metrics.csv',index=False);subject.to_csv(OUT/'subject_metrics.csv',index=False)
selected=mean[mean.method.str.endswith('_sel')]
print(selected[['dataset','stride','offset','method','savings_fraction','reference_disagreement','deployable_full_models_per_window','all_checks_passed']].to_string(index=False),flush=True)
for lock in ('prior.json','inputs.json'):
    assert all(sha(ROOT/n)==h for n,h in read(OUT/lock).items())
write(OUT/'summary_complete.json',dict(case_grids=396,selected_streams=3900,metrics_rows=len(rows),
    scenarios=9,summary_rows=len(mean),selected_packets_serialized=packet_count,
    exact_rational_recovery_weights=True,prior_paper_files_preserved=True,new_independent_subjects=0,drill_opened=False))
