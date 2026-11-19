"""Matched adaptations of Cooper & Marculescu (IoTDI2024) and margin control.

The paper's confidence-bin expected-accuracy-gain allocation is retained.
Frozen MiniRocket scores and fixed primary replace the original CNNs and rotating
initial sensor. Training/selection uses B subjects exclusively. Native cascades
return the primary prediction when declining the ensemble. Cache adaptations
reuse cached fusion and enforce the common maximum age. Actual paid calls,
score packets, request pairs and serialized policy tables are all charged.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,struct,time,subprocess
import numpy as np
import pandas as pd
from numba import njit
from run_oracle_imitation_v29 import original_cases,external_cases,load,events_for,read,write,sha
from run_nested_survival import subject_frame
from run_matched_calibration import fixed_baseline_name
from continuous_screen_accounting import macro_f1
from run_validity_diagnostic import onset_metrics
from audit_guided_design_v27 import score_lengths
from protocol_replay_v26 import RiceWidthCodec
from causal_score_codec import HEADER

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/published_baselines_v33'
AUTHOR=ROOT/'research/baseline_v33/BeyondThresholds'
FAMILIES=('bt_supervised_native','bt_unlabeled_native','bt_supervised_cache','bt_unlabeled_cache')
METHODS=FAMILIES+('margin',)
DELTAS=(4,8,16,24,32,48)
BINS=10

def confidence(f):
    logits=f['q'][:,0,:len(f['classes'])].astype(np.float64)*(8/255)
    probabilities=np.exp(logits-logits.max(1,keepdims=True))
    return probabilities.max(1)/probabilities.sum(1)

def fit_bins(b):
    conf=confidence(b);primary=np.argmax(b['q'][:,0],axis=1)
    known=b['endpoint_targets']!=-1
    # Equal-frequency confidence bins and accuracy gains, as in author code.
    values=np.sort(conf[known]);assert len(values)>=BINS
    marks=values[::(len(values)//BINS)][:BINS+1].copy()
    if len(marks)<BINS+1:marks=np.r_[marks,1.]
    marks[0]=0.;marks[-1]=1.;marks=marks.astype('<f4')
    ids=np.clip(np.searchsorted(marks[1:-1],conf,side='left'),0,BINS-1)
    sensor_correct=b['classes'][primary]==b['endpoint_targets']
    ensemble_correct=b['classes'][b['reference']]==b['endpoint_targets']
    supervised=np.zeros(BINS);unlabeled=np.zeros(BINS)
    for j in range(BINS):
        at=known&(ids==j)
        if at.any():supervised[j]=ensemble_correct[at].mean()-sensor_correct[at].mean()
        all_at=ids==j
        if all_at.any():unlabeled[j]=1-np.mean(primary[all_at]==b['reference'][all_at])
    return dict(marks=marks,supervised=supervised,unlabeled=unlabeled)

def policy_mask(gains,budget):
    mask=np.zeros(BINS,np.uint8)
    # Stable descending gain with bin-index tie order fixed in advance.
    order=np.lexsort((np.arange(BINS),-gains))
    chosen=[int(j) for j in order if gains[j]>0][:budget]
    mask[chosen]=1
    return mask

def package(marks,mask,budget,cache):
    body=np.asarray(marks,'<f4').tobytes()+np.asarray(mask,np.uint8).tobytes()+struct.pack('<BB',budget,int(cache))
    return HEADER.pack(103,0,0,0)+hashlib.sha256(body).digest()[:16]+body

@njit(cache=True)
def schedule(q,reset,conf,marks,mask,budget,cache,margin):
    n=len(q);send=np.zeros((n,3),np.bool_);send[:,0]=True
    pull=np.zeros(n,np.bool_);pred=np.empty(n,np.int16)
    anchor=0;requested=0
    for t in range(n):
        query=bool(reset[t]);policy_query=False
        if not query:
            if margin>=0:
                top=-1;second=-1
                for c in range(12):
                    value=int(q[t,0,c])+int(q[anchor,1,c])+int(q[anchor,2,c])
                    if value>top:second=top;top=value
                    elif value>second:second=value
                policy_query=top-second<=margin
            else:
                j=0
                while j<9 and conf[t]>marks[j+1]:j+=1
                # Runtime ensemble-frequency cap adapted to a fixed primary.
                policy_query=bool(mask[j]) and (requested+1)*10<=budget*(t+1)
            query=policy_query or (cache and t-anchor>=16)
        if query:
            send[t,1:]=True;anchor=t;requested+=1
            if not reset[t]:pull[t]=True
        best=-1;winner=0
        for c in range(12):
            value=int(q[t,0,c])
            if cache or query:value+=int(q[anchor,1,c])+int(q[anchor,2,c])
            if value>best:best=value;winner=c
        pred[t]=winner
    return send,pull,pred

def costs(f,send,pull,size,codec='rice2',complete=False):
    score=score_lengths(f,send,12,codec)
    fixed=45*f['status'].astype(np.int64)+(116 if complete else 134)*f['session_start'].astype(np.int64)
    return score+30*pull.astype(np.int64)+fixed+size*f['session_start'].astype(np.int64)

def candidates(model):
    for family in FAMILIES:
        cache=family.endswith('_cache');gain=model['supervised' if 'supervised' in family else 'unlabeled']
        for budget in range(11):
            mask=policy_mask(gain,budget)
            yield dict(family=family,method=family+f'_b{budget}',budget=budget,cache=cache,margin=-1,
                       marks=model['marks'],mask=mask,size=len(package(model['marks'],mask,budget,cache)))
    for delta in DELTAS:
        yield dict(family='margin',method=f'margin_d{delta}',budget=10,cache=True,margin=delta,
                   marks=model['marks'],mask=np.ones(BINS,np.uint8),size=34)

def cheap_metrics(f,conf,c,baseline):
    send,pull,pred=schedule(f['q'],f['reset'],conf,c['marks'],c['mask'],c['budget'],c['cache'],c['margin'])
    total=costs(f,send,pull,c['size']);known=f['endpoint_targets']!=-1;classes=f['classes']
    ref=macro_f1(f['endpoint_targets'][known],classes[f['reference'][known]],classes)
    actual=macro_f1(f['endpoint_targets'][known],classes[pred[known]],classes)
    row=dict(bytes_per_window=float(total.mean()),baseline_bytes_per_window=float(baseline.mean()),
        reference_disagreement=float(np.mean(pred!=f['reference'])),known_endpoint_macro_f1=actual,
        known_endpoint_reference_f1=ref,known_endpoint_macro_f1_drop=ref-actual,
        deployable_full_models_per_window=float(send.sum()/len(send)),positive_savings=float(total.sum()<baseline.sum()),
        pull_fraction=float(pull.mean()),score_bytes=int(score_lengths(f,send,12,'rice2').sum()),
        total_bytes=int(total.sum()),package_bytes=c['size'])
    return row,send,pull,pred,total

def select_on_b(b,model,codec):
    rows=[];pool=list(candidates(model))
    for subject in np.unique(b['subjects']):
        sf=subject_frame(b,subject);conf=confidence(sf)
        baseline=costs(sf,np.ones((len(sf['q']),3),bool),np.zeros(len(sf['q']),bool),0,codec,True)
        for c in pool:
            row,*_=cheap_metrics(sf,conf,c,baseline)
            rows.append(dict(subject=int(subject),method=c['method'],family=c['family'],**row))
    view=pd.DataFrame(rows);mean=view.groupby(['family','method'],as_index=False).mean(numeric_only=True)
    mean['saving']=1-mean.bytes_per_window/mean.baseline_bytes_per_window
    mean['eligible']=(mean.saving>=.1)&(mean.positive_savings>=.75)&(mean.deployable_full_models_per_window<=2.7)&(mean.reference_disagreement<=.01)&(mean.known_endpoint_macro_f1_drop<=.005)
    chosen={}
    for family in METHODS:
        group=mean[(mean.family==family)&mean.eligible].sort_values(['bytes_per_window','method'])
        chosen[family]=None if group.empty else next(c for c in pool if c['method']==group.iloc[0].method)
    return chosen,mean

def independent_verify(f,c,send,pull,pred,total):
    conf=confidence(f);anchor=0;requested=0;packets=0
    for t in range(len(send)):
        if f['reset'][t]:query=True;want_pull=False
        else:
            if c['margin']>=0:
                fused=np.sort(f['q'][t,0].astype(int)+f['q'][anchor,1].astype(int)+f['q'][anchor,2].astype(int))
                ask=fused[-1]-fused[-2]<=c['margin']
            else:
                j=min(9,int(np.searchsorted(c['marks'][1:-1],conf[t],side='left')))
                ask=bool(c['mask'][j]) and (requested+1)*10<=c['budget']*(t+1)
            query=ask or (c['cache'] and t-anchor>=16);want_pull=query
        assert bool(send[t,1])==query and bool(send[t,2])==query and bool(pull[t])==want_pull
        if query:anchor=t;requested+=1
        fused=f['q'][t,0].astype(int)
        if query or c['cache']:fused+=f['q'][anchor,1].astype(int)+f['q'][anchor,2].astype(int)
        assert int(np.argmax(fused))==int(pred[t])
    # All packets serialized for selected rules, with actual residual codec state.
    counted=np.zeros(len(send),np.int64)
    for node in range(3):
        enc=RiceWidthCodec('rice2',12);decoder=RiceWidthCodec('rice2',12)
        for t in np.flatnonzero(send[:,node]):
            if f['reset'][t]:enc.previous=None;decoder.previous=None
            packet=enc.encode(f['q'][t,node],node,int(t),int(t))
            assert np.array_equal(decoder.decode(packet,node,int(t),int(t)),f['q'][t,node])
            counted[t]+=len(packet);packets+=1
    counted+=30*pull+45*f['status']+(134+c['size'])*f['session_start']
    assert np.array_equal(counted,total)
    return packets

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);items=list(original_cases())+list(external_cases())
    inputs={str(Path(item[role]).relative_to(ROOT)):sha(item[role]) for item in items for role in ('a','b')}
    old={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper_v14').rglob('*') if p.is_file()}
    original_commit=subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD']).decode().strip()
    spec=dict(version='published_baselines_v33',client_date='2026-11-18',created_utc=datetime.now(timezone.utc).isoformat(),
        designed_after='V32 evaluation known; post-development benchmark',cases=396,new_subjects=0,
        reference='Cooper and Marculescu, Beyond Thresholds, IoTDI2024, DOI10.1109/IoTDI61053.2024.00005',
        author_repository='https://github.com/SLDGroup/BeyondThresholds',author_commit=original_commit,
        families=list(FAMILIES),bins=10,ensemble_bin_budgets=list(range(11)),margin_deltas=list(DELTAS),
        confidence='Softmax of quantized primary Ridge scores mapped back to [-4,4], over active classes, temperature1',
        calibration='B known endpoints: equal-frequency bins and empirical ensemble-minus-primary accuracy; unlabeled gains use B primary/ensemble agreement',
        adaptation=['Frozen node models; fixed primary instead of rotating initial sensor','B calibration subjects rather than target-subject calibration',
            'Native no-query output is primary only; cache wrappers reuse fused scores and force age16',
            'Fixed-primary adaptation caps ensemble fraction from a count of all requests, with protocol resets and cache age taking priority',
            'Full codec and control/model costs included; byte targets and existing quality/recovery gates unchanged'],
        selection='B only minimum bytes meeting B byte/positive/call/disagreement/F1 conditions, otherwise matched complete-push fallback',
        no_tuning_on_new_A=True,source_sha256=sha(__file__),drill_opened=False)
    if (OUT/'frozen.json').exists():
        frozen_source=read(OUT/'frozen.json')['source_sha256']
        amendment=OUT/'summary_dispatch_amendment.json'
        accepted=read(amendment)['updated_source_sha256'] if amendment.exists() else frozen_source
        assert accepted==spec['source_sha256'];assert read(OUT/'inputs.json')==inputs
    else:
        write(OUT/'frozen.json',spec);write(OUT/'inputs.json',inputs);write(OUT/'prior.json',old)
    return items

def run_one(item):
    dest=OUT/'cases'/item['key'];done=dest/'complete.json'
    if done.exists():
        assert all(sha(dest/n)==h for n,h in read(done)['files_sha256'].items());return item['key'],'verified'
    b=load(item,'b');model=fit_bins(b)
    codec='rice3' if item['external'] else fixed_baseline_name(item['case']['dataset'],item['stride'],item['offset'])
    chosen,table=select_on_b(b,model,codec)
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/'b_selection.json',dict(model=model,choices=chosen,A_loaded=False))
    table.to_csv(dest/'b_candidates.csv',index=False)
    a=load(item,'a');meta=dict(dataset=item['case']['dataset'],outer_fold=item['case']['outer_fold'],
        group=0 if item['external'] else item['case']['group'],direction=0 if item['external'] else item['case']['direction'],stride=item['stride'],offset=item['offset'])
    rows=[];events=[];packet_count=0
    for subject in item['case']['evaluation_subjects']:
        f=subject_frame(a,subject);conf=confidence(f);n=len(f['q']);annotation=events_for(item,subject)
        baseline=costs(f,np.ones((n,3),bool),np.zeros(n,bool),0,codec,True)
        todo=[(c['method'],c,False) for c in candidates(model)]+[(family+'_sel',chosen[family],True) for family in METHODS]
        for method,c,selected in todo:
            fallback=c is None
            if fallback:
                send=np.ones((n,3),bool);pull=np.zeros(n,bool);pred=f['reference'].copy();total=baseline
                known=f['endpoint_targets']!=-1;f1=macro_f1(f['endpoint_targets'][known],f['classes'][pred[known]],f['classes'])
                row=dict(bytes_per_window=float(total.mean()),baseline_bytes_per_window=float(baseline.mean()),reference_disagreement=0.,
                    known_endpoint_macro_f1=f1,known_endpoint_reference_f1=f1,known_endpoint_macro_f1_drop=0.,deployable_full_models_per_window=3.,
                    positive_savings=0.,pull_fraction=0.,score_bytes=int(score_lengths(f,send,12,codec).sum()),total_bytes=int(total.sum()),package_bytes=0)
            else:
                row,send,pull,pred,total=cheap_metrics(f,conf,c,baseline)
                if selected:packet_count+=independent_verify(f,c,send,pull,pred,total)
            common=dict(**meta,subject=int(subject),method=method)
            rows.append(dict(**common,windows=n,fallback=fallback,**row))
            events+=onset_metrics(f,pred,np.zeros(n),annotation,common)
            if selected:np.savez_compressed(dest/f'subject{subject}_{method}.npz',send=send,pull=pull,prediction=pred,total_bytes=total,continuous_global_indices=f['continuous_global_indices'])
    write(dest/'metrics.json',rows);write(dest/'onsets.json',events)
    write(done,dict(files_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'},selected_packets_serialized=packet_count))
    return item['key'],'done'

def selftest():
    # Analytic optimum over all bin subsets for each uniform-bin utilization cap.
    import itertools
    gains=np.array([.12,-.03,.08,.02,-.2,.3,.11,0.,.19,-.01])
    for budget in range(11):
        mask=policy_mask(gains,budget)
        best=max(sum(gains[i] for i in subset) for count in range(budget+1) for subset in itertools.combinations(range(BINS),count))
        assert abs(float(mask@gains)-best)<1e-12
    item=next(external_cases());b=load(item,'b');model=fit_bins(b)
    f=subject_frame(b,int(b['subjects'][0]));conf=confidence(f);base=costs(f,np.ones((len(f['q']),3),bool),np.zeros(len(f['q']),bool),0,'rice3',True)
    for c in list(candidates(model))[::7]:
        _,send,pull,pred,total=cheap_metrics(f,conf,c,base);independent_verify(f,c,send,pull,pred,total)
    print('B-only tests: all11bin budgets match exhaustive masks; native/cache/margin actual packet replay passed',flush=True)

def summarize():
    import runpy
    runpy.run_path(str(ROOT/'summarize_published_baselines_v33.py'),run_name='__main__')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['selftest','freeze','run','summarize'])
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if args.action=='selftest':selftest()
    elif args.action=='freeze':freeze()
    elif args.action=='summarize':summarize()
    else:
        items=freeze()
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,result in enumerate(pool.map(run_one,items,chunksize=1),1):print(i,'/',len(items),*result,flush=True)
