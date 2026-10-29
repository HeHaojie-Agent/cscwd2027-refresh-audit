"""Post hoc matched offline opportunity diagnostic; never an online policy."""
from pathlib import Path
import hashlib,json,sys,itertools
import numpy as np,pandas as pd
from numba import njit
from datetime import datetime,timezone
from validity_survival import frame
from run_nested_survival import subject_frame
from run_matched_calibration import fixed_baseline_name
from summarize_matched_calibration import equal_summary
from protocol_replay_v26 import RiceWidthCodec
from causal_score_codec import HEADER

C=Path(__file__).resolve().parent;O=C/'results/matched_oracle_v28'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

@njit(cache=True)
def edge(q,a,b,width):
    total=0
    for node in (1,2):
        bits=0
        for c in range(width):
            d=np.int64(q[b,node,c])-np.int64(q[a,node,c]);z=2*d if d>=0 else -2*d-1
            bits+=(z>>2)+3
        total+=14+min(width,(bits+7)//8)
    return total

@njit(cache=True)
def optimal(q,reset,width,charge):
    n=len(q);sent=np.zeros((n,3),np.bool_);sent[:,0]=True
    grants=np.zeros(n,np.int16);total=0;starts=0
    hs=np.array([1,2,4,8,16])
    while starts<n:
        end=starts+1
        while end<n and not reset[end]:end+=1
        cost=np.zeros(end-starts+1,np.int64);choice=np.ones(end-starts,np.int16)
        for a in range(end-1,starts-1,-1):
            first=end
            for t in range(a+1,min(a+16,end)):
                cw=0;fw=0;cb=-1;fb=-1
                for c in range(12):
                    cv=int(q[t,0,c])+int(q[a,1,c])+int(q[a,2,c])
                    fv=int(q[t,0,c])+int(q[t,1,c])+int(q[t,2,c])
                    if cv>cb:cb=cv;cw=c
                    if fv>fb:fb=fv;fw=c
                if cw!=fw:first=t;break
            best=np.int64(1<<60)
            for h in hs:
                b=min(a+h,end)
                if first<b:continue
                value=charge if a+1<end else 0
                if b<end:value+=edge(q,a,b,width)+cost[b-starts]
                if value<best or (value==best and h>choice[a-starts]):best=value;choice[a-starts]=h
            cost[a-starts]=best
        total+=cost[0];sent[starts,1:]=True;a=starts
        while a<end:
            h=choice[a-starts]
            if a+1<end:grants[a+1]=h
            a+=h
            if a<end:sent[a,1:]=True
        starts=end
    return sent,grants,total

def score_cost(q,sent,reset,width,k):
    result=np.zeros(len(q),np.int64)
    for node in range(3):
        at=np.flatnonzero(sent[:,node]);v=q[at,node,:width].astype(np.int64)
        d=np.diff(v,axis=0,prepend=v[:1]);z=2*np.abs(d)-(d<0)
        length=14+np.minimum(width,(((z>>k)+k+1).sum(1)+7)//8)
        length[reset[at]]=14+width;result[at]+=length
    return result

def verify(f,sent,grants,width,charge,optimum):
    n=len(sent);q=f['q'];pos=np.arange(n)
    assert sent[:,0].all() and sent[f['reset']].all()
    assert np.array_equal(sent[:,1],sent[:,2])
    anchor=np.maximum.accumulate(np.where(sent[:,1],pos,-1))
    pred=np.argmax(q[:,0].astype(int)+q[anchor,1].astype(int)+q[anchor,2].astype(int),axis=1)
    np.testing.assert_array_equal(pred,f['reference'])
    segment=np.cumsum(f['reset']);dec=np.flatnonzero(grants)
    assert np.isin(grants[dec],[1,2,4,8,16]).all()
    assert (anchor[dec-1]==dec-1).all()
    expected=[]
    for a in np.flatnonzero(sent[:,1]):
        if a+1<n and not f['reset'][a+1]:
            assert grants[a+1]>0
            due=a+int(grants[a+1])
            if due<n and segment[due]==segment[a]:expected.append(due)
    assert np.array_equal(np.sort(np.r_[np.flatnonzero(f['reset']),expected]),np.flatnonzero(sent[:,1]))
    base=np.zeros_like(sent);base[:,0]=True;base[f['reset']]=True
    score=score_cost(q,sent,f['reset'],width,2)
    assert int(score.sum()+charge*len(dec)-score_cost(q,base,f['reset'],width,2).sum())==optimum
    # Actual encode/decode at deterministic quantiles and every reset.
    inverses=0
    for node in range(3):
        at=np.flatnonzero(sent[:,node]);selected=set(np.linspace(0,len(at)-1,min(12,len(at)),dtype=int))
        selected.update(np.flatnonzero(f['reset'][at]))
        for j in sorted(selected):
            t=int(at[j]);enc=RiceWidthCodec('rice2',width);decoder=RiceWidthCodec('rice2',width)
            if not f['reset'][t]:
                enc.previous=q[at[j-1],node,:width].copy();decoder.previous=enc.previous.copy()
            packet=enc.encode(q[t,node,:width],node,t,t)
            assert np.array_equal(decoder.decode(packet,node,t,t),q[t,node,:width])
            inverses+=1
    return score,inverses

def cases():
    V=C/'results/matched_calibration_v20'
    for case in read(V/'plan.json')['cases']:
        for s,o in ((1,0),(2,0),(2,1)):
            yield dict(**case,stride=s,offset=o),V/'score_views'/case['id']/'evaluation.npz'
    for a in range(1,5):
        for s,o in ((1,0),(2,0),(2,1)):
            p=C/'results/opportunity_v23/controllers'/f'A{a}'/f'stride{s}_offset{o}'
            case=read(p/'lineage.json')['case']
            yield {**case,'id':f'opportunity/A{a}','stride':s,'offset':o},p/'A_frame.npz'

def short_tests():
    checked=0
    for seed,width,charge in itertools.product(range(8),(4,12),(0,10,30)):
        q=np.random.default_rng(seed).integers(0,256,(9,3,12),dtype=np.uint8)
        if width==4:q[:,:,4:]=0
        reset=np.zeros(9,bool);reset[0]=True;reference=np.argmax(q.astype(int).sum(1),axis=1)
        def exhaustive(a):
            best=10**12
            for h in (1,2,4,8,16):
                b=min(a+h,len(q))
                if any(np.argmax(q[t,0].astype(int)+q[a,1].astype(int)+q[a,2].astype(int))!=reference[t] for t in range(a+1,b)):continue
                cost=charge if a+1<len(q) else 0
                if b<len(q):
                    encs=[RiceWidthCodec('rice2',width) for _ in (1,2)]
                    for node,enc in zip((1,2),encs):
                        enc.previous=q[a,node,:width].copy();cost+=len(enc.encode(q[b,node,:width],node,b,b))
                    cost+=exhaustive(b)
                best=min(best,cost)
            return best
        sent,grants,total=optimal(q,reset,width,charge);assert total==exhaustive(0)
        verify(dict(q=q,reset=reset,reference=reference),sent,grants,width,charge,total);checked+=1
    return checked

def run():
    O.mkdir(exist_ok=True);allcases=list(cases())
    inputs={str(p.relative_to(C)):sha(p) for _,p in allcases}
    prior={str(p.relative_to(C)):sha(p) for folder in ('paper_v9','paper_v10','paper_v11') for p in (C/folder).rglob('*') if p.is_file()}
    protocol=dict(client_date='2026-10-27',created_utc=datetime.now(timezone.utc).isoformat(),post_hoc=True,
        all_old_outcomes_known=True,new_independent_subjects=0,future_scores_used=True,
        original_case_grids=384,external_case_grids=12,controls=[0,30],external_widths=[12,4],
        original_width=12,intervals=[1,2,4,8,16],quality='zero reference disagreement',
        actual_oracle_model_calls=3,oracle_model_package_bytes=0,tail_known=True,
        claimed_scope='Exact minimum bytes in the specified zero-conflict deadline family, with clairvoyant free planning; not a bound on the 1-percent disagreement-feasible policy family',
        source_sha256=sha(__file__))
    if (O/'frozen.json').exists():
        old=read(O/'frozen.json')
        if old['source_sha256']!=protocol['source_sha256']:
            amendment=O/'pre_execution_source_amendment.json'
            if not amendment.exists():
                assert not list(O.glob('traces/**/*.npz'))
                write(amendment,dict(original_source=old['source_sha256'],corrected_source=protocol['source_sha256'],
                    reason='Tiny exhaustive tests caught unsigned subtraction in Numba edge costs before full-stream execution',
                    full_streams_scored_before_correction=0))
            accepted=read(amendment)['corrected_source'];second=O/'external_baseline_amendment.json'
            assert protocol['source_sha256']==(read(second)['corrected_source'] if second.exists() else accepted)
        assert read(O/'inputs.json')==inputs
    else:
        write(O/'inputs.json',inputs);write(O/'prior.json',prior);write(O/'frozen.json',protocol)
    tests=short_tests();rows=[];inverses=0
    for ci,(case,path) in enumerate(allcases):
        dataset=case['dataset'];s=case['stride'];o=case['offset'];external=dataset=='opportunity'
        if external:
            with np.load(path) as z:f={k:z[k] for k in z.files}
        else:f=frame(path,dataset,s,o)
        for subject in case['evaluation_subjects']:
            sf=subject_frame(f,subject);q=sf['q'];n=len(q)
            for width in ((12,4) if external else (12,)):
                assert width==12 or not q[:,:,4:].any()
                # Original protocol: zero extension. Narrow packing:54B schema/session.
                extra=54 if width==4 else 0
                fixed=45*sf['status'].astype(int)+(134+extra)*sf['session_start'].astype(int)
                baseline=score_cost(q,np.ones((n,3),bool),sf['reset'],width,3 if external or fixed_baseline_name(dataset,s,o)=='rice3' else 2)
                baseline+=45*sf['status'].astype(int)+(116+extra)*sf['session_start'].astype(int)
                for charge in (0,30):
                    sent,grants,optimum=optimal(q,sf['reset'],width,charge)
                    score,inv=verify(sf,sent,grants,width,charge,optimum);inverses+=inv
                    total=score+charge*(grants>0)+fixed
                    row={k:case.get(k,0) for k in ('dataset','outer_fold','group','direction','stride','offset')}
                    row.update(subject=int(subject),width=width,charge=charge,windows=n,bytes_per_window=float(total.mean()),
                        baseline_bytes_per_window=float(baseline.mean()),saving=1-float(total.sum()/baseline.sum()),
                        actual_oracle_calls=3.,hypothetical_known_schedule_calls=float(sent.sum()/n),reference_disagreement=0.,
                        grant_decisions=int((grants>0).sum()),edge_optimum=int(optimum),score_bytes=int(score.sum()),
                        fixed_bytes=int(fixed.sum()),total_bytes=int(total.sum()),sampled_packet_inverses=inv)
                    rows.append(row)
                    dest=O/'traces'/case['id']/f'stride{s}_offset{o}';dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/f'subject{subject}_w{width}_c{charge}.npz',send=sent,grants=grants,total_bytes=total,
                        continuous_global_indices=sf['continuous_global_indices'])
        if ci%24==0:print(f'V28 {ci+1}/{len(allcases)} case-grids; {len(rows)} traces verified',flush=True)
    table=pd.DataFrame(rows);keys=['dataset','stride','offset','width','charge']
    _,subject,mean=equal_summary(table,keys)
    mean['saving']=1-mean.bytes_per_window/mean.baseline_bytes_per_window
    positive=subject.assign(positive=subject.bytes_per_window<subject.baseline_bytes_per_window).groupby(keys).positive.mean().reset_index(name='positive_subject_fraction')
    mean=mean.merge(positive,on=keys)
    for name,df in [('view_metrics',table),('subject_metrics',subject),('dataset_equal_metrics',mean)]:df.to_csv(O/(name+'.csv'),index=False)
    assert all(sha(C/p)==h for p,h in read(O/'inputs.json').items())
    assert all(sha(C/p)==h for p,h in read(O/'prior.json').items())
    write(O/'verification.json',dict(all_passed=True,exhaustive_tiny_cases=tests,full_stream_traces=len(rows),
        case_grids=len(allcases),sampled_packet_inverses=inverses,zero_disagreement_all=True,old_papers_preserved=True,
        future_information_used=True,actual_oracle_calls=3.,new_independent_subjects=0))
    print(mean[['dataset','stride','offset','width','charge','saving','hypothetical_known_schedule_calls']].to_string(index=False),flush=True)

if __name__=='__main__':run()
