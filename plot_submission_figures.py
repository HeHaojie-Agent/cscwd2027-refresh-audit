"""Draw the CSCWD manuscript's single-view figures from the released source data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
DATA = HERE / 'figure_source'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7.2,
    'axes.labelsize':7.5,'legend.fontsize':6.2,'axes.linewidth':.65,'xtick.labelsize':7,
    'ytick.labelsize':7,'xtick.major.size':2.5,'ytick.major.size':2.5,
    'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'svg.fonttype':'none',
    'mathtext.fontset':'stix','text.color':'#26323B','axes.labelcolor':'#26323B',
    'axes.edgecolor':'#68727A','xtick.color':'#26323B','ytick.color':'#26323B'})
INK='#26323B'; MUTED='#576674'; GRID='#DEEAEE'; BLUE='#325D89'; TEAL='#06776F'; GOLD='#A06420'; PLUM='#805B98'
W=3.48


def emit(fig, name):
    for suffix in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)


def box(ax, x, y, w, h, title, lines, face, edge):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018,rounding_size=.12',
        facecolor=face,edgecolor=edge,linewidth=.7))
    ax.text(x+.22,y+h-.27,title,weight='bold',fontsize=7.7,va='top',color=INK)
    ax.text(x+.22,y+h-.67,lines,fontsize=7.0,va='top',linespacing=1.32,color=INK)


def arrow(ax, a, b, color, style='-', curve=0, shrink=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,linewidth=.9,
        color=color,linestyle=style,connectionstyle=f'arc3,rad={curve}',shrinkA=shrink,shrinkB=shrink))


def mechanism():
    fig=plt.figure(figsize=(W,1.65)); ax=fig.add_axes([.01,.02,.98,.96]); ax.axis('off'); ax.set(xlim=(0,10),ylim=(0,5))
    box(ax,.16,2.57,3.1,1.78,'Auxiliary 1','Local change test\nInfer on push or pull','#F1F5F9',BLUE)
    box(ax,.16,.49,3.1,1.78,'Auxiliary 2','Local change test\nInfer on push or pull','#F1F5F9',BLUE)
    box(ax,5.76,.74,4.02,3.16,'Primary','Cache + local scores\nFuse and check margin\nPull oldest stale peer\nRecheck after its reply','#EDF7F4',TEAL)
    arrow(ax,(3.26,3.55),(5.76,3.48),BLUE)
    arrow(ax,(3.26,1.75),(5.76,1.78),BLUE)
    arrow(ax,(5.76,2.91),(3.26,2.90),TEAL,'--')
    arrow(ax,(5.76,1.15),(3.26,1.13),TEAL,'--')
    ax.text(4.5,3.64,'score packet',ha='center',fontsize=6.7,color=BLUE)
    ax.text(4.48,2.54,'pull notice',ha='center',fontsize=6.7,color=TEAL)
    ax.text(5.86,4.63,'Current window',weight='bold',fontsize=7.5)
    arrow(ax,(8.91,4.59),(8.91,3.9),TEAL)
    arrow(ax,(8.91,.74),(8.91,.42),TEAL)
    ax.text(8.91,.10,'Activity output',ha='center',fontsize=7.0)
    emit(fig,'refresh_mechanism')


def example():
    r=json.loads((DATA/'refresh_trace_v36.json').read_text())
    d=r['threshold']; a,b=r['age']; pre,post=r['margin_before'],r['margin_after']
    assert pre<=d<post and a>b and r['output']==r['complete_winner']
    for key,margin in [('pre_scores',pre),('post_scores',post)]:
        assert np.diff(np.sort(np.asarray(r[key]).sum(0))[-2:])[0]==margin
    fig=plt.figure(figsize=(W,1.59)); ax=fig.add_axes([.01,.01,.98,.98]); ax.axis('off'); ax.set(xlim=(0,10),ylim=(0,5))
    # A single timeline of one recorded window, ordered from top to bottom.
    ax.plot([.46,.46],[.79,4.44],color='#BCD4D0',lw=1.4,zorder=1)
    events=[(4.43,'Both auxiliary scores are stale',f'Cache ages: {a} and {b} windows. Neither peer pushes.'),
        (2.92,f'Pull auxiliary 1  |  margin {pre} $\\leq$ {d}',f'Oldest cache first. Send one {r["compact_notice_bytes"]}-byte notice.'),
        (1.41,f'Stop after the reply  |  margin {post} $>$ {d}','Keep auxiliary 2 cached. Output agrees with full fusion.')]
    for i,(y,title,sub) in enumerate(events,1):
        ax.scatter([.46],[y],s=66,c=TEAL,zorder=2,edgecolor='white',lw=.5)
        ax.text(.46,y,str(i),color='white',ha='center',va='center',fontsize=6.3,zorder=3)
        ax.text(.98,y,title,weight='bold',fontsize=8,va='center')
        ax.text(.98,y-.58,sub,fontsize=7.3,va='center')
    emit(fig,'refresh_example')


def axes(width=W,height=1.70, rect=(.158,.23,.808,.65)):
    fig=plt.figure(figsize=(width,height)); ax=fig.add_axes(rect)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=GRID,lw=.55); ax.set_axisbelow(True)
    return fig,ax


def envelope():
    f=pd.read_csv(DATA/'figure_envelope_points.csv')
    f=f[f.method.eq('survival_t0.01')]; assert len(f)==12 and (f.notice_pair_cap<0).all()
    fig,ax=axes(height=1.73,rect=(.164,.24,.799,.66))
    ax.axhspan(-32,0,color='#F7EEE6',lw=0,zorder=0)
    ax.axhline(0,color=MUTED,lw=.8)
    ax.axhline(30,color=MUTED,lw=.7,ls='--')
    ax.text(1.4,30.9,'30 B implemented',ha='left',va='bottom',fontsize=6.6,color=MUTED)
    for name,color,mark in [('pamap2',BLUE,'o'),('mhealth',TEAL,'s'),('opportunity',GOLD,'D')]:
        d=f[f.dataset.eq(name)].sort_values('extra_payload'); assert len(d)==4
        ax.plot(d.extra_payload,d.notice_pair_cap,marker=mark,ms=3.6,lw=1.15,color=color,
                label={'pamap2':'PAMAP2','mhealth':'MHEALTH','opportunity':'OPPORTUNITY'}[name])
    ax.set(xlim=(-1.5,66),ylim=(-32,39),xticks=[0,16,32,48,64],yticks=[-30,-15,0,15,30],
           xlabel='Bytes added per score packet',ylabel='Affordable notice pair (B)')
    ax.legend(loc='lower left',bbox_to_anchor=(.014,.50),borderaxespad=0,ncol=2,columnspacing=.8,
              handlelength=1.6,frameon=False)
    emit(fig,'hazard_cost_envelope')


def frontier(dataset):
    f=pd.read_csv(DATA/'frontier_points.csv'); f=f[(f.stride==1)&(f.offset==0)&f.dataset.eq(dataset)].copy()
    aux=pd.concat([pd.read_csv(DATA/f'aux_trigger_{v}_points.csv') for v in ['v30','v30b']])
    aux=aux[(aux.stride==1)&(aux.offset==0)&aux.dataset.eq(dataset)&~aux.method.eq('auxtrig_sel')].copy()
    aux['kappa']=aux.method.str.extract(r'k([0-9.]+)$')[0].astype(float); aux=aux.sort_values('kappa')
    fig,ax=axes(height=1.93,rect=(.144,.207,.83,.568))
    ax.axvspan(-.25,1,color='#EAF5F2',lw=0,zorder=0)
    ax.axhline(10,lw=.8,color=MUTED,ls=':')
    ax.text(12.7,11.8,'10% target',ha='right',va='bottom',fontsize=6.1,color=MUTED)
    xy=lambda g: (100*g.reference_disagreement,100*g.savings_fraction)
    lines=[]
    def marker(g,name,mark,color,layer=3,empty=False,size=13):
        assert len(g)>0, name
        h=ax.scatter(*xy(g),s=size,marker=mark,facecolor='white' if empty else color,
                edgecolor=color if empty else 'white',linewidth=.62,zorder=layer,label=name)
        lines.append(h)
    for fn,method,marker_char,color,label in [('twoside_trigger_v32_points','twoside_sel','D',GOLD,'Paired pull'),
            ('v36_dataset_equal_metrics','recovery_compact_sel','P',TEAL,'Compact sequential')]:
        d=pd.read_csv(DATA/(fn+'.csv')); d=d[(d.stride==1)&(d.offset==0)&d.dataset.eq(dataset)&d.method.eq(method)]
        marker(d,label,marker_char,color,10,size=38 if marker_char=='P' else 28)
    marker(f[f.source.eq('V28 offline')],'Offline schedule','*','#8F3B30',9,size=52)
    ax.plot(*xy(aux),color='#B48340',lw=1,zorder=4); marker(aux,'Auxiliary push','^','#B48340',6,size=20)
    fc=f[f.source.isin(['V20','V23'])&f.method.str.match(r'(survival|direct|isotonic|platt)_t')]
    marker(fc,'Forecast thresholds','o','#727984',3,empty=True,size=14)
    g=f[f.source.isin(['V20','V23'])&f.method.eq('periodic_h2')]
    h=ax.scatter(*xy(g),s=15,marker='x',color=INK,lw=.75,zorder=5,label='Period 2'); lines.append(h)
    for name,mark,col,lab in [('imit_linear','o',BLUE,'Linear imitation'),('imit_boost','s','#557D50','Boosted imitation'),('imit_boost_diag','D',PLUM,'Fee-free imitation')]:
        g=f[f.source.eq('V29 charged')&f.method.str.startswith(name+'_g')].copy()
        g['gamma']=g.method.str.extract(r'_g([0-9.]+)$')[0].astype(float); g=g.sort_values('gamma')
        ax.plot(*xy(g),color=col,lw=.88,zorder=4); marker(g,lab,mark,col,6,size=14)
    order=[1,0,3,2,4,5,6,7,8]
    fig.legend(handles=[lines[i] for i in order],loc='upper center',bbox_to_anchor=(.51,.993),frameon=False,
        ncol=3,columnspacing=.85,handletextpad=.18,handlelength=1.25,fontsize=6.05,labelspacing=.36)
    ax.set(xlim=(-.45,13),ylim=(-61,49),xticks=[0,2,4,6,8,10,12],yticks=[-50,-25,0,25],
           xlabel='Disagreement with complete push (%)',ylabel='Byte saving (%)')
    emit(fig,'frontier_'+dataset)


def subject_example():
    f=pd.read_csv(DATA/'design_subjects.csv'); f=f[(f.stride==1)&(f.offset==0)]
    fig,ax=axes(height=1.92,rect=(.148,.222,.82,.635))
    marker=['o','s','^','D']
    selectors=[('fixed12','Fixed',MUTED),('call_tuned','Calls',BLUE),('audit','Cost',TEAL),('fixed_matched','Matched',GOLD)]
    for i,(key,label,c) in enumerate(selectors):
        for _,r in f[f.selector.eq(key)].iterrows():
            ax.scatter(r.deployable_full_models_per_window,r.saving_percent,s=36-6*i,marker=marker[int(r.subject)-1],
                      facecolor='none',edgecolor=c,lw=.90,zorder=3+i)
    ax.axhline(0,color='#94A2AA',lw=.7); ax.axvline(2.7,color='#94A2AA',lw=.7,ls='--')
    ax.set(xlabel='Model calls per window',ylabel='Byte saving (%)')
    hs=[Line2D([],[],marker='o',ms=4,mfc='none',mec=c,linestyle='none',label=label) for _,label,c in selectors]
    fig.legend(handles=hs,loc='upper center',bbox_to_anchor=(.54,.997),frameon=False,ncol=4,handletextpad=.25,columnspacing=1)
    emit(fig,'design_subject_tradeoff')

if __name__=='__main__':
    mechanism(); example(); envelope()
    for dataset in ('pamap2','mhealth','opportunity'): frontier(dataset)
    subject_example()
