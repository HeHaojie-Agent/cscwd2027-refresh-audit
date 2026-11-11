"""Fig. 2 (fixed-trace envelope) and Fig. 3 (saving-disagreement frontier) in one shared style.
Run in a folder containing figure_envelope_points.csv and frontier_points.csv."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

def _font():
    names={f.name for f in fm.fontManager.ttflist}
    for n in ('Times New Roman','Liberation Serif','TeX Gyre Termes'):
        if n in names: return n
    return 'serif'
FONT=_font()
plt.rcParams.update({'font.family':FONT,'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8.5,
  'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':0.6,
  'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':3,'ytick.major.size':3,
  'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42,'ps.fonttype':42,
  'axes.edgecolor':'#333333','xtick.color':'#333333','ytick.color':'#333333','text.color':'#222222','axes.labelcolor':'#222222'})
INK='#222222'; MUTED='#6E6E6E'; GRID='#E3E3E3'
HAZ='#D55E00'; DIR='#0072B2'
TITLES=[('pamap2','(a) PAMAP2'),('mhealth','(b) MHEALTH'),('opportunity','(c) OPPORTUNITY')]
W=7.16

def style(ax):
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.grid(axis='y',color=GRID,lw=0.5,zorder=0); ax.set_axisbelow(True)


AUXC='#E69F00'
def _aux():
    a=pd.concat([pd.read_csv('aux_trigger_v30_points.csv'),pd.read_csv('aux_trigger_v30b_points.csv')],ignore_index=True)
    a=a[(a.stride==1)&(a.offset==0)&(a.method!='auxtrig_sel')].copy()
    a['kappa']=a.method.str.extract(r'k([0-9.]+)$')[0].astype(float)
    a=a[a.kappa<=6]
    a['x']=100*a.reference_disagreement; a['y']=100*a.savings_fraction
    return a
AUX=_aux()
def frontier(out='imitation_frontier'):
    f=pd.read_csv('frontier_points.csv'); f=f[(f.stride==1)&(f.offset==0)].copy()
    f['x']=100*f.reference_disagreement; f['y']=100*f.savings_fraction
    fam={'imit_boost_diag':('#CC79A7','D','Imitation, diagnostic (fee-free)'),
         'imit_boost':('#009E73','s','Imitation, boosted'),'imit_linear':(DIR,'o','Imitation, linear')}
    fig,axes=plt.subplots(1,3,figsize=(W,2.2),sharey=True)
    for ax,(ds,title) in zip(axes,TITLES):
        style(ax); d=f[f.dataset==ds]
        ax.axvspan(-0.4,1,color='#EFEFEF',lw=0,zorder=0)
        ax.axhline(10,color=MUTED,lw=0.9,ls=(0,(1,2)),zorder=2); ax.axhline(0,color='#BDBDBD',lw=0.6,zorder=1)
        fc=d[d.source.isin(['V20','V23'])&d.method.str.match(r'(survival|direct|isotonic|platt)_t')]
        ax.scatter(fc.x,fc.y,s=15,facecolor='white',edgecolor=MUTED,lw=0.8,zorder=3)
        pe=d[d.source.isin(['V20','V23'])&d.method.eq('periodic_h2')]
        ax.scatter(pe.x,pe.y,s=20,marker='x',color=INK,lw=0.9,zorder=3)
        for name,(c,m,_) in fam.items():
            g=d[(d.source=='V29 charged')&d.method.str.startswith(name+'_g')].copy()
            g['gamma']=g.method.str.extract(r'_g([0-9.]+)$')[0].astype(float); g=g.sort_values('gamma')
            ax.plot(g.x,g.y,color=c,lw=1.2,zorder=4)
            ax.scatter(g.x,g.y,s=16,marker=m,facecolor=c,edgecolor='white',lw=0.5,zorder=5)
        g=AUX[AUX.dataset==ds].sort_values('kappa')
        ax.plot(g.x,g.y,color=AUXC,lw=1.7,zorder=6)
        ax.scatter(g.x,g.y,s=18,marker='^',facecolor=AUXC,edgecolor='white',lw=0.5,zorder=7)
        o=d[d.source=='V28 offline']
        ax.scatter(o.x,o.y,s=80,marker='*',color=HAZ,edgecolor='white',lw=0.5,zorder=6)
        ax.annotate(f"{o.y.iloc[0]:.1f}%",(o.x.iloc[0],o.y.iloc[0]),xytext=(6,-3),textcoords='offset points',fontsize=7.5,color=INK)
        ax.set_xlim(-0.4,13); ax.set_xticks([0,2,4,6,8,10,12]); ax.set_ylim(-60,50); ax.set_yticks([-50,-25,0,25,50])
        ax.set_title(title,pad=3); ax.set_xlabel('Reference disagreement (%)')
    axes[0].set_ylabel('Byte saving (%)')
    axes[0].text(1.3,-57,'1% limit',fontsize=7,color=MUTED); axes[0].text(12.8,11.5,'10% target',fontsize=7,color=MUTED,ha='right')
    handles=[Line2D([],[],ls='',marker='*',ms=8,color=HAZ,mec='white',label='Offline zero-conflict'),
             *[Line2D([],[],color=c,lw=1.2,marker=m,ms=4.2,mfc=c,mec='white',label=lab) for c,m,lab in fam.values()],
             Line2D([],[],ls='',marker='o',ms=4.2,mfc='white',mec=MUTED,label='Threshold forecasters'),
             Line2D([],[],ls='',marker='x',ms=4.5,color=INK,label='Periodic, interval 2'),
             Line2D([],[],color=AUXC,lw=1.7,marker='^',ms=4.5,mfc=AUXC,mec='white',label='Auxiliary self-trigger (ours)')]
    order=[6,0,4,1,2,3,5]
    fig.legend(handles=[handles[i] for i in order],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(0.5,1.0),handlelength=2.0,columnspacing=1.6,handletextpad=0.5)
    fig.subplots_adjust(left=0.075,right=0.99,bottom=0.18,top=0.72,wspace=0.12)
    fig.savefig(out+'.pdf'); fig.savefig(out+'.png',dpi=300); plt.close(fig)


if __name__=='__main__':
    frontier()
