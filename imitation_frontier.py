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

def envelope(out='cost_envelope'):
    d=pd.read_csv('figure_envelope_points.csv')
    fig,axes=plt.subplots(1,3,figsize=(W,1.85),sharey=True)
    for ax,(ds,title) in zip(axes,TITLES):
        style(ax)
        ax.axhline(30,color=MUTED,lw=0.9,ls=(0,(1,2)),zorder=2)
        ax.axhline(0,color=MUTED,lw=0.9,ls=(0,(4,2)),zorder=2)
        for m,c,mk in (('direct_t0.01',DIR,'o'),('survival_t0.01',HAZ,'^')):
            g=d[(d.dataset==ds)&(d.method==m)].sort_values('extra_payload')
            ax.plot(g.extra_payload,g.notice_pair_cap,color=c,lw=1.4,marker=mk,ms=4.2,mec='white',mew=0.6,zorder=4)
        ax.set_xlim(-3,67); ax.set_xticks([0,16,32,48,64]); ax.set_ylim(-30,36); ax.set_yticks([-30,-20,-10,0,10,20,30])
        ax.set_title(title,pad=3); ax.set_xlabel('Bytes added per score packet, $u$')
    axes[0].set_ylabel('Affordable notice-pair charge (B)')
    handles=[Line2D([],[],color=HAZ,lw=1.4,marker='^',ms=4.2,mec='white',mew=0.6,label='Hazard 0.01'),
             Line2D([],[],color=DIR,lw=1.4,marker='o',ms=4.2,mec='white',mew=0.6,label='Direct 0.01'),
             Line2D([],[],color=MUTED,lw=0.9,ls=(0,(4,2)),label='Free control (0 B)'),
             Line2D([],[],color=MUTED,lw=0.9,ls=(0,(1,2)),label='Implemented charge (30 B)')]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(0.5,1.0),handlelength=2.2,columnspacing=1.6)
    fig.subplots_adjust(left=0.075,right=0.99,bottom=0.21,top=0.8,wspace=0.12)
    fig.savefig(out+'.pdf'); fig.savefig(out+'.png',dpi=300); plt.close(fig)

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
             Line2D([],[],ls='',marker='x',ms=4.5,color=INK,label='Periodic, interval 2')]
    order=[0,4,5,1,2,3]
    fig.legend(handles=[handles[i] for i in order],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.0),handlelength=2.0,columnspacing=1.6,handletextpad=0.5)
    fig.subplots_adjust(left=0.075,right=0.99,bottom=0.18,top=0.72,wspace=0.12)
    fig.savefig(out+'.pdf'); fig.savefig(out+'.png',dpi=300); plt.close(fig)

if __name__=='__main__':
    envelope(); frontier()
