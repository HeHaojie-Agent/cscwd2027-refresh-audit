import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# Run in this folder: python imitation_frontier.py  (Times New Roman on Windows, TeX Gyre Termes elsewhere)
import os
family='Times New Roman' if any('Times New Roman' in f.name for f in fm.fontManager.ttflist) else 'TeX Gyre Termes'
for f in ['/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf']:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({'font.family':family,'mathtext.fontset':'stix','font.size':8,'axes.linewidth':0.6,
  'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42})
f=pd.read_csv('frontier_points.csv'); f=f[(f.stride==1)&(f.offset==0)].copy()
f['x']=100*f.reference_disagreement; f['y']=100*f.savings_fraction
fig,axes=plt.subplots(1,3,figsize=(7.16,2.05),sharey=True)
fam={'imit_linear':('#0072B2','o','Imitation, linear'),'imit_boost':('#009E73','s','Imitation, boosted'),'imit_boost_diag':('#CC79A7','D','Imitation, diagnostic (fee-free)')}
handles={}
for ax,(ds,title) in zip(axes,[('pamap2','(a) PAMAP2'),('mhealth','(b) MHEALTH'),('opportunity','(c) OPPORTUNITY')]):
    d=f[f.dataset==ds]
    ax.axvspan(0,1,color='#EDEDED',lw=0,zorder=0)
    ax.axhline(10,color='#7A7A7A',lw=0.6,ls=':',zorder=1); ax.axhline(0,color='#BDBDBD',lw=0.5,zorder=1)
    fc=d[d.source.isin(['V20','V23'])&d.method.str.match(r'(survival|direct|isotonic|platt)_t')]
    h=ax.scatter(fc.x,fc.y,s=14,facecolor='white',edgecolor='#6E6E6E',lw=0.7,zorder=3,label='Learned threshold forecasters')
    handles['fc']=h
    pe=d[d.source.isin(['V20','V23'])&d.method.isin(['periodic_h2'])]
    h=ax.scatter(pe.x,pe.y,s=18,marker='x',color='#222222',lw=0.8,zorder=3,label='Periodic, interval 2'); handles['pe']=h
    for name,(c,m,lab) in fam.items():
        src='V29 charged'
        g=d[(d.source==src)&d.method.str.startswith(name+'_g')].copy()
        g['gamma']=g.method.str.extract(r'_g([0-9.]+)$').astype(float); g=g.sort_values('gamma')
        ax.plot(g.x,g.y,color=c,lw=1.0,zorder=4)
        h=ax.scatter(g.x,g.y,s=13,marker=m,facecolor=c,edgecolor='white',lw=0.5,zorder=5,label=lab); handles[name]=h
    o=d[d.source=='V28 offline']
    h=ax.scatter(o.x,o.y,s=60,marker='*',color='#D55E00',edgecolor='white',lw=0.4,zorder=6,label='Offline zero-conflict schedule'); handles['or']=h
    ax.annotate(f"{o.y.iloc[0]:.1f}%",(o.x.iloc[0],o.y.iloc[0]),xytext=(5,-3),textcoords='offset points',fontsize=7,color='#333333')
    ax.set_xlim(-0.3,13); ax.set_xticks([0,2,4,6,8,10,12]); ax.set_ylim(-60,50); ax.set_title(title,fontsize=8,pad=3)
    ax.set_xlabel('Reference disagreement (%)')
    for s in ('top','right'): ax.spines[s].set_visible(False)
axes[0].set_ylabel('Byte saving (%)')
axes[0].text(1.2,-56,'1% limit',fontsize=6.5,color='#555555'); axes[0].text(8.2,12,'10% target',fontsize=6.5,color='#555555')
order=['or','imit_boost_diag','imit_boost','imit_linear','fc','pe']
fig.legend([handles[k] for k in order],[handles[k].get_label() for k in order],loc='upper center',ncol=6,frameon=False,fontsize=6.6,
           bbox_to_anchor=(0.5,1.04),handletextpad=0.2,columnspacing=0.9)
fig.subplots_adjust(left=0.07,right=0.995,bottom=0.19,top=0.83,wspace=0.08)
fig.savefig('imitation_frontier.pdf'); fig.savefig('imitation_frontier.png',dpi=300)
