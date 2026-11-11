"""Fig. 2: fixed-trace control budget c_max(u) for a 10% saving, original primary grids.
Run from paper_v14 with figure_envelope_points.csv beside it: python cost_envelope.py
c_max(u) = (A + uK)/rho is exactly linear in u, so lines beyond the computed points (u > 64) are the same algebra."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

names = {f.name for f in fm.fontManager.ttflist}
FONT = next((n for n in ('Times New Roman', 'Liberation Serif', 'TeX Gyre Termes') if n in names), 'serif')
plt.rcParams.update({'font.family': FONT, 'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'pdf.fonttype': 42, 'axes.edgecolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333'})
import sys, os
src = sys.argv[1] if len(sys.argv) > 1 else 'figure_envelope_points.csv'
out = sys.argv[2] if len(sys.argv) > 2 else 'cost_envelope'
d = pd.read_csv(src)
DS = [('pamap2', 'PAMAP2', '#0072B2', 'o'), ('mhealth', 'MHEALTH', '#009E73', 's'), ('opportunity', 'OPPORTUNITY', '#CC79A7', 'D')]
UMAX, YLO, YHI = 96, -32, 38

fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
for ax, (meth, title) in zip(axes, [('direct_t0.01', '(a) Direct prediction, $\\epsilon=0.01$'),
                                    ('survival_t0.01', '(b) Hazard scheduler, $\\epsilon=0.01$')]):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.axhspan(YLO, 0, color='#C0392B', alpha=0.07, lw=0, zorder=0)
    ax.axhline(0, color='#888888', lw=0.6, zorder=1)
    ax.axhline(30, color='#333333', lw=0.8, ls=(0, (1.2, 1.6)), zorder=1)
    for ds, lab, c, mk in DS:
        g = d[(d.dataset == ds) & (d.method == meth)].sort_values('extra_payload')
        k, a = np.polyfit(g.extra_payload, g.notice_pair_cap, 1)
        uu = np.array([0, 64]); ax.plot(uu, a + k * uu, color=c, lw=1.5, zorder=3, solid_capstyle='round')
        uu = np.array([64, UMAX]); ax.plot(uu, a + k * uu, color=c, lw=1.0, ls=(0, (3, 2)), zorder=3)
        ax.scatter(g.extra_payload, g.notice_pair_cap, s=15, marker=mk, color=c, edgecolor='white', lw=0.5, zorder=4, label=lab)
        if k > 0 and a + k * UMAX >= 30:
            us = (30 - a) / k
            ax.scatter([us], [30], s=34, marker='*', color=c, edgecolor='white', lw=0.4, zorder=5)
            ax.annotate(f'$u^*$={us:.1f} B', (us, 30), xytext=(-2, 6 if ds == 'pamap2' else -10), textcoords='offset points',
                        ha='right' if ds == 'pamap2' else 'left', fontsize=6.5, color=c)
        if meth == 'direct_t0.01' and ds == 'opportunity':
            ax.annotate('needs >10 kB', (UMAX, a + k * UMAX), xytext=(0, -9), textcoords='offset points', ha='right', fontsize=6.5, color=c)
    ax.set_xlim(-2, UMAX + 2); ax.set_xticks([0, 32, 64, 96]); ax.set_ylim(YLO, YHI); ax.set_yticks([-30, -15, 0, 15, 30])
    ax.set_xlabel('Bytes added per score packet, $u$', labelpad=1.5)
    ax.set_title(title, fontsize=7.5, pad=3)
    left = meth == 'direct_t0.01'
    ax.text(2 if left else UMAX, 28.8 if left else 31.2, 'implemented 30 B', ha='left' if left else 'right', va='top' if left else 'bottom', fontsize=6.3, color='#333333')
axes[0].text(2, -29, 'unaffordable even with free control', fontsize=6.3, color='#A93226', va='bottom')
axes[1].annotate('budget falls as\npayload grows', (80, -22), xytext=(58, 8), fontsize=6.5, color='#444444', ha='center',
                 arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#444444', mutation_scale=6))
axes[0].set_ylabel('Affordable notice-pair charge (B)', labelpad=2)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.01), handletextpad=0.2, columnspacing=1.2, fontsize=7)
fig.subplots_adjust(left=0.13, right=0.985, bottom=0.17, top=0.82, wspace=0.08)
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=300)
