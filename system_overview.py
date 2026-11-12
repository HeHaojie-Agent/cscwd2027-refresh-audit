"""Fig. 1: audited three-node system (a) and schematic refresh actions with measured PAMAP2 costs (b).
Run from paper_v14: python system_overview.py
Timelines are schematic; byte ratios, disagreement and call rates are measured PAMAP2 primary-grid values
(frontier_points.csv: complete_rice2, survival_t0.01, V28 offline, imit_boost_diag_g0.8, imit_linear_g0.5)."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Polygon, Ellipse
import numpy as np
from matplotlib.colors import to_rgb

names = {f.name for f in fm.fontManager.ttflist}
family = 'Times New Roman' if 'Times New Roman' in names else 'Liberation Serif'
for f in ['/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf',
          '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf',
          '/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf']:
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': family, 'mathtext.fontset': 'stix', 'font.size': 7, 'pdf.fonttype': 42})

BLUE, ORANGE, GREEN, PINK, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#C0392B', '#8A8A8A'
W, H = 3.5, 2.62
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

def txt(x, y, s, **k):
    k.setdefault('fontsize', 7); k.setdefault('va', 'center'); k.setdefault('ha', 'left'); ax.text(x, y, s, **k)


def shade(c, f):
    r = np.array(to_rgb(c)); return tuple(r + (1 - r) * f) if f > 0 else tuple(r * (1 + f))

def _lerp(a, b, t):
    a, b = np.array(a), np.array(b); return a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]

def sphere(x, y, r, c, z=5, light=(-0.4, 0.45), shadow=True):
    n = 160; yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    t = np.clip(np.hypot(xx - light[0], yy - light[1]) / 1.55, 0, 1)
    hi, base, dk = shade(c, 0.75), to_rgb(c), shade(c, -0.5)
    img = np.where((t < 0.45)[..., None], _lerp(hi, base, t / 0.45), _lerp(base, dk, np.clip((t - 0.45) / 0.55, 0, 1)))
    if shadow: ax.add_patch(Ellipse((x + 0.18 * r, y - 0.85 * r), 1.5 * r, 0.45 * r, fc='black', alpha=0.12, ec='none', zorder=z - 0.2))
    im = ax.imshow(img, extent=[x - r, x + r, y - r, y + r], origin='lower', zorder=z, interpolation='bicubic')
    clip = Circle((x, y), r, transform=ax.transData, fc='none', ec='none'); ax.add_patch(clip); im.set_clip_path(clip)

def gradient_patch(patch, c_top, c_bot, z=3, horizontal=False):
    ax.add_patch(patch); patch.set_zorder(z + 0.1); patch.set_facecolor('none')
    bb = patch.get_extents().transformed(ax.transData.inverted())
    t = np.linspace(0, 1, 128)
    img = (np.outer(np.ones(64), t) if horizontal else np.outer(t, np.ones(64)))
    img = _lerp(to_rgb(c_bot), to_rgb(c_top), img) if not horizontal else _lerp(to_rgb(c_top), to_rgb(c_bot), img)
    im = ax.imshow(img, extent=[bb.x0, bb.x1, bb.y0, bb.y1], origin='lower', zorder=z, interpolation='bicubic', aspect='auto')
    im.set_clip_path(patch)

def limb(xs, ys, lw, z=2):
    xs, ys = np.array(xs), np.array(ys)
    kw = dict(solid_capstyle='round', solid_joinstyle='round', zorder=z)
    ax.plot(xs + 0.010, ys - 0.010, color='#A9A9A9', lw=lw, **kw)
    ax.plot(xs, ys, color='#CFCFCF', lw=lw, **kw)
    ax.plot(xs - 0.010, ys + 0.006, color='#ECECEC', lw=lw * 0.35, **kw)

def cube(xc, y, c, s=0.046, d=0.016, z=3):
    x0, y0 = xc - s / 2 - d / 3, y - s / 2 - d / 3
    ax.add_patch(Polygon([(x0, y0 + s), (x0 + d, y0 + s + d), (x0 + s + d, y0 + s + d), (x0 + s, y0 + s)], fc=shade(c, 0.45), ec='none', zorder=z))
    ax.add_patch(Polygon([(x0 + s, y0), (x0 + s + d, y0 + d), (x0 + s + d, y0 + s + d), (x0 + s, y0 + s)], fc=shade(c, -0.35), ec='none', zorder=z))
    ax.add_patch(Rectangle((x0, y0), s, s, fc=c, ec='none', zorder=z))

# ---------------- (a) body-worn system ----------------
txt(0.02, 2.53, '(a) Audited three-node system (PAMAP2 placement)', fontsize=7.4, fontweight='bold')
cx, sh, hip = 0.40, 2.23, 1.88
ax.add_patch(Ellipse((cx + 0.03, 1.535), 0.36, 0.05, fc='black', alpha=0.10, ec='none', zorder=1))   # ground shadow
limb([cx - 0.08, cx - 0.15, cx - 0.17], [sh, 2.07, 1.92], 5.5)                 # far arm
limb([cx - 0.045, cx - 0.06, cx - 0.07], [hip + 0.02, 1.72, 1.56], 6)           # legs
limb([cx + 0.045, cx + 0.07, cx + 0.09], [hip + 0.02, 1.72, 1.56], 6)
ax.add_patch(FancyBboxPatch((cx - 0.075, hip - 0.012), 0.17, sh - hip + 0.05, boxstyle='round,pad=0,rounding_size=0.06', fc='#A9A9A9', ec='none', zorder=2.5))
gradient_patch(FancyBboxPatch((cx - 0.085, hip), 0.17, sh - hip + 0.05, boxstyle='round,pad=0,rounding_size=0.06', ec='none'),
               '#EDEDED', '#B9B9B9', z=2.6, horizontal=True)
limb([cx + 0.08, cx + 0.17, cx + 0.27], [sh, 2.10, 2.00], 5.5, z=2.8)          # near arm (hand forward)
sphere(cx, 2.36, 0.075, '#CFCFCF', z=2.7, shadow=False)
P = (cx + 0.27, 2.00); A1 = (cx, 2.12); A2 = (cx + 0.085, 1.62)
for (x, y), c, lab in [(P, ORANGE, 'P'), (A1, BLUE, '1'), (A2, BLUE, '2')]:
    sphere(x, y, 0.052, c, z=5)
    txt(x, y - 0.003, lab, ha='center', color='white', fontsize=5.8, fontweight='bold', zorder=6)
for a, rad in [(A1, 0.45), (A2, -0.35)]:
    ax.add_patch(FancyArrowPatch(P, a, connectionstyle=f'arc3,rad={rad}', arrowstyle='<|-|>', mutation_scale=4.5,
                                 color='#666666', lw=0.6, ls=(0, (2, 1.4)), shrinkA=3.5, shrinkB=3.5, zorder=4))
xt = 0.80
txt(xt, 2.36, r'$\mathbf{P}$ primary, hand: infers every window, caches', fontsize=6.6)
txt(xt + 0.09, 2.25, 'auxiliary class scores, and decides when to refresh', fontsize=6.6)
txt(xt, 2.13, r'$\mathbf{1,2}$ auxiliaries, chest and ankle: infer on request,', fontsize=6.6)
txt(xt + 0.09, 2.02, '(ours) push on own signal change; primary pulls if fragile', fontsize=6.6, color='#B07800')
# decision flow
y0 = 1.875
spec = [(xt, 0.52, 'paid state $x_t$'), (xt + 0.62, 0.44, r'risk $F(h)$'), (xt + 1.16, 0.76, r'interval $H=\Psi_\epsilon(F)$')]
for x, w, s in spec:
    ax.add_patch(FancyBboxPatch((x + 0.018, y0 - 0.093), w, 0.15, boxstyle='round,pad=0.008,rounding_size=0.03', fc='black', alpha=0.15, ec='none', zorder=2))
    gradient_patch(FancyBboxPatch((x, y0 - 0.075), w, 0.15, boxstyle='round,pad=0.008,rounding_size=0.03', ec='#555555', lw=0.5), '#FFFFFF', '#D9D9D9', z=3)
    txt(x + w / 2, y0, s, ha='center', fontsize=6.6, zorder=5)
for xa, xb in [(xt + 0.53, xt + 0.61), (xt + 1.07, xt + 1.15)]:
    ax.annotate('', (xb, y0), (xa, y0), arrowprops=dict(arrowstyle='-|>', lw=0.6, color='#333333', mutation_scale=6, shrinkA=0, shrinkB=0))
xb = xt + 1.54
txt(xt, 1.735, r'$\rightarrow$ refresh after $H$ windows: 30 B notice pair + score packets', fontsize=6.4)
txt(xt, 1.630, r'$\rightarrow$ until then: fuse cached auxiliary scores, 0 B', fontsize=6.4)

# ---------------- (b) timeline ----------------
txt(0.02, 1.47, '(b) One 16-window segment under five schedules', fontsize=7.4, fontweight='bold')
x0, dx = 0.97, 0.077
bx0, bscale = 2.30, 0.0027
rows = [
    ('Complete push',             'all',   [],                                       [],      100.0, '0',     None),
    ('Hazard scheduler',          'sched', [1,2,3,5,6,8,9,10,11,12,13,14,15,16],     [],      136.1, '0.08%', None),
    ('Offline schedule$^\\dagger$','sched', [1,9,10,14],                             [],      59.2,  '0',     ORANGE),
    ('Imitation, best $\\leq$1%', 'sched', [1,2,4,6,8,9,10,11,13,15],                [],      104.4, '0.96%', PINK),
    ('Two-sided trigger (ours)',  'push',  [1,2,3,5,6,8,9,10,11,13,14,16],           [],      87.6,  '1.02%', '#E69F00'),
]
PULLS = (9,)
ys = 1.27
for w in range(1, 17):
    ax.add_patch(Rectangle((x0 + (w - 1) * dx, ys - 0.045), dx, 0.09, fc='#ECECEC' if w < 10 else '#CFCFCF', ec='white', lw=0.5))
txt(x0 + 4.5 * dx, ys, 'activity A', ha='center', fontsize=6)
txt(x0 + 12.5 * dx, ys, 'activity B', ha='center', fontsize=6)
txt(0.02, ys, 'true activity', fontsize=6.6, color='#444444')
ax.plot([x0 + 9 * dx] * 2, [ys - 0.055, ys - 0.20 - 4 * 0.19 - 0.06], color=RED, lw=0.5, ls=(0, (2, 1.5)), zorder=0.5)
txt(bx0, ys, 'bytes vs. complete push', fontsize=6.2, color='#444444')
txt(3.48, ys, 'disagr.', fontsize=6.2, color='#444444', ha='right')

dy = 0.19
for i, (lab, kind, ref, bad, byt, dis, col) in enumerate(rows):
    y = ys - 0.20 - i * dy
    txt(0.02, y, lab, fontsize=6.8, color=col or '#000000')
    ax.plot([x0, x0 + 16 * dx], [y, y], color='#E3E3E3', lw=0.5, zorder=1)
    for w in range(1, 17):
        xc = x0 + (w - 0.5) * dx
        if kind == 'all' or w in ref:
            cube(xc, y, BLUE)
            if kind == 'sched' or (kind == 'push' and w in PULLS):
                ax.plot([xc, xc], [y + 0.05, y + 0.085], color='#444444', lw=0.9, zorder=3, solid_capstyle='butt')
        else:
            ax.add_patch(Circle((xc, y), 0.015, fc='white', ec='#9A9A9A', lw=0.5, zorder=3))
        if w in bad:
            ax.plot(xc, y, marker='x', color=RED, ms=5, mew=1.2, zorder=6)
    ax.add_patch(Rectangle((bx0, y - 0.042), byt * bscale, 0.084, fc=col or '#B5B5B5', ec='none', alpha=0.9))
    d = byt - 100
    s = '100%' if i == 0 else (f'+{d:.1f}%' if d > 0 else f'−{-d:.1f}%')
    txt(bx0 + byt * bscale + 0.03, y, s, fontsize=6.4)
    txt(3.48, y, dis, ha='right', fontsize=6.4)
ax.plot([bx0 + 100 * bscale] * 2, [ys - 0.27, ys - 0.20 - 4 * dy + 0.07], color='#555555', lw=0.5, ls=(0, (1.5, 1)), zorder=5)

yl = 0.07
cube(0.07, yl, BLUE)
txt(0.12, yl, 'score packet', fontsize=6.2)
ax.plot([0.66, 0.66], [yl - 0.04, yl + 0.04], color='#444444', lw=0.9); txt(0.71, yl, 'notice pair', fontsize=6.2)
ax.add_patch(Circle((1.26, yl), 0.015, fc='white', ec='#9A9A9A', lw=0.5)); txt(1.31, yl, 'cache reuse', fontsize=6.2)
ax.plot([1.88, 1.88], [yl - 0.045, yl + 0.045], color=RED, lw=0.6, ls=(0, (2, 1.5))); txt(1.93, yl, 'activity change at window 10', fontsize=6.2)

fig.savefig('system_overview.pdf', dpi=600); fig.savefig('system_overview.png', dpi=300)
