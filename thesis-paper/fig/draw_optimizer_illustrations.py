"""
Illustrative figures for PSO and SA.
SA includes radii q1/q2 as neighbourhood brackets.
Text is kept in strict vertical zones to prevent any overlap.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5})

FW  = 5.90
rng = np.random.default_rng(3)

ASSET_COLS = ['#1565C0', '#6A1B9A', '#C55A11', '#C00000', '#375623']
W_OPT  = np.array([0.38, 0.27, 0.18, 0.11, 0.06])
W_MID  = np.array([0.28, 0.24, 0.22, 0.16, 0.10])
W_INIT = np.array([0.23, 0.21, 0.21, 0.19, 0.16])

STRIP_W = 0.28   # fixed width for all weight strips in SA


def weight_strip(ax, cx, y0, weights, total_w, height):
    """Proportional colour strip centred at cx."""
    x = cx - total_w / 2
    for w, c in zip(weights, ASSET_COLS):
        seg = w * total_w
        ax.add_patch(Rectangle((x, y0), seg * 0.96, height,
                                color=c, zorder=9, clip_on=False))
        x += seg


def bracket(ax, pos, q, y_br, label, col='#555555'):
    """Horizontal bracket at y_br showing the search radius."""
    h = 0.05
    x0, x1 = pos - q, pos + q
    ax.plot([x0, x1], [y_br, y_br],     color=col, lw=0.9, zorder=6)
    ax.plot([x0, x0], [y_br-h, y_br+h], color=col, lw=0.9, zorder=6)
    ax.plot([x1, x1], [y_br-h, y_br+h], color=col, lw=0.9, zorder=6)
    ax.text(pos, y_br - h - 0.07, label,
            ha='center', va='top', fontsize=6.2, color=col, zorder=7)


# =============================================================================
# PSO – 4 panels: Iteracija 1 · Iteracija 2 · Iteracija t · Konvergenca
# =============================================================================
def landscape(x, y):
    return (0.50*(x-0.72)**2 + 0.80*(y-0.58)**2
            + 0.35*np.exp(-12*((x-0.22)**2+(y-0.78)**2))
            + 0.25*np.exp(-10*((x-0.82)**2+(y-0.18)**2)))

gx = np.linspace(0, 1, 220);  gy = np.linspace(0, 1, 220)
GX, GY = np.meshgrid(gx, gy);  GZ = landscape(GX, GY)
OPT = np.array([0.72, 0.58])

# SW positions: each row = one panel, each column = one particle.
# All particles move monotonically toward OPT=[0.72,0.58] so velocity arrows
# always point in a sensible direction.
# Panel 0 → 1: ~35% of the way from start to OPT
# Panel 1 → 2: ~60% of the way from start to OPT
# Panel 2 → 3: converged cluster near OPT
_S0 = np.array([[0.13, 0.18], [0.18, 0.80], [0.66, 0.12],
                [0.88, 0.42], [0.48, 0.88]])
_t1, _t2 = 0.22, 0.62
SW = [
    _S0,
    _S0 + (_t1) * (OPT - _S0),   # panel 1
    _S0 + (_t2) * (OPT - _S0),   # panel 2  — HL particle ends at ≈[0.48,0.42]
    np.array([[0.70, 0.55], [0.71, 0.60], [0.73, 0.57],
              [0.72, 0.60], [0.74, 0.58]]),
]
# Slightly jitter panel-2 non-HL particles so they don't land on arrow endpoints.
# HL=0 arrow endpoints (vp/vg/vt) computed below; keep P1,P2 away from them.
SW[2] = SW[2].copy()
# HL particle (P1, starts top-left): place it in centre so pb (near start) pulls
# upper-left while g (OPT) pulls right → clean fan, physically motivated.
SW[2][1] = np.array([0.48, 0.52])
SW[2][2] += np.array([+0.04, -0.04])   # keep P2 clear of arrow endpoints

# Personal bests: after step 1, each particle's best is still near its start
PBESTS_1 = _S0.copy()
# Panel 2: pb stays near starting area — particle found good spot early, moved on.
PBESTS_2 = np.array([
    [0.20, 0.24],   # P0: near start [0.13,0.18]
    [0.22, 0.74],   # P1 (HL): near start [0.18,0.80] → vp pulls up-left, vg pulls right
    [0.68, 0.16],   # P2: near start [0.66,0.12]
    [0.84, 0.44],   # P3: near start [0.88,0.42]
    [0.50, 0.82],   # P4: near start [0.48,0.88]
])

HL = 1   # P1: starts top-left, pb is up there, now mid-field → clean decomposition

PSO_TITLES = ['Iteracija 1', 'Iteracija 2', 'Iteracija $t$', 'Konvergenca']
PSO_TCOLS  = ['#C00000', '#C55A11', '#6A1B9A', '#375623']

fig, axes = plt.subplots(1, 4, figsize=(FW, FW * 0.46),
                          gridspec_kw={'wspace': 0.12})
fig.patch.set_facecolor('white')

for col, (ax, title, tcol) in enumerate(zip(axes, PSO_TITLES, PSO_TCOLS)):

    ax.contourf(GX, GY, GZ, levels=10, cmap='Blues_r', alpha=0.40, zorder=0)
    cs = ax.contour(GX, GY, GZ, levels=10, colors='#1565C0',
                    linewidths=0.25, alpha=0.32, zorder=1)
    ax.clabel(cs, cs.levels[2:-2:3], fmt='%.2f', fontsize=4.0,
              colors='#1565C0', inline=True, inline_spacing=1)

    pts = SW[col]

    # Draw particles — consistent size and no border across all panels
    ax.scatter(pts[:, 0], pts[:, 1], s=40, color='#1565C0',
               edgecolors='none', zorder=5)
    # Subtle roman numerals — all panels except the last (convergence)
    if col < 3:
        _roman = ['I', 'II', 'III', 'IV', 'V']
        for m, p in enumerate(pts):
            ax.text(p[0], p[1], _roman[m],
                    ha='center', va='center', fontsize=3.2,
                    color='white', alpha=0.55, zorder=8)

    # Personal bests: small, subtle diamonds
    if col == 1:
        ax.scatter(PBESTS_1[:, 0], PBESTS_1[:, 1], s=18, marker='D',
                   color='#9C4DCC', alpha=0.55, linewidths=0, zorder=4)
    if col == 2:
        ax.scatter(PBESTS_2[:, 0], PBESTS_2[:, 1], s=18, marker='D',
                   color='#9C4DCC', alpha=0.55, linewidths=0, zorder=4)

    # Global optimum star
    ax.scatter(*OPT, s=160, marker='*', color='#C00000',
               edgecolors='white', linewidths=0.6, zorder=7)

    if col == 1:
        _arrow_len = 0.18   # fixed display length so shrinkA never eats short arrows
        for m in range(len(pts)):
            x_cur = SW[1][m]
            v_dir = SW[2][m] - x_cur
            if np.linalg.norm(v_dir) < 0.01:
                continue
            v_draw = v_dir / np.linalg.norm(v_dir) * _arrow_len
            ax.annotate('', xy=x_cur + v_draw, xytext=x_cur,
                        arrowprops=dict(arrowstyle='->', color='#C55A11',
                                        lw=1.0, mutation_scale=8,
                                        shrinkA=4, shrinkB=2),
                        zorder=6)

    if col == 2:
        xi = SW[2][HL]
        pb = PBESTS_2[HL]
        scale = 0.55
        vp = (pb - xi) * scale
        vg = (OPT - xi) * scale
        vt = vp + vg
        ep_p = xi + vp
        ep_g = xi + vg
        ep_t = xi + vt

        def arr(end, color, ls='-'):
            ax.annotate('', xy=end, xytext=xi,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.0, linestyle=ls,
                                        mutation_scale=8,
                                        shrinkA=4, shrinkB=2),
                        zorder=7)

        arr(ep_p, '#6A1B9A')
        arr(ep_g, '#C00000')
        arr(ep_t, '#C55A11')

        ax.text(OPT[0] + 0.04, OPT[1] + 0.04, '$g$',
                ha='left', va='bottom', fontsize=6.0, color='#C00000', zorder=9)

    if col == 3:
        ax.text(OPT[0] + 0.04, OPT[1] + 0.05, '$\\vec{w}^*$',
                ha='left', va='bottom', fontsize=6.2, color='#C00000', zorder=10)

    # contour label in panel 0 only
    if col == 0:
        ax.text(0.50, 0.03, 'izolinije: $f(\\vec{w})=-$Sharpe',
                ha='center', va='bottom', fontsize=5.0,
                color='#1565C0', transform=ax.transAxes, zorder=9)

    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xticks([]);   ax.set_yticks([])
    ax.set_title(title, fontsize=7.0, pad=5, color=tcol)
    if col == 0:
        ax.set_ylabel('$f(\\vec{w})$', fontsize=6.5, labelpad=3)
    if col == 1:
        ax.set_xlabel('Utež $w_i$ / $w_j$  (2D prikaz)', fontsize=6.0, labelpad=3)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
    for s in ['bottom', 'left']:  ax.spines[s].set_linewidth(0.6)

leg_h = [
    plt.scatter([], [], s=40, color='#1565C0', edgecolors='none'),
    plt.scatter([], [], s=18, marker='D', color='#9C4DCC', alpha=0.55),
    plt.scatter([], [], s=160, marker='*', color='#C00000', edgecolors='white', lw=0.6),
    Line2D([0], [0], color='#C55A11', lw=1.0, marker='>', markersize=5),
    Line2D([0], [0], color='#6A1B9A', lw=1.0, marker='>', markersize=5),
    Line2D([0], [0], color='#C00000', lw=1.0, marker='>', markersize=5),
]
leg_l = [
    'kandidatni portfelj (delec roja)',
    'osebni optimum $p_i$',
    'najboljši portfelj roja $g$',
    'skupna hitrost $v_m$',
    'privlak $p_i$ (komponenta $v_p$)',
    'privlak $g$ (komponenta $v_g$)',
]
fig.legend(leg_h, leg_l, loc='lower center', ncol=3,
           fontsize=5.8, framealpha=0.0, handlelength=1.4,
           columnspacing=1.0, bbox_to_anchor=(0.5, -0.08))

fig.savefig('optimizer_pso_illus.pdf', bbox_inches='tight', dpi=200)
plt.close()
print('OK  optimizer_pso_illus.pdf')


# =============================================================================
# SA – 3 panels with strict y-zones so nothing overlaps:
#   TOP    y in [Y_STRIP, Y_STRIP+0.20]   weight strip
#   MIDDLE y in [0, Y_STRIP-0.10]         landscape + arcs
#   BOTTOM y in [Y_BR-0.20, 0]            radius bracket + label
# =============================================================================
def curve1d(x):
    return (0.55*np.sin(3.5*x) + 0.38*np.sin(7*x)
            + 1.1*(x - 0.65)**2 + 0.35)

xs1d   = np.linspace(0, 1, 400)
ys1d   = curve1d(xs1d)
GMIN_X = xs1d[np.argmin(ys1d)]
GMIN_Y = ys1d.min()

C_MAX  = ys1d.max()
Y_BR   = GMIN_Y - 0.26    # y of radius bracket
YMIN   = Y_BR   - 0.20    # plot bottom (label room)
YMAX   = C_MAX  + 0.38    # plot top
Y_GAP  = C_MAX  + 0.08    # fixed y for annotation

Q1 = 0.25    # large radius (initial)
Q2 = 0.09    # small radius (after A < A_min)

# Consecutive positions: panel N starts where panel N-1's step ended.
#   Panel 1: 0.28 ->(+0.22)-> 0.50  (accepted, downhill)
#   Panel 2: 0.50 ->(-0.22)-> stay  (rejected, uphill)  -> A_min -> q2
#   Panel 3: 0.50 ->(+0.08)-> 0.58  (accepted, small q2 step)
P1_START = 0.28
P1_END   = P1_START + 0.22   # 0.50  (accepted)
P2_END   = P1_END             # 0.50  (rejected, stays)
stages = [
    dict(label='Visok $T$',
         pos=P1_START, q=Q1, q_label='$q_1$', q_col='#C00000', T_col='#C00000',
         jumps=[(+0.22, True)]),
    dict(label='Srednji $T$',
         pos=P1_END, q=Q1, q_label='$q_1$', q_col='#C55A11', T_col='#C55A11',
         jumps=[(-0.22, False)]),
    dict(label='Nizek $T$',
         pos=P2_END, q=Q2, q_label='$q_2$', q_col='#375623', T_col='#375623',
         jumps=[(+0.08, True)]),
]

fig, axes = plt.subplots(1, 3, figsize=(FW, FW * 0.44),
                          gridspec_kw={'wspace': 0.16})
fig.patch.set_facecolor('white')

for col, (ax, st) in enumerate(zip(axes, stages)):
    pos = st['pos']
    cy  = curve1d(np.array([pos]))[0]

    # landscape
    ax.plot(xs1d, ys1d, color='#1565C0', lw=1.5, zorder=3)
    ax.fill_between(xs1d, ys1d, YMIN, alpha=0.09, color='#1565C0', zorder=2)

    # neighbourhood shading (vertical stripe)
    ax.axvspan(np.clip(pos - st['q'], 0, 1),
               np.clip(pos + st['q'], 0, 1),
               ymin=0, ymax=1, alpha=0.15,
               color=st['q_col'], zorder=1)

    # global optimum
    ax.scatter(GMIN_X, GMIN_Y, s=100, marker='*', color='#C00000',
               edgecolors='white', linewidths=0.5, zorder=6)

    # current portfolio dot
    ax.scatter(pos, cy, s=62, color='#1565C0',
               edgecolors='white', linewidths=0.8, zorder=7)

    # jump arcs
    cur = pos
    for arc_idx, (delta, accepted) in enumerate(st['jumps']):
        nxt = np.clip(cur + delta, 0.02, 0.98)
        cy0 = curve1d(np.array([cur]))[0]
        cy1 = curve1d(np.array([nxt]))[0]
        color = '#375623' if accepted else '#C00000'
        rad = -0.35 * np.sign(delta)
        ax.annotate('', xy=(nxt, cy1), xytext=(cur, cy0),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color, lw=1.0,
                        linestyle='-' if accepted else '--',
                        connectionstyle=f'arc3,rad={rad}',
                        mutation_scale=10),
                    zorder=5)
        # label first arc of panel 1 only
        if col == 0 and arc_idx == 0:
            mx = (cur + nxt) / 2
            my = C_MAX + 0.28
            ax.text(mx, my, "sosed $\\vec{w}'$",
                    ha='center', va='bottom', fontsize=5.5,
                    color='#375623', zorder=8)
        if accepted:
            cur = nxt

    # label q switch in panel 3 only
    if col == 2:
        ax.text(0.50, Y_GAP,
                '$T$ pada  $\\Rightarrow$  $q_1 \\to q_2$',
                ha='center', va='bottom', fontsize=5.5,
                color='#375623', zorder=10)

    # radius bracket (fixed y = Y_BR)
    bracket(ax, pos, st['q'], Y_BR, st['q_label'], st['q_col'])

    ax.set_xlim(0, 1);  ax.set_ylim(YMIN, YMAX)
    ax.set_xticks([]);   ax.set_yticks([])
    ax.set_title(st['label'], fontsize=6.8, pad=5, color=st['T_col'])
    if col == 0:
        ax.set_ylabel('$f(\\vec{w})$ = $-$Sharpe', fontsize=6.5, labelpad=3)
    if col == 1:
        ax.set_xlabel('Portfelj $\\vec{w}$', fontsize=6.5, labelpad=3)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
    for s in ['bottom', 'left']:  ax.spines[s].set_linewidth(0.6)


leg2_h = [
    plt.scatter([], [], s=55, color='#1565C0', edgecolors='white', lw=0.7),
    plt.scatter([], [], s=100, marker='*', color='#C00000', edgecolors='white', lw=0.5),
    Line2D([0], [0], color='#375623', lw=1.2, marker='>', markersize=6),
    Line2D([0], [0], color='#C00000', lw=1.0, ls='--', marker='>', markersize=6),
]
leg2_l = [
    'portfelj $\\vec{w}$',
    'optimum $\\vec{w}^*$',
    'sprejet sosed $\\vec{w}\'$',
    'zavrnjen sosed $\\vec{w}\'$',
]
fig.legend(leg2_h, leg2_l, loc='lower center', ncol=4,
           fontsize=6.5, framealpha=0.0, handlelength=1.4,
           columnspacing=1.2, bbox_to_anchor=(0.5, -0.03))

fig.savefig('optimizer_sa_illus.pdf', bbox_inches='tight', dpi=200)
plt.close()
print('OK  optimizer_sa_illus.pdf')
