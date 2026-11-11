"""
Illustrative figure for SA (the PSO figure is TikZ: optimizer_pso_illus.tex).
SA includes radii q1/q2 as neighbourhood brackets.
Text is kept in strict vertical zones to prevent any overlap.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5})

FW  = 5.90


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
# SA – 3 panels with strict y-zones so nothing overlaps:
#   TOP    y in [Y_STRIP, Y_STRIP+0.20]   weight strip
#   MIDDLE y in [0, Y_STRIP-0.10]         landscape + arcs
#   BOTTOM y in [Y_BR-0.20, 0]            radius bracket + label
# =============================================================================
# Qualitative 1D stand-in for the KONP objective g(w) from
# eq:solver-objective: not the literal quadratic form, but shaped with
# several local optima so the figure can show SA escaping one before settling
# near the global optimum (star).
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
    ax.text(pos - 0.04, cy + 0.06, '$\\vec{w}$', ha='right', va='bottom',
            fontsize=6.2, color='#1565C0', zorder=9)

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
        # candidate neighbour w': hollow circle, green = accepted, red dashed = rejected
        ax.scatter(nxt, cy1, s=62, facecolors='white', edgecolors=color,
                   linewidths=1.1, linestyle='-' if accepted else '--', zorder=8)
        ax.text(nxt + 0.035*np.sign(delta), cy1 - 0.06, "$\\vec{w}'$",
                ha='left' if delta > 0 else 'right', va='top', fontsize=6.2,
                color=color, zorder=9)
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
        ax.set_ylabel('$g(\\vec{w})$', fontsize=6.5, labelpad=3)
    if col == 1:
        ax.set_xlabel('Portfelj $\\vec{w}$', fontsize=6.5, labelpad=3)
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
    for s in ['bottom', 'left']:  ax.spines[s].set_linewidth(0.6)


leg2_h = [
    Line2D([0], [0], ls='', marker='o', markersize=6, markerfacecolor='#1565C0', markeredgecolor='white'),
    Line2D([0], [0], ls='', marker='o', markersize=6, markerfacecolor='white', markeredgecolor='#375623', markeredgewidth=1.1),
    Line2D([0], [0], ls='', marker='o', markersize=6, markerfacecolor='white', markeredgecolor='#C00000', markeredgewidth=1.1),
    Line2D([0], [0], ls='', marker='*', markersize=9, markerfacecolor='#C00000', markeredgecolor='white'),
]
leg2_l = [
    'trenutni portfelj $\\vec{w}$',
    'sprejet sosed $\\vec{w}\'$',
    'zavrnjen sosed $\\vec{w}\'$',
    'optimum $\\vec{w}^*$',
]
fig.legend(leg2_h, leg2_l, loc='lower center', ncol=4,
           fontsize=6.5, framealpha=0.0, handlelength=1.4,
           columnspacing=1.2, bbox_to_anchor=(0.5, -0.03))

fig.savefig('optimizer_sa_illus.pdf', bbox_inches='tight', dpi=200)
plt.close()
print('OK  optimizer_sa_illus.pdf')
