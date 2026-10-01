"""
Illustrative figure for the walk-forward evaluation protocol (Slika fig:proto-schema).
Same visual language as draw_optimizer_illustrations.py (PSO/SA): serif font,
shared colour palette, no numeric axis clutter, bracket-style span labels,
legend at the bottom.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5})

FW = 5.90

# Pastel fills, with a darker tone of the same hue for brackets/labels so
# text stays legible against white.
TRAIN_COL = '#D4D4D4'
INPUT_COL = '#8FB8E8'
MU_COL    = '#9FC79A'
MU_DARK   = '#4C7A46'
COV_COL   = '#C3A8DE'
COV_DARK  = '#7A56A3'
HOLD_COL  = '#F2B183'
HOLD_DARK = '#B06A2E'
DEC_COL   = '#B33F3F'

# Abstract (non-numeric) time positions -- only relative order matters.
T_START   = -1.00
LMAX_START = -0.60
L_START   = -0.35
VAL_START = -0.12
DECISION  = 0.0
H_END     = 0.32

XLIM = (-1.14, 0.46)
BAR_H = 0.34
STEP  = 0.95

# Top to bottom: conditioning/holding windows first, training window last.
# 'train' sits further below 'hold' than the regular STEP so the time axis
# and the H bracket/label between them both have clear room.
ROWS = {
    'input': 3 * STEP,
    'mu':    2 * STEP,
    'cov':   1 * STEP,
    'hold':  0 * STEP,
    'train': -1.35 * STEP,
}


def bracket(ax, x0, x1, y, col, label):
    """Horizontal bracket with end ticks, labelled underneath -- same idiom
    as the q1/q2 radius brackets in the SA illustration."""
    h = 0.05
    ax.plot([x0, x1], [y, y], color=col, lw=0.9, zorder=6)
    ax.plot([x0, x0], [y - h, y + h], color=col, lw=0.9, zorder=6)
    ax.plot([x1, x1], [y - h, y + h], color=col, lw=0.9, zorder=6)
    ax.text((x0 + x1) / 2, y - h - 0.11, label, ha='center', va='top',
             fontsize=7.0, color=col, zorder=6)


fig, ax = plt.subplots(1, 1, figsize=(FW, FW * 0.34))
fig.patch.set_facecolor('white')

top_edge = ROWS['input'] + BAR_H

# Future (post-decision) shading -- the one thing that must stay unmissable:
# nothing computed for a decision may look right of this line.
ax.axvspan(DECISION, XLIM[1], color='#9E9E9E', alpha=0.15, zorder=0)
ax.text(DECISION + 0.03, top_edge - 0.10,
        'prihodnost (ni na voljo)', ha='left', va='top',
        fontsize=6.3, color='#616161', style='italic', zorder=2)

# Decision-time marker (labelled only in the legend, to avoid clutter above
# the plot).
ax.axvline(DECISION, color='black', lw=1.1, ls='--', zorder=5)

# -- Conditioning windows (L): model input & matched historical baseline
ax.add_patch(Rectangle((L_START, ROWS['input']), DECISION - L_START, BAR_H,
                        facecolor=INPUT_COL, edgecolor='none', zorder=3))
ax.add_patch(Rectangle((L_START, ROWS['mu']), DECISION - L_START, BAR_H,
                        facecolor=MU_COL, edgecolor='none', zorder=3))
bracket(ax, L_START, DECISION, ROWS['mu'] - 0.12, MU_DARK, r'$L$')

# -- Covariance window (L_max) ------------------------------------------
ax.add_patch(Rectangle((LMAX_START, ROWS['cov']), DECISION - LMAX_START, BAR_H,
                        facecolor=COV_COL, edgecolor='none', zorder=3))
bracket(ax, LMAX_START, DECISION, ROWS['cov'] - 0.12, COV_DARK, r'$L_{\max}$')

# -- Holding period (H) --------------------------------------------------
ax.add_patch(Rectangle((DECISION, ROWS['hold']), H_END - DECISION, BAR_H,
                        facecolor=HOLD_COL, edgecolor='none', zorder=3))
bracket(ax, DECISION, H_END, ROWS['hold'] - 0.10, HOLD_DARK, r'$H$')

# -- Training window, with the validation tail hatched -- moved to the
# bottom row, below the decision-time axis arrow.
ax.add_patch(Rectangle((T_START, ROWS['train']), VAL_START - T_START, BAR_H,
                        facecolor=TRAIN_COL, edgecolor='none', zorder=3))
ax.add_patch(Rectangle((VAL_START, ROWS['train']), DECISION - VAL_START, BAR_H,
                        facecolor=TRAIN_COL, edgecolor='black', lw=0.4,
                        hatch='////', zorder=4))

# Baseline "time" arrow, between the holding row and the training row.
y_axis = ROWS['hold'] - 0.55
ax.annotate('', xy=(XLIM[1], y_axis), xytext=(XLIM[0], y_axis),
            arrowprops=dict(arrowstyle='-|>', color='black', lw=0.9,
                             mutation_scale=8), zorder=2)
ax.text(XLIM[1] + 0.02, y_axis, 'trgovalni dnevi', ha='left', va='center',
        fontsize=6.8, color='#333333')

ax.set_xlim(*XLIM)
ax.set_ylim(ROWS['train'] - 0.25, top_edge + 0.15)
ax.set_xticks([]); ax.set_yticks([])
for s in ('top', 'right', 'left', 'bottom'):
    ax.spines[s].set_visible(False)

leg_h = [
    Patch(facecolor=INPUT_COL, label=r'vhod v model ($L$ dni)'),
    Patch(facecolor=MU_COL, label=r'zgodovinsko povprečje $\hat{\mu}$ ($L$ dni)'),
    Patch(facecolor=COV_COL, label=r'kovarianca $\hat{\Sigma}$ ($L_{\max}$ dni)'),
    Patch(facecolor=HOLD_COL, label=r'napovedno obdobje ($H$ dni)'),
    Patch(facecolor=TRAIN_COL, label='učenje modela'),
    Patch(facecolor=TRAIN_COL, edgecolor='black', lw=0.4, hatch='////',
          label='validacija (zgodnja ustavitev)'),
    Line2D([0], [0], color='black', lw=1.1, ls='--', label='odločitveni trenutek'),
]
fig.legend(handles=leg_h, loc='lower center', ncol=4, fontsize=6.3,
           framealpha=0.0, handlelength=1.4, columnspacing=1.1,
           bbox_to_anchor=(0.5, -0.14))

fig.savefig('protocol_schema.pdf', bbox_inches='tight', dpi=200)
plt.close()
print('OK  protocol_schema.pdf')
