"""
Model architecture flow diagrams: PatchTST, TFT, MASTER.
Circles: white fill, thin border, DejaVu Sans bold — clean and centred.
TFT includes LSTM step between VSN and self-attention (per thesis text).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.0})

FW    = 5.90
FH    = 2.20
CY    = 0.85
EH    = 0.82
GAP   = 0.13
CR    = 0.108
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI']


# ── numbered circle ───────────────────────────────────────────────────────────

def _circ(ax, cx, elem_top, roman, bc):
    cy = elem_top + GAP + CR
    ax.add_patch(Circle((cx, cy), CR,
                        facecolor='white', edgecolor=bc,
                        lw=1.0, zorder=12, clip_on=False))
    ax.text(cx, cy - 0.007, roman,
            ha='center', va='center',
            fontsize=7.5, color=bc, fontweight='normal',
            fontfamily='DejaVu Sans',
            zorder=13, clip_on=False)


# ── primitives ────────────────────────────────────────────────────────────────

def _rbox(ax, cx, cy, w, h, text, bc, fc, fs=5.0):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle='round,pad=0.025',
                                fc=fc, ec=bc, lw=0.85, zorder=5))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color='#1a1a1a',
            multialignment='center', zorder=6)


def _tensor(ax, cx, cy, w, h, bc, fc, n=3, d=0.022,
            lbl_l='$N$', lbl_b='$T$', n_waves=5):
    for i in range(n - 1, -1, -1):
        alpha = 0.48 + 0.24 * (n - 1 - i)
        ax.add_patch(FancyBboxPatch(
            (cx - w/2 + i*d, cy - h/2 + i*d), w, h,
            boxstyle='round,pad=0.018', fc=fc, ec=bc,
            lw=0.65, alpha=alpha, zorder=3 + i))
    xs = np.linspace(cx - w/2 + 0.05, cx + w/2 - 0.05, 80)
    for k in range(n_waves):
        y0 = cy - h/2 + (k + 1) * h / (n_waves + 1)
        yw = y0 + 0.013 * np.sin(xs * 44 + k * 1.5)
        ax.plot(xs, yw, color=bc, lw=0.60, alpha=0.55, zorder=7)
    if lbl_l:
        ax.annotate('', xy=(cx - w/2 - 0.04, cy + h/2),
                    xytext=(cx - w/2 - 0.04, cy - h/2),
                    arrowprops=dict(arrowstyle='<->', color=bc,
                                   lw=0.50, mutation_scale=5))
        ax.text(cx - w/2 - 0.11, cy, lbl_l,
                ha='center', va='center', fontsize=4.3,
                color=bc, style='italic', zorder=8)
    if lbl_b:
        ax.annotate('', xy=(cx + w/2, cy - h/2 - 0.07),
                    xytext=(cx - w/2, cy - h/2 - 0.07),
                    arrowprops=dict(arrowstyle='<->', color=bc,
                                   lw=0.50, mutation_scale=5))
        ax.text(cx, cy - h/2 - 0.14, lbl_b,
                ha='center', va='top', fontsize=4.3,
                color=bc, style='italic', zorder=8)


def _patch_embed(ax, cx, cy, w, h, bc, fc, n_patches=5):
    """Rectangle with vertical patch dividers and per-patch sine waves."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle='round,pad=0.018',
                                fc=fc, ec=bc, lw=0.85, zorder=5))
    pw = w / n_patches
    for k in range(1, n_patches):
        x = cx - w/2 + k * pw
        ax.plot([x, x], [cy - h/2 + 0.06, cy + h/2 - 0.06],
                color=bc, lw=0.50, alpha=0.60, zorder=6)
    for k in range(n_patches):
        px = cx - w/2 + (k + 0.5) * pw
        xs2 = np.linspace(px - pw/2 + 0.011, px + pw/2 - 0.011, 20)
        yw2 = cy + 0.010 * np.sin(xs2 * 68 + k * 2.5)
        ax.plot(xs2, yw2, color=bc, lw=0.55, alpha=0.55, zorder=7)
    ax.text(cx, cy - h/2 - 0.10, '$L/S$ patches',
            ha='center', va='top', fontsize=4.0, color=bc, zorder=8)


def _tst_row(ax, cx, cy, w, h, bc, fc, n=3):
    """n TST blocks side-by-side (channel-independent application)."""
    gap = 0.022
    bw  = (w - (n - 1) * gap) / n
    for k in range(n):
        bx = cx - w/2 + k * (bw + gap) + bw / 2
        ax.add_patch(FancyBboxPatch((bx - bw/2, cy - h/2), bw, h,
                                    boxstyle='round,pad=0.012',
                                    fc=fc, ec=bc, lw=0.80, zorder=5))
        ax.text(bx, cy, 'TST', ha='center', va='center',
                fontsize=4.6, color='#1a1a1a', zorder=6)
        if k < n - 1:
            ax.text(bx + bw/2 + gap/2, cy, '$\\cdots$',
                    ha='center', va='center', fontsize=4.0,
                    color=bc, zorder=6)


def _parallel_boxes(ax, cx, cy, w, h, labels, bc, fc, fs=4.5):
    n  = len(labels)
    sh = (h - 0.026 * (n - 1)) / n
    for k, lbl in enumerate(labels):
        gy = cy - h/2 + k * (sh + 0.026)
        ax.add_patch(FancyBboxPatch((cx - w/2, gy), w, sh,
                                    boxstyle='round,pad=0.012',
                                    fc=fc, ec=bc, lw=0.80, zorder=5))
        ax.text(cx, gy + sh/2, lbl, ha='center', va='center',
                fontsize=fs, color='#1a1a1a', zorder=6)


def _vsn(ax, cx, cy, w, h, bc, fc, lfc, n=4):
    """Variable Selection Network: z_i → σ gate, stacked rows."""
    sh = (h - 0.020 * (n - 1)) / n
    bw_in  = w * 0.24
    bw_sig = w * 0.54
    for k in range(n):
        gy = cy - h/2 + k * (sh + 0.020)
        lbl = f'$z_{{{k+1}}}$' if k < n - 1 else '$\\vdots$'
        ax.text(cx - w/2 - 0.06, gy + sh/2, lbl,
                ha='right', va='center', fontsize=3.8, color=bc)
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, gy + sh*0.12), bw_in, sh * 0.76,
            boxstyle='round,pad=0.004', fc=lfc, ec=bc, lw=0.50, zorder=5))
        bx = cx - w/2 + bw_in + 0.018
        ax.add_patch(FancyBboxPatch(
            (bx, gy), bw_sig, sh,
            boxstyle='round,pad=0.006', fc=fc, ec=bc, lw=0.75, zorder=5))
        ax.text(bx + bw_sig/2, gy + sh/2, '$\\sigma$',
                ha='center', va='center', fontsize=5.5,
                color='#1a1a1a', zorder=6)
        ax.annotate('', xy=(bx, gy + sh/2),
                    xytext=(cx - w/2 + bw_in, gy + sh/2),
                    arrowprops=dict(arrowstyle='->', color=bc,
                                   lw=0.45, mutation_scale=5), zorder=6)


def _lstm(ax, cx, cy, w, h, bc, fc):
    """LSTM block with recurrence arc above (shows temporal feedback)."""
    _rbox(ax, cx, cy, w, h, 'LSTM', bc, fc, fs=5.5)
    # recurrence arc: from right top → over box → left top
    arc_y = cy + h/2 + 0.11
    lx, rx = cx - w*0.25, cx + w*0.25
    verts = [(rx, cy + h/2),
             (rx, arc_y),
             (lx, arc_y),
             (lx, cy + h/2)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    ax.add_patch(PathPatch(Path(verts, codes),
                           facecolor='none', edgecolor=bc,
                           lw=0.65, zorder=8, clip_on=False))
    ax.annotate('', xy=(lx, cy + h/2 + 0.015),
                xytext=(lx, cy + h/2 + 0.055),
                arrowprops=dict(arrowstyle='->', color=bc,
                                lw=0.65, mutation_scale=6),
                zorder=9, annotation_clip=False)


def _attn(ax, cx, cy, size, bc, fc_hex):
    """T×T self-attention matrix with diagonal-fade heatmap."""
    n = 7
    cell = size / n
    r0 = int(fc_hex[1:3], 16) / 255
    g0 = int(fc_hex[3:5], 16) / 255
    b0 = int(fc_hex[5:7], 16) / 255
    for row in range(n):
        for col in range(n):
            v = np.exp(-0.28 * (row - col)**2 / n)
            rgba = (r0, g0, b0, 0.05 + 0.86 * v)
            ax.add_patch(Rectangle(
                (cx - size/2 + col*cell + 0.003,
                 cy - size/2 + row*cell + 0.003),
                cell - 0.006, cell - 0.006,
                fc=rgba, ec='none', zorder=5))
    ax.add_patch(FancyBboxPatch(
        (cx - size/2, cy - size/2), size, size,
        boxstyle='round,pad=0.010', fc='none', ec=bc,
        lw=0.90, zorder=6))
    for k in range(1, n):
        t = cx - size/2 + k * cell
        ax.plot([t, t], [cy - size/2 - 0.010, cy - size/2],
                color=bc, lw=0.35, zorder=7)
        ax.plot([cx - size/2 - 0.010, cx - size/2],
                [cy - size/2 + k*cell, cy - size/2 + k*cell],
                color=bc, lw=0.35, zorder=7)
    ax.text(cx, cy - size/2 - 0.10, '$T\\!\\times\\!T$',
            ha='center', va='top', fontsize=4.5, color=bc, zorder=8)


def _col_vec(ax, cx, cy, w, h, bc, fc,
             top_lbl='', bot_lbl='', label_right=''):
    """Column vector with labelled top and bottom cells, dots in middle."""
    rows = [top_lbl, '$\\vdots$', bot_lbl]
    nrow = len(rows)
    ch   = h / nrow
    for i, lbl in enumerate(rows):
        dot = lbl in ('$\\vdots$', '')
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2 + i*ch), w, ch * 0.84,
            boxstyle='round,pad=0.006',
            fc='white' if dot else fc,
            ec='none' if dot else bc,
            lw=0.60, zorder=5))
        if lbl and lbl != '$\\vdots$':
            ax.text(cx, cy - h/2 + i*ch + ch*0.42,
                    f'${lbl}$', ha='center', va='center',
                    fontsize=3.8, color='#1a1a1a', zorder=6)
        elif lbl == '$\\vdots$':
            ax.text(cx, cy - h/2 + i*ch + ch*0.42,
                    '$\\vdots$', ha='center', va='center',
                    fontsize=5.5, color=bc, zorder=6)
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.010',
        fc='none', ec=bc, lw=0.80, zorder=7))
    if label_right:
        ax.text(cx + w/2 + 0.10, cy, label_right,
                ha='left', va='center',
                fontsize=10, style='italic', fontweight='normal', zorder=8)


def _arr(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color='#666',
                                lw=0.85, mutation_scale=9), zorder=4)


def _new_fig():
    fig = plt.figure(figsize=(FW, FH))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    return fig, ax


def _connect(ax, elems):
    for i in range(len(elems) - 1):
        x0 = elems[i]['x']   + elems[i]['w']   / 2
        x1 = elems[i+1]['x'] - elems[i+1]['w'] / 2
        _arr(ax, x0 + 0.012, x1 - 0.012, CY)


# ── PatchTST  (green) ─────────────────────────────────────────────────────────
# Text: patch the input → channel-independent TST → linear head → pretvorba v μ̂
BC_P  = '#2E6B35'
FC_P  = '#BFE0C3'
LFC_P = '#EAF5EC'


def draw_patchtst():
    fig, ax = _new_fig()
    elems = [
        dict(x=0.48, w=0.52, h=EH),    # I   input tensor n×T
        dict(x=1.38, w=0.60, h=0.72),  # II  patch embedding L/S
        dict(x=2.44, w=0.86, h=EH),    # III TST ×n (channel-independent)
        dict(x=3.48, w=0.30, h=EH),    # IV  head col-vec h_i
        dict(x=4.46, w=0.74, h=0.70),  # V   linear projection → pretvorba
        dict(x=5.50, w=0.28, h=EH),    # VI  output μ̂
    ]

    for ri, e in enumerate(elems):
        cx, w, h = e['x'], e['w'], e['h']
        if ri == 0:
            _tensor(ax, cx, CY, w, h, BC_P, FC_P,
                    lbl_l='$n$', lbl_b='$T$')
        elif ri == 1:
            _patch_embed(ax, cx, CY, w, h, BC_P, FC_P)
        elif ri == 2:
            _tst_row(ax, cx, CY, w, h, BC_P, FC_P)
            # label clarifying channel-independence
            ax.text(cx, CY - h/2 - 0.09, '(neodvisnost kanalov)',
                    ha='center', va='top', fontsize=3.6,
                    color=BC_P, style='italic', zorder=8)
        elif ri == 3:
            _col_vec(ax, cx, CY, w, h, BC_P, FC_P,
                     top_lbl='h_1', bot_lbl='h_R')
            ax.text(cx - w/2 - 0.07, CY, '$h_i$',
                    ha='right', va='center', fontsize=4.5,
                    color=BC_P, style='italic')
        elif ri == 4:
            _rbox(ax, cx, CY, w, h,
                  'Pretvorba\nv $\\hat{\\mu}$', BC_P, FC_P, fs=5.2)
        elif ri == 5:
            _col_vec(ax, cx, CY, w, h, BC_P, FC_P,
                     top_lbl='\\hat{\\mu}_1', bot_lbl='\\hat{\\mu}_n',
                     label_right='$\\hat{\\boldsymbol{\\mu}}$')
        _circ(ax, cx, CY + h/2, ROMAN[ri], BC_P)

    _connect(ax, elems)
    fig.savefig('model_flow_patchtst.pdf', bbox_inches='tight', dpi=200)
    plt.close()
    print('OK  model_flow_patchtst.pdf')


# ── TFT  (teal) ───────────────────────────────────────────────────────────────
# Text: VSN → LSTM (local patterns) → self-attention (long-range) → kvantili
BC_T  = '#00695C'
FC_T  = '#9ED4CE'
LFC_T = '#DFF4F1'


def draw_tft():
    fig, ax = _new_fig()
    SA = 0.64   # attention square side
    elems = [
        dict(x=0.44, w=0.50, h=EH),    # I   input tensor N×T
        dict(x=1.30, w=0.62, h=EH),    # II  VSN: z_i → σ
        dict(x=2.22, w=0.52, h=0.68),  # III LSTM (local temporal)
        dict(x=3.16, w=SA,   h=SA),    # IV  self-attention T×T
        dict(x=4.16, w=0.66, h=0.66),  # V   quantile distil → μ̂
        dict(x=5.20, w=0.28, h=EH),    # VI  output col-vec
    ]

    for ri, e in enumerate(elems):
        cx, w, h = e['x'], e['w'], e['h']
        if ri == 0:
            _tensor(ax, cx, CY, w, h, BC_T, FC_T,
                    lbl_l='$N$', lbl_b='$T$')
        elif ri == 1:
            _vsn(ax, cx, CY, w, h, BC_T, FC_T, LFC_T, n=4)
        elif ri == 2:
            _lstm(ax, cx, CY, w, h, BC_T, FC_T)
        elif ri == 3:
            _attn(ax, cx, CY, SA, BC_T, '#9ED4CE')
        elif ri == 4:
            _rbox(ax, cx, CY, w, h,
                  'Kvant. don.\n$\\to$ mediana\n$\\to\\hat{\\mu}$',
                  BC_T, FC_T, fs=4.8)
        elif ri == 5:
            _col_vec(ax, cx, CY, w, h, BC_T, FC_T,
                     top_lbl='\\hat{\\mu}_1', bot_lbl='\\hat{\\mu}_N',
                     label_right='$\\hat{\\boldsymbol{\\mu}}$')
        _circ(ax, cx, CY + h/2, ROMAN[ri], BC_T)

    _connect(ax, elems)
    fig.savefig('model_flow_tft.pdf', bbox_inches='tight', dpi=200)
    plt.close()
    print('OK  model_flow_tft.pdf')


# ── MASTER  (orange I/O, purple attention) ────────────────────────────────────
# Text: Alpha158+tržne znač. → market gating → intra-stock attn →
#        inter-stock (cross) attn → scaling → μ̂
BC_MO  = '#B84500'
FC_MO  = '#FDDCB0'
LFC_MO = '#FFF3E0'
BC_MP  = '#5E2A8C'
FC_MP  = '#D4BBEC'
LFC_MP = '#F0E8FA'


def draw_master():
    fig, ax = _new_fig()
    elems = [
        dict(x=0.48, w=0.58, h=EH,   bc=BC_MO, fc=FC_MO),  # I   N×T×F tensor
        dict(x=1.46, w=0.64, h=0.70, bc=BC_MO, fc=FC_MO),  # II  market gating
        dict(x=2.60, w=0.86, h=EH,   bc=BC_MP, fc=FC_MP),  # III intra-stock attn ×N
        dict(x=3.76, w=0.72, h=0.72, bc=BC_MP, fc=FC_MP),  # IV  inter-stock (cross) attn
        dict(x=4.76, w=0.62, h=0.62, bc=BC_MO, fc=FC_MO),  # V   skaliranje
        dict(x=5.56, w=0.28, h=EH,   bc=BC_MO, fc=FC_MO),  # VI  output
    ]

    for ri, e in enumerate(elems):
        cx, w, h = e['x'], e['w'], e['h']
        bc, fc = e['bc'], e['fc']
        if ri == 0:
            _tensor(ax, cx, CY, w, h, bc, fc,
                    lbl_l='$N$', lbl_b='$T{\\!\\times\\!}F$',
                    n_waves=5)
        elif ri == 1:
            _rbox(ax, cx, CY, w, h,
                  'Tržno-vod.\nuteževanje\n$\\vec{m}_t$', bc, fc, fs=4.6)
        elif ri == 2:
            # intra-stock: one TST-style attention per stock
            _parallel_boxes(ax, cx, CY, w, h,
                            ['attn  $i=1$', '$\\cdots$', 'attn  $i=N$'],
                            bc, fc, fs=4.5)
            ax.text(cx, CY - h/2 - 0.09, '(znotraj delnice)',
                    ha='center', va='top', fontsize=3.6,
                    color=bc, style='italic', zorder=8)
        elif ri == 3:
            # inter-stock (cross-stock) aggregation
            _rbox(ax, cx, CY, w, h,
                  'Presečna\npozornost\n(med delnicami)', bc, fc, fs=4.4)
        elif ri == 4:
            _rbox(ax, cx, CY, w, h,
                  'Skaliranje\ndonosov', bc, fc)
        elif ri == 5:
            _col_vec(ax, cx, CY, w, h, bc, fc,
                     top_lbl='\\hat{\\mu}_1', bot_lbl='\\hat{\\mu}_N',
                     label_right='$\\hat{\\boldsymbol{\\mu}}$')
        _circ(ax, cx, CY + h/2, ROMAN[ri], bc)

    _connect(ax, elems)
    fig.savefig('model_flow_master.pdf', bbox_inches='tight', dpi=200)
    plt.close()
    print('OK  model_flow_master.pdf')


draw_patchtst()
draw_tft()
draw_master()
