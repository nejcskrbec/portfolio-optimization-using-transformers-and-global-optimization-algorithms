import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.path import Path
import numpy as np

PAL = [
    ('#1565C0', '#DEEAF1'),   # I   blue
    ('#6A1B9A', '#F3E5F5'),   # II  purple
    ('#6A1B9A', '#F3E5F5'),   # III purple
    ('#C55A11', '#FCE4D6'),   # IV  orange
    ('#C00000', '#FFE7E7'),   # V   red
    ('#375623', '#E2EFDA'),   # VI  green
]
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI']

FW  = 5.90
BH  = 0.60
MH  = 0.18
MV  = 0.18   # slightly taller margin to make room for the loop arc above
FH  = BH + 2 * MV + 0.22   # extra vertical space for arc
CR  = 0.07
K   = 0.5523


def rounded_rect_path(cx, cy, w, h, r):
    x0, y0 = cx - w/2, cy - h/2
    x1, y1 = cx + w/2, cy + h/2
    verts = [
        (x0+r, y0),   (x1-r, y0),
        (x1-r+K*r, y0), (x1, y0+r-K*r), (x1, y0+r),
        (x1, y1-r),
        (x1, y1-r+K*r), (x1-r+K*r, y1), (x1-r, y1),
        (x0+r, y1),
        (x0+r-K*r, y1), (x0, y1-r+K*r), (x0, y1-r),
        (x0, y0+r),
        (x0, y0+r-K*r), (x0+r-K*r, y0), (x0+r, y0),
        (x0+r, y0),
    ]
    codes = [
        Path.MOVETO,  Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def make_fig(boxes, outfile, hgap, fs_text, loop_from, loop_to, fs_num=7.0):
    """
    loop_from / loop_to: box indices (0-based) for the dashed feedback arc.
    The arc goes from the right edge of loop_from back to the left edge of loop_to,
    drawn above the boxes.
    """
    n  = len(boxes)
    BW = (FW - 2*MH - (n-1)*hgap) / n

    fig = plt.figure(figsize=(FW, FH))
    ax  = fig.add_subplot(111)
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    cy = MV + BH / 2

    def cx_of(i):
        return MH + BW/2 + i*(BW + hgap)

    def draw_box(i, text, pi):
        cx = cx_of(i)
        bc, fc = PAL[pi]
        p = rounded_rect_path(cx, cy, BW, BH, CR)
        ax.add_patch(PathPatch(p, facecolor=fc, edgecolor=bc,
                               linewidth=0.9, zorder=2))
        ax.text(cx - BW/2 + 0.06, cy + BH/2 - 0.06,
                ROMAN[i], ha='left', va='top',
                fontsize=fs_num, fontweight='bold', color=bc, zorder=3)
        ax.text(cx, cy - 0.05, text,
                ha='center', va='center',
                fontsize=fs_text, multialignment='center',
                color='#1A1A1A', zorder=3)
        return cx

    def draw_arrow(x0, x1):
        ax.annotate('',
                    xy=(x1 - 0.02, cy), xytext=(x0 + 0.02, cy),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                   lw=0.9, mutation_scale=10),
                    zorder=1)

    centres = [draw_box(i, txt, pi) for i, (txt, pi) in enumerate(boxes)]
    for i in range(n - 1):
        draw_arrow(centres[i] + BW/2, centres[i+1] - BW/2)

    # ── dashed loop arc above boxes ──────────────────────────────────────────
    x_start = centres[loop_from] + BW/2    # right edge of loop_from box
    x_end   = centres[loop_to]   - BW/2    # left  edge of loop_to   box
    y_top   = cy + BH/2 + 0.18             # arc peak height above boxes

    # cubic bezier: start straight up, arc over, come straight back down
    ctrl_h = 0.14   # control point vertical offset from arc peak
    path_verts = [
        (x_start, cy + BH/2),
        (x_start, y_top + ctrl_h),
        (x_end,   y_top + ctrl_h),
        (x_end,   cy + BH/2),
    ]
    path_codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    arc_path = Path(path_verts, path_codes)
    ax.add_patch(PathPatch(arc_path, facecolor='none', edgecolor='#555555',
                           linewidth=0.9, linestyle='dashed', zorder=1))
    # arrowhead at the end of the arc (pointing downward into loop_to box)
    ax.annotate('',
                xy=(x_end, cy + BH/2 + 0.02),
                xytext=(x_end, cy + BH/2 + 0.10),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=0.9, mutation_scale=10),
                zorder=1)

    fig.savefig(outfile, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'OK  {outfile}  BW={BW:.3f}"  gap={hgap}"')


# ── PSO  (5 boxes, loop IV→II) ───────────────────────────────────────────────
make_fig([
    ('Inicializacija\nroja',                    0),
    ('Posodobi\ndelec $v,x$',                   1),
    ('Popravi +\noceni $f(x)$',                 3),
    ('Posodobi\n$p_i, g$',                      4),
    (r'Vrni $\vec{w}^*$',                       5),
], 'optimizer_pso.pdf', hgap=0.50, fs_text=6.0,
   loop_from=3, loop_to=1)

# ── SA  (5 boxes, loop IV→II) ────────────────────────────────────────────────
make_fig([
    ('Inicializacija\n+ $T_0$',                 0),
    ('Ustvari\nsoseda $x\'$',                   1),
    ('Metropolis\nsprejemanje',                  3),
    ('Ohlajanje\n$T \\leftarrow \\alpha T$',    4),
    (r'Vrni $\vec{w}^*$',                       5),
], 'optimizer_sa.pdf', hgap=0.50, fs_text=6.0,
   loop_from=3, loop_to=1)
