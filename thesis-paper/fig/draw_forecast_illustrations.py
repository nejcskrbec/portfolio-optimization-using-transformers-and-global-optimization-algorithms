"""
Ponazoritve za razdelek 2 (napovedovanje pričakovanih donosov), v slogu
optimizer_*_illus. Trije predlogi, ki še niso vključeni v main.tex:

  forecast_models_illus.pdf  - kako PatchTST, TFT in MASTER berejo podatke
  forecast_to_mu_illus.pdf   - od izhoda modela do vhoda optimizatorja
  forecast_model_flows.pdf   - posodobljen diagram poteka treh modelov

Zaženi:  python draw_forecast_illustrations.py   (v mapi thesis-paper/fig)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5,
                     'mathtext.fontset': 'cm'})

FW = 5.90
BLUE, RED, ORANGE, PURPLE, GRAY, GREEN = '#1A63C6', '#C00000', '#D2691E', '#6A1B9A', '#6B6B6B', '#375623'
LIGHT = '#E8F0FA'


def panel(ax, title):
    ax.set_title(title, fontsize=8.5, pad=4)
    for s in ax.spines.values():
        s.set_color('#999999'); s.set_linewidth(0.6)
    ax.set_xticks([]); ax.set_yticks([])


def arrow(ax, p0, p1, col=GRAY, lw=0.9, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=6,
                                 color=col, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', zorder=5))


def box(ax, x, y, w, h, text, fc, ec='#555555', fs=6.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.02',
                                fc=fc, ec=ec, lw=0.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=4, linespacing=1.15)


rng = np.random.default_rng(7)

# =============================================================================
# 1) Kako modeli berejo podatke
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(FW, FW * 0.40), gridspec_kw={'wspace': 0.10})
fig.patch.set_facecolor('white')

# --- PatchTST: ena časovna vrsta, razrezana na prekrivajoče se odseke --------
ax = axes[0]; panel(ax, 'PatchTST')
L, H, B, Q = 32, 8, 8, 4
past = np.cumsum(rng.normal(0, 0.5, L)) * 0.25
x_past = np.arange(L)
ax.plot(x_past, past, color=BLUE, lw=0.9)
fut = past[-1] + np.cumsum(rng.normal(0.03, 0.35, H)) * 0.25
ax.plot(np.arange(L - 1, L + H), np.r_[past[-1], fut], color=ORANGE, lw=0.9, ls='--')
ax.axvspan(L - 0.5, L + H - 0.5, color='#FBEFE3', zorder=0)
ax.axvline(L - 0.5, color=GRAY, lw=0.6, ls=':')
ylo, yhi = past.min() - 0.9, past.max() + 1.6
ax.set_ylim(ylo, yhi); ax.set_xlim(-1, L + H)
# odseki (žetoni): pravokotniki nad vrsto, izmenično v dveh vrsticah
for k, s0 in enumerate(range(0, L - B + 1, Q)):
    yb = past.max() + 0.35 + (k % 2) * 0.45
    ax.add_patch(Rectangle((s0 - 0.4, yb), B - 0.2, 0.32, fc=LIGHT, ec=BLUE, lw=0.5, zorder=3))
# oznaki B in Q na prvih dveh odsekih
y0 = past.max() + 0.35
ax.annotate('', xy=(-0.4, y0 - 0.12), xytext=(B - 0.6, y0 - 0.12),
            arrowprops=dict(arrowstyle='<->', lw=0.5, color=GRAY))
ax.text(B / 2 - 0.5, y0 - 0.25, '$B$', ha='center', va='top', fontsize=6.3, color=GRAY)
yq = y0 + 0.45 + 0.32 + 0.12
ax.annotate('', xy=(-0.4, yq), xytext=(Q - 0.4, yq),
            arrowprops=dict(arrowstyle='<->', lw=0.5, color=GRAY))
ax.text(Q / 2 - 0.4, yq + 0.06, '$Q$', ha='center', va='bottom', fontsize=6.3, color=GRAY)
ax.text(L / 2, ylo + 0.15, 'vhodno okno ($L$ dni)', ha='center', va='bottom', fontsize=6.0, color=BLUE)
ax.text(L + H / 2 - 0.5, ylo + 0.15, '$H$ dni', ha='center', va='bottom', fontsize=6.0, color=ORANGE)
ax.text(L - 0.5, yhi - 0.05, '$t$', ha='center', va='top', fontsize=6.3, color=GRAY)

# --- TFT: napoved kvantilov za vsak dan napovednega obdobja --------------------
ax = axes[1]; panel(ax, 'TFT')
L2 = 24
past2 = np.cumsum(rng.normal(0, 0.5, L2)) * 0.22
ax.plot(np.arange(L2), past2, color=BLUE, lw=0.9)
h = np.arange(0, H + 1)
x_f = L2 - 1 + h
spread = 0.16 * np.sqrt(h)
c = past2[-1] + 0.02 * h
# levo nesimetrična porazdelitev: spodnji kvantili segajo dlje kot zgornji
bands = [(0.02, 0.98, 2.05, 1.45, '#DCE7F5'), (0.10, 0.90, 1.28, 0.95, '#BCD2EE'), (0.25, 0.75, 0.67, 0.55, '#93B6E3')]
for _, _, lo, hi, col in bands:
    ax.fill_between(x_f, c - lo * spread, c + hi * spread, color=col, lw=0, zorder=1)
ax.plot(x_f, c, color=GRAY, lw=0.8, ls='--', zorder=2)
mean = c - 0.18 * spread
ax.plot(x_f, mean, color=RED, lw=1.0, zorder=3)
ax.axvline(L2 - 0.5, color=GRAY, lw=0.6, ls=':')
ylo2, yhi2 = past2.min() - 0.7, past2.max() + 0.9
ax.set_ylim(ylo2, yhi2); ax.set_xlim(-1, L2 + H + 0.5)
ax.text(L2 - 0.5, yhi2 - 0.05, '$t$', ha='center', va='top', fontsize=6.3, color=GRAY)
ax.text(L2 + H - 0.2, c[-1] + 1.45 * spread[-1] + 0.05, '$0{,}98$', fontsize=5.5, color=GRAY, ha='right', va='bottom')
ax.text(L2 + H - 0.2, c[-1] - 2.05 * spread[-1] - 0.05, '$0{,}02$', fontsize=5.5, color=GRAY, ha='right', va='top')
ax.text(L2 / 2, ylo2 + 0.12, 'opazovane značilke', ha='center', va='bottom', fontsize=6.0, color=BLUE)
ax.plot([], [], color=GRAY, ls='--', lw=0.8, label='mediana')
ax.plot([], [], color=RED, lw=1.0, label='povprečje')
ax.legend(loc='upper left', fontsize=5.6, frameon=False, handlelength=1.6, borderaxespad=0.3)

# --- MASTER: mreža sredstva x dnevi, pozornost znotraj in med sredstvi ---------
ax = axes[2]; panel(ax, 'MASTER')
nA, nD = 5, 8
ax.set_xlim(-0.8, nD + 2.4); ax.set_ylim(-1.2, nA + 1.6)
for a in range(nA):
    for d in range(nD):
        ax.add_patch(Rectangle((d, a), 0.86, 0.78, fc=LIGHT, ec='#9DB7DE', lw=0.4, zorder=1))
# tržni vektor m_t: vrstica zgoraj, ki uteži značilke vseh sredstev
ax.add_patch(Rectangle((0, nA + 0.55), nD - 0.14, 0.45, fc='#FDEBD7', ec=ORANGE, lw=0.5, zorder=1))
ax.text(nD / 2 - 0.07, nA + 0.775, r'tržni vektor $\mathbf{m}_t$', ha='center', va='center', fontsize=5.8, color=ORANGE)
arrow(ax, (nD / 2, nA + 0.52), (nD / 2, nA - 0.12), col=ORANGE, lw=0.7)
# agregacija znotraj sredstva: loki po času v vrstici 2
ra = 2
for d in range(nD - 1):
    arrow(ax, (d + 0.43, ra + 0.82), (nD - 1 + 0.43, ra + 0.82), col=PURPLE, lw=0.45, rad=-0.35)
# agregacija med sredstvi: loki po sredstvih v zadnjem stolpcu
dc = nD - 1
for a in range(nA):
    if a != ra:
        arrow(ax, (dc + 0.9, a + 0.39), (dc + 0.9, ra + 0.39), col=RED, lw=0.45, rad=0.45)
ax.add_patch(Rectangle((-0.06, ra - 0.05), nD - 0.08, 0.88, fc='none', ec=PURPLE, lw=0.8, zorder=4))
ax.add_patch(Rectangle((dc - 0.05, -0.06), 0.96, nA - 0.1, fc='none', ec=RED, lw=0.8, zorder=4))
# izhod: en signal na sredstvo
sig = [0.55, -0.35, 0.95, 0.15, -0.70]
for a, v in enumerate(sig):
    x0 = nD + 1.25
    ax.add_patch(Rectangle((x0, a + 0.2), v * 0.85, 0.38, fc=GREEN if v > 0 else '#B85450', lw=0, zorder=2))
ax.plot([nD + 1.25] * 2, [0, nA], color=GRAY, lw=0.5)
ax.text(nD + 1.25, -0.25, 'signal', ha='center', va='top', fontsize=5.8, color=GREEN)
ax.text(nD / 2 - 0.07, -0.25, 'dnevi $t-7,\\dots,t$', ha='center', va='top', fontsize=5.8, color=BLUE)
ax.text(-0.35, nA / 2, 'sredstva', rotation=90, ha='center', va='center', fontsize=5.8, color=BLUE)
ax.text(-0.6, -0.75, 'znotraj sredstva', ha='left', va='top', fontsize=5.6, color=PURPLE)
ax.text(nD + 2.3, -0.75, 'med sredstvi', ha='right', va='top', fontsize=5.6, color=RED)

fig.savefig('forecast_models_illus.pdf', bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 2) Od izhoda modela do vhoda optimizatorja (številke iz tab:std-example)
# =============================================================================
fig, ax = plt.subplots(figsize=(FW, FW * 0.34))
ax.set_xlim(-0.01, 1.02); ax.set_ylim(0, 1); ax.axis('off')
rows = [('Zgodovinsko\npovprečje', 'povprečni dnevni\ndonos $\\times H$', 0.80),
        ('PatchTST, TFT', 'dnevni donosi $\\to$\nkumulativni donos (2)', 0.52),
        ('MASTER', 'signal relativnega\ndonosa', 0.24)]
for name, how, y in rows:
    box(ax, 0.00, y - 0.09, 0.19, 0.18, name, '#E8F0FA', bold=True, fs=6.0)
    box(ax, 0.23, y - 0.09, 0.27, 0.18, how, '#FFF6E5', fs=6.0)
    arrow(ax, (0.19, y), (0.23, y))
    arrow(ax, (0.50, y), (0.565, 0.52), rad=0.0)
box(ax, 0.565, 0.24, 0.20, 0.56, '', '#E5F3EA')
ax.text(0.665, 0.74, 'standardizacija\nmed sredstvi (9)', ha='center', va='center', fontsize=6.3, zorder=5)
vals_in, vals_out = [1.8, 0.6, 0.2, -0.6], [1.50, 0.12, -0.35, -1.27]
for k, (vi, vo) in enumerate(zip(vals_in, vals_out)):
    yy = 0.60 - k * 0.085
    ax.text(0.585, yy, f'{vi:+.1f} %'.replace('.', ','), fontsize=5.9, ha='left', va='center', zorder=5)
    ax.text(0.665, yy, r'$\to$', fontsize=6.3, ha='center', va='center', zorder=5)
    ax.text(0.745, yy, f'{vo:+.2f}'.replace('.', ','), fontsize=5.9, ha='right', va='center', zorder=5, color=GREEN)
arrow(ax, (0.765, 0.52), (0.815, 0.52))
box(ax, 0.815, 0.40, 0.185, 0.24, r'$\tilde{\mu}_t$ in $\tilde{\Sigma}_t$' + '\n' + 'optimizator\n(RD, SO)', '#F3E8F7', fs=6.1)
fig.savefig('forecast_to_mu_illus.pdf', bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 3) Posodobljen diagram poteka treh modelov
# =============================================================================
cols = {
    'PatchTST': ['dnevni logaritemski donosi\n(okno $L$, vsako sredstvo ločeno)',
                 'odseki dolžine $B$ s korakom $Q$\n+ transformerski kodirnik',
                 '$H$ dnevnih logaritemskih\ndonosov',
                 'kumulativni donos (2)'],
    'TFT': ['oznaka sredstva; donosi,\n20-dnevni donos, volatilnost',
            'izbira spremenljivk + LSTM\n+ časovna samopozornost',
            '$H$ dnevnih donosov\n(pričakovana vrednost, RMSE)',
            'kumulativni donos (2)'],
    'MASTER': ['158 značilk Alpha158\n+ 63 tržnih značilk',
               'tržno vodeno uteževanje\n$\\to$ znotraj in med sredstvi\n$\\to$ časovna agregacija',
               'signal relativnega\n21-dnevnega donosa',
               'brez pretvorbe\n(signal že relativen)'],
}
fills = ['#E8F0FA', '#FFF6E5', '#E5F3EA', '#FBE9E9']
fig, ax = plt.subplots(figsize=(FW, FW * 0.62))
ax.set_xlim(0, 3); ax.set_ylim(-1.35, 4.35); ax.axis('off')
for ci, (name, cells) in enumerate(cols.items()):
    x = ci + 0.06
    ax.text(ci + 0.5, 4.2, name, ha='center', va='center', fontsize=8.5, fontweight='bold')
    for ri, (txt, fc) in enumerate(zip(cells, fills)):
        y = 3.3 - ri * 0.95
        box(ax, x, y, 0.88, 0.68, txt, fc, fs=6.2)
        if ri < 3:
            arrow(ax, (ci + 0.5, y), (ci + 0.5, y - 0.27))
    arrow(ax, (ci + 0.5, 3.3 - 3 * 0.95), (ci + 0.5, -0.33))
box(ax, 0.06, -0.92, 2.88, 0.58, 'standardizacija med sredstvi (9) $\\to$ $\\hat{\\mu}$ za optimizator', '#EAEAEA', fs=6.6)
for k, lab in enumerate(['vhod', 'arhitektura', 'izhod modela', 'pretvorba v $\\hat{\\mu}$']):
    ax.add_patch(Rectangle((0.06 + k * 0.74, -1.25), 0.08, 0.08, fc=fills[k], ec='#555555', lw=0.5))
    ax.text(0.17 + k * 0.74, -1.21, lab, fontsize=5.8, va='center')
fig.savefig('forecast_model_flows.pdf', bbox_inches='tight')
plt.close(fig)
print('ok')
