---
name: pooled-significance-across-regimes
description: Cross-regime pooled significance (signal IC + portfolio returns) — the strongest statistical evidence the transformer beats baselines
metadata:
  node_type: memory
  type: project
  originSessionId: 7d1407b9-6189-45d1-86b7-0594e44d6916
---

`benchmark/significance_pooled.py` — standalone cross-regime meta-analysis over the 6 pickled regime results (`REGIME_MAP`: GFC-2008, 2013-2017, COVID-2020, 2022-2025, divuniverse, headline-megacap). Answers "how to better prove significance" (the per-regime n≈46 windows never reach p<0.05; pooling across regimes gives n=254 and power). Run: `python benchmark/significance_pooled.py`; it prints two tables AND exports `test_results/significance_pooled_{signal.csv,returns.csv}` + `significance_pooled.md` (the `_export()` call regenerates them each run — this is the "auto-regenerate" wiring the user asked for). Requires all 6 `results_<tag>.pkl` present. Ties into [[proposal-baselines-bl-lstm]] and [[strong-baselines-and-covid-config]].

**Two tests implemented** (the user's ranked #1 + #2):
- **#1 signal level** — pooled rank-IC per scenario with Newey-West HAC t (`_nw_tstat`, Bartlett, auto lag) + regime-clustered t (`_cluster_tstat`, 6 clusters, G−1 df); plus paired Transformer−baseline IC diffs.
- **#2 portfolio level** — pooled per-window cumulative-return differences, reported with t-test / Newey-West / Wilcoxon / sign / cluster p, for all 4 metaheuristics.

**Headline results (2026-07-21, n=254 pooled windows):**

#1 — **Transformer is the ONLY μ source with positive pooled OOS IC (+0.0198)**; cluster-t=2.49, cluster-p=0.055. Everything else is ≤0: Ansambel +0.010, LSTM −0.003, SimpleML −0.001, BlackLitterman −0.019, Zgodovinski −0.021. Paired IC diffs all positive but per-window NW p≈0.12–0.19 (IC is noisy window-to-window; the cluster/level test is the cleaner signal-side evidence).

#2 — **portfolio-level pooled per-window return diffs (PBILDE), the strongest result:**
| Comparison | Δ/window | t p | Wilcoxon | cluster p | |
|---|---|---|---|---|---|
| Transf − SimpleML | +1.67% | <0.001 | 0.0002 | 0.019 | *** |
| Transf − LSTM | +1.17% | 0.0018 | 0.0008 | 0.058 | *** |
| Transf − BlackLitterman | +1.22% | 0.032 | 0.026 | 0.093 | ** |
| Transf − Zgodovinski | +1.01% | 0.056 | 0.051 | 0.097 | * |
Robust across ALL 4 metaheuristics (pso/sa/ga nearly identical; Transf−Zgod even reaches p≈0.03 under pso/ga).

**Correct framing for the thesis (honest, not overstated):** treating non-overlapping monthly windows as ~independent (n≈254) the transformer significantly beats LSTM and SimpleML (p<0.01) and BlackLitterman (p<0.05), and marginally beats the classical Zgodovinski (p≈0.06). The **regime-clustered** test (each of 6 regimes = 1 unit, most conservative) softens these to marginal (p≈0.02–0.10) — report BOTH so a reviewer sees the honest range. The single cleanest sentence: "the transformer is the only μ source with positive out-of-sample IC pooled across six regimes, and its realized-return edge over every simpler ML / classical baseline is significant at the per-window level and directionally consistent (all 6 regimes) at the regime level."

**Three MORE tests added 2026-07-21** (all on the same pickles, no re-run — the pickles store per-window daily-return arrays; concatenating windows across regimes = each strategy's full pooled daily stream, T≈5385 days). Exports `significance_pooled_{sharpe,dsr,fdr}.csv` + sections in `significance_pooled.md`:

- **#3 Sharpe-difference test** (`test3_sharpe_diff`, Ledoit-Wolf 2008 / Memmel 2003 HAC delta method + Politis-Romano stationary bootstrap). Transformer annualized Sharpe **0.96** on pooled daily returns. ΔSharpe vs: **SimpleML +0.50 (HAC p=0.001, boot 0.000)***, **LSTM +0.39 (p=0.007, boot 0.011)***, **BlackLitterman +0.38 (p=0.055)***, Zgodovinski +0.30 (p=0.105, n.s.). HAC and bootstrap p agree tightly. This tests the exact metric we headline (Sharpe) — upgrades every "1.26 vs 0.80" into a p-value on the ratio. **The classical Zgodovinski is the one baseline the transformer does NOT significantly out-Sharpe pooled** (consistent with #2's p≈0.06) — the edge is vs *ML/BL* baselines, softer vs plain history.
- **#4 Deflated Sharpe Ratio** (`test4_deflated_sharpe`, Bailey & López de Prado 2014 — selection-adjusted over N=32 scenario×algo trials, uses return skew/kurtosis). **Transformer DSR=0.996, Ansambel 0.997 — both clear the 0.95 bar** ⇒ Sharpe stays significantly >0 AFTER correcting for having searched ~32 strategies. **Every classical/simpler baseline FAILS the deflation**: Zgodovinski 0.888, BlackLitterman 0.809, LSTM 0.803, SimpleML 0.630. This directly answers the "you tried many configs and picked the winner" reviewer attack — only the transformer/ensemble survive it. SR0 (expected max Sharpe under null across 32 trials) = 0.39 annualized.
- **#5 Benjamini-Hochberg FDR** across all 16 #2 return tests (4 algo × 4 baseline): **13/16 survive FDR<0.05**; the only 3 that drop to FDR<0.10 are the marginal Transf−Zgodovinski rows (pbilde/sa/ga), consistent everywhere. So the multiplicity of tests is not driving the result — the ML/BL wins are robust to FDR correction.

Net upgrade to the verdict: the transformer's risk-adjusted edge is (a) significant on the Sharpe metric itself vs ML/BL, (b) survives selection-bias deflation where NO baseline does, (c) survives FDR across all 16 tests. The honest soft spot remains **transformer-vs-plain-history**, which is economic/marginal (p≈0.055–0.11) but not conventionally significant pooled.
