---
name: thesis-paper-latex
description: Thesis LaTeX lives in thesis-paper/main.tex; synced to current methodology 2026-07-21
metadata:
  node_type: memory
  type: project
---

The written thesis is `thesis-paper/main.tex` (FRI EMAI `friteza` class, Slovene main language, doc. dr. Luka Fürst supervisor). Bib in `thesis-paper/bib/bibliography.bib`. Build with `latexmk -pdf main.tex` from `thesis-paper/` (TeXLive present at /opt/homebrew/bin; compiles clean, ~9 pages). Figures in `thesis-paper/fig/` — `walkforward_divuniverse.png` + `walkforward_gfc.png` are copies of `test_results/combined_walkforward_*` (regenerate from there if plots change).

**Brought fully up to date 2026-07-21** (it had been describing the OLD codebase). Now reflects: single MASTER-lite cross-sectional transformer emitting μ (rank-IC/ListNet) + shared-rep factor covariance Σ=BBᵀ+diag(ψ); Hedge forecast-combination ensemble; the 3 native-Σ scenarios + strong baselines (Ledoit-Wolf, Black-Litterman, LSTM, LightGBM); 6-regime walk-forward with pooled-across-regime significance (n≈254) — IC, return-diff, Sharpe-difference (Ledoit-Wolf/Memmel), Deflated Sharpe, Benjamini-Hochberg FDR (see [[pooled-significance-across-regimes]], [[proposal-baselines-bl-lstm]]). Removed the old univariate/multivariate/TimesFM + Cholesky-cov narrative and the single 2017–2018 result tables. Honest framing preserved: transformer significantly beats ML/BL baselines + only survivor of DSR; edge over plain history is marginal (p≈0.06). Added ~16 new bib entries (li2024master, yoo2021dtml, cao2007listnet, grinold2000, freund1997, fan2008, hochreiter1997lstm, ke2017lightgbm, black1992/helitterman1999, neweywest1987, ledoitwolf2008/memmel2003, baileylopez2014, benjamini1995, politis1994).
