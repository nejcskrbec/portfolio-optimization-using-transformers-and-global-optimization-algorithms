# Pooled significance across regimes

Regimes pooled (6): GFC-2008, 2013-2017, COVID-2020, 2022-2025, divuniverse, headline-megacap

## #1 Signal level — pooled rank-IC (mean IC / t)

| Scenario | mean IC | Newey-West t | NW p | cluster-t | cluster-p | N |
|---|---|---|---|---|---|---|
| Transformer | +0.0268 | 1.73 | 0.0847 | 2.22 | 0.0767 | 254 |
| Ansambel | +0.0218 | 1.51 | 0.1323 | 1.76 | 0.1381 | 254 |
| LSTM | +0.0043 | 0.26 | 0.7965 | 0.38 | 0.7183 | 254 |
| SimpleML | -0.0037 | -0.39 | 0.6946 | -0.10 | 0.9230 | 254 |
| BlackLitterman | -0.0182 | -1.14 | 0.2557 | -1.12 | 0.3141 | 254 |
| Zgodovinski | -0.0186 | -1.26 | 0.2081 | -1.27 | 0.2584 | 254 |

## #2 Portfolio level — pooled per-window return diff (algo=pso)

| Comparison | Δ/window% | t-test p | Wilcoxon p | sign p | cluster-p | N |
|---|---|---|---|---|---|---|
| Transf − LSTM | +1.26 | 0.0010 | 0.0006 | 0.0069 | 0.0742 | 254 |
| Transf − BlackLitterman | +1.38 | 0.0231 | 0.0143 | 0.0279 | 0.0848 | 254 |
| Transf − Zgodovinski | +1.22 | 0.0307 | 0.0206 | 0.0900 | 0.1241 | 254 |
| Transf − SimpleML | +2.03 | 0.0000 | 0.0000 | 0.0020 | 0.0122 | 254 |

_Per-window pool treats non-overlapping monthly windows as ~independent (n≈254, well-powered); cluster-p treats each regime as one unit (6 clusters, most conservative). Report both._

## #3 Sharpe-difference test (pooled daily, algo=pso)

Ledoit-Wolf (2008) / Memmel (2003) HAC delta method + stationary bootstrap. ΔSR annualized.

| Comparison | SR Transf | SR base | ΔSR | z | HAC p | boot p | T |
|---|---|---|---|---|---|---|---|
| Transf − LSTM | 0.97 | 0.58 | +0.39 | 2.90 | 0.0038 | 0.0040 | 5385 |
| Transf − BlackLitterman | 0.97 | 0.58 | +0.39 | 1.97 | 0.0491 | 0.0560 | 5385 |
| Transf − Zgodovinski | 0.97 | 0.64 | +0.33 | 1.80 | 0.0716 | 0.0680 | 5385 |
| Transf − SimpleML | 0.97 | 0.35 | +0.62 | 4.25 | 0.0000 | 0.0000 | 5385 |

## #4 Deflated Sharpe Ratio (Bailey & López de Prado 2014)

Selection-adjusted over N=24 trials (scenario×algo). DSR>0.95 ⇒ Sharpe stays significantly >0 after correcting for the search. SR annualized.

| Strategy (pso) | SR (ann) | SR0 (ann) | DSR=P(SR>0) | T |
|---|---|---|---|---|
| Transformer | 0.97 | 0.43 | 0.994 | 5385 |
| Ansambel | 1.01 | 0.43 | 0.996 | 5385 |
| Zgodovinski | 0.64 | 0.43 | 0.834 | 5385 |
| LSTM | 0.58 | 0.43 | 0.762 | 5385 |
| BlackLitterman | 0.58 | 0.43 | 0.762 | 5385 |
| SimpleML | 0.35 | 0.43 | 0.360 | 5385 |

## #5 Benjamini-Hochberg FDR (across all #2 pairwise tests)

12/12 of the 4-algo × 4-baseline return tests survive FDR<0.05.

| Test | raw p | BH q | FDR<0.05 |
|---|---|---|---|
| pso: Transf−SimpleML | 0.0000 | 0.0000 | ✓ |
| sa: Transf−SimpleML | 0.0000 | 0.0000 | ✓ |
| ga: Transf−SimpleML | 0.0000 | 0.0000 | ✓ |
| pso: Transf−LSTM | 0.0010 | 0.0021 | ✓ |
| sa: Transf−LSTM | 0.0009 | 0.0021 | ✓ |
| ga: Transf−LSTM | 0.0009 | 0.0021 | ✓ |
| pso: Transf−BlackLitterman | 0.0231 | 0.0323 | ✓ |
| sa: Transf−BlackLitterman | 0.0242 | 0.0323 | ✓ |
| ga: Transf−BlackLitterman | 0.0241 | 0.0323 | ✓ |
| pso: Transf−Zgodovinski | 0.0307 | 0.0369 | ✓ |
| sa: Transf−Zgodovinski | 0.0469 | 0.0494 | ✓ |
| ga: Transf−Zgodovinski | 0.0494 | 0.0494 | ✓ |
