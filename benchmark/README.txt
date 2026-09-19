COMPACT BENCHMARK LAYOUT
========================

Expected repository structure:

benchmark/
  core.py
  run.py
  configs/
    config_*.json
    ticker_alias_registry.json
  orlib_data/
    port1.txt ... port5.txt
    portef1.txt ... portef5.txt

estimators/
  master_us.py       # existing project file
  tft_mu.py          # existing project file
  patchtst_mu.py     # supplied by this patch

portfolio_optimizers/
  config.json
  ... C++ optimizer ...

data/
  ... benchmark datasets ...

Only MASTER, TFT and PatchTST are learned forecasting models. All three are
Transformer architectures. Historical is the non-learned baseline.

BUILD OPTIMIZER
---------------
cd portfolio_optimizers
make
cd ..

PRIMARY BENCHMARKS
------------------
python -u benchmark/run.py predictive
python -u benchmark/run.py orlib
python -u benchmark/run.py orlib --orlib-full
python -u benchmark/run.py leow
python -u benchmark/run.py aprea
python -u benchmark/run.py wang
python -u benchmark/run.py practical --cost-bps 10

GROUPS
------
# The three external literature comparisons only: Leow + Aprea + Wang
python -u benchmark/run.py literature

# Thesis evaluation: predictive + OR-Library + literature + practical
python -u benchmark/run.py all

FAST CHECKS
-----------
python -u benchmark/run.py leow --smoke
python -u benchmark/run.py aprea --smoke
python -u benchmark/run.py wang --smoke
python -u benchmark/run.py practical --smoke
python -u benchmark/run.py all --dry-run

UTILITY COMMANDS
----------------
python -u benchmark/run.py wang-risk-sweep
python -u benchmark/run.py wang-plot
python -u benchmark/run.py equity-curves
python -u benchmark/run.py timing

NOTES
-----
- predictive is the IC/RankIC/long-short protocol used in the current thesis.
- The older predictive_eval.py experiment was intentionally removed; it also
  contained an LSTM option and is not the standard predictive table currently
  used in the thesis.
- frontier_plots.py and verify_literature_setup.py were removed as standalone
  utilities because they are not required for the current thesis figures.
- Ticker alias registry now lives under benchmark/configs/.
