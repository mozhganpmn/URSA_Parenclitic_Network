# Calibrated parenclitic networks for short multivariate records

Code and data for *Detecting joint departures in short multivariate records with calibrated parenclitic networks*
(M. PourmoradNasseri, A. Gholidoust and A. Albadvi).

The method builds a parenclitic network for each target year from bivariate references fitted to earlier years only.
An edge is activated when a pair of series leaves its reference by more than an exact finite-sample threshold, and the
share of activated pairs (NAD) is tested against refitted increment-shift surrogates. The two-channel test (NET2)
separates changes in level from changes in the relationship between series. The application uses the URSA urban
resilience indicators for Vancouver and Calgary, 2011 to 2024.

## Contents

```
data/                         Vancouver and Calgary inputs (207 sub-indicators, 2011-2024), hierarchy files
parenclitic_core.py           engine: leave-future-out references, thresholds, NAD, surrogates, node and domain scores
variants.py                   mean and linear-trend references, level (CON) and relationship (REL) channels
parenclitic_urban_resilience.ipynb   Vancouver analysis, robustness checks, Table 1 and Fig. 4
experiments/                  synthetic experiments, benchmark (including NET2), Calgary analysis, figures
experiments/PLAN_improvements.md     protocol of the benchmark, written before the confirmatory runs
experiments/plan.sha256       SHA-256 of the protocol (also stored in every benchmark output)
results/                      stored outputs of the notebook and scripts, figures in results/figures/
```

## Setup

Python 3.13 and the packages in `requirements.txt` (the versions used for the paper).

```
pip install -r requirements.txt
```

Run every script from the repository root.

## Data

`data/Input.xlsx` (Vancouver) and `data/Input_Calgary.xlsx` (Calgary) have the same 207 rows in the same order.
`Sheet1` holds the values for 2024 to 2011. The `Harmonization` sheet gives the definition, geography, source and
status of every row, and the `Read me` sheet explains the layout and the status labels. `data/names.csv` and
`data/adjacency.csv` hold the indicator abbreviations and the indicator hierarchy. All series come from the public
sources named in the `Source` column of the `Harmonization` sheet.

## Reproducing the paper

| Paper item | Code | Stored output |
|---|---|---|
| Table 1, held-out false-positive rates | notebook sections 1, 5 and 10 | `results/synthetic_validation_output.json`, `results/method_comparison_output.json` |
| Fig. 1 | `experiments/make_figure_1.py` | `results/figures/Fig1_framework.*` |
| Fig. 2, Fig. 3, Supplementary Fig. S1 | `experiments/make_figures_2_3.py` | `results/figures/Fig2_*`, `Fig3_*`, `FigS1_*` |
| Pair-level power (Fig. 2c, Supplementary Table S5) | `experiments/power_pair.py` | `experiments/power_pair.json` |
| Calibration of the network test (Fig. 3, Supplementary Tables S6 and S7) | `experiments/null_calibration.py`, `null_calibration_levels.py`, `null_z_quantiles.py`, `power_network.py` | `experiments/null_calibration*.json`, `null_z_quantiles.json`, `power_network.json` |
| Alternative thresholds and non-normal marginals (Supplementary Table S4) | notebook section 10 | `results/method_comparison_output.json`, `results/nonnormal_robustness_output.json` |
| Table 2, Supplementary Tables S11 to S18 | `experiments/improvement_study.py`, `improvement_summary.py`, `make_improvement_tables.py` | `experiments/improvement_*.json` |
| Fig. 4, Vancouver Monte Carlo tests | notebook sections 8 and 12 | `results/monte_carlo_10k.json`, `results/figures/Fig4_*` |
| Table 3, Supplementary Table S8 | `experiments/city_replication.py` | `experiments/city_replication.json` |
| Figs 5 and 6, Supplementary Table S10 | `experiments/domain_matrix.py`, `experiments/city_figures.py` | `results/domain_matrix.json`, `results/city_figure_numbers.json`, `results/figures/Fig5_*`, `Fig6_*` |
| Supplementary Table S9, minimum training window and geography | `experiments/robustness_checks.py` | `results/robustness_checks.json` |
| Clip, CV cutoff and support-weight checks (Supplementary Note on robustness) | notebook sections 9 and 11 | `results/clip_check_output.json`, `results/sensitivity_cv_lambda.json` |
| Edges of 2023 that were inactive in 2022 | `experiments/new_edges_2023.py` | printed |

Commands used for the stored outputs:

```
python experiments/null_calibration.py 200 199
python experiments/null_calibration_levels.py
python experiments/null_z_quantiles.py
python experiments/power_network.py 150 199
python experiments/power_pair.py
python experiments/city_replication.py 10000
python experiments/domain_matrix.py 10000
python experiments/robustness_checks.py . 2000 results/robustness_checks.json
python experiments/improvement_study.py gate 200 199
python experiments/improvement_study.py power 5000 10000
python experiments/improvement_study.py real 10000
python experiments/improvement_summary.py
python experiments/make_improvement_tables.py experiments tables_si.tex tables_main.tex
python experiments/city_figures.py
python experiments/new_edges_2023.py .
```

All random numbers come from fixed seeds. The benchmark `power` run saves the raw statistics of every scenario to
`experiments/improvement_raw/` (about 38 MB), which `improvement_summary.py` reads. They are not tracked in this
repository and can be regenerated with the `power` command above.

## Checks

```
python experiments/test_engine.py
python experiments/test_variants.py
```

`test_engine.py` compares the vectorised engine with a direct per-pair implementation. `test_variants.py` checks that
the mean-reference statistics reproduce the engine and the monitors exactly and that the thresholds hold the 5% rate
under independent bivariate normal sampling.

## Licence

The code is released under the MIT licence (`LICENSE`).
