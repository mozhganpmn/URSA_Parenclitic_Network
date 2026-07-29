# Calibrated parenclitic networks for short, non-stationary multivariate systems

Reference implementation for *Calibrated Parenclitic Networks for Short,
Non-Stationary Multivariate Systems: An Application to Urban Resilience*,
applied to the URSA Vancouver indicator system (186 sub-indicators, seven
domains, 2011–2024).

## Method

Each pairwise relationship between sub-indicators is encoded as a deviation from
a **leave-future-out (LFO) Mahalanobis baseline**: for target year `t`, the
bivariate reference is estimated from co-observed years strictly before `t`, so
the reference is never contaminated by the events being detected.

An edge is *activated* when its Mahalanobis distance exceeds an exact
small-sample threshold derived from the scaled `F(2, n-2)` law, rather than the
asymptotic `chi^2(2)` approximation. At the minimum training window `n = 3` the
corrected threshold is 32.619 against a naive 2.448; using the asymptotic value
produces a 55.6% false-positive rate under the null.

Network anomaly density is baseline-corrected against a **circular-block
permutation null** that preserves each series' within-series autocorrelation and
trend while breaking cross-indicator phase alignment.

The three components address distinct error sources: the LFO baseline removes
reference contamination, the `F` correction removes the bulk of the small-sample
inflation, and the permutation null absorbs the residual autocorrelation-driven
inflation.

## Layout

```
parenclitic_urban_resilience.ipynb   single notebook, runs top to bottom
data/
  Input.xlsx                         186 sub-indicators x 14 years
  v1_urban_indicators_master_with_explanation.csv
results/                             created on first run
requirements.txt
```

## Running

```bash
pip install -r requirements.txt
jupyter lab parenclitic_urban_resilience.ipynb
```

Set `DATA_DIR` and `RESULTS_DIR` in the configuration cell. Execute all cells in
order. The synthetic calibration, the permutation null (`N = 500`, with an
`N = 10^4` rerun for the single confirmed year), and the clip-robustness sweep
dominate the runtime; the final supplementary-experiments cell adds a further
high-`N` permutation pass and can be skipped if
`results/supplementary_experiments.json` is not needed.

## Outputs

| File | Contents |
|---|---|
| `results/paper_numbers.json` | every quantity reported in the paper, to six decimals |
| `results/indicator_duration.csv` | indicator-level anomaly durations (Table 5) |
| `results/clip_check_output.json` | training-correlation clip sensitivity (Appendix A) |
| `results/supplementary_experiments.json` | supplementary robustness experiments (high-N permutation, DCI BH family sizes, 2023 drivers) |
| `results/*.png` | manuscript figures |


## Citation

A citation will be available when the paper is accepted.

## License

MIT for the code. The Vancouver indicator dataset in `data/` is released
alongside the code for research and replication use.
