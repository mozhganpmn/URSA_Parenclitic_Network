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
order; the permutation null (`N = 500`) and the clip-robustness sweep dominate
the runtime.

## Outputs

| File | Contents |
|---|---|
| `results/paper_numbers.json` | every quantity reported in the paper, to six decimals |
| `results/indicator_duration.csv` | indicator-level anomaly durations (Table V) |
| `results/clip_check_output.json` | training-correlation clip sensitivity (Supp. S1) |
| `results/*.png` | manuscript figures |

## Implementation notes

**The `20σ` cap is an aggregation-layer device.** Indicator and domain anomaly
scores, TAV, and the edge-weight plots are means of `w`, so a single extreme
pair would dominate them; the cap prevents that. The cap must not be applied
before the activation comparison, because the corrected threshold at `n = 3`
(32.619) exceeds it. Activation and the permutation null therefore use the
uncapped weight, which is also the quantity whose square follows the `F(2, n-2)`
law that the synthetic calibration validates. `CAP_APPLIES_TO_ACTIVATION` in the
configuration cell controls this; the notebook prints how many activations a
pre-threshold cap would suppress.

**Two lag-1 autocorrelations appear in the paper and are not interchangeable.**
The mean lag-1 autocorrelation of the *raw sub-indicator series* (≈ 0.59)
motivates the circular-block null. The mean lag-1 autocorrelation of the
*anomaly series* (≈ 0.35) is a separate quantity. Both are reported in
`paper_numbers.json` under distinct keys.

**`n_eff` can exceed `n`.** The Bartlett correction
`n_eff = n(1 - r1*r2)/(1 + r1*r2)` raises the effective sample size above the
nominal count when the two domain series have lag-1 autocorrelations of opposite
sign.

**`EWS_CSD` is undefined before 2018** under expanding-window normalisation and
is not plotted for 2015–2017. Only `AHI` and `EWS_CSD` are min-max normalised in
the EWS panel; lag-1 autocorrelation and correlation length are shown in native
units.

**Table V ties.** Eleven indicator pairs reach five consecutive anomalous years
and are not separable from one another. The published table is cut at duration
≥ 6; the full ranking, with the aggregation rule and tie-break stated
explicitly, is written to `indicator_duration.csv`.

## Citation

Will be updated on acceptance.

## License

MIT for the code. The Vancouver indicator dataset is available from the
corresponding author on reasonable request.
