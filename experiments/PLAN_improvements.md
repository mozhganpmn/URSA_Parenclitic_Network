# Pre-specified plan for the method-improvement study

Written 2026-10-17T20:34Z, before any new variant was implemented or run. Sizes in section 6 are fixed after a
timing run on development seeds and before the first confirmatory run; the final file is hashed (SHA-256) and the
hash is stored in every confirmatory output file.

## 1. Question

The first benchmark (benchmark_monitors.py) showed that the network anomaly density (NAD) is not consistently more
powerful than three standard monitors. This study asks whether three changes to the method itself, specified below,
improve power without losing calibration, and how the improved method compares with equally improved baselines.
All outcomes are reported, including losses.

## 2. Variants (hyperparameters fixed here, not tuned)

Reference (how the held-out year is predicted from earlier co-observed years):
* M, mean reference. The current leave-future-out (LFO) reference.
* T, trend reference. Each series of a pair is regressed on calendar year over the earlier co-observed years. The
  held-out residual vector is scored with the residual covariance (n - 2 degrees of freedom) and the prediction
  leverage h = 1/n + (t0 - mean t)^2 / Sxx. The Mahalanobis distance is divided by (1 + h), and the activation
  threshold is the 95% point of 2(n-2)/(n-3) F(2, n-3), multiplied by (1 + h). Requires n >= 4.

Aggregate over eligible pairs of a year:
* count. Share of pairs above their threshold (the current NAD).
* cont. Mean of u = -ln(max(p, 1e-6)), where p is the upper tail of the pair F law (F(2, n-2) for M, F(2, n-3) for T).
* rel. Relationship channel. In each pair the Mahalanobis distance is split along the two principal axes of the
  training correlation: the major-axis part (movement along the usual co-movement) and the minor-axis part
  (departure from the usual co-movement). The channel is the mean of -ln(max(p, 1e-6)) of the minor-axis part, with p
  from F(1, df) (df = n - 1 for M, n - 2 for T), over pairs with training |r| >= 0.7. At least 5 such pairs are
  required, otherwise the channel is undefined for that year.

Named variants: NAD-M (current), NAD-T, CON-M, CON-T, REL-M, REL-T.

Omnibus: OMNI = min(P_X, P_SPE), where X is the NAD-family variant adopted by section 5 and SPE is the PCA residual
with the matching reference. Calibrated by the same surrogates (Westfall-Young min-p; each surrogate's own rank-based
P values give the null distribution of the minimum). For the matched-power study the same rule is applied as a union
rejection region with equal per-test level a, where a is solved so that the null datasets of the same process give a
5% union false-positive rate.

Baselines get the same two references so that the comparison is not tilted by detrending:
UED-M, dT2-M, SPE-M (as in the first benchmark) and UED-T, dT2-T, SPE-T (series regressed on year, prediction
leverage in the scaling, n >= 4, PCA on training residuals with at most n - 3 components, 80% variance). The
comparator of a variant in any scenario family is the better of M and T for that baseline in that family.

Not in this study: domain-level scans. They need structured modules and are left for the real-data analysis.

## 3. Scenario families

A (existing): level shift, 10% or 30% of series, delta 2, 3, 4, stationary and trending base. 12 scenarios.
B (existing): relationship break in the three-block factor model, loading 0.9 and 0.7, share 30%, 60%, 100%,
  both bases. 12 scenarios.
C (existing): opposing shift, delta 1.0, 1.5, 2.0, both bases. 6 scenarios.
D (new): missingness. Level shift, 30% of series, delta 2.5, missing share 10%, 30%, 50%, both bases (6 scenarios).
  Opposing shift delta 1.5, missing share 10%, 30%, 50%, stationary base (3 scenarios). 9 scenarios.
E (new): sparse pair-specific breaks without block factor structure. Fifteen disjoint pairs of series with within-pair
  correlation 0.9 (AR(1) shared component), no other correlation. In the event year, one member of a share
  (one third, two thirds, all) of the pairs takes an independent draw of the shared component. Both bases.
  6 scenarios. This family favours pairwise methods by design and is labelled so in every report.
F (new): non-linear bases. Level shift, 30% of series, delta 2.5 and 3, on saturating-trend and random-walk bases.
  4 scenarios.
Total 49 scenarios. Event in year index 10 of 14, 30 series, AR(1) with phi 0.5, as in the first benchmark.
The generators of families A to C are those of benchmark_monitors.py without change.

Families D to F were chosen after seeing the first benchmark, to test the weakness it exposed (sparse and missing
data) and the scenario type where pairwise methods are expected to help. They are not independent of that result.

## 4. Calibration gate

Type-I error of the surrogate test (refitted increment-shift surrogates) for every statistic, in the seven regimes of
null_calibration (10% missing) plus iid and independent trends with 50% missing. 200 datasets, 199 surrogates,
tested year indices 5 to 13. A variant is admitted only if its rate is at most 0.07 in each regime except the
stationary common factor (known to be liberal for the whole family and reported separately). A variant that fails the
gate is reported and excluded from the adoption rule.

## 5. Adoption and reporting rules

1. Primary endpoint: matched-false-positive-rate power at 5%, averaged within a family and then across families A to F
   with equal weight per family.
2. Adopted NAD-family variant = the admitted variant with the highest primary endpoint. A tie within 0.01 goes to the
   simpler variant (order NAD-M, CON-M, NAD-T, CON-T, REL-M, REL-T).
3. Wording rule for comparisons with a baseline in a family: "more powerful" only if the family mean exceeds the
   comparator by at least 0.05 and the variant is ahead in at least two thirds of the family scenarios; "less
   powerful" under the mirror condition; "comparable" otherwise.
4. Real data (Vancouver, Calgary) are analysed once, last, with: the adopted variant, the matching SPE, the
   current NAD-M for continuity, and OMNI. 10^4 surrogates, Holm across the ten reporting years. No change to any
   rule after seeing them. If the adopted variant gives different findings from Table 2, both are reported and the
   manuscript text follows the adopted variant.
5. All scenarios and all variants appear in the supplement, losses included.

## 6. Seeds and sizes

Development and debugging use seeds below 1000 and never feed a reported number. Confirmatory seeds: data for the
calibration gate 20261101, surrogates 20261102; null datasets of power group g use 5000000 + g; event datasets of
scenario s use 6000000 + s. Sizes (to be fixed after timing, see top): event datasets D, null datasets Dn per
process group.

## 7. Addendum written 2026-10-17 before the first confirmatory run

Made after implementing the statistics and running small development checks (seeds below 1000, 300 event and 800
null datasets in seven scenarios, used only to find coding errors). Those checks showed that the relationship channel
and the level channel respond to different kinds of events, which motivated item 2. No hyperparameter of section 2 was
changed after the development runs.

1. Sizes: event datasets D = 5000 per scenario, null datasets Dn = 10000 per process group (16 groups), calibration
   gate 200 datasets and 199 surrogates per regime (nine regimes). Timing on the development seeds: 7 ms per dataset
   and 2.4 s per surrogate test.
2. Added omnibus statistics, calibrated like OMNI: NET2-M = min-p of (CON-M, REL-M) and NET2-T = min-p of (CON-T, REL-T).
   They use the network only, with a level channel and a relationship channel. OMNI(X) for the six single variants,
   each with the SPE of the same reference, stays as the companion with a standard monitor. The adoption rule of
   section 5 item 2 now ranks the admitted single variants and NET2-M, NET2-T together; OMNI(X) is reported but does
   not compete.
3. An undefined statistic (for example REL with fewer than 5 gated pairs, or a monitor without enough eligible series)
   counts as no evidence in the power study (ranked below every defined value). In the gate, a year in which the
   statistic is undefined is not tested, and the rate is taken over the tested years. For an omnibus statistic an
   undefined channel contributes P = 1, and the year is not tested only if both channels are undefined.
4. Matched power of an omnibus statistic: the statistic is minus the smaller of the two mid-p values of the channels
   against the process null, so that it is handled exactly like any other statistic.
5. The surrogate test uses the refitted increment-shift null (10^4 for real data); single-statistic P values use
   random tie-breaking in synthetic studies and conservative ties in real data.
