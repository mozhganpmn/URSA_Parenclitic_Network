# Temporal Deviation Networks for Urban Resilience
### Parenclitic Network Framework — URSA Vancouver

This project applies **parenclitic network analysis** to measure urban resilience in Vancouver over time. Rather than tracking individual indicators in isolation, it models the *relational structure* between sub-indicators — detecting when normal co-variation patterns break down as a signal of systemic stress.

---

## Overview

Each year, a network is constructed where:
- **Nodes** = urban sub-indicators (housing, income, environment, health, etc.)
- **Edges** = anomaly scores `w_ij(t)` measuring how much a pair of sub-indicators deviates from their expected relationship

A leave-future-out (LFO) baseline ensures every edge weight is a genuine out-of-sample anomaly — the reference is fitted only on years before `t`.

Aggregate metrics (NAD, CW-NAD, DCI, EWS) summarize network-wide stress levels year by year.

---

## Methodology

| Step | Description |
|---|---|
| LFO Reference Estimation | OLS + LOWESS + Newey-West HAC for autocorrelated pairs |
| Edge Construction | `w_ij(t) = \|x_j(t) - expected_ij(t)\| / s_ij(t)` |
| Hierarchical Aggregation | Sub-indicator → Indicator → Domain |
| Network Metrics | NAD, CW-NAD, APD, DCI (lags 0/1/2) |
| Early Warning Signals | Rising variance, lag-1 autocorrelation, correlation length |
| Node Metrics | Anomaly score, TAV, anomaly degree, betweenness |
| Edge Metrics | Consecutive duration, autocorrelation, recovery rate |

---

## Validation Scenarios

Three disruption scenarios anchor the analysis (2011–2024):

1. **Housing structural break (2015–2016)** — slow-building relational stress from foreign buyer activity, price-income decoupling, and the provincial foreign buyer tax
2. **COVID-19 (2020)** — acute cross-domain shock
3. **Post-COVID disruption persistence (2022–2024)** — relational breakdown persisting after apparent recovery

Heat Dome (2021) is retained as a secondary annotation.

---

## Repository Structure

```
├── URSA_Parenclitic_Network_7.ipynb   # Current version (v7)
├── URSA_Parenclitic_Network_6.ipynb   # Previous version (v6)
├── URSA_Parenclitic_Network.ipynb     # Original version
├── Input.xlsx                          # Main input data
├── adjacency.csv                       # Sub-indicator adjacency structure
├── names.csv                           # Sub-indicator names/labels
└── data/                               # Supporting data files
```

---

## Requirements

- Python 3.x
- `pandas`, `numpy`, `scipy`, `statsmodels`
- `networkx`
- `matplotlib`, `seaborn`
- `openpyxl` (for Excel input)

---

## Version History

| Version | Key Changes |
|---|---|
| v7 | LFO baseline, Newey-West HAC, three validation scenarios, EWS statistics |
| v6 | Previous baseline approach |
| v1 | Initial implementation |
