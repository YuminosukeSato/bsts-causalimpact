# R CausalImpact Compatibility Matrix

Comparison of features between R CausalImpact (bsts 1.4.1) and this Python implementation.

## State Components

| Feature | R | Python | Notes |
|---|---|---|---|
| Local level | Yes | Yes | Identical algorithm |
| Local linear trend | Yes | Yes | `state_model="local_linear_trend"` |
| Seasonality | Yes | Yes | State-space model matching R bsts `AddSeasonal()` |
| Dynamic regression | Yes | Yes | `dynamic_regression=True` |
| Regression (static) | Yes | Yes | Identical algorithm |

## MCMC Parameters

| Parameter | R | Python | Notes |
|---|---|---|---|
| niter | Yes | Yes | Same default (1000) |
| nseasons | Yes | Yes | `ModelOptions.nseasons` or `model_args["nseasons"]` |
| season.duration | Yes | Yes | `ModelOptions.season_duration` or `model_args["season_duration"]` (R compat: `"season.duration"`) |
| prior.level.sd | Yes | Yes | Same default (0.01) |
| standardize.data | Yes | Yes | Same default (True) |
| expected.model.size | Yes | Yes | Unified default `2` |
| state model selection | Via bsts state spec | Yes | `state_model="local_level"` or `"local_linear_trend"` |

## Warmup Semantics

| Aspect | R | Python | Match |
|---|---|---|---|
| Default warmup | niter / 2 | niter / 2 | Yes |
| Warmup discarded | First N samples | First N samples | Yes |
| Post-warmup used | niter - nwarmup | niter - nwarmup | Yes |

## Summary and Plot Parity

| Feature | R | Python | Match |
|---|---|---|---|
| Summary table | Yes | Yes | Same format |
| Narrative report | Yes | Yes | Same structure |
| Original + counterfactual plot | Yes | Yes | Yes |
| Pointwise effect plot | Yes | Yes | Yes |
| Cumulative effect plot | Yes | Yes | Yes |
| CI bands on plots | Yes | Yes | Yes |

## Data Handling

| Feature | R | Python | Notes |
|---|---|---|---|
| zoo time series | Yes | No | Use pandas DataFrame |
| pandas DataFrame | No | Yes | - |
| numpy ndarray | No | Yes | - |
| Date string periods | Yes | Yes | - |
| Integer index periods | No | Yes | - |
| Missing data (NA) | Handled | Not handled | Raise error on NaN |
| Multiple covariates | Yes | Yes | - |
| No covariates | Yes | Yes | - |

## Spike-and-Slab Variable Selection

| Feature | R | Python | Notes |
|---|---|---|---|
| Coordinate-wise sampling | Yes | Yes | Same algorithm |
| expected.model.size | Yes | Yes | Same prior calculation |
| Posterior inclusion probs | Via bsts | `ci.posterior_inclusion_probs` | - |
| Fallback to blocked g-prior | pi >= 1.0 | pi >= 1.0 | Same threshold |

## Numerical Equivalence (CI-Verified)

| Metric | Tolerance | Status |
|---|---|---|
| point_effect_mean | ±3% relative | Passing |
| cumulative_effect_total | ±3% relative | Passing |
| ci_lower / ci_upper | Tight parity (`±1%` no-cov, `±1%` covariates, `±1%` seasonal) | Passing |
| p_value significance | Match at alpha=0.05 | Passing |

Tests run against R CausalImpact 1.4.1 fixtures on every PR.

## Python-Only Extensions

Features that extend beyond R CausalImpact. No R equivalent exists.

| Feature | Method | Description |
|---|---|---|
| DATE decomposition | `ci.decompose()` | Decomposes causal effect into spot/persistent/trend (arXiv:2602.00836) |
| Retrospective mode | `mode="retrospective"` | Treatment indicators as covariates; effects from beta posteriors (arXiv:2602.00836) |
| Placebo test | `ci.run_placebo_test()` | Validates effect against null distribution from pre-period splits |
| Conformal inference | `ci.run_conformal_analysis()` | Distribution-free prediction intervals (Vovk et al., 2005) |
| DTW control selection | `select_controls()` | Automatic covariate selection via Dynamic Time Warping |
