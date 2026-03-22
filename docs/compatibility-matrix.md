# R CausalImpact Compatibility Matrix

Comparison of features between R CausalImpact (bsts 1.4.1) and this Python implementation.

## State Components

| Feature | R | Python | Notes |
|---|---|---|---|
| Local level | Yes | Yes | Identical algorithm |
| Local linear trend | Yes | No | Not yet implemented |
| Seasonality | Yes | Yes | R-compatible API with seasonal fixture coverage |
| Dynamic regression | Yes | No | Not yet implemented |
| Regression (static) | Yes | Yes | Identical algorithm |

## MCMC Parameters

| Parameter | R | Python | Notes |
|---|---|---|---|
| niter | Yes | Yes | Same default (1000) |
| nseasons | Yes | Yes | `ModelOptions.nseasons` or `model_args["nseasons"]` |
| season.duration | Yes | Yes | `ModelOptions.season_duration` or `model_args["season.duration"]` |
| prior.level.sd | Yes | Yes | Same default (0.01) |
| standardize.data | Yes | Yes | Same default (True) |
| expected.model.size | Yes | Yes | Legacy `CausalImpact` default is 2; `ModelOptions` keeps 1 |

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
| ci_lower / ci_upper | Tight parity (`±1.5%` no-cov, `±1%` covariates, `±5%` seasonal) | Passing |
| p_value significance | Match at alpha=0.05 | Passing |

Tests run against R CausalImpact 1.4.1 fixtures on every PR.
