# API Reference

## `CausalImpact`

```python
from causal_impact import CausalImpact

ci = CausalImpact(data, pre_period, post_period, model_args=None, alpha=0.05)
```

### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `data` | `DataFrame` or `ndarray` | First column is the response variable, remaining columns are covariates |
| `pre_period` | `list[str \| int]` | `[start, end]` of the pre-intervention period |
| `post_period` | `list[str \| int]` | `[start, end]` of the post-intervention period |
| `model_args` | `dict` or `ModelOptions` | MCMC parameters (see below) |
| `alpha` | `float` | Significance level for credible intervals (default: 0.05) |

### Methods

| Method | Returns | Description |
|---|---|---|
| `summary(output="summary", digits=2)` | `str` | Tabular summary of causal effects. Set `output="report"` for narrative form. |
| `report()` | `str` | Narrative interpretation of results (shortcut for `summary(output="report")`) |
| `plot(metrics=None)` | `Figure` | Matplotlib figure with original/pointwise/cumulative panels. Pass a list like `["original", "cumulative"]` to select panels. |

### Properties

| Property | Type | Description |
|---|---|---|
| `inferences` | `DataFrame` | Per-timestep actuals, predictions, prediction s.d., and effect intervals |
| `summary_stats` | `dict` | Aggregate statistics (effect mean, CI, p-value, etc.) |
| `posterior_inclusion_probs` | `ndarray \| None` | Posterior inclusion probability per covariate (requires covariates) |

## `ModelOptions`

```python
from causal_impact import ModelOptions

opts = ModelOptions(niter=5000, seed=123)
ci = CausalImpact(data, pre_period, post_period, model_args=opts)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `niter` | `int` | 1000 | Total MCMC iterations |
| `nwarmup` | `int` | 500 | Burn-in iterations to discard |
| `nchains` | `int` | 1 | Number of MCMC chains |
| `seed` | `int` | 0 | Random seed for reproducibility |
| `prior_level_sd` | `float` | 0.01 | Prior standard deviation for the local level |
| `standardize_data` | `bool` | `True` | Standardize data before fitting |
| `expected_model_size` | `int` | 2 | Expected number of active covariates for spike-and-slab prior |
| `dynamic_regression` | `bool` | `False` | Enable time-varying regression coefficients |
| `state_model` | `str` | `"local_level"` | `"local_level"` or `"local_linear_trend"` |
| `nseasons` | `int \| None` | `None` | Seasonal cycle count |
| `season_duration` | `int \| None` | `None` | Duration of each seasonal block; defaults to 1 when `nseasons` is set |

## `CausalImpactResults`

Returned by `ci._results`. A frozen dataclass containing all computed quantities.

### Fields

| Field | Type | Description |
|---|---|---|
| `actual` | `ndarray` | Observed y values in the post period |
| `point_effects` | `ndarray` | Mean effect per time point |
| `point_effect_lower` | `ndarray` | Lower pointwise credible interval per time point |
| `point_effect_upper` | `ndarray` | Upper pointwise credible interval per time point |
| `point_effect_mean` | `float` | Mean of point effects across time |
| `ci_lower` | `float` | Lower CI bound on average effect |
| `ci_upper` | `float` | Upper CI bound on average effect |
| `cumulative_effect_total` | `float` | Total cumulative effect |
| `relative_effect_mean` | `float` | Relative effect (effect / predicted) |
| `p_value` | `float` | Bayesian one-sided tail probability |
| `predictions_mean` | `ndarray` | Mean counterfactual prediction |
| `predictions_sd` | `ndarray` | Posterior standard deviation of the counterfactual prediction |
| `predictions_lower` | `ndarray` | Lower CI on counterfactual |
| `predictions_upper` | `ndarray` | Upper CI on counterfactual |
