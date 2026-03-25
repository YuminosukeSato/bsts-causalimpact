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
| `plot(metrics=None)` | `Figure` | Matplotlib figure with original/pointwise/cumulative panels. Pass a list like `["original", "cumulative"]` to select panels. Add `"decomposition"` for the DATE panel. |
| `decompose(alpha=None)` | `DateDecomposition` | DATE decomposition of pointwise effects into spot/persistent/trend. Call before plotting with `"decomposition"` metric. |
| `run_placebo_test(n_placebos=None, min_pre_length=3)` | `PlaceboTestResults` | Validates the causal effect against a null distribution from pre-period splits. |
| `run_conformal_analysis(alpha=None)` | `ConformalResults` | Distribution-free prediction intervals via split conformal inference. |

### Properties

| Property | Type | Description |
|---|---|---|
| `inferences` | `DataFrame` | Per-timestep actuals, predictions, prediction s.d., and effect intervals |
| `summary_stats` | `dict` | Aggregate statistics (effect mean, CI, p-value, etc.) |
| `posterior_inclusion_probs` | `ndarray \| None` | Posterior inclusion probability per covariate (spike-and-slab only; returns `None` for horseshoe) |
| `posterior_shrinkage` | `ndarray \| None` | Mean shrinkage factor kappa_j per covariate (horseshoe only; returns `None` for spike-and-slab). Values near 0 = weakly shrunk (included), near 1 = strongly shrunk. |

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
| `prior_type` | `str` | `"spike_slab"` | `"spike_slab"` (discrete variable selection) or `"horseshoe"` (continuous shrinkage). Horseshoe is recommended for dense DGP settings. |
| `state_model` | `str` | `"local_level"` | `"local_level"` or `"local_linear_trend"` |
| `mode` | `str` | `"forward"` | `"forward"` (counterfactual prediction) or `"retrospective"` (treatment indicators as covariates). Retrospective mode adds spot/persistent/trend columns to X and fits on the entire series. Effects are extracted from beta posteriors. |
| `nseasons` | `int \| None` | `None` | Seasonal cycle count. `nseasons=1` is equivalent to no seasonal component. |
| `season_duration` | `int \| None` | `None` | Duration of each seasonal block; defaults to 1 when `nseasons` is set. Requires `nseasons` to be set. |

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

## Horseshoe Prior (alternative to spike-and-slab)

CausalImpact supports the horseshoe prior (Carvalho, Polson & Scott 2010)
applied to BSTS regression, following the formulation of
Kohns & Bhattacharjee (2022) (arXiv:2011.00938).

### When to use horseshoe

| Scenario | Recommended prior |
|---|---|
| Few true covariates (sparse DGP) | `spike_slab` (default) |
| Many true covariates (dense DGP) | `horseshoe` |

### Usage

```python
from causal_impact import CausalImpact, ModelOptions

ci = CausalImpact(
    data, pre_period, post_period,
    model_args=ModelOptions(prior_type='horseshoe'),
)
print(ci.posterior_shrinkage)   # mean(kappa_j), 0=included 1=shrunk
# ci.posterior_inclusion_probs is None for horseshoe (spike-slab only)
```

### Shrinkage diagnostics

| Property | prior_type | Meaning |
|---|---|---|
| `posterior_inclusion_probs` | `spike_slab` | E[gamma_j] — discrete inclusion probability |
| `posterior_inclusion_probs` | `horseshoe` | `None` (not applicable) |
| `posterior_shrinkage` | `horseshoe` | E[kappa_j] — continuous shrinkage factor kappa_j = 1/(1+lambda_j^2 * tau^2). Values close to 0 indicate the covariate is weakly shrunk (effectively included). |
| `posterior_shrinkage` | `spike_slab` | `None` (not applicable) |

### Incompatible combinations

- `prior_type='horseshoe'` + `dynamic_regression=True` raises `ValueError`
- `prior_type='horseshoe'` + `mode='retrospective'` raises `ValueError`

### References

- Kohns, D. & Bhattacharjee, A. (2022). Horseshoe Prior for Sparse Bayesian Structural Time Series. arXiv:2011.00938.
- Makalic, E. & Schmidt, D.F. (2015). A simple sampler for the horseshoe estimator. IEEE Signal Processing Letters, 23(1), 179-182.
- Carvalho, C.M., Polson, N.G. & Scott, J.G. (2010). The horseshoe estimator for sparse signals. Biometrika, 97(2), 465-480.

---

## Beyond R Extensions

### Retrospective Mode

In retrospective mode (`mode="retrospective"`), treatment indicator columns
(spot, persistent, trend) are added as covariates and the BSTS model is fit on
the entire time series. Treatment effects are extracted directly from the beta
posteriors for the treatment columns, rather than from counterfactual predictions.

Key differences from forward mode:

- The model fits on all data (pre + post), not just pre-period
- Spike-and-slab variable selection is auto-disabled
- `ci._decomposition` is populated automatically (no need to call `ci.decompose()`)
- `ci.decompose()` is still available but uses the forward-mode OLS projection

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.

### `DateDecomposition`

Returned by `ci.decompose()` or auto-populated in retrospective mode.
A frozen dataclass containing the DATE decomposition results.

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.

| Field | Type | Description |
|---|---|---|
| `spot` | `EffectComponent` | Immediate impulse effect at intervention |
| `persistent` | `EffectComponent` | Sustained level shift from intervention onward |
| `trend` | `EffectComponent \| None` | Linearly growing/decaying effect. `None` when `T_post < 3`. |
| `residual_mean` | `ndarray` | Mean residual trajectory after decomposition |
| `design_matrix` | `ndarray` | The D matrix used for OLS projection |
| `alpha` | `float` | Significance level used for credible intervals |

### `EffectComponent`

Each component of the DATE decomposition.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Component name: `"spot"`, `"persistent"`, or `"trend"` |
| `coefficient` | `float` | Posterior mean of the OLS coefficient |
| `ci_lower` | `float` | Lower credible interval bound on coefficient |
| `ci_upper` | `float` | Upper credible interval bound on coefficient |
| `samples` | `ndarray` | Per-sample trajectories, shape `(n_samples, T_post)` |
| `coefficients` | `ndarray` | OLS coefficient per sample, shape `(n_samples,)` |
| `mean` | `ndarray` | Posterior mean trajectory, shape `(T_post,)` |
| `lower` | `ndarray` | Lower CI on trajectory, shape `(T_post,)` |
| `upper` | `ndarray` | Upper CI on trajectory, shape `(T_post,)` |

### `PlaceboTestResults`

Returned by `ci.run_placebo_test()`. Validates the effect against a null distribution.

| Field | Type | Description |
|---|---|---|
| `p_value` | `float` | Fraction of placebo effects >= real effect |
| `rank_ratio` | `float` | Conservative p-value estimate |
| `effect_distribution` | `ndarray` | Absolute average effects from each placebo split |
| `real_effect` | `float` | Absolute average effect from the real intervention |
| `n_placebos` | `int` | Number of placebo splits evaluated |

### `ConformalResults`

Returned by `ci.run_conformal_analysis()`. Distribution-free prediction intervals.

| Field | Type | Description |
|---|---|---|
| `q_hat` | `float` | Conformal quantile threshold |
| `lower` | `ndarray` | Lower conformal prediction interval for post-period |
| `upper` | `ndarray` | Upper conformal prediction interval for post-period |
| `n_calibration` | `int` | Number of calibration points used |
