# Advanced Features

This library reproduces R's CausalImpact faithfully, but it doesn't stop there.
Five additional capabilities extend the analysis beyond what R offers.

| Feature | Method | What it adds |
|---|---|---|
| DATE decomposition | `ci.decompose()` | Break effects into spot / persistent / trend |
| Retrospective mode | `mode="retrospective"` | Estimate effects from beta posteriors |
| Placebo test | `ci.run_placebo_test()` | Validate effects against null distribution |
| Conformal inference | `ci.run_conformal_analysis()` | Distribution-free prediction intervals |
| DTW control selection | `select_controls()` | Automatic covariate selection |
| Horseshoe prior | `ModelOptions(prior_type='horseshoe')` | Continuous shrinkage for dense DGP |

---

## DATE Decomposition

### What it does

Standard CausalImpact reports a single effect trajectory: "how large is the
effect at each time point?" DATE decomposition breaks this into three
interpretable components:

| Component | Shape | What it captures |
|---|---|---|
| Spot | Impulse at t=0 only | Immediate, one-time impact |
| Persistent | Step function from t=0 onward | Permanent baseline shift |
| Trend | Linear ramp 0, 1, 2, ... | Growing or decaying effect |

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.

### How it works

The decomposition fits a simple linear model to each MCMC sample's pointwise
effect vector. The design matrix has three columns:

```
t=0:  [1, 1, 0]   <- spot + persistent
t=1:  [0, 1, 1]   <- persistent + trend*1
t=2:  [0, 1, 2]   <- persistent + trend*2
...
t=T:  [0, 1, T]   <- persistent + trend*T
```

A pseudoinverse is computed once and applied to all posterior samples,
producing posterior distributions for each component's coefficient.

### Usage

```python
ci = CausalImpact(data, pre_period, post_period, model_args={"seed": 42})
dec = ci.decompose()

print(f"Spot:       {dec.spot.coefficient:+.2f}  "
      f"[{dec.spot.ci_lower:+.2f}, {dec.spot.ci_upper:+.2f}]")
print(f"Persistent: {dec.persistent.coefficient:+.2f}  "
      f"[{dec.persistent.ci_lower:+.2f}, {dec.persistent.ci_upper:+.2f}]")
if dec.trend is not None:
    print(f"Trend:      {dec.trend.coefficient:+.2f}  "
          f"[{dec.trend.ci_lower:+.2f}, {dec.trend.ci_upper:+.2f}]")
```

### Plotting

Add `"decomposition"` to the metrics list to see a fourth panel:

```python
fig = ci.plot(metrics=["original", "pointwise", "cumulative", "decomposition"])
```

The default 3-panel plot is unchanged.

### Business examples

| Scenario | Expected pattern |
|---|---|
| Ad campaign launch day | Large spot, small persistent |
| Permanent price reduction | Small spot, large persistent |
| New feature with gradual adoption | Small persistent, positive trend |
| One-time promotion | Large spot, near-zero persistent and trend |

---

## Retrospective Mode

### What it does

Standard (forward) mode builds a counterfactual prediction from the
pre-period and compares it against post-period observations. Retrospective
mode takes a different approach: it adds treatment indicator columns directly
as covariates and fits the model on the entire time series. Treatment effects
are extracted from the beta posteriors.

| | Forward mode (default) | Retrospective mode |
|---|---|---|
| Model fit | Pre-period only | Entire series |
| Effect estimation | Counterfactual subtraction | Beta posteriors for treatment columns |
| Decomposition | Post-hoc via `ci.decompose()` | Built-in during initialization |
| Use case | Standard causal inference | Direct effect attribution |

Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.

### Usage

```python
ci = CausalImpact(
    data, pre_period, post_period,
    model_args={
        "mode": "retrospective",
        "prior_level_sd": 0.001,
        "niter": 2000,
        "seed": 42,
    },
)

# Decomposition is auto-populated
dec = ci._decomposition
print(f"Persistent effect: {dec.persistent.coefficient:+.2f}")

# Standard summary and plot still work
print(ci.summary())
```

### The `prior_level_sd` parameter

In retrospective mode, the local level state (random walk) and the persistent
treatment indicator (step function) compete to explain level shifts. Setting
`prior_level_sd` to a small value (e.g. 0.001) constrains the random walk and
forces the model to attribute level shifts to the treatment columns.

```python
# Recommended for retrospective mode
model_args={"mode": "retrospective", "prior_level_sd": 0.001}
```

### When to use which mode

Use forward mode (default) when:

- You want counterfactual predictions
- You are following the standard CausalImpact workflow
- You want to run placebo tests or conformal analysis

Use retrospective mode when:

- You want spot/persistent/trend decomposition from the model itself
- You prefer a single-model-fit approach
- You want coefficients with direct causal interpretation

---

## Placebo Test

### What it does

A placebo test validates whether the detected effect is real by checking
whether similar "effects" appear in the pre-period where no intervention
occurred. The pre-period is split at multiple points, and a CausalImpact
analysis is run for each split. The real effect is ranked against this
null distribution.

### Usage

```python
ci = CausalImpact(data, pre_period, post_period, model_args={"seed": 42})
placebo = ci.run_placebo_test()

print(f"Placebo p-value: {placebo.p_value:.3f}")
print(f"Real effect:     {placebo.real_effect:.2f}")
print(f"# placebos:      {placebo.n_placebos}")
```

### Interpretation

| p-value | Meaning |
|---|---|
| < 0.05 | Effect is unlikely to be a pre-period artifact |
| > 0.10 | Effect may be indistinguishable from pre-period noise |

A small p-value means the real effect is larger than most placebo effects,
strengthening the causal claim.

---

## Conformal Inference

### What it does

Conformal inference provides distribution-free prediction intervals that
make no parametric assumptions about the data. It splits the pre-period into
training and calibration halves, computes nonconformity scores, and derives
a conformal quantile for the post-period.

### Usage

```python
ci = CausalImpact(data, pre_period, post_period, model_args={"seed": 42})
conformal = ci.run_conformal_analysis()

print(f"Conformal q_hat:      {conformal.q_hat:.2f}")
print(f"Calibration points:   {conformal.n_calibration}")
print(f"Lower bound shape:    {conformal.lower.shape}")
print(f"Upper bound shape:    {conformal.upper.shape}")
```

### When to use

- As a robustness check alongside Bayesian credible intervals
- When you want guarantees that do not depend on model assumptions
- When the data may not satisfy the Bayesian model's priors

Reference: Vovk et al. (2005), "Algorithmic Learning in a Random World."

---

## DTW Control Selection

### What it does

When you have many candidate control series and need to choose which ones
to include as covariates, DTW (Dynamic Time Warping) control selection
automatically ranks candidates by their similarity to the treated series
in the pre-period.

### Usage

```python
from causal_impact import select_controls

# candidates: DataFrame with candidate control series as columns
selected = select_controls(
    treated=data["y"],
    candidates=candidates,
    n_controls=3,
    pre_period=pre_period,
)

# Use selected controls in CausalImpact
analysis_data = data[["y"]].join(selected)
ci = CausalImpact(analysis_data, pre_period, post_period)
```

Reference: Sakoe & Chiba (1978), "Dynamic programming algorithm optimization
for spoken word recognition."

---

## Horseshoe Prior

### What it does

The horseshoe prior (Carvalho, Polson & Scott 2010) is a continuous shrinkage
alternative to spike-and-slab variable selection. While spike-and-slab performs
discrete inclusion/exclusion of covariates (gamma_j in {0,1}), the horseshoe
applies adaptive shrinkage that can handle dense DGP settings where many
covariates have true effects.

Reference: Kohns & Bhattacharjee (2022), "Horseshoe Prior for Sparse Bayesian
Structural Time Series" (arXiv:2011.00938).

### Hierarchical model

The horseshoe hierarchy uses Half-Cauchy priors decomposed into InvGamma
auxiliary variables (Makalic & Schmidt 2015):

```
beta_j | lambda_j, tau, sigma2  ~ N(0, lambda_j^2 * tau^2 * sigma2_obs)
lambda_j^2 | nu_j               ~ InvGamma(1/2, 1/nu_j)
nu_j                             ~ InvGamma(1/2, 1)
tau^2 | xi                       ~ InvGamma(1/2, 1/xi)
xi                               ~ InvGamma(1/2, 1)
```

The conditional posteriors used in the Gibbs sampler:

```
lambda_j^2 | .  ~ InvGamma(1,       1/nu_j + beta_j^2 / (2 * tau^2 * sigma2))
nu_j       | .  ~ InvGamma(1,       1 + 1/lambda_j^2)
tau^2      | .  ~ InvGamma((k+1)/2, 1/xi + sum(beta_j^2 / (2 * lambda_j^2 * sigma2)))
xi         | .  ~ InvGamma(1,       1 + 1/tau^2)
```

### Beta joint update

Unlike spike-and-slab (coordinate-wise), horseshoe uses a joint beta update:

```
A = X'X + diag(1 / (lambda_j^2 * tau^2))    (precision matrix)
b = X'(y - state - seasonal)                 (right-hand side)
beta ~ N(A^{-1} b, sigma2_obs * A^{-1})     (sampled via Cholesky)
```

### Shrinkage factor

The shrinkage factor kappa_j measures how much each covariate is shrunk:

```
kappa_j = 1 / (1 + lambda_j^2 * tau^2)
```

- kappa_j close to 1: strong shrinkage (covariate effectively excluded)
- kappa_j close to 0: weak shrinkage (covariate effectively included)

The `posterior_shrinkage` property returns E[kappa_j] averaged over post-warmup
MCMC iterations.

### When to use

| Scenario | Recommended prior |
|---|---|
| Few true covariates among many candidates (sparse DGP) | `spike_slab` (default) |
| Many covariates with true effects (dense DGP) | `horseshoe` |
| Time-varying coefficients | `spike_slab` (horseshoe + dynamic_regression not supported) |

### Usage

```python
from causal_impact import CausalImpact, ModelOptions

ci = CausalImpact(
    data, pre_period, post_period,
    model_args=ModelOptions(prior_type='horseshoe', niter=2000, seed=42),
)

# Shrinkage diagnostics
print(ci.posterior_shrinkage)   # E[kappa_j] per covariate
# posterior_inclusion_probs is None for horseshoe
```

### References

- Carvalho, C.M., Polson, N.G. & Scott, J.G. (2010). The horseshoe estimator
  for sparse signals. Biometrika, 97(2), 465-480.
- Kohns, D. & Bhattacharjee, A. (2022). Horseshoe Prior for Sparse Bayesian
  Structural Time Series. arXiv:2011.00938.
- Makalic, E. & Schmidt, D.F. (2015). A simple sampler for the horseshoe
  estimator. IEEE Signal Processing Letters, 23(1), 179-182.

---

## Citation

```bibtex
@article{schaffeodeleye2026dynamic,
  title   = {Dynamic Causal Inference with Time Series Data},
  author  = {Schaffe-Odeleye, Ayo and others},
  journal = {arXiv preprint arXiv:2602.00836},
  year    = {2026}
}
```
