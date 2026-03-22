# Migration from R CausalImpact

This guide helps R CausalImpact users transition to the Python implementation.

## Installation

```bash
pip install bsts-causalimpact
```

## API Mapping

### R

```r
library(CausalImpact)

impact <- CausalImpact(data, pre.period, post.period,
                       model.args = list(niter = 1000))
summary(impact)
summary(impact, "report")
plot(impact)
```

### Python

```python
from causal_impact import CausalImpact

ci = CausalImpact(data, pre_period, post_period,
                  model_args={"niter": 1000})
print(ci.summary())
print(ci.report())
fig = ci.plot()
```

## Parameter Mapping

| R (model.args) | Python (model_args / ModelOptions) | Default |
|---|---|---|
| `niter` | `niter` | 1000 |
| `nseasons` | `nseasons` | `None` |
| `season.duration` | `season_duration` or `model_args["season.duration"]` | `1` when `nseasons` is set |
| `prior.level.sd` | `prior_level_sd` | 0.01 |
| `standardize.data` | `standardize_data` | True |
| `expected.model.size` | `expected_model_size` | 2 |

## Data Format

### R

```r
data <- zoo(cbind(y, x1, x2), dates)
pre.period <- c(as.Date("2020-01-01"), as.Date("2020-03-14"))
post.period <- c(as.Date("2020-03-15"), as.Date("2020-04-14"))
```

### Python

```python
data = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=dates)
pre_period = ["2020-01-01", "2020-03-14"]
post_period = ["2020-03-15", "2020-04-14"]
```

Key differences:

- Use pandas DataFrame instead of zoo object
- Periods are string lists instead of Date vectors
- Integer indices are also supported

## Output Mapping

| R | Python | Notes |
|---|---|---|
| `summary(impact)` | `ci.summary()` | Same tabular format |
| `summary(impact, "report")` | `ci.report()` | Same narrative output |
| `plot(impact)` | `ci.plot()` | Returns matplotlib Figure |
| `impact$summary` | `ci.summary_stats` | Dict with aggregate stats |
| `impact$series` | `ci.inferences` | DataFrame with per-timestep data |

## Numerical Equivalence

This library verifies ±3% agreement with R CausalImpact on point estimates and cumulative effects across multiple test scenarios, including a seasonal fixture. Tests run on every PR.

Differences arise from independent RNG implementations (R's `set.seed` vs Rust's `ChaCha8Rng`), not from algorithmic differences.

Use `model_args={"state_model": "local_linear_trend"}` when you want the
optional local linear trend state instead of the default local level model.

## What Is Not Supported

- Custom bsts model objects

These features may be added in future versions.
