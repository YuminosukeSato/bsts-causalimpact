# Migration from tfp-causalimpact

This guide helps you migrate from `tfp-causalimpact` (TensorFlow Probability) to `bsts-causalimpact`.

## Installation Change

```bash
# Before
pip install tfcausalimpact

# After
pip install bsts-causalimpact
```

This removes the TensorFlow dependency (~3GB).

## API Differences

### Basic Usage

tfp-causalimpact:

```python
from causalimpact import CausalImpact

ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
ci.plot()
```

bsts-causalimpact:

```python
from causal_impact import CausalImpact

ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
fig = ci.plot()
```

Key differences:

- Import path: `causal_impact` (underscore), not `causalimpact`
- `plot()` returns a matplotlib `Figure` object instead of displaying inline
- Call `fig.savefig("output.png")` or display with `fig` in a notebook

### Data Format

Both libraries accept pandas DataFrames with the same structure:

- First column: response variable (y)
- Remaining columns: covariates (optional)
- Index: DatetimeIndex or integer index

No changes needed in data preparation.

### Model Configuration

tfp-causalimpact:

```python
ci = CausalImpact(data, pre_period, post_period, model_args={"nseasons": 7})
```

bsts-causalimpact:

```python
from causal_impact import ModelOptions

opts = ModelOptions(niter=1000, nwarmup=500, seed=42)
ci = CausalImpact(data, pre_period, post_period, model_args=opts)

# dict also works
ci = CausalImpact(data, pre_period, post_period, model_args={"niter": 1000})
```

Note: `bsts-causalimpact` does not support seasonal components (`nseasons`) in the current version. If your analysis requires seasonality, you need to handle it via covariates (e.g., Fourier terms).

### Output Access

| Feature | tfp-causalimpact | bsts-causalimpact |
|---|---|---|
| Summary table | `ci.summary()` | `ci.summary()` |
| Narrative report | `ci.summary(output="report")` | `ci.report()` |
| Inferences DataFrame | `ci.inferences` | `ci.inferences` |
| Aggregate statistics | N/A | `ci.summary_stats` |
| Plot | `ci.plot()` | `fig = ci.plot()` |

### What Is Not Supported

- Seasonal state components
- Custom prior specification beyond `prior_level_sd`
- TensorFlow-based model customization

## Algorithm Comparison

Both libraries use a Gibbs sampler for Bayesian structural time series. The key difference is implementation:

- tfp-causalimpact: Gibbs sampler via TensorFlow Probability
- bsts-causalimpact: Gibbs sampler in Rust (same algorithm as R's bsts package)

Results will not be numerically identical due to different random number generators, but both converge to the same posterior with sufficient iterations.
