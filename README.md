# bsts-causalimpact

Bayesian structural time series for causal inference in Python.
A faithful port of Google's [CausalImpact](https://google.github.io/CausalImpact/) R package. No TensorFlow required.

The Gibbs sampler is implemented in Rust (via PyO3), reproducing the same algorithm as R's `bsts` package while achieving 10-30x speedup.

## When to Use (and When Not to)

This method is valid only when all of the following hold:

- Control series are not contaminated by the intervention
- The relationship between treated and control series is stable across the pre- and post-intervention periods
- The pre-intervention period is sufficiently long (rule of thumb: at least 3x the post-intervention period)

If any of these assumptions are violated, the causal estimate will be unreliable. Consider a difference-in-differences or synthetic control approach instead.

## Installation

Requires Python 3.10+ and a Rust toolchain.

```bash
pip install bsts-causalimpact
```

For development (builds Rust extension locally):

```bash
git clone https://github.com/YuminosukeSato/bsts-causalimpact.git
cd bsts-causalimpact

# Install with uv (recommended)
uv sync --all-extras

# Or install with pip (builds Rust extension via maturin)
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from causal_impact import CausalImpact

# Prepare your data: first column = response, remaining columns = covariates
data = pd.read_csv("your_data.csv", index_col="date", parse_dates=True)

# Define pre- and post-intervention periods
pre_period = ["2020-01-01", "2020-03-14"]
post_period = ["2020-03-15", "2020-04-14"]

# Run the analysis
ci = CausalImpact(data, pre_period, post_period)

# Print a summary table
print(ci.summary())

# Print a narrative report
print(ci.report())

# Plot the results
fig = ci.plot()
fig.savefig("causal_impact.png")
```

## Comparison with Alternatives

| Aspect | bsts-causalimpact | tfp-causalimpact |
|---|---|---|
| Algorithm | Gibbs sampler (Rust, same as R bsts) | Gibbs sampler (via TensorFlow Probability) |
| Dependencies | numpy, pandas, matplotlib | tensorflow, tensorflow-probability (3GB+) |
| R compatibility tests | ±3% verified, CI-enforced | Not published |
| pip install | Works out of the box | Works out of the box |
| Python version | 3.10+ | 3.8+ |
| API | R CausalImpact compatible | Custom API |

The key differentiator is not the algorithm (both use Gibbs sampling) but the dependency footprint and verified R compatibility.

## Numerical Equivalence with R

This library verifies numerical equivalence with R CausalImpact (bsts 1.4.1) across 4 scenarios (basic, covariates, strong_effect, no_effect):

| Metric | Tolerance | Justification |
|---|---|---|
| `point_effect_mean` | ±3% relative | MC-SE with independent RNG, 4-sigma bound |
| `cumulative_effect_total` | ±3% relative | Same ratio as point_effect |
| `ci_lower` / `ci_upper` | ±15% relative | Systematic CI computation difference (Jensen) |
| `p_value` | Significance match | Classification at alpha=0.05 |

Tests run on every PR. Fixtures regenerated weekly from R CausalImpact 1.4.1.

## API

### `CausalImpact(data, pre_period, post_period, model_args=None, alpha=0.05)`

| Parameter | Type | Description |
|---|---|---|
| `data` | `DataFrame` or `ndarray` | First column is the response variable, remaining columns are covariates |
| `pre_period` | `list[str \| int]` | `[start, end]` of the pre-intervention period |
| `post_period` | `list[str \| int]` | `[start, end]` of the post-intervention period |
| `model_args` | `dict` or `ModelOptions` | MCMC parameters (see below) |
| `alpha` | `float` | Significance level for credible intervals (default: 0.05) |

#### Model Arguments

| Key | Default | Description |
|---|---|---|
| `niter` | 1000 | Total MCMC iterations |
| `nwarmup` | 500 | Burn-in iterations to discard |
| `nchains` | 1 | Number of MCMC chains |
| `seed` | 0 | Random seed for reproducibility |
| `prior_level_sd` | 0.01 | Prior standard deviation for the local level |
| `standardize_data` | `True` | Standardize data before fitting |
| `expected_model_size` | 1 | Expected number of active covariates (spike-and-slab prior) |

#### Methods and Properties

| Name | Returns | Description |
|---|---|---|
| `summary(output="summary")` | `str` | Tabular summary of causal effects |
| `report()` | `str` | Narrative interpretation of results |
| `plot(metrics=None)` | `Figure` | Matplotlib figure with original/pointwise/cumulative panels |
| `inferences` | `DataFrame` | Per-timestep effects, predictions, and credible intervals |
| `summary_stats` | `dict` | Aggregate statistics (effect mean, CI, p-value, etc.) |
| `posterior_inclusion_probs` | `ndarray \| None` | Posterior inclusion probability per covariate |

## Benchmark Results

| T | k | niter | This (Rust) | R (bsts) | vs R |
|--:|--:|------:|-----------:|---------:|----:|
| 100 | 0 | 1000 | 0.008s | 0.213s | 26x |
| 500 | 0 | 1000 | 0.033s | 0.997s | 30x |
| 1000 | 0 | 1000 | 0.069s | 2.108s | 31x |
| 1000 | 5 | 1000 | 0.197s | 2.171s | 11x |
| 5000 | 0 | 1000 | 0.330s | 10.264s | 31x |

Median of 3 runs. Reproduce: `python benchmarks/benchmark.py`

## Architecture

```
python/causal_impact/
    __init__.py          # Public API: CausalImpact, ModelOptions, __version__
    data.py              # DataProcessor: validation, standardization, period parsing
    main.py              # CausalImpact facade class
    options.py           # ModelOptions: typed MCMC configuration
    analysis.py          # CausalAnalysis: effect computation, CI, p-values
    summary.py           # SummaryFormatter: tabular and narrative reports
    plot.py              # Plotter: matplotlib visualization

src/ (Rust)
    lib.rs               # PyO3 entry point: run_gibbs_sampler()
    sampler.rs           # Gibbs sampler (R bsts-compatible algorithm)
    kalman.rs            # Kalman filter and simulation smoother
    state_space.rs       # State space model representation
    distributions.rs     # Posterior sampling distributions
```

## Development

```bash
git config core.hooksPath .githooks
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Numerical equivalence only
uv run pytest tests/test_numerical_equivalence.py -v

# Rust tests
cargo test
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, PR workflow, and test requirements.

## License

MIT
