# CausalImpact

A Python implementation of Google's [CausalImpact](https://google.github.io/CausalImpact/) (R package) for causal inference using Bayesian structural time series models.

The Gibbs sampler is written in Rust (via PyO3), achieving 10-30x speedup over R while maintaining numerical equivalence (±3% on point estimates, CI-verified).

## Why this library over tfp-causalimpact?

| Aspect | This library | tfp-causalimpact |
|---|---|---|
| Install | `pip install` (numpy, pandas only) | TensorFlow required (3GB+) |
| Speed (T=1000, k=0) | 0.07s | N/A (heavy TF overhead) |
| R compatibility | ±3% verified (CI-enforced) | Not published |
| Python version | 3.12+ | 3.8+ |
| Dependencies | numpy, pandas, matplotlib | tensorflow, tensorflow-probability |
| API | R CausalImpact compatible | Custom API |
| Algorithm | Rust Gibbs sampler (same as R bsts) | TFP-NUTS (different) |

## Numerical Equivalence with R

This library proves numerical equivalence with R CausalImpact (bsts) across 4 scenarios (basic, covariates, strong_effect, no_effect):

| Metric | Tolerance | Justification |
|---|---|---|
| `point_effect_mean` | ±3% relative | MC-SE with independent RNG, 4sigma bound |
| `cumulative_effect_total` | ±3% relative | Same ratio as point_effect |
| `ci_lower` / `ci_upper` | ±15% relative | Systematic CI computation difference (Jensen) |
| `p_value` | Significance match | Classification at alpha=0.05 |

Tests run on every PR. Fixtures regenerated weekly from R CausalImpact 1.4.1.

## Benchmark Results

| T | k | niter | This (Rust) | R (bsts) | vs R |
|--:|--:|------:|-----------:|---------:|----:|
| 100 | 0 | 1000 | 0.008s | 0.213s | 26x |
| 500 | 0 | 1000 | 0.033s | 0.997s | 30x |
| 1000 | 0 | 1000 | 0.069s | 2.108s | 31x |
| 1000 | 5 | 1000 | 0.197s | 2.171s | 11x |
| 5000 | 0 | 1000 | 0.330s | 10.264s | 31x |

Median of 3 runs. Reproduce: `python benchmarks/benchmark.py`

## Installation

Requires Python 3.12+ and a Rust toolchain.

```bash
git clone https://github.com/YuminosukeSato/CausalImpact.git
cd CausalImpact

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

## API

### `CausalImpact(data, pre_period, post_period, model_args=None, alpha=0.05)`

| Parameter | Type | Description |
|---|---|---|
| `data` | `DataFrame` or `ndarray` | First column is the response variable, remaining columns are covariates |
| `pre_period` | `list[str \| int]` | `[start, end]` of the pre-intervention period |
| `post_period` | `list[str \| int]` | `[start, end]` of the post-intervention period |
| `model_args` | `dict` (optional) | MCMC parameters (see below) |
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

#### Methods and Properties

| Name | Returns | Description |
|---|---|---|
| `summary(output="summary")` | `str` | Tabular summary of causal effects |
| `report()` | `str` | Narrative interpretation of results |
| `plot(metrics=None)` | `Figure` | Matplotlib figure with original/pointwise/cumulative panels |
| `inferences` | `DataFrame` | Per-timestep effects, predictions, and credible intervals |
| `summary_stats` | `dict` | Aggregate statistics (effect mean, CI, p-value, etc.) |

## Architecture

```
python/causal_impact/
    __init__.py          # Public API: CausalImpact, __version__
    data.py              # DataProcessor: validation, standardization, period parsing
    main.py              # CausalImpact facade class
    analysis.py          # CausalAnalysis: effect computation, CI, p-values
    summary.py           # SummaryFormatter: tabular and narrative reports
    plot.py              # Plotter: matplotlib visualization

src/ (Rust)
    lib.rs               # PyO3 entry point: run_gibbs_sampler()
    sampler.rs            # Gibbs sampler (R bsts-compatible algorithm)
    kalman.rs             # Kalman filter and simulation smoother
    state_space.rs        # State space model representation
    distributions.rs      # Posterior sampling distributions
```

## Setup

```bash
git config core.hooksPath .githooks
```

## Running Tests

```bash
# All tests (75 tests)
uv run pytest tests/ -v

# Numerical equivalence only (22 tests)
uv run pytest tests/test_numerical_equivalence.py -v

# Rust tests (13 tests)
cargo test
```

## License

MIT
