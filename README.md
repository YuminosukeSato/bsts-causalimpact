# bsts-causalimpact

[![PyPI version](https://img.shields.io/pypi/v/bsts-causalimpact)](https://pypi.org/project/bsts-causalimpact/)
[![Python](https://img.shields.io/pypi/pyversions/bsts-causalimpact)](https://pypi.org/project/bsts-causalimpact/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R Numerical Equivalence](https://github.com/YuminosukeSato/bsts-causalimpact/actions/workflows/numerical-equivalence.yml/badge.svg?branch=main&event=push)](https://github.com/YuminosukeSato/bsts-causalimpact/actions/workflows/numerical-equivalence.yml)

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

Requires Python 3.10+.
Binary wheels are intended for supported platforms, so Rust is only required
when building from source.

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

### Example Output

![CausalImpact plot example](docs/images/causal_impact_example.png)

```
Posterior inference {CausalImpact}

                         Average        Cumulative
Actual                   136.32          3953.19
Prediction (s.d.)        125.42 (0.66)   3637.07 (19.08)
95% CI                   [124.18, 126.71]  [3601.33, 3674.59]

Absolute effect (s.d.)   10.90 (0.66)    316.13 (19.08)
95% CI                   [9.61, 12.13]   [278.60, 351.86]

Relative effect (s.d.)   8.69% (0.57%) 8.69% (0.57%)
95% CI                   [7.58%, 9.77%] [7.58%, 9.77%]

Posterior tail-area probability p: 0.001
Posterior prob. of a causal effect: 99.90%
```

## Comparison with Alternatives

| | R CausalImpact | bsts-causalimpact (this) | tfp-causalimpact | tfcausalimpact | pycausalimpact |
|---|---|---|---|---|---|
| Maintainer | Google | OSS | Google | WillianFuks | dafiti (stale) |
| Algorithm | Gibbs (bsts/C++) | Gibbs (Rust) | TFP-based | VI default / HMC | MLE (statsmodels) |
| Dependencies | R, bsts | numpy, pandas, matplotlib | TF, TFP (3 GB+) | TF, TFP (3 GB+) | statsmodels |
| Spike-and-slab | Yes | Yes | Unknown | No | No |
| Horseshoe prior | No | Yes (`prior_type='horseshoe'`) | No | No | No |
| Seasonal component | Yes | Yes (`nseasons`, `season_duration`) | Unknown | Yes (TFP STS) | No |
| Dynamic regression | Yes | Yes (`dynamic_regression=True`) | Unknown | No | No |
| R numerical test | Reference | ±1% CI-enforced + TOST/ROPE | Not published | Visual comparison (~8% diff) | Not tested |
| Speed (T=1000) | 2.1 s | 0.07 s (30x) | Seconds | Minutes (HMC: hours) | Sub-second |
| Python version | N/A (R) | 3.10+ | 3.8+ | 3.7-3.11 | 3.6-3.8 (stale) |
| Last release | Active | Active | 2023 | 2025-01 | 2020-05 |

### Why this library exists

Existing Python ports have fundamental limitations:

- pycausalimpact uses MLE (not MCMC), producing results that diverge substantially from R
- tfcausalimpact uses variational inference by default (not Gibbs sampling), and requires TensorFlow (3 GB+)
- tfp-causalimpact (Google's own Python port) does not publish numerical equivalence tests with R
- None of the above implement spike-and-slab variable selection matching R's bsts

This library reproduces the core Gibbs-sampler workflow from R's bsts package in Rust, with CI-enforced numerical equivalence tests on every commit.

## Numerical Equivalence with R

[![R Numerical Equivalence](https://github.com/YuminosukeSato/bsts-causalimpact/actions/workflows/numerical-equivalence.yml/badge.svg?branch=main&event=push)](https://github.com/YuminosukeSato/bsts-causalimpact/actions/workflows/numerical-equivalence.yml)

Verified against R CausalImpact 1.4.1 (bsts 0.9.10, R 4.5) across 5 scenarios.
Enforced on every commit via CI.

### Test Matrix

| Scenario | point_effect | cum_effect | ci_lower | ci_upper | rel_effect | p_value |
|---|---|---|---|---|---|---|
| basic | ±1% | ±1% | ±1% | ±1% | ±1% | alpha=0.05 |
| covariates | ±1% | ±1% | ±1% | ±1% | ±1% | alpha=0.05 |
| strong_effect | ±1% | ±1% | ±1% | ±1% | ±1% | alpha=0.05 |
| no_effect | abs<0.5 | abs<0.5 | abs<0.5 | abs<0.5 | abs<0.5 | alpha=0.05 |
| seasonal | ±1% | ±1% | ±1% | ±1% | ±1% | alpha=0.05 |

### Three-Layer Equivalence Verification

No other Python CausalImpact implementation has statistical equivalence tests.
This library provides three layers of verification, exceeding even Google's own Python port.

| Layer | Method | What it proves | Reference |
|---|---|---|---|
| 1. Deterministic | Seed-fixed ±1% threshold | Same seed, same result, every commit | Regression testing |
| 2. FDA TOST | 90% CI upper < delta (N=30 seeds) | Mean error is statistically below delta | Schuirmann (1987), FDA Guidance (2001) |
| 3. Bayesian ROPE | 95% HDI within [0, delta] (N=30 seeds) | Posterior of error is practically equivalent | Kruschke (2018) AMPPS |

Layer 1 runs on every commit (CI-blocking). Layers 2-3 run with `--runslow` flag.

```bash
# Layer 1: deterministic (runs in CI)
uv run pytest tests/test_numerical_equivalence.py -v

# Layers 2+3: TOST + ROPE (30 seeds x 4 scenarios, ~80s)
uv run pytest tests/test_equivalence_tost_rope.py -v --runslow
```

### CI Enforcement

Two-layer CI enforcement:

1. Fixture-based (`ci.yml`): Compares Python output against committed R reference data. Blocking on every PR/push.
2. Live R comparison (`numerical-equivalence.yml`): Installs R, regenerates fixtures from scratch, and compares. Blocking when R is available. Weekly auto-regeneration.

### How to Reproduce

1. Install R 4.5+ and packages: `install.packages(c("CausalImpact", "jsonlite"))`
2. Generate R reference: `Rscript scripts/generate_r_reference.R`
3. Run equivalence tests: `.venv/bin/pytest tests/test_numerical_equivalence.py -v`

### Equivalence Verification: Comparison with Other Implementations

| | R CausalImpact | bsts-causalimpact (this) | tfp-causalimpact (Google) | tfcausalimpact | pycausalimpact (dafiti) |
|---|---|---|---|---|---|
| R reference fixtures | N/A (is reference) | 5 scenarios, CI-enforced | None | None | None |
| Deterministic R test | N/A | ±1% all metrics (seed=42) | None | README demo only (~8% diff) | None |
| FDA TOST (Schuirmann 1987) | N/A | 30-seed, delta=1-2% | None | None | None |
| Bayesian ROPE (Kruschke 2018) | N/A | 30-seed, 95% HDI in ROPE | None | None | None |
| Self-consistency tolerance | `tolerance=0.01` | N/A | `rtol=0.2` (20%) | `assert_almost_equal` | `assert_array_equal` |
| CI-blocking R check | N/A | Every commit + weekly live R | None | None | None |

Evidence per implementation (all verified from source code, not documentation claims):

- tfp-causalimpact (Google official Python port): 47 tests across 7 files. No R reference fixtures. Self-consistency checks use `rtol=0.2` (±20%) and `atol=0.01`. No R output comparison exists in the test suite. README states "designed to produce results close to the R package" but this is not verified by any automated test. ([source](https://github.com/google/tfp-causalimpact))
- tfcausalimpact (WillianFuks): 64 tests across 7 files. `tests/fixtures/comparison_data.csv` exists but is only used in README demo, not in automated tests. The README demo itself shows ~8% difference in AbsEffect (R: -657 vs Python: -708.51). ([source](https://github.com/WillianFuks/tfcausalimpact))
- pycausalimpact (dafiti): 62 tests across 5 files. Uses MLE (not MCMC), fundamentally different algorithm. No R comparison of any kind. Repository archived. ([source](https://github.com/dafiti/causalimpact))
- causalimpact (jamalsenouci): 54 tests across 4 files. No R comparison. Open issue #7 reports "significantly different results from R". ([source](https://github.com/jamalsenouci/causalimpact))

### What is matching R and what is not

| R feature | Status | Detail |
|---|---|---|
| Local level model (Gibbs sampler) | Matching | Same algorithm as bsts: Kalman filter + simulation smoother |
| SdPrior(sample.size=32) for sigma2_level | Matching | InvGamma(16, 16 * sigma_guess^2) |
| Post-period Random Walk propagation | Matching | Forward simulation from last pre-period state |
| Data standardization (standardize.data=TRUE) | Matching | (y - mean) / sd using pre-period moments |
| prior.level.sd = 0.01 | Matching | Same default, same semantics |
| Spike-and-slab variable selection | Matching | Coordinate-wise sampling with StudentSpikeSlabPrior defaults (`expected.r2=0.8`, `prior.df=50`, `prior.information.weight=0.01`, `diagonal.shrinkage=0.5`) |
| expected.model.size | Matching | Unified default `2` in `CausalImpact` and `ModelOptions` |
| expected.r2 = 0.8, prior.df = 50 | Matching | Same documented residual variance prior defaults as BoomSpikeSlab / bsts |
| Seasonal component (`nseasons`, `season_duration`) | Matching | State-space model matching R bsts `AddSeasonal()` (±1% CI parity) |
| Dynamic regression | Supported | Time-varying coefficients via random-walk FFBS; `dynamic_regression=True` |
| Local linear trend | Supported | Opt in with `state_model="local_linear_trend"` |
| DATE decomposition | Extended | Decomposes effects into spot/persistent/trend (arXiv:2602.00836) |
| Retrospective mode | Extended | Treatment indicators as covariates; effects from beta posteriors (arXiv:2602.00836) |
| Placebo test | Extended | Null distribution from pre-period splits |
| Horseshoe prior | Extended | Continuous shrinkage alternative to spike-and-slab (Kohns & Bhattacharjee 2022) |
| Conformal inference | Extended | Distribution-free prediction intervals |
| DTW control selection | Extended | Automatic covariate selection via Dynamic Time Warping |

Matching = CI-enforced numerical equivalence with R bsts (±1% or tighter).
Supported = Feature implemented, no R parity fixture yet.
Extended = Python-only feature with no R equivalent.

### Beyond R: Python-Only Extensions

Features that go beyond R's CausalImpact. These have no R equivalent.

| Feature | Method | What it does | Reference |
|---|---|---|---|
| DATE decomposition | `ci.decompose()` | Decomposes causal effect into spot, persistent, and trend | Schaffe-Odeleye et al. (2026), arXiv:2602.00836 |
| Retrospective mode | `mode="retrospective"` | Treatment indicators as covariates; effects extracted from beta posteriors | Schaffe-Odeleye et al. (2026), arXiv:2602.00836 |
| Placebo test | `ci.run_placebo_test()` | Validates effect against null distribution from pre-period splits | |
| Conformal inference | `ci.run_conformal_analysis()` | Distribution-free prediction intervals | Vovk et al. (2005) |
| DTW control selection | `select_controls()` | Automatic covariate selection via Dynamic Time Warping | Sakoe & Chiba (1978) |
| Horseshoe prior | `ModelOptions(prior_type='horseshoe')` | Continuous shrinkage alternative to spike-and-slab for dense DGP | Kohns & Bhattacharjee (2022), arXiv:2011.00938 |

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
| `expected_model_size` | 2 | Expected number of active covariates (spike-and-slab prior) |
| `nseasons` | `None` | Optional seasonal cycle count (R-compatible API) |
| `season_duration` | `None` | Optional duration of each seasonal block; defaults to `1` when `nseasons` is set |
| `dynamic_regression` | `False` | Enable time-varying regression coefficients (random-walk beta) |
| `state_model` | `"local_level"` | `"local_level"` or `"local_linear_trend"` |
| `prior_type` | `"spike_slab"` | `"spike_slab"` or `"horseshoe"` (continuous shrinkage for dense DGP) |
| `mode` | `"forward"` | `"forward"` (counterfactual prediction) or `"retrospective"` (treatment indicators as covariates) |

#### Methods and Properties

| Name | Returns | Description |
|---|---|---|
| `summary(output="summary")` | `str` | Tabular summary of causal effects |
| `report()` | `str` | Narrative interpretation of results |
| `plot(metrics=None)` | `Figure` | Matplotlib figure with original/pointwise/cumulative panels |
| `inferences` | `DataFrame` | Per-timestep actuals, predictions, prediction s.d., and effect intervals |
| `summary_stats` | `dict` | Aggregate statistics (effect mean, CI, p-value, etc.) |
| `posterior_inclusion_probs` | `ndarray \| None` | Posterior inclusion probability per covariate (spike-and-slab only) |
| `posterior_shrinkage` | `ndarray \| None` | Mean shrinkage factor per covariate (horseshoe only) |
| `decompose(alpha=None)` | `DateDecomposition` | DATE decomposition into spot/persistent/trend components |
| `run_placebo_test(...)` | `PlaceboTestResults` | Placebo test for effect validation |
| `run_conformal_analysis(...)` | `ConformalResults` | Distribution-free conformal prediction intervals |

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
    decomposition.py     # DATE decomposition (spot/persistent/trend)
    retrospective.py     # Retrospective attribution mode

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
