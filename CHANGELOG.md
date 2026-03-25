# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- DATE decomposition (`ci.decompose()`): decomposes pointwise causal effects
  into spot, persistent, and trend components via OLS projection on the
  MCMC posterior. Pure Python, no Rust changes.
  Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.
- Decomposition plot panel: `ci.plot(metrics=[..., "decomposition"])` adds a
  fourth panel showing the three components with credible intervals.
- `DateDecomposition` and `EffectComponent` exported from `causal_impact`.
- Retrospective attribution mode (`mode="retrospective"`): treatment indicator
  columns (spot, persistent, trend) are added as covariates and the model is fit
  on the entire time series. Effects are extracted from beta posteriors.
  Reference: Schaffe-Odeleye et al. (2026), arXiv:2602.00836.
- "Beyond R" documentation section covering all Python-only extensions
  (DATE decomposition, retrospective mode, placebo test, conformal inference,
  DTW control selection).
- `docs/theory.md`: mathematical background for DATE decomposition.
- MathJax support in mkdocs for LaTeX rendering.

## [1.3.1] - 2026-03-23

### Changed

- Reduced Rust-side boundary copies by passing contiguous NumPy arrays through
  the PyO3 entry point and borrowing the response vector during sampling.
- Reduced hot-path temporary allocations in the sampler and Kalman smoother by
  reusing scratch buffers and replacing clone-heavy state handoff.
- Switched release orchestration to semantic-release driven versioning and
  added a `clippy::perf` gate in CI and pre-push checks.

## [1.1.0] - 2026-03-23

### Changed

- Seasonal component: migrated from dummy regression to Kalman state-space
  model matching R bsts `AddSeasonal()` algorithm.
- Seasonal numerical equivalence tolerance tightened from ±5% to ±1%.

### Fixed

- `__version__` in Rust module now correctly tracks release version.

## [1.0.0] - 2026-03-23

### Added

- Added `state_model` with `local_level` and `local_linear_trend` support.
- Added `actual` and `predictions_sd` to `ci.inferences`.
- Added `py.typed` for PEP 561 type discovery.
- Added wheel smoke tests to the release workflow.

### Changed

- Made summary and report CI labels respect `alpha`.
- Unified `expected_model_size` defaults at `2`.
- Clarified installation guidance around binary wheels versus source builds.

## [0.3.0]

### Added

- Added dynamic regression with time-varying coefficients.
- Added seasonal components and MkDocs documentation site.

### Changed

- Improved numerical equivalence coverage against R fixtures.
