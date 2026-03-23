# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

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
