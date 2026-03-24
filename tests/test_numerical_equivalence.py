"""Numerical equivalence tests against R CausalImpact (bsts).

Proof basis:
  Local Level Model + identical prior + identical conjugate update rules
  -> converges to the same true posterior with sufficient niter (MCMC ergodic theorem)
  -> numerical equivalence verified per metric with tolerance bands

Tolerances:
  point_effect_mean     +-3%  relative error
  cumulative_effect     +-3%  relative error
  ci_lower / ci_upper
    - no-covariates     +-1.5%  relative error
      Converged value ~0.83% at niter=50000, threshold accounts for +-0.5% MCMC variance.
      Due to prior correction (SdPrior sample.size=32) + post-period Random Walk propagation.
    - covariates        +-1% relative error
      Threshold aligned with R SpikeSlabPrior for the static regression prior.
    - Consistent with summary CI definition in Google R source
  p_value               significance classification match (alpha=0.05)

no_effect scenario (true_effect=0):
  Effect ~ 0 causes relative error to diverge. Compared by absolute error < 2.0.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from causal_impact import CausalImpact

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# R: niter=5000, SuggestBurn(0.1) ≈ 500 → 4500 post-warmup
# Python: niter=20000, nwarmup=2000 → 18000 post-warmup
# niter=20000 is needed to keep CI bounds MCMC variance within +-0.5%
MCMC_ARGS = {
    "niter": 20000,
    "nwarmup": 2000,
    "seed": 42,
    "prior_level_sd": 0.01,
}

TOL_POINT = 0.03  # ±3% for point estimates
# ±1.0% for no-covariates CI bounds
# (actual error ~0.44%, MCMC variance margin included)
TOL_CI_NO_COV = 0.01
TOL_CI_COV = 0.01  # ±1% after aligning with the R static regression prior
TOL_CI_PHASE2 = 0.03  # explicit Phase 2 benchmark for covariate CI bounds
# State-space seasonal (DK simulation smoother): actual error < 0.1%
TOL_POINT_SEASONAL = 0.01
TOL_CI_SEASONAL = 0.01
ABS_TOL_NO_EFFECT = 2.0  # absolute tolerance when true_effect=0


def _load_fixture(scenario: str) -> dict:
    path = FIXTURES_DIR / f"r_reference_{scenario}.json"
    if not path.exists():
        msg = f"R fixture not found: {path}. Run scripts/generate_r_reference.R"
        if os.environ.get("CI") == "true":
            pytest.fail(msg)
        else:
            pytest.skip(msg)
    return json.loads(path.read_text())


def _build_df(
    fixture: dict,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    y = np.array(fixture["data"]["y"])
    n_pre = fixture["n_pre"]
    n = fixture["n"]

    x_data = fixture["data"].get("x")
    has_x = x_data and isinstance(x_data, dict) and len(x_data) > 0
    if has_x:
        cols = {"y": y}
        for xname, xvals in x_data.items():
            cols[xname] = np.array(xvals)
        df = pd.DataFrame(cols)
    else:
        df = pd.DataFrame({"y": y})

    pre_period = [0, n_pre - 1]
    post_period = [n_pre, n - 1]
    return df, pre_period, post_period


def _run_causal_impact(fixture: dict) -> dict:
    df, pre_period, post_period = _build_df(fixture)
    model_args = {**MCMC_ARGS}
    model_args.update(
        {
            key.replace(".", "_"): value
            for key, value in fixture.get("model_args", {}).items()
        }
    )
    ci = CausalImpact(df, pre_period, post_period, model_args=model_args)
    return ci.summary_stats


def _assert_relative(py_val: float, r_val: float, tol: float, metric: str) -> None:
    """Compare by relative error. Falls back to absolute error when r_val ~ 0."""
    abs_threshold = 0.5
    if abs(r_val) < abs_threshold:
        abs_diff = abs(py_val - r_val)
        assert abs_diff < abs_threshold, (
            f"{metric}: abs diff {abs_diff:.6f} >= {abs_threshold}"
            f" (py={py_val:.6f}, r={r_val:.6f})"
        )
    else:
        rel_err = abs(py_val - r_val) / abs(r_val)
        assert rel_err < tol, (
            f"{metric}: rel err {rel_err:.4%} >= {tol:.0%}"
            f" (py={py_val:.6f}, r={r_val:.6f})"
        )


def _assert_absolute(py_val: float, r_val: float, tol: float, metric: str) -> None:
    """Compare by absolute error. For the true_effect=0 scenario."""
    abs_diff = abs(py_val - r_val)
    assert abs_diff < tol, (
        f"{metric}: abs diff {abs_diff:.6f} >= {tol} (py={py_val:.6f}, r={r_val:.6f})"
    )


def _assert_significance(py_stats: dict, r_output: dict) -> None:
    r_sig = r_output["p_value"] < 0.05
    py_sig = py_stats["p_value"] < 0.05
    r_label = "sig" if r_sig else "ns"
    py_label = "sig" if py_sig else "ns"
    assert r_sig == py_sig, (
        f"Significance mismatch: "
        f"R p={r_output['p_value']:.4f} ({r_label}), "
        f"Py p={py_stats['p_value']:.4f} ({py_label})"
    )


# module-scope cache: run Gibbs once per scenario
_cache: dict[str, tuple[dict, dict]] = {}


def _get_scenario(scenario: str) -> tuple[dict, dict]:
    if scenario not in _cache:
        fixture = _load_fixture(scenario)
        py_stats = _run_causal_impact(fixture)
        _cache[scenario] = (fixture, py_stats)
    return _cache[scenario]


class TestEquivalenceBasic:
    SCENARIO = "basic"

    def test_point_effect_mean(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["point_effect_mean"],
            r["point_effect_mean"],
            TOL_POINT,
            "point_effect_mean",
        )

    def test_ci_lower(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_lower"], r["ci_lower"], TOL_CI_NO_COV, "ci_lower")

    def test_ci_upper(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_upper"], r["ci_upper"], TOL_CI_NO_COV, "ci_upper")

    def test_cumulative_effect(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["cumulative_effect_total"],
            r["cumulative_effect_total"],
            TOL_POINT,
            "cumulative_effect_total",
        )

    def test_p_value_significance(self):
        fixture, py = _get_scenario(self.SCENARIO)
        _assert_significance(py, fixture["r_output"])


class TestEquivalenceCovariates:
    SCENARIO = "covariates"

    def test_point_effect_mean(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["point_effect_mean"],
            r["point_effect_mean"],
            TOL_POINT,
            "point_effect_mean",
        )

    def test_ci_lower(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_lower"], r["ci_lower"], TOL_CI_COV, "ci_lower")

    def test_ci_upper(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_upper"], r["ci_upper"], TOL_CI_COV, "ci_upper")

    def test_cumulative_effect(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["cumulative_effect_total"],
            r["cumulative_effect_total"],
            TOL_POINT,
            "cumulative_effect_total",
        )

    def test_p_value_significance(self):
        fixture, py = _get_scenario(self.SCENARIO)
        _assert_significance(py, fixture["r_output"])


class TestEquivalenceStrongEffect:
    SCENARIO = "strong_effect"

    def test_point_effect_mean(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["point_effect_mean"],
            r["point_effect_mean"],
            TOL_POINT,
            "point_effect_mean",
        )

    def test_ci_lower(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_lower"], r["ci_lower"], TOL_CI_NO_COV, "ci_lower")

    def test_ci_upper(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_upper"], r["ci_upper"], TOL_CI_NO_COV, "ci_upper")

    def test_cumulative_effect(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["cumulative_effect_total"],
            r["cumulative_effect_total"],
            TOL_POINT,
            "cumulative_effect_total",
        )

    def test_p_value_significance(self):
        fixture, py = _get_scenario(self.SCENARIO)
        _assert_significance(py, fixture["r_output"])


class TestEquivalenceCovariatesPhase2:
    SCENARIO = "covariates"

    def test_point_effect_mean(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["point_effect_mean"],
            r["point_effect_mean"],
            TOL_POINT,
            "point_effect_mean",
        )

    def test_ci_lower(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_lower"], r["ci_lower"], TOL_CI_PHASE2, "ci_lower")

    def test_ci_upper(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_upper"], r["ci_upper"], TOL_CI_PHASE2, "ci_upper")

    def test_cumulative_effect(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["cumulative_effect_total"],
            r["cumulative_effect_total"],
            TOL_POINT,
            "cumulative_effect_total",
        )

    def test_p_value_significance(self):
        fixture, py = _get_scenario(self.SCENARIO)
        _assert_significance(py, fixture["r_output"])


class TestEquivalenceSeasonal:
    SCENARIO = "seasonal"

    def test_point_effect_mean(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["point_effect_mean"],
            r["point_effect_mean"],
            TOL_POINT_SEASONAL,
            "point_effect_mean",
        )

    def test_ci_lower(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_lower"], r["ci_lower"], TOL_CI_SEASONAL, "ci_lower")

    def test_ci_upper(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(py["ci_upper"], r["ci_upper"], TOL_CI_SEASONAL, "ci_upper")

    def test_cumulative_effect(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_relative(
            py["cumulative_effect_total"],
            r["cumulative_effect_total"],
            TOL_POINT_SEASONAL,
            "cumulative_effect_total",
        )

    def test_p_value_significance(self):
        fixture, py = _get_scenario(self.SCENARIO)
        _assert_significance(py, fixture["r_output"])


class TestEquivalenceNoEffect:
    """no_effect: compared by absolute error since true_effect=0."""

    SCENARIO = "no_effect"

    def test_point_effect_mean(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_absolute(
            py["point_effect_mean"],
            r["point_effect_mean"],
            ABS_TOL_NO_EFFECT,
            "point_effect_mean",
        )

    def test_ci_lower(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_absolute(
            py["ci_lower"],
            r["ci_lower"],
            ABS_TOL_NO_EFFECT,
            "ci_lower",
        )

    def test_ci_upper(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_absolute(
            py["ci_upper"],
            r["ci_upper"],
            ABS_TOL_NO_EFFECT,
            "ci_upper",
        )

    def test_cumulative_effect(self):
        fixture, py = _get_scenario(self.SCENARIO)
        r = fixture["r_output"]
        _assert_absolute(
            py["cumulative_effect_total"],
            r["cumulative_effect_total"],
            ABS_TOL_NO_EFFECT,
            "cumulative_effect_total",
        )

    def test_p_value_significance(self):
        fixture, py = _get_scenario(self.SCENARIO)
        _assert_significance(py, fixture["r_output"])


class TestEquivalenceBoundary:
    def test_fixture_missing_causes_skip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tests.test_numerical_equivalence.FIXTURES_DIR",
            tmp_path,
        )
        monkeypatch.delenv("CI", raising=False)
        with pytest.raises(pytest.skip.Exception):
            _load_fixture("nonexistent_scenario")

    def test_fixture_missing_causes_fail_in_ci(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tests.test_numerical_equivalence.FIXTURES_DIR",
            tmp_path,
        )
        monkeypatch.setenv("CI", "true")
        with pytest.raises(pytest.fail.Exception):
            _load_fixture("nonexistent_scenario")

    def test_tolerance_uses_absolute_when_r_val_near_zero(self):
        _assert_relative(0.05, 0.0, 0.01, "near_zero")
        _assert_relative(-0.05, 0.0, 0.01, "near_zero_neg")
        with pytest.raises(AssertionError):
            _assert_relative(0.6, 0.0, 0.01, "should_fail")
