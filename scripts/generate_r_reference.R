#!/usr/bin/env Rscript
# Generate R CausalImpact reference data for numerical equivalence tests.
#
# Usage:
#   Rscript scripts/generate_r_reference.R
#
# Output:
#   tests/fixtures/r_reference_{scenario}.json (4 files)
#
# Why: Python Gibbs sampler (Rust) and R bsts use the same algorithm.
#      By running both on identical data, we prove numerical equivalence.

library(CausalImpact)
library(jsonlite)

output_dir <- file.path("tests", "fixtures")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Scenario definitions
# Each scenario generates data with known properties.
# Python test loads this exact data (no regeneration) to eliminate RNG differences.
scenarios <- list(
  basic = list(
    seed = 42, effect = 3.0, noise = 1.0, k = 0,
    n = 100, n_pre = 70
  ),
  covariates = list(
    seed = 42, effect = 3.0, noise = 1.0, k = 2,
    n = 100, n_pre = 70
  ),
  strong_effect = list(
    seed = 42, effect = 8.0, noise = 0.5, k = 0,
    n = 100, n_pre = 70
  ),
  no_effect = list(
    seed = 42, effect = 0.0, noise = 1.0, k = 0,
    n = 100, n_pre = 70
  ),
  seasonal = list(
    seed = 7, effect = 3.0, noise = 0.3, k = 0,
    n = 112, n_pre = 84, nseasons = 7, season_duration = 1
  )
)

for (name in names(scenarios)) {
  s <- scenarios[[name]]
  set.seed(s$seed)

  n <- s$n
  n_pre <- s$n_pre
  t_seq <- 1:n

  # Data generation: y = 1.0 + effect*(t > n_pre) + N(0, noise^2) + covariates
  noise <- rnorm(n, 0, s$noise)
  y <- rep(1.0, n) + noise

  if (!is.null(s$nseasons)) {
    season_levels <- ((1:s$nseasons) - mean(1:s$nseasons)) * 0.8
    seasonal_pattern <- rep(season_levels, each = s$season_duration)
    y <- y + rep(seasonal_pattern, length.out = n)
  }

  y[(n_pre + 1):n] <- y[(n_pre + 1):n] + s$effect

  x_data <- NULL
  if (s$k > 0) {
    x_data <- list()
    for (j in 1:s$k) {
      set.seed(s$seed + j)
      xj <- rnorm(n, 0, 1)
      coeff <- 0.5 / j  # coefficients: 0.5, 0.25, ...
      y <- y + coeff * xj
      x_data[[paste0("x", j)]] <- xj
    }
  }

  # Build data frame for CausalImpact
  if (is.null(x_data)) {
    df <- data.frame(y = y)
  } else {
    df <- data.frame(y = y)
    for (xname in names(x_data)) {
      df[[xname]] <- x_data[[xname]]
    }
  }

  # Run R CausalImpact
  pre_period <- c(1, n_pre)
  post_period <- c(n_pre + 1, n)

  model_args <- list(niter = 5000, prior.level.sd = 0.01)
  if (!is.null(s$nseasons)) {
    model_args$nseasons <- s$nseasons
    model_args$season.duration <- s$season_duration
  }

  ci <- CausalImpact(
    df, pre_period, post_period,
    model.args = model_args
  )

  # Extract summary statistics
  s_table <- ci$summary
  avg_row <- s_table["Average", ]
  cum_row <- s_table["Cumulative", ]

  r_output <- list(
    point_effect_mean = as.numeric(avg_row["AbsEffect"]),
    ci_lower = as.numeric(avg_row["AbsEffect.lower"]),
    ci_upper = as.numeric(avg_row["AbsEffect.upper"]),
    cumulative_effect_total = as.numeric(cum_row["AbsEffect"]),
    relative_effect_mean = as.numeric(avg_row["RelEffect"]),
    p_value = as.numeric(ci$summary["Average", "p"])
  )

  # Build fixture JSON
  fixture <- list(
    scenario = name,
    n = n,
    n_pre = n_pre,
    seed = s$seed,
    true_effect = s$effect,
    noise_sd = s$noise,
    k = s$k,
    model_args = model_args[names(model_args) != "niter" & names(model_args) != "prior.level.sd"],
    data = list(
      y = as.numeric(y),
      x = x_data
    ),
    r_output = r_output
  )

  # Write JSON
  json_path <- file.path(output_dir, paste0("r_reference_", name, ".json"))
  write_json(fixture, json_path, pretty = TRUE, auto_unbox = TRUE, digits = 10)
  cat(sprintf("Generated: %s\n", json_path))
  cat(sprintf("  point_effect_mean: %.6f\n", r_output$point_effect_mean))
  cat(sprintf("  cumulative_effect: %.6f\n", r_output$cumulative_effect_total))
  cat(sprintf("  ci: [%.6f, %.6f]\n", r_output$ci_lower, r_output$ci_upper))
  cat(sprintf("  relative_effect:  %.6f\n", r_output$relative_effect_mean))
  cat(sprintf("  p_value:          %.6f\n", r_output$p_value))
  cat("\n")
}

cat("Done. All fixtures written to tests/fixtures/\n")
