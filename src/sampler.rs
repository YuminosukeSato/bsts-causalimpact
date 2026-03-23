use crate::distributions::{sample_from_precision, sample_inv_gamma, sample_normal};
use crate::kalman::{dynamic_beta_smoother, local_linear_trend_smoother, simulation_smoother};
use crate::state_space::{SeasonalConfig, StateModel, StateSpaceModel};
use rand::distributions::Bernoulli;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

const SEED_PRIME: u64 = 6_364_136_223_846_793_005;
const STATIC_REGRESSION_EXPECTED_R2: f64 = 0.8;
const STATIC_REGRESSION_PRIOR_DF: f64 = 50.0;
const STATIC_REGRESSION_PRIOR_INFORMATION_WEIGHT: f64 = 0.01;
const STATIC_REGRESSION_DIAGONAL_SHRINKAGE: f64 = 0.5;
const LOCAL_LINEAR_TREND_SLOPE_SD_RATIO: f64 = 0.1;

struct StaticRegressionPrior {
    beta_mean: Vec<f64>,
    beta_precision: Vec<Vec<f64>>,
    sigma_guess: f64,
    prior_df: f64,
}

struct ChainResult {
    chain_id: usize,
    states: Vec<Vec<f64>>,
    sigma_obs: Vec<f64>,
    sigma_level: Vec<f64>,
    beta: Vec<Vec<f64>>,
    gamma: Vec<Vec<bool>>,
    predictions: Vec<Vec<f64>>,
}

pub struct GibbsResult {
    pub states: Vec<Vec<f64>>,
    pub sigma_obs: Vec<f64>,
    pub sigma_level: Vec<f64>,
    pub beta: Vec<Vec<f64>>,
    pub gamma: Vec<Vec<bool>>,
    pub predictions: Vec<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_sampler(
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
    pre_end: usize,
    niter: usize,
    nwarmup: usize,
    nchains: usize,
    seed: u64,
    prior_level_sd: f64,
    expected_model_size: f64,
    nseasons: Option<f64>,
    season_duration: Option<f64>,
    dynamic_regression: bool,
    state_model: &str,
) -> Result<GibbsResult, String> {
    validate_inputs(&y, pre_end, nchains)?;

    let k = x.len();
    if k > 0 && expected_model_size <= 0.0 && !dynamic_regression {
        return Err("expected_model_size must be > 0 when covariates are present".to_string());
    }

    let seasonal = SeasonalConfig::from_optional(nseasons, season_duration)?;
    let state_model = StateModel::from_name(state_model)?;
    let model = StateSpaceModel::new(y.clone(), x, seasonal);
    let n_samples = niter * nchains;
    let chain_results: Vec<ChainResult> = (0..nchains)
        .into_par_iter()
        .map(|chain_id| {
            run_single_chain(
                &y,
                &model,
                pre_end,
                niter,
                nwarmup,
                seed,
                prior_level_sd,
                chain_id,
                expected_model_size,
                dynamic_regression,
                state_model,
            )
        })
        .collect();

    Ok(flatten_chain_results(chain_results, n_samples))
}

fn validate_inputs(y: &[f64], pre_end: usize, nchains: usize) -> Result<(), String> {
    if y.is_empty() {
        return Err("y must not be empty".to_string());
    }
    if pre_end == 0 {
        return Err("pre_end must be at least 1".to_string());
    }
    if pre_end >= y.len() {
        return Err("pre_end must be less than length of y (no post-period)".to_string());
    }
    if nchains == 0 {
        return Err("nchains must be at least 1".to_string());
    }

    for (index, value) in y.iter().enumerate().take(pre_end) {
        if value.is_nan() {
            return Err(format!("NaN found in y at pre-period index {index}"));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_single_chain(
    y: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    niter: usize,
    nwarmup: usize,
    seed: u64,
    prior_level_sd: f64,
    chain_id: usize,
    expected_model_size: f64,
    dynamic_regression: bool,
    state_model: StateModel,
) -> ChainResult {
    if dynamic_regression {
        run_single_chain_dynamic(
            y,
            model,
            pre_end,
            niter,
            nwarmup,
            seed,
            prior_level_sd,
            chain_id,
            state_model,
        )
    } else {
        run_single_chain_static(
            y,
            model,
            pre_end,
            niter,
            nwarmup,
            seed,
            prior_level_sd,
            chain_id,
            expected_model_size,
            state_model,
        )
    }
}

/// Dynamic regression chain: time-varying β_t via multivariate FFBS.
/// Spike-and-slab is disabled; gamma is always empty.
#[allow(clippy::too_many_arguments)]
fn run_single_chain_dynamic(
    y: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    niter: usize,
    nwarmup: usize,
    seed: u64,
    prior_level_sd: f64,
    chain_id: usize,
    state_model: StateModel,
) -> ChainResult {
    let t_total = y.len();
    let k = model.num_covariates();
    let k_seasonal = model.num_seasonal_covariates();
    let y_sd = sample_standard_deviation(&y[..pre_end]);
    let seasonal_regression_prior =
        (k_seasonal > 0).then(|| build_static_regression_prior(model.seasonal_covariates(), y_sd));
    let chain_seed = seed ^ (chain_id as u64).wrapping_mul(SEED_PRIME);
    let mut rng = StdRng::seed_from_u64(chain_seed);
    let mut sigma2_obs = 1.0;
    let mut sigma2_level = (prior_level_sd * prior_level_sd).max(1e-6);
    let sigma2_slope = sample_sigma2_slope(prior_level_sd, y_sd);
    let mut seasonal_beta = vec![0.0; k_seasonal];

    // Dynamic regression state variables
    let mut beta_t: Vec<Vec<f64>> = vec![vec![0.0; k]; pre_end];
    let sigma_guess_beta = y_sd / (pre_end as f64).sqrt();
    let initial_sigma2_beta = sigma_guess_beta * sigma_guess_beta;
    let mut sigma2_beta = vec![initial_sigma2_beta.max(1e-8); k];
    let init_beta_mean = vec![0.0; k];

    // σ²_β prior: InvGamma(shape=16, scale=16 * sigma_guess_beta²)
    let sigma2_beta_prior_shape = 16.0;
    let sigma2_beta_prior_scale = sigma2_beta_prior_shape * initial_sigma2_beta;

    let mut chain_result = ChainResult {
        chain_id,
        states: Vec::with_capacity(niter),
        sigma_obs: Vec::with_capacity(niter),
        sigma_level: Vec::with_capacity(niter),
        beta: Vec::with_capacity(niter),
        gamma: Vec::with_capacity(0), // spike-and-slab disabled
        predictions: Vec::with_capacity(niter),
    };

    // Pre-compute X^TX for seasonal regression (constant across iterations).
    let xtx_seasonal = if k_seasonal > 0 {
        Some(cross_product_matrix(model.seasonal_covariates(), pre_end))
    } else {
        None
    };

    for iter in 0..(niter + nwarmup) {
        // Step 1: State sampling — subtract time-varying x'β_t from y
        let y_adj = adjusted_observations_dynamic(y, model, &beta_t, &seasonal_beta, pre_end);
        let (states, slopes) = sample_state_path(
            &mut rng,
            &y_adj,
            sigma2_obs,
            sigma2_level,
            sigma2_slope,
            y[0],
            y_sd * y_sd,
            state_model,
        );

        // Step 2: Dynamic β_t sampling via multivariate FFBS
        if k > 0 {
            let y_adj_beta: Vec<f64> = y[..pre_end]
                .iter()
                .zip(states.iter())
                .enumerate()
                .map(|(t, (&y_val, &mu))| {
                    y_val - mu - block_contribution(model.seasonal_covariates(), &seasonal_beta, t)
                })
                .collect();

            beta_t = dynamic_beta_smoother(
                &mut rng,
                &y_adj_beta,
                model.covariates(),
                sigma2_obs,
                &sigma2_beta,
                &init_beta_mean,
                1e2, // Diffuse but numerically stable (std dev = 10)
            );
        }

        // Step 3: σ²_obs sampling
        let sse_obs: f64 = y[..pre_end]
            .iter()
            .zip(states.iter())
            .enumerate()
            .map(|(t, (&y_val, &mu))| {
                let reg = if k > 0 {
                    block_contribution(model.covariates(), &beta_t[t], t)
                } else {
                    0.0
                };
                let seasonal = block_contribution(model.seasonal_covariates(), &seasonal_beta, t);
                let err = y_val - mu - reg - seasonal;
                err * err
            })
            .sum();
        let sigma_guess = sample_standard_deviation(&y[..pre_end]);
        let obs_prior_df = 0.01;
        let obs_a = obs_prior_df / 2.0;
        let obs_b = obs_a * sigma_guess * sigma_guess;
        let obs_shape = obs_a + pre_end as f64 / 2.0;
        let obs_scale = obs_b + sse_obs / 2.0;
        sigma2_obs = sample_inv_gamma(&mut rng, obs_shape, obs_scale).max(1e-12);

        // Step 4: σ²_level sampling (same as static)
        sigma2_level = sample_sigma2_level_by_state_model(
            &mut rng,
            &states,
            &slopes,
            pre_end,
            prior_level_sd,
            y_sd,
            state_model,
        );

        // Step 5: σ²_β sampling per covariate (InvGamma posterior)
        if k > 0 {
            for j in 0..k {
                let ssd: f64 = beta_t
                    .windows(2)
                    .map(|w| {
                        let d = w[1][j] - w[0][j];
                        d * d
                    })
                    .sum();
                let shape = sigma2_beta_prior_shape + (pre_end - 1) as f64 / 2.0;
                let scale = sigma2_beta_prior_scale + ssd / 2.0;
                sigma2_beta[j] = sample_inv_gamma(&mut rng, shape, scale).max(1e-12);
            }
        }

        // Seasonal beta sampling (static, same as non-dynamic path)
        if k_seasonal > 0 {
            let prior = seasonal_regression_prior
                .as_ref()
                .expect("seasonal regression prior must exist when seasonal regressors exist");
            let baseline: Vec<f64> = states
                .iter()
                .enumerate()
                .map(|(t, &mu)| {
                    mu + if k > 0 {
                        block_contribution(model.covariates(), &beta_t[t], t)
                    } else {
                        0.0
                    }
                })
                .collect();
            seasonal_beta = sample_beta_with_normal_prior(
                &mut rng,
                &y[..pre_end],
                &baseline,
                model.seasonal_covariates(),
                sigma2_obs,
                &prior.beta_mean,
                &prior.beta_precision,
                xtx_seasonal.as_ref(),
            );
        }

        if iter >= nwarmup {
            let states_post = sample_post_period_states_by_state_model(
                &mut rng,
                states[pre_end - 1],
                slopes[pre_end - 1],
                t_total - pre_end,
                sigma2_level,
                sigma2_slope,
                state_model,
            );
            let full_states = combine_state_paths(&states, &states_post);

            // Dynamic predictions: propagate β via random walk
            let beta_last = if k > 0 {
                beta_t[pre_end - 1].clone()
            } else {
                vec![]
            };
            let predictions = generate_predictions_dynamic(
                &states_post,
                &beta_last,
                &sigma2_beta,
                &seasonal_beta,
                model,
                pre_end,
                sigma2_obs,
                &mut rng,
            );

            chain_result.states.push(full_states);
            chain_result.sigma_obs.push(sigma2_obs.sqrt());
            chain_result.sigma_level.push(sigma2_level.sqrt());
            chain_result.beta.push(if k > 0 {
                beta_t[pre_end - 1].clone()
            } else {
                vec![]
            });
            // gamma stays empty for dynamic regression
            chain_result.predictions.push(predictions);
        }
    }

    chain_result
}

/// Static regression chain: original Gibbs sampler with fixed β.
#[allow(clippy::too_many_arguments)]
fn run_single_chain_static(
    y: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    niter: usize,
    nwarmup: usize,
    seed: u64,
    prior_level_sd: f64,
    chain_id: usize,
    expected_model_size: f64,
    state_model: StateModel,
) -> ChainResult {
    let t_total = y.len();
    let k = model.num_covariates();
    let k_seasonal = model.num_seasonal_covariates();
    let y_sd = sample_standard_deviation(&y[..pre_end]);
    let static_regression_prior =
        (k > 0).then(|| build_static_regression_prior(model.covariates(), y_sd));
    let seasonal_regression_prior =
        (k_seasonal > 0).then(|| build_static_regression_prior(model.seasonal_covariates(), y_sd));
    let chain_seed = seed ^ (chain_id as u64).wrapping_mul(SEED_PRIME);
    let mut rng = StdRng::seed_from_u64(chain_seed);
    let mut sigma2_obs = 1.0;
    let mut sigma2_level = (prior_level_sd * prior_level_sd).max(1e-6);
    let sigma2_slope = sample_sigma2_slope(prior_level_sd, y_sd);
    let mut beta = vec![0.0; k];
    let mut seasonal_beta = vec![0.0; k_seasonal];
    let mut gamma = vec![true; k];
    let mut chain_result = ChainResult {
        chain_id,
        states: Vec::with_capacity(niter),
        sigma_obs: Vec::with_capacity(niter),
        sigma_level: Vec::with_capacity(niter),
        beta: Vec::with_capacity(niter),
        gamma: Vec::with_capacity(niter),
        predictions: Vec::with_capacity(niter),
    };

    // Compute pi and log_prior_odds once
    let pi = if k > 0 {
        (expected_model_size / k as f64).min(1.0)
    } else {
        1.0
    };
    let use_spike_slab = k > 0 && pi < 1.0;
    let log_prior_odds = if use_spike_slab {
        (pi / (1.0 - pi)).ln()
    } else {
        0.0
    };
    let g = pre_end as f64; // Zellner's g-prior parameter

    // Pre-compute X^TX for static and seasonal regression.
    // X is constant across all Gibbs iterations, so this is computed once.
    let xtx_static = if k > 0 {
        Some(cross_product_matrix(model.covariates(), pre_end))
    } else {
        None
    };
    let xtx_seasonal = if k_seasonal > 0 {
        Some(cross_product_matrix(model.seasonal_covariates(), pre_end))
    } else {
        None
    };

    // Pre-compute spike-and-slab constant statistics per covariate.
    // x_mean and n_j = sum((x_j - x_mean)^2) are constant across iterations.
    let slab_stats: Vec<(f64, f64)> = if use_spike_slab {
        model
            .covariates()
            .iter()
            .map(|x_col| {
                let x_sum: f64 = x_col.iter().take(pre_end).sum();
                let x_mean = x_sum / pre_end as f64;
                let sum_x2: f64 = x_col.iter().take(pre_end).map(|v| v * v).sum();
                let n_j = sum_x2 - pre_end as f64 * x_mean * x_mean;
                (x_mean, n_j)
            })
            .collect()
    } else {
        vec![]
    };

    for iter in 0..(niter + nwarmup) {
        // Step 1: State sampling (simulation smoother)
        let y_adj = adjusted_observations(y, model, &beta, &seasonal_beta, pre_end);
        let (states, slopes) = sample_state_path(
            &mut rng,
            &y_adj,
            sigma2_obs,
            sigma2_level,
            sigma2_slope,
            y[0],
            y_sd * y_sd,
            state_model,
        );

        // Step 2-3: (sigma2_obs, gamma, beta) sampling
        // Gibbs ordering depends on whether spike-and-slab is active:
        // - spike-and-slab: sample sigma2_obs BEFORE beta to prevent
        //   sigma2_obs spikes when beta jumps from 0 (excluded) to nonzero (included)
        // - blocked g-prior (pi>=1.0): sample beta BEFORE sigma2_obs
        //   (original ordering, preserves backward compatibility)
        if k > 0 && use_spike_slab {
            sigma2_obs = sample_sigma2_obs(
                &mut rng,
                &y[..pre_end],
                &states,
                model,
                &beta,
                &seasonal_beta,
                y_sd,
                0.01,
            );

            let mut residual = user_residual(&y[..pre_end], &states, model, &beta, &seasonal_beta);

            sample_spike_and_slab(
                &mut rng,
                &mut gamma,
                &mut beta,
                &mut residual,
                model.covariates(),
                k,
                pre_end,
                sigma2_obs,
                g,
                log_prior_odds,
                &slab_stats,
            );
            if k_seasonal > 0 {
                let prior = seasonal_regression_prior
                    .as_ref()
                    .expect("seasonal regression prior must exist when seasonal regressors exist");
                let baseline = baseline_from_state_and_block(&states, model.covariates(), &beta);
                seasonal_beta = sample_beta_with_normal_prior(
                    &mut rng,
                    &y[..pre_end],
                    &baseline,
                    model.seasonal_covariates(),
                    sigma2_obs,
                    &prior.beta_mean,
                    &prior.beta_precision,
                    xtx_seasonal.as_ref(),
                );
            }
        } else if k > 0 || k_seasonal > 0 {
            if k > 0 {
                let prior = static_regression_prior
                    .as_ref()
                    .expect("static regression prior must exist when k > 0");
                let baseline = baseline_from_state_and_block(
                    &states,
                    model.seasonal_covariates(),
                    &seasonal_beta,
                );
                beta = sample_beta_with_normal_prior(
                    &mut rng,
                    &y[..pre_end],
                    &baseline,
                    model.covariates(),
                    sigma2_obs,
                    &prior.beta_mean,
                    &prior.beta_precision,
                    xtx_static.as_ref(),
                );
                for gj in gamma.iter_mut() {
                    *gj = true;
                }
            }
            if k_seasonal > 0 {
                let prior = seasonal_regression_prior
                    .as_ref()
                    .expect("seasonal regression prior must exist when seasonal regressors exist");
                let baseline = baseline_from_state_and_block(&states, model.covariates(), &beta);
                seasonal_beta = sample_beta_with_normal_prior(
                    &mut rng,
                    &y[..pre_end],
                    &baseline,
                    model.seasonal_covariates(),
                    sigma2_obs,
                    &prior.beta_mean,
                    &prior.beta_precision,
                    xtx_seasonal.as_ref(),
                );
            }
            let sigma_guess = static_regression_prior
                .as_ref()
                .map(|prior| prior.sigma_guess)
                .or_else(|| {
                    seasonal_regression_prior
                        .as_ref()
                        .map(|prior| prior.sigma_guess)
                })
                .unwrap_or(y_sd);
            let prior_df = static_regression_prior
                .as_ref()
                .map(|prior| prior.prior_df)
                .or_else(|| {
                    seasonal_regression_prior
                        .as_ref()
                        .map(|prior| prior.prior_df)
                })
                .unwrap_or(0.01);
            sigma2_obs = sample_sigma2_obs(
                &mut rng,
                &y[..pre_end],
                &states,
                model,
                &beta,
                &seasonal_beta,
                sigma_guess,
                prior_df,
            );
        } else {
            let sigma_guess = sample_standard_deviation(&y[..pre_end]);
            sigma2_obs = sample_sigma2_obs(
                &mut rng,
                &y[..pre_end],
                &states,
                model,
                &beta,
                &seasonal_beta,
                sigma_guess,
                0.01,
            );
        }

        // Step 4: sigma^2_level sampling
        sigma2_level = sample_sigma2_level_by_state_model(
            &mut rng,
            &states,
            &slopes,
            pre_end,
            prior_level_sd,
            y_sd,
            state_model,
        );

        if iter >= nwarmup {
            let states_post = sample_post_period_states_by_state_model(
                &mut rng,
                states[pre_end - 1],
                slopes[pre_end - 1],
                t_total - pre_end,
                sigma2_level,
                sigma2_slope,
                state_model,
            );
            let full_states = combine_state_paths(&states, &states_post);
            let predictions = generate_predictions(
                &states_post,
                &beta,
                &seasonal_beta,
                model,
                pre_end,
                sigma2_obs,
                &mut rng,
            );

            chain_result.states.push(full_states);
            chain_result.sigma_obs.push(sigma2_obs.sqrt());
            chain_result.sigma_level.push(sigma2_level.sqrt());
            chain_result.beta.push(beta.clone());
            if k > 0 {
                chain_result.gamma.push(gamma.clone());
            }
            chain_result.predictions.push(predictions);
        }
    }

    chain_result
}

fn flatten_chain_results(mut chain_results: Vec<ChainResult>, n_samples: usize) -> GibbsResult {
    chain_results.sort_unstable_by_key(|chain_result| chain_result.chain_id);

    let mut result = GibbsResult {
        states: Vec::with_capacity(n_samples),
        sigma_obs: Vec::with_capacity(n_samples),
        sigma_level: Vec::with_capacity(n_samples),
        beta: Vec::with_capacity(n_samples),
        gamma: Vec::with_capacity(n_samples),
        predictions: Vec::with_capacity(n_samples),
    };

    for chain_result in chain_results {
        result.states.extend(chain_result.states);
        result.sigma_obs.extend(chain_result.sigma_obs);
        result.sigma_level.extend(chain_result.sigma_level);
        result.beta.extend(chain_result.beta);
        result.gamma.extend(chain_result.gamma);
        result.predictions.extend(chain_result.predictions);
    }

    result
}

fn sample_state_path<R: rand::Rng>(
    rng: &mut R,
    y_adj: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    sigma2_slope: f64,
    initial_state_mean: f64,
    initial_state_variance: f64,
    state_model: StateModel,
) -> (Vec<f64>, Vec<f64>) {
    match state_model {
        StateModel::LocalLevel => (
            simulation_smoother(
                rng,
                y_adj,
                sigma2_obs,
                sigma2_level,
                initial_state_mean,
                initial_state_variance,
            ),
            vec![0.0; y_adj.len()],
        ),
        StateModel::LocalLinearTrend => local_linear_trend_smoother(
            rng,
            y_adj,
            sigma2_obs,
            sigma2_level,
            sigma2_slope,
            initial_state_mean,
            initial_state_variance,
        ),
    }
}

fn sample_sigma2_level_by_state_model<R: rand::Rng>(
    rng: &mut R,
    states: &[f64],
    slopes: &[f64],
    pre_end: usize,
    prior_level_sd: f64,
    y_sd: f64,
    state_model: StateModel,
) -> f64 {
    match state_model {
        StateModel::LocalLevel => sample_sigma2_level(rng, states, pre_end, prior_level_sd, y_sd),
        StateModel::LocalLinearTrend => {
            sample_sigma2_trend_level(rng, states, slopes, pre_end, prior_level_sd, y_sd)
        }
    }
}

fn sample_post_period_states_by_state_model<R: rand::Rng>(
    rng: &mut R,
    last_pre_state: f64,
    last_pre_slope: f64,
    post_len: usize,
    sigma2_level: f64,
    sigma2_slope: f64,
    state_model: StateModel,
) -> Vec<f64> {
    match state_model {
        StateModel::LocalLevel => {
            sample_post_period_states(rng, last_pre_state, post_len, sigma2_level)
        }
        StateModel::LocalLinearTrend => sample_post_period_trend_states(
            rng,
            last_pre_state,
            last_pre_slope,
            post_len,
            sigma2_level,
            sigma2_slope,
        ),
    }
}

fn adjusted_observations(
    y: &[f64],
    model: &StateSpaceModel,
    beta: &[f64],
    seasonal_beta: &[f64],
    pre_end: usize,
) -> Vec<f64> {
    y.iter()
        .take(pre_end)
        .enumerate()
        .map(|(t, y_value)| y_value - model.regression_contribution(t, beta, seasonal_beta))
        .collect()
}

/// Adjusted observations for dynamic regression: subtract time-varying x'β_t.
fn adjusted_observations_dynamic(
    y: &[f64],
    model: &StateSpaceModel,
    beta_t: &[Vec<f64>],
    seasonal_beta: &[f64],
    pre_end: usize,
) -> Vec<f64> {
    y.iter()
        .take(pre_end)
        .enumerate()
        .map(|(t, y_value)| {
            let dynamic_reg = if !beta_t.is_empty() && !beta_t[0].is_empty() {
                block_contribution(model.covariates(), &beta_t[t], t)
            } else {
                0.0
            };
            let seasonal_reg = block_contribution(model.seasonal_covariates(), seasonal_beta, t);
            y_value - dynamic_reg - seasonal_reg
        })
        .collect()
}

fn baseline_from_state_and_block(states: &[f64], x: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    states
        .iter()
        .enumerate()
        .map(|(t, state)| state + block_contribution(x, beta, t))
        .collect()
}

fn user_residual(
    y_pre: &[f64],
    states: &[f64],
    model: &StateSpaceModel,
    beta: &[f64],
    seasonal_beta: &[f64],
) -> Vec<f64> {
    y_pre
        .iter()
        .zip(states.iter())
        .enumerate()
        .map(|(t, (y_value, state))| {
            y_value - state - model.regression_contribution(t, beta, seasonal_beta)
        })
        .collect()
}

fn block_contribution(x: &[Vec<f64>], beta: &[f64], t: usize) -> f64 {
    x.iter()
        .zip(beta.iter())
        .map(|(x_col, beta_value)| x_col[t] * beta_value)
        .sum::<f64>()
}

#[allow(clippy::too_many_arguments)]
fn sample_beta_with_normal_prior<R: rand::Rng>(
    rng: &mut R,
    y_pre: &[f64],
    baseline: &[f64],
    x: &[Vec<f64>],
    sigma2_obs: f64,
    prior_mean: &[f64],
    prior_precision: &[Vec<f64>],
    xtx_precomputed: Option<&Vec<Vec<f64>>>,
) -> Vec<f64> {
    let k = x.len();
    // Use pre-computed X^TX if available (avoids O(k^2 T) per iteration)
    let xtx = match xtx_precomputed {
        Some(pre) => pre.clone(),
        None => cross_product_matrix(x, y_pre.len()),
    };

    let residuals: Vec<f64> = y_pre
        .iter()
        .zip(baseline.iter())
        .map(|(y_value, baseline_value)| y_value - baseline_value)
        .collect();

    let mut xtr = vec![0.0; k];
    for (j, x_col) in x.iter().enumerate() {
        xtr[j] = x_col
            .iter()
            .zip(residuals.iter())
            .take(y_pre.len())
            .map(|(x_value, residual)| x_value * residual)
            .sum();
    }

    let mut posterior_precision = xtx;
    for (i, row) in posterior_precision.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value += prior_precision[i][j];
        }
    }

    let mut rhs = xtr;
    let prior_precision_times_mean = matrix_vector_product(prior_precision, prior_mean);
    for (value, prior_value) in rhs.iter_mut().zip(prior_precision_times_mean.iter()) {
        *value += prior_value;
    }

    // Sample from N(A^{-1}b, sigma2 * A^{-1}) via Cholesky of precision.
    // Replaces Gauss-Jordan inversion + separate mvnormal sampling.
    sample_from_precision(rng, &posterior_precision, &rhs, sigma2_obs)
}

/// Coordinate-wise spike-and-slab sampling for (gamma, beta).
///
/// For each covariate j:
///   1. Add back x_j * beta_j to get partial residual r_j
///   2. Compute log Bayes factor for inclusion using centered statistics
///   3. Sample gamma_j from Bernoulli(sigmoid(log_prior_odds + log_bf))
///   4. If included: sample beta_j from posterior; otherwise set beta_j = 0
///   5. Subtract x_j * new_beta_j from residual
///
/// Uses centered sum of squares: n_j = Σ(x_j - x̄_j)²
/// This ensures the Bayes factor is invariant to the location of x_j,
/// matching R's bsts which standardizes covariates before spike-and-slab.
#[allow(clippy::too_many_arguments)]
fn sample_spike_and_slab<R: rand::Rng>(
    rng: &mut R,
    gamma: &mut [bool],
    beta: &mut [f64],
    residual: &mut [f64],
    x: &[Vec<f64>],
    k: usize,
    t_pre: usize,
    sigma2_obs: f64,
    g: f64,
    log_prior_odds: f64,
    precomputed_stats: &[(f64, f64)],
) {
    let one_plus_g = 1.0 + g;
    let log_shrinkage = -0.5 * one_plus_g.ln(); // 0.5 * log(1/(1+g))

    for j in 0..k {
        let x_col = &x[j];
        let old_beta_j = beta[j];

        // Rank-1 update: add back old contribution to get partial residual
        for (t, r) in residual.iter_mut().enumerate().take(t_pre) {
            *r += x_col[t] * old_beta_j;
        }

        // Use pre-computed centered statistics (x_mean, n_j) for O(1) lookup
        let (x_mean, n_j) = precomputed_stats[j];

        // Guard against zero/near-zero variance covariates
        if n_j < 1e-12 {
            gamma[j] = false;
            beta[j] = 0.0;
            for (t, r) in residual.iter_mut().enumerate().take(t_pre) {
                *r -= x_col[t] * beta[j];
            }
            continue;
        }

        // Centered cross-product: Σ(x_j - x̄) * r = Σ x_j*r - x̄ * Σr
        let xtr_raw: f64 = x_col
            .iter()
            .zip(residual.iter())
            .take(t_pre)
            .map(|(x_val, r_val)| x_val * r_val)
            .sum();
        let r_sum: f64 = residual.iter().take(t_pre).sum();
        let xtr_j = xtr_raw - x_mean * r_sum;

        // Log Bayes factor: log_shrinkage + 0.5 * (x̃_j^T r_j)² / (ñ_j * σ²_obs * (1+g))
        let log_bf = log_shrinkage + 0.5 * xtr_j * xtr_j / (n_j * sigma2_obs * one_plus_g);

        // Log odds of inclusion
        let log_odds = log_prior_odds + log_bf;

        // Stable sigmoid: avoid exp overflow
        let include = if log_odds > 40.0 {
            true
        } else if log_odds < -40.0 {
            false
        } else {
            let prob = 1.0 / (1.0 + (-log_odds).exp());
            let dist = Bernoulli::new(prob).unwrap_or_else(|_| Bernoulli::new(0.5).unwrap());
            rng.sample(dist)
        };

        gamma[j] = include;

        if include {
            // Posterior: beta_j | gamma_j=1 ~ N(mu_j, sigma2_j)
            // Using centered statistics for the slab
            let mu_j = xtr_j * g / (n_j * one_plus_g);
            let sigma2_j = sigma2_obs * g / (n_j * one_plus_g);
            beta[j] = sample_normal(rng, mu_j, sigma2_j);
        } else {
            beta[j] = 0.0;
        }

        // Rank-1 update: subtract new contribution
        for (t, r) in residual.iter_mut().enumerate().take(t_pre) {
            *r -= x_col[t] * beta[j];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_sigma2_obs<R: rand::Rng>(
    rng: &mut R,
    y_pre: &[f64],
    states: &[f64],
    model: &StateSpaceModel,
    beta: &[f64],
    seasonal_beta: &[f64],
    sigma_guess: f64,
    prior_df: f64,
) -> f64 {
    let a_prior = prior_df / 2.0;
    let b_prior = a_prior * sigma_guess * sigma_guess;

    let sse = y_pre
        .iter()
        .zip(states.iter())
        .enumerate()
        .map(|(t, (y_value, state))| {
            let fitted = state + model.regression_contribution(t, beta, seasonal_beta);
            let error = y_value - fitted;
            error * error
        })
        .sum::<f64>();

    let shape = a_prior + y_pre.len() as f64 / 2.0;
    let scale = b_prior + sse / 2.0;
    sample_inv_gamma(rng, shape, scale).max(1e-12)
}

fn build_static_regression_prior(x: &[Vec<f64>], y_sd: f64) -> StaticRegressionPrior {
    let normalized_xtx = normalized_cross_product_matrix(x);
    let diagonal_xtx = diagonal_matrix(&normalized_xtx);
    let mut beta_precision = vec![vec![0.0; x.len()]; x.len()];

    for (i, row) in beta_precision.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let diagonal_component = diagonal_xtx[i][j];
            let dense_component = normalized_xtx[i][j];
            *value = STATIC_REGRESSION_PRIOR_INFORMATION_WEIGHT
                * (STATIC_REGRESSION_DIAGONAL_SHRINKAGE * diagonal_component
                    + (1.0 - STATIC_REGRESSION_DIAGONAL_SHRINKAGE) * dense_component);
        }
    }

    StaticRegressionPrior {
        beta_mean: vec![0.0; x.len()],
        beta_precision,
        sigma_guess: (1.0 - STATIC_REGRESSION_EXPECTED_R2).sqrt() * y_sd,
        prior_df: STATIC_REGRESSION_PRIOR_DF,
    }
}

fn sample_sigma2_level<R: rand::Rng>(
    rng: &mut R,
    states: &[f64],
    pre_end: usize,
    prior_level_sd: f64,
    y_sd: f64,
) -> f64 {
    let sigma_guess = prior_level_sd * y_sd;
    let sample_size = 32.0;
    let a_prior = sample_size / 2.0;
    let b_prior = a_prior * sigma_guess * sigma_guess;
    let ssd = states[..pre_end]
        .windows(2)
        .map(|window| {
            let diff = window[1] - window[0];
            diff * diff
        })
        .sum::<f64>();

    let shape = a_prior + (pre_end - 1) as f64 / 2.0;
    let scale = b_prior + ssd / 2.0;
    sample_inv_gamma(rng, shape, scale).max(1e-12)
}

fn sample_sigma2_trend_level<R: rand::Rng>(
    rng: &mut R,
    states: &[f64],
    slopes: &[f64],
    pre_end: usize,
    prior_level_sd: f64,
    y_sd: f64,
) -> f64 {
    let sigma_guess = prior_level_sd * y_sd;
    let sample_size = 32.0;
    let a_prior = sample_size / 2.0;
    let b_prior = a_prior * sigma_guess * sigma_guess;
    let ssd = states[..pre_end]
        .windows(2)
        .zip(slopes[..pre_end.saturating_sub(1)].iter())
        .map(|(window, slope)| {
            let innovation = window[1] - window[0] - slope;
            innovation * innovation
        })
        .sum::<f64>();

    let shape = a_prior + (pre_end - 1) as f64 / 2.0;
    let scale = b_prior + ssd / 2.0;
    sample_inv_gamma(rng, shape, scale).max(1e-12)
}

fn sample_sigma2_slope(prior_level_sd: f64, y_sd: f64) -> f64 {
    let slope_sd = (prior_level_sd * y_sd * LOCAL_LINEAR_TREND_SLOPE_SD_RATIO).max(1e-6);
    slope_sd * slope_sd
}

fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 1.0;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let centered_sum_of_squares = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>();
    let variance = centered_sum_of_squares / (n - 1.0);
    if variance <= 0.0 {
        1.0
    } else {
        variance.sqrt()
    }
}

fn cross_product_matrix(x: &[Vec<f64>], t: usize) -> Vec<Vec<f64>> {
    let k = x.len();
    let mut xtx = vec![vec![0.0; k]; k];
    for (i, row) in xtx.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = x[i]
                .iter()
                .zip(x[j].iter())
                .take(t)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum();
        }
    }
    xtx
}

fn normalized_cross_product_matrix(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x.first().map_or(1, Vec::len) as f64;
    let mut xtx = cross_product_matrix(x, x.first().map_or(0, Vec::len));
    for row in &mut xtx {
        for value in row {
            *value /= n;
        }
    }
    xtx
}

fn diagonal_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = matrix.len();
    let mut diagonal = vec![vec![0.0; k]; k];
    for (i, row) in diagonal.iter_mut().enumerate() {
        row[i] = matrix[i][i];
    }
    diagonal
}

fn matrix_vector_product(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum()
        })
        .collect()
}



fn sample_post_period_states<R: rand::Rng>(
    rng: &mut R,
    last_pre_state: f64,
    post_len: usize,
    sigma2_level: f64,
) -> Vec<f64> {
    let mut current_state = last_pre_state;
    (0..post_len)
        .map(|_| {
            current_state += sample_normal(rng, 0.0, sigma2_level);
            current_state
        })
        .collect()
}

fn sample_post_period_trend_states<R: rand::Rng>(
    rng: &mut R,
    last_pre_state: f64,
    last_pre_slope: f64,
    post_len: usize,
    sigma2_level: f64,
    sigma2_slope: f64,
) -> Vec<f64> {
    let mut current_state = last_pre_state;
    let mut current_slope = last_pre_slope;

    (0..post_len)
        .map(|_| {
            current_state += current_slope + sample_normal(rng, 0.0, sigma2_level);
            current_slope += sample_normal(rng, 0.0, sigma2_slope);
            current_state
        })
        .collect()
}

fn combine_state_paths(states_pre: &[f64], states_post: &[f64]) -> Vec<f64> {
    let mut full_states = Vec::with_capacity(states_pre.len() + states_post.len());
    full_states.extend_from_slice(states_pre);
    full_states.extend_from_slice(states_post);
    full_states
}

fn generate_predictions<R: rand::Rng>(
    states_post: &[f64],
    beta: &[f64],
    seasonal_beta: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    sigma2_obs: f64,
    rng: &mut R,
) -> Vec<f64> {
    states_post
        .iter()
        .enumerate()
        .map(|(offset, state)| {
            let t = pre_end + offset;
            model.observe(t, *state, beta, seasonal_beta) + sample_normal(rng, 0.0, sigma2_obs)
        })
        .collect()
}

/// Generate post-period predictions with dynamic β propagation.
/// β_{T+j} = β_{T+j-1} + N(0, diag(σ²_β)) — random walk continues.
#[allow(clippy::too_many_arguments)]
fn generate_predictions_dynamic<R: rand::Rng>(
    states_post: &[f64],
    beta_last: &[f64],
    sigma2_beta: &[f64],
    seasonal_beta: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    sigma2_obs: f64,
    rng: &mut R,
) -> Vec<f64> {
    let k = beta_last.len();
    let mut beta_current = beta_last.to_vec();

    states_post
        .iter()
        .enumerate()
        .map(|(offset, &state)| {
            let t = pre_end + offset;
            // Propagate β via random walk
            for j in 0..k {
                beta_current[j] += sample_normal(rng, 0.0, sigma2_beta[j]);
            }
            let dynamic_reg = block_contribution(model.covariates(), &beta_current, t);
            let seasonal_reg = block_contribution(model.seasonal_covariates(), seasonal_beta, t);
            state + dynamic_reg + seasonal_reg + sample_normal(rng, 0.0, sigma2_obs)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_sampler_basic() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.1 * i as f64).collect();
        let result = run_sampler(
            y,
            vec![],
            15,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            false,
            "local_level",
        )
        .unwrap();
        assert_eq!(result.states.len(), 10);
        assert_eq!(result.sigma_obs.len(), 10);
        assert_eq!(result.predictions.len(), 10);
        assert_eq!(result.predictions[0].len(), 5);
        assert!(result.gamma.is_empty() || result.gamma[0].is_empty());
    }

    #[test]
    fn test_run_sampler_with_covariates() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.5 * i as f64).collect();
        let x1: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let result = run_sampler(
            y,
            vec![x1],
            15,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            false,
            "local_level",
        )
        .unwrap();
        assert_eq!(result.beta.len(), 10);
        assert_eq!(result.beta[0].len(), 1);
        assert_eq!(result.gamma.len(), 10);
        assert_eq!(result.gamma[0].len(), 1);
    }

    #[test]
    fn test_run_sampler_empty_y() {
        let result = run_sampler(
            vec![],
            vec![],
            0,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            false,
            "local_level",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_sampler_pre_end_equals_t() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = run_sampler(
            y,
            vec![],
            3,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            false,
            "local_level",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_sampler_expected_model_size_validation() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.1 * i as f64).collect();
        let x1: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        // k>0 with expected_model_size=0 should fail (static only)
        let result = run_sampler(
            y.clone(),
            vec![x1.clone()],
            15,
            10,
            5,
            1,
            42,
            0.01,
            0.0,
            None,
            None,
            false,
            "local_level",
        );
        assert!(result.is_err());
        // k>0 with negative should fail
        let result = run_sampler(
            y,
            vec![x1],
            15,
            10,
            5,
            1,
            42,
            0.01,
            -1.0,
            None,
            None,
            false,
            "local_level",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_standard_deviation_uses_sample_scale_because_prior_must_match_python_standardization(
    ) {
        let values = vec![1.0, 2.0, 3.0];
        let sample_sd = sample_standard_deviation(&values);
        assert!((sample_sd - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sample_post_period_states_moves_after_pre_period_because_random_walk_noise_must_propagate(
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let states_post = sample_post_period_states(&mut rng, 10.0, 5, 0.1);
        let rounded_unique_states: std::collections::BTreeSet<i64> = states_post
            .iter()
            .map(|state| (state * 1_000_000_000.0).round() as i64)
            .collect();
        assert!(rounded_unique_states.len() > 1);
    }

    #[test]
    fn test_build_static_regression_prior_matches_r_spike_slab_formula() {
        let x = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 1.0, 0.0, -1.0]];
        let prior = build_static_regression_prior(&x, 1.0);

        assert!((prior.sigma_guess - (1.0 - 0.8_f64).sqrt()).abs() < 1e-12);
        assert!((prior.prior_df - 50.0).abs() < 1e-12);

        let normalized_xtx = normalized_cross_product_matrix(&x);
        let expected_00 = 0.01 * normalized_xtx[0][0];
        let expected_11 = 0.01 * normalized_xtx[1][1];
        let expected_01 = 0.005 * normalized_xtx[0][1];

        assert!((prior.beta_precision[0][0] - expected_00).abs() < 1e-12);
        assert!((prior.beta_precision[1][1] - expected_11).abs() < 1e-12);
        assert!((prior.beta_precision[0][1] - expected_01).abs() < 1e-12);
        assert!((prior.beta_precision[1][0] - expected_01).abs() < 1e-12);
    }

    #[test]
    fn test_run_sampler_dynamic_regression_basic() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.5 * i as f64).collect();
        let x1: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let result = run_sampler(
            y,
            vec![x1],
            15,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            true,
            "local_level",
        )
        .unwrap();
        assert_eq!(result.predictions.len(), 10);
        assert_eq!(result.predictions[0].len(), 5);
        // gamma should be empty (spike-and-slab disabled)
        assert!(result.gamma.is_empty());
    }

    #[test]
    fn test_run_sampler_dynamic_regression_k0_same_as_static() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.1 * i as f64).collect();
        let result_dyn = run_sampler(
            y.clone(),
            vec![],
            15,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            true,
            "local_level",
        )
        .unwrap();
        let result_static = run_sampler(
            y,
            vec![],
            15,
            10,
            5,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            false,
            "local_level",
        )
        .unwrap();
        // With k=0, both paths should produce identical predictions
        assert_eq!(
            result_dyn.predictions.len(),
            result_static.predictions.len()
        );
    }

    #[test]
    fn test_run_sampler_dynamic_regression_predictions_finite() {
        let y: Vec<f64> = (0..30).map(|i| 5.0 + (i as f64 * 0.1).sin()).collect();
        let x1: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).cos()).collect();
        let result = run_sampler(
            y,
            vec![x1],
            20,
            20,
            10,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            true,
            "local_level",
        )
        .unwrap();
        for pred_row in &result.predictions {
            for &val in pred_row {
                assert!(val.is_finite(), "Non-finite prediction: {}", val);
            }
        }
    }

    #[test]
    fn test_run_sampler_local_linear_trend_predictions_follow_post_period_length() {
        let y: Vec<f64> = (0..24).map(|i| 10.0 + 0.3 * i as f64).collect();
        let result = run_sampler(
            y,
            vec![],
            18,
            20,
            10,
            1,
            42,
            0.01,
            1.0,
            None,
            None,
            false,
            "local_linear_trend",
        )
        .unwrap();

        assert_eq!(result.predictions.len(), 20);
        assert_eq!(result.predictions[0].len(), 6);
    }
}
