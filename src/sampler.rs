use crate::distributions::{sample_inv_gamma, sample_mvnormal, sample_normal};
use crate::kalman::simulation_smoother;
use crate::state_space::StateSpaceModel;
use rand::distributions::Bernoulli;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

const SEED_PRIME: u64 = 6_364_136_223_846_793_005;

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
) -> Result<GibbsResult, String> {
    validate_inputs(&y, pre_end, nchains)?;

    let k = x.len();
    if k > 0 && expected_model_size <= 0.0 {
        return Err(
            "expected_model_size must be > 0 when covariates are present".to_string(),
        );
    }

    let model = StateSpaceModel::new(y.clone(), x);
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
) -> ChainResult {
    let t_total = y.len();
    let k = model.num_covariates();
    let y_sd = sample_standard_deviation(&y[..pre_end]);
    let chain_seed = seed ^ (chain_id as u64).wrapping_mul(SEED_PRIME);
    let mut rng = StdRng::seed_from_u64(chain_seed);
    let mut sigma2_obs = 1.0;
    let mut sigma2_level = (prior_level_sd * prior_level_sd).max(1e-6);
    let mut beta = vec![0.0; k];
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

    for iter in 0..(niter + nwarmup) {
        // Step 1: State sampling (simulation smoother)
        let y_adj = adjusted_observations(y, model, &beta, pre_end);
        let states = simulation_smoother(&mut rng, &y_adj, sigma2_obs, sigma2_level);

        // Step 2-3: (sigma2_obs, gamma, beta) sampling
        // Gibbs ordering depends on whether spike-and-slab is active:
        // - spike-and-slab: sample sigma2_obs BEFORE beta to prevent
        //   sigma2_obs spikes when beta jumps from 0 (excluded) to nonzero (included)
        // - blocked g-prior (pi>=1.0): sample beta BEFORE sigma2_obs
        //   (original ordering, preserves backward compatibility)
        if k > 0 && use_spike_slab {
            sigma2_obs =
                sample_sigma2_obs(&mut rng, &y[..pre_end], &states, &beta, model.covariates());

            let mut residual: Vec<f64> = y[..pre_end]
                .iter()
                .zip(states.iter())
                .map(|(y_val, s)| y_val - s)
                .collect();
            for j in 0..k {
                let x_col = &model.covariates()[j];
                for (t, r) in residual.iter_mut().enumerate().take(pre_end) {
                    *r -= x_col[t] * beta[j];
                }
            }

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
            );
        } else if k > 0 {
            // pi >= 1.0: all variables always included (backward compatible)
            beta = sample_beta(
                &mut rng,
                &y[..pre_end],
                &states,
                model.covariates(),
                sigma2_obs,
            );
            for gj in gamma.iter_mut() {
                *gj = true;
            }
            sigma2_obs =
                sample_sigma2_obs(&mut rng, &y[..pre_end], &states, &beta, model.covariates());
        } else {
            sigma2_obs =
                sample_sigma2_obs(&mut rng, &y[..pre_end], &states, &beta, model.covariates());
        }

        // Step 4: sigma^2_level sampling
        sigma2_level = sample_sigma2_level(&mut rng, &states, pre_end, prior_level_sd, y_sd);

        if iter >= nwarmup {
            let states_post = sample_post_period_states(
                &mut rng,
                states[pre_end - 1],
                t_total - pre_end,
                sigma2_level,
            );
            let full_states = combine_state_paths(&states, &states_post);
            let predictions =
                generate_predictions(&states_post, &beta, model, pre_end, sigma2_obs, &mut rng);

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

fn adjusted_observations(
    y: &[f64],
    model: &StateSpaceModel,
    beta: &[f64],
    pre_end: usize,
) -> Vec<f64> {
    y.iter()
        .take(pre_end)
        .enumerate()
        .map(|(t, y_value)| {
            y_value
                - beta
                    .iter()
                    .enumerate()
                    .map(|(j, beta_value)| model.x_at(j, t) * beta_value)
                    .sum::<f64>()
        })
        .collect()
}

fn sample_beta<R: rand::Rng>(
    rng: &mut R,
    y_pre: &[f64],
    states: &[f64],
    x: &[Vec<f64>],
    sigma2_obs: f64,
) -> Vec<f64> {
    let k = x.len();
    let mut xtx = vec![vec![0.0; k]; k];
    for (i, row) in xtx.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = x[i]
                .iter()
                .zip(x[j].iter())
                .take(y_pre.len())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum();
        }
    }

    let residuals: Vec<f64> = y_pre
        .iter()
        .zip(states.iter())
        .map(|(y_value, state)| y_value - state)
        .collect();

    let mut xtr = vec![0.0; k];
    for (j, x_col) in x.iter().enumerate() {
        xtr[j] = x_col
            .iter()
            .zip(residuals.iter())
            .map(|(x_value, residual)| x_value * residual)
            .sum();
    }

    let g = y_pre.len() as f64;
    let scale = (1.0 + 1.0 / g) / sigma2_obs;
    let mut precision = vec![vec![0.0; k]; k];
    for (i, row) in precision.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = xtx[i][j] * scale;
        }
    }

    let covariance = invert_matrix(&precision);
    let mut mean = vec![0.0; k];
    for (i, mean_value) in mean.iter_mut().enumerate() {
        *mean_value = covariance[i]
            .iter()
            .zip(xtr.iter())
            .map(|(covariance_value, xtr_value)| covariance_value * xtr_value / sigma2_obs)
            .sum();
    }

    sample_mvnormal(rng, &mean, &covariance)
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
) {
    let one_plus_g = 1.0 + g;
    let log_shrinkage = -0.5 * one_plus_g.ln(); // 0.5 * log(1/(1+g))
    let t_pre_f = t_pre as f64;

    for j in 0..k {
        let x_col = &x[j];
        let old_beta_j = beta[j];

        // Rank-1 update: add back old contribution to get partial residual
        for (t, r) in residual.iter_mut().enumerate().take(t_pre) {
            *r += x_col[t] * old_beta_j;
        }

        // Compute centered statistics for covariate j
        let x_sum: f64 = x_col.iter().take(t_pre).sum();
        let x_mean = x_sum / t_pre_f;
        let sum_x2: f64 = x_col.iter().take(t_pre).map(|v| v * v).sum();
        // Centered sum of squares: Σ(x_j - x̄)² = Σx² - T*x̄²
        let n_j = sum_x2 - t_pre_f * x_mean * x_mean;

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

fn sample_sigma2_obs<R: rand::Rng>(
    rng: &mut R,
    y_pre: &[f64],
    states: &[f64],
    beta: &[f64],
    x: &[Vec<f64>],
) -> f64 {
    let a_prior = 0.01;
    let b_prior = 0.01;

    let sse = y_pre
        .iter()
        .zip(states.iter())
        .enumerate()
        .map(|(t, (y_value, state))| {
            let fitted = state
                + x.iter()
                    .zip(beta.iter())
                    .map(|(x_col, beta_value)| x_col[t] * beta_value)
                    .sum::<f64>();
            let error = y_value - fitted;
            error * error
        })
        .sum::<f64>();

    let shape = a_prior + y_pre.len() as f64 / 2.0;
    let scale = b_prior + sse / 2.0;
    sample_inv_gamma(rng, shape, scale).max(1e-12)
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

fn combine_state_paths(states_pre: &[f64], states_post: &[f64]) -> Vec<f64> {
    let mut full_states = Vec::with_capacity(states_pre.len() + states_post.len());
    full_states.extend_from_slice(states_pre);
    full_states.extend_from_slice(states_post);
    full_states
}

fn generate_predictions<R: rand::Rng>(
    states_post: &[f64],
    beta: &[f64],
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
            model.observe(t, *state, beta) + sample_normal(rng, 0.0, sigma2_obs)
        })
        .collect()
}

fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = a.len();
    if k == 1 {
        return vec![vec![1.0 / a[0][0]]];
    }

    let width = 2 * k;
    let mut augmented = vec![vec![0.0; width]; k];
    let mut row_index = 0;
    while row_index < k {
        let mut col_index = 0;
        while col_index < k {
            augmented[row_index][col_index] = a[row_index][col_index];
            col_index += 1;
        }
        augmented[row_index][k + row_index] = 1.0;
        row_index += 1;
    }

    let mut pivot_col = 0;
    while pivot_col < k {
        let mut max_row = pivot_col;
        let mut max_value = augmented[pivot_col][pivot_col].abs();
        let mut candidate_row = pivot_col + 1;
        while candidate_row < k {
            let candidate_value = augmented[candidate_row][pivot_col].abs();
            if candidate_value > max_value {
                max_value = candidate_value;
                max_row = candidate_row;
            }
            candidate_row += 1;
        }
        augmented.swap(pivot_col, max_row);

        if augmented[pivot_col][pivot_col].abs() < 1e-15 {
            augmented[pivot_col][pivot_col] += 1e-8;
        }
        let pivot = augmented[pivot_col][pivot_col];

        let mut normalize_col = 0;
        while normalize_col < width {
            augmented[pivot_col][normalize_col] /= pivot;
            normalize_col += 1;
        }

        let mut eliminate_row = 0;
        while eliminate_row < k {
            if eliminate_row != pivot_col {
                let factor = augmented[eliminate_row][pivot_col];
                let mut eliminate_col = 0;
                while eliminate_col < width {
                    augmented[eliminate_row][eliminate_col] -=
                        factor * augmented[pivot_col][eliminate_col];
                    eliminate_col += 1;
                }
            }
            eliminate_row += 1;
        }

        pivot_col += 1;
    }

    let mut inverse = vec![vec![0.0; k]; k];
    let mut inverse_row = 0;
    while inverse_row < k {
        let mut inverse_col = 0;
        while inverse_col < k {
            inverse[inverse_row][inverse_col] = augmented[inverse_row][k + inverse_col];
            inverse_col += 1;
        }
        inverse_row += 1;
    }
    inverse
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_sampler_basic() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.1 * i as f64).collect();
        let result = run_sampler(y, vec![], 15, 10, 5, 1, 42, 0.01, 1.0).unwrap();
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
        let result = run_sampler(y, vec![x1], 15, 10, 5, 1, 42, 0.01, 1.0).unwrap();
        assert_eq!(result.beta.len(), 10);
        assert_eq!(result.beta[0].len(), 1);
        assert_eq!(result.gamma.len(), 10);
        assert_eq!(result.gamma[0].len(), 1);
    }

    #[test]
    fn test_run_sampler_empty_y() {
        let result = run_sampler(vec![], vec![], 0, 10, 5, 1, 42, 0.01, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_sampler_pre_end_equals_t() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = run_sampler(y, vec![], 3, 10, 5, 1, 42, 0.01, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_sampler_expected_model_size_validation() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.1 * i as f64).collect();
        let x1: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        // k>0 with expected_model_size=0 should fail
        let result = run_sampler(y.clone(), vec![x1.clone()], 15, 10, 5, 1, 42, 0.01, 0.0);
        assert!(result.is_err());
        // k>0 with negative should fail
        let result = run_sampler(y, vec![x1], 15, 10, 5, 1, 42, 0.01, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_standard_deviation_uses_sample_scale_because_prior_must_match_python_standardization() {
        let values = vec![1.0, 2.0, 3.0];
        let sample_sd = sample_standard_deviation(&values);
        assert!((sample_sd - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sample_post_period_states_moves_after_pre_period_because_random_walk_noise_must_propagate() {
        let mut rng = StdRng::seed_from_u64(42);
        let states_post = sample_post_period_states(&mut rng, 10.0, 5, 0.1);
        let rounded_unique_states: std::collections::BTreeSet<i64> = states_post
            .iter()
            .map(|state| (state * 1_000_000_000.0).round() as i64)
            .collect();
        assert!(rounded_unique_states.len() > 1);
    }

    #[test]
    fn test_invert_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let inv = invert_matrix(&a);
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[1][1] - 1.0).abs() < 1e-10);
    }
}
