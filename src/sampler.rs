use crate::distributions::{sample_inv_gamma, sample_mvnormal, sample_normal};
use crate::kalman::simulation_smoother;
use crate::state_space::StateSpaceModel;
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
    predictions: Vec<Vec<f64>>,
}

pub struct GibbsResult {
    pub states: Vec<Vec<f64>>,
    pub sigma_obs: Vec<f64>,
    pub sigma_level: Vec<f64>,
    pub beta: Vec<Vec<f64>>,
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
) -> Result<GibbsResult, String> {
    validate_inputs(&y, pre_end, nchains)?;

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
) -> ChainResult {
    let t_total = y.len();
    let chain_seed = seed ^ (chain_id as u64).wrapping_mul(SEED_PRIME);
    let mut rng = StdRng::seed_from_u64(chain_seed);
    let mut sigma2_obs = 1.0;
    let mut sigma2_level = (prior_level_sd * prior_level_sd).max(1e-6);
    let mut beta = vec![0.0; model.num_covariates()];
    let mut chain_result = ChainResult {
        chain_id,
        states: Vec::with_capacity(niter),
        sigma_obs: Vec::with_capacity(niter),
        sigma_level: Vec::with_capacity(niter),
        beta: Vec::with_capacity(niter),
        predictions: Vec::with_capacity(niter),
    };

    for iter in 0..(niter + nwarmup) {
        let y_adj = adjusted_observations(y, model, &beta, pre_end);
        let states = simulation_smoother(&mut rng, &y_adj, sigma2_obs, sigma2_level);

        if model.num_covariates() > 0 {
            beta = sample_beta(
                &mut rng,
                &y[..pre_end],
                &states,
                model.covariates(),
                sigma2_obs,
            );
        }

        sigma2_obs = sample_sigma2_obs(&mut rng, &y[..pre_end], &states, &beta, model.covariates());
        sigma2_level = sample_sigma2_level(&mut rng, &states, pre_end, prior_level_sd);

        if iter >= nwarmup {
            let full_states = extend_states_to_full(&states, pre_end, t_total);
            let predictions =
                generate_predictions(&full_states, &beta, model, pre_end, sigma2_obs, &mut rng);

            chain_result.states.push(full_states);
            chain_result.sigma_obs.push(sigma2_obs.sqrt());
            chain_result.sigma_level.push(sigma2_level.sqrt());
            chain_result.beta.push(beta.clone());
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
        predictions: Vec::with_capacity(n_samples),
    };

    for chain_result in chain_results {
        result.states.extend(chain_result.states);
        result.sigma_obs.extend(chain_result.sigma_obs);
        result.sigma_level.extend(chain_result.sigma_level);
        result.beta.extend(chain_result.beta);
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
) -> f64 {
    let a_prior = 0.5;
    let b_prior = 0.5 * prior_level_sd * prior_level_sd;
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

fn extend_states_to_full(states_pre: &[f64], pre_end: usize, t_total: usize) -> Vec<f64> {
    let mut full = vec![states_pre[pre_end - 1]; t_total];
    full[..pre_end].copy_from_slice(&states_pre[..pre_end]);
    full
}

fn generate_predictions<R: rand::Rng>(
    states: &[f64],
    beta: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    sigma2_obs: f64,
    rng: &mut R,
) -> Vec<f64> {
    states
        .iter()
        .enumerate()
        .skip(pre_end)
        .map(|(t, state)| model.observe(t, *state, beta) + sample_normal(rng, 0.0, sigma2_obs))
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
        let result = run_sampler(y, vec![], 15, 10, 5, 1, 42, 0.01).unwrap();
        assert_eq!(result.states.len(), 10);
        assert_eq!(result.sigma_obs.len(), 10);
        assert_eq!(result.predictions.len(), 10);
        assert_eq!(result.predictions[0].len(), 5);
    }

    #[test]
    fn test_run_sampler_with_covariates() {
        let y: Vec<f64> = (0..20).map(|i| 10.0 + 0.5 * i as f64).collect();
        let x1: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let result = run_sampler(y, vec![x1], 15, 10, 5, 1, 42, 0.01).unwrap();
        assert_eq!(result.beta.len(), 10);
        assert_eq!(result.beta[0].len(), 1);
    }

    #[test]
    fn test_run_sampler_empty_y() {
        let result = run_sampler(vec![], vec![], 0, 10, 5, 1, 42, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_sampler_pre_end_equals_t() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let result = run_sampler(y, vec![], 3, 10, 5, 1, 42, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_invert_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let inv = invert_matrix(&a);
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[1][1] - 1.0).abs() < 1e-10);
    }
}
