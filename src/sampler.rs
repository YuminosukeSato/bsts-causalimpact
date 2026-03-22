/// Gibbs sampler for Bayesian Structural Time Series (Local Level + Regression).
///
/// Each iteration:
///   1. Sample states via Durbin-Koopman simulation smoother
///   2. Sample β (regression coefficients) from conjugate Normal
///   3. Sample σ²_obs from conjugate InverseGamma
///   4. Sample σ²_level from conjugate InverseGamma
use crate::distributions::{sample_inv_gamma, sample_mvnormal, sample_normal};
use crate::kalman::simulation_smoother;
use crate::state_space::StateSpaceModel;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Large prime for deterministic child seed generation.
const SEED_PRIME: u64 = 6_364_136_223_846_793_005;

/// Result of the Gibbs sampler across all chains and iterations.
pub struct GibbsResult {
    /// Sampled states: (n_samples, T) flattened as Vec of Vec.
    pub states: Vec<Vec<f64>>,
    /// Sampled observation noise variance.
    pub sigma_obs: Vec<f64>,
    /// Sampled level noise variance.
    pub sigma_level: Vec<f64>,
    /// Sampled regression coefficients: (n_samples, k).
    pub beta: Vec<Vec<f64>>,
    /// Posterior predictions for post-period: (n_samples, T_post).
    pub predictions: Vec<Vec<f64>>,
}

/// Run the Gibbs sampler.
///
/// # Arguments
/// * `y` - Full time series (pre + post)
/// * `x` - Covariates as column-major Vec<Vec<f64>>, or empty
/// * `pre_end` - Index of last pre-period observation (exclusive: 0..pre_end is pre)
/// * `niter` - Number of MCMC iterations (post-warmup)
/// * `nwarmup` - Number of warmup iterations to discard
/// * `nchains` - Number of independent chains
/// * `seed` - Base random seed
/// * `prior_level_sd` - Prior standard deviation for level noise (R default: 0.01)
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
    let t_total = y.len();
    if t_total == 0 {
        return Err("y must not be empty".to_string());
    }
    if pre_end == 0 {
        return Err("pre_end must be at least 1".to_string());
    }
    if pre_end >= t_total {
        return Err("pre_end must be less than length of y (no post-period)".to_string());
    }

    // Check for NaN in pre-period
    for i in 0..pre_end {
        if y[i].is_nan() {
            return Err(format!("NaN found in y at pre-period index {i}"));
        }
    }

    let k = if x.is_empty() { 0 } else { x.len() };
    let t_post = t_total - pre_end;
    let n_samples = niter * nchains;
    let total_iter = niter + nwarmup;

    let model = StateSpaceModel::new(y.clone(), x.clone());

    let mut result = GibbsResult {
        states: Vec::with_capacity(n_samples),
        sigma_obs: Vec::with_capacity(n_samples),
        sigma_level: Vec::with_capacity(n_samples),
        beta: Vec::with_capacity(n_samples),
        predictions: Vec::with_capacity(n_samples),
    };

    for chain in 0..nchains {
        let chain_seed = seed ^ (chain as u64).wrapping_mul(SEED_PRIME);
        let mut rng = StdRng::seed_from_u64(chain_seed);

        // Initialize parameters
        let mut sigma2_obs = 1.0;
        let mut sigma2_level = (prior_level_sd * prior_level_sd).max(1e-6);
        let mut beta = vec![0.0; k];

        for iter in 0..total_iter {
            // --- Step 1: Sample states via simulation smoother ---
            // Adjust y for regression: y_adj = y - X * beta (pre-period only)
            let y_adj: Vec<f64> = (0..pre_end)
                .map(|t| {
                    let mut v = y[t];
                    for j in 0..k {
                        v -= model.x_at(j, t) * beta[j];
                    }
                    v
                })
                .collect();

            let states =
                simulation_smoother(&mut rng, &y_adj, sigma2_obs, sigma2_level);

            // --- Step 2: Sample β (regression coefficients) ---
            if k > 0 {
                beta = sample_beta(
                    &mut rng,
                    &y[..pre_end],
                    &states,
                    &x,
                    k,
                    pre_end,
                    sigma2_obs,
                );
            }

            // --- Step 3: Sample σ²_obs ---
            sigma2_obs = sample_sigma2_obs(
                &mut rng,
                &y[..pre_end],
                &states,
                &beta,
                &x,
                k,
                pre_end,
            );

            // --- Step 4: Sample σ²_level ---
            sigma2_level =
                sample_sigma2_level(&mut rng, &states, pre_end, prior_level_sd);

            // Store post-warmup samples
            if iter >= nwarmup {
                // Extend states to full time series for storage
                let full_states = extend_states_to_full(&states, pre_end, t_total);

                // Generate post-period predictions
                let preds = generate_predictions(
                    &full_states,
                    &beta,
                    &model,
                    pre_end,
                    t_total,
                    sigma2_obs,
                    &mut rng,
                );

                result.states.push(full_states);
                result.sigma_obs.push(sigma2_obs.sqrt());
                result.sigma_level.push(sigma2_level.sqrt());
                result.beta.push(beta.clone());
                result.predictions.push(preds);
            }
        }
    }

    Ok(result)
}

/// Sample regression coefficients β from the posterior.
/// β | y, μ, σ²_obs ~ Normal(μ_β, Σ_β)
/// where Σ_β = (X'X/σ²_obs + Σ_prior⁻¹)⁻¹
///       μ_β = Σ_β * X'r / σ²_obs
///       r = y - μ (residuals)
fn sample_beta<R: rand::Rng>(
    rng: &mut R,
    y_pre: &[f64],
    states: &[f64],
    x: &[Vec<f64>],
    k: usize,
    t_pre: usize,
    sigma2_obs: f64,
) -> Vec<f64> {
    // Compute X'X
    let mut xtx = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            for t in 0..t_pre {
                xtx[i][j] += x[i][t] * x[j][t];
            }
        }
    }

    // Residuals: r = y - mu
    let residuals: Vec<f64> = (0..t_pre).map(|t| y_pre[t] - states[t]).collect();

    // X'r
    let mut xtr = vec![0.0; k];
    for j in 0..k {
        for t in 0..t_pre {
            xtr[j] += x[j][t] * residuals[t];
        }
    }

    // Prior precision: use Zellner's g-prior with g = t_pre
    // Σ_prior⁻¹ = (1/g) * X'X / σ²_obs → weak prior
    let g = t_pre as f64;

    // Posterior precision: Σ_β⁻¹ = X'X/σ²_obs + X'X/(g * σ²_obs)
    //                            = X'X/σ²_obs * (1 + 1/g)
    let scale = (1.0 + 1.0 / g) / sigma2_obs;
    let mut precision = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            precision[i][j] = xtx[i][j] * scale;
        }
    }

    // Invert precision to get covariance
    let cov = invert_matrix(&precision, k);

    // Posterior mean: Σ_β * X'r / σ²_obs
    let mut mean = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            mean[i] += cov[i][j] * xtr[j] / sigma2_obs;
        }
    }

    sample_mvnormal(rng, &mean, &cov)
}

/// Sample observation noise variance from posterior.
/// σ²_obs ~ InvGamma(a + T/2, b + Σe²/2)
fn sample_sigma2_obs<R: rand::Rng>(
    rng: &mut R,
    y_pre: &[f64],
    states: &[f64],
    beta: &[f64],
    x: &[Vec<f64>],
    k: usize,
    t_pre: usize,
) -> f64 {
    let a_prior = 0.01;
    let b_prior = 0.01;

    let mut sse = 0.0;
    for t in 0..t_pre {
        let mut fitted = states[t];
        for j in 0..k {
            fitted += x[j][t] * beta[j];
        }
        let e = y_pre[t] - fitted;
        sse += e * e;
    }

    let shape = a_prior + t_pre as f64 / 2.0;
    let scale = b_prior + sse / 2.0;
    sample_inv_gamma(rng, shape, scale).max(1e-12)
}

/// Sample level noise variance from posterior.
/// σ²_level ~ InvGamma(a + (T-1)/2, b + Σd²/2)
/// with half-normal prior encoded as InvGamma approximation.
fn sample_sigma2_level<R: rand::Rng>(
    rng: &mut R,
    states: &[f64],
    t_pre: usize,
    prior_level_sd: f64,
) -> f64 {
    // Half-normal prior on σ_level → InvGamma(0.5, 0.5 * prior_level_sd²)
    let a_prior = 0.5;
    let b_prior = 0.5 * prior_level_sd * prior_level_sd;

    let mut ssd = 0.0;
    for t in 1..t_pre {
        let d = states[t] - states[t - 1];
        ssd += d * d;
    }

    let shape = a_prior + (t_pre - 1) as f64 / 2.0;
    let scale = b_prior + ssd / 2.0;
    sample_inv_gamma(rng, shape, scale).max(1e-12)
}

/// Extend pre-period states to the full time series using a random walk.
fn extend_states_to_full(states_pre: &[f64], pre_end: usize, t_total: usize) -> Vec<f64> {
    let mut full = vec![0.0; t_total];
    for t in 0..pre_end {
        full[t] = states_pre[t];
    }
    // Post-period: continue from last pre-period state
    let last_state = states_pre[pre_end - 1];
    for t in pre_end..t_total {
        full[t] = last_state;
    }
    full
}

/// Generate predictions for the post-period.
fn generate_predictions<R: rand::Rng>(
    states: &[f64],
    beta: &[f64],
    model: &StateSpaceModel,
    pre_end: usize,
    t_total: usize,
    sigma2_obs: f64,
    rng: &mut R,
) -> Vec<f64> {
    let t_post = t_total - pre_end;
    let mut preds = Vec::with_capacity(t_post);
    for t in pre_end..t_total {
        let mu = model.observe(t, states[t], beta);
        // Add observation noise
        let pred = mu + sample_normal(rng, 0.0, sigma2_obs);
        preds.push(pred);
    }
    preds
}

/// Naive matrix inversion for small k x k matrices (k typically < 10).
fn invert_matrix(a: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    if k == 1 {
        return vec![vec![1.0 / a[0][0]]];
    }

    // Gauss-Jordan elimination
    let mut aug = vec![vec![0.0; 2 * k]; k];
    for i in 0..k {
        for j in 0..k {
            aug[i][j] = a[i][j];
        }
        aug[i][k + i] = 1.0;
    }

    for col in 0..k {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..k {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            // Near-singular: add small ridge
            aug[col][col] += 1e-8;
            let pivot = aug[col][col];
            for j in 0..(2 * k) {
                aug[col][j] /= pivot;
            }
        } else {
            for j in 0..(2 * k) {
                aug[col][j] /= pivot;
            }
        }

        for row in 0..k {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * k) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            inv[i][j] = aug[i][k + j];
        }
    }
    inv
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
        let inv = invert_matrix(&a, 2);
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[1][1] - 1.0).abs() < 1e-10);
    }
}
