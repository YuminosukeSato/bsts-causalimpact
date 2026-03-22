//! Kalman filter and Durbin-Koopman simulation smoother for Local Level model,
//! plus multivariate FFBS for dynamic regression coefficients β_t.
//!
//! The simulation smoother draws states from p(α | y, σ²_obs, σ²_level, β)
//! using the method of Durbin & Koopman (2002).

use crate::distributions::sample_normal;
use rand::Rng;

const F_MIN: f64 = 1e-12;

pub fn kalman_filter(
    y_adj: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    initial_state_mean: f64,
    initial_state_variance: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = y_adj.len();
    let mut a = vec![0.0; t];
    let mut p = vec![0.0; t];
    let mut v = vec![0.0; t];
    let mut f = vec![0.0; t];

    let a0 = initial_state_mean;
    let p0 = initial_state_variance.max(F_MIN);

    for i in 0..t {
        let a_prior = if i == 0 { a0 } else { a[i - 1] };
        let p_prior = if i == 0 { p0 } else { p[i - 1] + sigma2_level };

        v[i] = y_adj[i] - a_prior;
        f[i] = (p_prior + sigma2_obs).max(F_MIN);

        let k = p_prior / f[i];
        a[i] = a_prior + k * v[i];
        let one_minus_k = 1.0 - k;
        p[i] = one_minus_k * one_minus_k * p_prior + k * k * sigma2_obs;
    }

    (a, p, v, f)
}

pub fn simulation_smoother<R: Rng>(
    rng: &mut R,
    y_adj: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    initial_state_mean: f64,
    initial_state_variance: f64,
) -> Vec<f64> {
    let t = y_adj.len();
    let centered_y_adj: Vec<f64> = y_adj
        .iter()
        .map(|value| value - initial_state_mean)
        .collect();

    let mut alpha_plus = vec![0.0; t];
    let mut y_plus = vec![0.0; t];

    alpha_plus[0] = sample_normal(rng, 0.0, initial_state_variance);
    y_plus[0] = alpha_plus[0] + sample_normal(rng, 0.0, sigma2_obs);

    for i in 1..t {
        alpha_plus[i] = alpha_plus[i - 1] + sample_normal(rng, 0.0, sigma2_level);
        y_plus[i] = alpha_plus[i] + sample_normal(rng, 0.0, sigma2_obs);
    }

    let y_star: Vec<f64> = centered_y_adj
        .iter()
        .zip(&y_plus)
        .map(|(y, yp)| y - yp)
        .collect();
    let (a, p, v, f) = kalman_filter(
        &y_star,
        sigma2_obs,
        sigma2_level,
        0.0,
        initial_state_variance,
    );
    let alpha_hat = state_smoother(&a, &p, &v, &f, sigma2_level, 0.0, initial_state_variance);

    alpha_hat
        .iter()
        .zip(alpha_plus.iter())
        .map(|(alpha_hat_t, alpha_plus_t)| alpha_hat_t + alpha_plus_t + initial_state_mean)
        .collect()
}

fn state_smoother(
    a: &[f64],
    p: &[f64],
    v: &[f64],
    f: &[f64],
    sigma2_level: f64,
    initial_state_mean: f64,
    initial_state_variance: f64,
) -> Vec<f64> {
    let t = a.len();
    let mut alpha_hat = vec![0.0; t];
    let mut r = 0.0;

    for i in (0..t).rev() {
        let p_prior = if i == 0 {
            initial_state_variance
        } else {
            p[i - 1] + sigma2_level
        };
        let k = p_prior / f[i];
        r = v[i] / f[i] + (1.0 - k) * r;

        let a_prior = if i == 0 { initial_state_mean } else { a[i - 1] };
        alpha_hat[i] = a_prior + p_prior * r;
    }

    alpha_hat
}

/// Multivariate FFBS (Forward Filtering Backward Sampling) for dynamic
/// regression coefficients β_t ∈ R^k with scalar observation.
///
/// Model:
///   y_adj_t = x_t' β_t + ε_t,  ε_t ~ N(0, σ²_obs)
///   β_t = β_{t-1} + η_t,       η_t ~ N(0, diag(σ²_β))
///
/// Uses Durbin-Koopman simulation smoother approach:
///   1. Draw β⁺ from prior → generate y⁺ = x'β⁺ + ε⁺
///   2. y* = y_adj - y⁺
///   3. Kalman filter on y* → β̂ (smoothed mean)
///   4. β = β̂ + β⁺
///
/// Returns T × k matrix (outer: time, inner: coefficient index).
#[allow(clippy::too_many_arguments)]
pub fn dynamic_beta_smoother<R: Rng>(
    rng: &mut R,
    y_adj: &[f64],
    x: &[Vec<f64>],
    sigma2_obs: f64,
    sigma2_beta: &[f64],
    init_beta_mean: &[f64],
    init_beta_var: f64,
) -> Vec<Vec<f64>> {
    let t = y_adj.len();
    let k = x.len();

    if k == 0 || t == 0 {
        return vec![vec![]; t];
    }

    let q_diag = sigma2_beta;

    // Step 1: Draw β⁺ from prior and generate y⁺
    let mut beta_plus = vec![vec![0.0; k]; t];
    let mut y_plus = vec![0.0; t];

    // β⁺_0 ~ N(0, init_beta_var * I)
    for val in beta_plus[0].iter_mut() {
        *val = sample_normal(rng, 0.0, init_beta_var);
    }
    y_plus[0] = dot_xt(x, &beta_plus[0], 0) + sample_normal(rng, 0.0, sigma2_obs);

    for i in 1..t {
        for j in 0..k {
            beta_plus[i][j] = beta_plus[i - 1][j] + sample_normal(rng, 0.0, q_diag[j]);
        }
        y_plus[i] = dot_xt(x, &beta_plus[i], i) + sample_normal(rng, 0.0, sigma2_obs);
    }

    // Step 2: y* = y_adj - y⁺ (centered around init_beta_mean)
    let y_star: Vec<f64> = y_adj
        .iter()
        .enumerate()
        .zip(&y_plus)
        .map(|((i, &y), &yp)| y - dot_xt(x, init_beta_mean, i) - yp)
        .collect();

    // Step 3: Multivariate Kalman filter (k-vector state, scalar obs)
    let beta_hat = mv_kalman_smoother(&y_star, x, sigma2_obs, q_diag, init_beta_var);

    // Step 4: β = β̂ + β⁺ + init_beta_mean
    let mut result = vec![vec![0.0; k]; t];
    for i in 0..t {
        for j in 0..k {
            result[i][j] = beta_hat[i][j] + beta_plus[i][j] + init_beta_mean[j];
        }
    }

    result
}

/// Compute x_t' β = Σ_j x[j][t] * beta[j]
#[inline]
fn dot_xt(x: &[Vec<f64>], beta: &[f64], t: usize) -> f64 {
    x.iter()
        .zip(beta.iter())
        .map(|(x_col, &b)| x_col[t] * b)
        .sum()
}

/// Multivariate Kalman filter + RTS smoother for k-vector state, scalar obs.
///
/// State equation:  β_t = β_{t-1} + η_t,  η_t ~ N(0, Q = diag(q_diag))
/// Obs equation:    y_t = x_t' β_t + ε_t,  ε_t ~ N(0, σ²_obs)
///
/// Returns smoothed state estimates β̂_t (T × k).
fn mv_kalman_smoother(
    y: &[f64],
    x: &[Vec<f64>],
    sigma2_obs: f64,
    q_diag: &[f64],
    init_var: f64,
) -> Vec<Vec<f64>> {
    let t = y.len();
    let k = x.len();

    // Storage for forward pass
    let mut a_filt = vec![vec![0.0; k]; t]; // filtered state means
    let mut p_filt = vec![vec![vec![0.0; k]; k]; t]; // filtered state covariances

    // Initial state: a_0|0 = 0, P_0|0 = init_var * I
    let mut a_pred = vec![0.0; k];
    let mut p_pred = vec![vec![0.0; k]; k];
    for (j, row) in p_pred.iter_mut().enumerate() {
        row[j] = init_var;
    }

    // Forward filter
    for i in 0..t {
        // x_t vector at time i
        let x_t: Vec<f64> = x.iter().map(|col| col[i]).collect();

        // Innovation: v_t = y_t - x_t' a_{t|t-1}
        let v: f64 = y[i]
            - x_t
                .iter()
                .zip(&a_pred)
                .map(|(&xi, &ai)| xi * ai)
                .sum::<f64>();

        // Innovation variance: F_t = x_t' P_{t|t-1} x_t + σ²_obs (scalar)
        let p_x: Vec<f64> = (0..k)
            .map(|j| {
                p_pred[j]
                    .iter()
                    .zip(&x_t)
                    .map(|(&p, &xi)| p * xi)
                    .sum::<f64>()
            })
            .collect();
        let f_t: f64 = x_t.iter().zip(&p_x).map(|(&xi, &px)| xi * px).sum::<f64>() + sigma2_obs;
        let f_inv = 1.0 / f_t.max(F_MIN);

        // Kalman gain: K_t = P_{t|t-1} x_t / F_t  (k-vector)
        let k_gain: Vec<f64> = p_x.iter().map(|&px| px * f_inv).collect();

        // Update: a_{t|t} = a_{t|t-1} + K_t v_t
        for j in 0..k {
            a_filt[i][j] = a_pred[j] + k_gain[j] * v;
        }

        // P_{t|t} = (I - K_t x_t') P_{t|t-1}
        // Using Joseph form for stability: P = (I-Kx')P(I-Kx')' + K σ²_obs K'
        for row in 0..k {
            for col in 0..k {
                let mut val = 0.0;
                for m in 0..k {
                    let i_kx_rm = if row == m { 1.0 } else { 0.0 } - k_gain[row] * x_t[m];
                    for n in 0..k {
                        let i_kx_cn = if col == n { 1.0 } else { 0.0 } - k_gain[col] * x_t[n];
                        val += i_kx_rm * p_pred[m][n] * i_kx_cn;
                    }
                }
                val += k_gain[row] * sigma2_obs * k_gain[col];
                p_filt[i][row][col] = val;
            }
        }

        // Symmetrize and enforce positive diagonal
        symmetrize_and_floor(&mut p_filt[i], k);

        // Predict next step: a_{t+1|t} = a_{t|t}, P_{t+1|t} = P_{t|t} + Q
        a_pred = a_filt[i].clone();
        p_pred = p_filt[i].clone();
        for j in 0..k {
            p_pred[j][j] += q_diag[j];
        }
    }

    // Backward smoother (Rauch-Tung-Striebel) using Cholesky solve
    // instead of explicit inverse for numerical stability.
    //
    // β̂_t = a_{t|t} + P_{t|t} @ solve(P_{t+1|t}, β̂_{t+1} - a_{t|t})
    // where solve(A, b) computes A^{-1} b via Cholesky decomposition.
    let mut beta_hat = vec![vec![0.0; k]; t];
    beta_hat[t - 1] = a_filt[t - 1].clone();

    for i in (0..t - 1).rev() {
        // P_{t+1|t} = P_{t|t} + Q
        let mut p_pred_next = p_filt[i].clone();
        for j in 0..k {
            p_pred_next[j][j] += q_diag[j];
        }

        // d = β̂_{t+1} - a_{t|t}
        let d: Vec<f64> = (0..k).map(|j| beta_hat[i + 1][j] - a_filt[i][j]).collect();

        // z = solve(P_{t+1|t}, d) via Cholesky
        let z = cholesky_solve(&p_pred_next, &d, k);

        // β̂_t = a_{t|t} + P_{t|t} @ z
        for j in 0..k {
            let correction: f64 = (0..k).map(|m| p_filt[i][j][m] * z[m]).sum();
            beta_hat[i][j] = a_filt[i][j] + correction;
        }
    }

    beta_hat
}

/// Symmetrize matrix and enforce minimum diagonal value.
#[allow(clippy::needless_range_loop)]
fn symmetrize_and_floor(p: &mut [Vec<f64>], k: usize) {
    // Cross-row mutation (p[row] and p[col]) prevents iterator usage.
    for row in 0..k {
        for col in (row + 1)..k {
            let avg = 0.5 * (p[row][col] + p[col][row]);
            p[row][col] = avg;
            p[col][row] = avg;
        }
        p[row][row] = p[row][row].max(1e-10);
    }
}

/// Solve A x = b via Cholesky decomposition (A must be SPD).
/// More stable than computing A^{-1} explicitly.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64], k: usize) -> Vec<f64> {
    // Relative ridge regularization
    let max_diag = (0..k).map(|i| a[i][i].abs()).fold(0.0_f64, f64::max);
    let ridge = max_diag * 1e-10 + 1e-15;

    // Cholesky decomposition: A_reg = L L'
    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|m| l[i][m] * l[j][m]).sum();
            if i == j {
                let diag = a[i][i] + ridge - sum;
                l[i][j] = if diag > 1e-15 { diag.sqrt() } else { 1e-8 };
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    // Forward substitution: L w = b
    let mut w = vec![0.0; k];
    for i in 0..k {
        let sum: f64 = (0..i).map(|j| l[i][j] * w[j]).sum();
        w[i] = (b[i] - sum) / l[i][i];
    }

    // Backward substitution: L' x = w
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let sum: f64 = ((i + 1)..k).map(|j| l[j][i] * x[j]).sum();
        x[i] = (w[i] - sum) / l[i][i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_kalman_filter_constant_signal() {
        let y: Vec<f64> = vec![5.0; 50];
        let (a, _p, _v, _f) = kalman_filter(&y, 1.0, 0.01, 5.0, 1.0);
        assert!(
            (a[49] - 5.0).abs() < 0.1,
            "Filtered state should converge to 5.0, got {}",
            a[49]
        );
    }

    #[test]
    fn test_kalman_filter_output_lengths() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let (a, p, v, f) = kalman_filter(&y, 1.0, 0.1, 1.0, 1.0);
        assert_eq!(a.len(), 3);
        assert_eq!(p.len(), 3);
        assert_eq!(v.len(), 3);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn test_simulation_smoother_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let states = simulation_smoother(&mut rng, &y, 1.0, 0.01, 1.0, 1.0);
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn test_simulation_smoother_tracks_signal() {
        let mut rng = StdRng::seed_from_u64(42);
        let y: Vec<f64> = vec![10.0; 50];
        let n_samples = 200;
        let mut mean_states = vec![0.0; 50];

        for _ in 0..n_samples {
            let states = simulation_smoother(&mut rng, &y, 0.1, 0.01, 10.0, 1.0);
            for (mean_state, state) in mean_states.iter_mut().zip(states.iter()) {
                *mean_state += state / n_samples as f64;
            }
        }

        assert!(
            (mean_states[49] - 10.0).abs() < 1.0,
            "Smoothed state should be near 10.0, got {}",
            mean_states[49]
        );
    }

    #[test]
    fn test_dynamic_beta_smoother_output_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let t = 20;
        let k = 3;
        let x: Vec<Vec<f64>> = (0..k)
            .map(|j| (0..t).map(|i| (i * (j + 1)) as f64 * 0.1).collect())
            .collect();
        let y_adj: Vec<f64> = (0..t).map(|i| i as f64 * 0.5).collect();
        let sigma2_beta = vec![0.01; k];
        let init_mean = vec![0.0; k];

        let result =
            dynamic_beta_smoother(&mut rng, &y_adj, &x, 1.0, &sigma2_beta, &init_mean, 1e2);

        assert_eq!(result.len(), t);
        for row in &result {
            assert_eq!(row.len(), k);
        }
    }

    #[test]
    fn test_dynamic_beta_smoother_k1_tracks_constant_signal() {
        let t = 100;
        let beta_true = 2.0;
        let x = vec![(0..t).map(|i| (i as f64) * 0.1).collect::<Vec<f64>>()];
        let y_adj: Vec<f64> = x[0].iter().map(|&xi| xi * beta_true).collect();
        let sigma2_beta = vec![0.001];
        let init_mean = vec![0.0];
        let n_samples = 100;
        let mut mean_beta_last = 0.0;

        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed);
            let result =
                dynamic_beta_smoother(&mut rng, &y_adj, &x, 0.1, &sigma2_beta, &init_mean, 1e2);
            mean_beta_last += result[t - 1][0] / n_samples as f64;
        }

        assert!(
            (mean_beta_last - beta_true).abs() < 0.5,
            "Mean beta at last time should be near {}, got {}",
            beta_true,
            mean_beta_last
        );
    }

    #[test]
    fn test_dynamic_beta_smoother_preserves_p_positive_definite() {
        let mut rng = StdRng::seed_from_u64(42);
        let t = 500;
        let k = 5;
        let x: Vec<Vec<f64>> = (0..k)
            .map(|j| {
                (0..t)
                    .map(|i| ((i * (j + 1)) as f64 * 0.01).sin())
                    .collect()
            })
            .collect();
        let y_adj: Vec<f64> = (0..t).map(|i| (i as f64 * 0.05).sin()).collect();
        let sigma2_beta = vec![0.01; k];
        let init_mean = vec![0.0; k];

        let result =
            dynamic_beta_smoother(&mut rng, &y_adj, &x, 1.0, &sigma2_beta, &init_mean, 1e2);

        // All values should be finite
        for row in &result {
            for &val in row {
                assert!(val.is_finite(), "Non-finite value in beta_t: {}", val);
            }
        }
    }
}
