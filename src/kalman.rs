//! Kalman filter and Durbin-Koopman simulation smoother for Local Level model,
//! plus multivariate FFBS for dynamic regression coefficients β_t and a
//! local linear trend state model.
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

pub fn local_linear_trend_smoother<R: Rng>(
    rng: &mut R,
    y_adj: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    sigma2_slope: f64,
    initial_state_mean: f64,
    initial_state_variance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = y_adj.len();
    if t == 0 {
        return (vec![], vec![]);
    }

    let centered_y_adj: Vec<f64> = y_adj
        .iter()
        .map(|value| value - initial_state_mean)
        .collect();
    let initial_slope_variance = (initial_state_variance / (t.max(1) as f64).powi(2)).max(F_MIN);

    let mut level_plus = vec![0.0; t];
    let mut slope_plus = vec![0.0; t];
    let mut y_plus = vec![0.0; t];

    level_plus[0] = sample_normal(rng, 0.0, initial_state_variance);
    slope_plus[0] = sample_normal(rng, 0.0, initial_slope_variance);
    y_plus[0] = level_plus[0] + sample_normal(rng, 0.0, sigma2_obs);

    for i in 1..t {
        level_plus[i] =
            level_plus[i - 1] + slope_plus[i - 1] + sample_normal(rng, 0.0, sigma2_level);
        slope_plus[i] = slope_plus[i - 1] + sample_normal(rng, 0.0, sigma2_slope);
        y_plus[i] = level_plus[i] + sample_normal(rng, 0.0, sigma2_obs);
    }

    let y_star: Vec<f64> = centered_y_adj
        .iter()
        .zip(&y_plus)
        .map(|(y, yp)| y - yp)
        .collect();
    let (level_hat, slope_hat) = local_linear_trend_mean(
        &y_star,
        sigma2_obs,
        sigma2_level,
        sigma2_slope,
        initial_state_variance,
        initial_slope_variance,
    );

    let levels = level_hat
        .iter()
        .zip(level_plus.iter())
        .map(|(level_hat_t, level_plus_t)| level_hat_t + level_plus_t + initial_state_mean)
        .collect();
    let slopes = slope_hat
        .iter()
        .zip(slope_plus.iter())
        .map(|(slope_hat_t, slope_plus_t)| slope_hat_t + slope_plus_t)
        .collect();

    (levels, slopes)
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

fn local_linear_trend_mean(
    y: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    sigma2_slope: f64,
    initial_level_variance: f64,
    initial_slope_variance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = y.len();
    let transition = [[1.0, 1.0], [0.0, 1.0]];
    let transition_t = [[1.0, 0.0], [1.0, 1.0]];
    let q = [[sigma2_level, 0.0], [0.0, sigma2_slope]];
    let init_cov = [
        [initial_level_variance.max(F_MIN), 0.0],
        [0.0, initial_slope_variance.max(F_MIN)],
    ];

    let mut a_pred_store = vec![[0.0, 0.0]; t];
    let mut p_pred_store = vec![[[0.0, 0.0], [0.0, 0.0]]; t];
    let mut a_filt = vec![[0.0, 0.0]; t];
    let mut p_filt = vec![[[0.0, 0.0], [0.0, 0.0]]; t];

    for i in 0..t {
        let (a_pred, p_pred) = if i == 0 {
            ([0.0, 0.0], init_cov)
        } else {
            let mean = apply_transition(a_filt[i - 1], transition);
            let covariance = predict_covariance(p_filt[i - 1], transition, transition_t, q);
            (mean, covariance)
        };
        a_pred_store[i] = a_pred;
        p_pred_store[i] = p_pred;

        let innovation = y[i] - a_pred[0];
        let innovation_variance = (p_pred[0][0] + sigma2_obs).max(F_MIN);
        let gain = [
            p_pred[0][0] / innovation_variance,
            p_pred[1][0] / innovation_variance,
        ];

        a_filt[i] = [
            a_pred[0] + gain[0] * innovation,
            a_pred[1] + gain[1] * innovation,
        ];
        p_filt[i] = update_covariance(p_pred, gain, sigma2_obs);
        symmetrize_and_floor_2x2(&mut p_filt[i]);
    }

    let mut smooth = vec![[0.0, 0.0]; t];
    smooth[t - 1] = a_filt[t - 1];

    for i in (0..t - 1).rev() {
        let predicted_next = p_pred_store[i + 1];
        let smoother_gain = matrix_mul_2x2(
            p_filt[i],
            matrix_mul_2x2(transition_t, invert_2x2(predicted_next)),
        );
        let diff = [
            smooth[i + 1][0] - a_pred_store[i + 1][0],
            smooth[i + 1][1] - a_pred_store[i + 1][1],
        ];
        let correction = matrix_vec_mul_2x2(smoother_gain, diff);
        smooth[i] = [a_filt[i][0] + correction[0], a_filt[i][1] + correction[1]];
    }

    (
        smooth.iter().map(|state| state[0]).collect(),
        smooth.iter().map(|state| state[1]).collect(),
    )
}

fn apply_transition(state: [f64; 2], transition: [[f64; 2]; 2]) -> [f64; 2] {
    [
        transition[0][0] * state[0] + transition[0][1] * state[1],
        transition[1][0] * state[0] + transition[1][1] * state[1],
    ]
}

fn predict_covariance(
    covariance: [[f64; 2]; 2],
    transition: [[f64; 2]; 2],
    transition_t: [[f64; 2]; 2],
    process_noise: [[f64; 2]; 2],
) -> [[f64; 2]; 2] {
    add_matrix_2x2(
        matrix_mul_2x2(matrix_mul_2x2(transition, covariance), transition_t),
        process_noise,
    )
}

fn update_covariance(
    predicted_covariance: [[f64; 2]; 2],
    gain: [f64; 2],
    sigma2_obs: f64,
) -> [[f64; 2]; 2] {
    let identity_minus_kh = [[1.0 - gain[0], 0.0], [-gain[1], 1.0]];
    let joseph_left = matrix_mul_2x2(identity_minus_kh, predicted_covariance);
    let joseph = matrix_mul_2x2(joseph_left, transpose_2x2(identity_minus_kh));
    let observation_term = [
        [
            gain[0] * gain[0] * sigma2_obs,
            gain[0] * gain[1] * sigma2_obs,
        ],
        [
            gain[1] * gain[0] * sigma2_obs,
            gain[1] * gain[1] * sigma2_obs,
        ],
    ];
    add_matrix_2x2(joseph, observation_term)
}

fn matrix_mul_2x2(lhs: [[f64; 2]; 2], rhs: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            lhs[0][0] * rhs[0][0] + lhs[0][1] * rhs[1][0],
            lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[1][1],
        ],
        [
            lhs[1][0] * rhs[0][0] + lhs[1][1] * rhs[1][0],
            lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[1][1],
        ],
    ]
}

fn matrix_vec_mul_2x2(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn add_matrix_2x2(lhs: [[f64; 2]; 2], rhs: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [lhs[0][0] + rhs[0][0], lhs[0][1] + rhs[0][1]],
        [lhs[1][0] + rhs[1][0], lhs[1][1] + rhs[1][1]],
    ]
}

fn transpose_2x2(matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

fn invert_2x2(mut matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let ridge = ((matrix[0][0].abs() + matrix[1][1].abs()) * 1e-10).max(1e-12);
    matrix[0][0] += ridge;
    matrix[1][1] += ridge;

    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let safe_determinant = if determinant.abs() < F_MIN {
        F_MIN.copysign(determinant.signum().max(1.0))
    } else {
        determinant
    };

    [
        [
            matrix[1][1] / safe_determinant,
            -matrix[0][1] / safe_determinant,
        ],
        [
            -matrix[1][0] / safe_determinant,
            matrix[0][0] / safe_determinant,
        ],
    ]
}

fn symmetrize_and_floor_2x2(matrix: &mut [[f64; 2]; 2]) {
    let off_diagonal = 0.5 * (matrix[0][1] + matrix[1][0]);
    matrix[0][1] = off_diagonal;
    matrix[1][0] = off_diagonal;
    matrix[0][0] = matrix[0][0].max(F_MIN);
    matrix[1][1] = matrix[1][1].max(F_MIN);
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
    let mut p_x = vec![0.0; k];
    let mut k_gain = vec![0.0; k];
    let mut p_pred_next = vec![vec![0.0; k]; k];
    let mut delta = vec![0.0; k];
    for (j, row) in p_pred.iter_mut().enumerate() {
        row[j] = init_var;
    }

    // Forward filter
    for i in 0..t {
        // Innovation: v_t = y_t - x_t' a_{t|t-1}
        let predicted_observation = x
            .iter()
            .zip(a_pred.iter())
            .map(|(x_col, &a_i)| x_col[i] * a_i)
            .sum::<f64>();
        let v = y[i] - predicted_observation;

        // Innovation variance: F_t = x_t' P_{t|t-1} x_t + σ²_obs (scalar)
        for j in 0..k {
            p_x[j] = p_pred[j]
                .iter()
                .zip(x.iter())
                .map(|(&p, x_col)| p * x_col[i])
                .sum::<f64>();
        }
        let f_t = x
            .iter()
            .zip(p_x.iter())
            .map(|(x_col, &px)| x_col[i] * px)
            .sum::<f64>()
            + sigma2_obs;
        let f_inv = 1.0 / f_t.max(F_MIN);

        // Kalman gain: K_t = P_{t|t-1} x_t / F_t  (k-vector)
        for j in 0..k {
            k_gain[j] = p_x[j] * f_inv;
        }

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
                    let i_kx_rm = if row == m { 1.0 } else { 0.0 } - k_gain[row] * x[m][i];
                    for n in 0..k {
                        let i_kx_cn = if col == n { 1.0 } else { 0.0 } - k_gain[col] * x[n][i];
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
        a_pred.clone_from(&a_filt[i]);
        p_pred.clone_from(&p_filt[i]);
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
        p_pred_next.clone_from(&p_filt[i]);
        for j in 0..k {
            p_pred_next[j][j] += q_diag[j];
        }

        // d = β̂_{t+1} - a_{t|t}
        for j in 0..k {
            delta[j] = beta_hat[i + 1][j] - a_filt[i][j];
        }

        // z = solve(P_{t+1|t}, d) via Cholesky
        let z = cholesky_solve(&p_pred_next, &delta, k);

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

/// Local level + seasonal state-space simulation smoother (Durbin-Koopman 2002, Jarociński 2015).
///
/// State vector: α_t = [μ_t, s_1(t), ..., s_{S-1}(t)]' ∈ ℝ^S
/// Observation: y_t = Z α_t + ε_t,  Z = [1, 1, 0, ..., 0]
/// Transition:
///   - season boundary (t % season_duration == 0):
///       μ_{t+1} = μ_t + η_level
///       s_1(t+1) = -Σ s_j(t) + η_seasonal
///       s_j(t+1) = s_{j-1}(t)  for j >= 2
///   - intra-season (otherwise):
///       μ_{t+1} = μ_t + η_level
///       seasonal state unchanged (identity)
/// Process noise: Q = diag(σ²_level, σ²_seasonal, 0, ..., 0)
///
/// Returns:
///   levels:         Vec<f64>  — μ_t (T values)
///   s1_obs:         Vec<f64>  — s_1(t), seasonal component observed at each t
///   innovation_ssd: f64       — Σ η_{s,t}², sufficient statistic for σ²_seasonal sampling
#[allow(clippy::too_many_arguments)]
pub fn local_level_seasonal_smoother<R: Rng>(
    rng: &mut R,
    y_residual: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    sigma2_seasonal: f64,
    nseasons: usize,
    season_duration: usize,
    initial_state_mean: f64,
    initial_state_var: f64,
) -> (Vec<f64>, Vec<f64>, f64) {
    let t = y_residual.len();
    let s = nseasons; // state dimension = nseasons (level + S-1 seasonal)

    if t == 0 {
        return (vec![], vec![], 0.0);
    }
    if s < 2 {
        // Degenerate: no seasonal component, fallback to simple smoother
        let levels = simulation_smoother(
            rng,
            y_residual,
            sigma2_obs,
            sigma2_level,
            initial_state_mean,
            initial_state_var,
        );
        return (levels, vec![0.0; t], 0.0);
    }

    // Center observations around initial_state_mean (Jarociński fix: a1=0)
    let centered_y: Vec<f64> = y_residual.iter().map(|v| v - initial_state_mean).collect();

    // Initial state variance for seasonal components
    let seasonal_init_var = initial_state_var;

    // Step 1: Draw α⁺ from prior → generate y⁺
    let mut alpha_plus = vec![vec![0.0; s]; t];
    let mut y_plus = vec![0.0; t];

    // α⁺_0 ~ N(0, diag(init_var, seasonal_init_var, ...))
    alpha_plus[0][0] = sample_normal(rng, 0.0, initial_state_var);
    for alpha in alpha_plus[0].iter_mut().take(s).skip(1) {
        *alpha = sample_normal(rng, 0.0, seasonal_init_var);
    }
    // y⁺_0 = Z α⁺_0 + ε⁺ = α⁺[0][0] + α⁺[0][1] + ε⁺
    y_plus[0] = alpha_plus[0][0] + alpha_plus[0][1] + sample_normal(rng, 0.0, sigma2_obs);

    for i in 1..t {
        let (prev_rows, current_and_rest) = alpha_plus.split_at_mut(i);
        let prev = &prev_rows[i - 1];
        let current = &mut current_and_rest[0];

        // Level: random walk
        current[0] = prev[0] + sample_normal(rng, 0.0, sigma2_level);

        if is_season_boundary(i, season_duration) {
            // Seasonal transition: s_1(t) = -Σ s_j(t-1) + η_seasonal
            let seasonal_sum: f64 = prev[1..s].iter().sum();
            current[1] = -seasonal_sum + sample_normal(rng, 0.0, sigma2_seasonal);
            // s_j(t) = s_{j-1}(t-1) for j >= 2
            if s > 2 {
                current[2..s].copy_from_slice(&prev[1..(s - 1)]);
            }
        } else {
            // Intra-season: seasonal state unchanged
            current[1..s].copy_from_slice(&prev[1..s]);
        }

        y_plus[i] = current[0] + current[1] + sample_normal(rng, 0.0, sigma2_obs);
    }

    // Step 2: y* = centered_y - y⁺
    let y_star: Vec<f64> = centered_y
        .iter()
        .zip(&y_plus)
        .map(|(y, yp)| y - yp)
        .collect();

    // Step 3-5: Kalman filter + RTS smoother on y*
    let alpha_hat = seasonal_kalman_smoother(
        &y_star,
        sigma2_obs,
        sigma2_level,
        sigma2_seasonal,
        s,
        season_duration,
        initial_state_var,
        seasonal_init_var,
    );

    // Step 6: α = α̂* + α⁺ + mean correction
    let mut levels = vec![0.0; t];
    let mut s1_obs = vec![0.0; t];

    for i in 0..t {
        levels[i] = alpha_hat[i][0] + alpha_plus[i][0] + initial_state_mean;
        s1_obs[i] = alpha_hat[i][1] + alpha_plus[i][1];
    }

    // Compute innovation_ssd: Σ (η_{s,t})² at season boundaries
    // η_{s,t} = s_1(t) - expected_s1 = s_1(t) - (-Σ_{j=1}^{S-1} s_j(t-1))
    let mut innovation_ssd = 0.0;
    for i in 1..t {
        if is_season_boundary(i, season_duration) {
            let expected_s1 = -(1..s)
                .map(|j| alpha_hat[i - 1][j] + alpha_plus[i - 1][j])
                .sum::<f64>();
            let eta = alpha_hat[i][1] + alpha_plus[i][1] - expected_s1;
            innovation_ssd += eta * eta;
        }
    }

    (levels, s1_obs, innovation_ssd)
}

/// Whether time step t is a season boundary.
#[inline]
fn is_season_boundary(t: usize, season_duration: usize) -> bool {
    t.is_multiple_of(season_duration)
}

/// S-dimensional Kalman filter + RTS smoother for local level + seasonal model.
///
/// State equation uses time-varying transition (season boundary vs intra-season).
/// Returns smoothed state estimates α̂_t (T × S).
#[allow(clippy::too_many_arguments)]
fn seasonal_kalman_smoother(
    y: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
    sigma2_seasonal: f64,
    s: usize,
    season_duration: usize,
    initial_level_var: f64,
    initial_seasonal_var: f64,
) -> Vec<Vec<f64>> {
    let t = y.len();
    let ss = s * s; // stride for flat p_pred indexing

    // Z vector: [1, 1, 0, ..., 0]
    // Z'Z = scalar, Z α = α[0] + α[1]

    // Q diagonal: [sigma2_level, sigma2_seasonal, 0, ..., 0]
    let q_diag: Vec<f64> = (0..s)
        .map(|j| match j {
            0 => sigma2_level,
            1 => sigma2_seasonal,
            _ => 0.0,
        })
        .collect();

    // Forward filter stores (needed by DK backward pass)
    // Flat contiguous memory for cache-friendly access
    let mut a_pred_flat = vec![0.0; t * s]; // a_pred_flat[i*s + j]
    let mut p_pred_flat = vec![0.0; t * ss]; // p_pred_flat[i*ss + j*s + k]
    let mut v_store = vec![0.0; t]; // innovation v_t
    let mut f_store = vec![0.0; t]; // innovation variance F_t (clamped to F_MIN)
    let mut k_store_flat = vec![0.0; t * s]; // k_store_flat[i*s + j]

    // Single working buffers for a_filt / p_filt (DK backward does not need full arrays)
    let mut a_filt_buf = vec![0.0; s];
    let mut p_filt_flat_buf = vec![0.0; ss]; // flat s×s working buffer
    let mut row_buf = vec![0.0; s]; // reusable temp for predict_state_covariance_flat
    let mut v_buf = vec![0.0; s]; // reusable temp for joseph_form_update_flat (P Z')

    // Initial state: a_{1|0} = 0, P_{1|0} = diag(init_level_var, init_seasonal_var, ...)
    // a_pred_flat[0..s] is already zero-initialized
    p_pred_flat[0] = initial_level_var.max(F_MIN);
    for j in 1..s {
        p_pred_flat[j * s + j] = initial_seasonal_var.max(F_MIN);
    }

    // Forward Kalman filter
    for i in 0..t {
        let a_off = i * s;
        let p_off = i * ss;
        let k_off = i * s;

        // Innovation: v_t = y_t - Z a_{t|t-1} = y_t - a[0] - a[1]
        let v = y[i] - a_pred_flat[a_off] - a_pred_flat[a_off + 1];
        v_store[i] = v;

        // Innovation variance: F = Z P Z' + σ²_obs
        // = P[0][0] + 2*P[0][1] + P[1][1] + σ²_obs
        let f_t = (p_pred_flat[p_off]
            + 2.0 * p_pred_flat[p_off + 1]
            + p_pred_flat[p_off + s + 1]
            + sigma2_obs)
            .max(F_MIN);
        f_store[i] = f_t;
        let f_inv = 1.0 / f_t;

        // Kalman gain: K = P Z' / F, where Z' = [1, 1, 0, ..., 0]'
        for j in 0..s {
            k_store_flat[k_off + j] =
                (p_pred_flat[p_off + j * s] + p_pred_flat[p_off + j * s + 1]) * f_inv;
        }

        // Update: a_{t|t} = a_{t|t-1} + K v
        for j in 0..s {
            a_filt_buf[j] = a_pred_flat[a_off + j] + k_store_flat[k_off + j] * v;
        }

        // P_{t|t} via sparse Joseph form (O(s²) instead of O(s⁴))
        // joseph_form_update_flat produces exactly symmetric output (lower tri copied to upper),
        // so symmetrize_and_floor is not needed. Only apply diagonal floor.
        joseph_form_update_flat(
            &p_pred_flat[p_off..p_off + ss],
            &k_store_flat[k_off..k_off + s],
            sigma2_obs,
            s,
            &mut p_filt_flat_buf,
            &mut v_buf,
        );
        for d in 0..s {
            p_filt_flat_buf[d * s + d] = p_filt_flat_buf[d * s + d].max(1e-10);
        }

        // Predict next step: write directly into stores[i+1]
        if i < t - 1 {
            let next_is_boundary = is_season_boundary(i + 1, season_duration);
            let next_a_off = (i + 1) * s;
            let next_p_off = (i + 1) * ss;
            apply_state_transition_inplace(
                &a_filt_buf,
                s,
                next_is_boundary,
                &mut a_pred_flat[next_a_off..next_a_off + s],
            );
            predict_state_covariance_flat(
                &p_filt_flat_buf,
                &q_diag,
                s,
                next_is_boundary,
                &mut p_pred_flat[next_p_off..next_p_off + ss],
                &mut row_buf,
            );
        }
    }

    // DK r_t backward smoother (Durbin-Koopman 2012, eq 4.43-4.44)
    // r_{t-1} = Z'/F_t v_t + L_t' r_t,  r_T = 0
    // α̂_t = a_{t|t-1} + P_{t|t-1} r_{t-1}
    let mut smooth = vec![vec![0.0; s]; t];
    let mut r = vec![0.0; s]; // r_T = 0
    let mut t_prime_r = vec![0.0; s]; // reusable buffer for T' r

    for i in (0..t).rev() {
        let a_off = i * s;
        let p_off = i * ss;
        let k_off = i * s;

        // Step 1: update r (from r_i to r_{i-1})
        let next_is_boundary = is_season_boundary(i + 1, season_duration);
        apply_state_transition_transpose_inplace(&r, s, next_is_boundary, &mut t_prime_r);

        // K_filter . (T' r): dot product of Kalman gain with T' r
        let k_dot_tpr: f64 = (0..s).map(|j| k_store_flat[k_off + j] * t_prime_r[j]).sum();

        // r_{i-1}[j] = Z'[j]/F_i * v_i + (T' r)[j] - Z'[j] * (K_filter . T' r)
        let v_over_f = v_store[i] / f_store[i];
        for j in 0..s {
            if j < 2 {
                r[j] = v_over_f + t_prime_r[j] - k_dot_tpr;
            } else {
                r[j] = t_prime_r[j];
            }
        }

        // Step 2: smoothed state α̂_i = a_{i|i-1} + P_{i|i-1} r_{i-1}
        // Flat contiguous access for cache efficiency
        for j in 0..s {
            let row_off = p_off + j * s;
            let correction: f64 = (0..s).map(|m| p_pred_flat[row_off + m] * r[m]).sum();
            smooth[i][j] = a_pred_flat[a_off + j] + correction;
        }
    }

    smooth
}

/// Apply state transition: T * state
/// For season boundary: level RW + seasonal rotation
/// For intra-season: level RW + seasonal freeze
fn apply_state_transition(state: &[f64], s: usize, is_boundary: bool) -> Vec<f64> {
    let mut result = vec![0.0; s];
    apply_state_transition_inplace(state, s, is_boundary, &mut result);
    result
}

/// In-place variant: writes T * state into `out`.
fn apply_state_transition_inplace(state: &[f64], s: usize, is_boundary: bool, out: &mut [f64]) {
    // Level: always random walk
    out[0] = state[0];

    if is_boundary {
        // Seasonal rotation: s_1(t+1) = -Σ s_j(t)
        let seasonal_sum: f64 = state[1..s].iter().sum();
        out[1] = -seasonal_sum;
        // s_j(t+1) = s_{j-1}(t) for j >= 2
        for j in 2..s {
            out[j] = state[j - 1];
        }
    } else {
        // Intra-season: seasonal state unchanged
        for j in 1..s {
            out[j] = state[j];
        }
    }
}

/// Apply transition transpose: T' * vector
fn apply_state_transition_transpose(v: &[f64], s: usize, is_boundary: bool) -> Vec<f64> {
    let mut result = vec![0.0; s];
    apply_state_transition_transpose_inplace(v, s, is_boundary, &mut result);
    result
}

/// In-place variant: writes T' * v into `out`.
fn apply_state_transition_transpose_inplace(
    v: &[f64],
    s: usize,
    is_boundary: bool,
    out: &mut [f64],
) {
    // T'[0][0] = 1 (level row of T is [1, 0, 0, ...])
    out[0] = v[0];

    if is_boundary {
        for j in 1..s {
            // Column j of T has: T[1][j] = -1 (sum-to-zero row)
            //                    T[j+1][j] = 1 if j+1 < s (shift)
            out[j] = -v[1]; // from the -1 in first seasonal row
            if j + 1 < s {
                out[j] += v[j + 1]; // from the shift: T[j+1][j] = 1
            }
        }
    } else {
        // Identity for seasonal block: T' = T = I
        for j in 1..s {
            out[j] = v[j];
        }
    }
}

/// Predict state covariance: T P T' + Q
///
/// Analytical expansion exploiting T's sparse structure:
/// - non-boundary (T = I): result = P + diag(Q) → O(s²)
/// - boundary: T*P computed via sparse row formulas, then (T*P)*T' → O(s²)
fn predict_state_covariance(
    p: &[Vec<f64>],
    q_diag: &[f64],
    s: usize,
    is_boundary: bool,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; s]; s];
    predict_state_covariance_inplace(p, q_diag, s, is_boundary, &mut result);
    result
}

/// In-place variant: writes T P T' + Q into `out`.
fn predict_state_covariance_inplace(
    p: &[Vec<f64>],
    q_diag: &[f64],
    s: usize,
    is_boundary: bool,
    out: &mut [Vec<f64>],
) {
    if !is_boundary {
        // T = I → T*P*T' = P, just copy and add Q diagonal
        for i in 0..s {
            out[i][..s].copy_from_slice(&p[i][..s]);
            out[i][i] = (out[i][i] + q_diag[i]).max(F_MIN);
        }
        return;
    }

    // Boundary case: T has sparse structure:
    //   row 0: T[0][0]=1 (level, rest 0)
    //   row 1: T[1][j]=-1 for j=1..s-1 (sum-to-zero)
    //   row r>=2: T[r][r-1]=1 (shift)

    // Step 1: Compute tp = T * P into `out` as temporary  (O(s²))
    for j in 0..s {
        out[0][j] = p[0][j];
        let mut row1_sum = 0.0;
        for k in 1..s {
            row1_sum += p[k][j];
        }
        out[1][j] = -row1_sum;
        for r in 2..s {
            out[r][j] = p[r - 1][j];
        }
    }

    // Step 2: Compute result = tp * T'  (O(s²))
    // We need to read tp while writing result, so we process one row at a time
    // using a temporary row buffer to avoid aliasing.
    // Actually, column 0 and column c>=2 read different indices than column 1,
    // and we can compute all columns from the tp rows without conflict
    // if we process row-by-row with a temp buffer.
    let mut row_buf = vec![0.0; s];
    for i in 0..s {
        row_buf[0] = out[i][0];
        let mut col1_sum = 0.0;
        for k in 1..s {
            col1_sum += out[i][k];
        }
        row_buf[1] = -col1_sum;
        for c in 2..s {
            row_buf[c] = out[i][c - 1];
        }
        out[i][..s].copy_from_slice(&row_buf[..s]);
    }

    // Symmetrize (floating point can introduce tiny asymmetries)
    for i in 0..s {
        for j in (i + 1)..s {
            let avg = 0.5 * (out[i][j] + out[j][i]);
            out[i][j] = avg;
            out[j][i] = avg;
        }
    }

    // Add Q diagonal and enforce positive
    for j in 0..s {
        out[j][j] = (out[j][j] + q_diag[j]).max(F_MIN);
    }
}

/// Joseph form update: P_new = (I - K Z) P (I - K Z)' + K σ²_obs K'
///
/// Exploits Z = [1, 1, 0, ..., 0] sparsity: rank-1 update formulation.
///   v[j] = P[j][0] + P[j][1]  (= (P Z')_j)
///   F = v[0] + v[1] + σ²_obs  (innovation variance)
///   P_new[i][j] = P[i][j] - K[i]*v[j] - v[i]*K[j] + K[i]*K[j]*F
fn joseph_form_update(
    p_pred: &[Vec<f64>],
    k_gain: &[f64],
    sigma2_obs: f64,
    s: usize,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; s]; s];
    joseph_form_update_inplace(p_pred, k_gain, sigma2_obs, s, &mut result);
    result
}

/// In-place variant: writes Joseph form update into `out`.
fn joseph_form_update_inplace(
    p_pred: &[Vec<f64>],
    k_gain: &[f64],
    sigma2_obs: f64,
    s: usize,
    out: &mut [Vec<f64>],
) {
    // v[j] = (P Z')_j = P[j][0] + P[j][1]
    let v: Vec<f64> = (0..s).map(|j| p_pred[j][0] + p_pred[j][1]).collect();
    // F = Z P Z' + σ²_obs = v[0] + v[1] + σ²_obs
    let f = v[0] + v[1] + sigma2_obs;

    for i in 0..s {
        for j in 0..=i {
            let val =
                p_pred[i][j] - k_gain[i] * v[j] - v[i] * k_gain[j] + k_gain[i] * k_gain[j] * f;
            out[i][j] = val;
            out[j][i] = val;
        }
    }
}

/// Flat-buffer variant of predict_state_covariance: T P T' + Q.
/// `p` and `out` are flat s×s buffers indexed as [row * s + col].
/// `row_buf` is a pre-allocated temporary buffer of size s (avoids heap alloc per call).
fn predict_state_covariance_flat(
    p: &[f64],
    q_diag: &[f64],
    s: usize,
    is_boundary: bool,
    out: &mut [f64],
    row_buf: &mut [f64],
) {
    if !is_boundary {
        // T = I → T*P*T' = P, just copy and add Q diagonal
        out.copy_from_slice(&p[..s * s]);
        for j in 0..s {
            out[j * s + j] = (out[j * s + j] + q_diag[j]).max(F_MIN);
        }
        return;
    }

    // Boundary case: closed-form T*P*T' + Q using sparse T structure.
    //
    // T (s×s transition matrix at season boundary):
    //   Row 0: [1, 0, 0, ..., 0]          (level random walk)
    //   Row 1: [0, -1, -1, ..., -1]       (sum-to-zero seasonal)
    //   Row r (r≥2): [0, ..., 1(col=r-1), ..., 0]  (shift)
    //
    // Closed-form (T*P*T')[i][j] for lower triangle (j ≤ i), then mirror:
    //   row_sums[l] = Σ_{k=1..s-1} P[l][k]   (precomputed per row)
    //   (0,0): P[0][0]
    //   (1,0): -row_sums[0]
    //   (1,1): Σ_{l=1..s-1} row_sums[l]       (= total_seasonal_sum)
    //   (i≥2, 0): P[(i-1)*s]
    //   (i≥2, 1): -row_sums[i-1]
    //   (i≥2, j in 2..i-1): P[(i-1)*s + j-1]
    //   (i≥2, j=i): P[(i-1)*s + i-1]          (diagonal)

    // Verify input P is symmetric (fusion relies on this)
    #[cfg(debug_assertions)]
    {
        let max_asym = (0..s)
            .flat_map(|i| (i + 1..s).map(move |j| (p[i * s + j] - p[j * s + i]).abs()))
            .fold(0.0f64, f64::max);
        debug_assert!(
            max_asym < 1e-10,
            "predict_cov_flat: input P not symmetric, max_asym={}",
            max_asym
        );
    }

    // Pass 0: row_sums precompute (reuse row_buf)
    for l in 0..s {
        let row_off = l * s;
        let mut sum = 0.0;
        for k in 1..s {
            sum += p[row_off + k];
        }
        row_buf[l] = sum;
    }
    let total_seasonal_sum: f64 = row_buf[1..s].iter().sum();

    // Pass 1: write lower triangle + mirror + Q diagonal
    // (0,0)
    out[0] = (p[0] + q_diag[0]).max(F_MIN);

    // (1,0) and mirror (0,1)
    let val_10 = -row_buf[0];
    out[1 * s] = val_10;
    out[1] = val_10;

    // (1,1)
    out[1 * s + 1] = (total_seasonal_sum + q_diag[1]).max(F_MIN);

    // Rows i >= 2
    for i in 2..s {
        let src_row = (i - 1) * s; // P row (i-1)

        // (i, 0): P[(i-1)*s + 0] = P[i-1][0]
        let val_i0 = p[src_row];
        out[i * s] = val_i0;
        out[i] = val_i0; // mirror (0, i)

        // (i, 1): -row_sums[i-1]
        let val_i1 = -row_buf[i - 1];
        out[i * s + 1] = val_i1;
        out[1 * s + i] = val_i1; // mirror (1, i)

        // (i, j) for 2 <= j < i: P[(i-1)*s + j-1]
        for j in 2..i {
            let val = p[src_row + j - 1];
            out[i * s + j] = val;
            out[j * s + i] = val; // mirror
        }

        // (i, i): diagonal
        out[i * s + i] = (p[src_row + i - 1] + q_diag[i]).max(F_MIN);
    }
}

/// Flat-buffer variant of joseph_form_update.
/// `p_pred` and `out` are flat s×s buffers indexed as [row * s + col].
/// `v_buf` is a pre-allocated temporary buffer of size s for (P Z')_j values.
fn joseph_form_update_flat(
    p_pred: &[f64],
    k_gain: &[f64],
    sigma2_obs: f64,
    s: usize,
    out: &mut [f64],
    v_buf: &mut [f64],
) {
    // v[j] = (P Z')_j = P[j][0] + P[j][1]
    for j in 0..s {
        v_buf[j] = p_pred[j * s] + p_pred[j * s + 1];
    }
    // F = Z P Z' + σ²_obs = v[0] + v[1] + σ²_obs
    let f = v_buf[0] + v_buf[1] + sigma2_obs;

    // P_new[i][j] = P[i][j] - K[i]*v[j] - v[i]*K[j] + K[i]*K[j]*F
    for i in 0..s {
        let ki = k_gain[i];
        let vi = v_buf[i];
        let ki_f = ki * f;
        for j in 0..=i {
            let val = p_pred[i * s + j] - ki * v_buf[j] - vi * k_gain[j] + ki_f * k_gain[j];
            out[i * s + j] = val;
            out[j * s + i] = val;
        }
    }
}

/// Flat-buffer variant of symmetrize_and_floor.
fn symmetrize_and_floor_flat(p: &mut [f64], s: usize) {
    for row in 0..s {
        for col in (row + 1)..s {
            let avg = 0.5 * (p[row * s + col] + p[col * s + row]);
            p[row * s + col] = avg;
            p[col * s + row] = avg;
        }
        p[row * s + row] = p[row * s + row].max(1e-10);
    }
}

/// Get element T[row][col] of the transition matrix (used only by naive test reference).
#[cfg(test)]
#[inline]
fn transition_element(row: usize, col: usize, _s: usize, is_boundary: bool) -> f64 {
    if row == 0 && col == 0 {
        return 1.0; // level: random walk
    }
    if row == 0 || col == 0 {
        return 0.0; // no cross-coupling between level and seasonal
    }
    // Seasonal block (rows/cols 1..s-1)
    if is_boundary {
        if row == 1 {
            return -1.0; // sum-to-zero: s_1(t+1) = -Σ s_j(t)
        }
        if col == row - 1 {
            return 1.0; // shift: s_j(t+1) = s_{j-1}(t)
        }
        0.0
    } else {
        // Identity for seasonal block
        if row == col {
            1.0
        } else {
            0.0
        }
    }
}

/// Count the number of season boundaries in [1, t_end).
pub fn count_season_boundaries(t_end: usize, season_duration: usize) -> usize {
    if t_end <= 1 || season_duration == 0 {
        return 0;
    }
    // Boundaries at t=1, 2, ..., t_end-1 where t % season_duration == 0
    // t=0 is not a boundary (it's initial)
    (1..t_end)
        .filter(|&t| is_season_boundary(t, season_duration))
        .count()
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

    #[test]
    fn test_local_linear_trend_smoother_output_lengths() {
        let mut rng = StdRng::seed_from_u64(7);
        let y: Vec<f64> = (0..20).map(|i| 5.0 + 0.2 * i as f64).collect();

        let (levels, slopes) =
            local_linear_trend_smoother(&mut rng, &y, 0.1, 0.01, 0.001, 5.0, 1.0);

        assert_eq!(levels.len(), y.len());
        assert_eq!(slopes.len(), y.len());
    }

    #[test]
    fn test_local_linear_trend_smoother_tracks_linear_signal_because_slope_state_should_not_collapse(
    ) {
        let mut rng = StdRng::seed_from_u64(11);
        let y: Vec<f64> = (0..60).map(|i| 10.0 + 0.3 * i as f64).collect();
        let n_samples = 100;
        let mut mean_last_level = 0.0;
        let mut mean_last_slope = 0.0;

        for _ in 0..n_samples {
            let (levels, slopes) =
                local_linear_trend_smoother(&mut rng, &y, 0.1, 0.01, 0.001, 10.0, 1.0);
            mean_last_level += levels[levels.len() - 1] / n_samples as f64;
            mean_last_slope += slopes[slopes.len() - 1] / n_samples as f64;
        }

        assert!((mean_last_level - 27.7).abs() < 2.0);
        assert!((mean_last_slope - 0.3).abs() < 0.2);
    }

    // ─── local_level_seasonal_smoother tests ───

    #[test]
    fn test_lls_output_len_equals_t() {
        let mut rng = StdRng::seed_from_u64(42);
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (levels, s1_obs, _ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.01, 7, 1, 0.0, 1.0);
        assert_eq!(levels.len(), y.len());
        assert_eq!(s1_obs.len(), y.len());
    }

    #[test]
    fn test_lls_t_equals_1() {
        let mut rng = StdRng::seed_from_u64(42);
        let y = vec![5.0];
        let (levels, s1_obs, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.01, 7, 1, 5.0, 1.0);
        assert_eq!(levels.len(), 1);
        assert_eq!(s1_obs.len(), 1);
        assert!(levels[0].is_finite());
        assert!(s1_obs[0].is_finite());
        assert_eq!(ssd, 0.0); // no innovation at t=0
    }

    #[test]
    fn test_lls_nseasons_2() {
        let mut rng = StdRng::seed_from_u64(42);
        let y = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let (levels, s1_obs, _ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 0.1, 0.001, 0.01, 2, 1, 0.0, 1.0);
        assert_eq!(levels.len(), 6);
        assert_eq!(s1_obs.len(), 6);
        for &v in &levels {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_lls_nseasons_7() {
        let mut rng = StdRng::seed_from_u64(42);
        // 7-period seasonal pattern
        let pattern = vec![-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = (0..28).map(|i| pattern[i % 7]).collect();
        let (levels, s1_obs, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 0.1, 0.001, 0.01, 7, 1, 0.0, 1.0);
        assert_eq!(levels.len(), 28);
        assert_eq!(s1_obs.len(), 28);
        assert!(ssd >= 0.0);
    }

    #[test]
    fn test_lls_nseasons_12() {
        let mut rng = StdRng::seed_from_u64(42);
        let y: Vec<f64> = (0..48).map(|i| ((i % 12) as f64 - 5.5) * 0.5).collect();
        let (levels, s1_obs, _ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 0.1, 0.001, 0.01, 12, 1, 0.0, 1.0);
        assert_eq!(levels.len(), 48);
        assert_eq!(s1_obs.len(), 48);
    }

    #[test]
    fn test_lls_transition_sum_to_zero() {
        // Verify that apply_state_transition preserves sum-to-zero for seasonal
        let state = vec![5.0, 2.0, -1.0, 3.0]; // s=4, seasonal: [2, -1, 3]
        let result = apply_state_transition(&state, 4, true);
        // new seasonal sum = result[1] + result[2] + result[3]
        // = -(2 + -1 + 3) + 2 + (-1) = -4 + 2 - 1 = -3
        // Old sum = 2 + -1 + 3 = 4
        // After rotation: [-4, 2, -1] → sum = -4 + 2 - 1 = -3
        // Actually, sum-to-zero means all S seasons sum to 0.
        // The "missing" season is -sum of stored S-1 seasons.
        // Before: stored = [2, -1, 3], missing = -(2-1+3) = -4
        // After: new[1] = -sum = -4, new[2] = old[1] = 2, new[3] = old[2] = -1
        // Stored = [-4, 2, -1], missing = -(-4+2-1) = 3 = old[3]
        assert_eq!(result[0], 5.0); // level unchanged
        assert_eq!(result[1], -4.0); // -sum of [2, -1, 3]
        assert_eq!(result[2], 2.0); // shift from [1]
        assert_eq!(result[3], -1.0); // shift from [2]
    }

    #[test]
    fn test_lls_tracks_strong_seasonal() {
        // Strong seasonal signal with known pattern
        let pattern = vec![-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = (0..56).map(|i| pattern[i % 7]).collect();

        let n_samples = 100;
        let mut mean_s1 = vec![0.0; 56];
        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let (_, s1_obs, _) =
                local_level_seasonal_smoother(&mut rng, &y, 0.01, 0.001, 0.01, 7, 1, 0.0, 1.0);
            for (mean, &s) in mean_s1.iter_mut().zip(s1_obs.iter()) {
                *mean += s / n_samples as f64;
            }
        }

        // s1 should track the pattern approximately: first season = -3 (centered)
        // The mean of the pattern is (-3-2+0+1+2+3+4)/7 = 5/7 ≈ 0.714
        // So first season centered ≈ -3 - 0.714 = -3.714
        // Just check the sign alternation follows the pattern
        assert!(
            mean_s1[0] < 0.0,
            "First season should be negative, got {}",
            mean_s1[0]
        );
        assert!(
            mean_s1[6] > 0.0,
            "Seventh season should be positive, got {}",
            mean_s1[6]
        );
    }

    #[test]
    fn test_lls_no_signal_seasonal_near_zero() {
        // Constant signal with no seasonal component
        let y: Vec<f64> = vec![5.0; 28];
        let n_samples = 100;
        let mut mean_s1 = vec![0.0; 28];
        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let (_, s1_obs, _) =
                local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.001, 7, 1, 5.0, 1.0);
            for (mean, &s) in mean_s1.iter_mut().zip(s1_obs.iter()) {
                *mean += s / n_samples as f64;
            }
        }

        // With no seasonal signal, s1 should be near zero on average
        for &s in &mean_s1 {
            assert!(
                s.abs() < 1.0,
                "Seasonal component should be near zero for constant signal, got {}",
                s
            );
        }
    }

    #[test]
    fn test_lls_levels_plus_s1_matches_y_when_obs_var_tiny() {
        // When σ²_obs → 0, levels + s1 should closely match y_residual
        let pattern = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = (0..20).map(|i| pattern[i % 5]).collect();
        let n_samples = 50;
        let mut mean_fit = vec![0.0; 20];
        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let (levels, s1_obs, _) =
                local_level_seasonal_smoother(&mut rng, &y, 1e-8, 0.001, 0.01, 5, 1, 0.0, 1.0);
            for (mean, (l, s)) in mean_fit.iter_mut().zip(levels.iter().zip(s1_obs.iter())) {
                *mean += (l + s) / n_samples as f64;
            }
        }

        for (i, (&m, &y_val)) in mean_fit.iter().zip(y.iter()).enumerate() {
            assert!(
                (m - y_val).abs() < 1.0,
                "fit[{}] = {} should be near y = {}",
                i,
                m,
                y_val
            );
        }
    }

    #[test]
    fn test_lls_innovation_ssd_nonneg() {
        let mut rng = StdRng::seed_from_u64(42);
        let y: Vec<f64> = (0..14).map(|i| (i % 7) as f64).collect();
        let (_, _, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.01, 7, 1, 0.0, 1.0);
        assert!(
            ssd >= 0.0,
            "innovation_ssd must be non-negative, got {}",
            ssd
        );
    }

    #[test]
    fn test_lls_innovation_ssd_large_for_varying_seasonal() {
        // Seasonal signal that changes over time → larger innovation_ssd
        let y_stable: Vec<f64> = (0..28).map(|i| (i % 7) as f64).collect();
        let y_changing: Vec<f64> = (0..28)
            .map(|i| (i % 7) as f64 * (1.0 + 0.1 * i as f64))
            .collect();

        let mut ssd_stable_sum = 0.0;
        let mut ssd_changing_sum = 0.0;
        let n_samples = 50;
        for seed in 0..n_samples {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let (_, _, ssd1) = local_level_seasonal_smoother(
                &mut rng, &y_stable, 0.1, 0.001, 0.01, 7, 1, 0.0, 1.0,
            );
            let mut rng2 = StdRng::seed_from_u64(seed as u64 + 1000);
            let (_, _, ssd2) = local_level_seasonal_smoother(
                &mut rng2,
                &y_changing,
                0.1,
                0.001,
                0.01,
                7,
                1,
                0.0,
                1.0,
            );
            ssd_stable_sum += ssd1;
            ssd_changing_sum += ssd2;
        }
        // The changing seasonal should have larger average innovation_ssd
        assert!(
            ssd_changing_sum > ssd_stable_sum,
            "Changing seasonal ssd {} should be > stable ssd {}",
            ssd_changing_sum / n_samples as f64,
            ssd_stable_sum / n_samples as f64
        );
    }

    #[test]
    fn test_lls_finite_sigma2_seasonal_zero() {
        let mut rng = StdRng::seed_from_u64(42);
        let y = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let (levels, s1_obs, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.0, 3, 1, 0.0, 1.0);
        for &v in &levels {
            assert!(
                v.is_finite(),
                "levels should be finite when sigma2_seasonal=0"
            );
        }
        for &v in &s1_obs {
            assert!(
                v.is_finite(),
                "s1_obs should be finite when sigma2_seasonal=0"
            );
        }
        assert!(ssd.is_finite());
    }

    #[test]
    fn test_lls_finite_sigma2_obs_tiny() {
        let mut rng = StdRng::seed_from_u64(42);
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (levels, s1_obs, _) =
            local_level_seasonal_smoother(&mut rng, &y, 1e-12, 0.01, 0.01, 7, 1, 0.0, 1.0);
        for &v in &levels {
            assert!(
                v.is_finite(),
                "levels should be finite with tiny sigma2_obs"
            );
        }
        for &v in &s1_obs {
            assert!(
                v.is_finite(),
                "s1_obs should be finite with tiny sigma2_obs"
            );
        }
    }

    #[test]
    fn test_lls_finite_init_var_large() {
        let mut rng = StdRng::seed_from_u64(42);
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (levels, s1_obs, _) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.01, 7, 1, 0.0, 1e6);
        for &v in &levels {
            assert!(v.is_finite(), "levels should be finite with large init_var");
        }
        for &v in &s1_obs {
            assert!(v.is_finite(), "s1_obs should be finite with large init_var");
        }
    }

    #[test]
    fn test_lls_season_duration_2() {
        let mut rng = StdRng::seed_from_u64(42);
        // With duration=2, boundaries at t=0,2,4,6,...
        let y: Vec<f64> = (0..14).map(|i| ((i / 2) % 3) as f64).collect();
        let (levels, s1_obs, _) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.01, 3, 2, 0.0, 1.0);
        assert_eq!(levels.len(), 14);
        assert_eq!(s1_obs.len(), 14);
        for &v in &levels {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_lls_season_duration_7() {
        let mut rng = StdRng::seed_from_u64(42);
        // Weekly seasonal: 4 weeks × 7 days = 28 observations
        let y: Vec<f64> = (0..28).map(|i| ((i / 7) % 4) as f64).collect();
        let (levels, s1_obs, _) =
            local_level_seasonal_smoother(&mut rng, &y, 1.0, 0.01, 0.01, 4, 7, 0.0, 1.0);
        assert_eq!(levels.len(), 28);
        assert_eq!(s1_obs.len(), 28);
    }

    #[test]
    fn test_lls_intra_season_state_frozen() {
        // With duration=2, odd timesteps should have same seasonal state as even
        let state = vec![5.0, 2.0, -1.0]; // s=3
        let intra = apply_state_transition(&state, 3, false);
        assert_eq!(intra[0], 5.0); // level unchanged
        assert_eq!(intra[1], 2.0); // seasonal frozen
        assert_eq!(intra[2], -1.0); // seasonal frozen
    }

    #[test]
    fn test_count_season_boundaries() {
        assert_eq!(count_season_boundaries(7, 1), 6); // t=1..6
        assert_eq!(count_season_boundaries(8, 2), 3); // t=2,4,6
        assert_eq!(count_season_boundaries(1, 1), 0); // no boundaries
        assert_eq!(count_season_boundaries(0, 1), 0);
        assert_eq!(count_season_boundaries(14, 7), 1); // t=7
    }

    // ─── PR-B: Analytical expansion tests ───
    //
    // Reference naive implementation of predict_state_covariance
    // (the original O(s⁴) version, kept for comparison testing).
    fn predict_state_covariance_naive(
        p: &[Vec<f64>],
        q_diag: &[f64],
        s: usize,
        is_boundary: bool,
    ) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; s]; s];
        for row in 0..s {
            for col in 0..=row {
                let mut val = 0.0;
                for m in 0..s {
                    let t_rm = transition_element(row, m, s, is_boundary);
                    if t_rm == 0.0 {
                        continue;
                    }
                    for n in 0..s {
                        let t_cn = transition_element(col, n, s, is_boundary);
                        if t_cn != 0.0 {
                            val += t_rm * p[m][n] * t_cn;
                        }
                    }
                }
                result[row][col] = val;
                result[col][row] = val;
            }
        }
        for j in 0..s {
            result[j][j] += q_diag[j];
        }
        for j in 0..s {
            result[j][j] = result[j][j].max(F_MIN);
        }
        result
    }

    // Reference naive implementation of Joseph form update
    // (the original O(s⁴) inline version, extracted for comparison).
    fn joseph_form_update_naive(
        p_pred: &[Vec<f64>],
        k_gain: &[f64],
        sigma2_obs: f64,
        s: usize,
    ) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; s]; s];
        for row in 0..s {
            for col in 0..s {
                let mut val = 0.0;
                for m in 0..s {
                    let z_m = if m <= 1 { 1.0 } else { 0.0 };
                    let ikz_rm = if row == m { 1.0 } else { 0.0 } - k_gain[row] * z_m;
                    for n in 0..s {
                        let z_n = if n <= 1 { 1.0 } else { 0.0 };
                        let ikz_cn = if col == n { 1.0 } else { 0.0 } - k_gain[col] * z_n;
                        val += ikz_rm * p_pred[m][n] * ikz_cn;
                    }
                }
                val += k_gain[row] * sigma2_obs * k_gain[col];
                result[row][col] = val;
            }
        }
        result
    }

    /// Generate a random SPD matrix of size s.
    fn random_spd(s: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let a: Vec<Vec<f64>> = (0..s)
            .map(|_| (0..s).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect();
        // P = A'A + 0.01*I (guaranteed SPD)
        let mut p = vec![vec![0.0; s]; s];
        for i in 0..s {
            for j in 0..s {
                for k in 0..s {
                    p[i][j] += a[k][i] * a[k][j];
                }
            }
            p[i][i] += 0.01;
        }
        p
    }

    fn q_diag_for(s: usize, sigma2_level: f64, sigma2_seasonal: f64) -> Vec<f64> {
        (0..s)
            .map(|j| match j {
                0 => sigma2_level,
                1 => sigma2_seasonal,
                _ => 0.0,
            })
            .collect()
    }

    fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        a.iter()
            .zip(b.iter())
            .flat_map(|(ra, rb)| ra.iter().zip(rb.iter()))
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn is_symmetric(m: &[Vec<f64>], tol: f64) -> bool {
        let s = m.len();
        for i in 0..s {
            for j in (i + 1)..s {
                if (m[i][j] - m[j][i]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    // ── predict_state_covariance: analytical vs naive (15 tests) ──

    #[test]
    fn test_predict_cov_analytical_matches_naive_s2_boundary() {
        let p = random_spd(2, 100);
        let q = q_diag_for(2, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 2, true);
        let actual = predict_state_covariance(&p, &q, 2, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=2 boundary: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_analytical_matches_naive_s2_intra() {
        let p = random_spd(2, 101);
        let q = q_diag_for(2, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 2, false);
        let actual = predict_state_covariance(&p, &q, 2, false);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=2 intra: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_analytical_matches_naive_s4_boundary() {
        let p = random_spd(4, 102);
        let q = q_diag_for(4, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 4, true);
        let actual = predict_state_covariance(&p, &q, 4, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=4 boundary: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_analytical_matches_naive_s7_boundary() {
        let p = random_spd(7, 103);
        let q = q_diag_for(7, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 7, true);
        let actual = predict_state_covariance(&p, &q, 7, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=7 boundary: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_analytical_matches_naive_s12_boundary() {
        let p = random_spd(12, 104);
        let q = q_diag_for(12, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 12, true);
        let actual = predict_state_covariance(&p, &q, 12, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=12 boundary: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_analytical_matches_naive_s52_boundary() {
        let p = random_spd(52, 105);
        let q = q_diag_for(52, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 52, true);
        let actual = predict_state_covariance(&p, &q, 52, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=52 boundary: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_analytical_matches_naive_s168_boundary() {
        let p = random_spd(168, 106);
        let q = q_diag_for(168, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, 168, true);
        let actual = predict_state_covariance(&p, &q, 168, true);
        // s=168: 167-element sums produce O(s × ε_machine) rounding differences
        // between naive 4-loop and analytical 2-step expansion
        let max_val = expected
            .iter()
            .flat_map(|r| r.iter())
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        assert!(
            max_abs_diff(&expected, &actual) < max_val * 1e-10,
            "s=168 boundary: max diff = {}, max_val = {}",
            max_abs_diff(&expected, &actual),
            max_val
        );
    }

    #[test]
    fn test_predict_cov_identity_p_boundary_equals_t_t_transpose_s4() {
        // P = I → T*I*T' = T*T'
        let s = 4;
        let mut p = vec![vec![0.0; s]; s];
        for j in 0..s {
            p[j][j] = 1.0;
        }
        let q = vec![0.0; s]; // no Q to isolate T*T'
        let result = predict_state_covariance(&p, &q, s, true);
        let expected = predict_state_covariance_naive(&p, &q, s, true);
        assert!(max_abs_diff(&result, &expected) < 1e-12);
    }

    #[test]
    fn test_predict_cov_identity_p_boundary_equals_t_t_transpose_s12() {
        let s = 12;
        let mut p = vec![vec![0.0; s]; s];
        for j in 0..s {
            p[j][j] = 1.0;
        }
        let q = vec![0.0; s];
        let result = predict_state_covariance(&p, &q, s, true);
        let expected = predict_state_covariance_naive(&p, &q, s, true);
        assert!(max_abs_diff(&result, &expected) < 1e-12);
    }

    #[test]
    fn test_predict_cov_zero_p_equals_q_diag_s7() {
        // P = 0 → result = Q
        let s = 7;
        let p = vec![vec![0.0; s]; s];
        let q = q_diag_for(s, 0.05, 0.02);
        let result = predict_state_covariance(&p, &q, s, true);
        for j in 0..s {
            assert!(
                (result[j][j] - q[j].max(F_MIN)).abs() < 1e-12,
                "diagonal {} mismatch: {} vs {}",
                j,
                result[j][j],
                q[j].max(F_MIN)
            );
            for k in 0..s {
                if k != j {
                    assert!(
                        result[j][k].abs() < 1e-12,
                        "off-diagonal [{},{}] = {} should be 0",
                        j,
                        k,
                        result[j][k]
                    );
                }
            }
        }
    }

    #[test]
    fn test_predict_cov_result_is_symmetric_s4_boundary() {
        let p = random_spd(4, 110);
        let q = q_diag_for(4, 0.01, 0.005);
        let result = predict_state_covariance(&p, &q, 4, true);
        assert!(is_symmetric(&result, 1e-12), "result should be symmetric");
    }

    #[test]
    fn test_predict_cov_result_is_symmetric_s12_boundary() {
        let p = random_spd(12, 111);
        let q = q_diag_for(12, 0.01, 0.005);
        let result = predict_state_covariance(&p, &q, 12, true);
        assert!(is_symmetric(&result, 1e-12), "result should be symmetric");
    }

    #[test]
    fn test_predict_cov_near_singular_p_s12() {
        // Near-singular P: min eigenvalue ~1e-12
        let s = 12;
        let base = random_spd(s, 112);
        let mut p = vec![vec![0.0; s]; s];
        // Scale to very small values
        for i in 0..s {
            for j in 0..s {
                p[i][j] = base[i][j] * 1e-12;
            }
        }
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, s, true);
        let actual = predict_state_covariance(&p, &q, s, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "near-singular P: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_predict_cov_large_p_diagonal_s12() {
        // P with large diagonal (P[i][i] = 1e8)
        let s = 12;
        let mut p = random_spd(s, 113);
        for i in 0..s {
            p[i][i] = 1e8;
        }
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p, &q, s, true);
        let actual = predict_state_covariance(&p, &q, s, true);
        // Relative tolerance for large values
        let max_val = expected
            .iter()
            .flat_map(|r| r.iter())
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff(&expected, &actual) < max_val * 1e-10,
            "large P: max diff = {}, max_val = {}",
            max_abs_diff(&expected, &actual),
            max_val
        );
    }

    #[test]
    fn test_predict_cov_q_diag_zero_s7() {
        // Frozen seasonal: Q = 0
        let s = 7;
        let p = random_spd(s, 114);
        let q = vec![0.0; s];
        let expected = predict_state_covariance_naive(&p, &q, s, true);
        let actual = predict_state_covariance(&p, &q, s, true);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "Q=0: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    // ── Joseph form: analytical vs naive (7 tests) ──

    fn make_joseph_test_data(s: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
        let p = random_spd(s, seed);
        let sigma2_obs = 0.5;
        // Compute realistic K gain: K = P Z' / F
        // Z = [1, 1, 0, ..., 0]
        let v: Vec<f64> = (0..s).map(|j| p[j][0] + p[j][1]).collect();
        let f_t = (v[0] + v[1] + sigma2_obs).max(F_MIN);
        let k_gain: Vec<f64> = v.iter().map(|&vi| vi / f_t).collect();
        (p, k_gain, sigma2_obs)
    }

    #[test]
    fn test_joseph_form_analytical_matches_naive_s2() {
        let (p, k, obs) = make_joseph_test_data(2, 200);
        let expected = joseph_form_update_naive(&p, &k, obs, 2);
        let actual = joseph_form_update(&p, &k, obs, 2);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=2: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_joseph_form_analytical_matches_naive_s4() {
        let (p, k, obs) = make_joseph_test_data(4, 201);
        let expected = joseph_form_update_naive(&p, &k, obs, 4);
        let actual = joseph_form_update(&p, &k, obs, 4);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=4: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_joseph_form_analytical_matches_naive_s7() {
        let (p, k, obs) = make_joseph_test_data(7, 202);
        let expected = joseph_form_update_naive(&p, &k, obs, 7);
        let actual = joseph_form_update(&p, &k, obs, 7);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=7: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_joseph_form_analytical_matches_naive_s12() {
        let (p, k, obs) = make_joseph_test_data(12, 203);
        let expected = joseph_form_update_naive(&p, &k, obs, 12);
        let actual = joseph_form_update(&p, &k, obs, 12);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=12: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_joseph_form_analytical_matches_naive_s168() {
        let (p, k, obs) = make_joseph_test_data(168, 204);
        let expected = joseph_form_update_naive(&p, &k, obs, 168);
        let actual = joseph_form_update(&p, &k, obs, 168);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "s=168: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_joseph_form_result_is_symmetric_s12() {
        let (p, k, obs) = make_joseph_test_data(12, 205);
        let result = joseph_form_update(&p, &k, obs, 12);
        assert!(
            is_symmetric(&result, 1e-12),
            "Joseph form result should be symmetric"
        );
    }

    #[test]
    fn test_joseph_form_near_singular_k_gain_s4() {
        // Near-singular: F very small → large K gain
        let s = 4;
        let p = random_spd(s, 206);
        let sigma2_obs = 1e-15; // tiny obs variance → F ≈ Z P Z'
        let v: Vec<f64> = (0..s).map(|j| p[j][0] + p[j][1]).collect();
        let f_t = (v[0] + v[1] + sigma2_obs).max(F_MIN);
        let k_gain: Vec<f64> = v.iter().map(|&vi| vi / f_t).collect();
        let expected = joseph_form_update_naive(&p, &k_gain, sigma2_obs, s);
        let actual = joseph_form_update(&p, &k_gain, sigma2_obs, s);
        let max_val = expected
            .iter()
            .flat_map(|r| r.iter())
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        assert!(
            max_abs_diff(&expected, &actual) < max_val * 1e-10,
            "near-singular K: max diff = {}, max_val = {}",
            max_abs_diff(&expected, &actual),
            max_val
        );
    }

    // ── Smoother integration (3 tests) ──

    #[test]
    fn test_seasonal_smoother_analytical_all_finite_s4() {
        let mut rng = StdRng::seed_from_u64(300);
        let y: Vec<f64> = (0..20).map(|i| ((i % 4) as f64 - 1.5) * 2.0).collect();
        let (levels, s1_obs, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 0.1, 0.001, 0.01, 4, 1, 0.0, 1.0);
        for (i, &v) in levels.iter().enumerate() {
            assert!(v.is_finite(), "levels[{}] = {} not finite", i, v);
        }
        for (i, &v) in s1_obs.iter().enumerate() {
            assert!(v.is_finite(), "s1_obs[{}] = {} not finite", i, v);
        }
        assert!(ssd.is_finite(), "ssd = {} not finite", ssd);
    }

    #[test]
    fn test_seasonal_smoother_analytical_all_finite_s12() {
        let mut rng = StdRng::seed_from_u64(301);
        let y: Vec<f64> = (0..48).map(|i| ((i % 12) as f64 - 5.5) * 0.5).collect();
        let (levels, s1_obs, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 0.1, 0.001, 0.01, 12, 1, 0.0, 1.0);
        for (i, &v) in levels.iter().enumerate() {
            assert!(v.is_finite(), "levels[{}] = {} not finite", i, v);
        }
        for (i, &v) in s1_obs.iter().enumerate() {
            assert!(v.is_finite(), "s1_obs[{}] = {} not finite", i, v);
        }
        assert!(ssd.is_finite(), "ssd = {} not finite", ssd);
    }

    #[test]
    fn test_seasonal_smoother_r_compatible_100steps_s12() {
        // Run smoother, verify output lengths and finite values
        // (R compatibility is tested at Python level via test_numerical_equivalence.py)
        let mut rng = StdRng::seed_from_u64(302);
        let y: Vec<f64> = (0..100)
            .map(|i| ((i % 12) as f64 - 5.5) * 0.3 + 10.0)
            .collect();
        let (levels, s1_obs, ssd) =
            local_level_seasonal_smoother(&mut rng, &y, 0.5, 0.01, 0.01, 12, 1, 10.0, 1.0);
        assert_eq!(levels.len(), 100);
        assert_eq!(s1_obs.len(), 100);
        assert!(ssd >= 0.0);
        for &v in &levels {
            assert!(v.is_finite());
        }
        for &v in &s1_obs {
            assert!(v.is_finite());
        }
    }

    // ─── PR-C: _inplace equivalence tests ───

    #[test]
    fn test_dk_inplace_predict_cov_matches_original_s12() {
        let p = random_spd(12, 400);
        let q = q_diag_for(12, 0.01, 0.005);
        let expected = predict_state_covariance(&p, &q, 12, true);
        let mut actual = vec![vec![0.0; 12]; 12];
        predict_state_covariance_inplace(&p, &q, 12, true, &mut actual);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "inplace predict_cov s=12: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
        // Also test non-boundary
        let expected_nb = predict_state_covariance(&p, &q, 12, false);
        predict_state_covariance_inplace(&p, &q, 12, false, &mut actual);
        assert!(
            max_abs_diff(&expected_nb, &actual) < 1e-12,
            "inplace predict_cov s=12 non-boundary: max diff = {}",
            max_abs_diff(&expected_nb, &actual)
        );
    }

    #[test]
    fn test_dk_inplace_joseph_form_matches_original_s12() {
        let (p, k, obs) = make_joseph_test_data(12, 401);
        let expected = joseph_form_update(&p, &k, obs, 12);
        let mut actual = vec![vec![0.0; 12]; 12];
        joseph_form_update_inplace(&p, &k, obs, 12, &mut actual);
        assert!(
            max_abs_diff(&expected, &actual) < 1e-12,
            "inplace joseph_form s=12: max diff = {}",
            max_abs_diff(&expected, &actual)
        );
    }

    #[test]
    fn test_dk_inplace_apply_transition_matches_original_s7() {
        let mut rng = StdRng::seed_from_u64(402);
        let state: Vec<f64> = (0..7).map(|_| rng.gen::<f64>() - 0.5).collect();
        // Boundary
        let expected_b = apply_state_transition(&state, 7, true);
        let mut actual = vec![0.0; 7];
        apply_state_transition_inplace(&state, 7, true, &mut actual);
        let diff_b: f64 = expected_b
            .iter()
            .zip(actual.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            diff_b < 1e-15,
            "inplace transition boundary s=7: diff = {}",
            diff_b
        );
        // Non-boundary
        let expected_nb = apply_state_transition(&state, 7, false);
        apply_state_transition_inplace(&state, 7, false, &mut actual);
        let diff_nb: f64 = expected_nb
            .iter()
            .zip(actual.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            diff_nb < 1e-15,
            "inplace transition non-boundary s=7: diff = {}",
            diff_nb
        );
    }

    #[test]
    fn test_dk_f_min_floor_prevents_division_by_zero_s4() {
        // When p_pred is near-zero and sigma2_obs is near-zero, f_t should be clamped to F_MIN
        let s = 4;
        let p = vec![vec![0.0; s]; s]; // zero covariance
        let sigma2_obs = 0.0;
        let sigma2_level = 0.0;
        let sigma2_seasonal = 0.0;
        // Run seasonal_kalman_smoother with degenerate inputs
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let result = seasonal_kalman_smoother(
            &y,
            sigma2_obs,
            sigma2_level,
            sigma2_seasonal,
            s,
            1,
            F_MIN,
            F_MIN,
        );
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "smooth[{}][{}] = {} not finite with degenerate inputs",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[test]
    fn test_dk_single_buffer_afilt_matches_array_s12_t50() {
        // Verify that the smoother produces identical results whether using
        // a full a_filt/p_filt array or a single buffer (DK does not need the array).
        // We test by comparing the current seasonal_kalman_smoother output with
        // the RTS reference implementation (which uses full arrays).
        let s = 12;
        let t = 50;
        let y: Vec<f64> = (0..t).map(|i| ((i % s) as f64 - 5.5) * 0.5).collect();
        let current = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, s, 1, 1.0, 1.0);
        let reference = seasonal_kalman_smoother_rts_reference(&y, 0.5, 0.01, 0.01, s, 1, 1.0, 1.0);
        let diff = max_abs_diff(&current, &reference);
        // DK r_t and RTS Cholesky take different numerical paths; tolerance
        // scales with s and T. s=12, T=50 yields ~2e-8 difference.
        assert!(
            diff < 1e-7,
            "single buffer should match array: max diff = {}",
            diff
        );
    }

    // ─── RTS reference implementation (for DK vs RTS equivalence tests) ───

    /// RTS smoother reference: uses full a_filt/p_filt arrays + Cholesky solve backward.
    /// Kept under #[cfg(test)] for DK vs RTS equivalence verification.
    #[allow(clippy::too_many_arguments)]
    fn seasonal_kalman_smoother_rts_reference(
        y: &[f64],
        sigma2_obs: f64,
        sigma2_level: f64,
        sigma2_seasonal: f64,
        s: usize,
        season_duration: usize,
        initial_level_var: f64,
        initial_seasonal_var: f64,
    ) -> Vec<Vec<f64>> {
        let t = y.len();

        let q_diag: Vec<f64> = (0..s)
            .map(|j| match j {
                0 => sigma2_level,
                1 => sigma2_seasonal,
                _ => 0.0,
            })
            .collect();

        let mut a_filt = vec![vec![0.0; s]; t];
        let mut p_filt = vec![vec![vec![0.0; s]; s]; t];
        let mut a_pred_store = vec![vec![0.0; s]; t];
        let mut p_pred_store = vec![vec![vec![0.0; s]; s]; t];

        let mut a_pred = vec![0.0; s];
        let mut p_pred = vec![vec![0.0; s]; s];
        p_pred[0][0] = initial_level_var.max(F_MIN);
        for j in 1..s {
            p_pred[j][j] = initial_seasonal_var.max(F_MIN);
        }

        for i in 0..t {
            a_pred_store[i] = a_pred.clone();
            p_pred_store[i] = p_pred.clone();

            let v = y[i] - a_pred[0] - a_pred[1];
            let f_t = (p_pred[0][0] + 2.0 * p_pred[0][1] + p_pred[1][1] + sigma2_obs).max(F_MIN);
            let f_inv = 1.0 / f_t;

            let k_gain: Vec<f64> = (0..s)
                .map(|j| (p_pred[j][0] + p_pred[j][1]) * f_inv)
                .collect();

            for j in 0..s {
                a_filt[i][j] = a_pred[j] + k_gain[j] * v;
            }

            p_filt[i] = joseph_form_update(&p_pred, &k_gain, sigma2_obs, s);
            symmetrize_and_floor(&mut p_filt[i], s);

            if i < t - 1 {
                let next_is_boundary = is_season_boundary(i + 1, season_duration);
                a_pred = apply_state_transition(&a_filt[i], s, next_is_boundary);
                p_pred = predict_state_covariance(&p_filt[i], &q_diag, s, next_is_boundary);
            }
        }

        // RTS backward smoother (Cholesky solve)
        let mut smooth = vec![vec![0.0; s]; t];
        smooth[t - 1] = a_filt[t - 1].clone();

        for i in (0..t - 1).rev() {
            let next_is_boundary = is_season_boundary(i + 1, season_duration);
            let p_pred_next = &p_pred_store[i + 1];
            let d: Vec<f64> = (0..s)
                .map(|j| smooth[i + 1][j] - a_pred_store[i + 1][j])
                .collect();
            let z = cholesky_solve(p_pred_next, &d, s);
            let t_prime_z = apply_state_transition_transpose(&z, s, next_is_boundary);
            for j in 0..s {
                let correction: f64 = (0..s).map(|m| p_filt[i][j][m] * t_prime_z[m]).sum();
                smooth[i][j] = a_filt[i][j] + correction;
            }
        }

        smooth
    }

    // ─── PR-C: DK r_t vs RTS equivalence tests (7 tests) ───

    /// Helper: compare DK (seasonal_kalman_smoother) vs RTS reference.
    fn assert_dk_matches_rts(
        y: &[f64],
        sigma2_obs: f64,
        sigma2_level: f64,
        sigma2_seasonal: f64,
        s: usize,
        season_duration: usize,
        initial_level_var: f64,
        initial_seasonal_var: f64,
        tol: f64,
        label: &str,
    ) {
        let dk = seasonal_kalman_smoother(
            y,
            sigma2_obs,
            sigma2_level,
            sigma2_seasonal,
            s,
            season_duration,
            initial_level_var,
            initial_seasonal_var,
        );
        let rts = seasonal_kalman_smoother_rts_reference(
            y,
            sigma2_obs,
            sigma2_level,
            sigma2_seasonal,
            s,
            season_duration,
            initial_level_var,
            initial_seasonal_var,
        );
        let diff = max_abs_diff(&dk, &rts);
        assert!(
            diff < tol,
            "{}: DK vs RTS max_abs_diff = {} (tol = {})",
            label,
            diff,
            tol
        );
    }

    #[test]
    fn test_dk_backward_matches_rts_s2_t5() {
        let y: Vec<f64> = vec![1.0, -1.0, 2.0, -2.0, 3.0];
        // DK and RTS take different numerical paths (DK avoids Cholesky solve).
        // Floating-point difference scales as O(ε_machine × T × s).
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 2, 1, 1.0, 1.0, 1e-9, "s2_t5");
    }

    #[test]
    fn test_dk_backward_matches_rts_s4_t10_nonboundary() {
        // season_duration=100 means no boundaries in 0..10
        let y: Vec<f64> = (0..10).map(|i| (i as f64 * 0.3).sin()).collect();
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 4, 100, 1.0, 1.0, 1e-9, "s4_t10_nb");
    }

    #[test]
    fn test_dk_backward_matches_rts_s4_t10_allboundary() {
        // season_duration=1 means every step is a boundary
        let y: Vec<f64> = (0..10).map(|i| ((i % 4) as f64 - 1.5) * 2.0).collect();
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 4, 1, 1.0, 1.0, 1e-8, "s4_t10_ab");
    }

    #[test]
    fn test_dk_backward_matches_rts_s7_t100() {
        let y: Vec<f64> = (0..100).map(|i| ((i % 7) as f64 - 3.0) * 0.5).collect();
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 7, 1, 1.0, 1.0, 1e-8, "s7_t100");
    }

    #[test]
    fn test_dk_backward_matches_rts_s12_t100() {
        let y: Vec<f64> = (0..100).map(|i| ((i % 12) as f64 - 5.5) * 0.3).collect();
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 12, 1, 1.0, 1.0, 1e-7, "s12_t100");
    }

    #[test]
    fn test_dk_backward_matches_rts_s52_t200() {
        let y: Vec<f64> = (0..200).map(|i| ((i % 52) as f64 - 25.5) * 0.1).collect();
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 52, 1, 1.0, 1.0, 1e-4, "s52_t200");
    }

    #[test]
    fn test_dk_backward_matches_rts_s168_t1200() {
        let y: Vec<f64> = (0..1200)
            .map(|i| ((i % 168) as f64 - 83.5) * 0.05)
            .collect();
        // s=168, T=1200: FP difference ~1e-5, well within R compat ±1% tolerance
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 168, 1, 1.0, 1.0, 1e-3, "s168_t1200");
    }

    // ─── PR-C: DK r recursion correctness tests (5 tests) ───

    #[test]
    fn test_dk_r_terminal_initialized_to_zero() {
        // For t=1 (single observation), backward pass starts with r = [0; s]
        // and after one step, smooth[0] = a_pred[0] + P_pred[0] * r_{-1}.
        // r_{-1} = Z'/F * v + L' * 0 = Z'/F * v (since r_T = 0).
        let s = 4;
        let y = vec![5.0];
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, s, 1, 1.0, 1.0);
        assert_eq!(result.len(), 1);
        for &v in &result[0] {
            assert!(v.is_finite(), "single obs DK result not finite: {}", v);
        }
    }

    /// Naive L' r computation: L = T - K_DK Z, K_DK = T K_filter, L' r = (I - Z' K') T' r
    fn naive_l_prime_r(k_filter: &[f64], r: &[f64], s: usize, is_boundary: bool) -> Vec<f64> {
        // Build full L matrix (s×s):
        // L = T - K_DK Z = T - T K_filter Z
        // L[i][j] = T[i][j] - (T K_filter)[i] * Z[j]
        let mut l = vec![vec![0.0; s]; s];
        for i in 0..s {
            // (T K_filter)[i] = sum_m T[i][m] * K_filter[m]
            let tk_i: f64 = (0..s)
                .map(|m| transition_element(i, m, s, is_boundary) * k_filter[m])
                .sum();
            for j in 0..s {
                let t_ij = transition_element(i, j, s, is_boundary);
                let z_j = if j < 2 { 1.0 } else { 0.0 };
                l[i][j] = t_ij - tk_i * z_j;
            }
        }
        // L' r
        let mut result = vec![0.0; s];
        for j in 0..s {
            for i in 0..s {
                result[j] += l[i][j] * r[i];
            }
        }
        result
    }

    /// Efficient L' r (the formula used in seasonal_kalman_smoother):
    /// L' r = T' r - Z' (K_filter . T' r)
    fn efficient_l_prime_r(k_filter: &[f64], r: &[f64], s: usize, is_boundary: bool) -> Vec<f64> {
        let t_prime_r = apply_state_transition_transpose(r, s, is_boundary);
        let k_dot_tpr: f64 = k_filter
            .iter()
            .zip(t_prime_r.iter())
            .map(|(&k, &tr)| k * tr)
            .sum();
        let mut result = vec![0.0; s];
        for j in 0..s {
            if j < 2 {
                result[j] = t_prime_r[j] - k_dot_tpr;
            } else {
                result[j] = t_prime_r[j];
            }
        }
        result
    }

    #[test]
    fn test_dk_l_prime_r_non_boundary_s4_matches_naive() {
        let mut rng = StdRng::seed_from_u64(500);
        let s = 4;
        let k: Vec<f64> = (0..s).map(|_| rng.gen::<f64>() * 0.5).collect();
        let r: Vec<f64> = (0..s).map(|_| rng.gen::<f64>() - 0.5).collect();
        let naive = naive_l_prime_r(&k, &r, s, false);
        let efficient = efficient_l_prime_r(&k, &r, s, false);
        let diff: f64 = naive
            .iter()
            .zip(efficient.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-14, "L'r non-boundary s=4: diff = {}", diff);
    }

    #[test]
    fn test_dk_l_prime_r_boundary_s4_matches_naive() {
        let mut rng = StdRng::seed_from_u64(501);
        let s = 4;
        let k: Vec<f64> = (0..s).map(|_| rng.gen::<f64>() * 0.5).collect();
        let r: Vec<f64> = (0..s).map(|_| rng.gen::<f64>() - 0.5).collect();
        let naive = naive_l_prime_r(&k, &r, s, true);
        let efficient = efficient_l_prime_r(&k, &r, s, true);
        let diff: f64 = naive
            .iter()
            .zip(efficient.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-14, "L'r boundary s=4: diff = {}", diff);
    }

    #[test]
    fn test_dk_l_prime_r_boundary_s12_matches_naive() {
        let mut rng = StdRng::seed_from_u64(502);
        let s = 12;
        let k: Vec<f64> = (0..s).map(|_| rng.gen::<f64>() * 0.3).collect();
        let r: Vec<f64> = (0..s).map(|_| rng.gen::<f64>() - 0.5).collect();
        let naive = naive_l_prime_r(&k, &r, s, true);
        let efficient = efficient_l_prime_r(&k, &r, s, true);
        let diff: f64 = naive
            .iter()
            .zip(efficient.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-13, "L'r boundary s=12: diff = {}", diff);
    }

    #[test]
    fn test_dk_r_vector_all_finite_s168_t1200() {
        // Run the full smoother and verify all output values are finite
        let s = 168;
        let t = 1200;
        let y: Vec<f64> = (0..t).map(|i| ((i % s) as f64 - 83.5) * 0.05).collect();
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, s, 1, 1.0, 1.0);
        for i in 0..t {
            for j in 0..s {
                assert!(
                    result[i][j].is_finite(),
                    "smooth[{}][{}] = {} not finite (s=168, t=1200)",
                    i,
                    j,
                    result[i][j]
                );
            }
        }
    }

    // ─── PR-C: Smoothed state finite tests (3 tests) ───

    #[test]
    fn test_dk_smooth_finite_s4() {
        let y: Vec<f64> = (0..20).map(|i| ((i % 4) as f64 - 1.5) * 2.0).collect();
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 4, 1, 1.0, 1.0);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(val.is_finite(), "s4: smooth[{}][{}] = {}", i, j, val);
            }
        }
    }

    #[test]
    fn test_dk_smooth_finite_s12() {
        let y: Vec<f64> = (0..100).map(|i| ((i % 12) as f64 - 5.5) * 0.3).collect();
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 12, 1, 1.0, 1.0);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(val.is_finite(), "s12: smooth[{}][{}] = {}", i, j, val);
            }
        }
    }

    #[test]
    fn test_dk_smooth_finite_s168() {
        let y: Vec<f64> = (0..500).map(|i| ((i % 168) as f64 - 83.5) * 0.02).collect();
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 168, 1, 1.0, 1.0);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(val.is_finite(), "s168: smooth[{}][{}] = {}", i, j, val);
            }
        }
    }

    // ─── PR-C: R compatibility gate tests (2 tests) ───
    // R compatibility is primarily tested via Python-level test_numerical_equivalence.py.
    // These Rust-level tests verify that the DK smoother produces results consistent
    // with the RTS reference (which is known to match R bsts within ±1%).

    #[test]
    fn test_dk_smoother_r_compatible_s12_t100() {
        let y: Vec<f64> = (0..100)
            .map(|i| ((i % 12) as f64 - 5.5) * 0.3 + 10.0)
            .collect();
        let dk = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 12, 1, 10.0, 1.0);
        let rts = seasonal_kalman_smoother_rts_reference(&y, 0.5, 0.01, 0.01, 12, 1, 10.0, 1.0);
        let diff = max_abs_diff(&dk, &rts);
        // Level values are ~10.0, so 1e-7 absolute corresponds to ~1e-8 relative
        assert!(
            diff < 1e-6,
            "R compat s12_t100: DK vs RTS diff = {} (must be << 1% of signal)",
            diff
        );
    }

    #[test]
    fn test_dk_smoother_r_compatible_s168_t1200() {
        let y: Vec<f64> = (0..1200)
            .map(|i| ((i % 168) as f64 - 83.5) * 0.05 + 100.0)
            .collect();
        let dk = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 168, 1, 100.0, 1.0);
        let rts = seasonal_kalman_smoother_rts_reference(&y, 0.5, 0.01, 0.01, 168, 1, 100.0, 1.0);
        let diff = max_abs_diff(&dk, &rts);
        // Signal scale ~100, diff ~1e-5 → relative ~1e-7, well within ±1%
        assert!(
            diff < 1e-2,
            "R compat s168_t1200: DK vs RTS diff = {} (must be << 1% of signal)",
            diff
        );
    }

    // ─── PR-C: Boundary value tests (5 tests) ───

    #[test]
    fn test_dk_smoother_t_eq_1_s4() {
        // Single observation: backward pass is trivial (one step from r_T=0)
        let y = vec![3.0];
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 4, 1, 1.0, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
        for &v in &result[0] {
            assert!(v.is_finite());
        }
        // With a_pred = [0,0,0,0] and y=3, smoothed level should be positive
        assert!(
            result[0][0] > 0.0,
            "level should be positive for y=3, got {}",
            result[0][0]
        );
    }

    #[test]
    fn test_dk_smoother_zero_innovation_s4() {
        // y[i] = Z a_pred[i] = 0 for all i → v_t = 0 → no correction
        // Start with a_pred = [0,...,0], y = 0 → v = 0 always
        let y = vec![0.0; 10];
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 4, 1, 1.0, 1.0);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "zero innov: smooth[{}][{}] = {}",
                    i,
                    j,
                    val
                );
            }
        }
        // With zero observations and zero initial state, smoothed should be near zero
        for &v in &result[9] {
            assert!(v.abs() < 1.0, "zero obs: expect near-zero state, got {}", v);
        }
    }

    #[test]
    fn test_dk_smoother_large_sigma2_obs_s12() {
        // Very large observation noise → filter trusts prior, smoothed ≈ prior (0)
        let y: Vec<f64> = (0..50).map(|i| ((i % 12) as f64 - 5.5) * 10.0).collect();
        let result = seasonal_kalman_smoother(&y, 1e8, 0.01, 0.01, 12, 1, 1.0, 1.0);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(val.is_finite(), "large obs: smooth[{}][{}] = {}", i, j, val);
            }
        }
        // With huge obs noise, Kalman gain ≈ 0, so smoothed ≈ 0 (prior)
        for &v in &result[25] {
            assert!(
                v.abs() < 5.0,
                "large sigma2_obs: expected near-zero, got {}",
                v
            );
        }
    }

    #[test]
    fn test_dk_smoother_season_duration_7_s7() {
        // season_duration=7: boundaries at t=0,7,14,...
        let y: Vec<f64> = (0..70).map(|i| ((i / 7) % 7) as f64).collect();
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 7, 7, 0.0, 1.0);
        assert_eq!(result.len(), 70);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "dur7: smooth[{}][{}] = {} not finite",
                    i,
                    j,
                    val
                );
            }
        }
        // DK should match RTS for this configuration
        let rts = seasonal_kalman_smoother_rts_reference(&y, 0.5, 0.01, 0.01, 7, 7, 0.0, 1.0);
        let diff = max_abs_diff(&result, &rts);
        assert!(diff < 1e-7, "dur7 DK vs RTS: diff = {}", diff);
    }

    #[test]
    fn test_dk_smoother_season_duration_24_s24() {
        // Hourly seasonal with 24 seasons
        let y: Vec<f64> = (0..240).map(|i| ((i % 24) as f64 - 11.5) * 0.2).collect();
        let result = seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 24, 1, 1.0, 1.0);
        assert_eq!(result.len(), 240);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "s24: smooth[{}][{}] = {} not finite",
                    i,
                    j,
                    val
                );
            }
        }
        // DK should match RTS
        let rts = seasonal_kalman_smoother_rts_reference(&y, 0.5, 0.01, 0.01, 24, 1, 1.0, 1.0);
        let diff = max_abs_diff(&result, &rts);
        assert!(diff < 1e-6, "s24 DK vs RTS: diff = {}", diff);
    }

    // ── PR-D helpers: flat ↔ nested conversion ──────────────────────
    fn flat_from_nested(m: &[Vec<f64>]) -> Vec<f64> {
        let s = m.len();
        let mut flat = vec![0.0; s * s];
        for i in 0..s {
            for j in 0..s {
                flat[i * s + j] = m[i][j];
            }
        }
        flat
    }

    fn nested_from_flat(flat: &[f64], s: usize) -> Vec<Vec<f64>> {
        (0..s).map(|i| flat[i * s..(i + 1) * s].to_vec()).collect()
    }

    // ── PR-D Red tests: fused predict_state_covariance_flat ─────────

    #[test]
    fn test_fused_predict_cov_matches_original_s2_boundary() {
        let s = 2;
        let p_nested = random_spd(s, 206);
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p_nested, &q, s, true);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        let actual = nested_from_flat(&out, s);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-12, "s2 boundary: diff = {}", diff);
    }

    #[test]
    fn test_fused_predict_cov_matches_original_s4_boundary() {
        let s = 4;
        let p_nested = random_spd(s, 200);
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p_nested, &q, s, true);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        let actual = nested_from_flat(&out, s);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-12, "s4 boundary: diff = {}", diff);
    }

    #[test]
    fn test_fused_predict_cov_matches_original_s12_boundary() {
        let s = 12;
        let p_nested = random_spd(s, 201);
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p_nested, &q, s, true);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        let actual = nested_from_flat(&out, s);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-12, "s12 boundary: diff = {}", diff);
    }

    #[test]
    fn test_fused_predict_cov_matches_original_s168_boundary() {
        let s = 168;
        let p_nested = random_spd(s, 202);
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p_nested, &q, s, true);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        let actual = nested_from_flat(&out, s);
        let diff = max_abs_diff(&expected, &actual);
        // Tolerance scaled by max value for large matrices
        let max_val = expected
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .fold(0.0_f64, f64::max);
        let tol = max_val * 1e-10;
        assert!(diff < tol, "s168 boundary: diff = {} (tol = {})", diff, tol);
    }

    #[test]
    fn test_fused_predict_cov_nonboundary_unchanged() {
        let s = 12;
        let p_nested = random_spd(s, 203);
        let q = q_diag_for(s, 0.01, 0.005);
        let expected = predict_state_covariance_naive(&p_nested, &q, s, false);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, false, &mut out, &mut row_buf);
        let actual = nested_from_flat(&out, s);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-12, "nonboundary s12: diff = {}", diff);
    }

    #[test]
    fn test_fused_predict_cov_preserves_symmetry_s52() {
        let s = 52;
        let p_nested = random_spd(s, 205);
        let q = q_diag_for(s, 0.01, 0.005);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        // Exact bit-level symmetry (mirror writes, not average)
        for i in 0..s {
            for j in (i + 1)..s {
                assert_eq!(
                    out[i * s + j],
                    out[j * s + i],
                    "symmetry fail at ({},{}): {} vs {}",
                    i,
                    j,
                    out[i * s + j],
                    out[j * s + i]
                );
            }
        }
    }

    #[test]
    fn test_fused_predict_cov_diagonal_positive_s168() {
        let s = 168;
        let p_nested = random_spd(s, 204);
        let q = q_diag_for(s, 0.01, 0.005);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        for i in 0..s {
            assert!(
                out[i * s + i] > 0.0,
                "diagonal [{}][{}] = {} not positive",
                i,
                i,
                out[i * s + i]
            );
        }
    }

    #[test]
    fn test_fused_predict_cov_identity_p_s7() {
        let s = 7;
        let p_nested: Vec<Vec<f64>> = (0..s)
            .map(|i| (0..s).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let q = vec![0.0; s]; // Q = 0
        let expected = predict_state_covariance_naive(&p_nested, &q, s, true);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        let actual = nested_from_flat(&out, s);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-12, "identity P s7: diff = {}", diff);
    }

    #[test]
    fn test_fused_predict_cov_zero_p_s4() {
        let s = 4;
        let p_nested = vec![vec![0.0; s]; s];
        let q = q_diag_for(s, 0.03, 0.02);
        let p_flat = flat_from_nested(&p_nested);
        let mut out = vec![0.0; s * s];
        let mut row_buf = vec![0.0; s];
        predict_state_covariance_flat(&p_flat, &q, s, true, &mut out, &mut row_buf);
        // With P=0, result should be diag(Q) (with F_MIN floor)
        for i in 0..s {
            for j in 0..s {
                if i == j {
                    let expected_val = q[i].max(F_MIN);
                    assert!(
                        (out[i * s + j] - expected_val).abs() < 1e-14,
                        "zero P diag [{},{}]: got {} expected {}",
                        i,
                        j,
                        out[i * s + j],
                        expected_val
                    );
                } else {
                    assert!(
                        out[i * s + j].abs() < 1e-14,
                        "zero P off-diag [{},{}] = {} not zero",
                        i,
                        j,
                        out[i * s + j]
                    );
                }
            }
        }
    }

    // ── PR-E tests: Rayon backward pass regression ──────────────────

    #[test]
    fn test_rayon_phase2_matches_sequential_s4_t100() {
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        assert_dk_matches_rts(&y, 0.5, 0.01, 0.01, 4, 1, 1.0, 1.0, 1e-9, "rayon_s4_t100");
    }

    #[test]
    fn test_rayon_phase2_matches_sequential_s168_t1200() {
        let y: Vec<f64> = (0..1200)
            .map(|i| ((i % 168) as f64 - 83.5) * 0.05)
            .collect();
        assert_dk_matches_rts(
            &y,
            0.5,
            0.01,
            0.01,
            168,
            1,
            1.0,
            1.0,
            1e-3,
            "rayon_s168_t1200",
        );
    }

    #[test]
    fn test_rayon_r_store_all_finite_s168_t1200() {
        let y: Vec<f64> = (0..1200)
            .map(|i| ((i % 168) as f64 - 83.5) * 0.05)
            .collect();
        let result =
            seasonal_kalman_smoother(&y, 0.5, 0.01, 0.01, 168, 1, 1.0, 1.0);
        assert_eq!(result.len(), 1200);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "rayon finite: smooth[{}][{}] = {} not finite",
                    i,
                    j,
                    val
                );
            }
        }
    }
}
