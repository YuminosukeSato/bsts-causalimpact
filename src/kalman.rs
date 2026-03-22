/// Kalman filter and Durbin-Koopman simulation smoother for Local Level model.
///
/// The simulation smoother draws states from p(α | y, σ²_obs, σ²_level, β)
/// using the method of Durbin & Koopman (2002).

use crate::distributions::sample_normal;
use rand::Rng;

/// Minimum value for forecast error variance F_t to prevent division by zero.
const F_MIN: f64 = 1e-12;

/// Forward Kalman filter for the local level model.
///
/// Model: y_t - X_t β = μ_t + ε_t, μ_t = μ_{t-1} + η_t
///
/// Returns (filtered_state, filtered_var, forecast_err, forecast_var)
/// Each is a Vec of length T.
pub fn kalman_filter(
    y_adj: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = y_adj.len();
    let mut a = vec![0.0; t]; // filtered state mean
    let mut p = vec![0.0; t]; // filtered state variance
    let mut v = vec![0.0; t]; // forecast error
    let mut f = vec![0.0; t]; // forecast error variance

    // Initialize: diffuse prior
    let a0 = 0.0;
    let p0 = 1e7; // large initial variance (diffuse)

    for i in 0..t {
        let a_prior = if i == 0 { a0 } else { a[i - 1] };
        let p_prior = if i == 0 { p0 } else { p[i - 1] + sigma2_level };

        // Forecast error
        v[i] = y_adj[i] - a_prior;
        f[i] = (p_prior + sigma2_obs).max(F_MIN);

        // Kalman gain
        let k = p_prior / f[i];

        // Update
        a[i] = a_prior + k * v[i];
        // Joseph form for numerical stability: P = (1-K)*P_prior*(1-K)' + K*H*K'
        let one_minus_k = 1.0 - k;
        p[i] = one_minus_k * one_minus_k * p_prior + k * k * sigma2_obs;
    }

    (a, p, v, f)
}

/// Durbin-Koopman simulation smoother.
///
/// Draws a sample of the state vector from p(α | y_adj, σ²_obs, σ²_level).
///
/// Algorithm:
/// 1. Draw synthetic states α⁺ and observations y⁺ from the model
/// 2. Run Kalman filter on (y_adj - y⁺)
/// 3. Run state smoother on the filtered output
/// 4. Return α̂_smooth + α⁺
pub fn simulation_smoother<R: Rng>(
    rng: &mut R,
    y_adj: &[f64],
    sigma2_obs: f64,
    sigma2_level: f64,
) -> Vec<f64> {
    let t = y_adj.len();

    // Step 1: Generate synthetic states and observations
    let mut alpha_plus = vec![0.0; t];
    let mut y_plus = vec![0.0; t];

    alpha_plus[0] = sample_normal(rng, 0.0, 1e7); // diffuse
    y_plus[0] = alpha_plus[0] + sample_normal(rng, 0.0, sigma2_obs);

    for i in 1..t {
        alpha_plus[i] = alpha_plus[i - 1] + sample_normal(rng, 0.0, sigma2_level);
        y_plus[i] = alpha_plus[i] + sample_normal(rng, 0.0, sigma2_obs);
    }

    // Step 2: Compute y* = y_adj - y_plus
    let y_star: Vec<f64> = y_adj.iter().zip(y_plus.iter()).map(|(y, yp)| y - yp).collect();

    // Step 3: Kalman filter on y*
    let (a, p, v, f) = kalman_filter(&y_star, sigma2_obs, sigma2_level);

    // Step 4: State smoother (backward pass)
    let alpha_hat = state_smoother(&a, &p, &v, &f, sigma2_level);

    // Step 5: α_sampled = α̂_smooth(y*) + α⁺
    let mut alpha_sampled = vec![0.0; t];
    for i in 0..t {
        alpha_sampled[i] = alpha_hat[i] + alpha_plus[i];
    }

    alpha_sampled
}

/// Classical fixed-interval state smoother (backward recursion).
///
/// Computes E[α_t | Y_1:T] from filtered quantities.
fn state_smoother(
    a: &[f64],
    p: &[f64],
    v: &[f64],
    f: &[f64],
    sigma2_level: f64,
) -> Vec<f64> {
    let t = a.len();
    let mut alpha_hat = vec![0.0; t];

    // Backward recursion for r_t
    // r_{t-1} = v_t/F_t + (1 - K_t) * r_t
    // where K_t = P_prior_t / F_t
    let mut r = 0.0; // r_T = 0

    for i in (0..t).rev() {
        let p_prior = if i == 0 { 1e7 } else { p[i - 1] + sigma2_level };
        let k = p_prior / f[i];
        r = v[i] / f[i] + (1.0 - k) * r;

        let a_prior = if i == 0 { 0.0 } else { a[i - 1] };
        alpha_hat[i] = a_prior + p_prior * r;
    }

    alpha_hat
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_kalman_filter_constant_signal() {
        // Constant signal: y_t = 5.0 for all t
        // Filter should converge to state ≈ 5.0
        let y: Vec<f64> = vec![5.0; 50];
        let (a, _p, _v, _f) = kalman_filter(&y, 1.0, 0.01);
        assert!(
            (a[49] - 5.0).abs() < 0.1,
            "Filtered state should converge to 5.0, got {}",
            a[49]
        );
    }

    #[test]
    fn test_kalman_filter_output_lengths() {
        let y: Vec<f64> = vec![1.0, 2.0, 3.0];
        let (a, p, v, f) = kalman_filter(&y, 1.0, 0.1);
        assert_eq!(a.len(), 3);
        assert_eq!(p.len(), 3);
        assert_eq!(v.len(), 3);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn test_simulation_smoother_shape() {
        let mut rng = StdRng::seed_from_u64(42);
        let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let states = simulation_smoother(&mut rng, &y, 1.0, 0.01);
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn test_simulation_smoother_tracks_signal() {
        let mut rng = StdRng::seed_from_u64(42);
        // Strong signal, low noise
        let y: Vec<f64> = (0..50).map(|i| 10.0 + 0.0 * i as f64).collect();
        let n_samples = 200;
        let mut mean_states = vec![0.0; 50];

        for _ in 0..n_samples {
            let states = simulation_smoother(&mut rng, &y, 0.1, 0.01);
            for t in 0..50 {
                mean_states[t] += states[t] / n_samples as f64;
            }
        }

        // Average of smoothed states near the end should be close to 10.0
        assert!(
            (mean_states[49] - 10.0).abs() < 1.0,
            "Smoothed state should be near 10.0, got {}",
            mean_states[49]
        );
    }
}
