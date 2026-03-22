//! Kalman filter and Durbin-Koopman simulation smoother for Local Level model.
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
}
