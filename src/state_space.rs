/// State-space model definition for Local Level + Regression.
///
/// Observation equation:  y_t = Z_t * α_t + ε_t,  ε_t ~ N(0, σ²_obs)
/// State transition:      α_t = T * α_{t-1} + R * η_t,  η_t ~ N(0, Q)
///
/// For Local Level:
///   α_t = [μ_t]   (scalar state: trend level)
///   Z_t = 1        (observation loads directly on level)
///   T   = 1        (random walk)
///   R   = 1
///   Q   = σ²_level

/// Holds the time-varying parts of the state-space model.
pub struct StateSpaceModel {
    /// Observation vector y (length T).
    pub y: Vec<f64>,
    /// Covariate matrix X (T x k), or empty if no covariates.
    pub x: Vec<Vec<f64>>,
    /// Number of covariates.
    pub k: usize,
    /// Number of time points.
    pub t_len: usize,
}

impl StateSpaceModel {
    pub fn new(y: Vec<f64>, x: Vec<Vec<f64>>) -> Self {
        let t_len = y.len();
        let k = if x.is_empty() { 0 } else { x.len() };
        Self { y, x, k, t_len }
    }

    /// Compute observation at time t given state mu and beta.
    /// y_hat = mu + X[t] . beta
    #[inline]
    pub fn observe(&self, t: usize, mu: f64, beta: &[f64]) -> f64 {
        let mut val = mu;
        for j in 0..self.k {
            val += self.x[j][t] * beta[j];
        }
        val
    }

    /// Get covariate value x[j][t].
    #[inline]
    pub fn x_at(&self, j: usize, t: usize) -> f64 {
        self.x[j][t]
    }
}
