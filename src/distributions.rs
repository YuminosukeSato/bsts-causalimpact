/// Sampling from conjugate distributions used in the Gibbs sampler.
use rand::Rng;
use rand_distr::{Gamma, Normal, StandardNormal};

/// Sample from InverseGamma(shape, scale).
/// Algorithm: x ~ Gamma(shape, 1/scale), return 1/x.
pub fn sample_inv_gamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    let gamma = Gamma::new(shape, 1.0 / scale).expect("Invalid Gamma parameters");
    let x: f64 = rng.sample(gamma);
    1.0 / x
}

/// Sample from Normal(mean, variance).
pub fn sample_normal<R: Rng>(rng: &mut R, mean: f64, variance: f64) -> f64 {
    let std = variance.sqrt();
    let z: f64 = rng.sample(StandardNormal);
    mean + std * z
}

/// Sample multivariate normal: mean + L * z, where L = cholesky(cov).
/// Returns vector of length k.
pub fn sample_mvnormal<R: Rng>(rng: &mut R, mean: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let k = mean.len();
    let l = cholesky(cov, k);
    let z: Vec<f64> = (0..k).map(|_| rng.sample(StandardNormal)).collect();

    let mut result = vec![0.0; k];
    for i in 0..k {
        result[i] = mean[i];
        for j in 0..=i {
            result[i] += l[i][j] * z[j];
        }
    }
    result
}

/// Cholesky decomposition of a k x k positive-definite matrix.
/// Returns lower-triangular L such that A = L * L^T.
fn cholesky(a: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0;
            for m in 0..j {
                sum += l[i][m] * l[j][m];
            }
            if i == j {
                let diag = a[i][i] - sum;
                l[i][j] = if diag > 0.0 { diag.sqrt() } else { 1e-12 };
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    l
}

/// Sample from a Normal distribution with precision-parameterized posterior.
/// Given prior precision and data precision, compute posterior.
pub fn sample_normal_posterior<R: Rng>(
    rng: &mut R,
    prior_mean: f64,
    prior_precision: f64,
    data_sum: f64,
    data_precision: f64,
) -> f64 {
    let post_precision = prior_precision + data_precision;
    let post_mean = (prior_precision * prior_mean + data_precision * data_sum) / post_precision;
    let post_variance = 1.0 / post_precision;
    sample_normal(rng, post_mean, post_variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_inv_gamma_positive() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let x = sample_inv_gamma(&mut rng, 2.0, 1.0);
            assert!(x > 0.0, "InvGamma sample must be positive");
        }
    }

    #[test]
    fn test_normal_mean_convergence() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10_000;
        let mean = 5.0;
        let var = 2.0;
        let samples: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, mean, var)).collect();
        let sample_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        assert!((sample_mean - mean).abs() < 0.1, "Mean should converge to {mean}");
    }

    #[test]
    fn test_mvnormal_dimension() {
        let mut rng = StdRng::seed_from_u64(42);
        let mean = vec![1.0, 2.0];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let sample = sample_mvnormal(&mut rng, &mean, &cov);
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_cholesky_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky(&a, 2);
        assert!((l[0][0] - 1.0).abs() < 1e-12);
        assert!((l[1][1] - 1.0).abs() < 1e-12);
        assert!((l[1][0]).abs() < 1e-12);
    }
}
