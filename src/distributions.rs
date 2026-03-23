//! Sampling from conjugate distributions used in the Gibbs sampler.

use rand::Rng;
use rand_distr::{Gamma, StandardNormal};

pub fn sample_inv_gamma<R: Rng>(rng: &mut R, shape: f64, scale: f64) -> f64 {
    let gamma = Gamma::new(shape, 1.0 / scale).expect("Invalid Gamma parameters");
    let x: f64 = rng.sample(gamma);
    1.0 / x
}

pub fn sample_normal<R: Rng>(rng: &mut R, mean: f64, variance: f64) -> f64 {
    let std = variance.sqrt();
    let z: f64 = rng.sample(StandardNormal);
    mean + std * z
}

/// Cholesky decomposition: A = L L^T, returns L (lower triangular).
/// A must be symmetric positive definite.
/// Near-zero or negative diagonals are clamped to 1e-12 for numerical stability.
fn cholesky_lower(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = a.len();
    let mut lower = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let sum = lower[i]
                .iter()
                .zip(lower[j].iter())
                .take(j)
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f64>();
            if i == j {
                let diagonal = a[i][i] - sum;
                lower[i][j] = if diagonal > 0.0 {
                    diagonal.sqrt()
                } else {
                    1e-12
                };
            } else {
                lower[i][j] = (a[i][j] - sum) / lower[j][j];
            }
        }
    }
    lower
}

/// Solve L x = b via forward substitution (L is lower triangular).
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let k = b.len();
    let mut x = vec![0.0; k];
    for i in 0..k {
        let sum: f64 = l[i].iter().zip(x.iter()).take(i).map(|(a, b)| a * b).sum();
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

/// Solve L^T x = b via backward substitution (L is lower triangular).
fn backward_solve_lt(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let k = b.len();
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let sum: f64 = ((i + 1)..k).map(|j| l[j][i] * x[j]).sum();
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

/// Solve L L^T x = b via forward + backward substitution.
fn chol_solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = forward_solve(l, b);
    backward_solve_lt(l, &y)
}

/// Sample beta ~ N(A^{-1}b, sigma2 * A^{-1}) using Cholesky of precision A.
///
/// Algorithm (matches R bsts):
///   1. L L^T = A  (Cholesky of precision matrix)
///   2. y = L^{-1} b  (forward solve)
///   3. mean = L^{-T} y  (backward solve)
///   4. z ~ N(0, I_k)
///   5. eps = sqrt(sigma2) * L^{-T} z  (backward solve)
///   6. return mean + eps
pub fn sample_from_precision<R: Rng>(
    rng: &mut R,
    precision: &[Vec<f64>],
    rhs: &[f64],
    sigma2_obs: f64,
) -> Vec<f64> {
    let k = rhs.len();
    let l = cholesky_lower(precision);
    let mean = chol_solve_lower(&l, rhs);
    let z: Vec<f64> = (0..k).map(|_| rng.sample(StandardNormal)).collect();
    let scale = sigma2_obs.sqrt();
    let eps = backward_solve_lt(&l, &z);
    mean.iter()
        .zip(eps.iter())
        .map(|(m, e)| m + scale * e)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
        assert!(
            (sample_mean - mean).abs() < 0.1,
            "Mean should converge to {mean}"
        );
    }

    #[test]
    fn test_cholesky_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky_lower(&a);
        assert!((l[0][0] - 1.0).abs() < 1e-12);
        assert!((l[1][1] - 1.0).abs() < 1e-12);
        assert!((l[1][0]).abs() < 1e-12);
    }

    #[test]
    fn test_cholesky_2x2() {
        // A = [[4, 2], [2, 3]]  =>  L = [[2, 0], [1, sqrt(2)]]
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_lower(&a);
        assert!((l[0][0] - 2.0).abs() < 1e-12);
        assert!((l[1][0] - 1.0).abs() < 1e-12);
        assert!((l[1][1] - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_cholesky_near_singular() {
        // Near-singular: diagonal element becomes near-zero after subtraction
        let a = vec![vec![1.0, 1.0 - 1e-14], vec![1.0 - 1e-14, 1.0]];
        let l = cholesky_lower(&a);
        // Should not panic, result should be finite
        for row in &l {
            for val in row {
                assert!(val.is_finite(), "Cholesky result must be finite");
            }
        }
    }

    #[test]
    fn test_chol_solve_identity() {
        // L = I => solve I I^T x = b => x = b
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 7.0];
        let x = chol_solve_lower(&l, &b);
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_chol_solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [10, 8]
        // A^{-1} = [[3/8, -1/4], [-1/4, 1/2]]
        // x = A^{-1}b = [3/8*10 + (-1/4)*8, (-1/4)*10 + 1/2*8] = [1.75, 1.5]
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_lower(&a);
        let b = vec![10.0, 8.0];
        let x = chol_solve_lower(&l, &b);
        assert!((x[0] - 1.75).abs() < 1e-10);
        assert!((x[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_chol_solve_1x1() {
        // k=1: scalar case. A = [[5]], b = [15] => x = 3
        let l = cholesky_lower(&vec![vec![5.0]]);
        let x = chol_solve_lower(&l, &[15.0]);
        assert!((x[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_sample_from_precision_1x1() {
        // k=1: precision=2, rhs=6, sigma2=0.5
        // mean = rhs/precision = 3.0, variance = sigma2/precision = 0.25
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10_000;
        let precision = vec![vec![2.0]];
        let rhs = vec![6.0];
        let sigma2 = 0.5;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let s = sample_from_precision(&mut rng, &precision, &rhs, sigma2);
            sum += s[0];
            sum_sq += s[0] * s[0];
        }
        let sample_mean = sum / n as f64;
        let sample_var = sum_sq / n as f64 - sample_mean * sample_mean;
        assert!(
            (sample_mean - 3.0).abs() < 0.1,
            "Mean {sample_mean} should be near 3.0"
        );
        assert!(
            (sample_var - 0.25).abs() < 0.1,
            "Variance {sample_var} should be near 0.25"
        );
    }

    #[test]
    fn test_sample_from_precision_diagonal() {
        // Diagonal precision: each component independent
        // precision = diag(4, 9), rhs = [12, 27], sigma2 = 1.0
        // mean = [3, 3], variance = [1/4, 1/9]
        let mut rng = StdRng::seed_from_u64(123);
        let n = 20_000;
        let precision = vec![vec![4.0, 0.0], vec![0.0, 9.0]];
        let rhs = vec![12.0, 27.0];
        let sigma2 = 1.0;
        let mut sum = vec![0.0; 2];
        let mut sum_sq = vec![0.0; 2];
        for _ in 0..n {
            let s = sample_from_precision(&mut rng, &precision, &rhs, sigma2);
            for j in 0..2 {
                sum[j] += s[j];
                sum_sq[j] += s[j] * s[j];
            }
        }
        for j in 0..2 {
            let mean = sum[j] / n as f64;
            let var = sum_sq[j] / n as f64 - mean * mean;
            assert!(
                (mean - 3.0).abs() < 0.1,
                "Component {j}: mean {mean} should be near 3.0"
            );
            let expected_var = sigma2 / precision[j][j];
            assert!(
                (var - expected_var).abs() < 0.1,
                "Component {j}: var {var} should be near {expected_var}"
            );
        }
    }

    #[test]
    fn test_sample_from_precision_identity() {
        // precision = I, rhs = [5, -3], sigma2 = 2.0
        // mean = rhs = [5, -3], cov = 2*I
        let mut rng = StdRng::seed_from_u64(99);
        let n = 10_000;
        let precision = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let rhs = vec![5.0, -3.0];
        let sigma2 = 2.0;
        let mut sum = vec![0.0; 2];
        for _ in 0..n {
            let s = sample_from_precision(&mut rng, &precision, &rhs, sigma2);
            for j in 0..2 {
                sum[j] += s[j];
            }
        }
        let mean0 = sum[0] / n as f64;
        let mean1 = sum[1] / n as f64;
        assert!(
            (mean0 - 5.0).abs() < 0.2,
            "Mean[0] {mean0} should be near 5"
        );
        assert!(
            (mean1 - (-3.0)).abs() < 0.2,
            "Mean[1] {mean1} should be near -3"
        );
    }

    #[test]
    fn test_sample_from_precision_finite_k20() {
        // k=20: verify all samples are finite
        let mut rng = StdRng::seed_from_u64(42);
        let k = 20;
        let mut precision = vec![vec![0.0; k]; k];
        for i in 0..k {
            precision[i][i] = 10.0;
            if i > 0 {
                precision[i][i - 1] = 0.1;
                precision[i - 1][i] = 0.1;
            }
        }
        let rhs: Vec<f64> = (0..k).map(|i| i as f64).collect();
        let sigma2 = 1.0;
        for _ in 0..100 {
            let s = sample_from_precision(&mut rng, &precision, &rhs, sigma2);
            for (j, val) in s.iter().enumerate() {
                assert!(val.is_finite(), "k=20 sample[{j}] is not finite: {val}");
            }
        }
    }
}
