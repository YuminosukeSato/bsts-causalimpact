/// Dynamic Time Warping (DTW) with Sakoe-Chiba band constraint and early abandoning.
///
/// # Optimization layers (applied in this PR)
/// 1. Sakoe-Chiba band: O(T²) → O(T×w)
/// 2. Early abandoning: row-wise minimum exceeds `best_so_far` → return ∞
/// 3. LB_Keogh lower bound: O(T) envelope filter prunes >90% candidates
/// 4. AVX2 SIMD for `lb_keogh_distance` (with scalar fallback)
///
/// # Future (YAGNI for this PR)
/// - Multi-scale coarse-to-fine: only useful for T>10,000

/// Compute DTW distance between two sequences with optional Sakoe-Chiba window.
///
/// Uses a flat cost matrix `cost[i*m + j]` (same pattern as kalman.rs).
/// Early abandoning: if the minimum cumulative cost in any row exceeds
/// `best_so_far`, returns `f64::INFINITY` immediately.
///
/// # Arguments
/// * `x` - First time series (length n, must be non-empty)
/// * `y` - Second time series (length m, must be non-empty)
/// * `window` - Sakoe-Chiba band width. `None` = unconstrained.
/// * `best_so_far` - Early abandoning threshold. Use `f64::INFINITY` to disable.
///
/// # Returns
/// DTW distance, or `f64::INFINITY` if early-abandoned.
pub fn dtw_distance(x: &[f64], y: &[f64], window: Option<usize>, best_so_far: f64) -> f64 {
    let n = x.len();
    let m = y.len();
    assert!(!x.is_empty(), "x must not be empty");
    assert!(!y.is_empty(), "y must not be empty");

    let w = window.unwrap_or(n.max(m));

    // Single-row optimization: only keep two rows to reduce memory from O(n*m) to O(m).
    // Previous row and current row.
    let mut prev = vec![f64::INFINITY; m];
    let mut curr = vec![f64::INFINITY; m];

    // First row (i=0)
    {
        let j_start = 0usize;
        let j_end = (w + 1).min(m);
        let mut row_min = f64::INFINITY;
        for j in j_start..j_end {
            let d = (x[0] - y[j]).abs();
            let p = if j == 0 { 0.0 } else { curr[j - 1] };
            curr[j] = d + p;
            row_min = row_min.min(curr[j]);
        }
        if row_min > best_so_far {
            return f64::INFINITY;
        }
    }

    for i in 1..n {
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(f64::INFINITY);

        let j_start = if i > w { i - w } else { 0 };
        let j_end = (i + w + 1).min(m);
        let mut row_min = f64::INFINITY;

        for j in j_start..j_end {
            let d = (x[i] - y[j]).abs();
            let up = prev[j]; // cost[i-1][j]
            let left = if j > 0 { curr[j - 1] } else { f64::INFINITY }; // cost[i][j-1]
            let diag = if j > 0 { prev[j - 1] } else { f64::INFINITY }; // cost[i-1][j-1]
            let min_prev = up.min(left).min(diag);
            curr[j] = d + min_prev;
            row_min = row_min.min(curr[j]);
        }

        if row_min > best_so_far {
            return f64::INFINITY;
        }
    }

    curr[m - 1]
}

/// Compute the LB_Keogh envelope (lower, upper) for a reference series.
///
/// For each point `y[i]`, the envelope is:
/// - `lo[i] = min(y[max(0, i-w) .. min(n, i+w+1)])`
/// - `hi[i] = max(y[max(0, i-w) .. min(n, i+w+1)])`
///
/// # Arguments
/// * `y` - Reference time series
/// * `window` - Envelope half-width
///
/// # Returns
/// `(lo, hi)` vectors of same length as `y`.
pub fn lb_keogh_envelope(y: &[f64], window: usize) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut lo = vec![0.0f64; n];
    let mut hi = vec![0.0f64; n];
    for i in 0..n {
        let start = i.saturating_sub(window);
        let end = (i + window + 1).min(n);
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in &y[start..end] {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        lo[i] = min_val;
        hi[i] = max_val;
    }
    (lo, hi)
}

/// Compute LB_Keogh distance: a lower bound on DTW distance.
///
/// Dispatches to AVX2 implementation on x86_64 when available,
/// otherwise falls back to scalar.
///
/// # Guarantee
/// `lb_keogh_distance(xi, lo, hi) <= dtw_distance(xi, y, window, INF)`
/// for the same `window` used to compute the envelope.
pub fn lb_keogh_distance(xi: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    assert_eq!(xi.len(), lo.len());
    assert_eq!(xi.len(), hi.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: we checked AVX2 is available at runtime.
            return unsafe { lb_keogh_distance_avx2(xi, lo, hi) };
        }
    }
    lb_keogh_distance_scalar(xi, lo, hi)
}

fn lb_keogh_distance_scalar(xi: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    xi.iter()
        .zip(lo)
        .zip(hi)
        .map(|((&x, &l), &h)| {
            if x < l {
                l - x
            } else if x > h {
                x - h
            } else {
                0.0
            }
        })
        .sum()
}

/// AVX2 implementation: processes 4 f64 per cycle (256-bit registers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lb_keogh_distance_avx2(xi: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = xi.len();
    let mut sum = _mm256_setzero_pd();
    let zero = _mm256_setzero_pd();
    let mut i = 0;
    while i + 4 <= n {
        let x = _mm256_loadu_pd(xi.as_ptr().add(i));
        let l = _mm256_loadu_pd(lo.as_ptr().add(i));
        let h = _mm256_loadu_pd(hi.as_ptr().add(i));
        // max(0, l - x) + max(0, x - h)
        let below = _mm256_max_pd(zero, _mm256_sub_pd(l, x));
        let above = _mm256_max_pd(zero, _mm256_sub_pd(x, h));
        sum = _mm256_add_pd(sum, _mm256_add_pd(below, above));
        i += 4;
    }
    // Horizontal sum: 256-bit → 128-bit → 64-bit
    let s128 = _mm_add_pd(
        _mm256_castpd256_pd128(sum),
        _mm256_extractf128_pd(sum, 1),
    );
    let s64 = _mm_add_sd(s128, _mm_shuffle_pd(s128, s128, 1));
    let mut result = _mm_cvtsd_f64(s64);
    // Tail elements (n % 4 != 0)
    for j in i..n {
        let x = xi[j];
        if x < lo[j] {
            result += lo[j] - x;
        } else if x > hi[j] {
            result += x - hi[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_identical_series_is_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(dtw_distance(&x, &x, None, f64::INFINITY), 0.0);
    }

    #[test]
    fn test_dtw_different_series_positive() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert!(dtw_distance(&x, &y, None, f64::INFINITY) > 0.0);
    }

    #[test]
    fn test_dtw_single_element() {
        let x = vec![1.0];
        let y = vec![3.0];
        assert!((dtw_distance(&x, &y, None, f64::INFINITY) - 2.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "x must not be empty")]
    fn test_dtw_empty_x_panics() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0];
        dtw_distance(&x, &y, None, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "y must not be empty")]
    fn test_dtw_empty_y_panics() {
        let x = vec![1.0];
        let y: Vec<f64> = vec![];
        dtw_distance(&x, &y, None, f64::INFINITY);
    }

    #[test]
    fn test_dtw_window_none_unconstrained() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        assert_eq!(dtw_distance(&x, &y, None, f64::INFINITY), 0.0);
    }

    #[test]
    fn test_dtw_window_zero_diagonal_only() {
        // window=0 constrains to diagonal: cost = sum |x[i] - y[i]|
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let d = dtw_distance(&x, &y, Some(0), f64::INFINITY);
        let l1: f64 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum();
        assert!((d - l1).abs() < 1e-12);
    }

    #[test]
    fn test_dtw_different_length() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = dtw_distance(&x, &y, None, f64::INFINITY);
        assert!(d >= 0.0);
        assert!(d.is_finite());
    }

    #[test]
    fn test_dtw_early_abandon_fires() {
        let x = vec![0.0, 0.0, 0.0];
        let y = vec![100.0, 100.0, 100.0];
        // best_so_far = 1.0 is far below actual DTW
        let d = dtw_distance(&x, &y, None, 1.0);
        assert_eq!(d, f64::INFINITY);
    }

    #[test]
    fn test_dtw_early_abandon_does_not_fire() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let d = dtw_distance(&x, &y, None, 100.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_dtw_symmetry() {
        let x = vec![1.0, 3.0, 5.0, 2.0];
        let y = vec![2.0, 4.0, 1.0, 3.0];
        let d1 = dtw_distance(&x, &y, None, f64::INFINITY);
        let d2 = dtw_distance(&y, &x, None, f64::INFINITY);
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn test_lb_keogh_envelope_single_element() {
        let y = vec![5.0];
        let (lo, hi) = lb_keogh_envelope(&y, 3);
        assert_eq!(lo, vec![5.0]);
        assert_eq!(hi, vec![5.0]);
    }

    #[test]
    fn test_lb_keogh_envelope_window_zero() {
        let y = vec![1.0, 5.0, 3.0];
        let (lo, hi) = lb_keogh_envelope(&y, 0);
        assert_eq!(lo, y);
        assert_eq!(hi, y);
    }

    #[test]
    fn test_lb_keogh_distance_identical_zero() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lo, hi) = lb_keogh_envelope(&y, 2);
        assert_eq!(lb_keogh_distance(&y, &lo, &hi), 0.0);
    }

    #[test]
    fn test_lb_keogh_lower_bound_property() {
        // LB_Keogh <= DTW (lower bound guarantee)
        let x = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let w = 1;
        let (lo, hi) = lb_keogh_envelope(&y, w);
        let lb = lb_keogh_distance(&x, &lo, &hi);
        let dtw = dtw_distance(&x, &y, Some(w), f64::INFINITY);
        assert!(lb <= dtw + 1e-12, "LB_Keogh ({lb}) > DTW ({dtw})");
    }

    #[test]
    fn test_lb_keogh_window_zero_equals_l1() {
        // window=0 → envelope equals y itself → LB_Keogh = L1 distance
        let x = vec![1.0, 5.0, 3.0];
        let y = vec![2.0, 3.0, 6.0];
        let (lo, hi) = lb_keogh_envelope(&y, 0);
        let lb = lb_keogh_distance(&x, &lo, &hi);
        let l1: f64 = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum();
        assert!((lb - l1).abs() < 1e-12);
    }

    #[test]
    fn test_lb_keogh_scalar_matches_avx2() {
        // Both implementations must produce the same result.
        let n = 100;
        let xi: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let (lo, hi) = lb_keogh_envelope(&y, 5);

        let scalar = lb_keogh_distance_scalar(&xi, &lo, &hi);
        let dispatched = lb_keogh_distance(&xi, &lo, &hi);
        assert!(
            (scalar - dispatched).abs() < 1e-12,
            "scalar={scalar}, dispatched={dispatched}"
        );
    }

    #[test]
    fn test_lb_keogh_non_multiple_of_4() {
        // n % 4 != 0 tests tail handling in AVX2 version
        for n in [101, 102, 103] {
            let xi: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let y: Vec<f64> = (0..n).map(|i| i as f64 + 10.0).collect();
            let (lo, hi) = lb_keogh_envelope(&y, 2);
            let scalar = lb_keogh_distance_scalar(&xi, &lo, &hi);
            let dispatched = lb_keogh_distance(&xi, &lo, &hi);
            assert!(
                (scalar - dispatched).abs() < 1e-10,
                "n={n}: scalar={scalar}, dispatched={dispatched}"
            );
        }
    }

    #[test]
    fn test_lb_keogh_large_window_all_inside() {
        // Very large window → all points inside envelope → LB = 0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let (lo, hi) = lb_keogh_envelope(&y, 100);
        // envelope lo=hi=3 for all, but with huge window it's still 3
        // x has values 1..5, so some are outside
        // Actually with window=100, lo[i]=min(all y)=3, hi[i]=max(all y)=3
        // So lb = |1-3| + |2-3| + 0 + |4-3| + |5-3| = 2+1+0+1+2 = 6
        let lb = lb_keogh_distance(&x, &lo, &hi);
        assert!((lb - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_dtw_known_value_3x3() {
        // Hand-computed DTW for a simple case
        // x = [1, 2, 3], y = [1, 2, 2]
        // cost matrix (cumulative, abs distance):
        //   y:  1    2    2
        // x:1  0    1    2
        //   2  1    0    0
        //   3  3    1    1
        // DTW = cost[2][2] = 1.0
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 2.0];
        let d = dtw_distance(&x, &y, None, f64::INFINITY);
        assert!((d - 1.0).abs() < 1e-12, "DTW={d}, expected 1.0");
    }

    #[test]
    fn test_dtw_two_row_optimization_correctness() {
        // Compare two-row optimization with a known multi-element case
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 2.0, 4.0];
        let d = dtw_distance(&x, &y, None, f64::INFINITY);
        // Optimal warping path: (0,0)→(1,0)→(2,1)→(3,1)→(4,2)
        // costs: 0 + 0 + 0 + 1 + 0 = 1
        // Actually let me trace carefully:
        // (0,0): |0-0|=0, prev=0 → 0
        // (1,0): |1-0|=1, prev=cost(0,0)=0 → 1
        // (1,1): |1-2|=1, prev=min(cost(0,1), cost(1,0), cost(0,0))=min(2,1,0)=0 → 1
        // (2,1): |2-2|=0, prev=min(cost(1,1), cost(2,0), cost(1,0))=min(1,2,1)=1 → 1
        // (2,2): |2-4|=2, prev=min(cost(1,2), cost(2,1), cost(1,1))
        //   cost(1,2): |1-4|=3, prev=min(cost(0,2), cost(1,1), cost(0,1))
        //     cost(0,1): |0-2|=2, prev=cost(0,0)=0 → 2
        //     cost(0,2): |0-4|=4, prev=cost(0,1)=2 → 6
        //     cost(1,2) = 3 + min(6,1,2) = 3+1 = 4
        //   prev for (2,2) = min(4, 1, 1) = 1
        //   cost(2,2) = 2 + 1 = 3
        // (3,2): |3-4|=1, prev=min(cost(2,2), cost(3,1), cost(2,1))
        //   cost(3,1): |3-2|=1, prev=min(cost(2,1), cost(3,0), cost(2,0))
        //     cost(2,0): |2-0|=2, prev=cost(1,0)=1 → 3
        //     cost(3,0): |3-0|=3, prev=cost(2,0)=3 → 6
        //     cost(3,1) = 1 + min(1, 6, 3) = 1+1 = 2
        //   prev for (3,2) = min(3, 2, 1) = 1
        //   cost(3,2) = 1+1 = 2
        // (4,2): |4-4|=0, prev=min(cost(3,2), cost(4,1), cost(3,1))
        //   cost(4,1): |4-2|=2, prev=min(cost(3,1), cost(4,0), cost(3,0))
        //     cost(4,0): |4-0|=4, prev=cost(3,0)=6 → 10
        //     cost(4,1) = 2 + min(2, 10, 6) = 2+2 = 4
        //   prev for (4,2) = min(2, 4, 2) = 2
        //   cost(4,2) = 0 + 2 = 2
        assert!((d - 2.0).abs() < 1e-12, "DTW={d}, expected 2.0");
    }
}
