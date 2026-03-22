//! State-space model definition for Local Level + Regression + Seasonal regressors.
//!
//! The latent state remains a scalar local level:
//!   α_t = μ_t
//!   μ_t = μ_{t-1} + η_t
//!
//! Seasonal structure is represented as always-included static regressors using
//! sum-to-zero coding. This keeps the Gibbs sampler stable while exposing the
//! same public API shape as R's `nseasons` / `season.duration`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeasonalConfig {
    nseasons: usize,
    season_duration: usize,
}

impl SeasonalConfig {
    pub fn from_optional(
        nseasons: Option<f64>,
        season_duration: Option<f64>,
    ) -> Result<Option<Self>, String> {
        match (nseasons, season_duration) {
            (None, None) => Ok(None),
            (Some(nseasons_value), None) => Ok(Some(Self {
                nseasons: validate_whole_number("nseasons", nseasons_value)?,
                season_duration: 1,
            })),
            (None, Some(_)) => {
                Err("nseasons must be provided when season_duration is set".to_string())
            }
            (Some(nseasons_value), Some(season_duration_value)) => Ok(Some(Self {
                nseasons: validate_whole_number("nseasons", nseasons_value)?,
                season_duration: validate_whole_number("season_duration", season_duration_value)?,
            })),
        }
    }

    #[inline]
    pub fn nseasons(&self) -> usize {
        self.nseasons
    }

    #[inline]
    pub fn season_duration(&self) -> usize {
        self.season_duration
    }
}

fn validate_whole_number(name: &str, value: f64) -> Result<usize, String> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(format!("{name} must be an integer, got {value}"));
    }

    let integer_value = value as isize;
    if integer_value < 1 {
        return Err(format!("{name} must be >= 1, got {value}"));
    }
    Ok(integer_value as usize)
}

pub struct StateSpaceModel {
    x: Vec<Vec<f64>>,
    seasonal_x: Vec<Vec<f64>>,
}

impl StateSpaceModel {
    pub fn new(y: Vec<f64>, x: Vec<Vec<f64>>, seasonal: Option<SeasonalConfig>) -> Self {
        let seasonal_x = seasonal
            .map(|config| build_seasonal_design(y.len(), config))
            .unwrap_or_default();
        Self { x, seasonal_x }
    }

    #[inline]
    pub fn num_covariates(&self) -> usize {
        self.x.len()
    }

    #[inline]
    pub fn covariates(&self) -> &[Vec<f64>] {
        &self.x
    }

    #[inline]
    pub fn num_seasonal_covariates(&self) -> usize {
        self.seasonal_x.len()
    }

    #[inline]
    pub fn seasonal_covariates(&self) -> &[Vec<f64>] {
        &self.seasonal_x
    }

    #[inline]
    pub fn observe(&self, t: usize, mu: f64, beta: &[f64], seasonal_beta: &[f64]) -> f64 {
        mu + self.regression_contribution(t, beta, seasonal_beta)
    }

    #[inline]
    pub fn regression_contribution(&self, t: usize, beta: &[f64], seasonal_beta: &[f64]) -> f64 {
        self.x
            .iter()
            .zip(beta.iter())
            .map(|(x_col, beta_value)| x_col[t] * beta_value)
            .sum::<f64>()
            + self
                .seasonal_x
                .iter()
                .zip(seasonal_beta.iter())
                .map(|(x_col, beta_value)| x_col[t] * beta_value)
                .sum::<f64>()
    }
}

fn build_seasonal_design(t_total: usize, config: SeasonalConfig) -> Vec<Vec<f64>> {
    if config.nseasons() <= 1 {
        return vec![];
    }

    let ncols = config.nseasons() - 1;
    let mut design = vec![vec![0.0; t_total]; ncols];
    for t in 0..t_total {
        let season_index = (t / config.season_duration()) % config.nseasons();
        if season_index < ncols {
            design[season_index][t] = 1.0;
        } else {
            for column in &mut design {
                column[t] = -1.0;
            }
        }
    }
    design
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_whole_number_rejects_fractional_value() {
        let error = SeasonalConfig::from_optional(Some(7.5), Some(1.0)).unwrap_err();
        assert!(error.contains("nseasons"));
    }

    #[test]
    fn test_build_seasonal_design_uses_sum_to_zero_reference_column() {
        let design = build_seasonal_design(
            6,
            SeasonalConfig {
                nseasons: 3,
                season_duration: 1,
            },
        );
        assert_eq!(design.len(), 2);
        assert_eq!(design[0], vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
        assert_eq!(design[1], vec![0.0, 1.0, -1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_build_seasonal_design_respects_season_duration() {
        let design = build_seasonal_design(
            8,
            SeasonalConfig {
                nseasons: 2,
                season_duration: 2,
            },
        );
        assert_eq!(
            design,
            vec![vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]]
        );
    }
}
