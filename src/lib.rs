use pyo3::prelude::*;
use pyo3::types::PyList;

mod distributions;
mod kalman;
mod sampler;
mod state_space;

#[pyclass]
#[derive(Clone)]
pub struct GibbsSamples {
    #[pyo3(get)]
    pub states: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub sigma_obs: Vec<f64>,
    #[pyo3(get)]
    pub sigma_level: Vec<f64>,
    #[pyo3(get)]
    pub beta: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub gamma: Vec<Vec<bool>>,
    #[pyo3(get)]
    pub predictions: Vec<Vec<f64>>,
}

#[pyfunction]
#[pyo3(
    signature = (
        y,
        x,
        pre_end,
        niter,
        nwarmup,
        nchains,
        seed,
        prior_level_sd,
        expected_model_size=1.0,
        nseasons=None,
        season_duration=None,
        dynamic_regression=false
    )
)]
#[allow(clippy::too_many_arguments)]
fn run_gibbs_sampler(
    y: Vec<f64>,
    x: Option<&Bound<'_, PyList>>,
    pre_end: usize,
    niter: usize,
    nwarmup: usize,
    nchains: usize,
    seed: u64,
    prior_level_sd: f64,
    expected_model_size: f64,
    nseasons: Option<f64>,
    season_duration: Option<f64>,
    dynamic_regression: bool,
) -> PyResult<GibbsSamples> {
    let x_vecs: Vec<Vec<f64>> = match x {
        Some(list) => {
            let mut cols = Vec::new();
            for item in list.iter() {
                let col: Vec<f64> = item.extract()?;
                cols.push(col);
            }
            cols
        }
        None => vec![],
    };

    let result = sampler::run_sampler(
        y,
        x_vecs,
        pre_end,
        niter,
        nwarmup,
        nchains,
        seed,
        prior_level_sd,
        expected_model_size,
        nseasons,
        season_duration,
        dynamic_regression,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(GibbsSamples {
        states: result.states,
        sigma_obs: result.sigma_obs,
        sigma_level: result.sigma_level,
        beta: result.beta,
        gamma: result.gamma,
        predictions: result.predictions,
    })
}

/// CausalImpact Rust core module.
/// Provides Gibbs sampler for Bayesian structural time series.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.2.0")?;
    m.add_class::<GibbsSamples>()?;
    m.add_function(wrap_pyfunction!(run_gibbs_sampler, m)?)?;
    Ok(())
}
