use numpy::ndarray::Axis;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;

mod distributions;
mod kalman;
mod sampler;
mod state_space;

const MODULE_VERSION: &str = "1.3.1";

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct GibbsSamples {
    #[pyo3(get)]
    pub states: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub sigma_obs: Vec<f64>,
    #[pyo3(get)]
    pub sigma_level: Vec<f64>,
    #[pyo3(get)]
    pub sigma_seasonal: Vec<f64>,
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
        dynamic_regression=false,
        state_model="local_level"
    )
)]
#[allow(clippy::too_many_arguments)]
fn run_gibbs_sampler(
    y: &Bound<'_, PyAny>,
    x: Option<&Bound<'_, PyAny>>,
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
    state_model: &str,
) -> PyResult<GibbsSamples> {
    let y_values = extract_series(y)?;
    let y_slice = y_values.as_slice()?;
    let x_vecs = extract_covariates(x)?;

    let result = sampler::run_sampler(
        y_slice,
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
        state_model,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    Ok(GibbsSamples {
        states: result.states,
        sigma_obs: result.sigma_obs,
        sigma_level: result.sigma_level,
        sigma_seasonal: result.sigma_seasonal,
        beta: result.beta,
        gamma: result.gamma,
        predictions: result.predictions,
    })
}

enum SeriesInput<'py> {
    Borrowed(PyReadonlyArray1<'py, f64>),
    Owned(Vec<f64>),
}

impl<'py> SeriesInput<'py> {
    fn as_slice(&self) -> PyResult<&[f64]> {
        match self {
            Self::Borrowed(array) => Ok(array.as_slice()?),
            Self::Owned(values) => Ok(values.as_slice()),
        }
    }
}

fn extract_series<'py>(value: &Bound<'py, PyAny>) -> PyResult<SeriesInput<'py>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'py, f64>>() {
        return Ok(SeriesInput::Borrowed(array));
    }

    Ok(SeriesInput::Owned(value.extract::<Vec<f64>>()?))
}

fn extract_covariates(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<f64>>> {
    let Some(value) = value else {
        return Ok(vec![]);
    };

    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        let view = array.as_array();
        let mut cols = Vec::with_capacity(view.nrows());
        for row in view.axis_iter(Axis(0)) {
            cols.push(row.to_vec());
        }
        return Ok(cols);
    }

    value.extract::<Vec<Vec<f64>>>()
}

/// CausalImpact Rust core module.
/// Provides Gibbs sampler for Bayesian structural time series.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", MODULE_VERSION)?;
    m.add_class::<GibbsSamples>()?;
    m.add_function(wrap_pyfunction!(run_gibbs_sampler, m)?)?;
    Ok(())
}
