use pyo3::prelude::*;

mod distributions;
mod kalman;
mod state_space;

/// CausalImpact Rust core module.
/// Provides Gibbs sampler for Bayesian structural time series.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
