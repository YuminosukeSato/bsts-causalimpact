use pyo3::prelude::*;

/// CausalImpact Rust core module.
/// Provides Gibbs sampler for Bayesian structural time series.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
