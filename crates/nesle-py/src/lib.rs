mod env;
mod errors;
#[cfg(feature = "viewer")]
mod human_window;
mod interface;
mod metadata;
mod state;
mod vector;

#[cfg(feature = "viewer")]
mod viewer;

use pyo3::prelude::*;

#[pymodule]
fn _nesle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(metadata::backend_id, m)?)?;
    m.add_function(wrap_pyfunction!(metadata::game_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(metadata::start_state_metadata, m)?)?;
    m.add_class::<state::PyCoreState>()?;
    m.add_class::<state::PyEnvState>()?;
    m.add_class::<interface::PyNesInterface>()?;
    m.add_class::<env::PyNesEnv>()?;
    m.add_class::<vector::PyNesVectorEnv>()?;
    #[cfg(feature = "viewer")]
    m.add_class::<human_window::PyHumanWindow>()?;
    Ok(())
}
