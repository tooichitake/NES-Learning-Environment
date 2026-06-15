use nesle_core::state::CoreState;
use nesle_rl::NesEnvState;
use pyo3::prelude::*;

#[pyclass(name = "CoreState")]
#[derive(Clone)]
pub(crate) struct PyCoreState {
    pub(crate) state: CoreState,
}

#[pymethods]
impl PyCoreState {
    fn bytes(&self) -> Vec<u8> {
        self.state.as_bytes().to_vec()
    }
}

#[pyclass(name = "EnvState")]
#[derive(Clone)]
pub(crate) struct PyEnvState {
    pub(crate) state: NesEnvState,
}
