use nesle_common::NesleError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

pub(crate) fn map_error(err: NesleError) -> PyErr {
    match err {
        NesleError::InvalidRom(_)
        | NesleError::UnsupportedMapper(_)
        | NesleError::InvalidState(_) => PyValueError::new_err(err.to_string()),
        NesleError::Io(_) => PyRuntimeError::new_err(err.to_string()),
    }
}
