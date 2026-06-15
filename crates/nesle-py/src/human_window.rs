#![expect(
    clippy::useless_conversion,
    reason = "PyO3 macro expansion reports PyResult returns as conversions"
)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "HumanWindow", unsendable)]
pub(crate) struct PyHumanWindow {
    inner: crate::viewer::HumanWindow,
}

#[pymethods]
impl PyHumanWindow {
    #[new]
    #[pyo3(signature = (title="NESLE", scale=3))]
    fn new(title: &str, scale: u32) -> PyResult<Self> {
        crate::viewer::HumanWindow::open(title, scale)
            .map(|inner| Self { inner })
            .map_err(PyRuntimeError::new_err)
    }

    fn present(&mut self, rgb: &[u8]) -> PyResult<bool> {
        self.inner.present(rgb).map_err(PyRuntimeError::new_err)
    }

    fn close(&mut self) {}
}

impl nesle_rl::FrameSink for PyHumanWindow {
    fn present(&mut self, rgb: &[u8]) -> bool {
        self.inner.present(rgb).unwrap_or(true)
    }
}
