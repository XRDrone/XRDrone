#![allow(clippy::too_many_arguments)]
#![allow(deprecated)]

mod common;
mod geometry;
mod id_flicker;
mod udp;

use pyo3::prelude::*;

/// Python extension entrypoint for the native XRDrone helpers.
#[pymodule]
fn xrdrone_native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(udp::to_unity_udp_packet, m)?)?;
    m.add_class::<id_flicker::RobustIdFlickerMitigator>()?;
    Ok(())
}
