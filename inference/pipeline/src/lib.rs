#![allow(clippy::too_many_arguments)]
#![allow(deprecated)]

mod adaptive_tuning;
mod common;
mod geometry;
mod id_flicker;
mod smoothing;
mod udp;
mod world_projection;

use pyo3::prelude::*;

/// Python extension entrypoint for the native XRDrone helpers.
///
/// The public Python API remains stable even though the Rust implementation is
/// now split into focused modules:
/// - `id_flicker`: tracked-object continuity and coasting
/// - `world_projection`: foot-point extraction and ground-plane projection
/// - `udp`: Unity packet formatting
/// - `adaptive_tuning`: bounded runtime policy adaptation
/// - `smoothing`: One Euro filters and pose/world smoothing
#[pymodule]
fn xrdrone_native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(world_projection::clamp01_py, m)?)?;
    m.add_function(wrap_pyfunction!(world_projection::passes_udp_world_projection_filter, m)?)?;
    m.add_function(wrap_pyfunction!(world_projection::attach_foot_and_world, m)?)?;
    m.add_function(wrap_pyfunction!(udp::to_unity_udp_packet, m)?)?;

    m.add_class::<id_flicker::RobustIdFlickerMitigator>()?;
    m.add_class::<adaptive_tuning::AdaptiveTuningMetrics>()?;
    m.add_class::<adaptive_tuning::AdaptiveRuntimeTuner>()?;
    m.add_class::<smoothing::OneEuroVectorFilterPy>()?;
    m.add_class::<smoothing::OneEuroQuaternionFilterPy>()?;
    m.add_class::<smoothing::PoseMotionSmootherCore>()?;
    m.add_class::<smoothing::WorldTrackSmootherCore>()?;
    Ok(())
}
