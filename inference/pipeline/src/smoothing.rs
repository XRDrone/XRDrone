use crate::common::{clamp01, clone_py_dict, py_get, pylist_to_dicts, safe_bool_opt, safe_float_opt};
use crate::geometry::{
    alpha, flat_from_mat3, mat3_from_flat, pose_position_params_from_smoothness,
    pose_rotation_params_from_smoothness, quat_angle, quat_from_rotmat, quat_normalize,
    quat_slerp, rotmat_from_quat, vec_norm, world_params_from_smoothness, ypr_from_r_wc,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::HashMap;

/// Vector-valued One Euro filter core.
#[derive(Clone)]
pub(crate) struct VecFilterCore {
    min_cutoff_hz: f64,
    beta: f64,
    d_cutoff_hz: f64,
    x_prev: Option<Vec<f64>>,
    dx_hat: Option<Vec<f64>>,
}

impl VecFilterCore {
    pub(crate) fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self {
            min_cutoff_hz,
            beta,
            d_cutoff_hz,
            x_prev: None,
            dx_hat: None,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.x_prev = None;
        self.dx_hat = None;
    }

    pub(crate) fn set_params(&mut self, min_cutoff_hz: f64, beta: f64, d_cutoff_hz: Option<f64>) {
        self.min_cutoff_hz = min_cutoff_hz;
        self.beta = beta;
        if let Some(v) = d_cutoff_hz {
            self.d_cutoff_hz = v;
        }
    }

    pub(crate) fn filter_vec(&mut self, x: Vec<f64>, dt: f64) -> Vec<f64> {
        let dt = dt.max(1e-6);
        if self.x_prev.is_none() {
            self.x_prev = Some(x.clone());
            self.dx_hat = Some(vec![0.0; x.len()]);
            return x;
        }

        let prev = match self.x_prev.clone() {
            Some(prev) => prev,
            None => return x,
        };
        let mut dx = vec![0.0; x.len()];
        for i in 0..x.len() {
            dx[i] = (x[i] - prev[i]) / dt;
        }

        let alpha_d = alpha(self.d_cutoff_hz, dt);
        let mut dx_hat = self.dx_hat.clone().unwrap_or_else(|| vec![0.0; x.len()]);
        for i in 0..x.len() {
            dx_hat[i] = dx_hat[i] + alpha_d * (dx[i] - dx_hat[i]);
        }
        self.dx_hat = Some(dx_hat.clone());

        let speed = vec_norm(&dx_hat);
        let cutoff = self.min_cutoff_hz + self.beta * speed;
        let alpha_x = alpha(cutoff, dt);
        let mut x_hat = vec![0.0; x.len()];
        for i in 0..x.len() {
            x_hat[i] = prev[i] + alpha_x * (x[i] - prev[i]);
        }
        self.x_prev = Some(x_hat.clone());
        x_hat
    }
}

/// Quaternion-valued One Euro filter core.
#[derive(Clone)]
pub(crate) struct QuatFilterCore {
    min_cutoff_hz: f64,
    beta: f64,
    d_cutoff_hz: f64,
    q_prev: Option<[f64; 4]>,
    speed_hat: f64,
}

impl QuatFilterCore {
    pub(crate) fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self {
            min_cutoff_hz,
            beta,
            d_cutoff_hz,
            q_prev: None,
            speed_hat: 0.0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.q_prev = None;
        self.speed_hat = 0.0;
    }

    pub(crate) fn set_params(&mut self, min_cutoff_hz: f64, beta: f64, d_cutoff_hz: Option<f64>) {
        self.min_cutoff_hz = min_cutoff_hz;
        self.beta = beta;
        if let Some(v) = d_cutoff_hz {
            self.d_cutoff_hz = v;
        }
    }

    pub(crate) fn filter_quat(&mut self, q: [f64; 4], dt: f64) -> [f64; 4] {
        let q = quat_normalize(q);
        let dt = dt.max(1e-6);
        if self.q_prev.is_none() {
            self.q_prev = Some(q);
            self.speed_hat = 0.0;
            return q;
        }

        let prev = match self.q_prev {
            Some(prev) => prev,
            None => return q,
        };
        let speed = quat_angle(prev, q) / dt;
        let alpha_d = alpha(self.d_cutoff_hz, dt);
        self.speed_hat = self.speed_hat + alpha_d * (speed - self.speed_hat);
        let cutoff = self.min_cutoff_hz + self.beta * self.speed_hat.abs();
        let alpha_q = alpha(cutoff, dt);
        let q_hat = quat_slerp(prev, q, alpha_q);
        self.q_prev = Some(q_hat);
        q_hat
    }
}

#[pyclass(name = "OneEuroVectorFilter")]
pub struct OneEuroVectorFilterPy {
    inner: VecFilterCore,
}

#[pymethods]
impl OneEuroVectorFilterPy {
    #[new]
    #[pyo3(signature=(*, min_cutoff_hz, beta, d_cutoff_hz=1.0))]
    fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self {
            inner: VecFilterCore::new(min_cutoff_hz, beta, d_cutoff_hz),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[pyo3(signature=(*, min_cutoff_hz, beta, d_cutoff_hz=None))]
    fn set_params(&mut self, min_cutoff_hz: f64, beta: f64, d_cutoff_hz: Option<f64>) {
        self.inner.set_params(min_cutoff_hz, beta, d_cutoff_hz);
    }

    fn filter(&mut self, x: Vec<f64>, dt: f64) -> Vec<f64> {
        self.inner.filter_vec(x, dt)
    }
}

#[pyclass(name = "OneEuroQuaternionFilter")]
pub struct OneEuroQuaternionFilterPy {
    inner: QuatFilterCore,
}

#[pymethods]
impl OneEuroQuaternionFilterPy {
    #[new]
    #[pyo3(signature=(*, min_cutoff_hz, beta, d_cutoff_hz=1.0))]
    fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self {
            inner: QuatFilterCore::new(min_cutoff_hz, beta, d_cutoff_hz),
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[pyo3(signature=(*, min_cutoff_hz, beta, d_cutoff_hz=None))]
    fn set_params(&mut self, min_cutoff_hz: f64, beta: f64, d_cutoff_hz: Option<f64>) {
        self.inner.set_params(min_cutoff_hz, beta, d_cutoff_hz);
    }

    fn filter(&mut self, q: Vec<f64>, dt: f64) -> PyResult<Vec<f64>> {
        if q.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Quaternion must have length 4",
            ));
        }
        Ok(self
            .inner
            .filter_quat([q[0], q[1], q[2], q[3]], dt)
            .to_vec())
    }
}

/// Pose smoother that wraps vector/quaternion One Euro filters for pose output.
#[pyclass(name = "PoseMotionSmootherCore")]
pub struct PoseMotionSmootherCore {
    enabled: bool,
    smoothness: f64,
    derivative_cutoff_hz: f64,
    reset_timeout_s: f64,
    default_fps: f64,
    last_timestamp: Option<f64>,
    position_filter: VecFilterCore,
    rotation_filter: QuatFilterCore,
}

#[pymethods]
impl PoseMotionSmootherCore {
    #[new]
    #[pyo3(signature=(*, enabled=true, smoothness=0.5, derivative_cutoff_hz=1.0, reset_timeout_s=0.75, default_fps=30.0))]
    fn new(
        enabled: bool,
        smoothness: f64,
        derivative_cutoff_hz: f64,
        reset_timeout_s: f64,
        default_fps: f64,
    ) -> Self {
        let smoothness = clamp01(smoothness);
        let derivative_cutoff_hz = derivative_cutoff_hz.max(1e-3);
        let reset_timeout_s = reset_timeout_s.max(1e-3);
        let default_fps = default_fps.max(1.0);
        let (pos_cutoff, pos_beta) = pose_position_params_from_smoothness(smoothness);
        let (rot_cutoff, rot_beta) = pose_rotation_params_from_smoothness(smoothness);
        Self {
            enabled,
            smoothness,
            derivative_cutoff_hz,
            reset_timeout_s,
            default_fps,
            last_timestamp: None,
            position_filter: VecFilterCore::new(pos_cutoff, pos_beta, derivative_cutoff_hz),
            rotation_filter: QuatFilterCore::new(rot_cutoff, rot_beta, derivative_cutoff_hz),
        }
    }

    fn reset(&mut self) {
        self.last_timestamp = None;
        self.position_filter.reset();
        self.rotation_filter.reset();
    }

    fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.reset();
        }
    }

    fn set_smoothness(&mut self, smoothness: f64) {
        self.smoothness = clamp01(smoothness);
        let (pos_cutoff, pos_beta) = pose_position_params_from_smoothness(self.smoothness);
        let (rot_cutoff, rot_beta) = pose_rotation_params_from_smoothness(self.smoothness);
        self.position_filter
            .set_params(pos_cutoff, pos_beta, Some(self.derivative_cutoff_hz));
        self.rotation_filter
            .set_params(rot_cutoff, rot_beta, Some(self.derivative_cutoff_hz));
        self.reset();
    }

    #[pyo3(signature=(pose_data, c_w, r_wc, k, *, timestamp=None))]
    fn smooth_pose(
        &mut self,
        py: Python<'_>,
        pose_data: &PyDict,
        c_w: Vec<f64>,
        r_wc: Vec<f64>,
        k: Vec<f64>,
        timestamp: Option<f64>,
    ) -> PyResult<(Py<PyDict>, Vec<f64>, Vec<f64>, Vec<f64>)> {
        if c_w.len() != 3 || r_wc.len() != 9 || k.len() != 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid pose primitive sizes",
            ));
        }
        let dt = self.resolve_dt(timestamp);
        let c_w_smooth = self.position_filter.filter_vec(c_w.clone(), dt);
        let r_wc_mat = mat3_from_flat(&r_wc).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Rotation matrix must have length 9")
        })?;
        let q_wc = quat_from_rotmat(r_wc_mat);
        let q_wc_smooth = self.rotation_filter.filter_quat(q_wc, dt);
        let r_wc_smooth = rotmat_from_quat(q_wc_smooth);
        let [yaw, pitch, roll] = ypr_from_r_wc(r_wc_smooth);

        let smoothed_pose = clone_py_dict(py, pose_data)?;
        let pose_ref = smoothed_pose.bind(py);
        pose_ref.set_item("x", c_w_smooth[0])?;
        pose_ref.set_item("altitude", c_w_smooth[1])?;
        pose_ref.set_item("z", c_w_smooth[2])?;
        pose_ref.set_item("yaw", yaw)?;
        pose_ref.set_item("pitch", pitch)?;
        pose_ref.set_item("roll", roll)?;

        Ok((smoothed_pose, c_w_smooth, flat_from_mat3(r_wc_smooth), k))
    }
}

impl PoseMotionSmootherCore {
    fn resolve_dt(&mut self, timestamp: Option<f64>) -> f64 {
        let default_dt = 1.0 / self.default_fps;
        match timestamp {
            None => {
                self.last_timestamp = None;
                default_dt
            }
            Some(ts) => {
                if let Some(last_ts) = self.last_timestamp {
                    self.last_timestamp = Some(ts);
                    let dt = ts - last_ts;
                    if dt <= 0.0 || dt > self.reset_timeout_s {
                        self.reset();
                        self.last_timestamp = Some(ts);
                        default_dt
                    } else {
                        dt
                    }
                } else {
                    self.last_timestamp = Some(ts);
                    default_dt
                }
            }
        }
    }
}

#[derive(Clone)]
struct WorldTrackState {
    filt: VecFilterCore,
    last_timestamp: f64,
}

/// Per-track smoother for projected world-space detection positions.
#[pyclass(name = "WorldTrackSmootherCore")]
pub struct WorldTrackSmootherCore {
    enabled: bool,
    smoothness: f64,
    derivative_cutoff_hz: f64,
    reset_timeout_s: f64,
    max_track_age_s: f64,
    default_fps: f64,
    states: HashMap<i64, WorldTrackState>,
    min_cutoff_hz: f64,
    beta: f64,
}

#[pymethods]
impl WorldTrackSmootherCore {
    #[new]
    #[pyo3(signature=(*, enabled=true, smoothness=0.5, derivative_cutoff_hz=1.0, reset_timeout_s=0.75, max_track_age_s=1.5, default_fps=30.0))]
    fn new(
        enabled: bool,
        smoothness: f64,
        derivative_cutoff_hz: f64,
        reset_timeout_s: f64,
        max_track_age_s: f64,
        default_fps: f64,
    ) -> Self {
        let smoothness = clamp01(smoothness);
        let (min_cutoff_hz, beta) = world_params_from_smoothness(smoothness);
        Self {
            enabled,
            smoothness,
            derivative_cutoff_hz: derivative_cutoff_hz.max(1e-3),
            reset_timeout_s: reset_timeout_s.max(1e-3),
            max_track_age_s: max_track_age_s.max(1e-3),
            default_fps: default_fps.max(1.0),
            states: HashMap::new(),
            min_cutoff_hz,
            beta,
        }
    }

    fn reset(&mut self) {
        self.states.clear();
    }

    fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.reset();
        }
    }

    fn set_smoothness(&mut self, smoothness: f64) {
        self.smoothness = clamp01(smoothness);
        let (min_cutoff_hz, beta) = world_params_from_smoothness(self.smoothness);
        self.min_cutoff_hz = min_cutoff_hz;
        self.beta = beta;
        self.reset();
    }

    #[pyo3(signature=(detections, *, timestamp=None))]
    fn update_detections(
        &mut self,
        py: Python<'_>,
        detections: &PyAny,
        timestamp: Option<f64>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let mut out = Vec::new();
        if !self.enabled || self.smoothness <= 0.0 {
            for det in pylist_to_dicts(detections)? {
                out.push(clone_py_dict(py, det)?);
            }
            return Ok(out);
        }

        let default_dt = 1.0 / self.default_fps;
        for det in pylist_to_dicts(detections)? {
            let cloned = clone_py_dict(py, det)?;
            let det_out = cloned.as_ref(py);

            if !safe_bool_opt(py_get(det_out, "world_valid"), false) {
                out.push(cloned);
                continue;
            }

            let track_key = match py_get(det_out, "track_id").and_then(|v| v.extract::<i64>().ok()) {
                Some(v) => v,
                None => {
                    out.push(cloned);
                    continue;
                }
            };
            let xz = vec![
                safe_float_opt(py_get(det_out, "world_x"), 0.0),
                safe_float_opt(py_get(det_out, "world_z"), 0.0),
            ];

            let state = self.states.entry(track_key).or_insert_with(|| WorldTrackState {
                filt: VecFilterCore::new(self.min_cutoff_hz, self.beta, self.derivative_cutoff_hz),
                last_timestamp: timestamp.unwrap_or(0.0),
            });

            let xz_smooth = if state.filt.x_prev.is_none() {
                state.filt.filter_vec(xz.clone(), default_dt)
            } else {
                let dt = match timestamp {
                    None => default_dt,
                    Some(_ts) if state.last_timestamp <= 0.0 => default_dt,
                    Some(ts) => {
                        let dt = ts - state.last_timestamp;
                        if dt <= 0.0 || dt > self.reset_timeout_s {
                            state.filt.reset();
                            default_dt
                        } else {
                            dt
                        }
                    }
                };
                state.filt.filter_vec(xz.clone(), dt)
            };

            if let Some(ts) = timestamp {
                state.last_timestamp = ts;
            }

            det_out.set_item("world_x", xz_smooth[0])?;
            det_out.set_item("world_z", xz_smooth[1])?;
            out.push(cloned);
        }
        self.prune(timestamp);
        Ok(out)
    }
}

impl WorldTrackSmootherCore {
    fn prune(&mut self, timestamp: Option<f64>) {
        let Some(ts) = timestamp else {
            return;
        };
        let stale: Vec<i64> = self
            .states
            .iter()
            .filter_map(|(track_id, state)| {
                if ts - state.last_timestamp > self.max_track_age_s {
                    Some(*track_id)
                } else {
                    None
                }
            })
            .collect();
        for track_id in stale {
            self.states.remove(&track_id);
        }
    }
}
