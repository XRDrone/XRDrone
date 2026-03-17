#![allow(clippy::too_many_arguments)]
#![allow(deprecated)]

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::{HashMap, HashSet, VecDeque};

fn clamp01(x: f64) -> f64 {
    x.max(0.0).min(1.0)
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn safe_float_obj(obj: &PyAny, default: f64) -> f64 {
    match obj.extract::<f64>() {
        Ok(v) if v.is_finite() => v,
        _ => default,
    }
}

fn safe_float_opt(obj: Option<&PyAny>, default: f64) -> f64 {
    match obj {
        Some(v) => safe_float_obj(v, default),
        None => default,
    }
}

fn safe_bool_opt(obj: Option<&PyAny>, default: bool) -> bool {
    match obj {
        Some(v) => v.extract::<bool>().unwrap_or(default),
        None => default,
    }
}

fn py_get<'a>(det: &'a PyDict, key: &str) -> Option<&'a PyAny> {
    det.get_item(key).ok().flatten()
}

fn dict_string_lower_chain(det: &PyDict, keys: &[&str]) -> String {
    for key in keys {
        if let Some(value) = py_get(det, key) {
            if let Ok(s) = value.extract::<String>() {
                return s.to_lowercase();
            }
        }
    }
    String::new()
}

fn clone_py_dict(py: Python<'_>, det: &PyDict) -> PyResult<Py<PyDict>> {
    let out = PyDict::new_bound(py);
    for (key, value) in det.iter() {
        if let Ok(list) = value.downcast::<PyList>() {
            let copied = PyList::empty_bound(py);
            for item in list.iter() {
                copied.append(item)?;
            }
            out.set_item(key, copied)?;
        } else if let Ok(dict) = value.downcast::<PyDict>() {
            let copied = PyDict::new_bound(py);
            for (dk, dv) in dict.iter() {
                copied.set_item(dk, dv)?;
            }
            out.set_item(key, copied)?;
        } else {
            out.set_item(key, value)?;
        }
    }
    Ok(out.unbind())
}

fn clone_py_dict_with_items(
    py: Python<'_>,
    det: &PyDict,
    extra: &[(&str, PyObject)],
) -> PyResult<Py<PyDict>> {
    let out = clone_py_dict(py, det)?;
    let out_ref = out.as_ref(py);
    for (key, value) in extra {
        out_ref.set_item(key, value)?;
    }
    Ok(out)
}

fn any_to_string_set(values: Option<&PyAny>) -> PyResult<Option<HashSet<String>>> {
    match values {
        None => Ok(None),
        Some(obj) => {
            if obj.is_none() {
                return Ok(None);
            }
            let mut out = HashSet::new();
            for item in obj.iter()? {
                let item = item?;
                let s = item.extract::<String>()?;
                out.insert(s.to_lowercase());
            }
            Ok(Some(out))
        }
    }
}

fn any_to_class_map(values: Option<&PyAny>) -> PyResult<HashMap<String, i64>> {
    let mut out = HashMap::new();
    if let Some(obj) = values {
        if obj.is_none() {
            return Ok(out);
        }
        let dict: &PyDict = obj.downcast()?;
        for (k, v) in dict.iter() {
            out.insert(k.extract::<String>()?.to_lowercase(), v.extract::<i64>()?);
        }
    }
    Ok(out)
}

fn pylist_to_dicts<'a>(obj: &'a PyAny) -> PyResult<Vec<&'a PyDict>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let item = item?;
        let det: &PyDict = item.downcast()?;
        out.push(det);
    }
    Ok(out)
}

fn parse_bbox(det: &PyDict) -> Option<[f64; 4]> {
    let bbox = py_get(det, "bbox_xyxy")?;
    let mut vals = Vec::new();
    for item in bbox.iter().ok()? {
        vals.push(safe_float_obj(item.ok()?, 0.0));
    }
    if vals.len() != 4 {
        return None;
    }
    Some([vals[0], vals[1], vals[2], vals[3]])
}

fn xyxy_to_xywhn(x1: f64, y1: f64, x2: f64, y2: f64, width: f64, height: f64) -> (f64, f64, f64, f64) {
    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let cx = x1 + w / 2.0;
    let cy = y1 + h / 2.0;
    if width <= 0.0 || height <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    (
        clamp01(cx / width),
        clamp01(cy / height),
        clamp01(w / width),
        clamp01(h / height),
    )
}

fn mat3_from_flat(values: &[f64]) -> Option<[[f64; 3]; 3]> {
    if values.len() != 9 {
        return None;
    }
    Some([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ])
}

fn flat_from_mat3(m: [[f64; 3]; 3]) -> Vec<f64> {
    vec![
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ]
}

fn inverse_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let i = m[2][2];

    let a11 = e * i - f * h;
    let a12 = -(d * i - f * g);
    let a13 = d * h - e * g;
    let a21 = -(b * i - c * h);
    let a22 = a * i - c * g;
    let a23 = -(a * h - b * g);
    let a31 = b * f - c * e;
    let a32 = -(a * f - c * d);
    let a33 = a * e - b * d;

    let det = a * a11 + b * a12 + c * a13;
    if det.abs() <= 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;

    Some([
        [a11 * inv_det, a21 * inv_det, a31 * inv_det],
        [a12 * inv_det, a22 * inv_det, a32 * inv_det],
        [a13 * inv_det, a23 * inv_det, a33 * inv_det],
    ])
}

fn mat3_vec3_mul(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn vec3_normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = vec3_norm(v);
    if n <= 0.0 {
        None
    } else {
        Some([v[0] / n, v[1] / n, v[2] / n])
    }
}

fn pixel_ray_in_world(_c_w: [f64; 3], r_wc: [[f64; 3]; 3], k: [[f64; 3]; 3], u_px: f64, v_px: f64) -> Option<[f64; 3]> {
    let k_inv = inverse_3x3(k)?;
    let d_c = mat3_vec3_mul(k_inv, [u_px, v_px, 1.0]);
    let d_c = vec3_normalize(d_c)?;
    let r_cw = mat3_transpose(r_wc);
    let d_w = mat3_vec3_mul(r_cw, d_c);
    vec3_normalize(d_w)
}

fn intersect_plane_y0(c_w: [f64; 3], r_wc: [[f64; 3]; 3], k: [[f64; 3]; 3], u_px: f64, v_px: f64) -> Option<[f64; 3]> {
    let d_w = pixel_ray_in_world(c_w, r_wc, k, u_px, v_px)?;
    let denom = d_w[1];
    if denom.abs() < 1e-8 {
        return None;
    }
    let t = (0.0 - c_w[1]) / denom;
    if t <= 0.0 {
        return None;
    }
    Some([
        c_w[0] + t * d_w[0],
        c_w[1] + t * d_w[1],
        c_w[2] + t * d_w[2],
    ])
}

#[pyfunction]
#[pyo3(name = "clamp01")]
fn clamp01_py(x: f64) -> f64 {
    clamp01(x)
}

#[pyfunction]
#[pyo3(signature=(det, *, allowed_classes=None, min_conf=None))]
fn passes_udp_world_projection_filter(det: &PyDict, allowed_classes: Option<&PyAny>, min_conf: Option<f64>) -> PyResult<bool> {
    let cls_name = dict_string_lower_chain(det, &["class", "class_name"]);
    if let Some(allow) = any_to_string_set(allowed_classes)? {
        if !allow.contains(&cls_name) {
            return Ok(false);
        }
    }
    if let Some(min_conf_f) = min_conf {
        let conf = safe_float_opt(py_get(det, "confidence"), 0.0);
        if conf < min_conf_f {
            return Ok(false);
        }
    }
    Ok(true)
}

#[pyclass(name = "RobustIDFlickerMitigator")]
struct RobustIdFlickerMitigator {
    enabled: bool,
    apply_classes: Option<HashSet<String>>,
    tau_on: f64,
    tau_off: f64,
    coast_frames: i64,
    drop_frames: i64,
    ema_alpha: f64,
    use_conf_ema: bool,
    require_track_id: bool,
    coast_conf_decay: f64,
    states: HashMap<i64, TrackContinuityState>,
    step: i64,
}

struct TrackContinuityState {
    cls_name: String,
    conf_ema: f64,
    emitted: bool,
    miss_count: i64,
    last_seen_step: i64,
    last_gate_conf: f64,
    last_raw_conf: f64,
    last_observed_det: Option<Py<PyDict>>,
    last_stable_det: Option<Py<PyDict>>,
}

#[pymethods]
impl RobustIdFlickerMitigator {
    #[new]
    #[pyo3(signature=(*, enabled=true, apply_classes=None, tau_on=0.80, tau_off=0.55, coast_frames=6, drop_frames=45, ema_alpha=0.45, use_conf_ema=true, require_track_id=true, coast_conf_decay=0.985))]
    fn new(
        enabled: bool,
        apply_classes: Option<&PyAny>,
        tau_on: f64,
        tau_off: f64,
        coast_frames: i64,
        drop_frames: i64,
        ema_alpha: f64,
        use_conf_ema: bool,
        require_track_id: bool,
        coast_conf_decay: f64,
    ) -> PyResult<Self> {
        let apply_classes = any_to_string_set(apply_classes)?;
        let tau_on = clamp01(tau_on);
        let tau_off = clamp01(tau_off.min(tau_on));
        let coast_frames = coast_frames.max(0);
        let drop_frames = drop_frames.max(coast_frames);
        Ok(Self {
            enabled,
            apply_classes,
            tau_on,
            tau_off,
            coast_frames,
            drop_frames,
            ema_alpha: clamp01(ema_alpha),
            use_conf_ema,
            require_track_id,
            coast_conf_decay: clamp01(coast_conf_decay),
            states: HashMap::new(),
            step: 0,
        })
    }

    #[pyo3(signature=(*, tau_on=None, tau_off=None, coast_frames=None, coast_conf_decay=None))]
    fn set_runtime_policy(
        &mut self,
        tau_on: Option<f64>,
        tau_off: Option<f64>,
        coast_frames: Option<i64>,
        coast_conf_decay: Option<f64>,
    ) {
        if let Some(v) = tau_on {
            self.tau_on = clamp01(v);
        }
        if let Some(v) = tau_off {
            self.tau_off = clamp01(v.min(self.tau_on));
        }
        if let Some(v) = coast_frames {
            self.coast_frames = v.max(0);
            self.drop_frames = self.drop_frames.max(self.coast_frames);
        }
        if let Some(v) = coast_conf_decay {
            self.coast_conf_decay = clamp01(v);
        }
    }

    fn reset(&mut self) {
        self.states.clear();
        self.step = 0;
    }

    fn apply(&mut self, py: Python<'_>, detections: &PyAny) -> PyResult<Vec<Py<PyDict>>> {
        if !self.enabled {
            let mut out = Vec::new();
            for det in pylist_to_dicts(detections)? {
                out.push(clone_py_dict(py, det)?);
            }
            return Ok(out);
        }

        self.step += 1;
        let mut passthrough: Vec<Py<PyDict>> = Vec::new();
        let mut outputs: Vec<Py<PyDict>> = Vec::new();
        let mut seen_track_keys: HashSet<i64> = HashSet::new();
        let mut emitted_track_keys: HashSet<i64> = HashSet::new();

        for det in pylist_to_dicts(detections)? {
            let cls_name = dict_string_lower_chain(det, &["class", "class_name"]);
            let is_target = match &self.apply_classes {
                Some(classes) => classes.contains(&cls_name),
                None => true,
            };
            if !is_target {
                passthrough.push(clone_py_dict(py, det)?);
                continue;
            }

            let track_key = match py_get(det, "track_id").and_then(|v| v.extract::<i64>().ok()) {
                Some(v) => v,
                None => {
                    if !self.require_track_id {
                        passthrough.push(clone_py_dict(py, det)?);
                    }
                    continue;
                }
            };

            seen_track_keys.insert(track_key);
            let raw_conf = safe_float_opt(py_get(det, "confidence"), 0.0);
            let state = self.states.entry(track_key).or_insert_with(|| TrackContinuityState {
                cls_name: cls_name.clone(),
                conf_ema: raw_conf,
                emitted: false,
                miss_count: 0,
                last_seen_step: self.step,
                last_gate_conf: raw_conf,
                last_raw_conf: raw_conf,
                last_observed_det: None,
                last_stable_det: None,
            });

            state.cls_name = cls_name;
            state.conf_ema = self.ema_alpha * raw_conf + (1.0 - self.ema_alpha) * state.conf_ema;
            state.last_gate_conf = if self.use_conf_ema { state.conf_ema } else { raw_conf };
            state.last_raw_conf = raw_conf;
            state.last_seen_step = self.step;
            state.last_observed_det = Some(clone_py_dict(py, det)?);
            let gate_conf = state.last_gate_conf;

            if !state.emitted {
                if gate_conf >= self.tau_on {
                    state.emitted = true;
                    state.miss_count = 0;
                    state.last_stable_det = Some(clone_py_dict(py, det)?);
                    outputs.push(clone_py_dict_with_items(
                        py,
                        det,
                        &[
                            ("udp_confidence", gate_conf.into_py(py)),
                            ("force_udp_emit", true.into_py(py)),
                            ("continuity_state", "observed".into_py(py)),
                        ],
                    )?);
                    emitted_track_keys.insert(track_key);
                }
                continue;
            }

            if gate_conf >= self.tau_off {
                state.miss_count = 0;
                state.last_stable_det = Some(clone_py_dict(py, det)?);
                outputs.push(clone_py_dict_with_items(
                    py,
                    det,
                    &[
                        ("udp_confidence", clamp01(gate_conf).into_py(py)),
                        ("force_udp_emit", true.into_py(py)),
                        ("continuity_state", "observed".into_py(py)),
                    ],
                )?);
                emitted_track_keys.insert(track_key);
                continue;
            }

            state.miss_count += 1;
            if state.miss_count <= self.coast_frames {
                if let Some(last_stable) = &state.last_stable_det {
                    let decay_power = (state.miss_count - 1).max(0) as i32;
                    let coast_conf = state.last_gate_conf * self.coast_conf_decay.powi(decay_power);
                    let stable_ref = last_stable.as_ref(py);
                    outputs.push(clone_py_dict_with_items(
                        py,
                        stable_ref,
                        &[
                            ("udp_confidence", clamp01(coast_conf).into_py(py)),
                            ("force_udp_emit", true.into_py(py)),
                            ("continuity_state", "coasted".into_py(py)),
                        ],
                    )?);
                    emitted_track_keys.insert(track_key);
                }
            } else {
                state.emitted = false;
            }
        }

        let mut stale_keys = Vec::new();
        for (track_key, state) in self.states.iter_mut() {
            if seen_track_keys.contains(track_key) {
                continue;
            }
            let gap_steps = self.step - state.last_seen_step;
            if state.emitted {
                state.miss_count += 1;
                if state.miss_count <= self.coast_frames && !emitted_track_keys.contains(track_key) {
                    if let Some(last_stable) = &state.last_stable_det {
                        let decay_power = (state.miss_count - 1).max(0) as i32;
                        let coast_conf = state.last_gate_conf * self.coast_conf_decay.powi(decay_power);
                        let stable_ref = last_stable.as_ref(py);
                        outputs.push(clone_py_dict_with_items(
                            py,
                            stable_ref,
                            &[
                                ("udp_confidence", clamp01(coast_conf).into_py(py)),
                                ("force_udp_emit", true.into_py(py)),
                                ("continuity_state", "coasted".into_py(py)),
                            ],
                        )?);
                        emitted_track_keys.insert(*track_key);
                    }
                } else {
                    state.emitted = false;
                }
            }
            if gap_steps > self.drop_frames {
                stale_keys.push(*track_key);
            }
        }
        for key in stale_keys {
            self.states.remove(&key);
        }

        passthrough.extend(outputs);
        Ok(passthrough)
    }
}

#[pyfunction]
#[pyo3(signature=(merged_detections, *, frame_id, timestamp, width, height, class_map=None, allowed_classes=None, min_conf=None))]
fn to_unity_udp_packet(
    py: Python<'_>,
    merged_detections: &PyAny,
    frame_id: i64,
    timestamp: f64,
    width: i64,
    height: i64,
    class_map: Option<&PyAny>,
    allowed_classes: Option<&PyAny>,
    min_conf: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let class_map = any_to_class_map(class_map)?;
    let allow = any_to_string_set(allowed_classes)?;
    let min_conf_f = min_conf;

    let packet = PyDict::new_bound(py);
    let dets = PyList::empty_bound(py);

    for (i, det) in pylist_to_dicts(merged_detections)?.iter().enumerate() {
        let cls_name = dict_string_lower_chain(det, &["class", "class_name"]);
        if let Some(allow_set) = &allow {
            if !allow_set.contains(&cls_name) {
                continue;
            }
        }

        let conf = safe_float_opt(py_get(det, "udp_confidence").or_else(|| py_get(det, "confidence")), 0.0);
        let force_emit = safe_bool_opt(py_get(det, "force_udp_emit"), false);
        if let Some(min_conf_f) = min_conf_f {
            if conf < min_conf_f && !force_emit {
                continue;
            }
        }

        let bbox = match parse_bbox(det) {
            Some(v) => v,
            None => continue,
        };
        let (cx, cy, w, h) = xyxy_to_xywhn(bbox[0], bbox[1], bbox[2], bbox[3], width as f64, height as f64);
        let default_foot_x = cx;
        let default_foot_y = clamp01(cy + h / 2.0);

        let det_id = py_get(det, "track_id")
            .and_then(|v| v.extract::<i64>().ok())
            .unwrap_or(i as i64);

        let world_valid = safe_bool_opt(py_get(det, "world_valid"), false);
        let world_x = safe_float_opt(py_get(det, "world_x"), 0.0);
        let world_y = safe_float_opt(py_get(det, "world_y"), 0.0);
        let world_z = safe_float_opt(py_get(det, "world_z"), 0.0);

        let out_det = PyDict::new_bound(py);
        out_det.set_item("id", det_id)?;
        out_det.set_item("cls", *class_map.get(&cls_name).unwrap_or(&-1))?;
        out_det.set_item("conf", conf)?;
        out_det.set_item("cx", cx)?;
        out_det.set_item("cy", cy)?;
        out_det.set_item("w", w)?;
        out_det.set_item("h", h)?;
        out_det.set_item("foot_x", safe_float_opt(py_get(det, "foot_x"), default_foot_x))?;
        out_det.set_item("foot_y", safe_float_opt(py_get(det, "foot_y"), default_foot_y))?;
        out_det.set_item("world_valid", world_valid)?;
        out_det.set_item("world_x", world_x)?;
        out_det.set_item("world_y", world_y)?;
        out_det.set_item("world_z", world_z)?;
        dets.append(out_det)?;
    }

    packet.set_item("frame_id", frame_id)?;
    packet.set_item("timestamp", timestamp)?;
    packet.set_item("width", width)?;
    packet.set_item("height", height)?;
    packet.set_item("detections", dets)?;
    Ok(packet.into())
}

#[pyfunction]
#[pyo3(signature=(detections, *, pose_valid, pose_camera_world=None, pose_rotation_world_to_camera=None, pose_intrinsics=None, width, height, projection_classes=None, projection_min_conf=None))]
fn attach_foot_and_world(
    py: Python<'_>,
    detections: &PyAny,
    pose_valid: bool,
    pose_camera_world: Option<Vec<f64>>,
    pose_rotation_world_to_camera: Option<Vec<f64>>,
    pose_intrinsics: Option<Vec<f64>>,
    width: i64,
    height: i64,
    projection_classes: Option<&PyAny>,
    projection_min_conf: Option<f64>,
) -> PyResult<Vec<Py<PyDict>>> {
    let width = width.max(1) as f64;
    let height = height.max(1) as f64;
    let allow = any_to_string_set(projection_classes)?;

    let c_w = pose_camera_world.and_then(|v| if v.len() >= 3 { Some([v[0], v[1], v[2]]) } else { None });
    let r_wc = pose_rotation_world_to_camera.and_then(|v| mat3_from_flat(&v));
    let k = pose_intrinsics.and_then(|v| mat3_from_flat(&v));
    let pose_ready = pose_valid && c_w.is_some() && r_wc.is_some() && k.is_some();

    let mut out = Vec::new();
    for det in pylist_to_dicts(detections)? {
        let cloned = clone_py_dict(py, det)?;
        let det_out = cloned.as_ref(py);

        det_out.set_item("foot_x", safe_float_opt(py_get(det_out, "foot_x"), 0.0))?;
        det_out.set_item("foot_y", safe_float_opt(py_get(det_out, "foot_y"), 0.0))?;
        det_out.set_item("world_valid", safe_bool_opt(py_get(det_out, "world_valid"), false))?;
        det_out.set_item("world_x", safe_float_opt(py_get(det_out, "world_x"), 0.0))?;
        det_out.set_item("world_y", safe_float_opt(py_get(det_out, "world_y"), 0.0))?;
        det_out.set_item("world_z", safe_float_opt(py_get(det_out, "world_z"), 0.0))?;

        let bbox = match parse_bbox(det_out) {
            Some(v) => v,
            None => {
                out.push(cloned);
                continue;
            }
        };

        let foot_x_px = (bbox[0] + bbox[2]) / 2.0;
        let foot_y_px = bbox[3];
        det_out.set_item("foot_x", clamp01(foot_x_px / width))?;
        det_out.set_item("foot_y", clamp01(foot_y_px / height))?;

        let cls_name = dict_string_lower_chain(det_out, &["class", "class_name"]);
        let should_project = match &allow {
            Some(values) => values.contains(&cls_name),
            None => true,
        } && projection_min_conf
            .map(|min_conf_f| safe_float_opt(py_get(det_out, "confidence"), 0.0) >= min_conf_f)
            .unwrap_or(true);

        if !pose_ready || !should_project {
            det_out.set_item("world_valid", false)?;
            det_out.set_item("world_x", 0.0)?;
            det_out.set_item("world_y", 0.0)?;
            det_out.set_item("world_z", 0.0)?;
            out.push(cloned);
            continue;
        }

        let point_world = intersect_plane_y0(c_w.unwrap(), r_wc.unwrap(), k.unwrap(), foot_x_px, foot_y_px);
        if let Some(point) = point_world {
            det_out.set_item("world_valid", true)?;
            det_out.set_item("world_x", point[0])?;
            det_out.set_item("world_y", point[1])?;
            det_out.set_item("world_z", point[2])?;
        } else {
            det_out.set_item("world_valid", false)?;
            det_out.set_item("world_x", 0.0)?;
            det_out.set_item("world_y", 0.0)?;
            det_out.set_item("world_z", 0.0)?;
        }
        out.push(cloned);
    }
    Ok(out)
}

fn push_capped(hist: &mut VecDeque<f64>, value: f64, max_len: usize) {
    while hist.len() >= max_len {
        hist.pop_front();
    }
    hist.push_back(value);
}

fn mean_hist(hist: &VecDeque<f64>) -> f64 {
    if hist.is_empty() {
        0.0
    } else {
        hist.iter().copied().sum::<f64>() / hist.len() as f64
    }
}

fn wrap_angle_deg(angle_deg: f64) -> f64 {
    (angle_deg + 180.0).rem_euclid(360.0) - 180.0
}

fn bbox_iou_xyxy(a: [f64; 4], b: [f64; 4]) -> f64 {
    let inter_x1 = a[0].max(b[0]);
    let inter_y1 = a[1].max(b[1]);
    let inter_x2 = a[2].min(b[2]);
    let inter_y2 = a[3].min(b[3]);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;
    if inter_area <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let denom = area_a + area_b - inter_area;
    if denom <= 0.0 {
        0.0
    } else {
        inter_area / denom
    }
}

#[pyclass]
#[derive(Clone)]
struct AdaptiveTuningMetrics {
    #[pyo3(get)]
    pose_valid_ratio: f64,
    #[pyo3(get)]
    avg_markers_used: f64,
    #[pyo3(get)]
    pose_position_jitter_m: f64,
    #[pyo3(get)]
    pose_rotation_jitter_deg: f64,
    #[pyo3(get)]
    coast_ratio: f64,
    #[pyo3(get)]
    id_switch_rate: f64,
    #[pyo3(get)]
    avg_fps: f64,
    #[pyo3(get)]
    avg_drop_frames: f64,
}

struct TrackEntry {
    cls_name: String,
    track_id: i64,
    bbox_xyxy: [f64; 4],
}

#[pyclass]
struct AdaptiveRuntimeTuner {
    enabled: bool,
    target_classes: HashSet<String>,
    window_frames: usize,
    update_interval_frames: usize,
    cooldown_frames: usize,
    iou_match_threshold: f64,
    motion_smoothing_min: f64,
    motion_smoothing_max: f64,
    motion_smoothing_step: f64,
    id_tau_on_min: f64,
    id_tau_on_max: f64,
    id_tau_off_min: f64,
    id_tau_off_max: f64,
    id_tau_step: f64,
    id_coast_frames_min: i64,
    id_coast_frames_max: i64,
    id_coast_step: i64,
    base_motion_smoothing: f64,
    base_tau_on: f64,
    base_tau_off: f64,
    base_coast_frames: i64,
    frame_count: usize,
    last_applied_frame: isize,
    pose_valid_hist: VecDeque<f64>,
    markers_used_hist: VecDeque<f64>,
    pose_position_jitter_hist: VecDeque<f64>,
    pose_rotation_jitter_hist: VecDeque<f64>,
    coast_ratio_hist: VecDeque<f64>,
    id_switch_rate_hist: VecDeque<f64>,
    prev_valid_pose_vec: Option<[f64; 6]>,
    prev_track_entries: Vec<TrackEntry>,
}

#[pymethods]
impl AdaptiveRuntimeTuner {
    #[new]
    #[pyo3(signature=(*, enabled, target_classes, window_frames, update_interval_frames, cooldown_frames, iou_match_threshold, motion_smoothing_min, motion_smoothing_max, motion_smoothing_step, id_tau_on_min, id_tau_on_max, id_tau_off_min, id_tau_off_max, id_tau_step, id_coast_frames_min, id_coast_frames_max, id_coast_step, base_motion_smoothing, base_tau_on, base_tau_off, base_coast_frames))]
    fn new(
        enabled: bool,
        target_classes: &PyAny,
        window_frames: usize,
        update_interval_frames: usize,
        cooldown_frames: usize,
        iou_match_threshold: f64,
        motion_smoothing_min: f64,
        motion_smoothing_max: f64,
        motion_smoothing_step: f64,
        id_tau_on_min: f64,
        id_tau_on_max: f64,
        id_tau_off_min: f64,
        id_tau_off_max: f64,
        id_tau_step: f64,
        id_coast_frames_min: i64,
        id_coast_frames_max: i64,
        id_coast_step: i64,
        base_motion_smoothing: f64,
        base_tau_on: f64,
        base_tau_off: f64,
        base_coast_frames: i64,
    ) -> PyResult<Self> {
        let target_classes = any_to_string_set(Some(target_classes))?.unwrap_or_default();
        let mut out = Self {
            enabled,
            target_classes,
            window_frames: window_frames.max(10),
            update_interval_frames: update_interval_frames.max(1),
            cooldown_frames,
            iou_match_threshold: clamp(iou_match_threshold, 0.05, 0.95),
            motion_smoothing_min: clamp(motion_smoothing_min, 0.0, 1.0),
            motion_smoothing_max: 0.0,
            motion_smoothing_step: clamp(motion_smoothing_step, 0.01, 0.25),
            id_tau_on_min: clamp(id_tau_on_min, 0.0, 1.0),
            id_tau_on_max: 0.0,
            id_tau_off_min: clamp(id_tau_off_min, 0.0, 1.0),
            id_tau_off_max: 0.0,
            id_tau_step: clamp(id_tau_step, 0.005, 0.2),
            id_coast_frames_min: id_coast_frames_min.max(0),
            id_coast_frames_max: 0,
            id_coast_step: id_coast_step.max(1),
            base_motion_smoothing: 0.0,
            base_tau_on: 0.0,
            base_tau_off: 0.0,
            base_coast_frames: 0,
            frame_count: 0,
            last_applied_frame: -(1 << 30),
            pose_valid_hist: VecDeque::new(),
            markers_used_hist: VecDeque::new(),
            pose_position_jitter_hist: VecDeque::new(),
            pose_rotation_jitter_hist: VecDeque::new(),
            coast_ratio_hist: VecDeque::new(),
            id_switch_rate_hist: VecDeque::new(),
            prev_valid_pose_vec: None,
            prev_track_entries: Vec::new(),
        };
        out.motion_smoothing_max = clamp(motion_smoothing_max.max(out.motion_smoothing_min), out.motion_smoothing_min, 1.0);
        out.id_tau_on_max = clamp(id_tau_on_max.max(out.id_tau_on_min), 0.0, 1.0);
        out.id_tau_off_max = clamp(id_tau_off_max.max(out.id_tau_off_min), 0.0, 1.0);
        out.id_coast_frames_max = id_coast_frames_max.max(out.id_coast_frames_min);
        out.base_motion_smoothing = clamp(base_motion_smoothing, out.motion_smoothing_min, out.motion_smoothing_max);
        out.base_tau_on = clamp(base_tau_on, out.id_tau_on_min, out.id_tau_on_max);
        out.base_tau_off = clamp(base_tau_off, out.id_tau_off_min, out.id_tau_off_max);
        out.base_coast_frames = base_coast_frames.max(out.id_coast_frames_min).min(out.id_coast_frames_max);
        out.reset();
        Ok(out)
    }

    fn reset(&mut self) {
        self.frame_count = 0;
        self.last_applied_frame = -(1 << 30);
        self.pose_valid_hist.clear();
        self.markers_used_hist.clear();
        self.pose_position_jitter_hist.clear();
        self.pose_rotation_jitter_hist.clear();
        self.coast_ratio_hist.clear();
        self.id_switch_rate_hist.clear();
        self.prev_valid_pose_vec = None;
        self.prev_track_entries.clear();
    }

    #[pyo3(signature=(*, pose_data, raw_detections, udp_ready_detections))]
    fn record_frame(&mut self, pose_data: &PyDict, raw_detections: &PyAny, udp_ready_detections: &PyAny) -> PyResult<()> {
        if !self.enabled {
            return Ok(());
        }
        self.frame_count += 1;
        self.record_pose_metrics(pose_data);
        self.record_continuity_metrics(udp_ready_detections)?;
        let id_switch_rate = self.estimate_id_switch_rate(raw_detections)?;
        push_capped(&mut self.id_switch_rate_hist, id_switch_rate, self.window_frames);
        Ok(())
    }

    #[pyo3(signature=(*, current_motion_smoothing, current_tau_on, current_tau_off, current_coast_frames, avg_fps, avg_drop_frames))]
    fn propose_adjustment(
        &mut self,
        py: Python<'_>,
        current_motion_smoothing: f64,
        current_tau_on: f64,
        current_tau_off: f64,
        current_coast_frames: i64,
        avg_fps: f64,
        avg_drop_frames: f64,
    ) -> PyResult<Option<Py<PyDict>>> {
        if !self.enabled {
            return Ok(None);
        }
        if self.pose_valid_hist.len() < self.window_frames.min(self.update_interval_frames) {
            return Ok(None);
        }
        if self.frame_count % self.update_interval_frames != 0 {
            return Ok(None);
        }
        if (self.frame_count as isize - self.last_applied_frame) < self.cooldown_frames as isize {
            return Ok(None);
        }

        let metrics = AdaptiveTuningMetrics {
            pose_valid_ratio: mean_hist(&self.pose_valid_hist),
            avg_markers_used: mean_hist(&self.markers_used_hist),
            pose_position_jitter_m: mean_hist(&self.pose_position_jitter_hist),
            pose_rotation_jitter_deg: mean_hist(&self.pose_rotation_jitter_hist),
            coast_ratio: mean_hist(&self.coast_ratio_hist),
            id_switch_rate: mean_hist(&self.id_switch_rate_hist),
            avg_fps: avg_fps.max(0.0),
            avg_drop_frames: avg_drop_frames.max(0.0),
        };
        let mode = self.choose_mode(&metrics);

        let cur_smooth = clamp(current_motion_smoothing, self.motion_smoothing_min, self.motion_smoothing_max);
        let cur_tau_on = clamp(current_tau_on, self.id_tau_on_min, self.id_tau_on_max);
        let cur_tau_off = clamp(current_tau_off, self.id_tau_off_min, self.id_tau_off_max);
        let cur_coast = current_coast_frames.max(self.id_coast_frames_min).min(self.id_coast_frames_max);

        let stable_target_smooth = self.motion_smoothing_min.max(self.base_motion_smoothing - 0.10);
        let stable_target_coast = self.id_coast_frames_min.max(self.base_coast_frames - 1);
        let jitter_target_smooth = self.motion_smoothing_max.min(self.base_motion_smoothing + 0.15);
        let jitter_target_coast = self.id_coast_frames_max.min(self.base_coast_frames + 2);
        let low_trust_target_smooth = self.motion_smoothing_max.min(self.base_motion_smoothing + 0.10);
        let low_trust_target_coast = self.id_coast_frames_min.max(self.base_coast_frames - 1);

        let (mut new_smooth, mut new_tau_on, mut new_tau_off, mut new_coast) = if mode == "stable" {
            (
                step_toward(cur_smooth, stable_target_smooth, self.motion_smoothing_step),
                step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step),
                step_toward(cur_tau_off, self.id_tau_off_max.min(self.base_tau_off + 0.02), self.id_tau_step),
                step_toward_int(cur_coast, stable_target_coast, self.id_coast_step),
            )
        } else if mode == "jittery_visible" {
            (
                step_toward(cur_smooth, jitter_target_smooth, self.motion_smoothing_step),
                step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step),
                step_toward(cur_tau_off, self.id_tau_off_min.max(self.base_tau_off - 0.06), self.id_tau_step),
                step_toward_int(cur_coast, jitter_target_coast, self.id_coast_step),
            )
        } else if mode == "low_trust" {
            (
                step_toward(cur_smooth, low_trust_target_smooth, self.motion_smoothing_step),
                step_toward(cur_tau_on, self.id_tau_on_max.min(self.base_tau_on + 0.04), self.id_tau_step),
                step_toward(cur_tau_off, self.id_tau_off_max.min(self.base_tau_off + 0.04), self.id_tau_step),
                step_toward_int(cur_coast, low_trust_target_coast, self.id_coast_step),
            )
        } else {
            (
                step_toward(cur_smooth, self.base_motion_smoothing, self.motion_smoothing_step),
                step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step),
                step_toward(cur_tau_off, self.base_tau_off, self.id_tau_step),
                step_toward_int(cur_coast, self.base_coast_frames, self.id_coast_step),
            )
        };

        new_smooth = clamp(new_smooth, self.motion_smoothing_min, self.motion_smoothing_max);
        new_tau_on = clamp(new_tau_on, self.id_tau_on_min, self.id_tau_on_max);
        new_tau_off = clamp(new_tau_off.min(new_tau_on), self.id_tau_off_min, self.id_tau_off_max);
        new_coast = new_coast.max(self.id_coast_frames_min).min(self.id_coast_frames_max);

        let changed = (new_smooth - cur_smooth).abs() > 1e-9
            || (new_tau_on - cur_tau_on).abs() > 1e-9
            || (new_tau_off - cur_tau_off).abs() > 1e-9
            || new_coast != cur_coast;

        if changed {
            self.last_applied_frame = self.frame_count as isize;
        }

        let result = PyDict::new_bound(py);
        result.set_item("mode", mode)?;
        result.set_item("changed", changed)?;
        result.set_item("motion_smoothing", new_smooth)?;
        result.set_item("tau_on", new_tau_on)?;
        result.set_item("tau_off", new_tau_off)?;
        result.set_item("coast_frames", new_coast)?;
        result.set_item("metrics", Py::new(py, metrics)?)?;
        Ok(Some(result.into()))
    }
}

impl AdaptiveRuntimeTuner {
    fn is_target_class(&self, det: &PyDict) -> bool {
        self.target_classes.contains(&dict_string_lower_chain(det, &["class", "class_name"]))
    }

    fn record_pose_metrics(&mut self, pose_data: &PyDict) {
        let pose_valid = safe_bool_opt(py_get(pose_data, "pose_valid"), false);
        push_capped(&mut self.pose_valid_hist, if pose_valid { 1.0 } else { 0.0 }, self.window_frames);
        push_capped(
            &mut self.markers_used_hist,
            safe_float_opt(py_get(pose_data, "markers_used"), 0.0).max(0.0),
            self.window_frames,
        );
        if !pose_valid {
            self.prev_valid_pose_vec = None;
            return;
        }

        let cur_vec = [
            safe_float_opt(py_get(pose_data, "x"), 0.0),
            safe_float_opt(py_get(pose_data, "altitude"), 0.0),
            safe_float_opt(py_get(pose_data, "z"), 0.0),
            safe_float_opt(py_get(pose_data, "yaw"), 0.0),
            safe_float_opt(py_get(pose_data, "pitch"), 0.0),
            safe_float_opt(py_get(pose_data, "roll"), 0.0),
        ];
        if let Some(prev) = self.prev_valid_pose_vec {
            let d_pos = [cur_vec[0] - prev[0], cur_vec[1] - prev[1], cur_vec[2] - prev[2]];
            let d_rot = [
                wrap_angle_deg(cur_vec[3] - prev[3]),
                wrap_angle_deg(cur_vec[4] - prev[4]),
                wrap_angle_deg(cur_vec[5] - prev[5]),
            ];
            push_capped(&mut self.pose_position_jitter_hist, vec_norm(&d_pos), self.window_frames);
            push_capped(&mut self.pose_rotation_jitter_hist, vec_norm(&d_rot), self.window_frames);
        }
        self.prev_valid_pose_vec = Some(cur_vec);
    }

    fn extract_track_entries(&self, detections: &PyAny) -> PyResult<Vec<TrackEntry>> {
        let mut entries = Vec::new();
        for det in pylist_to_dicts(detections)? {
            if !self.is_target_class(det) {
                continue;
            }
            let bbox = match parse_bbox(det) {
                Some(v) => v,
                None => continue,
            };
            let track_id = match py_get(det, "track_id").and_then(|v| v.extract::<i64>().ok()) {
                Some(v) => v,
                None => continue,
            };
            entries.push(TrackEntry {
                cls_name: dict_string_lower_chain(det, &["class", "class_name"]),
                track_id,
                bbox_xyxy: bbox,
            });
        }
        Ok(entries)
    }

    fn estimate_id_switch_rate(&mut self, raw_detections: &PyAny) -> PyResult<f64> {
        let cur_entries = self.extract_track_entries(raw_detections)?;
        if cur_entries.is_empty() {
            self.prev_track_entries.clear();
            return Ok(0.0);
        }
        let mut switches = 0usize;
        let mut matched = 0usize;
        let mut used_prev: HashSet<usize> = HashSet::new();

        for cur in &cur_entries {
            let mut best_idx: Option<usize> = None;
            let mut best_iou = 0.0;
            for (prev_idx, prev) in self.prev_track_entries.iter().enumerate() {
                if used_prev.contains(&prev_idx) || prev.cls_name != cur.cls_name {
                    continue;
                }
                let iou = bbox_iou_xyxy(cur.bbox_xyxy, prev.bbox_xyxy);
                if iou > best_iou {
                    best_iou = iou;
                    best_idx = Some(prev_idx);
                }
            }
            if let Some(best_idx) = best_idx {
                if best_iou >= self.iou_match_threshold {
                    used_prev.insert(best_idx);
                    matched += 1;
                    if self.prev_track_entries[best_idx].track_id != cur.track_id {
                        switches += 1;
                    }
                }
            }
        }
        self.prev_track_entries = cur_entries;
        if matched == 0 {
            Ok(0.0)
        } else {
            Ok(switches as f64 / matched as f64)
        }
    }

    fn record_continuity_metrics(&mut self, udp_ready_detections: &PyAny) -> PyResult<()> {
        let mut total = 0usize;
        let mut coasted = 0usize;
        for det in pylist_to_dicts(udp_ready_detections)? {
            if !self.is_target_class(det) {
                continue;
            }
            total += 1;
            if dict_string_lower_chain(det, &["continuity_state"]) == "coasted" {
                coasted += 1;
            }
        }
        let ratio = if total == 0 { 0.0 } else { coasted as f64 / total as f64 };
        push_capped(&mut self.coast_ratio_hist, ratio, self.window_frames);
        Ok(())
    }

    fn choose_mode(&self, metrics: &AdaptiveTuningMetrics) -> &'static str {
        if metrics.pose_valid_ratio < 0.35 || metrics.avg_markers_used < 0.75 || metrics.avg_drop_frames > 3.0 {
            return "low_trust";
        }

        let stable_pose = metrics.pose_valid_ratio >= 0.85;
        let stable_jitter = metrics.pose_position_jitter_m <= 0.05;
        let stable_rot = metrics.pose_rotation_jitter_deg <= 2.5;
        let stable_coast = metrics.coast_ratio <= 0.12;
        let stable_ids = metrics.id_switch_rate <= 0.05;
        let stable_drops = metrics.avg_drop_frames <= 1.0;
        if stable_pose && stable_jitter && stable_rot && stable_coast && stable_ids && stable_drops {
            return "stable";
        }

        if metrics.pose_valid_ratio >= 0.35
            && (metrics.pose_position_jitter_m >= 0.08
                || metrics.pose_rotation_jitter_deg >= 4.0
                || metrics.coast_ratio >= 0.18
                || metrics.id_switch_rate >= 0.08)
        {
            return "jittery_visible";
        }
        "recovering"
    }
}

fn step_toward(current: f64, target: f64, step: f64) -> f64 {
    let step = step.abs();
    if (target - current).abs() <= step {
        target
    } else if target > current {
        current + step
    } else {
        current - step
    }
}

fn step_toward_int(current: i64, target: i64, step: i64) -> i64 {
    let step = step.max(1);
    if (target - current).abs() <= step {
        target
    } else if target > current {
        current + step
    } else {
        current - step
    }
}

fn alpha(cutoff_hz: f64, dt: f64) -> f64 {
    let cutoff = cutoff_hz.max(1e-6);
    let dt = dt.max(1e-6);
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

fn quat_normalize(mut q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n <= 0.0 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        q[0] /= n;
        q[1] /= n;
        q[2] /= n;
        q[3] /= n;
        q
    }
}

fn quat_from_rotmat(r: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let (w, x, y, z) = if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).sqrt();
        (
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        )
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = 2.0 * (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt();
        (
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        )
    } else if r[1][1] > r[2][2] {
        let s = 2.0 * (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt();
        (
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        )
    } else {
        let s = 2.0 * (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt();
        (
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        )
    };
    quat_normalize([w, x, y, z])
}

fn rotmat_from_quat(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]
}

fn quat_dot(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn quat_slerp(q0: [f64; 4], mut q1: [f64; 4], t: f64) -> [f64; 4] {
    let q0 = quat_normalize(q0);
    let mut dot = quat_dot(q0, q1);
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }
    let t = clamp01(t);
    if dot > 0.9995 {
        return quat_normalize([
            q0[0] + t * (q1[0] - q0[0]),
            q0[1] + t * (q1[1] - q0[1]),
            q0[2] + t * (q1[2] - q0[2]),
            q0[3] + t * (q1[3] - q0[3]),
        ]);
    }
    let theta_0 = dot.clamp(-1.0, 1.0).acos();
    let sin_theta_0 = theta_0.sin();
    if sin_theta_0 <= 1e-8 {
        return q0;
    }
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let s0 = (theta_0 - theta).sin() / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;
    quat_normalize([
        s0 * q0[0] + s1 * q1[0],
        s0 * q0[1] + s1 * q1[1],
        s0 * q0[2] + s1 * q1[2],
        s0 * q0[3] + s1 * q1[3],
    ])
}

fn quat_angle(q0: [f64; 4], q1: [f64; 4]) -> f64 {
    let q0 = quat_normalize(q0);
    let q1 = quat_normalize(q1);
    let dot = quat_dot(q0, q1).abs();
    2.0 * dot.clamp(-1.0, 1.0).acos()
}

fn ypr_from_r_wc(r_wc: [[f64; 3]; 3]) -> [f64; 3] {
    let r_cw = mat3_transpose(r_wc);
    let yaw = (r_cw[0][2]).atan2(r_cw[2][2]).to_degrees();
    let pitch = -(-r_cw[1][2]).clamp(-1.0, 1.0).asin().to_degrees();
    let roll = (r_cw[1][0]).atan2(r_cw[1][1]).to_degrees();
    [yaw, pitch, roll]
}

fn pose_position_params_from_smoothness(smoothness: f64) -> (f64, f64) {
    let s = clamp01(smoothness);
    (3.5 - 3.1 * s, 0.03 + 0.32 * s)
}

fn pose_rotation_params_from_smoothness(smoothness: f64) -> (f64, f64) {
    let s = clamp01(smoothness);
    (4.5 - 3.8 * s, 0.04 + 0.42 * s)
}

fn world_params_from_smoothness(smoothness: f64) -> (f64, f64) {
    let s = clamp01(smoothness);
    (3.0 - 2.6 * s, 0.02 + 0.24 * s)
}

#[derive(Clone)]
struct VecFilterCore {
    min_cutoff_hz: f64,
    beta: f64,
    d_cutoff_hz: f64,
    x_prev: Option<Vec<f64>>,
    dx_hat: Option<Vec<f64>>,
}

impl VecFilterCore {
    fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self { min_cutoff_hz, beta, d_cutoff_hz, x_prev: None, dx_hat: None }
    }
    fn reset(&mut self) {
        self.x_prev = None;
        self.dx_hat = None;
    }
    fn set_params(&mut self, min_cutoff_hz: f64, beta: f64, d_cutoff_hz: Option<f64>) {
        self.min_cutoff_hz = min_cutoff_hz;
        self.beta = beta;
        if let Some(v) = d_cutoff_hz {
            self.d_cutoff_hz = v;
        }
    }
    fn filter_vec(&mut self, x: Vec<f64>, dt: f64) -> Vec<f64> {
        let dt = dt.max(1e-6);
        if self.x_prev.is_none() {
            self.x_prev = Some(x.clone());
            self.dx_hat = Some(vec![0.0; x.len()]);
            return x;
        }
        let prev = self.x_prev.clone().unwrap();
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

#[derive(Clone)]
struct QuatFilterCore {
    min_cutoff_hz: f64,
    beta: f64,
    d_cutoff_hz: f64,
    q_prev: Option<[f64; 4]>,
    speed_hat: f64,
}

impl QuatFilterCore {
    fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self { min_cutoff_hz, beta, d_cutoff_hz, q_prev: None, speed_hat: 0.0 }
    }
    fn reset(&mut self) {
        self.q_prev = None;
        self.speed_hat = 0.0;
    }
    fn set_params(&mut self, min_cutoff_hz: f64, beta: f64, d_cutoff_hz: Option<f64>) {
        self.min_cutoff_hz = min_cutoff_hz;
        self.beta = beta;
        if let Some(v) = d_cutoff_hz {
            self.d_cutoff_hz = v;
        }
    }
    fn filter_quat(&mut self, q: [f64; 4], dt: f64) -> [f64; 4] {
        let q = quat_normalize(q);
        let dt = dt.max(1e-6);
        if self.q_prev.is_none() {
            self.q_prev = Some(q);
            self.speed_hat = 0.0;
            return q;
        }
        let prev = self.q_prev.unwrap();
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
struct OneEuroVectorFilterPy {
    inner: VecFilterCore,
}

#[pymethods]
impl OneEuroVectorFilterPy {
    #[new]
    #[pyo3(signature=(*, min_cutoff_hz, beta, d_cutoff_hz=1.0))]
    fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self { inner: VecFilterCore::new(min_cutoff_hz, beta, d_cutoff_hz) }
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
struct OneEuroQuaternionFilterPy {
    inner: QuatFilterCore,
}

#[pymethods]
impl OneEuroQuaternionFilterPy {
    #[new]
    #[pyo3(signature=(*, min_cutoff_hz, beta, d_cutoff_hz=1.0))]
    fn new(min_cutoff_hz: f64, beta: f64, d_cutoff_hz: f64) -> Self {
        Self { inner: QuatFilterCore::new(min_cutoff_hz, beta, d_cutoff_hz) }
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
            return Err(pyo3::exceptions::PyValueError::new_err("Quaternion must have length 4"));
        }
        Ok(self.inner.filter_quat([q[0], q[1], q[2], q[3]], dt).to_vec())
    }
}

#[pyclass(name = "PoseMotionSmootherCore")]
struct PoseMotionSmootherCore {
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
    fn new(enabled: bool, smoothness: f64, derivative_cutoff_hz: f64, reset_timeout_s: f64, default_fps: f64) -> Self {
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
        self.position_filter.set_params(pos_cutoff, pos_beta, Some(self.derivative_cutoff_hz));
        self.rotation_filter.set_params(rot_cutoff, rot_beta, Some(self.derivative_cutoff_hz));
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
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid pose primitive sizes"));
        }
        let dt = self.resolve_dt(timestamp);
        let c_w_smooth = self.position_filter.filter_vec(c_w.clone(), dt);
        let r_wc_mat = mat3_from_flat(&r_wc).unwrap();
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

#[pyclass(name = "WorldTrackSmootherCore")]
struct WorldTrackSmootherCore {
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
    fn new(enabled: bool, smoothness: f64, derivative_cutoff_hz: f64, reset_timeout_s: f64, max_track_age_s: f64, default_fps: f64) -> Self {
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
    fn update_detections(&mut self, py: Python<'_>, detections: &PyAny, timestamp: Option<f64>) -> PyResult<Vec<Py<PyDict>>> {
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
        let Some(ts) = timestamp else { return; };
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

#[pymodule]
fn xrdrone_native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clamp01_py, m)?)?;
    m.add_function(wrap_pyfunction!(passes_udp_world_projection_filter, m)?)?;
    m.add_function(wrap_pyfunction!(attach_foot_and_world, m)?)?;
    m.add_function(wrap_pyfunction!(to_unity_udp_packet, m)?)?;
    m.add_class::<RobustIdFlickerMitigator>()?;
    m.add_class::<AdaptiveTuningMetrics>()?;
    m.add_class::<AdaptiveRuntimeTuner>()?;
    m.add_class::<OneEuroVectorFilterPy>()?;
    m.add_class::<OneEuroQuaternionFilterPy>()?;
    m.add_class::<PoseMotionSmootherCore>()?;
    m.add_class::<WorldTrackSmootherCore>()?;
    Ok(())
}
