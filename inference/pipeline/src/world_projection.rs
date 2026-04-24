use crate::common::{
    any_to_string_set, clamp01, clone_py_dict, dict_string_lower_chain, parse_bbox, py_get,
    pylist_to_dicts, safe_bool_opt, safe_float_opt,
};
use crate::geometry::{intersect_plane_y0, mat3_from_flat};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// Expose the common `[0, 1]` clamp helper to Python.
#[pyfunction]
#[pyo3(name = "clamp01")]
pub fn clamp01_py(x: f64) -> f64 {
    clamp01(x)
}

/// Check whether a detection is eligible for world projection and UDP output.
#[pyfunction]
#[pyo3(signature=(det, *, allowed_classes=None, min_conf=None))]
pub fn passes_udp_world_projection_filter(
    det: &PyDict,
    allowed_classes: Option<&PyAny>,
    min_conf: Option<f64>,
) -> PyResult<bool> {
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

/// Attach normalized foot points and optional world-ground intersections to detections.
#[pyfunction]
#[pyo3(signature=(detections, *, pose_valid, pose_camera_world=None, pose_rotation_world_to_camera=None, pose_intrinsics=None, width, height, projection_classes=None, projection_min_conf=None))]
pub fn attach_foot_and_world(
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

    let c_w = pose_camera_world.and_then(|v| {
        if v.len() >= 3 {
            Some([v[0], v[1], v[2]])
        } else {
            None
        }
    });
    let r_wc = pose_rotation_world_to_camera.and_then(|v| mat3_from_flat(&v));
    let k = pose_intrinsics.and_then(|v| mat3_from_flat(&v));
    let pose_ready = match (pose_valid, c_w, r_wc, k) {
        (true, Some(c_w), Some(r_wc), Some(k)) => Some((c_w, r_wc, k)),
        _ => None,
    };

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

        if pose_ready.is_none() || !should_project {
            det_out.set_item("world_valid", false)?;
            det_out.set_item("world_x", 0.0)?;
            det_out.set_item("world_y", 0.0)?;
            det_out.set_item("world_z", 0.0)?;
            out.push(cloned);
            continue;
        }

        let point_world = match pose_ready {
            Some((c_w, r_wc, k)) => intersect_plane_y0(c_w, r_wc, k, foot_x_px, foot_y_px),
            None => None,
        };
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
