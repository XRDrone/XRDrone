use crate::common::{
    any_to_class_map, any_to_string_set, dict_string_lower_chain, parse_bbox, py_get, safe_bool_opt,
    safe_float_opt,
};
use crate::geometry::xyxy_to_xywhn;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};

/// Convert merged detections into the Unity-facing UDP packet schema.
#[pyfunction]
#[pyo3(signature=(merged_detections, *, frame_id, timestamp, width, height, class_map=None, allowed_classes=None, min_conf=None))]
pub fn to_unity_udp_packet(
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

    for (i, det) in crate::common::pylist_to_dicts(merged_detections)?.iter().enumerate() {
        let cls_name = dict_string_lower_chain(det, &["class", "class_name"]);
        if let Some(allow_set) = &allow {
            if !allow_set.contains(&cls_name) {
                continue;
            }
        }

        let conf = safe_float_opt(
            py_get(det, "udp_confidence").or_else(|| py_get(det, "confidence")),
            0.0,
        );
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
        let (cx, cy, w, h) =
            xyxy_to_xywhn(bbox[0], bbox[1], bbox[2], bbox[3], width as f64, height as f64);
        let default_foot_x = cx;
        let default_foot_y = crate::common::clamp01(cy + h / 2.0);

        let det_id = py_get(det, "track_id")
            .and_then(|v| v.extract::<i64>().ok())
            .unwrap_or(i as i64);

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
        dets.append(out_det)?;
    }

    packet.set_item("frame_id", frame_id)?;
    packet.set_item("timestamp", timestamp)?;
    packet.set_item("width", width)?;
    packet.set_item("height", height)?;
    packet.set_item("detections", dets)?;
    Ok(packet.into())
}
