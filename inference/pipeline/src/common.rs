use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::{HashMap, HashSet, VecDeque};

/// Clamp a scalar into the inclusive `[0, 1]` range.
pub(crate) fn clamp01(x: f64) -> f64 {
    x.max(0.0).min(1.0)
}

/// Clamp a scalar into an arbitrary inclusive range.
pub(crate) fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Extract a finite float from a Python object, falling back to a default.
pub(crate) fn safe_float_obj(obj: &PyAny, default: f64) -> f64 {
    match obj.extract::<f64>() {
        Ok(v) if v.is_finite() => v,
        _ => default,
    }
}

/// Extract an optional finite float from a Python object, falling back to a default.
pub(crate) fn safe_float_opt(obj: Option<&PyAny>, default: f64) -> f64 {
    match obj {
        Some(v) => safe_float_obj(v, default),
        None => default,
    }
}

/// Extract an optional boolean from a Python object, falling back to a default.
pub(crate) fn safe_bool_opt(obj: Option<&PyAny>, default: bool) -> bool {
    match obj {
        Some(v) => v.extract::<bool>().unwrap_or(default),
        None => default,
    }
}

/// Convenience accessor for Python dictionaries that gracefully ignores missing keys.
pub(crate) fn py_get<'a>(det: &'a PyDict, key: &str) -> Option<&'a PyAny> {
    det.get_item(key).ok().flatten()
}

/// Look up the first string field present in `keys` and normalize it to lowercase.
pub(crate) fn dict_string_lower_chain(det: &PyDict, keys: &[&str]) -> String {
    for key in keys {
        if let Some(value) = py_get(det, key) {
            if let Ok(s) = value.extract::<String>() {
                return s.to_lowercase();
            }
        }
    }
    String::new()
}

/// Create a shallow clone of a Python dict, copying nested dict/list containers.
pub(crate) fn clone_py_dict(py: Python<'_>, det: &PyDict) -> PyResult<Py<PyDict>> {
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

/// Clone a Python dict and append additional key/value pairs to the clone.
pub(crate) fn clone_py_dict_with_items(
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

/// Convert a Python iterable of strings to an optional lowercase set.
pub(crate) fn any_to_string_set(values: Option<&PyAny>) -> PyResult<Option<HashSet<String>>> {
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

/// Convert a Python dict of class-name to integer-ID mappings into a lowercase Rust map.
pub(crate) fn any_to_class_map(values: Option<&PyAny>) -> PyResult<HashMap<String, i64>> {
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

/// Materialize a Python iterable of dictionaries as borrowed `PyDict` references.
pub(crate) fn pylist_to_dicts<'a>(obj: &'a PyAny) -> PyResult<Vec<&'a PyDict>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let item = item?;
        let det: &PyDict = item.downcast()?;
        out.push(det);
    }
    Ok(out)
}

/// Parse a detection `bbox_xyxy` field into a fixed-size Rust array.
pub(crate) fn parse_bbox(det: &PyDict) -> Option<[f64; 4]> {
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

/// Append to a bounded history buffer, evicting oldest values first.
pub(crate) fn push_capped(hist: &mut VecDeque<f64>, value: f64, max_len: usize) {
    while hist.len() >= max_len {
        hist.pop_front();
    }
    hist.push_back(value);
}

/// Mean of a bounded history buffer.
pub(crate) fn mean_hist(hist: &VecDeque<f64>) -> f64 {
    if hist.is_empty() {
        0.0
    } else {
        hist.iter().copied().sum::<f64>() / hist.len() as f64
    }
}

/// Wrap an angle in degrees into the `[-180, 180)` interval.
pub(crate) fn wrap_angle_deg(angle_deg: f64) -> f64 {
    (angle_deg + 180.0).rem_euclid(360.0) - 180.0
}

/// Standard intersection-over-union for `xyxy` boxes.
pub(crate) fn bbox_iou_xyxy(a: [f64; 4], b: [f64; 4]) -> f64 {
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

/// Step a float toward a target by at most `step`.
pub(crate) fn step_toward(current: f64, target: f64, step: f64) -> f64 {
    let step = step.abs();
    if (target - current).abs() <= step {
        target
    } else if target > current {
        current + step
    } else {
        current - step
    }
}

/// Step an integer toward a target by at most `step`.
pub(crate) fn step_toward_int(current: i64, target: i64, step: i64) -> i64 {
    let step = step.max(1);
    if (target - current).abs() <= step {
        target
    } else if target > current {
        current + step
    } else {
        current - step
    }
}
