use crate::common::{
    any_to_string_set, clamp01, clone_py_dict, clone_py_dict_with_items, dict_string_lower_chain,
    py_get, pylist_to_dicts, safe_float_opt,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::{HashMap, HashSet};

/// Per-track continuity state used by the native ID flicker mitigator.
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

/// Rust implementation of tracked-object continuity filtering and coasting.
#[pyclass(name = "RobustIDFlickerMitigator")]
pub struct RobustIdFlickerMitigator {
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

    /// Apply hysteresis, coasting, and stale-state cleanup to a detection batch.
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
