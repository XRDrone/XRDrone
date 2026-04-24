use crate::common::{
    any_to_string_set, bbox_iou_xyxy, clamp, dict_string_lower_chain, mean_hist, parse_bbox,
    push_capped, py_get, pylist_to_dicts, safe_bool_opt, safe_float_opt, step_toward,
    step_toward_int, wrap_angle_deg,
};
use crate::geometry::vec_norm;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::collections::{HashSet, VecDeque};

#[derive(Clone)]
#[pyclass]
pub struct AdaptiveTuningMetrics {
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

/// Bounded runtime tuner that reacts to pose quality and continuity stability.
#[pyclass]
pub struct AdaptiveRuntimeTuner {
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
        out.motion_smoothing_max = clamp(
            motion_smoothing_max.max(out.motion_smoothing_min),
            out.motion_smoothing_min,
            1.0,
        );
        out.id_tau_on_max = clamp(id_tau_on_max.max(out.id_tau_on_min), 0.0, 1.0);
        out.id_tau_off_max = clamp(id_tau_off_max.max(out.id_tau_off_min), 0.0, 1.0);
        out.id_coast_frames_max = id_coast_frames_max.max(out.id_coast_frames_min);
        out.base_motion_smoothing = clamp(
            base_motion_smoothing,
            out.motion_smoothing_min,
            out.motion_smoothing_max,
        );
        out.base_tau_on = clamp(base_tau_on, out.id_tau_on_min, out.id_tau_on_max);
        out.base_tau_off = clamp(base_tau_off, out.id_tau_off_min, out.id_tau_off_max);
        out.base_coast_frames = base_coast_frames
            .max(out.id_coast_frames_min)
            .min(out.id_coast_frames_max);
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
    fn record_frame(
        &mut self,
        pose_data: &PyDict,
        raw_detections: &PyAny,
        udp_ready_detections: &PyAny,
    ) -> PyResult<()> {
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

        let cur_smooth = clamp(
            current_motion_smoothing,
            self.motion_smoothing_min,
            self.motion_smoothing_max,
        );
        let cur_tau_on = clamp(current_tau_on, self.id_tau_on_min, self.id_tau_on_max);
        let cur_tau_off = clamp(current_tau_off, self.id_tau_off_min, self.id_tau_off_max);
        let cur_coast = current_coast_frames
            .max(self.id_coast_frames_min)
            .min(self.id_coast_frames_max);

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
                step_toward(
                    cur_tau_off,
                    self.id_tau_off_max.min(self.base_tau_off + 0.02),
                    self.id_tau_step,
                ),
                step_toward_int(cur_coast, stable_target_coast, self.id_coast_step),
            )
        } else if mode == "jittery_visible" {
            (
                step_toward(cur_smooth, jitter_target_smooth, self.motion_smoothing_step),
                step_toward(cur_tau_on, self.base_tau_on, self.id_tau_step),
                step_toward(
                    cur_tau_off,
                    self.id_tau_off_min.max(self.base_tau_off - 0.06),
                    self.id_tau_step,
                ),
                step_toward_int(cur_coast, jitter_target_coast, self.id_coast_step),
            )
        } else if mode == "low_trust" {
            (
                step_toward(cur_smooth, low_trust_target_smooth, self.motion_smoothing_step),
                step_toward(
                    cur_tau_on,
                    self.id_tau_on_max.min(self.base_tau_on + 0.04),
                    self.id_tau_step,
                ),
                step_toward(
                    cur_tau_off,
                    self.id_tau_off_max.min(self.base_tau_off + 0.04),
                    self.id_tau_step,
                ),
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
        new_tau_off = clamp(
            new_tau_off.min(new_tau_on),
            self.id_tau_off_min,
            self.id_tau_off_max,
        );
        new_coast = new_coast
            .max(self.id_coast_frames_min)
            .min(self.id_coast_frames_max);

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
        self.target_classes
            .contains(&dict_string_lower_chain(det, &["class", "class_name"]))
    }

    fn record_pose_metrics(&mut self, pose_data: &PyDict) {
        let pose_valid = safe_bool_opt(py_get(pose_data, "pose_valid"), false);
        push_capped(
            &mut self.pose_valid_hist,
            if pose_valid { 1.0 } else { 0.0 },
            self.window_frames,
        );
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
            push_capped(
                &mut self.pose_position_jitter_hist,
                vec_norm(&d_pos),
                self.window_frames,
            );
            push_capped(
                &mut self.pose_rotation_jitter_hist,
                vec_norm(&d_rot),
                self.window_frames,
            );
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
        let ratio = if total == 0 {
            0.0
        } else {
            coasted as f64 / total as f64
        };
        push_capped(&mut self.coast_ratio_hist, ratio, self.window_frames);
        Ok(())
    }

    fn choose_mode(&self, metrics: &AdaptiveTuningMetrics) -> &'static str {
        if metrics.pose_valid_ratio < 0.35
            || metrics.avg_markers_used < 0.75
            || metrics.avg_drop_frames > 3.0
        {
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
