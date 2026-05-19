use crate::common::clamp01;

/// Convert an `xyxy` pixel box into normalized `xywh` values.
pub(crate) fn xyxy_to_xywhn(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    width: f64,
    height: f64,
) -> (f64, f64, f64, f64) {
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

/// Interpret a 9-element flat buffer as a row-major 3x3 matrix.
pub(crate) fn mat3_from_flat(values: &[f64]) -> Option<[[f64; 3]; 3]> {
    if values.len() != 9 {
        return None;
    }
    Some([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ])
}

/// Flatten a row-major 3x3 matrix to a 9-element vector.
pub(crate) fn flat_from_mat3(m: [[f64; 3]; 3]) -> Vec<f64> {
    vec![
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ]
}

/// Invert a 3x3 matrix using the adjugate/determinant formula.
pub(crate) fn inverse_3x3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
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

/// Matrix-vector multiply for a 3x3 matrix and 3-vector.
pub(crate) fn mat3_vec3_mul(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Matrix transpose for a 3x3 matrix.
pub(crate) fn mat3_transpose(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Euclidean norm of a dynamic vector.
pub(crate) fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Euclidean norm of a 3-vector.
pub(crate) fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Normalize a 3-vector if its norm is non-zero.
pub(crate) fn vec3_normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = vec3_norm(v);
    if n <= 0.0 {
        None
    } else {
        Some([v[0] / n, v[1] / n, v[2] / n])
    }
}

/// Convert a pixel coordinate into a unit ray expressed in world coordinates.
pub(crate) fn pixel_ray_in_world(
    _c_w: [f64; 3],
    r_wc: [[f64; 3]; 3],
    k: [[f64; 3]; 3],
    u_px: f64,
    v_px: f64,
) -> Option<[f64; 3]> {
    let k_inv = inverse_3x3(k)?;
    let d_c = mat3_vec3_mul(k_inv, [u_px, v_px, 1.0]);
    let d_c = vec3_normalize(d_c)?;
    let r_cw = mat3_transpose(r_wc);
    let d_w = mat3_vec3_mul(r_cw, d_c);
    vec3_normalize(d_w)
}

/// Intersect the camera ray through a pixel with the world `y = 0` plane.
pub(crate) fn intersect_plane_y0(
    c_w: [f64; 3],
    r_wc: [[f64; 3]; 3],
    k: [[f64; 3]; 3],
    u_px: f64,
    v_px: f64,
) -> Option<[f64; 3]> {
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

/// First-order low-pass alpha from cutoff frequency and timestep.
pub(crate) fn alpha(cutoff_hz: f64, dt: f64) -> f64 {
    let cutoff = cutoff_hz.max(1e-6);
    let dt = dt.max(1e-6);
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

/// Normalize a quaternion, defaulting to identity for degenerate inputs.
pub(crate) fn quat_normalize(mut q: [f64; 4]) -> [f64; 4] {
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

/// Convert a rotation matrix into a unit quaternion.
pub(crate) fn quat_from_rotmat(r: [[f64; 3]; 3]) -> [f64; 4] {
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

/// Convert a unit quaternion into a rotation matrix.
pub(crate) fn rotmat_from_quat(q: [f64; 4]) -> [[f64; 3]; 3] {
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

/// Quaternion dot product.
pub(crate) fn quat_dot(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Spherical linear interpolation between unit quaternions.
pub(crate) fn quat_slerp(q0: [f64; 4], mut q1: [f64; 4], t: f64) -> [f64; 4] {
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

/// Angular distance between two quaternions in radians.
pub(crate) fn quat_angle(q0: [f64; 4], q1: [f64; 4]) -> f64 {
    let q0 = quat_normalize(q0);
    let q1 = quat_normalize(q1);
    let dot = quat_dot(q0, q1).abs();
    2.0 * dot.clamp(-1.0, 1.0).acos()
}

/// Convert world-to-camera rotation into yaw, pitch, and roll in degrees.
pub(crate) fn ypr_from_r_wc(r_wc: [[f64; 3]; 3]) -> [f64; 3] {
    let r_cw = mat3_transpose(r_wc);
    let yaw = (r_cw[0][2]).atan2(r_cw[2][2]).to_degrees();
    let pitch = -(-r_cw[1][2]).clamp(-1.0, 1.0).asin().to_degrees();
    let roll = (r_cw[1][0]).atan2(r_cw[1][1]).to_degrees();
    [yaw, pitch, roll]
}

/// Map user-facing smoothness into One Euro position-filter parameters.
pub(crate) fn pose_position_params_from_smoothness(smoothness: f64) -> (f64, f64) {
    let s = clamp01(smoothness);
    (3.5 - 3.1 * s, 0.03 + 0.32 * s)
}

/// Map user-facing smoothness into One Euro rotation-filter parameters.
pub(crate) fn pose_rotation_params_from_smoothness(smoothness: f64) -> (f64, f64) {
    let s = clamp01(smoothness);
    (4.5 - 3.8 * s, 0.04 + 0.42 * s)
}

/// Map user-facing smoothness into One Euro world-track filter parameters.
pub(crate) fn world_params_from_smoothness(smoothness: f64) -> (f64, f64) {
    let s = clamp01(smoothness);
    (3.0 - 2.6 * s, 0.02 + 0.24 * s)
}
