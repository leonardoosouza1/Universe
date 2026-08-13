//! Ideal racing line (outside-in-outside) as a precomputed smoothed polyline.
//!
//! Built once from the track centerline (VoxelWorld::waypoints_world) plus the
//! track half-width. The racing line displaces the centerline toward the
//! INSIDE of each corner by a fraction of the half-width, proportional to the
//! local (smoothed) curvature sign. Straights ≈ centerline; corners pull to
//! apex — wide-apex-wide emerges naturally from the smoothing window.
//!
//! API:
//!   - `RacingLine::from_centerline(&centerline, half_width)` — builder.
//!   - `RacingLine::lateral_offset_from_line(pos, tangent)` — signed deviation
//!     from the ideal line at the car's nearest projection.
//!     Positive => car is to the LEFT of the ideal line (must steer right).
//!     Negative => car is to the RIGHT of the ideal line (must steer left).
//!
//! Everything is 2D (xz world plane, stored as Vec2(x,z)).

use bevy::prelude::*;

/// Offset factor relative to half-width: apex pulls at most 0.35 * half_width
/// toward the inside of the turn. Keeps the car on pavement even on tight
/// tracks (track_width 18m → ±3.15m from centerline).
const APEX_OFFSET_FRAC: f32 = 0.35;

/// Smoothing window radius (±N points) for the curvature sign field.
/// Dense polyline is resampled at RESAMPLE_STEP m, so ±8 points ≈ ±24m
/// of track around each sample — roughly one corner entry + apex + exit.
const CURV_SMOOTH_RADIUS: usize = 8;

/// Target spacing between racing line samples (meters). Small enough for
/// sub-meter lookup precision; large enough to keep the array tiny
/// (one F1 lap @ 3m ≈ a few hundred points).
const RESAMPLE_STEP: f32 = 3.0;

/// Precomputed ideal racing line — a closed polyline of world-space points.
#[derive(Clone, Debug, Default)]
pub struct RacingLine {
    /// Ideal line points in world (x,z). The curve is closed: last == first
    /// (matches the centerline contract from world.rs).
    pub points: Vec<Vec2>,
    /// Unit tangent at each point (finite diff). Same length as `points`.
    pub tangents: Vec<Vec2>,
    /// Cumulative arc length at each point (points[0] = 0.0). len == points.len().
    pub arc: Vec<f32>,
    /// Total line length.
    pub total_len: f32,
    /// Half of track width (for clamping offsets, scaling heuristics).
    pub half_width: f32,
}

impl RacingLine {
    /// Build racing line from a closed centerline polyline (last == first) and
    /// the track width in meters. Safe on malformed input (returns empty).
    pub fn from_centerline(centerline: &[Vec2], track_width: f32) -> Self {
        let half_width = track_width * 0.5;
        if centerline.len() < 3 {
            return Self::default();
        }

        // 1. Resample centerline at uniform arc length.
        let dense = resample_polyline(centerline, RESAMPLE_STEP);
        let n = dense.len();
        if n < 3 {
            return Self::default();
        }

        // 2. Compute signed curvature magnitude per point via cross of successive
        //    tangents. Positive = left turn, negative = right turn (xz plane).
        let mut curv_sign = vec![0.0f32; n];
        for i in 0..n {
            let prev = dense[(i + n - 1) % n];
            let cur = dense[i];
            let next = dense[(i + 1) % n];
            let a = (cur - prev).normalize_or_zero();
            let b = (next - cur).normalize_or_zero();
            // 2D cross product (z of cross in 3D).
            curv_sign[i] = a.x * b.y - a.y * b.x;
        }

        // 3. Smooth the curvature sign with a moving average window of
        //    ±CURV_SMOOTH_RADIUS. Produces a gentle, signed field in roughly
        //    [-1,1]; near-zero on straights, saturates through corners.
        let w = CURV_SMOOTH_RADIUS;
        let mut smoothed = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = 0.0f32;
            let mut cnt = 0.0f32;
            for k in 0..=2 * w {
                let idx = (i + n + k - w) % n;
                acc += curv_sign[idx];
                cnt += 1.0;
            }
            // Normalize by window size AND by a reasonable curvature scale
            // (clamp so straight segments stay tiny, corners saturate ~1).
            let avg = acc / cnt;
            smoothed[i] = (avg * 8.0).clamp(-1.0, 1.0);
        }

        // 4. Offset each centerline point by APEX_OFFSET_FRAC * half_width
        //    toward the INSIDE of the turn. Inside of a left turn (curv_sign > 0)
        //    is to the LEFT of the tangent → normal = rotate tangent by +90°.
        //    Inside of a right turn is to the right → opposite sign.
        let mut points = Vec::with_capacity(n + 1);
        let mut tangents = Vec::with_capacity(n + 1);
        for i in 0..n {
            let prev = dense[(i + n - 1) % n];
            let next = dense[(i + 1) % n];
            let tan = (next - prev).normalize_or_zero();
            // Left-hand normal: rotate tan +90° in xz plane => (-tan.y, tan.x).
            let normal_left = Vec2::new(-tan.y, tan.x);
            // curv_sign > 0 means left turn → inside is LEFT → move along normal_left.
            let offset_mag = APEX_OFFSET_FRAC * half_width * smoothed[i];
            let p = dense[i] + normal_left * offset_mag;
            points.push(p);
            tangents.push(tan);
        }
        // Close the loop explicitly (mirrors centerline convention).
        points.push(points[0]);
        tangents.push(tangents[0]);

        // 5. Arc-length table.
        let mut arc = vec![0.0f32; points.len()];
        let mut total = 0.0f32;
        for i in 1..points.len() {
            total += points[i - 1].distance(points[i]);
            arc[i] = total;
        }

        Self {
            points,
            tangents,
            arc,
            total_len: total,
            half_width,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.points.len() < 2
    }

    /// Project `pos` onto the ideal line, return (segment_index, t_in_seg[0..1], projected_point, segment_tangent).
    fn project(&self, pos: Vec2) -> Option<(usize, f32, Vec2, Vec2)> {
        if self.points.len() < 2 { return None; }
        let mut best = (0usize, 0.0f32, self.points[0], self.tangents[0]);
        let mut best_d = f32::INFINITY;
        for i in 0..self.points.len() - 1 {
            let a = self.points[i];
            let b = self.points[i + 1];
            let ab = b - a;
            let len2 = ab.length_squared().max(1e-6);
            let t = ((pos - a).dot(ab) / len2).clamp(0.0, 1.0);
            let p = a + ab * t;
            let d = pos.distance_squared(p);
            if d < best_d {
                best_d = d;
                let tan = ab.normalize_or_zero();
                best = (i, t, p, tan);
            }
        }
        Some(best)
    }

    /// Signed lateral deviation of `pos` from the ideal line.
    ///
    /// Sign convention: positive => car is to the LEFT of the ideal line
    /// relative to the racing-line tangent (car must steer RIGHT to reduce).
    /// Negative => car to the right (must steer LEFT).
    /// Units: meters. Clamped to ±half_width to stay in a sane range.
    ///
    /// `tangent` argument is currently UNUSED by the projection itself but
    /// kept in the signature per API contract — could be used to disambiguate
    /// projections on self-intersecting lines in future work.
    pub fn lateral_offset_from_line(&self, pos: Vec2, _tangent: Vec2) -> f32 {
        let Some((_, _, proj, tan)) = self.project(pos) else { return 0.0; };
        let delta = pos - proj;
        // Left-hand normal of the line tangent.
        let normal_left = Vec2::new(-tan.y, tan.x);
        let signed = delta.dot(normal_left);
        signed.clamp(-self.half_width, self.half_width)
    }

    /// Sample the ideal line at arc position `s` meters ahead of `pos`'s
    /// projection, and return the signed lateral deviation of the car from
    /// that look-ahead point (projected along the line). Positive = car is
    /// to the LEFT of where the ideal line will be in `s` meters.
    ///
    /// Practical use: anticipates corner turn-in while the car is still on
    /// straight, when instantaneous offset is zero.
    pub fn lookahead_offset(&self, pos: Vec2, s: f32) -> f32 {
        let Some((i, t, _proj, _tan)) = self.project(pos) else { return 0.0; };
        if self.arc.is_empty() { return 0.0; }
        // Current arc s0 = arc[i] + t * (arc[i+1]-arc[i]).
        let seg_len = (self.arc[i + 1] - self.arc[i]).max(1e-4);
        let s0 = self.arc[i] + t * seg_len;
        let target = (s0 + s).rem_euclid(self.total_len.max(1e-4));
        // Binary search / linear scan for target (small array, linear is fine).
        let mut j = 0usize;
        for k in 0..self.arc.len() - 1 {
            if self.arc[k] <= target && target <= self.arc[k + 1] {
                j = k;
                break;
            }
        }
        let seg_l = (self.arc[j + 1] - self.arc[j]).max(1e-4);
        let u = ((target - self.arc[j]) / seg_l).clamp(0.0, 1.0);
        let look_p = self.points[j] + (self.points[j + 1] - self.points[j]) * u;
        let tan = self.tangents[j];
        let normal_left = Vec2::new(-tan.y, tan.x);
        (pos - look_p).dot(normal_left).clamp(-self.half_width, self.half_width)
    }

    /// Return the world-space position on the racing line `s` meters ahead of `pos`.
    /// Returns `pos` if the projection fails or the line is empty.
    pub fn lookahead_point(&self, pos: Vec2, s: f32) -> Vec2 {
        let Some((i, t, _proj, _tan)) = self.project(pos) else { return pos; };
        if self.arc.is_empty() { return pos; }
        let seg_len = (self.arc[i + 1] - self.arc[i]).max(1e-4);
        let s0 = self.arc[i] + t * seg_len;
        let target = (s0 + s).rem_euclid(self.total_len.max(1e-4));
        let mut j = 0usize;
        for k in 0..self.arc.len() - 1 {
            if self.arc[k] <= target && target <= self.arc[k + 1] {
                j = k;
                break;
            }
        }
        let seg_l = (self.arc[j + 1] - self.arc[j]).max(1e-4);
        let u = ((target - self.arc[j]) / seg_l).clamp(0.0, 1.0);
        self.points[j] + (self.points[j + 1] - self.points[j]) * u
    }
}

/// Resample a polyline at uniform arc-length `step`. Returns at least the
/// first point of the input on degenerate cases.
fn resample_polyline(points: &[Vec2], step: f32) -> Vec<Vec2> {
    if points.len() < 2 || step <= 1e-3 {
        return points.to_vec();
    }
    let mut out = Vec::new();
    let mut acc = 0.0f32;
    out.push(points[0]);
    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let seg = b - a;
        let seg_len = seg.length();
        if seg_len < 1e-4 { continue; }
        let dir = seg / seg_len;
        let mut d = step - acc;
        while d < seg_len {
            out.push(a + dir * d);
            d += step;
        }
        acc = (acc + seg_len).rem_euclid(step);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_track() -> Vec<Vec2> {
        vec![
            Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0), Vec2::new(0.0, 100.0),
            Vec2::new(0.0, 0.0),
        ]
    }

    #[test]
    fn builds_non_empty_on_square() {
        let line = RacingLine::from_centerline(&square_track(), 18.0);
        assert!(!line.is_empty());
        assert!(line.total_len > 300.0); // ~400m square perimeter
    }

    #[test]
    fn offset_is_zero_at_ideal_point() {
        let line = RacingLine::from_centerline(&square_track(), 18.0);
        if line.points.is_empty() { return; }
        let p = line.points[line.points.len() / 2];
        let off = line.lateral_offset_from_line(p, Vec2::X);
        assert!(off.abs() < 0.01, "point ON line should have ~0 offset, got {}", off);
    }

    #[test]
    fn offset_sign_is_consistent() {
        let line = RacingLine::from_centerline(&square_track(), 18.0);
        if line.points.len() < 3 { return; }
        // Pick a point, move it along the LEFT normal of the tangent → expect positive offset.
        let i = line.points.len() / 4;
        let tan = line.tangents[i];
        let normal_left = Vec2::new(-tan.y, tan.x);
        let probe = line.points[i] + normal_left * 2.0;
        let off = line.lateral_offset_from_line(probe, tan);
        assert!(off > 0.0, "probe to the LEFT should be positive, got {}", off);
    }
}
