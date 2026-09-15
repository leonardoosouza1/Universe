/* Pure coordinate mapping for inspecting the packed retina without inventing rays. */
(function (root) {
  'use strict';
  const TAN_K = 1.25;

  function dimensions(meta) {
    const cols = Number(meta?.retina_cols);
    const fov = Number(meta?.fov_degrees);
    return Number.isInteger(cols) && cols > 1 && Number.isFinite(fov) && fov > 0
      ? { cols, halfFov: fov * Math.PI / 360 } : null;
  }

  function rayYaw(meta, pose, row, col) {
    const d = dimensions(meta);
    if (!d || !Number.isInteger(row) || row < 0 || !Number.isInteger(col) || col < 0 || col >= d.cols) return null;
    const normalized = col / (d.cols - 1) * 2 - 1;
    const gaze = row === 0 && Number.isFinite(pose?.gaze_yaw) ? pose.gaze_yaw : 0;
    return (row === 0 ? Math.tan(normalized * TAN_K) / Math.tan(TAN_K) * d.halfFov : normalized * d.halfFov) + gaze;
  }

  function sourceColumn(meta, pose, row, displayCol, mode) {
    const d = dimensions(meta);
    if (!d || !Number.isInteger(row) || row < 0 || !Number.isInteger(displayCol) || displayCol < 0 || displayCol >= d.cols) return null;
    if (mode === 'raw') return displayCol;
    if (mode !== 'angular') return null;
    if (row > 0) return displayCol;
    const gaze = pose?.gaze_yaw;
    if (!Number.isFinite(gaze)) return null;
    const target = (displayCol / (d.cols - 1) * 2 - 1) * d.halfFov;
    const local = target - gaze;
    if (Math.abs(local) > d.halfFov + 1e-12) return null;
    const normalized = Math.atan(local / d.halfFov * Math.tan(TAN_K)) / TAN_K;
    return Math.round((normalized + 1) * 0.5 * (d.cols - 1));
  }

  root.ReplayVisionProjection = Object.freeze({ sourceColumn, rayYaw });
})(window);
