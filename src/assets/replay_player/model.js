(function installReplayModel(root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.UniverseReplayModel = api;
})(typeof window === "undefined" ? globalThis : window, function createReplayModel() {
  "use strict";
  const COLORS = ["#73e0b3", "#7db8e5", "#f0bd6b", "#d68be8", "#ef776f", "#9bc86b", "#6fd3d8", "#e69b70"];
  const finite = (value) => Number.isFinite(value);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function beingId(being, index) { return being.id ?? being.being ?? index; }
  function beingLabel(being, index) {
    const id = beingId(being, index);
    return String(being.label || being.name || `Being ${id}`);
  }
  function allTicks(beings) {
    const ticks = new Set();
    for (const being of beings) for (const frame of being.frames || []) if (finite(frame.t)) ticks.add(frame.t);
    return [...ticks].sort((a, b) => a - b);
  }
  function frameAtOrBefore(being, tick) {
    const frames = being?.frames || [];
    let low = 0, high = frames.length - 1, found = null;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (frames[mid].t <= tick) { found = frames[mid]; low = mid + 1; } else high = mid - 1;
    }
    return found;
  }
  function normalizeReplay(track, beings, meta, brain) {
    const safeBeings = Array.isArray(beings) ? beings.filter(Boolean).map((being) => ({
      ...being, frames: Array.isArray(being.frames) ? being.frames.slice().sort((a, b) => a.t - b.t) : [],
    })) : [];
    return {
      track: Array.isArray(track) ? track.filter((p) => Array.isArray(p) && finite(p[0]) && finite(p[1])) : [],
      beings: safeBeings, meta: meta && typeof meta === "object" ? meta : {},
      brain: brain && typeof brain === "object" ? brain : null, ticks: allTicks(safeBeings),
    };
  }
  function sceneBounds(data) {
    const points = data.track.length ? data.track : data.beings.flatMap((being) => being.frames
      .map((frame) => [frame.x, frame.y]).filter((p) => finite(p[0]) && finite(p[1])));
    if (!points.length) return null;
    const xs = points.map((point) => point[0]), ys = points.map((point) => point[1]);
    const trackWidth = finite(data.meta.track_width) ? Math.max(data.meta.track_width, 0) : 0;
    return { minX: Math.min(...xs) - trackWidth / 2, maxX: Math.max(...xs) + trackWidth / 2,
      minY: Math.min(...ys) - trackWidth / 2, maxY: Math.max(...ys) + trackWidth / 2 };
  }
  function fitCamera(bounds, width, height, margin = 38) {
    if (!bounds || width <= 0 || height <= 0) return { x: 0, y: 0, scale: 1 };
    const spanX = Math.max(bounds.maxX - bounds.minX, 1), spanY = Math.max(bounds.maxY - bounds.minY, 1);
    const scale = Math.max(.001, Math.min((width - margin * 2) / spanX, (height - margin * 2) / spanY));
    return { x: width / 2 - ((bounds.minX + bounds.maxX) / 2) * scale,
      y: height / 2 + ((bounds.minY + bounds.maxY) / 2) * scale, scale };
  }
  function project(camera, x, y) { return { x: camera.x + x * camera.scale, y: camera.y - y * camera.scale }; }
  function sensorSegments(frame) {
    if (!frame || !finite(frame.x) || !finite(frame.y) || !finite(frame.fx) || !finite(frame.fy)) return [];
    const length = Math.hypot(frame.fx, frame.fy); if (length < 1e-6) return [];
    const fx = frame.fx / length, fy = frame.fy / length;
    const probes = [["left", frame.cl, frame.x + fy * frame.cl, frame.y - fx * frame.cl],
      ["front", frame.cf, frame.x + fx * frame.cf, frame.y + fy * frame.cf],
      ["right", frame.cr, frame.x - fy * frame.cr, frame.y + fx * frame.cr]];
    return probes.filter((probe) => finite(probe[1]) && probe[1] >= 0)
      .map(([name, distance, x, y]) => ({ name, distance, x, y }));
  }
  function actionRows(frame) {
    return [{ label: "Throttle", raw: frame?.ithr ?? null, applied: frame?.thr ?? null, signed: false },
      { label: "Steer", raw: frame?.istr ?? null, applied: frame?.str ?? null, signed: true },
      { label: "Brake", raw: frame?.ibrk ?? null, applied: frame?.brk ?? null, signed: false }];
  }
  function traceTick(frame, stepsPerTick) {
    if (!Number.isSafeInteger(frame?.step) || !Number.isSafeInteger(stepsPerTick) || stepsPerTick < 1) return null;
    if (frame.step < stepsPerTick || frame.step % stepsPerTick !== 0) return null;
    return frame.step / stepsPerTick - 1;
  }
  function brainSample(brain, being, tick) {
    const trace = brain?.trace;
    if (!trace || String(trace.being) !== String(being) || !Array.isArray(trace.frames)) return null;
    let match = null, matchTick = null;
    for (const frame of trace.frames) {
      const sampleTick = traceTick(frame, brain.traceStepsPerTick);
      if (sampleTick != null && sampleTick <= tick && (matchTick == null || sampleTick > matchTick)) {
        match = frame; matchTick = sampleTick;
      }
    }
    return match ? { frame: match, tick: matchTick, age: tick - matchTick } : null;
  }
  function nodeFields(node) {
    return Array.isArray(node) ? { id: node[0], type: node[1], pixel: node[2], group: node[3] }
      : { id: node.id, type: node.type, pixel: node.pixel, group: node.group };
  }
  function brainLayout(brain, width, height) {
    const nodes = (brain?.nodes || []).map(nodeFields).filter((node) => Number.isInteger(node.id));
    const featured = Array.isArray(brain?.flowTypeIds) ? brain.flowTypeIds : [];
    const types = featured.length ? [...featured, "other"] : [...new Set(nodes.map((node) => node.type))].sort((a, b) => a - b);
    const positions = new Map(), columns = new Map(types.map((type, index) => [type, index]));
    for (const node of nodes) {
      const column = columns.get(node.type) ?? types.length - 1;
      const hash = (Number(node.id) * 2654435761) >>> 0;
      positions.set(node.id, { x: 22 + (column + .5) * ((width - 44) / Math.max(types.length, 1)),
        y: 15 + (hash / 4294967296) * (height - 30) });
    }
    return { nodes, positions };
  }
  function frameDuration(data, index) {
    if (index >= data.ticks.length - 1) return 1000 / Math.max(data.meta.tick_rate || 64, 1);
    return Math.max(1, (data.ticks[index + 1] - data.ticks[index]) * 1000 / Math.max(data.meta.tick_rate || 64, 1));
  }
  function formatTime(tick, tickRate) {
    const millis = Math.max(0, tick) * 1000 / Math.max(tickRate || 64, 1);
    const minutes = Math.floor(millis / 60000), seconds = Math.floor(millis / 1000) % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(Math.floor(millis % 1000)).padStart(3, "0")}`;
  }
  function controllerLabel(meta) {
    if (meta.controller_kind === "diagnostic_pilot") return "Diagnostic pilot";
    const labels = { native: "Native neural brain", fly_spiking: "Fly spiking · experimental", fly_rate: "Fly rate · experimental" };
    return labels[meta.brain_model] || meta.source || "Controller not recorded";
  }
  function completedLaps(validation, id, tick) {
    if (validation?.schema_version !== 1) return null;
    const result = validation.beings?.find((being) => String(being.id ?? being.being) === String(id));
    if (!result || !Array.isArray(result.lap_completion_ticks)) return null;
    if (result.reason && !["stationary", "insufficient_sequential_progress", "no_forward_progress"].includes(result.reason)) return null;
    return result.lap_completion_ticks.filter((end) => finite(end) && end <= tick).length;
  }
  function motionSummary(data) {
    return data.beings.map((being, index) => {
      let distance = 0;
      for (let i = 1; i < being.frames.length; i++) {
        const a = being.frames[i - 1], b = being.frames[i];
        if ([a.x, a.y, b.x, b.y].every(finite)) distance += Math.hypot(b.x - a.x, b.y - a.y);
      }
      return { id: beingId(being, index), distance };
    });
  }
  return { COLORS, actionRows, beingId, beingLabel, brainLayout, brainSample, clamp, finite, fitCamera,
    formatTime, frameAtOrBefore, frameDuration, normalizeReplay, project, sceneBounds, sensorSegments, traceTick,
    controllerLabel, completedLaps, motionSummary };
});
