"use strict";
const assert = require("node:assert/strict");
const model = require("./model.js");
const data = model.normalizeReplay([[0, 0], [100, 50]], [
  { id: 7, label: "Scout", frames: [{ t: 20, x: 2, y: 3 }, { t: 10, x: 1, y: 1 }] },
  { id: 9, frames: [{ t: 10, x: 6, y: 7 }] },
], { tick_rate: 50, track_width: 10 }, null);
assert.deepEqual(data.ticks, [10, 20]);
assert.equal(model.frameAtOrBefore(data.beings[0], 15).t, 10);
assert.equal(model.frameAtOrBefore(data.beings[0], 9), null);
assert.equal(model.beingLabel(data.beings[0], 0), "Scout");
assert.deepEqual(model.sceneBounds(data), { minX: -5, maxX: 105, minY: -5, maxY: 55 });
assert.deepEqual(model.sensorSegments({ x: 10, y: 20, fx: 1, fy: 0, cl: 3, cr: 4, cf: 7 }), [
  { name: "left", distance: 3, x: 10, y: 17 }, { name: "front", distance: 7, x: 17, y: 20 },
  { name: "right", distance: 4, x: 10, y: 24 },
]);
const brain = { traceStepsPerTick: 156, trace: { being: 7, frames: [
  { step: 156, spikes: [0] }, { step: 468, spikes: [2] },
] } };
assert.equal(model.traceTick(brain.trace.frames[1], 156), 2);
assert.equal(model.brainSample(brain, 7, 4).tick, 2);
assert.equal(model.brainSample(brain, 7, 4).age, 2);
assert.equal(model.brainSample(brain, 9, 4), null);
assert.equal(model.traceTick({ step: 157 }, 156), null);
assert.deepEqual(model.actionRows({ ithr: null, thr: .8, istr: -.2, str: .1, ibrk: .4, brk: 0 }), [
  { label: "Throttle", raw: null, applied: .8, signed: false },
  { label: "Steer", raw: -.2, applied: .1, signed: true }, { label: "Brake", raw: .4, applied: 0, signed: false },
]);
assert.equal(model.formatTime(75, 50), "00:01.500");
console.log("replay model tests passed");
const verified = { schema_version: 1, beings: [{ id: 7, reason: null, lap_completion_ticks: [100, 200] }] };
assert.equal(model.completedLaps(verified, 7, 99), 0);
assert.equal(model.completedLaps(verified, 7, 100), 1);
assert.equal(model.completedLaps(verified, 7, 199), 1);
assert.equal(model.completedLaps(verified, 7, 250), 2);
assert.equal(model.completedLaps(null, 7, 250), null);
assert.equal(model.completedLaps(verified, 9, 250), null);
assert.equal(model.completedLaps({ schema_version: 1, beings: [{ id: 7, reason: 'implausible_jump', lap_completion_ticks: [100] }] }, 7, 250), null);
assert.equal(model.controllerLabel({ brain_model: 'native', controller_kind: 'diagnostic_pilot' }), 'Diagnostic pilot');
assert.equal(model.controllerLabel({ brain_model: 'fly_spiking' }), 'Fly spiking · experimental');
assert.equal(model.motionSummary(model.normalizeReplay([], [{ frames: [{ x: 0, y: 0 }, { x: 3, y: 4 }, { x: 0, y: 0 }] }], {}, null))[0].distance, 10);
