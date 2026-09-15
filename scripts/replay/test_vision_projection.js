#!/usr/bin/env node
/* CPU tests for the retina projection contract. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const sandbox = { window: {} };
vm.runInNewContext(fs.readFileSync('src/assets/replay_player/vision_projection.js', 'utf8'), sandbox);
const P = sandbox.window.ReplayVisionProjection;
const meta = { retina_cols: 126, fov_degrees: 120 };
const pose = { gaze_yaw: 0 };
const deg = r => r * 180 / Math.PI;

assert.equal(P.sourceColumn(meta, pose, 0, 0, 'raw'), 0);
assert.equal(P.sourceColumn(meta, pose, 1, 125, 'raw'), 125);
assert(Math.abs(deg(P.rayYaw(meta, pose, 1, 0)) + 60) < 1e-10);
assert(Math.abs(deg(P.rayYaw(meta, pose, 1, 125)) - 60) < 1e-10);
assert.equal(P.sourceColumn(meta, {}, 0, 62, 'angular'), null, 'missing gaze is explicit');

// A wall boundary at +/-30 degrees must reproject to the same source ray within one pixel.
for (const sceneDeg of [-30, 30]) {
  const target = Math.round((sceneDeg + 60) / 120 * 125);
  const source = P.sourceColumn(meta, pose, 0, target, 'angular');
  const sourceYaw = deg(P.rayYaw(meta, pose, 0, source));
  assert(Math.abs(sourceYaw - sceneDeg) < 1.1, `foveal boundary ${sceneDeg}: ${sourceYaw}`);
  const linearSource = P.sourceColumn(meta, pose, 1, target, 'angular');
  assert.equal(linearSource, target, 'non-foveal rows stay linear');
}

// A gaze shift changes only the foveal inverse mapping; no input array is touched.
const shifted = P.sourceColumn(meta, { gaze_yaw: 0.2 }, 0, 62, 'angular');
assert.notEqual(shifted, 62);
assert.equal(P.sourceColumn(meta, { gaze_yaw: 0.2 }, 1, 62, 'angular'), 62);
const immutablePose = { gaze_yaw: 0.2 };
const before = JSON.stringify(immutablePose);
P.sourceColumn(meta, immutablePose, 0, 62, 'angular');
assert.equal(JSON.stringify(immutablePose), before);
assert.equal(P.sourceColumn(meta, pose, 0, 0, 'angular'), 0);
assert.equal(P.sourceColumn(meta, pose, 0, 125, 'angular'), 125);
console.log('vision projection contract: PASS');
