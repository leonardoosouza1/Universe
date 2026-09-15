#!/usr/bin/env node
/* CPU contract tests for the replay retina angle labels. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('src/assets/replay_player/study.js', 'utf8');
const sandbox = { window: {}, console, requestAnimationFrame() {}, setTimeout() {} };
vm.runInNewContext(source, sandbox, { filename: 'study.js' });
const angle = (sample, col, row) =>
  sandbox.window.ReplayStudy.prototype.retinaAngle.call({ sample }, {
    fov_degrees: 120,
    retina_cols: 126,
  }, col, row);

const left = angle({}, 0, 1);
const right = angle({}, 125, 1);
assert(Math.abs(left.degrees + 60) < 1e-10, `left endpoint: ${left.degrees}`);
assert(Math.abs(right.degrees - 60) < 1e-10, `right endpoint: ${right.degrees}`);

const centerColumn = 62;
const u = centerColumn / 125;
const expectedRow0 = Math.tan((2 * u - 1) * 1.25) / Math.tan(1.25) * 60;
const center = angle({}, centerColumn, 0);
assert(Math.abs(center.degrees - expectedRow0) < 1e-10, 'row 0 tangent mapping');

const gaze = 0.2;
const row0WithGaze = angle({ sensor_pose: { gaze_yaw: gaze } }, 62, 0);
assert(Math.abs(row0WithGaze.degrees - (expectedRow0 + gaze * 180 / Math.PI)) < 1e-10,
  'gaze offset applies to row 0');
const otherRowWithGaze = angle({ sensor_pose: { gaze_yaw: gaze } }, 62, 1);
assert(Math.abs(otherRowWithGaze.degrees - ((u - 0.5) * 120)) < 1e-10,
  'gaze offset does not apply to other rows');

const unknown = angle({}, 62, 0);
assert.equal(unknown.reference, 'gaze-relative angle (gaze unrecorded)');
assert.equal(row0WithGaze.reference, 'body-relative azimuth');
console.log('vision angle contract: PASS');
