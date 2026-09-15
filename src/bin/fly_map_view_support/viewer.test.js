"use strict";

const assert = require("node:assert/strict");
const viewer = require("./viewer.js");

function row(being, tick, x, z) {
  return {
    being,
    tick,
    alive: true,
    coverage: 0.5,
    asymmetry: -0.25,
    leftPixels: 10,
    rightPixels: 5,
    visibleDistance: 2.5,
    bodyDistance: 3.0,
    overlapCount: 0,
    rawSteer: -0.75,
    appliedSteer: -0.5,
    throttle: 0.8,
    brake: 0.1,
    speed: 4.0,
    x,
    z,
    forwardX: 1,
    forwardZ: 0,
  };
}

assert.equal(viewer.traceStepToTick(156, 156), 0);
assert.equal(viewer.traceStepToTick(10_140, 156), 64);
assert.throws(() => viewer.traceStepToTick(155, 156), /before the first world tick/);
assert.throws(() => viewer.traceStepToTick(157, 156), /not divisible/);

const telemetry = [
  row(0, 0, 620, 720),
  row(1, 0, 621, 719),
  row(0, 64, 610, 710),
  row(1, 64, 611, 709),
  row(0, 128, 600, 700),
  row(1, 128, 601, 699),
];
const trace = {
  being: 0,
  frames: [
    { step: 156, commands: [0, 0.4, 0], v: [-52], spikes: [0] },
    { step: 10_140, commands: [-0.2, 0.5, 0], v: [-44], spikes: [1] },
  ],
};

const traceTimeline = viewer.buildTimeline({ trace, telemetry, traceStepsPerTick: 156 }, 0);
assert.deepEqual(traceTimeline.map((sample) => sample.tick), [0, 64]);
assert.equal(traceTimeline.length, 2, "the timeline indexes trace frames, not physical ticks");

const telemetryTimeline = viewer.buildTimeline({ trace: null, telemetry }, 1);
assert.deepEqual(telemetryTimeline.map((sample) => sample.tick), [0, 64, 128]);

const snapshot = viewer.bodySnapshot(telemetry, 64, 0);
assert.deepEqual(snapshot.current.map((item) => item.being), [0, 1]);
assert.deepEqual(snapshot.trail.map((item) => item.tick), [0, 64]);
assert.equal(snapshot.trail.some((item) => item.tick === 128), false);

assert.deepEqual(viewer.measuredBounds(telemetry), {
  minX: 600,
  maxX: 621,
  minZ: 699,
  maxZ: 720,
});

const badReset = {
  trace: {
    being: 0,
    frames: [
      { step: 10_140 },
      { step: 156 },
    ],
  },
  telemetry,
  traceStepsPerTick: 156,
};
assert.throws(() => viewer.buildTimeline(badReset, 0), /strictly increasing/);

class FakeElement {
  constructor(tagName, id = "") {
    this.tagName = tagName;
    this.id = id;
    this.children = [];
    this.listeners = {};
    this.textContent = "";
    this.value = "";
    this.disabled = false;
    this.width = id === "flow" ? 900 : 1200;
    this.height = id === "telemetry-chart" ? 640 : 500;
  }

  appendChild(child) {
    this.children.push(child);
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }

  dispatch(name) {
    this.listeners[name]();
  }

  getContext() {
    const noOperation = () => {};
    return new Proxy({}, {
      get(target, key) {
        return key in target ? target[key] : noOperation;
      },
      set(target, key, value) {
        target[key] = value;
        return true;
      },
    });
  }
}

const elementIds = [
  "kpis", "groups", "eye", "flow", "slider", "being", "play",
  "telemetry-info", "trace-warning", "frame-info", "graph", "body",
  "telemetry-chart",
];
const elements = Object.fromEntries(elementIds.map((id) => [id, new FakeElement("div", id)]));
const document = {
  createElement: (tagName) => new FakeElement(tagName),
  getElementById: (id) => elements[id],
};
const mountedData = {
  n: 1,
  m: 1,
  nTypes: 1,
  eyeBound: 1,
  hasPixels: true,
  flowTypeIds: [0],
  typeFlow: [1],
  nodes: [[0, 0, 0, 0]],
  edges: [[0, 0, 1_065_353_216]],
  groups: [[0]],
  trace,
  telemetry,
  traceStepsPerTick: 156,
};
const mountedState = viewer.mount(document, mountedData);
assert.equal(elements.slider.max, "1", "the mounted range uses two trace samples");
assert.equal(mountedState.selectedBeing, 0, "the trace being is selected by default");
elements.being.value = "1";
elements.being.dispatch("change");
assert.match(elements["frame-info"].textContent, /activation unavailable for being 1/);

console.log("viewer functional tests passed");
