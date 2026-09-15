(function initializeFlyMapViewModel(root, factory) {
  "use strict";

  const model = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = model;
  }
  root.FlyMapViewModel = model;
})(typeof window === "undefined" ? globalThis : window, function createModel() {
  "use strict";

  function finite(value) {
    return Number.isFinite(value);
  }

  function traceStepToTick(step, stepsPerTick) {
    if (!Number.isSafeInteger(step) || !Number.isSafeInteger(stepsPerTick) || stepsPerTick < 1) {
      throw new Error("trace step and steps-per-tick must be positive safe integers");
    }
    if (step < stepsPerTick) {
      throw new Error("trace step is before the first world tick");
    }
    if (step % stepsPerTick !== 0) {
      throw new Error("trace step is not divisible by steps-per-tick");
    }
    return step / stepsPerTick - 1;
  }

  function rowsForBeing(telemetry, being) {
    return telemetry
      .filter((row) => row.being === being)
      .slice()
      .sort((left, right) => left.tick - right.tick);
  }

  function buildTimeline(data, being) {
    const traceFrames = data.trace?.frames || [];
    if (traceFrames.length > 0) {
      let previousTick = -1;
      return traceFrames.map((frame, frameIndex) => {
        const tick = traceStepToTick(frame.step, data.traceStepsPerTick);
        if (tick <= previousTick) {
          throw new Error("trace ticks must be strictly increasing; a reset or duplicate was found");
        }
        previousTick = tick;
        return { tick, frameIndex, rowIndex: null };
      });
    }
    return rowsForBeing(data.telemetry || [], being).map((row, rowIndex) => ({
      tick: row.tick,
      frameIndex: null,
      rowIndex,
    }));
  }

  function rowAtTick(rows, tick) {
    return rows.find((row) => row.tick === tick) || null;
  }

  function bodySnapshot(telemetry, tick, selectedBeing) {
    const current = [];
    const seen = new Set();
    for (const row of telemetry) {
      if (row.tick === tick && !seen.has(row.being)) {
        current.push(row);
        seen.add(row.being);
      }
    }
    const trail = rowsForBeing(telemetry, selectedBeing).filter((row) => row.tick <= tick);
    return { current, trail };
  }

  function measuredBounds(telemetry) {
    const positioned = telemetry.filter((row) => finite(row.x) && finite(row.z));
    if (positioned.length === 0) {
      return null;
    }
    return {
      minX: Math.min(...positioned.map((row) => row.x)),
      maxX: Math.max(...positioned.map((row) => row.x)),
      minZ: Math.min(...positioned.map((row) => row.z)),
      maxZ: Math.max(...positioned.map((row) => row.z)),
    };
  }

  function equalAspectProjector(bounds, width, height, margin) {
    const spanX = Math.max(bounds.maxX - bounds.minX, 1);
    const spanZ = Math.max(bounds.maxZ - bounds.minZ, 1);
    const scale = Math.min((width - margin * 2) / spanX, (height - margin * 2) / spanZ);
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerZ = (bounds.minZ + bounds.maxZ) / 2;
    return {
      x: (value) => width / 2 + (value - centerX) * scale,
      z: (value) => height / 2 - (value - centerZ) * scale,
      scale,
    };
  }

  function floatFromBits(bits) {
    return new Float32Array(new Uint32Array([bits]).buffer)[0];
  }

  function createBrainLayout(data, canvas) {
    const typeIds = data.flowTypeIds || [];
    const hasOther = data.nTypes > typeIds.length;
    const columns = Math.max(typeIds.length + (hasOther ? 1 : 0), 1);
    const positions = new Map();
    for (const node of data.nodes) {
      const found = typeIds.indexOf(node[1]);
      const column = found >= 0 ? found : columns - 1;
      const hash = (node[0] * 2_654_435_761) >>> 0;
      positions.set(node[0], {
        x: 52 + (column + 0.5) * ((canvas.width - 84) / columns),
        y: 32 + (hash / 4_294_967_296) * (canvas.height - 52),
        column,
      });
    }
    return { positions, typeIds, columns };
  }

  function bodyScene(data, sample, selectedBeing) {
    if (!sample) {
      return null;
    }
    const snapshot = bodySnapshot(data.telemetry, sample.tick, selectedBeing);
    const bounds = measuredBounds([...snapshot.current, ...snapshot.trail]);
    if (!bounds) {
      return null;
    }
    return {
      bounds,
      current: snapshot.current,
      trail: snapshot.trail.filter((row) => finite(row.x) && finite(row.z)),
    };
  }

  function chartPanels(rows) {
    const speedMaximum = Math.max(1, ...rows.map((row) => finite(row.speed) ? row.speed : 0));
    return [
      [{ title: "Vision", minimum: -1, maximum: 1, unit: "ratio" }, [
        { key: "coverage", label: "coverage", color: "#58a6ff", legendX: 0 },
        { key: "asymmetry", label: "asymmetry", color: "#ffa657", legendX: 220 },
      ]],
      [{ title: "Steering", minimum: -1, maximum: 1, unit: "command" }, [
        { key: "rawSteer", label: "raw", color: "#bc8cff", legendX: 0 },
        { key: "appliedSteer", label: "applied", color: "#f2cc60", legendX: 190 },
      ]],
      [{ title: "Pedals", minimum: 0, maximum: 1, unit: "command" }, [
        { key: "throttle", label: "throttle", color: "#3fb950", legendX: 0 },
        { key: "brake", label: "brake", color: "#f85149", legendX: 210 },
      ]],
      [{ title: "Speed", minimum: 0, maximum: Number(speedMaximum.toFixed(2)), unit: "m/s" }, [
        { key: "speed", label: "speed", color: "#79c0ff", legendX: 0, digits: 2 },
      ]],
    ];
  }

  function availableBeings(data) {
    const beings = [...new Set((data.telemetry || []).map((row) => row.being))]
      .sort((left, right) => left - right);
    if (data.trace && !beings.includes(data.trace.being)) {
      beings.push(data.trace.being);
      beings.sort((left, right) => left - right);
    }
    return beings;
  }

  function telemetryStatus(data, selectedBeing, sample, row) {
    if (row) {
      return `tick ${row.tick} · ${row.alive ? "alive" : "dead"}`
        + ` · visible ${row.visibleDistance.toFixed(2)} m`
        + ` · body ${row.bodyDistance.toFixed(2)} m`
        + ` · retinal pixels ${row.leftPixels.toFixed(0)} left / ${row.rightPixels.toFixed(0)} right`
        + ` · overlap ${row.overlapCount}`;
    }
    if (data.telemetry.length === 0) {
      return "Body, sensor, and action telemetry unavailable; pass --telemetry <file.csv>.";
    }
    return sample
      ? `No measured telemetry row for being ${selectedBeing} at tick ${sample.tick}.`
      : `No measured telemetry rows for being ${selectedBeing}.`;
  }

  function staticDetails(data) {
    const percent = 100 * data.eyeBound / Math.max(data.n, 1);
    return {
      kpis: [
        [data.n, "neurons"], [data.m, "edges"], [data.nTypes, "cell types"],
        [data.eyeBound, "retina-bound neurons"], [data.nodes.length, "sampled neurons"],
        [data.edges.length, "drawn sampled edges"],
      ],
      groups: data.groups.map((members, index) => [
        `motor group ${index}`,
        `${members.length.toLocaleString()} neurons`,
      ]),
      eye: data.hasPixels
        ? `${data.eyeBound.toLocaleString()} / ${data.n.toLocaleString()} neurons `
          + `(${percent.toFixed(2)}%) have measured pixel bindings.`
        : "Pixel-binding data is unavailable in this map.",
    };
  }

  return {
    availableBeings,
    bodyScene,
    bodySnapshot,
    buildTimeline,
    chartPanels,
    createBrainLayout,
    equalAspectProjector,
    finite,
    floatFromBits,
    measuredBounds,
    rowAtTick,
    rowsForBeing,
    staticDetails,
    telemetryStatus,
    traceStepToTick,
  };
});
