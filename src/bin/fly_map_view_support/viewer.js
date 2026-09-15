(function initializeFlyMapViewer(root, factory) {
  "use strict";
  const commonJs = typeof module === "object" && module.exports;
  const model = commonJs ? require("./model.js") : root.FlyMapViewModel;
  const viewer = factory(model);
  if (commonJs) {
    module.exports = viewer;
  }
  if (root.document && root.FLY_MAP_DATA) {
    const start = () => viewer.mount(root.document, root.FLY_MAP_DATA);
    if (root.document.readyState === "loading") {
      root.document.addEventListener("DOMContentLoaded", start, { once: true });
    } else {
      start();
    }
  }
})(typeof window === "undefined" ? globalThis : window, function createViewer(model) {
  "use strict";
  const {
    availableBeings, bodyScene, bodySnapshot, buildTimeline,
    chartPanels, createBrainLayout, equalAspectProjector, finite,
    floatFromBits, measuredBounds, rowAtTick, rowsForBeing,
    staticDetails, telemetryStatus, traceStepToTick,
  } = model;
  const COLORS = [
    "#58a6ff", "#f2cc60", "#3fb950", "#f85149",
    "#bc8cff", "#ffa657", "#79c0ff", "#d2a8ff",
  ];
  function canvasContext(document, id) {
    const canvas = document.getElementById(id);
    return { canvas, context: canvas.getContext("2d") };
  }
  function clearCanvas(canvas, context) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#090d12";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.font = "12px system-ui";
  }
  function drawUnavailable(canvas, context, message) {
    clearCanvas(canvas, context);
    context.fillStyle = "#8b949e";
    context.textAlign = "center";
    context.fillText(message, canvas.width / 2, canvas.height / 2);
    context.textAlign = "start";
  }
  function drawFlow(document, data) {
    const { canvas, context } = canvasContext(document, "flow");
    clearCanvas(canvas, context);
    const ids = data.flowTypeIds || [];
    if (ids.length === 0) {
      drawUnavailable(canvas, context, "No cell-type flow is available.");
      return;
    }
    const left = 52;
    const top = 18;
    const plotWidth = canvas.width - left - 20;
    const plotHeight = canvas.height - 68;
    const cellWidth = plotWidth / ids.length;
    const cellHeight = plotHeight / ids.length;
    const maximum = Math.max(...data.typeFlow, 1);
    for (let source = 0; source < ids.length; source += 1) {
      for (let target = 0; target < ids.length; target += 1) {
        const count = data.typeFlow[source * ids.length + target];
        const intensity = count ? Math.log1p(count) / Math.log1p(maximum) : 0;
        context.fillStyle = `rgba(88,166,255,${0.03 + intensity * 0.97})`;
        context.fillRect(
          left + target * cellWidth,
          top + source * cellHeight,
          cellWidth - 0.5,
          cellHeight - 0.5,
        );
      }
    }
    context.fillStyle = "#8b949e";
    context.font = "9px system-ui";
    const labelEvery = Math.max(1, Math.ceil(ids.length / 16));
    for (let index = 0; index < ids.length; index += 1) {
      if (index % labelEvery !== 0) {
        continue;
      }
      context.fillText(`T${ids[index]}`, 3, top + (index + 0.7) * cellHeight);
      context.save();
      context.translate(left + (index + 0.55) * cellWidth, top + plotHeight + 8);
      context.rotate(-Math.PI / 2);
      context.fillText(`T${ids[index]}`, 0, 0);
      context.restore();
    }
  }
  function drawBrain(document, data, state) {
    const { canvas, context } = canvasContext(document, "graph");
    clearCanvas(canvas, context);
    const layout = state.brainLayout;
    const sample = state.timeline[state.sampleIndex] || null;
    const traceFrame = sample?.frameIndex == null
      ? null
      : data.trace?.frames[sample.frameIndex] || null;
    const activationAvailable = Boolean(traceFrame && state.selectedBeing === data.trace.being);
    const labelEvery = Math.max(1, Math.ceil(layout.columns / 18));
    for (let column = 0; column < layout.columns; column += 1) {
      if (column % labelEvery !== 0 && column !== layout.columns - 1) {
        continue;
      }
      const x = 52 + (column + 0.5) * ((canvas.width - 84) / layout.columns);
      context.fillStyle = COLORS[column % COLORS.length];
      const label = column < layout.typeIds.length ? `T${layout.typeIds[column]}` : "other";
      context.fillText(label, x - 10, 15);
    }
    for (const edge of data.edges) {
      const source = layout.positions.get(edge[0]);
      const target = layout.positions.get(edge[1]);
      if (!source || !target) {
        continue;
      }
      const weight = floatFromBits(edge[2]);
      const alpha = Math.min(0.48, 0.035 + Math.abs(weight) / 20);
      context.strokeStyle = weight >= 0
        ? `rgba(63,185,80,${alpha})`
        : `rgba(248,81,73,${alpha})`;
      context.lineWidth = Math.min(1.8, 0.35 + Math.abs(weight) / 100);
      context.beginPath();
      context.moveTo(source.x, source.y);
      context.lineTo(target.x, target.y);
      context.stroke();
    }
    let observed = 0;
    let active = 0;
    let maximumVoltage = -Infinity;
    for (const node of data.nodes) {
      const position = layout.positions.get(node[0]);
      const voltage = activationAvailable ? traceFrame.v?.[node[0]] : undefined;
      const spikes = activationAvailable ? traceFrame.spikes?.[node[0]] : undefined;
      const voltageAvailable = finite(voltage);
      if (voltageAvailable) {
        observed += 1;
        maximumVoltage = Math.max(maximumVoltage, voltage);
      }
      if (finite(spikes) && spikes > 0) {
        active += 1;
      }
      context.fillStyle = voltageAvailable
        ? (voltage > -45 ? "#f2cc60" : voltage > -52 ? "#3fb950" : "#264b73")
        : COLORS[position.column % COLORS.length];
      context.globalAlpha = voltageAvailable ? (spikes > 0 ? 1 : 0.68) : 0.58;
      context.beginPath();
      context.arc(position.x, position.y, spikes > 0 ? 4 : 2.1, 0, Math.PI * 2);
      context.fill();
    }
    context.globalAlpha = 1;
    const info = document.getElementById("frame-info");
    if (!data.trace || !traceFrame) {
      info.textContent = `sample ${state.sampleIndex + 1}/${state.timeline.length || 0}`
        + " · brain activation unavailable (no trace)";
    } else if (!activationAvailable) {
      info.textContent = `tick ${sample.tick} · brain activation unavailable for being ${state.selectedBeing}`
        + ` (trace contains being ${data.trace.being})`;
    } else {
      const voltageText = observed ? `${maximumVoltage.toFixed(2)} mV` : "unavailable";
      const commands = Array.isArray(traceFrame.commands)
        ? traceFrame.commands.map((value) => Number(value).toFixed(3)).join(" / ")
        : "unavailable";
      info.textContent = `sample ${state.sampleIndex + 1}/${state.timeline.length}`
        + ` · tick ${sample.tick} · LIF step ${traceFrame.step}`
        + ` · ${active} sampled-node spikes · max V ${voltageText}`
        + ` · motor ${commands}`;
    }
  }
  function drawArrow(context, projector, row, color, length) {
    if (![row.x, row.z, row.forwardX, row.forwardZ].every(finite)) {
      return;
    }
    const x = projector.x(row.x);
    const z = projector.z(row.z);
    context.strokeStyle = color;
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(x, z);
    context.lineTo(projector.x(row.x + row.forwardX * length),
      projector.z(row.z + row.forwardZ * length));
    context.stroke();
  }
  function drawBody(document, data, state) {
    const { canvas, context } = canvasContext(document, "body");
    const sample = state.timeline[state.sampleIndex] || null;
    const scene = bodyScene(data, sample, state.selectedBeing);
    if (!scene) {
      drawUnavailable(canvas, context, "Measured body positions are unavailable.");
      return;
    }
    clearCanvas(canvas, context);
    const projector = equalAspectProjector(scene.bounds, canvas.width, canvas.height, 34);
    if (scene.trail.length > 1) {
      context.strokeStyle = "rgba(242,204,96,0.7)";
      context.lineWidth = 2;
      context.beginPath();
      scene.trail.forEach((row, index) => {
        const x = projector.x(row.x);
        const z = projector.z(row.z);
        index === 0 ? context.moveTo(x, z) : context.lineTo(x, z);
      });
      context.stroke();
    }
    for (const row of scene.current) {
      if (!finite(row.x) || !finite(row.z)) {
        continue;
      }
      const selected = row.being === state.selectedBeing;
      const color = selected ? "#f2cc60" : "#58a6ff";
      context.fillStyle = color;
      context.beginPath();
      context.arc(projector.x(row.x), projector.z(row.z), selected ? 5 : 3.2, 0, Math.PI * 2);
      context.fill();
      drawArrow(context, projector, row, color, Math.max(2, 10 / projector.scale));
    }
    context.fillStyle = "#8b949e";
    context.fillText("equal x/z scale · measured positions and forward vectors", 10, 18);
  }
  function drawPanel(context, panel, rows, currentIndex, specs) {
    const left = 92;
    const top = panel.y + 23;
    const bottom = panel.y + panel.height - 20;
    const plotWidth = panel.width - left - 18;
    const plotHeight = bottom - top;
    context.fillStyle = "#10161f";
    context.fillRect(left, top, plotWidth, plotHeight);
    context.strokeStyle = "#30363d";
    context.strokeRect(left, top, plotWidth, plotHeight);
    context.fillStyle = "#c9d1d9";
    context.fillText(panel.title, 10, panel.y + 15);
    context.fillStyle = "#8b949e";
    context.fillText(`${panel.maximum} ${panel.unit}`, 10, top + 8);
    context.fillText(`${panel.minimum} ${panel.unit}`, 10, bottom);
    const xAt = (index) => left + index * plotWidth / Math.max(rows.length - 1, 1);
    const yAt = (value) => bottom
      - (value - panel.minimum) * plotHeight / Math.max(panel.maximum - panel.minimum, 1e-9);
    if (panel.minimum < 0 && panel.maximum > 0) {
      context.strokeStyle = "#3a424d";
      context.beginPath();
      context.moveTo(left, yAt(0));
      context.lineTo(left + plotWidth, yAt(0));
      context.stroke();
    }
    for (const spec of specs) {
      context.strokeStyle = spec.color;
      context.lineWidth = 1.7;
      context.beginPath();
      let started = false;
      rows.forEach((row, index) => {
        const value = row[spec.key];
        if (!finite(value)) {
          return;
        }
        const x = xAt(index);
        const y = yAt(value);
        started ? context.lineTo(x, y) : context.moveTo(x, y);
        started = true;
      });
      context.stroke();
      const current = rows[currentIndex]?.[spec.key];
      context.fillStyle = spec.color;
      const valueText = finite(current) ? Number(current).toFixed(spec.digits ?? 3) : "unavailable";
      context.fillText(`${spec.label}: ${valueText} ${panel.unit}`,
        left + spec.legendX, panel.y + 15);
    }
    context.strokeStyle = "rgba(242,204,96,0.7)";
    context.beginPath();
    context.moveTo(xAt(currentIndex), top);
    context.lineTo(xAt(currentIndex), bottom);
    context.stroke();
  }
  function drawCharts(document, data, state) {
    const { canvas, context } = canvasContext(document, "telemetry-chart");
    const sample = state.timeline[state.sampleIndex] || null;
    const rows = rowsForBeing(data.telemetry, state.selectedBeing);
    const currentIndex = sample ? rows.findIndex((row) => row.tick === sample.tick) : -1;
    if (rows.length === 0 || currentIndex < 0) {
      drawUnavailable(canvas, context, "Measured sensor and action series are unavailable at this sample.");
      return;
    }
    clearCanvas(canvas, context);
    const panelHeight = canvas.height / 4;
    chartPanels(rows).forEach(([panel, specs], index) => {
      drawPanel(context, {
        ...panel,
        y: panelHeight * index,
        width: canvas.width,
        height: panelHeight,
      }, rows, currentIndex, specs);
    });
  }
  function renderStaticDetails(document, data) {
    const details = staticDetails(data);
    const container = document.getElementById("kpis");
    for (const [value, label] of details.kpis) {
      const card = document.createElement("div");
      card.className = "card kpi";
      card.textContent = Number(value).toLocaleString();
      const caption = document.createElement("small");
      caption.textContent = label;
      card.appendChild(caption);
      container.appendChild(card);
    }
    const groups = document.getElementById("groups");
    details.groups.forEach(([label, value]) => {
      const row = document.createElement("tr");
      const name = document.createElement("td");
      const count = document.createElement("td");
      name.textContent = label;
      count.textContent = value;
      row.appendChild(name);
      row.appendChild(count);
      groups.appendChild(row);
    });
    document.getElementById("eye").textContent = details.eye;
  }
  function mount(document, data) {
    renderStaticDetails(document, data);
    drawFlow(document, data);
    const slider = document.getElementById("slider");
    const select = document.getElementById("being");
    const play = document.getElementById("play");
    const status = document.getElementById("telemetry-info");
    const beings = availableBeings(data);
    for (const being of beings) {
      const option = document.createElement("option");
      option.value = being;
      option.textContent = `being ${being}`;
      select.appendChild(option);
    }
    select.disabled = beings.length === 0;
    const defaultBeing = data.trace && beings.includes(data.trace.being)
      ? data.trace.being
      : beings[0] ?? 0;
    select.value = String(defaultBeing);
    const state = {
      selectedBeing: defaultBeing,
      sampleIndex: 0,
      timeline: [],
      brainLayout: createBrainLayout(data, document.getElementById("graph")),
      timer: null,
    };
    function refreshTimeline(reset) {
      state.timeline = buildTimeline(data, state.selectedBeing);
      state.sampleIndex = reset
        ? 0
        : Math.min(state.sampleIndex, Math.max(state.timeline.length - 1, 0));
      slider.max = String(Math.max(state.timeline.length - 1, 0));
      slider.value = String(state.sampleIndex);
      slider.disabled = state.timeline.length <= 1;
      play.disabled = state.timeline.length <= 1;
    }
    function drawAll() {
      const sample = state.timeline[state.sampleIndex] || null;
      const rows = rowsForBeing(data.telemetry || [], state.selectedBeing);
      const row = sample ? rowAtTick(rows, sample.tick) : null;
      document.getElementById("trace-warning").textContent = data.trace
        && state.selectedBeing !== data.trace.being
        ? `Brain activation is unavailable: this trace contains being ${data.trace.being}.`
        : "";
      status.textContent = telemetryStatus(data, state.selectedBeing, sample, row);
      drawBrain(document, data, state);
      drawBody(document, data, state);
      drawCharts(document, data, state);
    }
    slider.addEventListener("input", () => {
      state.sampleIndex = Number(slider.value);
      drawAll();
    });
    select.addEventListener("change", () => {
      state.selectedBeing = Number(select.value);
      refreshTimeline(true);
      drawAll();
    });
    play.addEventListener("click", () => {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
        play.textContent = "Play";
        return;
      }
      play.textContent = "Pause";
      state.timer = setInterval(() => {
        state.sampleIndex = (state.sampleIndex + 1) % state.timeline.length;
        slider.value = String(state.sampleIndex);
        drawAll();
      }, 250);
    });
    refreshTimeline(true);
    drawAll();
    return state;
  }
  return {
    bodySnapshot,
    buildTimeline,
    equalAspectProjector,
    measuredBounds,
    mount,
    traceStepToTick,
  };
});
