(function startReplay(root) {
  "use strict";
  const M = root.UniverseReplayModel;
  const data = M.normalizeReplay(root.REPLAY_TRACK || [], root.REPLAY_BEINGS || [], root.REPLAY_META || {}, root.REPLAY_BRAIN || null);
  const validation = root.REPLAY_VALIDATION || null;
  const $ = (id) => document.getElementById(id);
  const state = { index: 0, selected: 0, playing: false, trail: true, sensors: true, timer: null,
    camera: null, drag: null, moved: false, hits: [], following: false, playOriginTime: 0, playOriginTick: 0 };
  const scene = $("scene"), sceneContext = scene.getContext("2d");
  const study = new root.ReplayStudy(data, state, selectedId, tick);
  const canvasSizes = new WeakMap();

  function measureCanvas(canvas) {
    const ratio = Math.min(devicePixelRatio || 1, 2), rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const size = { width: rect.width, height: rect.height, ratio };
    canvasSizes.set(canvas, size); return size;
  }
  function resizeCanvas(canvas) { return canvasSizes.get(canvas) || measureCanvas(canvas); }
  function selectedBeing() { return data.beings[state.selected] || null; }
  function selectedId() { return selectedBeing() ? M.beingId(selectedBeing(), state.selected) : null; }
  function tick() { return data.ticks[state.index] ?? 0; }
  function frameFor(index) { return M.frameAtOrBefore(data.beings[index], tick()); }
  function wallProbesAvailable() { return data.meta.wall_probes_recorded === true || (data.meta.wall_probes_recorded == null && data.meta.sensor_phase !== "unspecified"); }
  function value(value, digits = 1, suffix = "") { return M.finite(value) ? `${value.toFixed(digits)}${suffix}` : "--"; }
  function setText(id, text) { $(id).textContent = text; }

  function installRoster() {
    const picker = $("being-picker"); picker.replaceChildren();
    data.beings.forEach((being, index) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "being-choice";
      button.style.setProperty("--being-color", M.COLORS[index % M.COLORS.length]);
      button.innerHTML = '<span class="being-dot"></span><span></span>';
      button.lastElementChild.textContent = M.beingLabel(being, index);
      button.addEventListener("click", () => selectBeing(index)); picker.appendChild(button);
    });
  }
  function selectBeing(index) {
    if (!data.beings[index]) return;
    state.selected = index; document.documentElement.style.setProperty("--being", M.COLORS[index % M.COLORS.length]); render();
  }
  function fitScene() {
    state.following = false; $("focus-being").setAttribute("aria-pressed", "false");
    const size = resizeCanvas(scene); state.camera = M.fitCamera(M.sceneBounds(data), size.width, size.height, 48); drawScene();
  }
  function followSelected() {
    const frame = frameFor(state.selected), size = resizeCanvas(scene); if (!frame) return;
    if (!state.camera) state.camera = { x: 0, y: 0, scale: 4 };
    state.camera.scale = M.clamp(state.camera.scale || 4, 3, 5);
    state.camera.x = size.width / 2 - frame.x * state.camera.scale;
    state.camera.y = size.height / 2 + frame.y * state.camera.scale;
  }
  function toggleFollow() {
    state.following = !state.following; $("focus-being").setAttribute("aria-pressed", String(state.following));
    if (state.following) { state.camera.scale = 4; followSelected(); } drawScene();
  }

  function drawBackground(ctx, width, height) {
    ctx.fillStyle = "#090d0e"; ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(190,220,207,.035)"; ctx.lineWidth = 1;
    for (let x = 20; x < width; x += 42) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
    for (let y = 20; y < height; y += 42) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
  }
  function traceTrack(ctx, camera) {
    if (data.track.length < 2) return false;
    ctx.beginPath(); let point = M.project(camera, data.track[0][0], data.track[0][1]); ctx.moveTo(point.x, point.y);
    for (let i = 1; i < data.track.length; i++) { point = M.project(camera, data.track[i][0], data.track[i][1]); ctx.lineTo(point.x, point.y); }
    ctx.closePath(); return true;
  }
  function drawTrack(ctx, camera) {
    if (!traceTrack(ctx, camera)) return;
    const trackWidth = M.finite(data.meta.track_width) ? data.meta.track_width : 18;
    ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.strokeStyle = "#24302f";
    ctx.lineWidth = Math.max(5, trackWidth * camera.scale + 4); ctx.stroke();
    traceTrack(ctx, camera); ctx.strokeStyle = "#161e20"; ctx.lineWidth = Math.max(3, trackWidth * camera.scale); ctx.stroke();
    traceTrack(ctx, camera); ctx.setLineDash([5, 10]); ctx.strokeStyle = "rgba(212,231,222,.18)"; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    const start = M.project(camera, data.track[0][0], data.track[0][1]);
    ctx.fillStyle = "#73e0b3"; ctx.fillRect(start.x - 3, start.y - 3, 6, 6);
  }
  function drawTerrain(ctx, camera) {
    const terrain=data.meta.terrain;
    if(!terrain || terrain.alignment!=='cell_center' || !Array.isArray(terrain.hole_cells)) return;
    const cell=terrain.cell_size_m;
    if(!Number.isFinite(cell) || cell<=0) return;
    ctx.save(); ctx.fillStyle='#35240f'; ctx.strokeStyle='#e2b65e'; ctx.lineWidth=1.1;
    for(const [x,z] of terrain.hole_cells) {
      const a=M.project(camera,(x-.5)*cell,(z+.5)*cell), side=cell*camera.scale;
      ctx.fillRect(a.x,a.y,side,side); ctx.strokeRect(a.x,a.y,side,side);
      if(side>6) {ctx.beginPath();ctx.moveTo(a.x+2,a.y+side-2);ctx.lineTo(a.x+side-2,a.y+2);ctx.stroke();}
    }
    ctx.restore();
  }
  function drawRetinalFootprint(ctx,camera) {
    const observation=study.sample, m=study.diagnostics?.meta;
    if(!state.sensors || !observation || observation.being!==selectedId() || observation.tick>tick()) return;
    const pose=observation.sensor_pose;
    if(!pose || !Array.isArray(pose.position) || !Array.isArray(pose.forward) || !Array.isArray(pose.right)) return;
    const origin=M.project(camera,pose.position[0],pose.position[2]);
    ctx.save();ctx.lineWidth=.6;ctx.strokeStyle='rgba(125,218,191,.12)';
    const selectedRows=[0,Math.floor(m.retina_rows/2),m.retina_rows-1];
    for(const row of selectedRows) for(let col=0;col<m.retina_cols;col+=3) {
      const angle=study.retinaAngle(m,col,row), pitch=m.retina_pitch_radians?.[row];
      if(!angle || !Number.isFinite(pitch)) continue;
      const yaw=angle.degrees*Math.PI/180, packed=observation.retina[row*m.retina_cols+col]>>>0;
      const length=(packed>>>8)/1000*Math.cos(pitch);
      const end=M.project(camera,pose.position[0]+(pose.forward[0]*Math.cos(yaw)+pose.right[0]*Math.sin(yaw))*length,
        pose.position[2]+(pose.forward[1]*Math.cos(yaw)+pose.right[1]*Math.sin(yaw))*length);
      ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(end.x,end.y);ctx.stroke();
      ctx.fillStyle=(packed&255)===3?'#e2b65e':(packed&255)===2?'#ed8775':'#76b5a3';ctx.fillRect(end.x-1,end.y-1,2,2);
    }
    const hover=study.hoverRay;
    if(hover&&hover.tick===observation.tick&&hover.being===observation.being) {
      const hit=study.hitWorld(hover.row,hover.col);
      if(hit) {
        const point=M.project(camera,hit.x,hit.z);
        ctx.strokeStyle='#f8e2a6';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(point.x,point.y,7,0,Math.PI*2);ctx.stroke();
        ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(point.x,point.y);ctx.stroke();
      }
    }
    ctx.restore();
  }
  function drawTrail(ctx, camera, being, frame) {
    if (!state.trail || !frame) return;
    const frames = being.frames || [], end = frames.lastIndexOf(frame); if (end < 1) return;
    const start = Math.max(0, end - 700); ctx.beginPath();
    for (let i = start; i <= end; i++) {
      const f = frames[i]; if (!M.finite(f.x) || !M.finite(f.y)) continue;
      const p = M.project(camera, f.x, f.y); if (i === start) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = `${M.COLORS[state.selected % M.COLORS.length]}72`; ctx.lineWidth = 2; ctx.stroke();
  }
  function drawSensors(ctx, camera, frame) {
    if (!state.sensors || !frame || !wallProbesAvailable()) return;
    const origin = M.project(camera, frame.x, frame.y); ctx.save(); ctx.setLineDash([4, 4]);
    for (const probe of M.sensorSegments(frame)) {
      const end = M.project(camera, probe.x, probe.y); ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(end.x, end.y);
      ctx.strokeStyle = probe.name === "front" ? "rgba(240,189,107,.72)" : "rgba(125,184,229,.62)"; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.setLineDash([]); ctx.beginPath(); ctx.arc(end.x, end.y, 2.5, 0, Math.PI * 2); ctx.fillStyle = ctx.strokeStyle; ctx.fill(); ctx.setLineDash([4, 4]);
    }
    ctx.restore();
  }
  function drawBeing(ctx, camera, index, frame) {
    if (!frame || !M.finite(frame.x) || !M.finite(frame.y)) return;
    const p = M.project(camera, frame.x, frame.y), color = M.COLORS[index % M.COLORS.length], active = index === state.selected;
    const angle = Math.atan2(-frame.fy, frame.fx), radius = active ? 8 : 5;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(angle); ctx.beginPath(); ctx.moveTo(radius + 4, 0); ctx.lineTo(-radius, radius * .72); ctx.lineTo(-radius * .55, 0); ctx.lineTo(-radius, -radius * .72); ctx.closePath();
    ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = active ? 16 : 7; ctx.fill(); ctx.restore();
    if (active) { ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2); ctx.strokeStyle = `${color}70`; ctx.lineWidth = 1; ctx.stroke(); }
    state.hits.push({ index, x: p.x, y: p.y });
  }
  function drawScene() {
    const size = resizeCanvas(scene), ctx = sceneContext; ctx.setTransform(size.ratio, 0, 0, size.ratio, 0, 0);
    if (!state.camera) state.camera = M.fitCamera(M.sceneBounds(data), size.width, size.height, 48);
    if (state.following) followSelected();
    drawBackground(ctx, size.width, size.height); drawTrack(ctx, state.camera); drawTerrain(ctx, state.camera); drawRetinalFootprint(ctx, state.camera); state.hits = [];
    const selectedFrame = frameFor(state.selected); drawTrail(ctx, state.camera, selectedBeing(), selectedFrame); drawSensors(ctx, state.camera, selectedFrame);
    data.beings.forEach((_, index) => { if (index !== state.selected) drawBeing(ctx, state.camera, index, frameFor(index)); });
    drawBeing(ctx, state.camera, state.selected, selectedFrame);
  }

  function actionPosition(number, signed) {
    if (!M.finite(number)) return null;
    const normalized = signed ? M.clamp(number, -1, 1) : M.clamp(number, 0, 1);
    return signed ? { left: normalized < 0 ? 50 + normalized * 50 : 50, width: Math.abs(normalized) * 50 }
      : { left: 0, width: normalized * 100 };
  }
  function updateActions(frame) {
    const host = $("action-bars"); host.replaceChildren(); let rawAvailable = false;
    for (const row of M.actionRows(frame)) {
      const wrapper = document.createElement("div"); wrapper.className = "action-row";
      const label = document.createElement("span"); label.textContent = row.label;
      const track = document.createElement("div"); track.className = `action-track${row.signed ? "" : " unsigned"}`;
      for (const [kind, amount] of [["raw", row.raw], ["applied", row.applied]]) {
        const position = actionPosition(amount, row.signed); if (!position) continue;
        if (kind === "raw") rawAvailable = true;
        const bar = document.createElement("i"); bar.className = `action-bar ${kind}`; bar.style.left = `${position.left}%`; bar.style.width = `${position.width}%`; track.appendChild(bar);
      }
      const output = document.createElement("output"); output.textContent = M.finite(row.applied) ? row.applied.toFixed(2) : "--";
      wrapper.append(label, track, output); host.appendChild(wrapper);
    }
    setText("action-note", rawAvailable ? "Outline shows brain intent; solid shows the command applied to physics." : "Raw brain intent was not recorded; solid bars show applied commands only.");
  }
  function retinaColor(number) {
    if (!M.finite(number)) return "#202729"; const t = M.clamp((number + 128) / 255, 0, 1);
    return `rgb(${Math.round(31 + t * 205)},${Math.round(55 + t * 180)},${Math.round(60 + t * 165)})`;
  }
  function updateRetina(id, values) {
    const host = $(id); host.replaceChildren();
    for (let i = 0; i < 8; i++) { const cell = document.createElement("i"); cell.className = "retina-cell"; cell.title = M.finite(values?.[i]) ? String(values[i]) : "Unavailable"; cell.style.background = retinaColor(values?.[i]); host.appendChild(cell); }
  }
  function meter(id, outputId, amount) {
    $(id).style.width = M.finite(amount) ? `${M.clamp(amount, 0, 1) * 100}%` : "0";
    setText(outputId, M.finite(amount) ? `${Math.round(M.clamp(amount, 0, 1) * 100)}%` : "--");
  }
  function updateFlags(frame) {
    const flags = [[1, "Governor", ""], [2, "Sentinel", "danger"], [4, "ECAS", ""], [8, "Reverse", ""]], host = $("safety-flags"); host.replaceChildren();
    if (!Number.isInteger(frame?.sf)) { const unavailable = document.createElement("span"); unavailable.className = "section-note"; unavailable.textContent = "Safety interventions were not recorded."; host.appendChild(unavailable); setText("intent-strength", "neural summary unavailable"); return; }
    for (const [bit, label, kind] of flags) { const flag = document.createElement("span"); flag.className = `flag ${kind}${frame && (frame.sf & bit) ? " active" : ""}`; flag.textContent = label; host.appendChild(flag); }
    const strength = M.finite(frame?.axiomStrength) ? `axiom ${Math.round(frame.axiomStrength * 100)}%` : "axiom unavailable"; setText("intent-strength", strength);
  }

  function updateInspector() {
    const being = selectedBeing(), frame = frameFor(state.selected), name = being ? M.beingLabel(being, state.selected) : "No being";
    setText("being-name", name);
    const lowLife = frame?.life === 0, lifeName = data.meta.life_semantics === "remaining_tick_budget" ? "Budget" : "Life";
    const stateBadge = $("being-state"); stateBadge.className = `state-badge${frame ? (lowLife ? " warning" : " active") : ""}`; stateBadge.textContent = frame ? (lowLife ? `${lifeName} <1%` : "Recorded") : "No frame";
    setText("stat-speed", M.finite(frame?.spd) ? (frame.spd * 3.6).toFixed(0) : "--"); setText("scene-speed", M.finite(frame?.spd) ? `${(frame.spd * 3.6).toFixed(0)} km/h` : "-- km/h");
    const laps = M.completedLaps(validation, selectedId(), tick());
    setText("stat-lap", laps === null ? "--" : String(laps));
    setText("scene-lap", laps === null ? "Circuits not verified" : `${laps} completed ${laps === 1 ? "lap" : "laps"}`);
    setText("stat-arc", value(frame?.arc, 0, " m")); setText("scene-progress", value(frame?.arc, 0, " m")); setText("stat-gear", M.finite(frame?.gear) ? (frame.gear === 0 ? "N/R" : String(frame.gear)) : "--"); setText("stat-yaw", value(frame?.yawRate, 2, " rad/s"));
    meter("life-meter", "life-value", frame?.life); meter("fuel-meter", "fuel-value", frame?.fuel);
    const lifeLabels = { health: "physical health", remaining_tick_budget: "remaining life budget", unavailable: "not recorded" };
    setText("body-status", lifeLabels[data.meta.life_semantics] || "Recorded state"); updateActions(frame);
    const probesAvailable = wallProbesAvailable();
    setText("sensor-phase", probesAvailable ? `${data.meta.sensor_phase.replaceAll("_", "-")} sample` : "unavailable");
    setText("probe-left", probesAvailable ? value(frame?.cl, 1, " m") : "--"); setText("probe-front", probesAvailable ? value(frame?.cf, 1, " m") : "--"); setText("probe-right", probesAvailable ? value(frame?.cr, 1, " m") : "--");
    updateRetina("retina-foveal", frame?.fov); updateRetina("retina-peripheral", frame?.peri); setText("retina-status", frame?.fov ? "recorded" : "unavailable"); updateFlags(frame);
    setText("phase-note", `Body/actions: ${data.meta.frame_phase || "phase unknown"}. Wall probes: ${data.meta.sensor_phase || "phase unknown"}. Track display uses the recorded centreline and nominal width, not collider boundaries.`); study.update();
    document.querySelectorAll(".being-choice").forEach((choice, index) => choice.setAttribute("aria-current", String(index === state.selected)));
  }
  function updateTimeline() {
    $("timeline").max = String(Math.max(0, data.ticks.length - 1)); $("timeline").value = String(state.index);
    setText("frame-readout", `Tick ${tick()} / ${data.ticks.at(-1) ?? 0}`);
    $("frame-readout").title = `Recorded frame ${data.ticks.length ? state.index + 1 : 0} of ${data.ticks.length}; simulation ticks and saved frames have different cadences.`;
    setText("time-readout", M.formatTime(tick() - (data.ticks[0] || 0), data.meta.tick_rate));
  }
  function render() { study.update(); drawScene(); updateInspector(); updateTimeline(); }

  function playbackStatus() { setText("playback-status", state.playing ? "Playing" : state.index === data.ticks.length - 1 ? "Finished · replay with Space" : "Paused · Space to play"); }
  function stop() { state.playing = false; cancelAnimationFrame(state.timer); $("play").classList.remove("playing"); $("play").setAttribute("aria-label", "Play replay"); playbackStatus(); }
  function indexAtTick(target) { let low = 0, high = data.ticks.length - 1, found = 0; while (low <= high) { const mid = (low + high) >> 1; if (data.ticks[mid] <= target) { found = mid; low = mid + 1; } else high = mid - 1; } return found; }
  function schedule(now) {
    if (!state.playing) return;
    const speed = Number($("playback-speed").value), elapsedTicks = (now - state.playOriginTime) * (data.meta.tick_rate || 64) * speed / 1000;
    const targetTick = state.playOriginTick + elapsedTicks, nextIndex = indexAtTick(targetTick);
    if (nextIndex !== state.index) { state.index = nextIndex; render(); }
    if (targetTick >= data.ticks.at(-1)) { state.index = Math.max(0, data.ticks.length - 1); render(); stop(); return; }
    state.timer = requestAnimationFrame(schedule);
  }
  function rebasePlayback() { state.playOriginTime = performance.now(); state.playOriginTick = tick(); }
  function togglePlay() { if (state.playing) { stop(); return; } if (state.index >= data.ticks.length - 1) state.index = 0; state.playing = true; rebasePlayback(); $("play").classList.add("playing"); $("play").setAttribute("aria-label", "Pause replay"); playbackStatus(); state.timer = requestAnimationFrame(schedule); }
  function go(delta) { stop(); state.index = M.clamp(state.index + delta, 0, Math.max(0, data.ticks.length - 1)); render(); }

  $("play").addEventListener("click", togglePlay); $("step-back").addEventListener("click", () => go(-1)); $("step-forward").addEventListener("click", () => go(1));
  $("playback-speed").addEventListener("change", () => { if (state.playing) rebasePlayback(); });
  $("timeline").addEventListener("input", (event) => { stop(); state.index = Number(event.target.value); render(); });
  $("show-trail").addEventListener("change", (event) => { state.trail = event.target.checked; drawScene(); });
  $("show-sensors").addEventListener("change", (event) => { state.sensors = event.target.checked; drawScene(); }); $("fit-track").addEventListener("click", fitScene); $("focus-being").addEventListener("click", toggleFollow);
  $("toggle-inspector").addEventListener("click", () => { const closed = $("inspector").classList.toggle("closed"); $("toggle-inspector").setAttribute("aria-expanded", String(!closed)); });
  scene.addEventListener("wheel", (event) => { event.preventDefault(); const rect = scene.getBoundingClientRect(), mx = event.clientX - rect.left, my = event.clientY - rect.top;
    const worldX = (mx - state.camera.x) / state.camera.scale, worldY = (state.camera.y - my) / state.camera.scale, factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    state.camera.scale = M.clamp(state.camera.scale * factor, .01, 200); state.camera.x = mx - worldX * state.camera.scale; state.camera.y = my + worldY * state.camera.scale; drawScene(); }, { passive: false });
  scene.addEventListener("pointerdown", (event) => { scene.setPointerCapture(event.pointerId); state.following = false; $("focus-being").setAttribute("aria-pressed", "false"); state.drag = { x: event.clientX, y: event.clientY }; state.moved = false; scene.classList.add("dragging"); });
  scene.addEventListener("pointermove", (event) => { if (!state.drag) return; const dx = event.clientX - state.drag.x, dy = event.clientY - state.drag.y; if (Math.abs(dx) + Math.abs(dy) > 2) state.moved = true;
    state.camera.x += dx; state.camera.y += dy; state.drag = { x: event.clientX, y: event.clientY }; drawScene(); });
  scene.addEventListener("pointerup", (event) => { scene.classList.remove("dragging"); state.drag = null; if (state.moved) return; const rect = scene.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
    const hit = state.hits.map((item) => ({ ...item, distance: Math.hypot(item.x - x, item.y - y) })).filter((item) => item.distance < 18).sort((a, b) => a.distance - b.distance)[0]; if (hit) selectBeing(hit.index); });
  scene.addEventListener("dblclick", fitScene); document.addEventListener("keydown", (event) => { if (event.target.matches("input,select,button")) return;
    if (event.code === "Space") { event.preventDefault(); togglePlay(); } else if (event.key === "ArrowRight") go(1); else if (event.key === "ArrowLeft") go(-1); });
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) measureCanvas(entry.target);
    if (!state.drag) fitScene();
  });
  resizeObserver.observe(scene);
  const sourceLabel = Number.isFinite(data.meta.fixture_tick_limit)
    ? `${M.controllerLabel(data.meta)} · recorded ${(data.meta.fixture_tick_limit / data.meta.tick_rate).toFixed(1)} s excerpt`
    : M.controllerLabel(data.meta);
  setText("run-source", sourceLabel); setText("run-rate", `${data.meta.tick_rate || "--"} Hz`); setText("run-seed", `seed ${data.meta.world_seed ?? "--"}`);
  const moving = M.motionSummary(data).filter((being) => being.distance >= 10).length;
  const duration = (data.ticks.at(-1) || 0) / (data.meta.tick_rate || 64);
  setText("run-outcome", validation?.schema_version === 1
    ? `${validation.passed}/${validation.total} completed ≥${validation.min_laps} lap(s) · ${duration.toFixed(1)} s recorded`
    : `${moving}/${data.beings.length} travelled ≥10 m · ${duration.toFixed(1)} s · circuits not verified`);
  $("run-outcome").classList.toggle("warning", validation ? !validation.valid : moving === 0);
  $("scene-empty").hidden = data.ticks.length > 0;
  if (matchMedia("(max-width:900px)").matches) { $("inspector").classList.add("closed"); $("toggle-inspector").setAttribute("aria-expanded", "false"); }
  const terrain=data.meta.terrain;
  let holeIndex=0;
  $('inspect-hole').disabled=!terrain?.hole_cells?.length;
  $('inspect-hole').addEventListener('click',()=>{
    if(!terrain?.hole_cells?.length) return;
    const [x,z]=terrain.hole_cells[holeIndex++%terrain.hole_cells.length], size=resizeCanvas(scene);
    state.following=false; $('focus-being').setAttribute('aria-pressed','false');
    const scale=7;
    state.camera={scale,x:size.width/2-x*terrain.cell_size_m*scale,y:size.height/2+z*terrain.cell_size_m*scale};
    drawScene();
  });
  $('terrain-legend').textContent=terrain?`Holes · ${terrain.hole_cells.length} recorded cells · ${terrain.cell_size_m} m each`:'Terrain cells were not recorded';
  installRoster(); if (data.beings.length) selectBeing(0); else render(); fitScene();
  root.__UNIVERSE_REPLAY__ = { study, data, state, render, selectBeing, drawScene };
  playbackStatus();
  if (data.ticks.length > 1 && !matchMedia("(prefers-reduced-motion: reduce)").matches) togglePlay();
})(window);
