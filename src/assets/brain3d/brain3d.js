/* Anatomical point/connectivity viewer. Activity is sampled, never interpolated. */
(function (root) {
  'use strict';
  const decode = (text, Type) => {
    const binary = atob(text), bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Type(bytes.buffer);
  };
  class Brain3D {
    constructor(canvas, data) {
      this.canvas = canvas; this.data = data; this.frame = -1; this.selection = 0;
      this.yaw = 0; this.pitch = -.18; this.zoom = 1; this.pan = [0, 0]; this.drag = null;
      this.gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
      if (!this.gl) throw new Error('WebGL2 is unavailable. Enable browser graphics acceleration and reload.');
      this.raw = decode(data.xyz, Float32Array);
      this.pre = decode(data.edgePre, Uint32Array); this.post = decode(data.edgePost, Uint32Array);
      if (this.raw.length !== data.n * 3 || this.pre.length !== data.m || this.post.length !== data.m)
        throw new Error('Brain coordinate/edge counts do not match the manifest.');
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      this.valid = new Uint8Array(data.n); this.visibleCount = 0;
      for (let i = 0; i < data.n; i++) {
        if (![0, 1, 2].every(k => Number.isFinite(this.raw[i * 3 + k]))) continue;
        this.valid[i] = 1; this.visibleCount++;
        for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], this.raw[i * 3 + k]); hi[k] = Math.max(hi[k], this.raw[i * 3 + k]); }
      }
      if (!this.visibleCount) throw new Error('No finite anatomical coordinates were recorded.');
      this.extent = Math.max(...hi.map((v, k) => v - lo[k]));
      const center = lo.map((v, k) => (v + hi[k]) / 2), scale = 1.65 / Math.max(this.extent, 1e-9);
      this.xyz = new Float32Array(this.raw.length);
      for (let i = 0; i < data.n; i++) for (let k = 0; k < 3; k++)
        this.xyz[i * 3 + k] = this.valid[i] ? (this.raw[i * 3 + k] - center[k]) * scale : 1000;
      // Precompute a compact adjacency index once; do not scan 15M edges each frame.
      this.offsets = new Uint32Array(data.n + 1);
      for (let i = 0; i < data.m; i++) {
        if (this.pre[i] >= data.n || this.post[i] >= data.n) throw new Error('Edge endpoint outside neuron index.');
        this.offsets[this.pre[i] + 1]++; this.offsets[this.post[i] + 1]++;
      }
      for (let i = 1; i <= data.n; i++) this.offsets[i] += this.offsets[i - 1];
      this.adjacency = new Uint32Array(this.offsets[data.n]);
      const cursor = this.offsets.slice();
      for (let i = 0; i < data.m; i++) { this.adjacency[cursor[this.pre[i]]++] = i; this.adjacency[cursor[this.post[i]]++] = i; }
      this.activity = new Float32Array(data.n).fill(-1);
      this._init(); this._bind(); this.setSelection(0); this.resize();
    }
    _init() {
      const g = this.gl;
      const shader = (kind, source) => {
        const result = g.createShader(kind); g.shaderSource(result, source); g.compileShader(result);
        if (!g.getShaderParameter(result, g.COMPILE_STATUS)) throw new Error(g.getShaderInfoLog(result));
        return result;
      };
      this.program = g.createProgram();
      const vs = shader(g.VERTEX_SHADER, `#version 300 es
        in vec3 position; in float activation;
        uniform vec2 orbit; uniform vec2 pan; uniform float zoom; uniform float aspect;
        uniform float pointSize; uniform int selected; out float level;
        void main() {
          float cy=cos(orbit.x), sy=sin(orbit.x), cp=cos(orbit.y), sp=sin(orbit.y);
          vec3 q=vec3(cy*position.x-sy*position.z,position.y,sy*position.x+cy*position.z);
          q=vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);
          float perspective=3.5/(3.5-q.z);
          gl_Position=vec4((q.xy*zoom*perspective+pan)*vec2(1.0/aspect,-1.0),-q.z*.25,1.0);
          gl_PointSize=gl_VertexID==selected?pointSize*3.5:pointSize;
          level=gl_VertexID==selected?2.0:activation;
        }`);
      const fs = shader(g.FRAGMENT_SHADER, `#version 300 es
        precision highp float; in float level; uniform bool lines; uniform vec4 lineColor; out vec4 color;
        void main() {
          if(lines) { color=lineColor; return; }
          if(distance(gl_PointCoord,vec2(.5))>.5) discard;
          if(level>1.5) color=vec4(1.,.72,.26,1.);
          else if(level<0.) color=vec4(.30,.44,.55,.55);
          else color=vec4(mix(vec3(.13,.31,.57),vec3(1.,.91,.57),clamp(level,0.,1.)),.9);
        }`);
      g.attachShader(this.program, vs); g.attachShader(this.program, fs); g.linkProgram(this.program);
      g.deleteShader(vs); g.deleteShader(fs);
      if (!g.getProgramParameter(this.program, g.LINK_STATUS)) throw new Error(g.getProgramInfoLog(this.program));
      this.loc = {};
      for (const name of ['orbit','pan','zoom','aspect','pointSize','selected','lines','lineColor']) this.loc[name] = g.getUniformLocation(this.program, name);
      this.positionLoc = g.getAttribLocation(this.program, 'position'); this.activityLoc = g.getAttribLocation(this.program, 'activation');
      this.posBuffer = g.createBuffer(); this.actBuffer = g.createBuffer(); this.edgeBuffer = g.createBuffer();
      g.bindBuffer(g.ARRAY_BUFFER, this.posBuffer); g.bufferData(g.ARRAY_BUFFER, this.xyz, g.STATIC_DRAW);
      g.bindBuffer(g.ARRAY_BUFFER, this.actBuffer); g.bufferData(g.ARRAY_BUFFER, this.activity, g.DYNAMIC_DRAW);
    }
    _bind() {
      const c = this.canvas;
      this.down = e => { this.drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2, originX:e.clientX, originY:e.clientY }; c.setPointerCapture(e.pointerId); };
      this.move = e => {
        if (!this.drag) return;
        const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
        if (this.drag.pan) { this.pan[0] += dx / Math.max(c.clientHeight, 1) * 2; this.pan[1] += dy / Math.max(c.clientHeight, 1) * 2; }
        else { this.yaw += dx * .008; this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * .008)); }
        this.drag.x = e.clientX; this.drag.y = e.clientY; this.draw();
      };
      this.up = e => {
        if(this.drag && !this.drag.pan && Math.hypot(e.clientX-this.drag.originX,e.clientY-this.drag.originY)<4) this.pick(e.clientX,e.clientY);
        this.drag = null;
      };
      this.wheel = e => { e.preventDefault(); this.zoom = Math.max(.15, Math.min(12, this.zoom * Math.exp(-e.deltaY * .001))); this.draw(); };
      this.context = e => e.preventDefault();
      for (const [name, fn] of [['pointerdown',this.down],['pointermove',this.move],['pointerup',this.up],['pointercancel',this.up],['wheel',this.wheel],['contextmenu',this.context]])
        c.addEventListener(name, fn, { passive: false });
      this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(c);
    }
    setFrame(index) {
      if (index === this.frame) return this;
      const f = this.data.frames?.[index]; this.frame = f ? index : -1;
      for (let i = 0; i < this.data.n; i++) {
        const voltage = f?.v?.[i], spikes = f?.spikes?.[i];
        this.activity[i] = Number.isFinite(voltage) ? (spikes > 0 ? 1 : Math.max(0, Math.min(1, (voltage + 70) / 28))) : -1;
      }
      const g = this.gl; g.bindBuffer(g.ARRAY_BUFFER, this.actBuffer); g.bufferSubData(g.ARRAY_BUFFER, 0, this.activity);
      this.draw(); return this;
    }
    setFrameByTick(tick, being) {
      let best = -1;
      if (String(being) === String(this.data.traceBeing))
        for (let i = 0; i < (this.data.frames || []).length; i++) {
          const f = this.data.frames[i];
          const t = f.tick ?? (f.step / this.data.traceStepsPerTick - 1);
          if (Number.isSafeInteger(t) && t <= tick) best = i;
        }
      return this.setFrame(best);
    }
    setSelection(index) {
      this.selection = Math.max(0, Math.min(this.data.n - 1, Number(index) || 0)) | 0;
      const edges = [], begin = this.offsets[this.selection], end = this.offsets[this.selection + 1];
      this.incidentCount = end - begin; this.drawnEdges = 0;
      for (let i = begin; i < end && this.drawnEdges < 6000; i++) {
        const e = this.adjacency[i], a = this.pre[e], b = this.post[e];
        if (!this.valid[a] || !this.valid[b]) continue;
        for (const n of [a,b]) edges.push(this.xyz[n * 3], this.xyz[n * 3 + 1], this.xyz[n * 3 + 2]);
        this.drawnEdges++;
      }
      const g = this.gl; g.bindBuffer(g.ARRAY_BUFFER, this.edgeBuffer); g.bufferData(g.ARRAY_BUFFER, new Float32Array(edges), g.STATIC_DRAW);
      this.draw(); return this;
    }
    pick(clientX,clientY) {
      const r=this.canvas.getBoundingClientRect(), x=clientX-r.left, y=clientY-r.top, aspect=r.width/r.height;
      const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
      let best=-1, distance=64;
      for(let i=0;i<this.data.n;i++) {
        if(!this.valid[i])continue;
        const a=this.xyz[i*3],b=this.xyz[i*3+1],c=this.xyz[i*3+2];
        const qx=cy*a-sy*c,qz=sy*a+cy*c,qy=cp*b-sp*qz,z=sp*b+cp*qz,k=3.5/(3.5-z);
        const px=((qx*this.zoom*k+this.pan[0])/aspect*.5+.5)*r.width;
        const py=((qy*this.zoom*k+this.pan[1])*.5+.5)*r.height;
        const d=(px-x)**2+(py-y)**2;
        if(d<distance) {best=i;distance=d;}
      }
      if(best>=0) {this.setSelection(best); if(this.onSelection)this.onSelection(best);}
      return best;
    }
    resetCamera() { this.yaw=0; this.pitch=-.18; this.zoom=1; this.pan=[0,0]; this.draw(); }
    resize() {
      const rect = this.canvas.getBoundingClientRect(), ratio = Math.min(root.devicePixelRatio || 1, 2);
      if (rect.width < 1 || rect.height < 1) return this;
      this.canvas.width = Math.round(rect.width * ratio); this.canvas.height = Math.round(rect.height * ratio);
      return this.draw();
    }
    draw() {
      const g=this.gl, c=this.canvas;
      g.viewport(0,0,c.width,c.height); g.clearColor(.027,.038,.056,1); g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);
      g.enable(g.DEPTH_TEST); g.enable(g.BLEND); g.blendFunc(g.SRC_ALPHA,g.ONE_MINUS_SRC_ALPHA); g.useProgram(this.program);
      g.uniform2f(this.loc.orbit,this.yaw,this.pitch); g.uniform2fv(this.loc.pan,this.pan);
      g.uniform1f(this.loc.zoom,this.zoom); g.uniform1f(this.loc.aspect,c.width/Math.max(c.height,1));
      g.uniform1f(this.loc.pointSize,Math.max(1.2,Math.min(3,c.height/400))); g.uniform1i(this.loc.selected,this.selection);
      g.uniform1i(this.loc.lines,0);
      g.bindBuffer(g.ARRAY_BUFFER,this.posBuffer); g.enableVertexAttribArray(this.positionLoc); g.vertexAttribPointer(this.positionLoc,3,g.FLOAT,false,0,0);
      g.bindBuffer(g.ARRAY_BUFFER,this.actBuffer); g.enableVertexAttribArray(this.activityLoc); g.vertexAttribPointer(this.activityLoc,1,g.FLOAT,false,0,0);
      g.drawArrays(g.POINTS,0,this.data.n);
      if (this.drawnEdges) {
        g.uniform1i(this.loc.lines,1); g.uniform4f(this.loc.lineColor,1,.57,.25,.45);
        g.disableVertexAttribArray(this.activityLoc); g.vertexAttrib1f(this.activityLoc,-1);
        g.bindBuffer(g.ARRAY_BUFFER,this.edgeBuffer); g.vertexAttribPointer(this.positionLoc,3,g.FLOAT,false,0,0);
        g.drawArrays(g.LINES,0,this.drawnEdges*2);
      }
      return this;
    }
    dispose() {
      for (const [name, fn] of [['pointerdown',this.down],['pointermove',this.move],['pointerup',this.up],['pointercancel',this.up],['wheel',this.wheel],['contextmenu',this.context]]) this.canvas.removeEventListener(name,fn);
      this.observer.disconnect(); const g=this.gl;
      for (const b of [this.posBuffer,this.actBuffer,this.edgeBuffer]) g.deleteBuffer(b);
      g.deleteProgram(this.program);
    }
  }
  root.Brain3D = Brain3D;
})(window);
