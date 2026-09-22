/* Native controller viewer. Positions are schematic; weights and sampled states are recorded data. */
(function (root) {
  'use strict';
  const LAYERS = [
    ['wave',192,-1.7,.72], ['sensory KAN',192,-1.1,.72], ['input CfC',96,-.5,.72],
    ['L1',96,.1,.72], ['L2',48,.7,.72], ['motor head',3,1.3,.72],
    ['slow input',48,-1.1,-1.15], ['slow CfC',64,-.3,-1.15],
    ['slow dense',16,.5,-1.15], ['slow output',3,1.3,-1.15]
  ];
  const OUTPUTS = ['throttle','steer','brake'];
  const SLOW_OUTPUTS = ['aggression signal','throttle-cap signal','steer-bias signal'];
  function buildLayout(){const offsets={},nodes=[],positions=[];let offset=0;for(const [name,count,x,y] of LAYERS){offsets[name]=offset;const columns=Math.min(16,Math.ceil(Math.sqrt(count*1.5))),rows=Math.ceil(count/columns);for(let i=0;i<count;i++){const col=i%columns,row=Math.floor(i/columns);nodes.push({layer:name,index:i});positions.push(x,y+(col-(columns-1)/2)*.075,(row-(rows-1)/2)*.105);}offset+=count;}return {offsets,nodes,positions,n:offset};}

  class NativeBrain3D {
    constructor(canvas, data) {
      this.canvas=canvas; this.data=data; this.selection=0; this.yaw=-.12; this.pitch=-.12;
      this.zoom=1; this.pan=[0,0]; this.drag=null; this.sample=null; this.selectedBeing=null;this.fitSize=null;
      const layout=buildLayout();this.offsets=layout.offsets;this.nodes=layout.nodes;this.positions=layout.positions;this.n=layout.n;
      this.xyz=new Float32Array(this.positions); this.activity=new Float32Array(this.n).fill(-2);
      this.gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true});
      if(!this.gl) throw new Error('WebGL2 is unavailable. Enable browser graphics acceleration and reload.');
      this._init(); this._bind(); this.setSelection(this.node('L1',0)); this.resize();
    }
    _init() {
      const g=this.gl, compile=(kind,source)=>{const s=g.createShader(kind);g.shaderSource(s,source);g.compileShader(s);if(!g.getShaderParameter(s,g.COMPILE_STATUS))throw new Error(g.getShaderInfoLog(s));return s;};
      this.program=g.createProgram();
      const vs=compile(g.VERTEX_SHADER,`#version 300 es
        in vec3 position; in float activation; uniform vec2 orbit; uniform vec2 pan;
        uniform float zoom; uniform float aspect; uniform float pointSize; uniform int selected; uniform bool lines;
        out float level;
        void main(){float cy=cos(orbit.x),sy=sin(orbit.x),cp=cos(orbit.y),sp=sin(orbit.y);
          vec3 q=vec3(cy*position.x-sy*position.z,position.y,sy*position.x+cy*position.z);
          q=vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);float p=4.5/(4.5-q.z);
          gl_Position=vec4((q.xy*zoom*p+pan)*vec2(1.0/aspect,-1.0),-q.z*.2,1.0);
          bool isSelectedPoint=!lines&&gl_VertexID==selected;
          gl_PointSize=isSelectedPoint?pointSize*3.2:pointSize;level=isSelectedPoint?4.0:activation;}`);
      const fs=compile(g.FRAGMENT_SHADER,`#version 300 es
        precision highp float;in float level;uniform bool lines;uniform bool gradientLines;uniform vec4 lineColor;out vec4 color;
        void main(){
          if(lines){
            if(gradientLines){
              // Edge magnitude is a normalised local contribution a*w, not a causal weight; see nodeInfo().
              float m=clamp(abs(level),0.0,1.0);
              vec3 rgb=level<0.0?mix(vec3(.38,.55,.78),vec3(.16,.32,.57),m):mix(vec3(.42,.68,.64),vec3(1.,.55,.30),m);
              color=vec4(rgb,.12+.78*m);return;
            }
            color=lineColor;return;
          }
          if(distance(gl_PointCoord,vec2(.5))>.5)discard;
          if(level>3.0)color=vec4(1.,.77,.28,1.);else if(level< -1.5)color=vec4(.29,.39,.49,.62);
          else if(level<0.)color=vec4(mix(vec3(.38,.55,.78),vec3(.16,.32,.57),-level),.95);
          else color=vec4(mix(vec3(.42,.68,.64),vec3(1.,.55,.30),level),.95);}`);
      g.attachShader(this.program,vs);g.attachShader(this.program,fs);g.linkProgram(this.program);g.deleteShader(vs);g.deleteShader(fs);
      if(!g.getProgramParameter(this.program,g.LINK_STATUS))throw new Error(g.getProgramInfoLog(this.program));
      this.loc={};for(const name of ['orbit','pan','zoom','aspect','pointSize','selected','lines','lineColor','gradientLines'])this.loc[name]=g.getUniformLocation(this.program,name);
      this.positionLoc=g.getAttribLocation(this.program,'position');this.activityLoc=g.getAttribLocation(this.program,'activation');
      this.posBuffer=g.createBuffer();this.actBuffer=g.createBuffer();this.edgeBuffers=[g.createBuffer(),g.createBuffer(),g.createBuffer()];
      this.contribPosBuffer=g.createBuffer();this.contribLevelBuffer=g.createBuffer();this.contribCount=0;
      g.bindBuffer(g.ARRAY_BUFFER,this.posBuffer);g.bufferData(g.ARRAY_BUFFER,this.xyz,g.STATIC_DRAW);
      g.bindBuffer(g.ARRAY_BUFFER,this.actBuffer);g.bufferData(g.ARRAY_BUFFER,this.activity,g.DYNAMIC_DRAW);
    }
    _bind() {
      const c=this.canvas;
      this.down=e=>{this.drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2,originX:e.clientX,originY:e.clientY};c.setPointerCapture(e.pointerId);};
      this.move=e=>{if(!this.drag)return;const dx=e.clientX-this.drag.x,dy=e.clientY-this.drag.y;if(this.drag.pan){this.pan[0]+=dx/Math.max(c.clientHeight,1)*2;this.pan[1]+=dy/Math.max(c.clientHeight,1)*2;}else{this.yaw+=dx*.008;this.pitch=Math.max(-1.45,Math.min(1.45,this.pitch+dy*.008));}this.drag.x=e.clientX;this.drag.y=e.clientY;this.draw();};
      this.up=e=>{if(this.drag&&!this.drag.pan&&Math.hypot(e.clientX-this.drag.originX,e.clientY-this.drag.originY)<4)this.pick(e.clientX,e.clientY);this.drag=null;};
      this.wheel=e=>{e.preventDefault();this.zoom=Math.max(.2,Math.min(5,this.zoom*Math.exp(-e.deltaY*.001)));this.draw();};
      this.context=e=>e.preventDefault();
      for(const [name,fn] of [['pointerdown',this.down],['pointermove',this.move],['pointerup',this.up],['pointercancel',this.up],['wheel',this.wheel],['contextmenu',this.context]])c.addEventListener(name,fn,{passive:false});
      this.observer=new ResizeObserver(()=>this.resize());this.observer.observe(c);
    }
    node(layer,index){return this.offsets[layer]+index;}
    edge(list,a,b,weight=null,contribution=null){
      const e={a,b,weight};
      if(contribution!==null&&Number.isFinite(contribution))e.contribution=contribution;
      list.push(e);
    }
    /* Value of `stages[stageName][i]` for the currently observed sample, only when that
       sample truly belongs to the being whose topology is attached (see setObservation). */
    recordedStage(stageName,i) {
      const s=this.sample;
      if(!s||String(this.selectedBeing)!==String(this.data.traceBeing)||String(s.being)!==String(this.selectedBeing))return null;
      const values=s.stages?.[stageName];
      return Array.isArray(values)&&Number.isFinite(values[i])?values[i]:null;
    }
    incident(index) {
      const out=[], node=this.nodes[index], i=node.index, f=this.data.fast, s=this.data.slow;
      if(node.layer==='wave') {
        this.edge(out,index,this.node('sensory KAN',i));
        for(let k=0;k<48;k++)if(s.gatherMap[k]===i)this.edge(out,index,this.node('slow input',k));
      } else if(node.layer==='sensory KAN') {
        this.edge(out,this.node('wave',i),index);
        if(i<96)this.edge(out,index,this.node('input CfC',i));
        else for(let j=0;j<96;j++)this.edge(out,index,this.node('L1',j),f.l1Weights[j*192+i]);
      } else if(node.layer==='input CfC') {
        this.edge(out,this.node('sensory KAN',i),index);
        for(let j=0;j<96;j++)this.edge(out,index,this.node('L1',j),f.l1Weights[j*192+i]);
      } else if(node.layer==='L1') {
        for(let k=0;k<192;k++)this.edge(out,k<96?this.node('input CfC',k):this.node('sensory KAN',k),index,f.l1Weights[i*192+k]);
        // l1_to_l2[i] is the exact i8 this L1 unit actually delivered to L2 on the sampled
        // tick; contribution = that recorded activation times the checkpoint weight.
        const activation=this.recordedStage('l1_to_l2',i);
        for(let j=0;j<48;j++){
          const weight=f.l2Weights[j*96+i];
          this.edge(out,index,this.node('L2',j),weight,activation!==null?activation*weight:null);
        }
      } else if(node.layer==='L2') {
        for(let k=0;k<96;k++){
          const weight=f.l2Weights[i*96+k], activation=this.recordedStage('l1_to_l2',k);
          this.edge(out,this.node('L1',k),index,weight,activation!==null?activation*weight:null);
        }
        if(i<3)this.edge(out,index,this.node('motor head',i));
      } else if(node.layer==='motor head') this.edge(out,this.node('L2',i),index);
      else if(node.layer==='slow input') {
        if(s.gatherMap[i]>=0)this.edge(out,this.node('wave',s.gatherMap[i]),index);
        this.edge(out,index,this.node('slow CfC',i));
      } else if(node.layer==='slow CfC') {
        if(i<48)this.edge(out,this.node('slow input',i),index);
        for(let j=0;j<16;j++)this.edge(out,index,this.node('slow dense',j),s.d1Weights[j*64+i]);
      } else if(node.layer==='slow dense') {
        for(let k=0;k<64;k++)this.edge(out,this.node('slow CfC',k),index,s.d1Weights[i*64+k]);
        for(let j=0;j<3;j++)this.edge(out,index,this.node('slow output',j),s.d2Weights[j*16+i]);
      } else if(node.layer==='slow output') for(let k=0;k<16;k++)this.edge(out,this.node('slow dense',k),index,s.d2Weights[i*16+k]);
      return out;
    }
    visibleConnections(index) {
      return this.incident(index).filter(edge=>edge.weight===null||(Number.isFinite(edge.weight)&&edge.weight!==0));
    }
    connectionSummary(index) {
      const edges=this.incident(index),summary={slots:edges.length,shown:0,incoming:0,outgoing:0,zero:0,invalid:0,positive:0,negative:0,fixed:0};
      for(const edge of edges) {
        if(edge.weight===0){summary.zero++;continue;}
        if(edge.weight!==null&&!Number.isFinite(edge.weight)){summary.invalid++;continue;}
        summary.shown++;summary[edge.b===index?'incoming':'outgoing']++;
        summary[edge.weight===null?'fixed':edge.weight<0?'negative':'positive']++;
      }
      return summary;
    }
    setSelection(index) {
      this.selection=Math.max(0,Math.min(this.n-1,Number(index)||0))|0;const edges=this.visibleConnections(this.selection);
      const groups=[[],[],[]], contribXyz=[], contribLevel=[];
      const contribMax=Math.max(1e-6,...edges.filter(e=>e.contribution!==undefined).map(e=>Math.abs(e.contribution)));
      for(const edge of edges){
        if(edge.contribution!==undefined){
          const level=Math.max(-1,Math.min(1,edge.contribution/contribMax));
          for(const n of [edge.a,edge.b]){contribXyz.push(this.xyz[n*3],this.xyz[n*3+1],this.xyz[n*3+2]);contribLevel.push(level);}
          continue;
        }
        const kind=edge.weight===null?2:(edge.weight<0?0:1);
        for(const n of [edge.a,edge.b])groups[kind].push(this.xyz[n*3],this.xyz[n*3+1],this.xyz[n*3+2]);
      }
      for(let i=0;i<3;i++){this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.edgeBuffers[i]);this.gl.bufferData(this.gl.ARRAY_BUFFER,new Float32Array(groups[i]),this.gl.STATIC_DRAW);}
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.contribPosBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,new Float32Array(contribXyz),this.gl.STATIC_DRAW);
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.contribLevelBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,new Float32Array(contribLevel),this.gl.STATIC_DRAW);
      this.contribCount=contribLevel.length;
      this.edgeCounts=groups.map(group=>group.length/6);this.focusSummary=this.connectionSummary(this.selection);this.incidentCount=this.focusSummary.slots;this.drawnEdges=edges.length;this.draw();return this;
    }
    /* Pure activity computation, kept free of GL calls so it can be unit-tested. Schema-2
       samples (with `stages`) light up L1/L2 from the exact recorded kernel trace; schema-1
       samples (no `stages`) fall back to the old motor-only behaviour: L1 stays grey and
       only the three L2 motor rows populate, from the legacy motor trace. */
    computeActivity(sample,being) {
      const activity=new Float32Array(this.n).fill(-2);
      if(!(sample&&String(being)===String(this.data.traceBeing)&&String(sample.being)===String(being)))return activity;
      for(let i=0;i<Math.min(192,sample.wave?.length||0);i++)activity[this.node('wave',i)]=Math.max(-1,Math.min(1,sample.wave[i]/127));
      for(let i=0;i<Math.min(96,sample.fast_state?.length||0);i++)activity[this.node('input CfC',i)]=Math.max(-1,Math.min(1,sample.fast_state[i]/2));
      const l1=sample.stages?.l1_effective;
      if(Array.isArray(l1)) {
        const scale=Math.max(1e-6,...l1.map(v=>Math.abs(v)));
        for(let i=0;i<Math.min(96,l1.length);i++)if(Number.isFinite(l1[i]))activity[this.node('L1',i)]=Math.max(-1,Math.min(1,l1[i]/scale));
      }
      const motor=sample.body?.motor_trace,raw=motor?.raw_intent||sample.body?.raw_motor,afterHead=motor?.after_head,slow=motor?.slow_output||sample.body?.slow;
      const l2=sample.stages?.l2_adapted;
      if(Array.isArray(l2)) {
        const scale=Math.max(1e-6,...l2.map(v=>Math.abs(v)));
        for(let i=0;i<Math.min(48,l2.length);i++)if(Number.isFinite(l2[i]))activity[this.node('L2',i)]=Math.max(-1,Math.min(1,l2[i]/scale));
      } else {
        for(let i=0;i<Math.min(3,raw?.length||0);i++)activity[this.node('L2',i)]=Math.max(-1,Math.min(1,raw[i]));
      }
      for(let i=0;i<Math.min(3,afterHead?.length||0);i++)activity[this.node('motor head',i)]=Math.max(-1,Math.min(1,afterHead[i]));
      for(let i=0;i<Math.min(3,slow?.length||0);i++)activity[this.node('slow output',i)]=Math.max(-1,Math.min(1,slow[i]));
      return activity;
    }
    setObservation(sample,being) {
      this.sample=sample;this.selectedBeing=being;this.activity=this.computeActivity(sample,being);
      const g=this.gl;g.bindBuffer(g.ARRAY_BUFFER,this.actBuffer);g.bufferSubData(g.ARRAY_BUFFER,0,this.activity);
      // Re-select the current node so incident() recomputes L1<->L2 edge contribution
      // (a*w) against the fresh sample; this also redraws.
      this.setSelection(this.selection);return this;
    }
    nodeInfo(index=this.selection) {
      const n=this.nodes[index],i=n.index,f=this.data.fast,s=this.data.slow;let detail='activation unavailable in this recording';
      if(n.layer==='wave')detail='recorded sensory code (i8 / 127 for colour)';
      else if(n.layer==='sensory KAN')detail=`fixed runtime identity prefilter · shared coefficients ${f.sensoryKnots.map(v=>Number(v).toFixed(1)).join(', ')} · output was not recorded`;
      else if(n.layer==='input CfC'){const rate=Number(f.tau[i]);detail=`recurrent sensory memory · decay rate ${rate.toPrecision(4)} s⁻¹ · time constant ${rate>0?(1/rate).toPrecision(4)+' s':'unavailable'}`;}
      else if(n.layer==='L1') {
        const value=this.recordedStage('l1_effective',i);
        const checkpoint=`checkpoint memory α ${Number(f.l1Alpha[i]).toPrecision(4)} · period ${Number(f.l1Period[i]).toPrecision(4)} ticks · threshold ${Number(f.l1Threshold[i]).toPrecision(4)} · activation ${this.activationName(f.l1Activation[i])}`;
        detail=value===null
          ? `activation unavailable in this recording (schema 1, or no fresh brain dispatch on this sample) · ${checkpoint}`
          : `recorded stage l1_effective = ${value.toPrecision(5)} (post-memory/threshold, pre-gate) · edges to L2: local contribution a·w, not causality · ${checkpoint}`;
      }
      else if(n.layer==='L2') {
        const value=this.recordedStage('l2_adapted',i);
        const role=i<3?`${OUTPUTS[i]} `:'';
        const checkpoint=`checkpoint memory α ${Number(f.l2Alpha[i]).toPrecision(4)} · activation ${this.activationName(f.l2Activation[i])}`;
        detail=value===null
          ? `${role}activation unavailable in this recording (schema 1, or no fresh brain dispatch on this sample) · ${checkpoint}`
          : `${role}recorded stage l2_adapted = ${value.toPrecision(5)} (post std/gain/fatigue, pre-activation) · edges from L1: local contribution a·w, not causality · ${checkpoint}`;
      }
      else if(n.layer==='motor head'){const mode=this.sample?.body?.motor_trace?.motor_head;detail=mode==='linear'?`${OUTPUTS[i]} linear clamp; spline knots inactive`:mode==='spline'?`${OUTPUTS[i]} ${f.outputKnotsSource || 'runtime spline'} · knots ${f.outputKnots[i].map(v=>Number(v).toFixed(2)).join(', ')}`:`${OUTPUTS[i]} head mode was not recorded; exported spline defaults are not assumed active`;}
      else if(n.layer==='slow input')detail=s.gatherMap[i]<0?'zero-filled: this source is absent in gpu_evolve':`declared source: wave channel ${s.gatherMap[i]}`;
      else if(n.layer==='slow CfC')detail='recurrent state was not recorded';
      else if(n.layer==='slow dense')detail=`ReLU hidden unit · bias ${Number(s.d1Bias[i]).toPrecision(4)}`;
      else if(n.layer==='slow output')detail=`${SLOW_OUTPUTS[i]} is recorded · bias ${Number(s.d2Bias[i]).toPrecision(4)}`;
      const measurement=this.measurement(index);return {layer:n.layer,index:i,detail,measurement};
    }
    activationName(value){return ['tanh','ReLU','linear','leaky ReLU'][Math.round(Number(value))]||`unknown id ${value}`;}
    measurement(index=this.selection) {
      const n=this.nodes[index],i=n.index,s=this.sample;if(!s||String(this.selectedBeing)!==String(this.data.traceBeing)||String(s.being)!==String(this.selectedBeing))return null;
      if(n.layer==='wave'&&Number.isFinite(s.wave?.[i]))return `${s.wave[i]} i8 code (${(s.wave[i]/127).toFixed(3)} normalized)`;
      if(n.layer==='input CfC'&&Number.isFinite(s.fast_state?.[i]))return `${Number(s.fast_state[i]).toPrecision(5)} dimensionless sensory memory`;
      if(n.layer==='L1') {
        const value=this.recordedStage('l1_effective',i);
        if(value!==null)return `${value.toPrecision(5)} dimensionless recorded stage l1_effective`;
      }
      const motor=s.body?.motor_trace,raw=motor?.raw_intent||s.body?.raw_motor,slow=motor?.slow_output||s.body?.slow;
      if(n.layer==='L2') {
        const value=this.recordedStage('l2_adapted',i);
        if(value!==null)return `${value.toPrecision(5)} dimensionless recorded stage l2_adapted${i<3?` (${OUTPUTS[i]} row)`:''}`;
        if(i<3&&Number.isFinite(raw?.[i]))return `${Number(raw[i]).toPrecision(5)} raw ${OUTPUTS[i]} intent (schema 1: no l2_adapted stage)`;
      }
      if(n.layer==='motor head'&&Number.isFinite(motor?.after_head?.[i]))return `${Number(motor.after_head[i]).toPrecision(5)} after ${motor.motor_head||'unidentified motor head'}`;
      if(n.layer==='slow output'&&Number.isFinite(slow?.[i]))return `${Number(slow[i]).toPrecision(5)} dimensionless ${SLOW_OUTPUTS[i]}`;
      return null;
    }
    pick(clientX,clientY) {
      const r=this.canvas.getBoundingClientRect(),x=clientX-r.left,y=clientY-r.top,aspect=r.width/r.height,cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);let best=-1,distance=81;
      for(let i=0;i<this.n;i++){const a=this.xyz[i*3],b=this.xyz[i*3+1],c=this.xyz[i*3+2],qx=cy*a-sy*c,qz=sy*a+cy*c,qy=cp*b-sp*qz,z=sp*b+cp*qz,k=4.5/(4.5-z),px=((qx*this.zoom*k+this.pan[0])/aspect*.5+.5)*r.width,py=((qy*this.zoom*k+this.pan[1])*.5+.5)*r.height,d=(px-x)**2+(py-y)**2;if(d<distance){best=i;distance=d;}}
      if(best>=0){this.setSelection(best);if(this.onSelection)this.onSelection(best);}return best;
    }
    rotatedPoint(index){const a=this.xyz[index*3],b=this.xyz[index*3+1],c=this.xyz[index*3+2],cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch),qx=cy*a-sy*c,qz=sy*a+cy*c,qy=cp*b-sp*qz,z=sp*b+cp*qz,k=4.5/(4.5-z);return {x:qx*k,y:qy*k,z};}
    projectPoint(index,width,height){const p=this.rotatedPoint(index),aspect=Math.max(width/Math.max(height,1),.01);return {x:(p.x*this.zoom+this.pan[0])/aspect,y:-(p.y*this.zoom+this.pan[1])};}
    fitCamera(width,height){const aspect=Math.max(width/Math.max(height,1),.01);let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(let i=0;i<this.n;i++){const p=this.rotatedPoint(i);minX=Math.min(minX,p.x);maxX=Math.max(maxX,p.x);minY=Math.min(minY,p.y);maxY=Math.max(maxY,p.y);}const dx=Math.max(maxX-minX,.001),dy=Math.max(maxY-minY,.001),margin=.78;this.zoom=Math.max(.1,Math.min(5,2*margin*aspect/dx,2*margin/dy));this.pan=[-(minX+maxX)*.5*this.zoom,-(minY+maxY)*.5*this.zoom];return this;}
    resetCamera(){this.yaw=-.12;this.pitch=-.12;const r=this.canvas.getBoundingClientRect();this.fitCamera(r.width,r.height);this.fitSize=`${Math.round(r.width)}x${Math.round(r.height)}`;this.draw();}
    resize(){const r=this.canvas.getBoundingClientRect(),ratio=Math.min(root.devicePixelRatio||1,2);if(r.width<1||r.height<1)return this;this.canvas.width=Math.round(r.width*ratio);this.canvas.height=Math.round(r.height*ratio);const size=`${Math.round(r.width)}x${Math.round(r.height)}`;if(size!==this.fitSize){this.fitCamera(r.width,r.height);this.fitSize=size;}return this.draw();}
    draw(){const g=this.gl,c=this.canvas;g.viewport(0,0,c.width,c.height);g.clearColor(.027,.038,.056,1);g.clear(g.COLOR_BUFFER_BIT|g.DEPTH_BUFFER_BIT);g.enable(g.DEPTH_TEST);g.enable(g.BLEND);g.blendFunc(g.SRC_ALPHA,g.ONE_MINUS_SRC_ALPHA);g.useProgram(this.program);g.uniform2f(this.loc.orbit,this.yaw,this.pitch);g.uniform2fv(this.loc.pan,this.pan);g.uniform1f(this.loc.zoom,this.zoom);g.uniform1f(this.loc.aspect,c.width/Math.max(c.height,1));g.uniform1f(this.loc.pointSize,Math.max(2,Math.min(5,c.height/150)));g.uniform1i(this.loc.selected,this.selection);g.uniform1i(this.loc.lines,0);g.bindBuffer(g.ARRAY_BUFFER,this.posBuffer);g.enableVertexAttribArray(this.positionLoc);g.vertexAttribPointer(this.positionLoc,3,g.FLOAT,false,0,0);g.bindBuffer(g.ARRAY_BUFFER,this.actBuffer);g.enableVertexAttribArray(this.activityLoc);g.vertexAttribPointer(this.activityLoc,1,g.FLOAT,false,0,0);g.drawArrays(g.POINTS,0,this.n);
      const colors=[[.30,.52,.86,.55],[.96,.45,.28,.55],[.68,.72,.77,.36]];g.uniform1i(this.loc.lines,1);g.uniform1i(this.loc.gradientLines,0);g.disableVertexAttribArray(this.activityLoc);g.vertexAttrib1f(this.activityLoc,-2);for(let i=0;i<3;i++){if(!this.edgeCounts?.[i])continue;g.uniform4fv(this.loc.lineColor,colors[i]);g.bindBuffer(g.ARRAY_BUFFER,this.edgeBuffers[i]);g.vertexAttribPointer(this.positionLoc,3,g.FLOAT,false,0,0);g.drawArrays(g.LINES,0,this.edgeCounts[i]*2);}
      if(this.contribCount) {
        // L1<->L2 synapses coloured by recorded local contribution a·w (see recordedStage/incident).
        g.uniform1i(this.loc.gradientLines,1);
        g.enableVertexAttribArray(this.activityLoc);
        g.bindBuffer(g.ARRAY_BUFFER,this.contribLevelBuffer);g.vertexAttribPointer(this.activityLoc,1,g.FLOAT,false,0,0);
        g.bindBuffer(g.ARRAY_BUFFER,this.contribPosBuffer);g.vertexAttribPointer(this.positionLoc,3,g.FLOAT,false,0,0);
        g.drawArrays(g.LINES,0,this.contribCount);
        g.uniform1i(this.loc.gradientLines,0);g.disableVertexAttribArray(this.activityLoc);g.vertexAttrib1f(this.activityLoc,-2);
      }
      return this;}
  }
  NativeBrain3D.layout=buildLayout;
  root.NativeBrain3D=NativeBrain3D;
})(window);
