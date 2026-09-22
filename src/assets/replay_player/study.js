/* One clock drives the body, sampled vision and recorded neural activity. */
(function (root) {
  'use strict';
  const $ = id => document.getElementById(id);
  const text = (id, value) => { $(id).textContent = value; };
  const latest = (rows, tick) => {
    let lo=0, hi=rows.length;
    while(lo<hi) { const m=(lo+hi)>>1; if(rows[m].tick<=tick) lo=m+1; else hi=m; }
    return lo ? rows[lo-1] : null;
  };
  // These IDs are recorded material IDs, not RGB classes.
  const materials = {0:'Grass',1:'Road',2:'Wall',3:'Hole',4:'Being'};
  const palette = {0:[76,144,111],1:[160,181,194],2:[240,115,98],3:[225,188,80],4:[180,124,219]};
  class ReplayStudy {
    constructor(data,state,selectedId,tick) {
      this.data=data; this.state=state; this.selectedId=selectedId; this.tick=tick;
      this.diagnostics=root.DRIVING_DATA; this.nativePayload=root.REPLAY_NATIVE_BRAIN; this.payload=data.meta.brain_model==='native'?null:root.REPLAY_BRAIN_3D; this.brain=null; this.sample=null; this.lastSample=null;
      this.projection='angular'; this.hoverRay=null; this.tab='vision'; this.rows=new Map(); this.windows=new Map(); this.generation=data.meta.generation ?? 0;
      for(const row of this.diagnostics?.observations || []) {
        if(row.generation !== this.generation) continue;
        const id=String(row.being); if(!this.rows.has(id)) this.rows.set(id,[]); this.rows.get(id).push(row);
      }
      for(const row of this.diagnostics?.windows || []) {
        if(row.generation !== this.generation) continue;
        const id=String(row.being); if(!this.windows.has(id)) this.windows.set(id,[]); this.windows.get(id).push(row);
      }
      for(const rows of this.rows.values()) rows.sort((a,b)=>a.tick-b.tick);
      document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>this.setView(button.dataset.view)));
      document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{
        this.tab=button.dataset.tab;
        $('next-impact').hidden=this.tab!=='behavior';
        document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
        for(const name of ['vision','behavior','signals']) $(`dock-${name}`).hidden=name!==this.tab;
        this.update();
      }));
      $('next-impact').addEventListener('click',()=>{
        const rows=(this.windows.get(String(this.selectedId()))||[]).filter(r=>r.impact_ticks>0);
        const next=rows.find(r=>r.last_tick>this.tick())||rows[0];
        if(!next){text('sample-stamp','No impacts recorded for this being');return;}
        if(this.state.playing)$('play').click();
        this.state.index=Math.max(0,this.data.ticks.findIndex(t=>t>=next.last_tick));
        root.__UNIVERSE_REPLAY__.render();
      });
      $('reset-brain').addEventListener('click',()=>this.brain?.resetCamera());
      $('select-neuron').addEventListener('click',()=>{
        if(!this.brain)return;
        const local=Number($('neuron-index').value), offset=this.nativePayload?this.brain.offsets[$('native-layer').value]||0:0;
        this.brain.setSelection(offset+local); this.updateBrain();
      });
      $('neuron-index').addEventListener('keydown',event=>{if(event.key==='Enter')$('select-neuron').click();});
      $('native-layer').addEventListener('change',()=>{
        if(!this.nativePayload||!this.brain)return;
        const layer=$('native-layer').value,count=this.brain.nodes.filter(node=>node.layer===layer).length;
        $('neuron-index').max=Math.max(0,count-1);$('neuron-index').value=0;this.brain.setSelection(this.brain.offsets[layer]);this.updateBrain();
      });
      $('jump-telemetry-being').addEventListener('click',()=>{
        const target=this.diagnostics?.meta?.selected_being;if(!Number.isInteger(target))return;
        const index=this.data.beings.findIndex((being,i)=>String(root.UniverseReplayModel.beingId(being,i))===String(target));
        if(index>=0)root.__UNIVERSE_REPLAY__?.selectBeing(index);
      });
      this.matches=[]; this.matchIndex=0;
      $('find-neuron').addEventListener('click',()=>{
        if(!this.brain||!this.payload) return;
        const query=$('neuron-query').value.trim().toLowerCase();
        this.matches=[]; this.matchIndex=0;
        if(query) this.payload.meta.forEach((node,i)=>{ if(node.rootId===query || (node.type||'').toLowerCase().includes(query)) this.matches.push(i); });
        if(this.matches.length) { this.brain.setSelection(this.matches[0]); this.updateBrain(); }
        text('neuron-matches',`${this.matches.length} matches`);
      });
      $('neuron-query').addEventListener('keydown',e=>{if(e.key==='Enter')$('find-neuron').click();});
      $('next-neuron').addEventListener('click',()=>{if(this.matches.length) {this.matchIndex=(this.matchIndex+1)%this.matches.length;this.brain.setSelection(this.matches[this.matchIndex]);this.updateBrain();text('neuron-matches',`${this.matchIndex+1}/${this.matches.length}`);}});
      $('vision-projection').addEventListener('change',()=>{
        this.projection=$('vision-projection').value; this.clearHover(); this.drawRetina(); this.updateVision(this.diagnostics?.meta);
        root.__UNIVERSE_REPLAY__?.drawScene();
      });
      $('retina-full').addEventListener('pointermove',event=>{
        if(!this.sample || !this.diagnostics?.meta) return;
        const m=this.diagnostics.meta, rect=event.target.getBoundingClientRect();
        const col=Math.max(0,Math.min(m.retina_cols-1,Math.floor((event.clientX-rect.left)/rect.width*m.retina_cols)));
        const row=Math.max(0,Math.min(m.retina_rows-1,Math.floor((event.clientY-rect.top)/rect.height*m.retina_rows)));
        const sourceCol=this.sourceColumn(row,col);
        if(sourceCol===null) {
          this.clearHover('This angle was not sampled; no pixel is invented.');
          root.__UNIVERSE_REPLAY__?.drawScene(); return;
        }
        const packed=this.sample.retina[row*m.retina_cols+sourceCol]>>>0;
        this.hoverRay={row,col:sourceCol,tick:this.sample.tick,being:this.sample.being};
        const pitch=m.retina_pitch_radians?.[row];
        const angle=this.retinaAngle(m,sourceCol,row);
        const azimuth=angle===null?'angle unavailable':`${angle.degrees.toFixed(2)}° ${angle.reference}`;
        const hit=this.hitWorld(row,sourceCol), terrain=this.data.meta.terrain;
        let mapNote='';
        if(hit&&terrain&&hit.material===3) {
          const cell=terrain.cell_size_m, index=v=>Math.sign(v)*Math.floor(Math.abs(v/cell)+.5);
          const x=index(hit.x), z=index(hit.z), matches=terrain.hole_cells.some(c=>c[0]===x&&c[1]===z);
          mapNote=` · map cell ${x},${z}: ${matches?'recorded hole':'not a recorded hole'}`;
        }
        text('retina-hover',`Row ${row} · sensor column ${sourceCol} · ${materials[packed&255]||`material ${packed&255}`} · ${((packed>>>8)/1000).toFixed(3)} m · ${azimuth}${Number.isFinite(pitch)?` · pitch ${pitch.toFixed(4)} rad`:''}${mapNote}`);
        root.__UNIVERSE_REPLAY__?.drawScene();
      });
      this.setView(this.payload||this.nativePayload?'split':'track');
      const m=this.diagnostics?.meta;
      this.updateVision(m);
      if(this.nativePayload) {
        $('fly-neuron-search').hidden=true;
        $('native-topology-summary').hidden=false;
        text('brain3d-kind','Native functional topology');
        text('brain3d-name',`Native controller · ${this.nativePayload.checkpoint}`);
        $('brain3d').title='Schematic functional layout. Positions are not anatomy; focused lines are exact stored weights or declared fixed mappings.';
        requestAnimationFrame(()=>setTimeout(()=>{
          try {
            this.brain=new root.NativeBrain3D($('brain3d'),this.nativePayload);
            this.brain.onSelection=()=>this.updateBrain();
            $('brain3d-message').hidden=true;this.updateBrain();
          } catch(error){text('brain3d-message',`Native 3D rendering failed: ${error.message}`);this.error=error.message;}
        },0));
      } else if(!this.payload) {
        $('native-layer-wrap').hidden=true;
        text('brain3d-message','No native controller checkpoint or anatomical fly trace is attached to this replay. This recording uses '+(data.meta.brain_model||'an unspecified controller')+'.');
        text('brain3d-name','Controller topology unavailable');
      } else {
        $('native-layer-wrap').hidden=true;
        text('brain3d-kind','Anatomical connectome');
        text('brain3d-name',`${this.payload.n.toLocaleString()} neurons · FlyWire`);
        $('neuron-index').max=this.payload.n-1;
        // Let loading feedback paint before preparing the complete adjacency index.
        requestAnimationFrame(()=>setTimeout(()=>{
          try {
            this.brain=new root.Brain3D($('brain3d'),this.payload);
            this.brain.onSelection=()=>this.updateBrain();
            $('brain3d-message').hidden=true;
            this.updateBrain();
          } catch(error) { text('brain3d-message',`3D rendering failed: ${error.message}`); this.error=error.message; }
        },0));
      }
    }
    clearHover(message='Hover a pixel to inspect its material, distance, azimuth and pitch.') {
      this.hoverRay=null; text('retina-hover',message);
    }
    retinaAngle(m,col,row) {
      const fov=Number.isFinite(m?.fov_degrees)?m.fov_degrees:null;
      if(fov===null || !m.retina_cols) return null;
      const u=m.retina_cols===1?0:col/(m.retina_cols-1);
      let angleRad=(u-.5)*(fov*Math.PI/180);
      if(row===0) angleRad=Math.tan((2*u-1)*1.25)/Math.tan(1.25)*(fov*Math.PI/360);
      const gaze=this.sample?.sensor_pose?.gaze_yaw;
      if(row!==0 || Number.isFinite(gaze)) return {degrees:(angleRad+(row===0?gaze:0))*180/Math.PI,reference:'body-relative azimuth'};
      return {degrees:angleRad*180/Math.PI,reference:'gaze-relative angle (gaze unrecorded)'};
    }
    hitWorld(row,col) {
      const pose=this.sample?.sensor_pose, m=this.diagnostics?.meta, pitch=m?.retina_pitch_radians?.[row];
      if(!pose || !Number.isFinite(pose.gaze_yaw) || !Number.isFinite(pitch)) return null;
      const angle=this.retinaAngle(m,col,row); if(!angle) return null;
      const yaw=angle.degrees*Math.PI/180, p=this.sample.retina[row*m.retina_cols+col]>>>0;
      const distance=(p>>>8)/1000, horizontal=distance*Math.cos(pitch);
      return {x:pose.position[0]+(pose.forward[0]*Math.cos(yaw)+pose.right[0]*Math.sin(yaw))*horizontal,
        z:pose.position[2]+(pose.forward[1]*Math.cos(yaw)+pose.right[1]*Math.sin(yaw))*horizontal,
        material:p&255,distance};
    }
    projectionMode() {
      return this.projection==='angular'&&Number.isFinite(this.sample?.sensor_pose?.gaze_yaw)?'angular':'raw';
    }
    sourceColumn(row,col) {
      return root.ReplayVisionProjection.sourceColumn(this.diagnostics?.meta,this.sample?.sensor_pose,row,col,this.projectionMode());
    }
    rowZeroRange(sample,m) {
      if(!sample?.retina || !m?.retina_cols) return null;
      let max=0; for(let col=0;col<m.retina_cols;col++) max=Math.max(max,(sample.retina[col]>>>8)/1000);
      return max;
    }
    updateVision(m) {
      if(!m) {
        text('vision-resolution','Retina not recorded in this replay');
        text('vision-explanation','Use a recording with driving diagnostics to inspect actual vision. Empty cells are not sensor readings.');
        return;
      }
      const actual=this.rowZeroRange(this.sample,m);
      const pitch=m.retina_pitch_radians?.[0], offset=m.eye_offset_m??m.eye_height_m;
      const bodyHeight=this.sample?.sensor_pose?.position?.[1];
      const height=Number.isFinite(offset)&&Number.isFinite(bodyHeight)?offset+bodyHeight:null;
      const horizon=Number.isFinite(pitch)&&pitch<0&&height!==null?height/-Math.tan(pitch):null;
      const projection=this.projectionMode();
      $('vision-projection').value=projection;
      $('vision-projection').options[0].disabled=!Number.isFinite(this.sample?.sensor_pose?.gaze_yaw);
      text('vision-axis',projection==='angular'?'−60° · 0° · +60°':'Column index · row 0 is foveated');
      const range=`${Number.isFinite(horizon)?` · actual ground horizon ≈ ${horizon.toFixed(2)} m`:''}${Number.isFinite(actual)?` · row 0 farthest return ${actual.toFixed(2)} m`:''}`;
      text('vision-resolution',`${m.retina_rows} rows × ${m.retina_cols} columns · ${m.retina_rows*m.retina_cols} rays${m.fov_degrees ? ` · ${m.fov_degrees}° FOV` : ''}${range}`);
      const geometry=m.retina_geometry==='physical-v1'?'Physical ground surfaces: yellow marks a sampled hole.':'Legacy geometry: holes can appear as vertical obstacles. Re-exporting cannot correct these recorded rays.';
      const projectionNote=projection==='angular'?'Aligned angles use the nearest recorded ray; neural inputs are unchanged.':'Raw columns have different angles on row 0; vertically aligned pixels do not represent the same direction.';
      text('vision-explanation',`${projectionNote} ${geometry} Eye offset and body height both determine the horizon. No lane-marking channel is present.`);
    }
    setView(mode) {
      $('study-workspace').dataset.view=mode;
      document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===mode)));
      requestAnimationFrame(()=>this.brain?.resize());
    }
    updateBrain() {
      if(!this.brain) return;
      const id=this.selectedId(), tick=this.tick();
      if(this.nativePayload) {
        const observed=this.nativePayload.observedGenomes?.[this.sample?.genome_file];
        if(observed && this.brain.data!==observed) {
          this.brain.data=observed;
          text('brain3d-name',`Native controller · ${this.sample.genome_file}`);
        }
        this.brain.setObservation(this.sample,id);
        const info=this.brain.nodeInfo();$('native-layer').value=info.layer;$('neuron-index').value=info.index;
        $('neuron-index').max=this.brain.nodes.filter(node=>node.layer===info.layer).length-1;
        text('neuron-detail',`${info.layer} ${info.index} · ${info.detail}`);
        const edges=this.brain.focusSummary;
        text('brain-connection-count',`${info.layer} ${info.index} only · ${edges.incoming} incoming / ${edges.outgoing} outgoing · ${edges.zero} zero weights hidden${edges.invalid?` · ${edges.invalid} invalid weights omitted`:''}`);
        const measured=info.measurement?`Measured: ${info.measurement}`:'Activation unavailable for this unit/sample';
        const motor=this.sample?.body?.motor_trace,motorNote=motor?` · mapper ${motor.source} · head ${motor.motor_head} · throttle arbitration ${motor.throttle_arbitration||'unrecorded'} · override ${motor.override_cause} · brake ${motor.brake_suppression}`:'';
        text('brain3d-state',`${measured}${motorNote}. Grey = unrecorded activation.`);
        this.updateTelemetryLink(id);
        return;
      }
      this.brain.setFrameByTick(tick,id);
      const n=this.brain.selection, node=this.payload.meta[n]||{}, f=this.payload.frames[this.brain.frame];
      text('brain-connection-count',`Neuron ${n} only · ${this.brain.drawnEdges.toLocaleString()} / ${this.brain.incidentCount.toLocaleString()} incident connections drawn`);
      $('neuron-index').value=n;
      const sampleTick=f?(f.tick??f.step/this.payload.traceStepsPerTick-1):null;
      const voltage=f?.v?.[n], spikes=f?.spikes?.[n];
      text('neuron-detail',`#${n} · root ${node.rootId} · ${node.type||'untyped'} · ${node.flow||'unclassified'} · ${this.brain.drawnEdges.toLocaleString()}/${this.brain.incidentCount.toLocaleString()} incident connections drawn`);
      const measured=f&&Number.isFinite(voltage)&&Number.isFinite(spikes)?`sample tick ${sampleTick} · age ${tick-sampleTick} ticks · ${voltage.toFixed(2)} mV · ${spikes} spikes`:`No activation for being ${id} at this tick; anatomy only`;
      text('brain3d-state',`${measured} · ${this.brain.visibleCount.toLocaleString()}/${this.payload.n.toLocaleString()} located neurons · ${this.payload.m.toLocaleString()} indexed edges`);
      $('brain3d').title='Anatomical soma/annotated positions; straight lines show connectivity, not reconstructed neurite paths. Missing coordinates: '+(this.payload.n-this.brain.visibleCount);
      this.updateTelemetryLink(id);
    }
    updateTelemetryLink(id) {
      const detailed=this.diagnostics?.meta?.selected_being, button=$('jump-telemetry-being');
      if(!Number.isInteger(detailed)) {text('telemetry-being','Detailed neural telemetry was not recorded. Selecting a being only changes the inspection target; it does not identify a winner.');button.hidden=true;return;}
      const associated=this.nativePayload?.traceBeing, relation=Number.isInteger(associated)?` The verified checkpoint topology is associated with Being ${associated} by the export argument.`:'';
      text('telemetry-being',`Detailed telemetry exists for Being ${detailed} only.${relation} Selection does not mean best or champion.`);
      button.hidden=String(id)===String(detailed);
    }
    drawRetina() {
      const c=$('retina-full'), ctx=c.getContext('2d'), m=this.diagnostics?.meta;
      if(!this.sample || !m) {ctx.clearRect(0,0,c.width,c.height); return;}
      c.width=m.retina_cols; c.height=m.retina_rows; const image=ctx.createImageData(c.width,c.height);
      for(let i=0;i<this.sample.retina.length;i++) {
        const row=Math.floor(i/m.retina_cols), col=this.sourceColumn(row,i%m.retina_cols);
        if(col===null) {image.data.set([23,29,36,255],i*4);continue;}
        const p=this.sample.retina[row*m.retina_cols+col]>>>0, distance=(p>>>8)/1000, color=palette[p&255]||[215,188,96];
        const gain=.30+.70*Math.exp(-distance/50);
        for(let k=0;k<3;k++) image.data[i*4+k]=Math.round(color[k]*gain);
        image.data[i*4+3]=255;
      }
      ctx.putImageData(image,0,0);
      c.title='Categorical material and encoded distance in metres; brightness indicates proximity, not RGB colour.';
    }
    drawSignals() {
      for(const [id,values,limit] of [['sensory-wave',this.sample?.wave,127],['fast-state',this.sample?.fast_state,1]]) {
        const c=$(id), ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
        if(!values) continue;
        const scale=limit===127?127:Math.max(.001,...values.map(Math.abs));
        values.forEach((v,i)=>{const level=Math.max(-1,Math.min(1,v/scale)); ctx.fillStyle=level>=0?'#d5b87a':'#7699c6'; ctx.fillRect(i*c.width/values.length,c.height/2,Math.max(1,c.width/values.length-1),-level*c.height*.48);});
      }
      text('signal-note',this.sample?.fast_state?'Native sensory CfC memory before L1, dimensionless; scale adapts to the sample. This is not the fly connectome.':'Native sensory CfC memory before L1 is unavailable for this sample.');
    }
    drawBehavior() {
      const rows=this.windows.get(String(this.selectedId()))||[], through=rows.filter(r=>r.last_tick<=this.tick());
      const c=$('behavior-chart'), rect=c.getBoundingClientRect(); c.width=Math.max(1,Math.round(rect.width)); c.height=Math.max(1,Math.round(rect.height));
      const ctx=c.getContext('2d'); ctx.fillStyle='#0b1017'; ctx.fillRect(0,0,c.width,c.height);
      const span=Math.max(1,rows.at(-1)?.last_tick||1), maxSpeed=Math.max(1,...rows.map(r=>r.mean_speed_mps));
      ctx.fillStyle='#dd70562b';
      for(const row of rows) if(row.impact_ticks>0) ctx.fillRect(row.first_tick/span*c.width,0,Math.max(1,(row.last_tick-row.first_tick+1)/span*c.width),c.height);
      for(const [key,color,scale] of [['mean_speed_mps','#9ab7da',maxSpeed],['mean_throttle','#cdb37d',1],['mean_brake','#e07d70',1]]) {
        ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.beginPath();
        rows.forEach((r,i)=>{const x=r.last_tick/span*c.width,y=c.height-12-r[key]/scale*(c.height-24); if(!i)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
      }
      ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(this.tick()/span*c.width,0);ctx.lineTo(this.tick()/span*c.width,c.height);ctx.stroke();
      const sum=key=>through.reduce((s,r)=>s+(r[key]||0),0), alive=sum('alive_ticks');
      const host=$('contact-stats');host.replaceChildren();
      for(const [label,val] of [['Contact',alive?(sum('contact_ticks')/alive*100).toFixed(2)+'%':'—'],['Impact ticks',sum('impact_ticks')],['Braking',alive?(sum('brake_ticks')/alive*100).toFixed(2)+'%':'—'],['Stopped',alive?(sum('stationary_ticks')/alive*100).toFixed(2)+'%':'—']]) {
        const box=document.createElement('div'),name=document.createElement('span'),value=document.createElement('strong');name.textContent=label;value.textContent=val;box.append(name,value);host.append(box);
      }
      const note=document.createElement('p');note.className='section-note';note.textContent=rows.length?`Alive ticks through last complete window: ${alive}. Blue: speed (0–${maxSpeed.toFixed(1)} m/s); gold: throttle; red: brake (0–1). Red bands: impact windows. Contacts are physical; proximity is separate.`:'No physical contact diagnostics attached.';host.append(note);
    }
    updateLifetimeTrace() {
      const host=$('lifetime-trace');if(!host)return;host.replaceChildren();
      const values=this.sample?.lifetime_state,flags=this.data.meta.lifetime_flags||0;
      if(!flags || !Array.isArray(values) || values.length!==512)return;
      const title=document.createElement('p');title.className='motor-trace-caption';
      title.textContent=`Lifetime adaptation · sample tick ${this.sample.tick} · ${flags&1?'plasticity ':''}${flags&2?'development ':''}${flags&4?'topology ':''}`;host.append(title);
      const table=document.createElement('table'),body=document.createElement('tbody');
      const maxAbs=(from,to)=>Math.max(0,...values.slice(from,to).map(Math.abs));
      for(const [label,value] of [
        ['Developmental age',values[375]],['Previous transition feedback',values[32]],
        ['Fuel fraction',values[33]],['Largest acquired weight',maxAbs(36,180)],
        ['Largest eligibility trace',maxAbs(180,324)],['Lifetime dispatches',values[380]],
        ['Plastic throttle residual',values[384]],['Plastic steering residual',values[385]],['Plastic brake residual',values[386]],
        ['Tank capacity',this.sample.body?.fuel_capacity],['Engine temperature °C',this.sample.body?.engine_temperature_c]
      ]) {const row=document.createElement('tr'),name=document.createElement('th'),cell=document.createElement('td');name.textContent=label;cell.textContent=Number.isFinite(value)?value.toFixed(5):'Unavailable';row.append(name,cell);body.append(row);}
      table.append(body);host.append(table);
      const note=document.createElement('p');note.className='section-note';note.textContent='Inherited topology and acquired motor weights are separate. The 3D graph shows inherited connections; this table reports measured resident lifetime state. Acquired weights reset at birth.';host.append(note);
    }
    updateMotorTrace() {
      this.updateLifetimeTrace();
      const host=$('motor-stage-trace'),motor=this.sample?.body?.motor_trace;host.replaceChildren();
      if(!motor){const note=document.createElement('p');note.className='section-note';note.textContent='Per-stage motor trace was not recorded for this being/sample.';host.append(note);return;}
      const caption=document.createElement('p'),speed=Number.isFinite(motor.pre_physics_speed_mps)?`${motor.pre_physics_speed_mps.toFixed(3)} m/s`:'speed unavailable';caption.className='motor-trace-caption';caption.textContent=`${motor.phase} · ${speed} · ${motor.motor_head} head · brake ${motor.brake_suppression}`;host.append(caption);
      const table=document.createElement('table'),head=document.createElement('thead'),headRow=document.createElement('tr');
      for(const label of ['Stage','Throttle','Steer','Brake']){const cell=document.createElement('th');cell.scope='col';cell.textContent=label;headRow.append(cell);}head.append(headRow);table.append(head);
      const body=document.createElement('tbody');
      for(const [label,key] of [['Raw intent','raw_intent'],['After head','after_head'],['After modulation','after_modulation'],['Final physics command','final_command']]){const row=document.createElement('tr'),name=document.createElement('th');name.scope='row';name.textContent=label;row.append(name);for(let i=0;i<3;i++){const cell=document.createElement('td'),value=motor[key]?.[i];cell.textContent=Number.isFinite(value)?Number(value).toFixed(3):'—';row.append(cell);}body.append(row);}table.append(body);host.append(table);
    }
    update() {
      const rows=this.rows.get(String(this.selectedId()))||[]; this.sample=latest(rows,this.tick());
      text('sample-stamp',this.sample?`Being ${this.sample.being} · sample ${this.sample.tick} · age ${this.tick()-this.sample.tick} ticks · pre-physics vision`:'No sensory sample for this being / tick');
      this.updateVision(this.diagnostics?.meta);
      if(this.sample!==this.lastSample) {this.clearHover();this.drawRetina();this.drawSignals();this.lastSample=this.sample;}
      if(this.tab==='behavior')this.drawBehavior();
      this.updateBrain();
      this.updateMotorTrace();
      const motor=this.sample?.body?.motor_trace;
      if(motor)text('action-note',`Recorded ${motor.phase}: raw intent → ${motor.motor_head} head → slow/hip modulation → steering filter and brake gating → final physics command. Throttle arbitration: ${motor.throttle_arbitration||'unrecorded'}; final override: ${motor.override_cause}; brake suppression: ${motor.brake_suppression}.`);
    }
  }
  root.ReplayStudy=ReplayStudy;
})(window);
