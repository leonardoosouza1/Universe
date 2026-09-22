// organism-city-observation-v2. Geometry only; never writes vehicle controls.
struct Params { n:u32, lights:u32, tick:u32, pad:u32 }
struct Light { position:vec2<f32>, approach:vec2<f32>, half_width:f32, offset:u32, green:u32, yellow:u32 }
struct State {
 start:vec2<f32>, initialized:u32, nearest:u32,
 distance:f32, phase:f32, time_left:f32, reserved:f32,
 red:u32, green:u32, yellow:u32, crossings:u32,
 last_light:u32, last_tick:u32, paid_fines:f32, pad1:u32,
}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> lights:array<Light>;
@group(0) @binding(2) var<storage,read_write> state:array<State>;
@group(0) @binding(3) var<storage,read> physical:array<f32>;
@group(0) @binding(4) var<storage,read_write> wave:array<u32>;
@group(0) @binding(5) var<storage,read_write> body:array<f32>;
fn phase_at(l:Light)->vec2<f32> {
 let half=l.green+l.yellow; let period=2u*half;
 let t=((p.tick%period)+(l.offset%period))%period;
 if(t<l.green){return vec2<f32>(1.0,f32(l.green-t)/f32(period));}
 if(t<half){return vec2<f32>(0.0,f32(half-t)/f32(period));}
 return vec2<f32>(-1.0,f32(period-t)/f32(period));
}
fn write_channel(b:u32,ch:u32,value:f32){
 let shift=(ch%4u)*8u;let idx=b*48u+ch/4u;
 let byte=u32(i32(round(clamp(value,-1.0,1.0)*127.0)))&255u;
 wave[idx]=(wave[idx]&~(255u<<shift))|(byte<<shift);
}
@compute @workgroup_size(64)
fn senses(@builtin(global_invocation_id) id:vec3<u32>){
 let b=id.x;if(b>=p.n){return;}let base=b*92u;
 let pos=vec2<f32>(physical[base+36u],physical[base+38u]);
 let heading=vec2<f32>(physical[base+40u],physical[base+42u]);
 state[b].start=pos;state[b].initialized=1u;
 var nearest=0u;var distance=100.0;var phase=vec2<f32>(0.0);
 for(var i=0u;i<p.lights;i++){
  let l=lights[i];let delta=l.position-pos;let d=dot(delta,l.approach);
  let side=abs(dot(delta,vec2<f32>(-l.approach.y,l.approach.x)));
  if(d>=0.0 && d<distance && side<=l.half_width && dot(heading,l.approach)>0.5){
   nearest=i+1u;distance=d;phase=phase_at(l);
  }
 }
 state[b].nearest=nearest;state[b].distance=distance;state[b].phase=phase.x;state[b].time_left=phase.y;
 write_channel(b,172u,select(0.0,1.0,nearest!=0u));
 write_channel(b,173u,distance/100.0);write_channel(b,182u,phase.x);write_channel(b,191u,phase.y);
}
@compute @workgroup_size(64)
fn post(@builtin(global_invocation_id) id:vec3<u32>){
 let b=id.x;if(b>=p.n || state[b].initialized==0u){return;}
 let base=b*56u;let end=vec2<f32>(physical[base+36u],physical[base+38u]);
 let start=state[b].start;
 for(var i=0u;i<p.lights;i++){
  let l=lights[i];let a=dot(start-l.position,l.approach);let z=dot(end-l.position,l.approach);
  if(a<0.0 && z>=0.0){
   let crossing=start+(end-start)*(-a/(z-a));
   if(abs(dot(crossing-l.position,vec2<f32>(-l.approach.y,l.approach.x)))<=l.half_width){
    let phase=phase_at(l).x;
    if(phase<0.0){
     state[b].red+=1u;
     // One event -> one cash-bounded payment. No debt, no pedal intervention.
     let fine=min(3.0,max(0.0,body[b*128u+64u]));
     body[b*128u+64u]-=fine;state[b].paid_fines+=fine;
    }else if(phase>0.0){state[b].green+=1u;}else{state[b].yellow+=1u;}
    state[b].crossings+=1u;state[b].last_light=i+1u;state[b].last_tick=p.tick;
   }
  }
 }
 // Updating start also makes repeated post dispatch idempotent at an unchanged pose.
 state[b].start=end;
}
