// Organism school judge (Stage 2, docs/SCHOOL-LESSONS-20260922.md).
// One invocation per being, scheduled after taxi.post and before the orientation
// feedback. It never writes a command; it reads the physics OUTPUT pose of this tick,
// the applied throttle from the physics INPUT and the body life state, then writes
// its own 32-float record and the task observation slots (65,70,71,80,81,90).
// Lesson 1 move, 2 stop, 3 turn_near, 4 turn_far. Status 0 running, 1 success,
// 2 failed, 3 truncated. Failure codes: 1 contact, 2 death, 3 corridor, 4 born
// inside goal, 5 overshoot, 6 drove through the junction, 7 wrong exit,
// 8 entered the stop area without ever reaching v_min.
// Params.flags: bit 0 bench (host-only), bit 1 stop-score shaping
// (docs/SCHOOL-STOP-SHAPING-20260922.md).
struct Params {
  n:u32, tick:u32, horizon:u32, flags:u32,
  goal_radius:f32, area_radius:f32, v_min:f32, v_stop:f32,
  hold_ticks:f32, corridor_half:f32, overshoot_along:f32, propelled_min:f32,
  box_half:f32, exit_overshoot:f32, align_min:f32, gate_radius:f32,
}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> physics_in:array<f32>;
@group(0) @binding(2) var<storage,read> physics_out:array<f32>;
@group(0) @binding(3) var<storage,read_write> body:array<f32>;
@group(0) @binding(4) var<storage,read_write> school:array<f32>;

const PI:u32=92u; const PO:u32=56u; const BS:u32=128u; const SS:u32=40u;
// School record slots (see src/gpu/school.rs for the host mirror).
const LESSON:u32=0u; const PHASE:u32=1u; const STATUS:u32=2u; const FAIL_CODE:u32=3u;
const P0X:u32=4u; const P0Z:u32=5u; const F0X:u32=6u; const F0Z:u32=7u;
const TX:u32=8u; const TZ:u32=9u; const EX:u32=10u; const EZ:u32=11u;
const EDX:u32=12u; const EDZ:u32=13u; const BEST_REMAINING:u32=14u; const DWELL:u32=15u;
const PROPELLED:u32=16u; const ELAPSED:u32=17u; const MAX_SPEED:u32=18u; const CONTACT_TICKS:u32=19u;
const COLLISION_J:u32=20u; const SCORE:u32=21u; const OUTCOME_TICK:u32=22u; const MAX_LAT:u32=23u;
const MAX_ALONG:u32=24u; const ENERGY_START:u32=25u; const DISTANCE_START:u32=26u; const INITIAL_REMAINING:u32=27u;
const PROGRESS:u32=28u; const ENTRY_SPEED:u32=29u; const WRONG_EX:u32=30u; const WRONG_EZ:u32=31u;
// Frozen with the outcome so consumption is reported for the judged span, not the whole episode.
const ENERGY_OUTCOME:u32=32u; const DISTANCE_OUTCOME:u32=33u;
// Stop shaping: slowest speed seen inside the area (1e9 until entered) and the
// longest held stop. Recorded always, scored only when flags bit 1 is set.
const MIN_SPEED_INSIDE:u32=34u; const MAX_DWELL:u32=35u;

fn close(k:u32, s:u32, status:f32, code:f32) {
  school[k+STATUS]=status; school[k+FAIL_CODE]=code; school[k+OUTCOME_TICK]=f32(p.tick);
  school[k+ENERGY_OUTCOME]=body[s]; school[k+DISTANCE_OUTCOME]=body[s+98u];
  let success=select(0.0,1.0,status==1.0);
  let failed=select(0.0,1.0,status==2.0);
  let integrity=clamp(body[s+8u],0.0,1.0);
  let alive=select(0.0,1.0,body[s+29u]>0.5);
  var shaping=0.0;
  if((p.flags&2u)!=0u && school[k+LESSON]>1.5 && school[k+LESSON]<2.5){
    shaping=300.0*clamp(1.0-school[k+MIN_SPEED_INSIDE]/max(p.v_min,0.001),0.0,1.0)
           +200.0*clamp(school[k+MAX_DWELL]/max(p.hold_ticks,1.0),0.0,1.0);
  }
  school[k+SCORE]=1000.0*success+500.0*school[k+PROGRESS]-200.0*failed-100.0*(1.0-integrity)-100.0*(1.0-alive)+shaping;
}

@compute @workgroup_size(64)
fn judge(@builtin(global_invocation_id) id:vec3<u32>) {
  let b=id.x; if(b>=p.n){return;}
  let i=b*PI; let o=b*PO; let s=b*BS; let k=b*SS;
  let lesson=school[k+LESSON];
  if(lesson<0.5){return;}
  let pos=vec2<f32>(physics_out[o+36u],physics_out[o+38u]);
  let p0=vec2<f32>(school[k+P0X],school[k+P0Z]);
  let f0=vec2<f32>(school[k+F0X],school[k+F0Z]);
  let l0=vec2<f32>(-f0.y,f0.x);
  let goal=vec2<f32>(school[k+TX],school[k+TZ]);
  let exit=vec2<f32>(school[k+EX],school[k+EZ]);
  let turning=lesson>=2.5;
  let in_box=abs(pos.x-goal.x)<=p.box_half && abs(pos.y-goal.y)<=p.box_half;
  // A turn's phase advances on the tick the box is entered, before the sub-goal is
  // published, so the observation switches to the exit gate on that same tick.
  if(turning && school[k+STATUS]<0.5 && school[k+PHASE]<0.5 && in_box && school[k+ELAPSED]>=0.5){school[k+PHASE]=1.0;}
  // The sub-goal is published every tick, before and after the outcome, so the
  // observation channels and the reference witness always see a defined task.
  var subgoal=goal;
  if(turning && school[k+PHASE]>=0.5){subgoal=exit;}
  body[s+65u]=0.0; body[s+70u]=subgoal.x; body[s+71u]=subgoal.y;
  body[s+80u]=subgoal.x-pos.x; body[s+81u]=subgoal.y-pos.y; body[s+90u]=1.0;
  if(school[k+STATUS]>0.5){return;}
  let speed=length(vec2<f32>(physics_out[o],physics_out[o+2u]));
  let heading_raw=vec2<f32>(physics_out[o+40u],physics_out[o+42u]);
  let heading=heading_raw/max(length(heading_raw),0.0001);
  let rel=pos-p0; let along=dot(rel,f0); let lat=dot(rel,l0);
  var remaining=distance(pos,goal);
  if(turning){
    if(school[k+PHASE]<0.5){remaining=distance(pos,goal)+distance(goal,exit);}
    else {remaining=distance(pos,exit);}
  }
  if(school[k+ELAPSED]<0.5){
    school[k+ENERGY_START]=body[s]; school[k+DISTANCE_START]=body[s+98u];
    school[k+INITIAL_REMAINING]=max(remaining,1.0); school[k+BEST_REMAINING]=remaining;
    school[k+MIN_SPEED_INSIDE]=1.0e9;
    let born_inside=select(distance(pos,goal)<=p.goal_radius,in_box,turning);
    if(born_inside){close(k,s,2.0,4.0);return;}
  }
  school[k+ELAPSED]+=1.0;
  school[k+MAX_SPEED]=max(school[k+MAX_SPEED],speed);
  school[k+MAX_LAT]=max(school[k+MAX_LAT],abs(lat));
  school[k+MAX_ALONG]=max(school[k+MAX_ALONG],along);
  if(physics_in[i+51u]>=0.1 && speed>=0.5){school[k+PROPELLED]+=1.0;}
  school[k+BEST_REMAINING]=min(school[k+BEST_REMAINING],remaining);
  school[k+PROGRESS]=clamp(1.0-school[k+BEST_REMAINING]/school[k+INITIAL_REMAINING],0.0,1.0);
  if(body[s+29u]<0.5){close(k,s,2.0,2.0);return;}
  let mask=u32(max(physics_out[o+39u],0.0));
  if(mask!=0u){
    school[k+CONTACT_TICKS]+=1.0; school[k+COLLISION_J]+=max(0.0,physics_out[o+55u]);
    close(k,s,2.0,1.0);return;
  }
  if(!turning){
    if(abs(lat)>p.corridor_half || along<-5.0){close(k,s,2.0,3.0);return;}
    if(along>p.overshoot_along){close(k,s,2.0,5.0);return;}
  }
  if(lesson<1.5){
    if(distance(pos,goal)<=p.goal_radius && school[k+PROPELLED]>=p.propelled_min){close(k,s,1.0,0.0);return;}
  } else if(lesson<2.5){
    let inside=distance(pos,goal)<=p.area_radius;
    if(school[k+PHASE]<0.5){
      if(speed>=p.v_min){school[k+PHASE]=1.0;}
      else if(inside){close(k,s,2.0,8.0);return;}
    }
    if(school[k+PHASE]>=0.5 && school[k+PHASE]<1.5 && inside){school[k+PHASE]=2.0; school[k+ENTRY_SPEED]=speed;}
    if(school[k+PHASE]>=1.5){
      if(inside && speed<=p.v_stop){school[k+DWELL]+=1.0;} else {school[k+DWELL]=0.0;}
      if(inside){school[k+MIN_SPEED_INSIDE]=min(school[k+MIN_SPEED_INSIDE],speed);}
      school[k+MAX_DWELL]=max(school[k+MAX_DWELL],school[k+DWELL]);
      if(school[k+DWELL]>=p.hold_ticks){close(k,s,1.0,0.0);return;}
    }
  } else {
    let along_junction=dot(goal-p0,f0);
    if(along>along_junction+p.exit_overshoot){close(k,s,2.0,6.0);return;}
    let wrong=vec2<f32>(school[k+WRONG_EX],school[k+WRONG_EZ]);
    if(distance(pos,wrong)<=p.gate_radius){close(k,s,2.0,7.0);return;}
    let e=vec2<f32>(school[k+EDX],school[k+EDZ]);
    if(school[k+PHASE]>=0.5 && distance(pos,exit)<=p.gate_radius && dot(heading,e)>=p.align_min){close(k,s,1.0,0.0);return;}
  }
  if(school[k+ELAPSED]>=f32(p.horizon)){close(k,s,3.0,0.0);}
}
