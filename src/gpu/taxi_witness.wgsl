// Bounded reference pilot. It emits only the existing native intent ABI.
// Packed wave channels 176..191 are words 44..47, signed i8 lanes.
struct Params { n:u32, wave_words:u32, _a:u32, _b:u32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> wave:array<u32>;
@group(0) @binding(2) var<storage,read> physics:array<f32>;
@group(0) @binding(3) var<storage,read> body:array<f32>;
@group(0) @binding(4) var<storage,read> routes:array<f32>;
@group(0) @binding(5) var<storage,read> world:array<f32>;
@group(0) @binding(6) var<storage,read_write> intent:array<f32>;

const INTENT_STRIDE:u32=96u;
const PHYSICS_STRIDE:u32=92u;
const BODY_STRIDE:u32=128u;

fn signed_byte(word:u32, lane:u32)->f32 {
  let raw=(word>>(lane*8u))&255u;
  let s=i32(raw)-select(0,256,i32(raw)>127);
  return clamp(f32(s)/127.0,-1.0,1.0);
}
fn channel(b:u32, ch:u32)->f32 {
  let w=b*p.wave_words+ch/4u;
  return signed_byte(wave[w],ch%4u);
}
fn station_row(i:u32)->u32 { return 4u+i*16u; }
fn station_for_target(goal:vec2<f32>, count:u32)->u32 {
  var selected=count;
  var best=0.05;
  for(var j=0u;j<count;j++) {
    let w=station_row(j);
    let d=distance(goal,vec2<f32>(world[w],world[w+1u]));
    if(d<best){best=d;selected=j;}
  }
  return selected;
}
fn route_value(pos:vec2<f32>, target_station:u32, size:u32, cell:f32)->f32 {
  let g=vec2<i32>(floor(pos/cell));
  if(any(g<vec2<i32>(0)) || any(g>=vec2<i32>(i32(size)))) { return -1.0; }
  return routes[4u+target_station*size*size+u32(g.y)*size+u32(g.x)];
}
fn route_waypoint(pos:vec2<f32>, target_station:u32, size:u32, cell:f32)->vec2<f32> {
  let g=vec2<i32>(floor(pos/cell));
  var best=route_value(pos,target_station,size,cell);
  var waypoint=pos;
  for(var k=0u;k<4u;k++) {
    var delta=vec2<i32>(-1,0);
    if(k==0u){delta=vec2<i32>(0,1);}else if(k==1u){delta=vec2<i32>(0,-1);}else if(k==2u){delta=vec2<i32>(1,0);}
    let q=g+delta;
    if(any(q<vec2<i32>(0)) || any(q>=vec2<i32>(i32(size)))){continue;}
    let d=routes[4u+target_station*size*size+u32(q.y)*size+u32(q.x)];
    if(d>=0.0 && (best<0.0 || d<best)){best=d;waypoint=(vec2<f32>(q)+vec2<f32>(0.5))*cell;}
  }
  return waypoint;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
  let b=id.x;if(b>=p.n){return;}
  let ib=b*INTENT_STRIDE;let pb=b*PHYSICS_STRIDE;let sb=b*BODY_STRIDE;
  // The controller owns all intent slots for this explicit pilot, so stale
  // neural requests cannot leak into pickup/dropoff/service arbitration.
  for(var k=0u;k<INTENT_STRIDE;k++){intent[ib+k]=0.0;}
  // This witness is a one-delivery feasibility check. Once the production
  // taxi kernel records one paid delivery, hold the vehicle at zero command;
  // this prevents a newly issued passenger from turning the witness into an
  // unbounded route/reward loop.
  if(body[sb+68u]>=1.0){intent[ib+2u]=1.0;return;}

  // Body state is the authoritative current task. The packed wave remains the
  // interaction/traffic input, so the pilot does not use hidden neural state.
  let onboard=body[sb+65u];
  if(onboard<0.5 && body[sb+90u]<0.5){intent[ib+2u]=1.0;return;}
  let goal=select(vec2<f32>(body[sb+70u],body[sb+71u]),vec2<f32>(body[sb+66u],body[sb+67u]),onboard>0.5);
  let position=vec2<f32>(physics[pb+36u],physics[pb+38u]);
  let direct_dist=distance(position,goal);
  let route_size=u32(routes[0]);
  let route_cell=routes[1];
  let station_count=u32(routes[2]);
  let target_station=station_for_target(goal,station_count);
  var waypoint=goal;
  if(route_size>0u && target_station<station_count){
    let route_dist=route_value(position,target_station,route_size,route_cell);
    if(route_dist>=route_cell*1.5){waypoint=route_waypoint(position,target_station,route_size,route_cell);}
  }
  let delta=waypoint-position;
  let dist=max(length(delta),0.001);
  let lat_raw=vec2<f32>(physics[pb+44u],physics[pb+46u]);
  let lat=lat_raw/max(length(lat_raw),0.0001);
  let desired_dir=delta/dist;
  let steer=clamp(dot(desired_dir,lat)*1.8,-1.0,1.0);
  let speed=length(vec2<f32>(physics[pb],physics[pb+2u]));
  let rpm=physics[pb+49u];
  let max_rpm=max(1.0,physics[pb+50u]);
  let red_ahead=channel(b,172u)>0.5 && channel(b,182u)<-0.5;
  let red_dist=channel(b,173u)*100.0;
  let yellow_ahead=channel(b,172u)>0.5 && abs(channel(b,182u))<0.5;
  let at_task=direct_dist<=3.0;
  let desired_speed=select(select(5.0,3.0,yellow_ahead),1.0,red_ahead && red_dist<12.0);
  let throttle=select(clamp(0.16+0.05*dist,0.12,0.62),0.0,at_task || (red_ahead && red_dist<5.0));
  let brake=select(0.0,select(clamp((speed-desired_speed)/2.0,0.0,1.0),1.0,speed>0.25 && at_task),at_task || (red_ahead && red_dist<8.0) || speed>desired_speed+0.5);
  intent[ib]=select(throttle,0.0,brake>0.0);intent[ib+1u]=steer;intent[ib+2u]=brake;

  // The production gearbox consumes edge-triggered requests through rows 3/4.
  if(speed>3.5 && rpm>max_rpm*0.82){intent[ib+3u]=1.0;}
  if(speed>3.5 && rpm<max_rpm*0.30){intent[ib+4u]=1.0;}

  // Action 6/7/5 are the production taxi interaction requests. The taxi
  // kernel checks exact distance, low speed, stock, money and passenger ID.
  let stopped=speed<0.5;
  if(stopped && onboard<0.5 && channel(b,180u)>0.5 && at_task){intent[ib+6u]=1.0;}
  if(stopped && onboard>0.5 && at_task){intent[ib+7u]=1.0;}
  // Compare dimensionless fractions, never joules directly with kilograms.
  let low_resource=min(body[sb]/max(body[sb+18u],1.0),min(body[sb+1u]/max(body[sb+19u],0.001),body[sb+2u]/max(body[sb+20u],0.001)));
  let near_service=length(vec2<f32>(body[sb+85u],body[sb+86u]))<=3.0;
  if(stopped && onboard<0.5 && channel(b,189u)>0.5 && near_service && low_resource<0.25){intent[ib+5u]=1.0;}
}
