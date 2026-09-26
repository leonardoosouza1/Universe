// organism-taxi-v1: GPU-only task/services, after physical + body transition.
// Body state 128f32; fields0..63 belong to organism,64..127 to this adapter.
// Physics output56f32: vel0..2,pos36..38; native intent96f32.
// World header4f32=[next_id,issued_total,paid_total,reserved], rows16f32:
// x,z,Ecap,Wcap,Mcap,Eregen,Wregen,Mregen,Estock,Wstock,Mstock,
// waiting_passenger_id,destination_station,cooldown_s,reserved,reserved.
// Monetary prices:0.00001/J,1/kgwater,5/kgmaterial; rates100000J/s,1kg/s,.2kg/s.
// Monetary genesis25; deliveryfare10+.02*direct_distance_m; no energy creation by fares.
// mode=0 production, 1 isolated motor bench, 2 isolated service bench. Benches do
// not consume shared genesis stocks or receive production passengers; their judges
// own the task slots. The production taxi world keeps conservation.
struct Params { n:u32, stations:u32, tick:u32, wave_words:u32, dt:f32, observation:u32, mode:u32, _c:u32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> body:array<f32>;
@group(0) @binding(2) var<storage,read_write> world:array<f32>;
@group(0) @binding(3) var<storage,read> physics:array<f32>;
@group(0) @binding(4) var<storage,read> intent:array<f32>;
@group(0) @binding(5) var<storage,read_write> wave:array<u32>;
@group(0) @binding(6) var<storage,read> routes:array<f32>;
fn station(i:u32)->u32 { return 4u+16u*i; }
fn distance_to(pos:vec2<f32>, j:u32)->f32 { let w=station(j); return distance(pos,vec2<f32>(world[w],world[w+1u])); }
fn requested(b:u32, action:u32)->bool { return intent[b*96u+action]>0.5; }
// Cell convention shared with world.rs/organism_sensors.wgsl: centre i*cell, half away from zero.
fn cell_index(v:vec2<f32>)->vec2<i32> { return vec2<i32>(sign(v)*floor(abs(v)+vec2<f32>(0.5))); }

// Immutable map metric: no runtime host search or hidden steering assistance.
fn route_distance(pos:vec2<f32>, destination:vec2<f32>)->f32 {
  if(routes[0]==0.0){return distance(pos,destination);}
  let size=u32(routes[0]);let cell=routes[1];
  let grid=cell_index(pos/cell);
  if(any(grid<vec2<i32>(0)) || any(grid>=vec2<i32>(i32(size)))){return -1.0;}
  var target_station=p.stations;
  for(var j=0u;j<p.stations;j++){
    let w=station(j);
    if(distance(destination,vec2<f32>(world[w],world[w+1u]))<0.01){target_station=j;break;}
  }
  if(target_station==p.stations){return -1.0;}
  return routes[4u+target_station*size*size+u32(grid.y)*size+u32(grid.x)];
}

// Strict v4 route lookup. Unlike route_distance, this never falls back to a
// straight line: an absent field, off-road cell, unknown station, or
// unreachable cell is an explicit missing navigation observation.
fn route_station_for(destination:vec2<f32>)->u32 {
  for(var j=0u;j<p.stations;j++){
    let w=station(j);
    if(distance(destination,vec2<f32>(world[w],world[w+1u]))<0.01){return j;}
  }
  return p.stations;
}

fn finite_v4(value:f32)->bool {
  return value==value && abs(value)<1e30;
}

fn finite_vec2_v4(value:vec2<f32>)->bool {
  return finite_v4(value.x) && finite_v4(value.y);
}

// Returns body-frame direction cos/sin, compressed remaining road distance,
// and validity. The local descent is cardinal and road-field-only; at equal
// distance choices, the candidate most aligned with current forward wins, then
// the fixed cardinal order below supplies a stable final tie break. This is a
// one-cell lookahead (at most routes[1] metres), so it has no turn anticipation:
// a corner is selected when the next legal cell descends. The field is a
// cell-centre graph and does not model body footprint, wall clearance, or
// curvature/steering limits; v4 is a navigation hint, not a collision-free
// trajectory guarantee. A legal-road width of 5 cells at 3 m/cell leaves
// nominal room for the approximately 4.5 m body only when the pose stays
// centred, which this observation cannot enforce.
fn route_frame_v4(
  pos:vec2<f32>,
  destination_point:vec2<f32>,
  fwd:vec2<f32>,
  lat:vec2<f32>,
  speed:f32
)->vec4<f32> {
  let missing=vec4<f32>(0.0,0.0,1.0,0.0);
  if(!finite_v4(routes[0]) || routes[0]<=0.0){return missing;}
  let size=u32(routes[0]);let cell=routes[1];
  if(size==0u || !finite_v4(cell) || cell<=0.0
    || !finite_vec2_v4(pos) || !finite_vec2_v4(destination_point)
    || !finite_vec2_v4(fwd) || !finite_vec2_v4(lat)){return missing;}
  let grid=cell_index(pos/cell);
  if(any(grid<vec2<i32>(0)) || any(grid>=vec2<i32>(i32(size)))){return missing;}
  let target_station=route_station_for(destination_point);
  if(target_station==p.stations){return missing;}
  let stride=size*size;
  let base=4u+target_station*stride;
  let current=routes[base+u32(grid.y)*size+u32(grid.x)];
  if(!finite_v4(current) || current<0.0){return missing;}

  let target_delta=destination_point-pos;
  let target_distance=length(target_delta);
  let f=fwd/max(length(fwd),0.0001);
  let l=lat/max(length(lat),0.0001);
  let remaining=current/(current+100.0);
  // A station cell has no lower neighbour. Preserve the near-station point
  // direction so pickup/dropoff can settle on the exact interaction centre.
  if(target_distance<=4.0){
    if(target_distance<0.001){return vec4<f32>(0.0,0.0,remaining,1.0);}
    let direct=target_delta/target_distance;
    return vec4<f32>(dot(direct,f),dot(direct,l),remaining,1.0);
  }
  if(current<=0.0){return missing;}

  if(p.observation==5u){
    if(!finite_v4(speed)){return missing;}
    let preview_m=clamp(6.0+1.5*speed,18.0,48.0);
    let steps=min(16u,max(1u,u32(ceil(preview_m/cell))));
    var cursor=grid;
    var remaining_at_cursor=current;
    var heading=f;
    var travelled=0u;
    for(var step=0u;step<16u;step++){
      if(step>=steps || remaining_at_cursor<=0.0){break;}
      var best_offset=vec2<i32>(0,0);
      var best_distance=1e30;
      var best_alignment=-2.0;
      var found=false;
      for(var k=0u;k<4u;k++){
        var offset=vec2<i32>(1,0);
        switch k {
          case 1u: {offset=vec2<i32>(0,1);}
          case 2u: {offset=vec2<i32>(-1,0);}
          case 3u: {offset=vec2<i32>(0,-1);}
          default: {}
        }
        let next=cursor+offset;
        if(any(next<vec2<i32>(0)) || any(next>=vec2<i32>(i32(size)))){continue;}
        let d=routes[base+u32(next.y)*size+u32(next.x)];
        if(!finite_v4(d) || d<0.0 || d>=remaining_at_cursor){continue;}
        let alignment=dot(vec2<f32>(offset),heading);
        if(!found || d<best_distance || (d==best_distance && alignment>best_alignment)){
          found=true;best_offset=offset;best_distance=d;best_alignment=alignment;
        }
      }
      if(!found){return missing;}
      cursor+=best_offset;
      heading=vec2<f32>(best_offset);
      remaining_at_cursor=best_distance;
      travelled+=1u;
    }
    if(travelled==0u){return missing;}
    let delta=vec2<f32>(cursor)*cell-pos;
    let direction=delta/max(length(delta),0.0001);
    return vec4<f32>(dot(direction,f),dot(direction,l),remaining,1.0);
  }

  var best=vec2<f32>(0.0,0.0);
  var best_distance=1e30;
  var best_alignment=-2.0;
  var found=false;
  for(var k=0u;k<4u;k++){
    var offset=vec2<i32>(1,0);
    switch k {
      case 1u: {offset=vec2<i32>(0,1);}
      case 2u: {offset=vec2<i32>(-1,0);}
      case 3u: {offset=vec2<i32>(0,-1);}
      default: {}
    }
    let next=grid+offset;
    if(any(next<vec2<i32>(0)) || any(next>=vec2<i32>(i32(size)))){continue;}
    let candidate_distance=routes[base+u32(next.y)*size+u32(next.x)];
    if(!finite_v4(candidate_distance) || candidate_distance<0.0
      || candidate_distance>=current){continue;}
    let step=vec2<f32>(f32(offset.x),f32(offset.y));
    let alignment=dot(step,f);
    if(!found || candidate_distance<best_distance
      || (candidate_distance==best_distance && alignment>best_alignment)){
      found=true;
      best=step;
      best_distance=candidate_distance;
      best_alignment=alignment;
    }
  }
  if(!found){return missing;}
  let direction=best/max(length(best),0.0001);
  return vec4<f32>(dot(direction,f),dot(direction,l),remaining,1.0);
}

// Explicit genesis adapter, called once AFTER organism reset, never on ordinaryticks.
// Desired birth reserves are transferred from station slot%station_count. Body
// construction first consumes0.01kg per kg of expressed baseline1350kg*massscale.
// Episodic worldreset creates the declared environmental stock; it is not an
// evolutionary gain or a per-tick replenishment. Marker93 prevents double funding.
@compute @workgroup_size(1)
fn birth() {
  if(p.stations<2u){return;}
  for(var b=0u;b<p.n;b++){
    let s=b*128u;
    if(body[s+93u]!=0.0){continue;}
    let desired=vec3<f32>(body[s],body[s+1u],body[s+2u]);
    body[s]=0.0;body[s+1u]=0.0;body[s+2u]=0.0;
    body[s+91u]=0.0;body[s+92u]=0.0;body[s+93u]=1.0;
    body[s+64u]=0.0;body[s+84u]=1.0;
    if(p.mode!=0u){
      body[s]=desired.x;body[s+1u]=desired.y;body[s+2u]=desired.z;
      body[s+92u]=desired.x;body[s+64u]=25.0;
      continue;
    }
    let w=station(b%p.stations);
    let construction=0.01*1350.0*body[s+24u];
    if(body[s+29u]<0.5 || world[w+10u]<construction || world[w+8u]<=0.0 || world[w+9u]<=0.0){
      body[s+29u]=0.0;body[s+69u]=7.0;continue;
    }
    world[w+10u]-=construction;body[s+91u]=construction;
    let energy=min(desired.x,world[w+8u]);
    let water=min(desired.y,world[w+9u]);
    let material=min(desired.z,world[w+10u]);
    world[w+8u]-=energy;world[w+9u]-=water;world[w+10u]-=material;
    body[s]=energy;body[s+1u]=water;body[s+2u]=material;
    body[s+92u]=energy;body[s+64u]=25.0;
  }
}

// Single invocation resolves shared stocks, IDs and payments without floatatomics.
// Rotation changesfirstbeing eachtick; resource priority withinonebeing is E,W,M.
@compute @workgroup_size(1)
fn post() {
  if(p.n==0u || p.stations<2u){return;}
  for(var j=0u;j<p.stations;j++) {
    let w=station(j);
    for(var r=0u;r<3u;r++){world[w+8u+r]=min(world[w+2u+r],world[w+8u+r]+p.dt*world[w+5u+r]);}
    world[w+13u]=max(0.0,world[w+13u]-p.dt);
    if(p.mode==0u && world[w+11u]==0.0 && world[w+13u]==0.0 && world[0]<16777216.0){
      let id=u32(world[0]); world[w+11u]=f32(id); world[0]+=1.0; world[1]+=1.0;
      world[w+12u]=f32((j+1u+(id%(p.stations-1u)))%p.stations);
    }
  }
  for(var turn=0u;turn<p.n;turn++) {
    let b=(turn+p.tick%p.n)%p.n; let s=b*128u; let o=b*56u;
    if(body[s+84u]==0.0){body[s+64u]=25.0;body[s+84u]=1.0;}
    body[s+69u]=0.0;
    // Task progress is a bounded fitness signal, never an action/controller
    // input. It is recomputed from the current pose each tick, so it cannot
    // accumulate or be farmed by oscillation.
    body[s+127u]=0.0;
    for(var r=0u;r<4u;r++){body[s+76u+r]=0.0;}
    let pos=vec2<f32>(physics[o+36u],physics[o+38u]);
    let speed=length(vec2<f32>(physics[o],physics[o+2u]));
    var near=0u; var near_dist=distance_to(pos,0u);
    var waiting=0u; var waiting_dist=1e30; var has_waiting=false;
    for(var j=0u;j<p.stations;j++) {
      let d=distance_to(pos,j);
      if(d<near_dist){near=j;near_dist=d;}
      if(world[station(j)+11u]>0.0 && d<waiting_dist){waiting=j;waiting_dist=d;has_waiting=true;}
    }
    if(p.mode==2u){
      has_waiting=body[s+90u]>0.5;
      waiting_dist=distance(pos,vec2<f32>(body[s+70u],body[s+71u]));
    }
    let w=station(near); body[s+82u]=f32(near);
    body[s+83u]=select(0.0,1.0,near_dist<=4.0 && (world[w+8u]>0.0 || world[w+9u]>0.0 || world[w+10u]>0.0));
    body[s+85u]=world[w]-pos.x; body[s+86u]=world[w+1u]-pos.y;
    body[s+87u]=world[w+8u]/max(1.0,world[w+2u]);
    body[s+88u]=world[w+9u]/max(1.0,world[w+3u]);
    body[s+89u]=world[w+10u]/max(1.0,world[w+4u]);
    // A school lesson measures one controller in a fixed environment. Letting
    // mutated classmates consume these shared stocks changes the elite's sensor
    // channel 187 and breaks exact re-evaluation even with an unchanged genome.
    if(p.mode==1u){continue;}
    if(p.mode==0u){
      if(body[s+65u]<0.5 && has_waiting){let q=station(waiting);body[s+70u]=world[q];body[s+71u]=world[q+1u];}
      body[s+90u]=select(0.0,1.0,has_waiting);
    }
    if(body[s+29u]<0.5){body[s+69u]=6.0;continue;}
    let any_request=requested(b,5u)||requested(b,6u)||requested(b,7u);
    if(any_request && speed>=0.5){body[s+69u]=1.0;}
    if(p.mode==0u && requested(b,5u) && speed<0.5){
      if(near_dist>4.0){body[s+69u]=2.0;}else{
        var received=0.0;
        for(var r=0u;r<3u;r++){
          var price=0.00001;var rate=100000.0;
          if(r==1u){price=1.0;rate=1.0;}
          if(r==2u){price=5.0;rate=0.2;}
          let deficit=max(0.0,body[s+18u+r]-body[s+r]);
          let amount=min(min(deficit,world[w+8u+r]),min(rate*p.dt,body[s+64u]/price));
          let cost=min(body[s+64u],amount*price);
          body[s+r]+=amount; world[w+8u+r]-=amount; body[s+64u]-=cost;
          body[s+76u+r]=amount; body[s+79u]+=cost;received+=amount;
        }
        if(received==0.0){body[s+69u]=select(4.0,3.0,body[s+64u]<=0.0);}
      }
    }
    if(requested(b,6u) && speed<0.5 && body[s+65u]<0.5){
      if(!has_waiting){body[s+69u]=5.0;}
      else if(waiting_dist>4.0){body[s+69u]=2.0;}
      else{
        body[s+65u]=1.0;body[s+74u]=0.0;body[s+90u]=0.0;
        if(p.mode==0u){
          let q=station(waiting);let dst=station(u32(world[q+12u]));
          body[s+72u]=world[q+11u];
          body[s+66u]=world[dst];body[s+67u]=world[dst+1u];
          body[s+70u]=world[q];body[s+71u]=world[q+1u];
          body[s+73u]=10.0+0.02*distance(vec2<f32>(world[q],world[q+1u]),vec2<f32>(world[dst],world[dst+1u]));
          world[q+11u]=0.0;world[q+13u]=30.0;
        }
      }
    }
    if(body[s+65u]>0.5){
      body[s+74u]+=p.dt;
      if(requested(b,7u) && speed<0.5){
        if(distance(pos,vec2<f32>(body[s+66u],body[s+67u]))>4.0){body[s+69u]=2.0;}else{
          if(body[s+72u]!=body[s+75u]){
            body[s+64u]+=body[s+73u];body[s+68u]+=1.0;body[s+75u]=body[s+72u];
            if(p.mode==0u){world[2]+=body[s+73u];}
          }
          body[s+65u]=0.0;body[s+72u]=0.0;
        }
      }
    }
    let task_position=select(vec2<f32>(body[s+70u],body[s+71u]),vec2<f32>(body[s+66u],body[s+67u]),body[s+65u]>0.5);
    body[s+80u]=task_position.x-pos.x;body[s+81u]=task_position.y-pos.y;
    // Task progress (127) is phase-aware with best-distance memory in body slots 31
    // (key), 61 (best remaining, m) and 62 (initial remaining, m). Phase A: approaching
    // the nearest waiting passenger pays 0..100 for road distance closed since that
    // passenger became the destination_point. Phase B: carrying pays 100 plus 0..100 toward the
    // destination. key = passenger_id*2 + phase; any change resets the memory at the
    // current distance, so a swapped destination_point, a return trip or oscillation earns nothing
    // new, a destination_point already within 8 m earns no approach credit, and an unreachable
    // start (route -1) earns none for that key. Dead beings skip this block above.
    var key=0.0; var goal=vec2<f32>(0.0,0.0);
    if(body[s+65u]>0.5){key=body[s+72u]*2.0+1.0;goal=vec2<f32>(body[s+66u],body[s+67u]);}
    else if(has_waiting){
      if(p.mode==2u){key=body[s+72u]*2.0;goal=vec2<f32>(body[s+70u],body[s+71u]);}
      else{let q=station(waiting);key=world[q+11u]*2.0;goal=vec2<f32>(world[q],world[q+1u]);}
    }
    if(key==0.0){body[s+31u]=0.0;body[s+61u]=0.0;body[s+62u]=0.0;body[s+127u]=0.0;}
    else{
      let remaining=route_distance(pos,goal);
      if(body[s+31u]!=key){body[s+31u]=key;body[s+61u]=remaining;body[s+62u]=remaining;}
      if(remaining>=0.0 && (body[s+61u]<0.0 || remaining<body[s+61u])){body[s+61u]=remaining;}
      var approach=0.0;
      if(body[s+62u]>=8.0 && body[s+61u]>=0.0){approach=100.0*clamp(1.0-body[s+61u]/body[s+62u],0.0,1.0);}
      body[s+127u]=select(0.0,100.0,body[s+65u]>0.5)+approach;
    }
  }
}
fn pack4(a:f32,b:f32,c:f32,d:f32)->u32 {
  return (u32(i32(round(clamp(a,-1.0,1.0)*127.0)))&255u)
    |((u32(i32(round(clamp(b,-1.0,1.0)*127.0)))&255u)<<8u)
    |((u32(i32(round(clamp(c,-1.0,1.0)*127.0)))&255u)<<16u)
    |((u32(i32(round(clamp(d,-1.0,1.0)*127.0)))&255u)<<24u);
}
// Explicit body channels160..175, same observation schema, all sampled previouspoststate:
// 160 energy/energycapacity,161 water/watercapacity,162 material/materialcapacity,
// 163 (temperature_K-293.15)/130K,164 fatigue,165 enginewear,166 tyrewear,167integrity,
// 168 maturity,169 age/7200s,170 effectiveengine/birthengine,171 effectivebrake/birthbrake,
// 172 repairedenginewear_fraction,173 repairedtyrewear_fraction,174 repairresource_scale[0,1],175 alive.
// Fractions/codes are explicit feature normalizations, then signed i8 quantization
// saturates only the observation. Canonical raw state is never overwritten here.
// Explicit scenario-only observation schema: taxi channels176..191 replacelegacy horizon.
// v2 (p.observation==2): 176 money/100,177 onboard,178 targetdx/100m,179 targetdz/100m,
// 180 requestavailable,181 taskage/300s,182 deliveries/100,183 refusal/6,
// 184 servicedx/100m,185 servicedz/100m,186 E stockfraction,187 Wstockfraction,
// 188 Mstockfraction,189 serviceavailable,190 fare/100,191 alive.
// v3 (p.observation==3): 178/179 = cos/sin of the task direction in the body frame
// (forward, lateral axes of the physics pose), 183 = d/(d+100) (1 when no task);
// 184/185/186 the same for the nearest service; 187 min station stock fraction;
// 188 refusal/6; 190 fare/100. 182 and 191 belong to traffic.wgsl in both versions.
// v4 (p.observation==4): 178/179 = body-frame direction of the local legal-road
// route descent (near a station, the exact station-point direction), 183 =
// remaining legal-road distance/(distance+100). Missing route field, off-road
// position, unreachable destination_point, unknown destination_point, and no task encode direction 0
// and distance 1. Other v3 channels retain their meanings.
// Senses read the physics INPUT record (stride 92: pos36/38, fwd40/42, lat44/46),
// which holds the initial pose at tick 0 and the fed-back pose afterwards.
fn body_frame(delta:vec2<f32>, fwd:vec2<f32>, lat:vec2<f32>)->vec3<f32> {
  let d=length(delta);
  if(d<0.001){return vec3<f32>(0.0,0.0,0.0);}
  let dir=delta/d;
  let f=fwd/max(length(fwd),0.0001);let l=lat/max(length(lat),0.0001);
  return vec3<f32>(dot(dir,f),dot(dir,l),d/(d+100.0));
}
@compute @workgroup_size(64)
fn senses(@builtin(global_invocation_id) gid:vec3<u32>){
 let b=gid.x;if(b>=p.n){return;}let s=b*128u;let words=b*p.wave_words;
 wave[words+40u]=pack4(body[s]/max(body[s+18u],0.000001),body[s+1u]/max(body[s+19u],0.000001),body[s+2u]/max(body[s+20u],0.000001),(body[s+9u]-293.15)/130.0);
 wave[words+41u]=pack4(body[s+5u],body[s+6u],body[s+7u],body[s+8u]);
 wave[words+42u]=pack4(body[s+4u],body[s+3u]/7200.0,body[s+32u]/max(body[s+16u],0.000001),body[s+33u]/max(body[s+17u],0.000001));
 wave[words+43u]=pack4(body[s+42u],body[s+43u],body[s+46u],body[s+29u]);
 let base=words+44u;
 if(p.observation==3u || p.observation==4u || p.observation==5u){
  let i=b*92u;
  let fwd=vec2<f32>(physics[i+40u],physics[i+42u]);
  let lat=vec2<f32>(physics[i+44u],physics[i+46u]);
  let has_task=body[s+65u]>0.5 || body[s+90u]>0.5;
  var task=vec3<f32>(0.0,0.0,1.0);
  if(has_task){task=body_frame(vec2<f32>(body[s+80u],body[s+81u]),fwd,lat);}
  let service=body_frame(vec2<f32>(body[s+85u],body[s+86u]),fwd,lat);
  let stock=min(body[s+87u],min(body[s+88u],body[s+89u]));
  wave[base]=pack4(body[s+64u]/100.0,body[s+65u],task.x,task.y);
  wave[base+1u]=pack4(body[s+90u],body[s+74u]/300.0,0.0,task.z);
  wave[base+2u]=pack4(service.x,service.y,service.z,stock);
  wave[base+3u]=pack4(body[s+69u]/6.0,body[s+83u],body[s+73u]/100.0,0.0);
  return;
 }
 wave[base]=pack4(body[s+64u]/100.0,body[s+65u],body[s+80u]/100.0,body[s+81u]/100.0);
 wave[base+1u]=pack4(body[s+90u],body[s+74u]/300.0,body[s+68u]/100.0,body[s+69u]/6.0);
 wave[base+2u]=pack4(body[s+85u]/100.0,body[s+86u]/100.0,body[s+87u],body[s+88u]);
 wave[base+3u]=pack4(body[s+89u],body[s+83u],body[s+73u]/100.0,body[s+29u]);
}

// A separate route pass keeps route search out of the dense body/service packer.
// On RADV NAVI22, compiling both in a single senses entry caused a driver crash.
// Sequential GPU passes preserve every non-route byte without host readback.
@compute @workgroup_size(64)
fn route_senses(@builtin(global_invocation_id) gid:vec3<u32>){
 let b=gid.x;if(b>=p.n){return;}let s=b*128u;let i=b*92u;
 var task=vec4<f32>(0.0,0.0,1.0,0.0);
 if(body[s+65u]>0.5 || body[s+90u]>0.5){
  let destination_point=select(vec2<f32>(body[s+70u],body[s+71u]),vec2<f32>(body[s+66u],body[s+67u]),body[s+65u]>0.5);
  task=route_frame_v4(vec2<f32>(physics[i+36u],physics[i+38u]),destination_point,
    vec2<f32>(physics[i+40u],physics[i+42u]),vec2<f32>(physics[i+44u],physics[i+46u]),
    length(vec2<f32>(physics[i],physics[i+2u])));
 }
 let base=b*p.wave_words+44u;
 wave[base]=(wave[base]&0x0000ffffu)|(pack4(0.0,0.0,task.x,task.y)&0xffff0000u);
 wave[base+1u]=(wave[base+1u]&0x00ffffffu)|(pack4(0.0,0.0,0.0,task.z)&0xff000000u);
}
