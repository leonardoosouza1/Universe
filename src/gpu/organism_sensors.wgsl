// organism-resident-extras-v1. temporal stride73; physical input stride92.
struct Params { n:u32, gs:u32, points:u32, line_offset:u32, horizons:u32, social:u32, continuous:u32, pad:u32, dt:f32, cell:f32, half_width:f32, total:f32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> physical:array<f32>;
@group(0) @binding(2) var<storage,read_write> body:array<f32>;
@group(0) @binding(3) var<storage,read> data:array<f32>;
@group(0) @binding(4) var<storage,read> digest:array<f32>;
@group(0) @binding(5) var<storage,read_write> temporal:array<f32>;
@group(0) @binding(6) var<storage,read_write> retina_pose:array<f32>;
@group(0) @binding(7) var<storage,read_write> perception_pose:array<f32>;
fn norm(v:vec2<f32>)->vec2<f32>{let l=length(v);return select(vec2<f32>(0.0),v/max(l,1e-30),l>0.0);}
// Rust f32::round uses halfway away from zero; WGSL round has different tie rules.
fn cell_round(v:vec2<f32>)->vec2<i32>{return vec2<i32>(sign(v)*floor(abs(v)+vec2<f32>(0.5)));}
fn voxel(c:vec2<i32>)->u32 {if(any(c<vec2<i32>(0))||any(c>=vec2<i32>(i32(p.gs)))){return 2u;}return u32(data[u32(c.y)*p.gs+u32(c.x)]);}
fn obstacle(c:vec2<i32>,holes:bool)->bool{let v=voxel(c);return v==2u||(holes&&v==3u);}
fn ray(pos:vec2<f32>,direction:vec2<f32>,horizon:f32,holes:bool)->f32{
 let magnitude=length(direction);if(magnitude<=1e-6){return horizon;}let d=direction/magnitude;
 var cell=cell_round(pos/p.cell);if(obstacle(cell,holes)){return 0.0;}
 var step=vec2<i32>(0);var next=vec2<f32>(1e30);var delta=vec2<f32>(1e30);
 for(var a=0u;a<2u;a++){if(abs(d[a])>1e-6){step[a]=select(-1,1,d[a]>0.0);let boundary=(f32(cell[a])+0.5*f32(step[a]))*p.cell;next[a]=max(0.0,(boundary-pos[a])/d[a]);delta[a]=p.cell/abs(d[a]);}}
 for(var k=0u;k<2u*p.gs+2u;k++){let crossing=min(next.x,next.y);if(crossing>horizon){return horizon;}for(var a=0u;a<2u;a++){if(next[a]==crossing){cell[a]+=step[a];next[a]+=delta[a];}}if(obstacle(cell,holes)){return crossing;}}
 return horizon;
}
fn coarse(pos:vec2<f32>,direction:vec2<f32>,front:bool)->f32{
 var distances=array<f32,10>(1.0,3.0,6.0,9.0,12.0,16.0,20.0,26.0,33.0,40.0);
 for(var k=0u;k<10u;k++){let d=select(distances[k],3.0*f32(k+1u),front);if(obstacle(cell_round((pos+direction*d)/p.cell),front)){return d;}}
 return select(40.0,50.0,front);
}
fn point(i:u32)->vec2<f32>{let o=p.line_offset+i*5u;return vec2<f32>(data[o],data[o+1u]);}
fn arc(i:u32)->f32{return data[p.line_offset+i*5u+2u];}
fn tangent(i:u32)->vec2<f32>{let o=p.line_offset+i*5u;return vec2<f32>(data[o+3u],data[o+4u]);}
fn wrap(v:f32)->f32{let len=max(p.total,0.0001);return v-floor(v/len)*len;}
fn curve(s:f32,heading:vec2<f32>,distance:f32)->f32{
 if(p.points<2u||p.total<=0.0001){return 0.0;}let target_arc=wrap(s+max(0.0,distance));var j=0u;
 for(var k=0u;k<p.points-1u;k++){if(arc(k)<=target_arc){j=k;}}
 let t=norm(point(j+1u)-point(j));let f=norm(heading);if(dot(t,t)<1e-6||dot(f,f)<1e-6){return 0.0;}
 return clamp(atan2(f.x*t.y-f.y*t.x,clamp(dot(f,t),-1.0,1.0))/1.5707963267948966,-1.0,1.0);
}
@group(0) @binding(8) var<storage,read> intent:array<f32>;
// Ordered sensor phases preserve the same equations. Driver/compiler diagnosis
// and the explicit pose-write workaround are recorded in the implementation ledger.
@compute @workgroup_size(64)
fn probes(@builtin(global_invocation_id) id:vec3<u32>){
 let b=id.x;if(b>=p.n){return;}let i=b*92u;let s=b*128u;let t=b*73u;let pose=b*8u;
 let pos=vec2<f32>(physical[i+36u],physical[i+38u]);let fwd=vec2<f32>(physical[i+40u],physical[i+42u]);let lat=vec2<f32>(physical[i+44u],physical[i+46u]);let velocity=vec2<f32>(physical[i],physical[i+2u]);let speed=length(velocity);
 var distances=vec3<f32>(40.0,40.0,50.0);let left=vec2<f32>(fwd.y,-fwd.x);
 if(p.continuous!=0u){distances=vec3<f32>(ray(pos,left,40.0,false),ray(pos,-left,40.0,false),ray(pos,fwd,50.0,true));}else{distances=vec3<f32>(coarse(pos,left,false),coarse(pos,-left,false),coarse(pos,fwd,true));}
 body[s+103u]=min(distances.x,min(distances.y,distances.z));
 body[s+124u]=distances.x;body[s+125u]=distances.y;body[s+126u]=distances.z;
 temporal[t+16u]=distances.x;temporal[t+17u]=distances.y;temporal[t+18u]=distances.z;
}

@compute @workgroup_size(64)
fn geometry(@builtin(global_invocation_id) id:vec3<u32>){
 let b=id.x;if(b>=p.n){return;}let i=b*92u;let s=b*128u;let t=b*73u;let pose=b*8u;
 let pos=vec2<f32>(physical[i+36u],physical[i+38u]);let fwd=vec2<f32>(physical[i+40u],physical[i+42u]);let lat=vec2<f32>(physical[i+44u],physical[i+46u]);let velocity=vec2<f32>(physical[i],physical[i+2u]);let speed=length(velocity);
 let distances=vec3<f32>(temporal[t+16u],temporal[t+17u],temporal[t+18u]);
 var risk=0.0;for(var k=10u;k<16u;k++){temporal[t+k]=0.0;}for(var k=66u;k<73u;k++){temporal[t+k]=0.0;}
 if(p.points>=2u){
  var best=1e30;var seg=0u;var fraction=0.0;var projection=point(0u);var heading=vec2<f32>(0.0);
  for(var k=0u;k<p.points-1u;k++){let a=point(k);let ab=point(k+1u)-a;let fraction_k=clamp(dot(pos-a,ab)/max(dot(ab,ab),1e-6),0.0,1.0);let q=a+ab*fraction_k;let d=dot(pos-q,pos-q);if(d<best){best=d;seg=k;fraction=fraction_k;projection=q;heading=norm(ab);}}
  let hw=max(p.half_width,1.0);let ss=arc(seg)+fraction*max(arc(seg+1u)-arc(seg),0.0001);
  temporal[t+10u]=clamp(dot(pos-projection,vec2<f32>(-heading.y,heading.x)),-p.half_width,p.half_width)/hw;
  let target_arc=wrap(ss+5.0);var j=0u;for(var k=0u;k<p.points-1u;k++){if(arc(k)<=target_arc&&target_arc<=arc(k+1u)){j=k;break;}}
  let u=clamp((target_arc-arc(j))/max(arc(j+1u)-arc(j),0.0001),0.0,1.0);let look=point(j)+(point(j+1u)-point(j))*u;let tan=tangent(j);
  temporal[t+12u]=clamp(dot(pos-look,vec2<f32>(-tan.y,tan.x)),-p.half_width,p.half_width)/hw;
  temporal[t+13u]=clamp(distances.z/(hw*3.0),0.0,1.0);temporal[t+14u]=clamp((distances.x+distances.y)/(hw*4.0),0.0,1.0);temporal[t+15u]=clamp(min(distances.x,distances.y)/(hw*1.5),0.0,1.0);
  let horizons=array<f32,7>(data[p.horizons+b*3u],data[p.horizons+b*3u+1u],data[p.horizons+b*3u+2u],350.0,200.0,275.0,260.0);
  for(var k=0u;k<7u;k++){temporal[t+66u+k]=curve(ss,heading,horizons[k]);}
  risk=clamp(speed*speed*abs(curve(ss,heading,30.0)*1.5707963267948966/30.0)/(0.9*9.81),0.0,1.0);
 }
 temporal[t+28u]=risk;
 // Explicit powertrain-v2 observation overlay, no racing-line reinterpretation
 // in legacy runs. Published channel manifest declares the changed semantics.
 if ((bitcast<u32>(physical[i+87u]) & 8192u)!=0u) {
   temporal[t+66u]=f32(bitcast<u32>(physical[i+57u]));
   temporal[t+67u]=clamp(physical[i+58u],0.0,1.0);
   temporal[t+68u]=clamp(physical[i+56u]/0.35,0.0,1.0);
 }
}

@compute @workgroup_size(64)
fn bridge(@builtin(global_invocation_id) id:vec3<u32>){
 let b=id.x;if(b>=p.n){return;}let i=b*92u;let s=b*128u;let t=b*73u;let pose=b*8u;
 let pos=vec2<f32>(physical[i+36u],physical[i+38u]);let fwd=vec2<f32>(physical[i+40u],physical[i+42u]);let lat=vec2<f32>(physical[i+44u],physical[i+46u]);let velocity=vec2<f32>(physical[i],physical[i+2u]);let speed=length(velocity);
 temporal[t+7u]=clamp(intent[b*96u],0.0,1.0);
 retina_pose[pose+0u]=pos.x;perception_pose[pose+0u]=pos.x;
 retina_pose[pose+1u]=physical[i+37u];perception_pose[pose+1u]=physical[i+37u];
 retina_pose[pose+2u]=pos.y;perception_pose[pose+2u]=pos.y;
 retina_pose[pose+3u]=0.0;perception_pose[pose+3u]=0.0;
 retina_pose[pose+4u]=fwd.x;perception_pose[pose+4u]=fwd.x;
 retina_pose[pose+5u]=fwd.y;perception_pose[pose+5u]=fwd.y;
 retina_pose[pose+6u]=lat.x;perception_pose[pose+6u]=lat.x;
 retina_pose[pose+7u]=lat.y;perception_pose[pose+7u]=lat.y;
 temporal[t+20u]=select(1.0,clamp(dot(velocity,fwd)/max(speed,0.000001),-1.0,1.0),speed>0.1);temporal[t+21u]=speed;temporal[t+22u]=dot(velocity,lat);
 // Scenario reserve is acquired integrity/alive, not the legacy fixed life budget.
 temporal[t+23u]=body[s+8u]*body[s+29u];temporal[t+24u]=clamp(physical[i+49u]/max(physical[i+50u],0.000001),0.0,1.0);temporal[t+25u]=f32(bitcast<u32>(physical[i+55u]));
 temporal[t+26u]=clamp((physical[i+8u]+physical[i+16u]+physical[i+24u]+physical[i+32u])*0.25,0.0,1.0);
 temporal[t+27u]=clamp((physical[i+60u]-20.0)/130.0,0.0,1.0);temporal[t+29u]=select(0.0,1.0,voxel(cell_round(pos/p.cell))==0u);temporal[t+30u]=clamp(abs(physical[i+53u]),0.0,1.0);
 temporal[t+31u]=select(0.0,min(1.0,temporal[t+31u]+p.dt),speed<0.5);
 // Actual reverse gear only. Slots33..48 are explicitly absent in native192 schema.
 temporal[t+32u]=select(0.0,1.0,bitcast<u32>(physical[i+55u])==0u);for(var k=33u;k<49u;k++){temporal[t+k]=0.0;}
 temporal[t+49u]=select(0.0,clamp((fwd.x*velocity.y-fwd.y*velocity.x)/max(speed,0.000001),-1.0,1.0),speed>1.0);
 // Previous-decision consequences; adapter feedback baseline94..97 is not mutated.
 var joy=0.0;if(body[s+97u]>0.5){joy=clamp(body[s+68u]-body[s+94u]-2.0*max(0.0,body[s+95u]-body[s+8u])+0.2*max(0.0,(body[s]-body[s+96u])/max(1.0,body[s+18u])),-1.0,1.0);}
 temporal[t+50u]=joy;temporal[t+51u]=select(0.0,min(1.0,temporal[t+51u]+p.dt/5.0),joy>0.0);temporal[t+52u]=select(0.0,1.0,abs(physical[i+3u])>1.5&&speed<2.0);
 var stress=0.0;if(p.social!=0u){for(var other=0u;other<p.n;other++){if(other==b){continue;}let o=other*92u;let rel=vec2<f32>(physical[o+36u],physical[o+38u])-pos;let d=length(rel);if(d<40.0){stress=max(stress,0.5);}if(d<20.0&&dot(rel,fwd)/max(d,0.01)>0.3){stress=1.0;}}}
 temporal[t+53u]=stress;temporal[t+54u]=digest[b*19u+1u];temporal[t+55u]=digest[b*19u+2u];temporal[t+56u]=clamp(physical[i+53u],-1.0,1.0);temporal[t+57u]=clamp(physical[i+52u],0.0,1.0);
 // Legacy BODY block is disabled in this schema; organism/taxi senses own160..191.
 // Keep source-compatible tyre/body temperatures if a consumer inspects temporal.
 for(var k=58u;k<66u;k++){temporal[t+k]=0.0;}temporal[t+61u]=body[s]/max(1.0,body[s+18u]);temporal[t+64u]=temporal[t+26u];temporal[t+65u]=temporal[t+27u];
}
