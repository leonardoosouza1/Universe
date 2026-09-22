// Explicit contact model city-v2: horizontal discs radius 1.5m, restitution .05.
// Mutual closest overlaps produce disjoint pairs: simultaneous impulses conserve
// pair momentum and dissipated energy. Multi-body piles resolve over later ticks.
struct Params{n:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> snapshot:array<f32>;
@group(0) @binding(2) var<storage,read> input:array<f32>;
@group(0) @binding(3) var<storage,read> body:array<f32>;
@group(0) @binding(4) var<storage,read_write> output:array<f32>;
@group(0) @binding(5) var<storage,read_write> nearest:array<u32>;
fn position(b:u32)->vec2<f32>{return vec2<f32>(snapshot[b*56u+36u],snapshot[b*56u+38u]);}
@compute @workgroup_size(64)
fn pairs(@builtin(global_invocation_id) gid:vec3<u32>){
 let b=gid.x;if(b>=p.n){return;}var candidate=p.n;var dist2=9.0;
 if(body[b*128u+29u]>0.5){
  let pos=position(b);
  for(var j=0u;j<p.n;j++){
   if(j==b || body[j*128u+29u]<=0.5){continue;}
   let delta=position(j)-pos;let d=dot(delta,delta);
   if(d<dist2 || (d==dist2 && j<candidate)){candidate=j;dist2=d;}
  }
 }
 nearest[b]=candidate;
}
@compute @workgroup_size(64)
fn contacts(@builtin(global_invocation_id) gid:vec3<u32>){
 let b=gid.x;if(b>=p.n){return;}let j=nearest[b];if(j>=p.n){return;}if(nearest[j]!=b){return;}
 let o=b*56u;let q=j*56u;let delta=position(j)-position(b);let d=length(delta);
 var normal=vec2<f32>(select(-1.0,1.0,b<j),0.0);if(d>0.000001){normal=delta/d;}
 let mass=input[b*92u+62u];let other_mass=input[j*92u+62u];
 if(mass<=0.0 || other_mass<=0.0){return;}
 let inv=1.0/mass;let inv_other=1.0/other_mass;let reduced=1.0/(inv+inv_other);
 let v=vec2<f32>(snapshot[o],snapshot[o+2u]);let other_v=vec2<f32>(snapshot[q],snapshot[q+2u]);
 let closing=max(0.0,dot(v-other_v,normal));let impulse=1.05*closing*reduced;
 let corrected=position(b)-normal*max(0.0,3.0-d)*(inv/(inv+inv_other));
 let next_v=v-normal*impulse*inv;
 output[o]=next_v.x;output[o+2u]=next_v.y;output[o+36u]=corrected.x;output[o+38u]=corrected.y;
 if(closing>0.0){
  output[o+39u]=bitcast<f32>(bitcast<u32>(snapshot[o+39u])|16u);
  output[o+43u]=max(snapshot[o+43u],closing);output[o+47u]=snapshot[o+47u]+impulse;
  output[o+55u]=snapshot[o+55u]+0.25*reduced*(1.0-0.05*0.05)*closing*closing;
 }
}
