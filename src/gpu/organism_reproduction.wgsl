// Single-invocation transaction. No host resource arithmetic writes resident state.
struct Params { parent:u32, child:u32, n:u32, pad:u32, energy:f32, cost:f32, reserve:f32, pad_water:f32 };
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> body:array<f32>;
@group(0) @binding(2) var<storage,read> newborn:array<f32>;
// status, parent E/W/M before, parent E/W/M after, child E/W/M,
// construction kg, dissipated J, discarded dead-slot E/W/M, reserved.
@group(0) @binding(3) var<storage,read_write> receipt:array<f32>;
@compute @workgroup_size(1)
fn reproduce() {
 for(var i=0u;i<16u;i++){receipt[i]=0.0;}
 if(p.parent>=p.n || p.child>=p.n || p.parent==p.child){return;}
 let a=p.parent*128u; let b=p.child*128u;
 let construction=13.5*newborn[24];
 let material=0.1*newborn[20];
 // Paid viable birth: exceed the body's strict 20%-capacity growth gate.
 let water=0.25*newborn[19];
 if(body[a+29u]!=1.0 || body[b+29u]!=0.0 || body[a+4u]<0.5){return;}
 if(p.energy<=0.0 || p.energy>newborn[18] || water<=0.0 || water>newborn[19]){return;}
 if(body[a]<p.energy+p.cost+p.reserve || body[a+1u]<water+1.0 || body[a+2u]<material+construction){return;}
 for(var i=0u;i<3u;i++){receipt[1u+i]=body[a+i];receipt[12u+i]=body[b+i];}
 body[a]-=p.energy+p.cost; body[a+1u]-=water; body[a+2u]-=material+construction;
 for(var i=0u;i<128u;i++){body[b+i]=newborn[i];}
 body[b]=p.energy;body[b+1u]=water;body[b+2u]=material;
 // Reproduction never mints genesis currency or re-runs station funding.
 body[b+64u]=0.0;body[b+84u]=1.0;body[b+91u]=construction;body[b+92u]=p.energy;body[b+93u]=1.0;
 for(var i=0u;i<3u;i++){receipt[4u+i]=body[a+i];receipt[7u+i]=body[b+i];}
 receipt[10]=construction;receipt[11]=p.cost;receipt[0]=1.0;
}
