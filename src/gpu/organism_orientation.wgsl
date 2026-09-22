// Scenario-only post-C orientation integration. CarOutput stride=56 f32.
// Match the historical headless yaw convention: (x,z)->(x*cos-z*sin,x*sin+z*cos).
// Preserve angular velocity, vertical components, contact padding and all other fields.
struct Params { n:u32, dt:f32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(64)
fn rotate(@builtin(global_invocation_id) id:vec3<u32>) {
 if(id.x>=p.n){return;}
 let b=id.x*56u;
 let theta=output[b+3u]*p.dt;
 let st=sin(theta);let ct=cos(theta);
 for(var axis=0u;axis<2u;axis++) {
  let o=b+40u+axis*4u;
  let x=output[o];let z=output[o+2u];
  let rx=x*ct-z*st;let rz=x*st+z*ct;
  let magnitude=max(sqrt(rx*rx+rz*rz),0.000001);
  output[o]=rx/magnitude;output[o+2u]=rz/magnitude;
 }
}

@group(0) @binding(2) var<storage,read_write> input:array<f32>;
// Scatter only acquired fields, equivalent to physics.rs::apply_output. Preserve
// hereditary capacities, commands, gear, world coefficients, dt and ABI padding.
@compute @workgroup_size(64)
fn feedback(@builtin(global_invocation_id) id:vec3<u32>) {
 if(id.x>=p.n){return;}let o=id.x*56u;let i=id.x*92u;
 for(var k=0u;k<48u;k++){input[i+k]=output[o+k];}
 // Contact mask is numeric on both sides; a bitcast here produced 1065353216 for 1.0.
 input[i+39u]=output[o+39u];
 input[i+49u]=output[o+48u];input[i+54u]=output[o+49u];
 input[i+56u]=output[o+50u];input[i+57u]=output[o+51u];
 input[i+58u]=output[o+52u];input[i+60u]=output[o+53u];
 input[i+61u]=output[o+54u];input[i+89u]=output[o+55u];
}
