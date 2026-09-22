// Feedback since the preceding neural decision; body slots94..97 are adapter-owned.
struct Params { n:u32, _a:u32, _b:u32, _c:u32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> body:array<f32>;
@group(0) @binding(2) var<storage,read_write> life:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let b=id.x;if(b>=p.n){return;}
 let s=b*128u;let l=b*512u+32u;
 var reward=0.0;
 if(body[s+97u]>0.5){
   let delivered=body[s+68u]-body[s+94u];
   let damage=max(0.0,body[s+95u]-body[s+8u]);
   let energy_change=(body[s]-body[s+96u])/max(1.0,body[s+18u]);
   reward=clamp(delivered-2.0*damage+0.2*max(0.0,energy_change)+0.01*min(0.0,energy_change),-1.0,1.0);
 }
 life[l]=reward;life[l+1u]=clamp(body[s]/max(1.0,body[s+18u]),0.0,1.0);
 life[l+2u]=body[s+29u];life[l+3u]=0.0;
 body[s+94u]=body[s+68u];body[s+95u]=body[s+8u];body[s+96u]=body[s];body[s+97u]=1.0;
}
