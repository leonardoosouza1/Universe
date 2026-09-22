// Feedback since the preceding neural decision; body slots 94..97 are adapter-owned.
// mode 0 (legacy): an energy gain pays 0.2 per unit of capacity and a loss 0.01, so a
// loss q followed by a recovery q nets +0.19 q -- a refuel cycle is rewarded
// (META/BRAINSTORM-ESCOLA-EVOLUCAO-UNIVERSE-2026-09-22.md section 4).
// mode 1 (symmetric): energy pays 0.01 per unit in both directions and money paid for
// services is a cost (0.01 per genesis purse of 25); a loss/recovery cycle nets <= 0.
// The money baseline lives in lifetime slot LIFE_AUX_MONEY (the body has no spare
// adapter slot). It is zero at birth, so the first decision sees the genesis purse as
// a gain, which the cost-only term ignores.
// mode 2 (progress): mode 1 plus 0.01 per point of bounded task progress (body 127)
// gained since the previous decision on the SAME task key (body 31). Progress only
// rises when the best road distance to the phase goal improves (taxi.wgsl), so the
// term pays for approach and for the pickup phase change, never for oscillation,
// parking or odometer; a key change (new target, drop-off) pays nothing. Without this
// term a life with no delivery and no damage sees rewards of ~1e-7 (energy per tick
// over capacity) and the plastic P stays at ~1e-9 (docs/SCHOOL-INCENTIVES-20260922.md).
struct Params { n:u32, mode:u32, _b:u32, _c:u32 }
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> body:array<f32>;
@group(0) @binding(2) var<storage,read_write> life:array<f32>;
const LIFE_AUX_MONEY:u32=400u;
const LIFE_AUX_PROGRESS:u32=401u;
const LIFE_AUX_TASK_KEY:u32=402u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let b=id.x;if(b>=p.n){return;}
 let s=b*128u;let l=b*512u+32u;let m=b*512u+LIFE_AUX_MONEY;
 let g=b*512u+LIFE_AUX_PROGRESS;let k=b*512u+LIFE_AUX_TASK_KEY;
 var reward=0.0;
 if(body[s+97u]>0.5){
   let delivered=body[s+68u]-body[s+94u];
   let damage=max(0.0,body[s+95u]-body[s+8u]);
   let energy_change=(body[s]-body[s+96u])/max(1.0,body[s+18u]);
   if(p.mode>=1u){
     let money_change=(body[s+64u]-life[m])/25.0;
     var progress=0.0;
     if(p.mode==2u && body[s+31u]!=0.0 && body[s+31u]==life[k]){progress=max(0.0,body[s+127u]-life[g]);}
     reward=clamp(delivered-2.0*damage+0.01*energy_change+0.01*min(0.0,money_change)+0.01*progress,-1.0,1.0);
   } else {
     reward=clamp(delivered-2.0*damage+0.2*max(0.0,energy_change)+0.01*min(0.0,energy_change),-1.0,1.0);
   }
 }
 life[l]=reward;life[l+1u]=clamp(body[s]/max(1.0,body[s+18u]),0.0,1.0);
 life[l+2u]=body[s+29u];life[l+3u]=0.0;
 body[s+94u]=body[s+68u];body[s+95u]=body[s+8u];body[s+96u]=body[s];body[s+97u]=1.0;
 life[m]=body[s+64u];life[g]=body[s+127u];life[k]=body[s+31u];
}
