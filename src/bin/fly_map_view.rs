//! Self-contained visual map for a `flymap` connectome.
//!
//! The map is static and lives in the browser; an optional `FSPKTR01` trace adds the
//! voltage and spike activity of one selected being over time. No GPU, server or external
//! JavaScript is needed. This is deliberately a diagnostic viewer: sampling edges and
//! compacting one being's state are explicit so opening a 15M-edge map does not freeze the
//! browser by trying to draw every edge.
//!
//! ```text
//! fly_map_view --map /path/to/map.flymap --html /media/llm-data/scratch/fly.html
//! fly_map_view --map /path/to/map.flymap --trace /media/llm-data/scratch/fly.fspike \
//!     --html /media/llm-data/scratch/fly.html
//! ```

use serde_json::json;
use std::fmt::Write as _;
use std::path::PathBuf;
use universe::fly_map::spiking::read_trace_sampled;
use universe::fly_map::{FlyMap, NONE};

const MAX_EDGES: usize = 12_000;
const MAX_NODES: usize = 6_000;
const MAX_FLOW_TYPES: usize = 48;
const MAX_TRACE_FRAMES: usize = 512;

fn usage() -> ! {
    eprintln!(
        "usage: fly_map_view --map <file.flymap> [--trace <file.fspike>] [--html <file.html>]"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut map_path: Option<PathBuf> = None;
    let mut trace_path: Option<PathBuf> = None;
    let mut html_path: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--map" => {
                i += 1;
                map_path = args.get(i).map(PathBuf::from);
            }
            "--trace" => {
                i += 1;
                trace_path = args.get(i).map(PathBuf::from);
            }
            "--html" => {
                i += 1;
                html_path = args.get(i).map(PathBuf::from);
            }
            _ => usage(),
        }
        i += 1;
    }
    let map_path = map_path.unwrap_or_else(|| usage());
    let map = FlyMap::load(map_path.to_str().unwrap_or_default()).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", map_path.display());
        std::process::exit(1);
    });
    let html_path = html_path.unwrap_or_else(|| {
        let mut p = map_path.clone();
        p.set_extension("html");
        p
    });

    let (edge_sample, flow_type_ids, type_flow) = summarize_edges(&map);

    let node_stride = map.n().div_ceil(MAX_NODES).max(1);
    let node_sample: Vec<[u32; 4]> = (0..map.n())
        .step_by(node_stride)
        .take(MAX_NODES)
        .map(|neuron| {
            [
                neuron as u32,
                map.type_of[neuron],
                map.pixel_of_neuron
                    .as_ref()
                    .and_then(|p| p.get(neuron))
                    .copied()
                    .unwrap_or(NONE),
                output_group(&map, neuron),
            ]
        })
        .collect();

    let output_groups: Vec<Vec<u32>> = (0..map.k_out())
        .map(|g| {
            let start = map.out_indptr[g] as usize;
            let end = map.out_indptr[g + 1] as usize;
            map.out_members[start..end].to_vec()
        })
        .collect();

    let trace_json = match trace_path {
        None => json!(null),
        Some(path) => {
            let trace = read_trace_sampled(path.to_str().unwrap_or_default(), MAX_TRACE_FRAMES)
                .unwrap_or_else(|e| {
                    eprintln!("failed to read trace {}: {e}", path.display());
                    std::process::exit(1);
                });
            if trace.n != map.n() {
                eprintln!(
                    "trace/map neuron mismatch: trace has {}, map has {}",
                    trace.n,
                    map.n()
                );
                std::process::exit(1);
            }
            json!({
                "being": trace.being,
                "frames": trace.frames.iter().map(|frame| json!({
                    "step": frame.step,
                    "commands": frame.commands,
                    "v": frame.v,
                    "spikes": frame.spike_counts,
                })).collect::<Vec<_>>(),
            })
        }
    };

    let data = json!({
        "n": map.n(),
        "m": map.m(),
        "nTypes": map.n_types,
        "kIn": map.k_in,
        "kOut": map.k_out(),
        "hasPreCsr": map.pre_csr.is_some(),
        "hasPixels": map.pixel_of_neuron.is_some(),
        "eyeBound": map.pixel_of_neuron.as_ref().map_or(0, |p| p.iter().filter(|&&x| x != NONE).count()),
        "typeFlow": type_flow,
        "flowTypeIds": flow_type_ids,
        "nodes": node_sample,
        "edges": edge_sample,
        "groups": output_groups,
        "trace": trace_json,
    });
    let html = render_html(&data.to_string());
    if let Err(e) = std::fs::write(&html_path, html) {
        eprintln!("could not write {}: {e}", html_path.display());
        std::process::exit(1);
    }
    println!(
        "→ {} · {} neurons · {} edges ({} drawn) · trace {}",
        html_path.display(),
        map.n(),
        map.m(),
        edge_sample.len(),
        if trace_path_was_present(&data) {
            "loaded"
        } else {
            "off"
        }
    );
}

fn trace_path_was_present(data: &serde_json::Value) -> bool {
    !data.get("trace").map_or(true, serde_json::Value::is_null)
}

/// Summarises the connectome in bounded memory.  The full FlyWire map has 8,866 cell
/// types, so a dense `n_types × n_types` matrix would be ~315 MiB and the browser would
/// try to paint 78 million cells.  We retain the most connected types and count only
/// their pairwise flow; all nodes outside that set are placed in an `other` column.
fn summarize_edges(map: &FlyMap) -> (Vec<[u32; 3]>, Vec<u32>, Vec<u32>) {
    let edge_stride = map.m().div_ceil(MAX_EDGES).max(1);
    let mut edge_sample = Vec::with_capacity(MAX_EDGES.min(map.m()));
    let mut incidence = vec![0u64; map.n_types];
    for_each_edge(map, |edge_idx, src, dst, weight| {
        let src_type = map.type_of[src as usize] as usize;
        let dst_type = map.type_of[dst as usize] as usize;
        incidence[src_type] = incidence[src_type].saturating_add(1);
        incidence[dst_type] = incidence[dst_type].saturating_add(1);
        if edge_idx % edge_stride == 0 && edge_sample.len() < MAX_EDGES {
            edge_sample.push([src, dst, weight.to_bits()]);
        }
    });

    let mut type_ids: Vec<usize> = (0..map.n_types).collect();
    type_ids.sort_unstable_by(|&a, &b| incidence[b].cmp(&incidence[a]).then_with(|| a.cmp(&b)));
    type_ids.truncate(MAX_FLOW_TYPES);
    type_ids.sort_unstable();
    let mut type_slot = vec![usize::MAX; map.n_types];
    for (slot, &type_id) in type_ids.iter().enumerate() {
        type_slot[type_id] = slot;
    }
    let width = type_ids.len();
    let mut flow = vec![0u32; width * width];
    for_each_edge(map, |_, src, dst, _| {
        let a = type_slot[map.type_of[src as usize] as usize];
        let b = type_slot[map.type_of[dst as usize] as usize];
        if a != usize::MAX && b != usize::MAX {
            flow[a * width + b] = flow[a * width + b].saturating_add(1);
        }
    });
    (
        edge_sample,
        type_ids.into_iter().map(|id| id as u32).collect(),
        flow,
    )
}

fn for_each_edge(map: &FlyMap, mut f: impl FnMut(usize, u32, u32, f32)) {
    if let Some(pre) = &map.pre_csr {
        let mut edge_idx = 0;
        for src in 0..map.n() {
            for edge in pre.indptr[src] as usize..pre.indptr[src + 1] as usize {
                f(edge_idx, src as u32, pre.indices[edge], pre.weights[edge]);
                edge_idx += 1;
            }
        }
    } else {
        let mut edge_idx = 0;
        for dst in 0..map.n() {
            for edge in map.indptr[dst] as usize..map.indptr[dst + 1] as usize {
                f(edge_idx, map.indices[edge], dst as u32, map.weights[edge]);
                edge_idx += 1;
            }
        }
    }
}

fn output_group(map: &FlyMap, neuron: usize) -> u32 {
    for group in 0..map.k_out() {
        let start = map.out_indptr[group] as usize;
        let end = map.out_indptr[group + 1] as usize;
        if map.out_members[start..end].contains(&(neuron as u32)) {
            return group as u32;
        }
    }
    NONE
}

fn render_html(data: &str) -> String {
    let mut h = String::new();
    write!(
        h,
        r##"<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Fly map viewer</title>
<style>
:root{{color-scheme:dark;--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--blue:#58a6ff}}
*{{box-sizing:border-box}} body{{margin:0;padding:22px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
h1{{font-size:21px;margin:0 0 4px}} h2{{font-size:15px;margin:24px 0 8px}}
.sub,.hint{{color:var(--dim);font-size:12px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px;margin-bottom:12px}}
.kpi{{font-size:22px;font-variant-numeric:tabular-nums}} .kpi small{{display:block;color:var(--dim);font-size:11px;font-weight:400}}
canvas{{display:block;max-width:100%;height:auto;background:#0a0e13;border-radius:5px}}
label{{color:var(--dim);font-size:12px}} input[type=range]{{width:min(640px,90%)}}
#tracebar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}} #frame-info{{color:var(--blue);font-variant-numeric:tabular-nums}}
table{{border-collapse:collapse;font-size:12px;width:100%}} td{{border-bottom:1px solid #21262d;padding:3px 6px}} td:last-child{{text-align:right;color:var(--dim)}}
.legend{{color:var(--dim);font-size:11px;margin-top:7px}} .warn{{color:#f2cc60}}
</style></head><body>
<h1>Fly map — connectome + atividade</h1><div class="sub">arquivo estático · arestas e neurônios são amostrados deterministicamente; o fluxo mostra os 48 tipos mais conectados e o trace usa no máximo 512 quadros</div>
<div class="grid" id="kpis"></div>
<h2>Fluxo entre tipos celulares</h2><div class="card"><canvas id="flow" width="900" height="420"></canvas><div class="legend">linha = tipo pré-sináptico · coluna = tipo pós-sináptico · intensidade logarítmica do número de arestas</div></div>
<h2>Topologia amostrada e tensão de membrana</h2><div class="card"><div id="tracebar"><label>amostra <input id="slider" type="range" min="0" value="0"></label><span id="frame-info">sem trace</span></div><canvas id="graph" width="1100" height="650"></canvas><div class="legend">cada coluna é um tipo; verde = tensão próxima/abaixo do limiar, amarelo = tensão alta, azul = sem trace; arestas fortes ficam mais opacas</div></div>
<div class="grid"><div class="card"><h2 style="margin-top:0">Saída motora</h2><table id="groups"></table></div><div class="card"><h2 style="margin-top:0">Olho</h2><div id="eye"></div><p class="hint">A ligação pixel→neurônio é anatômica e fica na VRAM. O trace só é mapeado quando solicitado.</p></div></div>
<script>const D=__DATA__;
const NONE=4294967295, colors=['#58a6ff','#f2cc60','#3fb950','#f85149','#bc8cff','#ffa657','#79c0ff','#d2a8ff'];
const k=document.getElementById('kpis');
function addK(v,l){{const e=document.createElement('div');e.className='card';e.innerHTML='<div class="kpi">'+v+'<small>'+l+'</small></div>';k.appendChild(e)}}
addK(D.n.toLocaleString(),'neurônios'); addK(D.m.toLocaleString(),'arestas'); addK(D.nTypes,'tipos celulares'); addK(D.eyeBound.toLocaleString(),'neurônios ligados à retina');
const flow=document.getElementById('flow'),fx=flow.getContext('2d'),fw=flow.width,fh=flow.height,flowIds=D.flowTypeIds||[],nTF=Math.max(flowIds.length,1),cell=Math.min(fw,fh-34)/nTF;
const maxFlow=Math.max(...D.typeFlow,1); fx.font='10px system-ui';
for(let a=0;a<nTF;a++)for(let b=0;b<nTF;b++){{const c=D.typeFlow[a*nTF+b];const z=c?Math.log1p(c)/Math.log1p(maxFlow):0;fx.fillStyle='rgba(88,166,255,'+(0.08+0.92*z)+')';fx.fillRect(b*cell,a*cell,cell-1,cell-1)}}
fx.fillStyle='#8b949e'; for(let a=0;a<nTF;a++){{const id=flowIds[a]===undefined?'?':flowIds[a];fx.fillText('T'+id, bnd(a), nTF*cell+18);fx.fillText('T'+id, 2, a*cell+12)}} function bnd(a){{return a*cell+Math.max(0,cell/2-8)}}
const groups=document.getElementById('groups'); D.groups.forEach((g,i)=>{{const tr=document.createElement('tr');tr.innerHTML='<td>grupo '+i+'</td><td>'+g.length.toLocaleString()+' neurônios</td>';groups.appendChild(tr)}});
document.getElementById('eye').innerHTML='<b>'+D.eyeBound.toLocaleString()+'</b> / '+D.n.toLocaleString()+' ('+(100*D.eyeBound/Math.max(D.n,1)).toFixed(2)+'%) com pixel binding'+(D.hasPixels?'':' <span class="warn">(bloco ausente)</span>');
const graph=document.getElementById('graph'),gx=graph.getContext('2d'),slider=document.getElementById('slider'),info=document.getElementById('frame-info');
const trace=D.trace, frames=trace?trace.frames:[]; slider.max=Math.max(frames.length-1,0); slider.disabled=!trace;
const hasOther=D.nTypes>flowIds.length, graphTypes=Math.max(flowIds.length+(hasOther?1:0),1);
function graphType(type){{const i=flowIds.indexOf(type);return i>=0?i:(hasOther?flowIds.length:0)}}
const pos=new Map(); for(const q of D.nodes){{const type=graphType(q[1]), x=55+(type+0.5)*((graph.width-100)/graphTypes); const y=25+((q[0]*2654435761>>>0)/4294967296)*(graph.height-50); pos.set(q[0],[x,y])}}
function draw(){{const fi=Number(slider.value), fr=frames[fi];gx.clearRect(0,0,graph.width,graph.height);gx.fillStyle='#0a0e13';gx.fillRect(0,0,graph.width,graph.height);
 for(let t=0;t<graphTypes;t++){{const x=55+(t+0.5)*((graph.width-100)/graphTypes);gx.fillStyle=colors[t%colors.length];gx.fillText(t<flowIds.length?'T'+flowIds[t]:'outros',x-8,15)}}
 for(const e of D.edges){{const a=pos.get(e[0]),b=pos.get(e[1]);if(!a||!b)continue;const w=new Float32Array(new Uint32Array([e[2]]).buffer)[0];gx.strokeStyle=w>=0?'rgba(63,185,80,'+Math.min(.5,.05+Math.abs(w)/20)+')':'rgba(248,81,73,'+Math.min(.5,.05+Math.abs(w)/20)+')';gx.lineWidth=Math.min(2,0.4+Math.abs(w)/100);gx.beginPath();gx.moveTo(a[0],a[1]);gx.lineTo(b[0],b[1]);gx.stroke()}}
 let active=0,maxV=-Infinity; for(const q of D.nodes){{const p=pos.get(q[0]);if(!p)continue;let v=0,sp=0;if(fr){{v=fr.v[q[0]]||0;sp=fr.spikes[q[0]]||0;maxV=Math.max(maxV,v);if(sp>0)active++}}let fill=trace?(v>-45?'#f2cc60':v>-52?'#3fb950':'#264b73'):colors[q[1]%colors.length];gx.fillStyle=fill;gx.globalAlpha=sp>0?1:.65;gx.beginPath();gx.arc(p[0],p[1],sp>0?3.5:2,0,Math.PI*2);gx.fill();gx.globalAlpha=1}}
 if(fr){{info.textContent='frame '+fi+'/'+(frames.length-1)+' · step '+fr.step+' · ativos '+active+' · max V '+maxV.toFixed(2)+' · cmd '+fr.commands.map(x=>x.toFixed(3)).join(' / ')}}else info.textContent=trace?'trace vazio':'sem trace — use --trace';}}
slider.addEventListener('input',draw); draw();
</script></body></html>"##,
    )
    .expect("static template is valid");
    h.replace("__DATA__", data)
}
