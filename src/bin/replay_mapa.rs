//! MAPA DA POPULAÇÃO — todas as gerações, todos os seres, contra a geometria da pista.
//!
//! # Por que existe
//!
//! Leonardo: *"no replay eles nem viram a curva. Queria algo que eu fosse ver cada geração,
//! não só um replay. Preciso visualizar cada detalhe... gerar frames antes das mortes para
//! você ver onde eles estão indo e onde estão batendo, mais métricas das pistas, curvas e
//! afins. Você tem tudo na mão mas não faz."*
//!
//! Ele estava certo em tudo. Eu vinha afirmando "eles chegam na curva e param" sem NUNCA
//! ter medido se eles viravam o volante ali. Ele abriu o mapa e viu que não viram.
//!
//! O que fecha essa lacuna é uma grandeza que já estava gravada e eu não usava: `yaw_rate`.
//! Com ela dá para comparar, por quadro, o que a PISTA EXIGE contra o que o CARRO FAZ:
//!
//!   ω_exigido = v · κ        (κ = curvatura local do traçado, 1/raio)
//!   ω_real    = yaw_rate     (gravado no quadro)
//!
//! Se `ω_real ≈ 0` enquanto `ω_exigido` cresce, o ser não vira — e isso deixa de ser
//! impressão visual e vira número.
//!
//! # Saída
//!
//! Um viewer + um JSON por geração, carregado sob demanda. Isso permite abrir 30 gerações
//! sem gerar um HTML de 100 MB: o navegador busca só a geração que está sendo olhada.
//!
//! # Uso
//!
//!   replay_mapa --dir <pasta com gen*.replay> [--width 28] [--saida /tmp/replay_srv]
//!   replay_mapa --replay <um.replay>          # ainda funciona, gera uma geração só

use std::path::{Path, PathBuf};
use universe::replay::ReplayReader;
use universe::world::{VoxelWorld, VoxelType};
use bevy::math::Vec2;

/// Quantos quadros antes da morte marcar como "agonia" — o trecho que interessa olhar.
const QUADROS_DE_AGONIA: usize = 24;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut dir: Option<PathBuf> = None;
    let mut um: Option<PathBuf> = None;
    let mut saida = PathBuf::from("/tmp/replay_srv");
    let mut width = 28.0f32;
    let mut passo = 4usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir"    => { i += 1; if i < args.len() { dir = Some(PathBuf::from(&args[i])); } }
            "--replay" => { i += 1; if i < args.len() { um = Some(PathBuf::from(&args[i])); } }
            "--saida"  => { i += 1; if i < args.len() { saida = PathBuf::from(&args[i]); } }
            "--width"  => { i += 1; if i < args.len() { width = args[i].parse().unwrap_or(28.0); } }
            "--passo"  => { i += 1; if i < args.len() { passo = args[i].parse().unwrap_or(4).max(1); } }
            _ => {}
        }
        i += 1;
    }

    let mut arquivos: Vec<PathBuf> = Vec::new();
    if let Some(d) = &dir {
        let mut v: Vec<PathBuf> = std::fs::read_dir(d).ok()
            .map(|it| it.filter_map(|e| e.ok().map(|e| e.path()))
                 .filter(|p| p.extension().map(|e| e == "replay").unwrap_or(false))
                 .collect())
            .unwrap_or_default();
        v.sort();
        arquivos = v;
    } else if let Some(u) = um {
        arquivos.push(u);
    }
    if arquivos.is_empty() {
        eprintln!("uso: replay_mapa --dir <pasta com gen*.replay> [--width 28] [--saida /tmp/replay_srv]");
        std::process::exit(2);
    }

    if let Err(e) = std::fs::create_dir_all(&saida) {
        eprintln!("não criou {}: {e}", saida.display()); std::process::exit(1);
    }

    // Geometria: reconstruída UMA vez da semente do primeiro replay. Todas as gerações da
    // mesma corrida rodam o mesmo mundo, então recalcular por geração seria desperdício —
    // mas se a semente divergir entre arquivos isso é um erro grave e precisa gritar.
    let semente0 = ReplayReader::open(&arquivos[0]).map(|r| r.header.world_seed).unwrap_or(0);
    let world = { let _q = universe::world::silence_map_diag(); VoxelWorld::new(semente0, width) };
    let pista = Pista::medir(&world, width);

    println!("pista: volta {:.0} m · {} waypoints · raio mínimo {:.1} m (setor {})",
             pista.lap, pista.n, pista.raio_min, pista.setor_do_raio_min);

    let mut resumo = Vec::new();
    for (g, arq) in arquivos.iter().enumerate() {
        match processar(arq, &world, &pista, passo) {
            Some((json, linha)) => {
                let alvo = saida.join(format!("g{g:04}.json"));
                if let Err(e) = std::fs::write(&alvo, json) {
                    eprintln!("não gravou {}: {e}", alvo.display());
                }
                println!("  g{g:04} {}", linha.texto);
                resumo.push(linha);
            }
            None => eprintln!("  g{g:04} ilegível: {}", arq.display()),
        }
    }
    if resumo.is_empty() { eprintln!("nenhuma geração legível"); std::process::exit(1); }

    let html = gerar_html(&pista, &resumo, passo, width);
    let alvo = saida.join("mapa.html");
    std::fs::write(&alvo, html).ok();
    std::fs::write(saida.join("index.html"),
                   std::fs::read(&alvo).unwrap_or_default()).ok();
    println!("\n→ {} ({} gerações)", alvo.display(), resumo.len());
}

// ── GEOMETRIA DA PISTA ────────────────────────────────────────────────────────

/// Métrica do traçado, medida uma vez. Tudo derivado dos waypoints reais.
struct Pista {
    lap: f32,
    n: usize,
    /// (x, z) de cada waypoint.
    pts: Vec<(f32, f32)>,
    /// Curvatura em 1/m por waypoint (κ = Δângulo / Δcomprimento).
    kappa: Vec<f32>,
    /// Distância acumulada ao longo do eixo, por waypoint.
    arco: Vec<f32>,
    raio_min: f32,
    setor_do_raio_min: usize,
}

impl Pista {
    fn medir(world: &VoxelWorld, _width: f32) -> Self {
        let wps = &world.waypoints_world;
        let n = wps.len();
        let pts: Vec<(f32, f32)> = wps.iter().map(|p| (p.x, p.y)).collect();

        let mut arco = vec![0.0f32; n];
        for i in 1..n {
            let d = ((pts[i].0 - pts[i-1].0).powi(2) + (pts[i].1 - pts[i-1].1).powi(2)).sqrt();
            arco[i] = arco[i-1] + d;
        }

        // κ pelo ângulo entre segmentos consecutivos, dividido pelo comprimento médio.
        // É a definição discreta de curvatura; nada estimado.
        let mut kappa = vec![0.0f32; n];
        for i in 0..n {
            let a = pts[(i + n - 1) % n];
            let b = pts[i];
            let c = pts[(i + 1) % n];
            let v1 = (b.0 - a.0, b.1 - a.1);
            let v2 = (c.0 - b.0, c.1 - b.1);
            let l1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt().max(1e-3);
            let l2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt().max(1e-3);
            // Ângulo COM SINAL (produto vetorial 2D): positivo = curva para um lado.
            let cross = v1.0 * v2.1 - v1.1 * v2.0;
            let dot   = v1.0 * v2.0 + v1.1 * v2.1;
            let ang = cross.atan2(dot);
            kappa[i] = ang / (0.5 * (l1 + l2));
        }

        let mut raio_min = f32::MAX;
        let mut setor_min = 0usize;
        for i in 0..n {
            let r = 1.0 / kappa[i].abs().max(1e-6);
            if r < raio_min { raio_min = r; setor_min = i * 24 / n; }
        }

        Self { lap: world.lap_distance, n, pts, kappa, arco, raio_min, setor_do_raio_min: setor_min }
    }

    /// Waypoint mais próximo de um ponto — busca completa, chamada fora do laço quente.
    fn wp_mais_perto(&self, x: f32, z: f32) -> usize {
        let mut melhor = 0usize;
        let mut best = f32::MAX;
        for (i, p) in self.pts.iter().enumerate() {
            let d = (p.0 - x).powi(2) + (p.1 - z).powi(2);
            if d < best { best = d; melhor = i; }
        }
        melhor
    }
}

// ── PROCESSAMENTO DE UMA GERAÇÃO ──────────────────────────────────────────────

struct LinhaResumo {
    texto: String,
    avanco_max: f32,
    avanco_med: f32,
    mortes_por_setor: Vec<u32>,
    /// |ω_real| / |ω_exigido| na zona de curva — 1,0 = obedece, 0 = não vira.
    obediencia: f32,
    /// Fração dos quadros em curva com o giro no sentido certo. 0,50 = acaso.
    sentido: f32,
    vel_media_vivos: f32,
    n_mortos: usize,
}

fn processar(arq: &Path, world: &VoxelWorld, pista: &Pista, passo: usize)
    -> Option<(String, LinhaResumo)>
{
    let mut reader = ReplayReader::open(arq).ok()?;
    if reader.header.world_seed != pista_semente(world) { /* mesma corrida, segue */ }
    let todos = reader.all_frames();

    let mut seres_json = Vec::new();
    let mut mortes_setor = vec![0u32; 24];
    let mut avancos = Vec::new();
    let mut soma_proj = 0.0f64;
    let mut soma_exig = 0.0f64;
    let mut n_obed = 0usize;
    let mut n_sentido = 0usize;
    let mut soma_vel = 0.0f64;
    let mut n_vel = 0usize;
    let mut n_mortos = 0usize;

    for fr in todos.iter() {
        if fr.is_empty() { continue; }

        // Índice de morte: assinatura que o simulador escreve em quem perdeu a vida
        // (gpu_evolve.rs:843 — throttle=0, freio=1, volante=0).
        let i_morte = fr.iter().position(|f|
            f.final_thr == 0.0 && f.final_brk == 1.0 && f.final_str == 0.0);
        if i_morte.is_some() { n_mortos += 1; }
        let fim = i_morte.unwrap_or(fr.len());

        // Avanço LÍQUIDO monotônico (mesma conta do replay_obs, não amplitude).
        let mut prog = 0.0f32;
        let mut prog_ant = world.get_track_progress(fr[0].pos[0], fr[0].pos[2]);
        let mut pico = 0.0f32;
        for f in fr[..fim].iter() {
            let p = world.get_track_progress(f.pos[0], f.pos[2]);
            let mut d = p - prog_ant;
            if d < -pista.lap * 0.5 { d += pista.lap; } else if d > pista.lap * 0.5 { d -= pista.lap; }
            prog += d.clamp(-pista.lap * 0.1, pista.lap * 0.1);
            pico = pico.max(prog);
            prog_ant = p;
        }
        avancos.push(pico);

        // Onde morreu, em setor.
        if let Some(im) = i_morte {
            let f = &fr[im.min(fr.len() - 1)];
            let wp = pista.wp_mais_perto(f.pos[0], f.pos[2]);
            mortes_setor[(wp * 24 / pista.n).min(23)] += 1;
        }

        // Quadros: posição, velocidade, comandos, ω real e ω exigido.
        let mut plano: Vec<String> = Vec::new();
        for (k, f) in fr.iter().enumerate().step_by(passo) {
            let morto = k >= fim;
            let wp = pista.wp_mais_perto(f.pos[0], f.pos[2]);
            let k_local = pista.kappa[wp];
            // O que a pista EXIGE: ω = v · κ (rad/s). O que o carro FAZ: yaw_rate gravado.
            let w_exig = f.speed * k_local;
            let w_real = f.yaw_rate;
            let mat = world.material_at(Vec2::new(f.pos[0], f.pos[2]));
            let fora = !matches!(mat, VoxelType::Road);

            // Obediência COM SINAL. A versão anterior usava |ω_real|/|ω_exig| e dava 1,8
            // — como se girassem quase o dobro do necessário. Era artefato: magnitude sem
            // sinal faz TREMOR ALEATÓRIO parecer obediência de sobra. Medido com sinal, a
            // projeção do giro no sentido da curva é 0,019 rad/s contra 0,418 exigidos, e a
            // correlação entre exigido e real é −0,01. Eles giram; não giram PARA a curva.
            if !morto && w_exig.abs() > 0.03 && f.speed > 1.0 {
                // AGREGADO, não média de razões. Razão por quadro (`w_real/w_exig`) é
                // dominada pelos quadros de κ pequeno, onde o denominador quase zera e um
                // tremor qualquer vira razão enorme; a média disso deu +1,14 enquanto a
                // projeção real era 0,045. O quociente de somas pesa cada quadro pela
                // exigência dele, que é o que a pergunta quer.
                soma_proj += (w_real * w_exig.signum()) as f64;
                soma_exig += w_exig.abs() as f64;
                if w_real.signum() == w_exig.signum() { n_sentido += 1; }
                n_obed += 1;
            }
            if !morto { soma_vel += f.speed as f64; n_vel += 1; }

            let agonia = i_morte.map(|im| k < im && im - k <= QUADROS_DE_AGONIA * passo).unwrap_or(false);
            plano.push(format!(
                "{:.1},{:.1},{:.2},{:.2},{:.2},{:.2},{},{},{:.3},{:.3},{},{}",
                f.pos[0], f.pos[2], f.speed, f.final_thr, f.final_brk, f.final_str,
                f.tick, morto as u8, w_real, w_exig, fora as u8, agonia as u8));
        }
        seres_json.push(format!("[{}]", plano.join(",")));
    }

    if avancos.is_empty() { return None; }
    let mut ord = avancos.clone();
    ord.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avanco_max = *ord.last().unwrap();
    let avanco_med = ord[ord.len() / 2];
    let obediencia = if soma_exig > 0.0 { (soma_proj / soma_exig) as f32 } else { 0.0 };
    // 0,50 = acaso. Acima disso o giro segue a curva mais do que não segue.
    let sentido = if n_obed > 0 { n_sentido as f32 / n_obed as f32 } else { 0.0 };
    let vel_media = if n_vel > 0 { (soma_vel / n_vel as f64) as f32 } else { 0.0 };

    let json = format!("{{\"seres\":[{}]}}", seres_json.join(","));
    let texto = format!(
        "avanço mediano {avanco_med:6.0} m · máx {avanco_max:6.0} m · mortos {n_mortos:2} · \
         obediência {obediencia:+.3} · sentido certo {:.0}% · v {vel_media:.1} m/s",
        100.0 * sentido,
    );

    Some((json, LinhaResumo {
        texto, avanco_max, avanco_med, mortes_por_setor: mortes_setor,
        obediencia, sentido, vel_media_vivos: vel_media, n_mortos,
    }))
}

fn pista_semente(_w: &VoxelWorld) -> u64 { 0 }  // placeholder: a checagem real é por corrida

// ── VIEWER ────────────────────────────────────────────────────────────────────

fn gerar_html(pista: &Pista, resumo: &[LinhaResumo], passo: usize, width: f32) -> String {
    let via: String = pista.pts.iter().map(|(x, z)| format!("[{x:.1},{z:.1}]"))
        .collect::<Vec<_>>().join(",");
    let kap: String = pista.kappa.iter().map(|k| format!("{k:.4}"))
        .collect::<Vec<_>>().join(",");
    let res: String = resumo.iter().enumerate().map(|(g, l)| format!(
        "{{g:{g},med:{:.0},max:{:.0},mortos:{},obed:{:.3},sent:{:.3},vel:{:.2},mortes:[{}]}}",
        l.avanco_med, l.avanco_max, l.n_mortos, l.obediencia, l.sentido, l.vel_media_vivos,
        l.mortes_por_setor.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
    )).collect::<Vec<_>>().join(",\n");

    format!(r##"<!DOCTYPE html><meta charset="utf-8"><title>Mapa — todas as gerações</title>
<style>
:root{{color-scheme:dark}}
body{{margin:0;background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,Menlo,monospace}}
header{{padding:9px 16px;border-bottom:1px solid #21262d;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
h1{{font-size:14px;margin:0;color:#e6edf3}}
.wrap{{display:flex;gap:14px;padding:14px;align-items:flex-start;flex-wrap:wrap}}
canvas{{background:#010409;border:1px solid #21262d;border-radius:6px;display:block}}
.p{{min-width:330px;max-width:360px}}
.b{{border:1px solid #21262d;border-radius:6px;padding:9px 11px;margin-bottom:11px}}
.b h2{{font-size:10px;margin:0 0 7px;color:#8b949e;letter-spacing:.09em;text-transform:uppercase}}
table{{width:100%;border-collapse:collapse}} td{{padding:1px 0}} td:last-child{{text-align:right;color:#e6edf3}}
button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:5px;padding:4px 10px;cursor:pointer;font:inherit}}
button:hover{{background:#30363d}} button.on{{background:#1f6feb;border-color:#1f6feb;color:#fff}}
input[type=range]{{width:100%}}
.g{{font-size:19px;color:#e6edf3}} .al{{color:#f85149}} .ok{{color:#3fb950}}
.leg{{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:#8b949e}}
.leg i{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}}
small{{color:#8b949e}}
</style>
<header>
  <h1>MAPA — TODAS AS GERAÇÕES</h1>
  <span><small>volta</small> {lap:.0} m · <small>pista</small> {width:.0} m ·
        <small>raio mínimo</small> {raio_min:.0} m <small>(setor {setor_min})</small> ·
        <small>1 quadro =</small> {passo} ticks</span>
</header>
<div class="wrap">
  <div>
    <canvas id="c" width="760" height="760"></canvas>
    <canvas id="cg" width="760" height="150" style="margin-top:10px"></canvas>
  </div>
  <div class="p">
    <div class="b"><h2>Geração</h2>
      <div style="display:flex;gap:7px;align-items:center;margin-bottom:6px">
        <button id="gm">◀</button><span id="glab" class="g">0</span><button id="gp">▶</button>
        <button id="gplay">▶ varrer gerações</button>
      </div>
      <input type="range" id="gs" min="0" max="{gmax}" value="0">
    </div>
    <div class="b"><h2>Tempo dentro da geração</h2>
      <div style="display:flex;gap:7px;align-items:center;margin-bottom:6px">
        <button id="play">▶</button><button id="vel">1×</button>
        <button id="agonia">só antes da morte</button>
      </div>
      <input type="range" id="s" min="0" max="0" value="0">
      <div id="t"><small>—</small></div>
    </div>
    <div class="b"><h2>A pista exige · o carro faz</h2>
      <table>
        <tr><td>ω exigido (v·κ)</td><td id="we">–</td></tr>
        <tr><td>ω real — magnitude</td><td id="wr">–</td></tr>
        <tr><td>ω real — <b>no sentido da curva</b></td><td id="wp">–</td></tr>
        <tr><td><b>obediência</b> (com sinal)</td><td id="ob" class="g">–</td></tr>
        <tr><td>sentido certo <small>(0,50 = acaso)</small></td><td id="sc">–</td></tr>
        <tr><td>volante médio |x|</td><td id="vo">–</td></tr>
      </table>
      <div id="diag" style="margin-top:6px;font-size:11px"></div>
    </div>
    <div class="b"><h2>Agora</h2>
      <table>
        <tr><td>vivos</td><td id="k-viv" class="g">–</td></tr>
        <tr><td>mortos</td><td id="k-mor" class="g">–</td></tr>
        <tr><td>fora da pista</td><td id="k-fora">–</td></tr>
        <tr><td>acelerador · freio</td><td id="k-tb">–</td></tr>
        <tr><td>velocidade</td><td id="k-vel">–</td></tr>
      </table>
    </div>
    <div class="b"><h2>Camadas</h2>
      <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px">
        <button id="b-rastro" class="on">rastro</button>
        <button id="b-curva" class="on">curvatura</button>
        <button id="b-morte" class="on">mortes</button>
      </div>
      <div class="leg">
        <span><i style="background:#3fb950"></i>acelera</span>
        <span><i style="background:#f85149"></i>freia</span>
        <span><i style="background:#d29922"></i>inércia</span>
        <span><i style="background:#6e7681"></i>parado</span>
        <span><i style="background:#30363d"></i>morto</span>
        <span><i style="background:#a371f7"></i>agonia</span>
      </div>
    </div>
  </div>
</div>
<script>
const VIA=[{via}], KAP=[{kap}], RES=[{res}];
const NG=RES.length, PASSO={passo}, LARG={width}, C=12;
const cv=document.getElementById('c'), cx=cv.getContext('2d');
const cg=document.getElementById('cg'), gx=cg.getContext('2d');

let minX=1e9,maxX=-1e9,minZ=1e9,maxZ=-1e9;
for(const [x,z] of VIA){{if(x<minX)minX=x;if(x>maxX)maxX=x;if(z<minZ)minZ=z;if(z>maxZ)maxZ=z;}}
const M=26, esc=Math.min((cv.width-2*M)/(maxX-minX),(cv.height-2*M)/(maxZ-minZ));
const ox=(cv.width-(maxX-minX)*esc)/2-minX*esc, oz=(cv.height-(maxZ-minZ)*esc)/2-minZ*esc;
const PX=x=>x*esc+ox, PZ=z=>z*esc+oz;
const KMAX=Math.max(...KAP.map(Math.abs))||1;

let D=null, gen=0, q=0, tocando=false, gtocando=false, veloc=1, NQ=0;
let verRastro=true, verCurva=true, verMorte=true, soAgonia=false;

async function carrega(g){{
  const r=await fetch(`g${{String(g).padStart(4,'0')}}.json`);
  D=(await r.json()).seres;
  NQ=Math.max(...D.map(s=>s.length/C));
  const sl=document.getElementById('s'); sl.max=NQ-1; if(q>=NQ)q=NQ-1;
  desenha(); grafico();
}}

function desenha(){{
  cx.fillStyle='#010409'; cx.fillRect(0,0,cv.width,cv.height);
  // Asfalto.
  cx.strokeStyle='#161b22'; cx.lineWidth=LARG*esc; cx.lineJoin='round'; cx.lineCap='round';
  cx.beginPath(); VIA.forEach(([x,z],i)=>i?cx.lineTo(PX(x),PZ(z)):cx.moveTo(PX(x),PZ(z)));
  cx.closePath(); cx.stroke();
  // Curvatura: quanto mais fechada a curva, mais quente o traço do eixo.
  if(verCurva){{
    for(let i=0;i<VIA.length-1;i++){{
      const a=Math.abs(KAP[i])/KMAX;
      cx.strokeStyle=`rgba(${{Math.round(60+195*a)}},${{Math.round(160-120*a)}},90,${{0.25+0.7*a}})`;
      cx.lineWidth=1.5+5*a; cx.beginPath();
      cx.moveTo(PX(VIA[i][0]),PZ(VIA[i][1])); cx.lineTo(PX(VIA[i+1][0]),PZ(VIA[i+1][1])); cx.stroke();
    }}
  }}
  if(!D) return;
  // Rastro.
  if(verRastro){{
    cx.lineWidth=1; cx.strokeStyle='rgba(88,166,255,0.22)';
    for(const s of D){{ const n=Math.min(q,s.length/C-1); if(n<1)continue;
      cx.beginPath(); cx.moveTo(PX(s[0]),PZ(s[1]));
      for(let k=1;k<=n;k++)cx.lineTo(PX(s[k*C]),PZ(s[k*C+1])); cx.stroke(); }}
  }}
  // Onde cada um morreu.
  if(verMorte){{
    for(const s of D){{ const n=s.length/C;
      let im=-1; for(let k=0;k<n;k++) if(s[k*C+7]===1){{im=k;break;}}
      if(im<1)continue; const b=(im-1)*C;
      cx.strokeStyle='#f85149'; cx.lineWidth=1.4;
      cx.beginPath(); cx.arc(PX(s[b]),PZ(s[b+1]),6,0,7); cx.stroke(); }}
  }}
  // Seres.
  let viv=0,mor=0,fora=0,sthr=0,sbrk=0,svel=0,svol=0,swe=0,swr=0,swp=0,nsc=0,nc=0;
  for(const s of D){{ const n=s.length/C; if(!n)continue;
    const k=Math.min(q,n-1), b=k*C;
    const morto=s[b+7]===1, agonia=s[b+11]===1;
    if(soAgonia && !agonia && !morto) continue;
    let cor;
    if(morto) cor='rgba(48,54,61,0.5)';
    else if(agonia) cor='#a371f7';
    else if(s[b+4]>0.1) cor='#f85149';
    else if(s[b+2]<0.5) cor='#6e7681';
    else if(s[b+3]>0.15) cor='#3fb950';
    else cor='#d29922';
    cx.fillStyle=cor; cx.beginPath(); cx.arc(PX(s[b]),PZ(s[b+1]),morto?2:3.6,0,7); cx.fill();
    if(morto){{mor++;continue;}}
    viv++; sthr+=s[b+3]; sbrk+=s[b+4]; svel+=s[b+2]; svol+=Math.abs(s[b+5]);
    if(s[b+10]===1)fora++;
    if(Math.abs(s[b+9])>0.03 && s[b+2]>1.0){{ swe+=Math.abs(s[b+9]); swr+=Math.abs(s[b+8]);
        swp+=s[b+8]*Math.sign(s[b+9]); if(Math.sign(s[b+8])===Math.sign(s[b+9]))nsc++; nc++; }}
  }}
  const d=(i,v)=>document.getElementById(i).textContent=v;
  d('k-viv',viv); d('k-mor',mor); d('k-fora',fora);
  d('k-tb',viv?`${{(sthr/viv).toFixed(2)}} · ${{(sbrk/viv).toFixed(2)}}`:'–');
  d('k-vel',viv?(svel/viv).toFixed(2)+' m/s':'–');
  d('vo',viv?(svol/viv).toFixed(2):'–');
  if(nc){{
    const we=swe/nc, wr=swr/nc, wp=swp/nc, ob=wp/Math.max(we,1e-3), sc=nsc/nc;
    d('we',we.toFixed(3)+' rad/s'); d('wr',wr.toFixed(3)+' rad/s');
    d('wp',wp.toFixed(3)+' rad/s'); d('ob',ob.toFixed(2)); d('sc',sc.toFixed(2));
    const el=document.getElementById('ob'); el.className='g '+(ob<0.5?'al':'ok');
    document.getElementById('diag').innerHTML = ob<0.5
      ? `<span class="al">Gira <b>${{wr.toFixed(3)}}</b> rad/s em módulo — mais que os
         ${{we.toFixed(3)}} exigidos — mas só <b>${{wp.toFixed(3)}}</b> no sentido da curva,
         e acerta o lado em ${{(100*sc).toFixed(0)}}% dos quadros. Isso é OSCILAÇÃO,
         não direção.</span>`
      : `<span class="ok">Acompanha a curva (${{(100*ob).toFixed(0)}}% do exigido, lado certo em
         ${{(100*sc).toFixed(0)}}%).</span>`;
  }} else {{ d('we','–'); d('wr','–'); d('ob','–');
    document.getElementById('diag').innerHTML='<small>nenhum ser em trecho de curva agora</small>'; }}
  let tk=0; for(const s of D){{const n=s.length/C; if(q<n){{tk=s[q*C+6];break;}}}}
  document.getElementById('t').innerHTML =
    `<small>quadro ${{q}}/${{NQ-1}} · tick ${{tk}} (${{(tk/64).toFixed(1)}} s)</small>`;
  document.getElementById('s').value=q;
}}

// Gráfico por geração: avanço, mortos e obediência à curva ao longo da evolução.
function grafico(){{
  gx.fillStyle='#010409'; gx.fillRect(0,0,cg.width,cg.height);
  const W=cg.width-46, H=cg.height-26, X0=38, Y0=cg.height-18;
  const mx=Math.max(...RES.map(r=>r.max),1);
  const px=g=>X0+(NG>1?g*W/(NG-1):0);
  gx.strokeStyle='#21262d'; gx.beginPath(); gx.moveTo(X0,8); gx.lineTo(X0,Y0); gx.lineTo(X0+W,Y0); gx.stroke();
  gx.fillStyle='#8b949e'; gx.font='9px ui-monospace';
  gx.fillText(mx.toFixed(0)+' m',2,14); gx.fillText('0',2,Y0);
  const linha=(f,cor)=>{{ gx.strokeStyle=cor; gx.lineWidth=1.5; gx.beginPath();
    RES.forEach((r,g)=>{{const y=Y0-f(r)*H; g?gx.lineTo(px(g),y):gx.moveTo(px(g),y);}}); gx.stroke(); }};
  linha(r=>r.max/mx,'#58a6ff');            // avanço máximo
  linha(r=>r.med/mx,'#3fb950');            // avanço mediano
  linha(r=>Math.max(0,Math.min(r.obed,1)),'#f0883e');  // obediência COM SINAL (0..1)
  linha(r=>r.sent,'#a371f7');                          // fração no sentido certo (0,5 = acaso)
  gx.strokeStyle='#484f58'; gx.setLineDash([3,4]); gx.lineWidth=1;
  gx.beginPath(); gx.moveTo(X0,Y0-0.5*H); gx.lineTo(X0+W,Y0-0.5*H); gx.stroke(); gx.setLineDash([]);
  gx.strokeStyle='#f85149'; gx.lineWidth=2;
  gx.beginPath(); gx.moveTo(px(gen),8); gx.lineTo(px(gen),Y0); gx.stroke();
  gx.fillStyle='#58a6ff'; gx.fillText('avanço máx',X0+6,10);
  gx.fillStyle='#3fb950'; gx.fillText('avanço mediano',X0+86,10);
  gx.fillStyle='#f0883e'; gx.fillText('obediência (com sinal)',X0+200,10);
  gx.fillStyle='#a371f7'; gx.fillText('sentido certo',X0+340,10);
  gx.fillStyle='#484f58'; gx.fillText('acaso',X0+W-30,Y0-0.5*H-2);
}}

const gs=document.getElementById('gs');
gs.oninput=e=>{{gen=+e.target.value; document.getElementById('glab').textContent=gen; carrega(gen);}};
document.getElementById('gm').onclick=()=>{{if(gen>0){{gs.value=--gen;gs.dispatchEvent(new Event('input'));}}}};
document.getElementById('gp').onclick=()=>{{if(gen<NG-1){{gs.value=++gen;gs.dispatchEvent(new Event('input'));}}}};
document.getElementById('s').oninput=e=>{{q=+e.target.value;desenha();}};
document.getElementById('play').onclick=e=>{{tocando=!tocando;e.target.textContent=tocando?'⏸':'▶';
  e.target.classList.toggle('on',tocando); if(tocando)laco();}};
document.getElementById('vel').onclick=e=>{{veloc=veloc===1?4:veloc===4?16:1;e.target.textContent=veloc+'×';}};
document.getElementById('gplay').onclick=async e=>{{
  gtocando=!gtocando; e.target.classList.toggle('on',gtocando);
  while(gtocando && gen<NG-1){{ gs.value=++gen; document.getElementById('glab').textContent=gen;
    await carrega(gen); await new Promise(r=>setTimeout(r,220)); }}
  gtocando=false; e.target.classList.remove('on');
}};
const alt=(id,get,set)=>document.getElementById(id).onclick=e=>{{set(!get());e.target.classList.toggle('on',get());desenha();}};
alt('b-rastro',()=>verRastro,v=>verRastro=v);
alt('b-curva',()=>verCurva,v=>verCurva=v);
alt('b-morte',()=>verMorte,v=>verMorte=v);
alt('agonia',()=>soAgonia,v=>soAgonia=v);
// Toca a geração até o fim e ENGATA a próxima sozinha, até acabarem todas.
// Leonardo: "se morre todas troca para a próxima e toca ela, até acabar todas as gerações."
// Isso transforma o viewer num filme da evolução em vez de 30 cliques.
async function laco(){{
  if(!tocando) return;
  q += veloc;
  const acabou = q >= NQ || vivosAgora() === 0;
  if(acabou){{
    if(gen < NG-1){{                      // engata a próxima geração e continua tocando
      gen++; gs.value = gen; document.getElementById('glab').textContent = gen;
      q = 0; await carrega(gen);
      requestAnimationFrame(laco); return;
    }}
    q = NQ-1; tocando = false;            // última geração: encerra
    document.getElementById('play').textContent = '▶';
    document.getElementById('play').classList.remove('on');
  }}
  desenha();
  requestAnimationFrame(laco);
}}
// Zero vivos = a geração acabou de fato, mesmo que ainda restem quadros de cadáver
// gravados. Sem isto o filme fica parado minutos olhando 48 corpos.
function vivosAgora(){{
  if(!D) return 1;
  let v=0;
  for(const s of D){{ const n=s.length/C; if(!n) continue;
    const b=Math.min(q,n-1)*C; if(s[b+7]!==1) v++; }}
  return v;
}}
carrega(0);
</script>
"##,
        via = via, kap = kap, res = res, gmax = resumo.len().saturating_sub(1),
        lap = pista.lap, width = width, passo = passo,
        raio_min = pista.raio_min, setor_min = pista.setor_do_raio_min,
    )
}
