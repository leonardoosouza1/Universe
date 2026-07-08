// IARA-VIDA — um SER cognitivo que realmente pensa (Leonardo 2026-07-07, branch iara-vida).
// Reusa os cérebros REAIS do Universe (CfCCell, BitLinear, DenseLayer) num mundo multi-necessidade
// (comida≠água≠ninho, dia/noite, predadores). Cérebro recorrente + neuromodulado + preditivo, glass-box.
// Modos:  --evolve [gens]   GA (esquema real do Universe) → salva vida_champion.dna + métricas
//         --record [dna] [ticks]  simula mundo vivo → vida/trace.json (mundo + cérebro focal por tick)
//         --serve [dna] [porta]   servidor HTTP: UI dirige o Rust real ao vivo (perturbar/spawnar/mundo)
//         --watch [dna]     render ASCII (pra mim validar)
use universe::cfc::CfCCell;
use universe::layers::BitLinear;
use universe::dense_layer::DenseLayer;
use std::io::{Read, Write};
use bevy::prelude::*;
use bevy::math::Isometry2d;
use std::sync::atomic::{AtomicBool, Ordering};
static LEARN: AtomicBool = AtomicBool::new(true);   // ablação: desliga o "comer=aprender" pra validar conhecimento=sobrevivência

// ---- dimensões do cérebro (com COMUNICAÇÃO POR BYTE: ouve 8 bits dos vizinhos, fala 8 bits) ----
const NMSG: usize = 8;   // bits da mensagem (um BYTE — a tese byte viva)
const NIN: usize = 27;   // 18 sensores + 8 bits ouvidos + viés
const NH1: usize = 20;   // córtex instinto (BitLinear)
const NH2: usize = 14;   // córtex associação (Dense)
const NOUT: usize = 13;  // 0 virar,1 acelerar,2 comer,3 beber,4 descansar + 8 bits de mensagem
const DT: f32 = 0.1;

const SENSE_NAMES: [&str; NIN] = ["comida_esq","comida_dir","agua_esq","agua_dir","pred_esq","pred_dir",
    "pred_prox","ninho_x","ninho_y","vizinhos","alinham","fome","sede","fadiga","medo","saude","dia_sin","dia_cos",
    "ouve0","ouve1","ouve2","ouve3","ouve4","ouve5","ouve6","ouve7","vies"];
const ACT_NAMES: [&str; NOUT] = ["virar","acelerar","comer","beber","descansar","fala0","fala1","fala2","fala3","fala4","fala5","fala6","fala7"];

// ---- DOMÍNIOS DE CONHECIMENTO (comida = frase/bytes; comer = aprender; saber = sobreviver) ----
const NDOM: usize = 6;
const DOM_NAMES: [&str; NDOM] = ["forrageio","agua","perigo","abrigo","social","navegacao"];
// a "frase" (bytes) que a comida de cada domínio ensina — o conhecimento fica VISÍVEL
const DOM_FACTS: [&str; NDOM] = ["o verde alimenta","o azul mata a sede","o vermelho te caca","a noite pede abrigo","o cardume protege","o terreno se atravessa"];

// ---- RNG determinístico ----
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) }
    fn f(&mut self) -> f32 { (self.u() >> 40) as f32 / (1u64 << 24) as f32 }
    fn g(&mut self) -> f32 { let (a, b) = (self.f().max(1e-7), self.f()); (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos() }
}

fn q8(x: f32) -> i8 { (x.clamp(-1.0, 1.0) * 127.0) as i8 }

// ================== CÉREBRO (camadas reais do Universe) ==================
const GENOME: usize = NIN*NH1 + NH1*NH2 + NH2 + NH2*NOUT + NOUT + NH2*NIN + NIN + 2;
struct Brain {
    l1: BitLinear,      // NIN -> NH1 (instinto ternário)
    cfc_p: CfCCell,     // NIN memória perceptual
    cfc_r: CfCCell,     // NH1 memória de trabalho (recorrente)
    l2: DenseLayer,     // NH1 -> NH2
    act: DenseLayer,    // NH2 -> NOUT
    wm: DenseLayer,     // NH2 -> NIN (world-model: prevê próxima percepção)
}
#[derive(Clone)]
struct Cognition { mem: [f32; NIN], h1: [f32; NH1], rec: [f32; NH1], h2: [f32; NH2], out: [f32; NOUT], pred: [f32; NIN] }
impl Brain {
    fn from_genome(g: &[f32]) -> Brain {
        let mut o = 0;
        let mut take = |n: usize, off: &mut usize| { let s = g[*off..*off+n].to_vec(); *off += n; s };
        let mut l1 = BitLinear::new(NIN, NH1); l1.weights = take(NIN*NH1, &mut o);
        let mut l2 = DenseLayer::new(NH1, NH2); l2.weights = take(NH1*NH2, &mut o); l2.bias = take(NH2, &mut o);
        let mut act = DenseLayer::new(NH2, NOUT); act.weights = take(NH2*NOUT, &mut o); act.bias = take(NOUT, &mut o);
        let mut wm = DenseLayer::new(NH2, NIN); wm.weights = take(NH2*NIN, &mut o); wm.bias = take(NIN, &mut o);
        let mut cfc_p = CfCCell::new(NIN); cfc_p.decay_rate = 0.4 + 3.0 * sig(g[o]); o += 1; cfc_p.reset();
        let mut cfc_r = CfCCell::new(NH1); cfc_r.decay_rate = 0.3 + 3.0 * sig(g[o]); cfc_r.reset();
        Brain { l1, cfc_p, cfc_r, l2, act, wm }
    }
    fn reset(&mut self) { self.cfc_p.reset(); self.cfc_r.reset(); }
    // senses em [-1,1]; dopa/corti modulam ganho (dopamina→exploração, cortisol→defesa)
    fn think(&mut self, senses: &[f32; NIN], dopa: f32, corti: f32) -> Cognition {
        let si: Vec<i8> = senses.iter().map(|&x| q8(x)).collect();
        let mut memi = vec![0i8; NIN]; self.cfc_p.update(&si, &mut memi, DT);
        let mem: [f32; NIN] = arr(&memi.iter().map(|&x| x as f32 / 127.0).collect::<Vec<_>>());
        let h1v = self.l1.forward(&memi); let h1: [f32; NH1] = arr(&h1v.iter().map(|x| x.tanh()).collect::<Vec<_>>());
        let h1i: Vec<i8> = h1.iter().map(|&x| q8(x)).collect();
        let mut reci = vec![0i8; NH1]; self.cfc_r.update(&h1i, &mut reci, DT);
        let recf: Vec<f32> = reci.iter().map(|&x| x as f32 / 127.0).collect(); let rec: [f32; NH1] = arr(&recf);
        let gain = (1.0 + 0.5 * dopa - 0.4 * corti).clamp(0.3, 2.0);
        let h2v: Vec<f32> = self.l2.forward(&recf).iter().map(|x| x.max(0.0) * gain).collect(); let h2: [f32; NH2] = arr(&h2v);
        let outv = self.act.forward(&h2v);
        let mut out = [0.0f32; NOUT];
        out[0] = outv[0].tanh();                          // virar ∈ [-1,1]
        for k in 1..NOUT { out[k] = sig(outv[k]); }       // acelerar/comer/beber/descansar + 8 bits de fala ∈ [0,1]
        let pred: [f32; NIN] = arr(&self.wm.forward(&h2v).iter().map(|x| x.tanh()).collect::<Vec<_>>());
        Cognition { mem, h1, rec, h2, out, pred }
    }
    fn genome(&self) -> Vec<f32> {
        let mut g = Vec::with_capacity(GENOME);
        g.extend_from_slice(&self.l1.weights); g.extend_from_slice(&self.l2.weights); g.extend_from_slice(&self.l2.bias);
        g.extend_from_slice(&self.act.weights); g.extend_from_slice(&self.act.bias);
        g.extend_from_slice(&self.wm.weights); g.extend_from_slice(&self.wm.bias);
        g.push(inv_sig((self.cfc_p.decay_rate - 0.4) / 3.0)); g.push(inv_sig((self.cfc_r.decay_rate - 0.3) / 3.0)); g
    }
}
fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn inv_sig(p: f32) -> f32 { let p = p.clamp(1e-4, 1.0 - 1e-4); (p / (1.0 - p)).ln() }
fn arr<const N: usize>(v: &[f32]) -> [f32; N] { let mut a = [0.0; N]; for i in 0..N { a[i] = v[i]; } a }

// ================== SER ==================
struct Being {
    id: u32, x: f32, y: f32, head: f32,
    energy: f32, hydra: f32, stamina: f32, health: f32, age: u32, gener: u32, alive: bool,
    dopa: f32, corti: f32, curio: f32, adeno: f32,       // neuromoduladores
    brain: Brain, cog: Cognition, senses: [f32; NIN], prev_pred: [f32; NIN], last_act: usize,
    // acumuladores p/ fitness/métricas
    sat_sum: f64, perr_sum: f64, act_cnt: [u32; NOUT], drink_thirsty: f64, eat_hungry: f64, nest_ticks: u32, born: u32,
    trail: Vec<[f32; 2]>, act_hist: Vec<u8>,   // rota + histórico de ações (só modos ao vivo)
    byte: u8, neigh: f32, neigh_sum: f64,       // byte que o ser FALA (8 bits) + nº de vizinhos + soma (agrupamento)
    knowledge: [f32; NDOM], learned: u32,       // CONHECIMENTO por domínio (0..1) + nº de aprendizados (comer=aprender)
}
fn new_cog() -> Cognition { Cognition { mem:[0.0;NIN], h1:[0.0;NH1], rec:[0.0;NH1], h2:[0.0;NH2], out:[0.0;NOUT], pred:[0.0;NIN] } }
impl Being {
    fn from_genome(id: u32, x: f32, y: f32, gener: u32, born: u32, g: &[f32], r: &mut Rng) -> Being {
        Being { id, x, y, head: r.f() * std::f32::consts::TAU,
            energy: 70.0, hydra: 70.0, stamina: 80.0, health: 100.0, age: 0, gener, alive: true,
            dopa: 0.0, corti: 0.0, curio: 0.0, adeno: 0.0,
            brain: Brain::from_genome(g), cog: new_cog(), senses: [0.0; NIN], prev_pred: [0.0; NIN], last_act: 0,
            sat_sum: 0.0, perr_sum: 0.0, act_cnt: [0; NOUT], drink_thirsty: 0.0, eat_hungry: 0.0, nest_ticks: 0, born,
            trail: Vec::new(), act_hist: Vec::new(), byte: 0, neigh: 0.0, neigh_sum: 0.0,
            knowledge: [0.0; NDOM], learned: 0 }
    }
    fn satisfaction(&self) -> f32 { (self.energy + self.hydra + self.stamina) / 300.0 }
    fn fitness(&self) -> f64 {
        let ticks = self.age.max(1) as f64;
        let sat = self.sat_sum / ticks;                         // satisfação média de necessidades
        let pred_acc = (1.0 - (self.perr_sum / ticks)).max(0.0); // acurácia do world-model
        let group = self.neigh_sum / ticks;                     // tamanho médio de grupo (inteligência coletiva)
        let know = self.knowledge.iter().sum::<f32>() as f64 / NDOM as f64;   // amplitude de conhecimento adquirido
        ticks * (0.5 + 0.5 * sat) + 120.0 * pred_acc + 22.0 * group + 300.0 * know
    }
}

// ================== MUNDO ==================
#[derive(Clone)]
struct World { size: f32, food: Vec<(f32, f32, bool, f32)>, water: Vec<(f32, f32)>, nests: Vec<(f32, f32, f32)>,
    preds: Vec<(f32, f32)>, day_len: u32, regrow: f32, metab: f32, night_drain: f32 }
impl World {
    fn new(nfood: usize, nwater: usize, nnest: usize, npred: usize, r: &mut Rng) -> World {
        World { size: 1.0,
            food: (0..nfood).map(|_| (r.f(), r.f(), true, 0.0)).collect(),
            water: (0..nwater).map(|_| (r.f(), r.f())).collect(),
            nests: (0..nnest).map(|_| (r.f(), r.f(), 0.09)).collect(),
            preds: (0..npred).map(|_| (r.f(), r.f())).collect(),
            day_len: 600, regrow: 40.0, metab: 3.0, night_drain: 5.5 }
    }
    fn is_night(&self, t: u32) -> bool { (t % self.day_len) >= self.day_len / 2 }
    fn day_phase(&self, t: u32) -> f32 { (t % self.day_len) as f32 / self.day_len as f32 * std::f32::consts::TAU }
}
// TERRENO POR FREQUÊNCIA (função livre): cada domínio é uma banda de frequência (baixa=continente, alta=ilha).
// domain_at = qual domínio "vence" em (x,y) → os domínios se formam no terreno pela frequência (ideia do codec).
fn domain_at(x: f32, y: f32) -> usize {
    use std::f32::consts::TAU;
    let mut best = f32::MIN; let mut bd = 0;
    for k in 0..NDOM {
        let f = 1.0 + k as f32 * 1.25;              // frequência cresce com o domínio
        let ph = k as f32 * 2.1;
        let v = (TAU*f*x + ph).sin() + (TAU*f*y + ph*1.7).cos() + 0.55*(TAU*f*1.6*(x+y) + ph).sin();
        if v > best { best = v; bd = k; }
    }
    bd
}
fn wd(a: f32, b: f32) -> f32 { let mut d = a - b; while d > 0.5 { d -= 1.0; } while d < -0.5 { d += 1.0; } d }
fn field(x: f32, y: f32, pts: &[(f32, f32)], sig: f32) -> f32 {
    let mut s = 0.0; for &(px, py) in pts { let (dx, dy) = (wd(x, px), wd(y, py)); s += (-(dx*dx+dy*dy)/(sig*sig)).exp(); } s.min(2.0)
}

// senses do ser (glass-box, [-1,1])
fn sense(b: &Being, w: &World, t: u32, neigh_d: f32, neigh_align: f32, heard: &[f32; NMSG]) -> [f32; NIN] {
    let hl = [b.x + 0.03*(b.head+0.6).cos(), b.y + 0.03*(b.head+0.6).sin()];
    let hr = [b.x + 0.03*(b.head-0.6).cos(), b.y + 0.03*(b.head-0.6).sin()];
    let foods: Vec<(f32,f32)> = w.food.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect();
    let (fl, fr) = (field(hl[0],hl[1],&foods,0.10), field(hr[0],hr[1],&foods,0.10));
    let (wl, wr) = (field(hl[0],hl[1],&w.water,0.10), field(hr[0],hr[1],&w.water,0.10));
    let (pl, pr) = (field(hl[0],hl[1],&w.preds,0.11), field(hr[0],hr[1],&w.preds,0.11));
    let pprox = field(b.x,b.y,&w.preds,0.10);
    let mut best = 9.0; let (mut ndx, mut ndy) = (0.0, 0.0);
    for &(nx,ny,_) in &w.nests { let (dx,dy)=(wd(nx,b.x),wd(ny,b.y)); let d=dx*dx+dy*dy; if d<best { best=d; ndx=dx; ndy=dy; } }
    let nn = best.sqrt().max(1e-3);
    let ph = w.day_phase(t);
    // CONHECIMENTO ABRE OS SENTIDOS: sem saber, o ser é quase cego pra água/perigo/abrigo/social
    let kw = &b.knowledge;
    let (gf,gw,gp,gn,gs) = (0.5+0.5*kw[0], 0.05+0.95*kw[1], 0.05+0.95*kw[2], 0.12+0.88*kw[3], 0.12+0.88*kw[4]);
    let (fl,fr)=(fl*gf,fr*gf); let (wl,wr)=(wl*gw,wr*gw); let (pl,pr,pprox)=(pl*gp,pr*gp,pprox*gp); let (ndx,ndy)=(ndx*gn,ndy*gn);
    let mut s = [0.0f32; NIN];
    s[0]=tanh(fl-fr); s[1]=tanh(fl+fr-0.5); s[2]=tanh(wl-wr); s[3]=tanh(wl+wr-0.5);
    s[4]=tanh(pl-pr); s[5]=tanh(pl+pr); s[6]=pprox.min(1.0); s[7]=ndx/nn; s[8]=ndy/nn;
    s[9]=tanh(neigh_d); s[10]=neigh_align; s[11]=1.0-b.energy/100.0; s[12]=1.0-b.hydra/100.0;
    s[13]=b.adeno; s[14]=b.corti; s[15]=b.health/100.0; s[16]=ph.sin(); s[17]=ph.cos();
    for k in 0..NMSG { s[18+k]=heard[k]*gs; }   // 8 bits ouvidos — só valem se o ser SABE do social
    s[NIN-1]=1.0;                               // viés
    s
}
fn tanh(x: f32) -> f32 { x.tanh() }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.iter().find(|a| a.starts_with("--")).map(|s| s.as_str()).unwrap_or("--evolve");
    match mode {
        "--evolve" => evolve(args.iter().position(|a| a=="--evolve").and_then(|i| args.get(i+1)).and_then(|s| s.parse().ok()).unwrap_or(40)),
        "--record" => record(),
        "--watch"  => watch(),
        "--serve"  => serve(),
        "--lab"    => lab(),
        "--validate" => validate(),
        _ => evolve(40),
    }
}

// VALIDAÇÃO por ABLAÇÃO: o mesmo campeão COM aprender (comer=conhecimento) vs SEM aprender → conhecimento = sobrevivência?
fn validate() {
    let mut r0 = Rng(1);
    let g = load_genome("vida_champion.dna").unwrap_or_else(|| { println!("(sem vida_champion.dna — evolua antes; aleatório)"); random_genome(&mut r0) });
    println!("VALIDAÇÃO — conhecimento = sobrevivência?  (ablação: comer=aprender LIGADO vs DESLIGADO, 5 mundos)\n");
    for (label, learn) in [("APRENDE (comer=conhecimento)", true), ("NAO aprende (ablacao)", false)] {
        LEARN.store(learn, Ordering::Relaxed);
        let (mut life, mut nb, mut learned) = (0f64, 0f64, 0f64); let mut know = [0f64; NDOM];
        for seed in 0..5u64 {
            let mut r = Rng(300 + seed);
            let mut w = World::new(80, 6, 4, 4, &mut r);
            let mut beings: Vec<Being> = (0..60).map(|i| { let mut cg=g.clone(); for v in cg.iter_mut(){ if r.f()<0.1 {*v+=r.g()*0.1;} } Being::from_genome(i, r.f(), r.f(), 0, 0, &cg, &mut r) }).collect();
            for t in 0..1800u32 { step_world(&mut beings, &mut w, t, &mut r, false, false); if beings.iter().all(|b| !b.alive) { break; } }
            for b in &beings { life += b.age as f64; nb += 1.0; learned += b.learned as f64; for k in 0..NDOM { know[k] += b.knowledge[k] as f64; } }
        }
        print!("  {:<32} vida media {:>6.0} ticks · aprendizados/ser {:>4.1} · conhecimento:", label, life/nb, learned/nb);
        for k in 0..NDOM { print!(" {} {:.2}", DOM_NAMES[k], know[k]/nb); }
        println!();
    }
    LEARN.store(true, Ordering::Relaxed);
    println!("\n→ se APRENDE vive bem mais que a ablação, CONHECIMENTO = SOBREVIVÊNCIA está validado numericamente.");
}

// ================== LAB GRÁFICO NATIVO (Bevy) — a engine Rust renderizando o cérebro por dentro ==================
#[derive(Resource)]
struct Sim { w: World, beings: Vec<Being>, tick: u32, sel: Option<u32>, paused: bool, speed: f32, acc: f32, rng: Rng, g: Vec<f32>, nextid: u32, follow: bool }
#[derive(Component)] struct Hud;
#[derive(Component)] struct Help;

fn lab() {
    let mut rng = Rng(3);
    let g = load_genome("vida_champion.dna").unwrap_or_else(|| { println!("(sem vida_champion.dna — evolua antes; usando aleatório)"); random_genome(&mut rng) });
    let mut w = World::new(130, 7, 5, 5, &mut rng);
    let beings: Vec<Being> = (0..70).map(|i| { let mut cg = g.clone();
        for v in cg.iter_mut() { if rng.f() < 0.12 { *v += rng.g() * 0.12; } } Being::from_genome(i, rng.f(), rng.f(), 0, 0, &cg, &mut rng) }).collect();
    println!("IARA-VIDA LAB (Bevy) — clique num ser p/ inspecionar · Espaço=pausa · ↑↓=veloc · .=passo · P=perturbar cérebro · S=spawn · F=comida · X=predador · R=reset · TAB=seguir");
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin { primary_window: Some(Window {
            title: "IARA-VIDA · o cérebro de um ser, por dentro".into(), resolution: (1400.0, 780.0).into(),
            present_mode: bevy::window::PresentMode::AutoVsync, ..default() }), ..default() }))
        .insert_resource(ClearColor(Color::srgb(0.02, 0.03, 0.05)))
        .insert_resource(Sim { w, beings, tick: 0, sel: None, paused: false, speed: 5.0, acc: 0.0, rng, g, nextid: 900, follow: true })
        .add_systems(Startup, setup_lab)
        .add_systems(Update, (input_lab, btn_click, step_lab, draw_lab, hud_lab).chain())
        .add_systems(Update, btn_hover)
        .run();
}
// ---- UI componentizada ----
#[derive(Clone, Copy, PartialEq)] enum Act { PlayPause, Faster, Slower, Step, Spawn, Food, Pred, Reset, Follow, Perturb }
#[derive(Component)] struct Btn(Act);
#[derive(Component)] struct SpeedTag;
#[derive(Component)] struct PlayTag;
const BTN_BG: Color = Color::srgb(0.12,0.16,0.24);
const BTN_HOV: Color = Color::srgb(0.20,0.28,0.42);
const BTN_PRESS: Color = Color::srgb(0.16,0.45,0.55);
const TOPH: f32 = 48.0; const BOTH: f32 = 150.0;

fn apply_act(a: Act, s: &mut Sim) {
    match a {
        Act::PlayPause => s.paused = !s.paused,
        Act::Faster => s.speed = (s.speed*1.5).min(60.0),
        Act::Slower => s.speed = (s.speed/1.5).max(0.3),
        Act::Follow => s.follow = !s.follow,
        Act::Step => { step_world(&mut s.beings,&mut s.w,s.tick,&mut s.rng,true,true); s.tick+=1; }
        Act::Reset => { let mut rr=Rng(s.tick as u64+7); let g=s.g.clone(); s.w=World::new(130,7,5,5,&mut rr);
            s.beings=(0..70).map(|i|{let mut cg=g.clone(); for v in cg.iter_mut(){if rr.f()<0.12{*v+=rr.g()*0.12;}} Being::from_genome(i,rr.f(),rr.f(),0,0,&cg,&mut rr)}).collect(); s.tick=0; s.sel=None; }
        Act::Spawn => { let mut cg=s.g.clone(); for v in cg.iter_mut(){if s.rng.f()<0.2{*v+=s.rng.g()*0.2;}} let (x,y)=(s.rng.f(),s.rng.f()); let id=s.nextid; s.nextid+=1; let mut b=Being::from_genome(id,x,y,0,s.tick,&cg,&mut s.rng); b.brain.reset(); s.beings.push(b); }
        Act::Food => { for _ in 0..8 { let (x,y)=(s.rng.f(),s.rng.f()); s.w.food.push((x,y,true,0.0)); } }
        Act::Pred => { let (x,y)=(s.rng.f(),s.rng.f()); s.w.preds.push((x,y)); }
        Act::Perturb => { if let Some(id)=s.sel { if let Some(b)=s.beings.iter_mut().find(|b|b.id==id){ let mut cg=b.brain.genome(); for v in cg.iter_mut(){if s.rng.f()<0.25{*v=(*v+s.rng.g()*0.5).clamp(-3.0,3.0);}} b.brain=Brain::from_genome(&cg);} } }
    }
}
fn btn_click(q: Query<(&Interaction,&Btn),(Changed<Interaction>,With<Button>)>, mut sim: ResMut<Sim>) {
    for (i,b) in &q { if *i==Interaction::Pressed { let s=&mut *sim; apply_act(b.0, s); } }
}
fn btn_hover(mut q: Query<(&Interaction,&mut BackgroundColor),(Changed<Interaction>,With<Button>)>) {
    for (i,mut bg) in &mut q { bg.0 = match *i { Interaction::Pressed=>BTN_PRESS, Interaction::Hovered=>BTN_HOV, Interaction::None=>BTN_BG }; }
}
fn node_btn(w: f32) -> Node { Node { width: Val::Px(w), height: Val::Px(30.0), justify_content: JustifyContent::Center,
    align_items: AlignItems::Center, margin: UiRect::right(Val::Px(4.0)), ..default() } }
fn tf(s: f32) -> TextFont { TextFont { font_size: s, ..default() } }
fn setup_lab(mut c: Commands) {
    c.spawn(Camera2d::default());
    // rótulos do cérebro em world-space (alinham com as colunas em qualquer tamanho de janela) + título do mundo
    for (x, name) in BLBL { c.spawn((Text2d::new(name), tf(15.0), TextColor(Color::srgb(0.55,0.68,0.85)), Transform::from_xyz(x, 332.0, 1.0))); }
    c.spawn((Text2d::new("MUNDO VIVO"), tf(15.0), TextColor(Color::srgb(0.5,0.6,0.75)), Transform::from_xyz(-360.0, 340.0, 1.0)));
    // ── BARRA DE CONTROLE (topo, botões clicáveis) ──
    c.spawn((Node { position_type: PositionType::Absolute, left: Val::Px(0.0), top: Val::Px(0.0), width: Val::Percent(100.0),
        height: Val::Px(TOPH), flex_direction: FlexDirection::Row, align_items: AlignItems::Center, padding: UiRect::horizontal(Val::Px(10.0)), ..default() },
        BackgroundColor(Color::srgba(0.05,0.07,0.11,0.97)))).with_children(|bar| {
        bar.spawn((Text::new("IARA-VIDA"), tf(16.0), TextColor(Color::srgb(0.35,0.85,0.7)), Node { margin: UiRect::right(Val::Px(14.0)), ..default() }));
        bar.spawn((Button, node_btn(74.0), BackgroundColor(BTN_BG), Btn(Act::PlayPause))).with_children(|b| { b.spawn((Text::new("PAUSAR"), tf(13.0), TextColor(Color::WHITE), PlayTag)); });
        bar.spawn((Button, node_btn(30.0), BackgroundColor(BTN_BG), Btn(Act::Slower))).with_children(|b| { b.spawn((Text::new("-"), tf(17.0), TextColor(Color::WHITE))); });
        bar.spawn((Node { width: Val::Px(96.0), justify_content: JustifyContent::Center, align_items: AlignItems::Center, ..default() },))
            .with_children(|b| { b.spawn((Text::new("veloc 5.0/s"), tf(12.0), TextColor(Color::srgb(0.7,0.82,0.98)), SpeedTag)); });
        bar.spawn((Button, node_btn(30.0), BackgroundColor(BTN_BG), Btn(Act::Faster))).with_children(|b| { b.spawn((Text::new("+"), tf(17.0), TextColor(Color::WHITE))); });
        for (lbl, act, w) in [("passo",Act::Step,58.0f32),("+ ser",Act::Spawn,60.0),("+ comida",Act::Food,86.0),("+ predador",Act::Pred,96.0),("perturbar",Act::Perturb,84.0),("seguir",Act::Follow,60.0),("reset",Act::Reset,56.0)] {
            bar.spawn((Button, node_btn(w), BackgroundColor(BTN_BG), Btn(act))).with_children(|b| { b.spawn((Text::new(lbl), tf(13.0), TextColor(Color::srgb(0.86,0.91,1.0)))); });
        }
    });
    // ── PAINEL INSPETOR (base) ──
    c.spawn((Node { position_type: PositionType::Absolute, left: Val::Px(0.0), bottom: Val::Px(0.0), width: Val::Percent(100.0),
        height: Val::Px(BOTH), padding: UiRect::all(Val::Px(12.0)), ..default() }, BackgroundColor(Color::srgba(0.05,0.07,0.11,0.95))))
        .with_children(|p| { p.spawn((Text::new("clique num ser (bolinha) pra inspecionar o cérebro dele."), tf(13.0), TextColor(Color::srgb(0.82,0.9,1.0)), Hud)); });
}
// colunas do cérebro (world-x) + rótulo — usadas por setup e draw_brain
const BLBL: [(f32, &str); 7] = [(30.0,"sensores"),(170.0,"percepção"),(300.0,"instinto"),(418.0,"memória"),(520.0,"pensamento"),(605.0,"ação"),(660.0,"previsão")];
// mapeia mundo [0,1]² → metade esquerda da tela; cérebro na metade direita
fn wpos(x: f32, y: f32) -> Vec2 { Vec2::new(-360.0 + (x - 0.5) * 640.0, (0.5 - y) * 640.0) }
fn npos(cx: f32, i: usize, n: usize) -> Vec2 { let span = 600.0; let y = if n > 1 { span * 0.5 - i as f32 * span / (n - 1) as f32 } else { 0.0 }; Vec2::new(cx, y) }
fn act_col(a: usize) -> Color { match a { 2 => Color::srgb(0.48,0.83,0.41), 3 => Color::srgb(0.29,0.64,1.0), 4 => Color::srgb(0.6,0.52,1.0), 5 => Color::srgb(1.0,0.71,0.33), _ => Color::srgb(0.42,0.5,0.62) } }
fn actf(v: f32) -> Color { let a = v.abs().min(1.0); if v >= 0.0 { Color::srgba(0.2,0.82,0.65,0.25+a*0.75) } else { Color::srgba(1.0,0.71,0.33,0.25+a*0.75) } }
fn gc(g: &mut Gizmos, p: Vec2, r: f32, c: Color) { g.circle_2d(Isometry2d::from_translation(p), r, c); }

fn step_lab(time: Res<Time>, mut sim: ResMut<Sim>) {
    if sim.paused { return; }
    let dt = time.delta_secs();
    sim.acc += sim.speed * dt;
    let mut budget = 0;
    while sim.acc >= 1.0 && budget < 40 {
        let s = &mut *sim;
        step_world(&mut s.beings, &mut s.w, s.tick, &mut s.rng, true, true);
        s.tick += 1; s.acc -= 1.0; budget += 1;
    }
    // segue o protagonista (ser vivo mais velho) se estiver no modo seguir ou se o selecionado morreu
    let alive_sel = sim.sel.map_or(false, |id| sim.beings.iter().any(|b| b.id == id));
    if sim.follow || !alive_sel {
        if let Some(id) = sim.beings.iter().max_by_key(|b| b.age).map(|b| b.id) { sim.sel = Some(id); }
    }
}

fn input_lab(mut sim: ResMut<Sim>, keys: Res<ButtonInput<KeyCode>>, mouse: Res<ButtonInput<MouseButton>>,
             windows: Query<&Window>, cams: Query<(&Camera, &GlobalTransform), With<Camera2d>>) {
    if keys.just_pressed(KeyCode::Space) { sim.paused = !sim.paused; }
    if keys.just_pressed(KeyCode::ArrowUp) { sim.speed = (sim.speed * 1.5).min(60.0); }
    if keys.just_pressed(KeyCode::ArrowDown) { sim.speed = (sim.speed / 1.5).max(0.3); }
    if keys.just_pressed(KeyCode::Tab) { sim.follow = !sim.follow; }
    if keys.just_pressed(KeyCode::Period) { let s = &mut *sim; step_world(&mut s.beings, &mut s.w, s.tick, &mut s.rng, true, true); s.tick += 1; }
    if keys.just_pressed(KeyCode::KeyR) { let mut rr = Rng(sim.tick as u64 + 7); let g = sim.g.clone();
        sim.w = World::new(130,7,5,5,&mut rr); sim.beings = (0..70).map(|i| { let mut cg=g.clone(); for v in cg.iter_mut(){ if rr.f()<0.12 {*v+=rr.g()*0.12;} } Being::from_genome(i, rr.f(), rr.f(), 0, 0, &cg, &mut rr) }).collect(); sim.tick=0; sim.sel=None; }
    if keys.just_pressed(KeyCode::KeyS) { let s=&mut *sim; let mut cg=s.g.clone(); for v in cg.iter_mut(){ if s.rng.f()<0.2 {*v+=s.rng.g()*0.2;} }
        let (x,y)=(s.rng.f(),s.rng.f()); let id=s.nextid; s.nextid+=1; let mut b=Being::from_genome(id,x,y,0,s.tick,&cg,&mut s.rng); b.brain.reset(); s.beings.push(b); }
    if keys.just_pressed(KeyCode::KeyF) { let s=&mut *sim; for _ in 0..8 { let (x,y)=(s.rng.f(),s.rng.f()); s.w.food.push((x,y,true,0.0)); } }
    if keys.just_pressed(KeyCode::KeyX) { let s=&mut *sim; let (x,y)=(s.rng.f(),s.rng.f()); s.w.preds.push((x,y)); }
    if keys.just_pressed(KeyCode::KeyP) { if let Some(id)=sim.sel { let s=&mut *sim; if let Some(b)=s.beings.iter_mut().find(|b| b.id==id) {
        let mut cg=b.brain.genome(); for v in cg.iter_mut(){ if s.rng.f()<0.25 { *v=(*v+s.rng.g()*0.5).clamp(-3.0,3.0); } } b.brain=Brain::from_genome(&cg); } } }
    // clique = inspecionar o ser mais próximo
    if mouse.just_pressed(MouseButton::Left) {
        if let (Ok(win), Ok((cam, ct))) = (windows.single(), cams.single()) {
            if let Some(cur) = win.cursor_position() {
                if cur.y <= TOPH || cur.y >= win.height() - BOTH { return; }   // ignora cliques na barra/painel
                if let Ok(wp) = cam.viewport_to_world_2d(ct, cur) {
                    let mut best = 22.0; let mut bid = None;
                    for b in &sim.beings { let p = wpos(b.x, b.y); let d = p.distance(wp); if d < best { best = d; bid = Some(b.id); } }
                    if bid.is_some() { sim.sel = bid; sim.follow = false; }
                }
            }
        }
    }
}

// cor a partir do BYTE (a "linguagem" vira cor): matiz = valor do byte
fn byte_col(byte: u8, l: f32, a: f32) -> Color { Color::hsla(byte as f32 / 255.0 * 360.0, 0.78, l.clamp(0.0,1.0), a.clamp(0.0,1.0)) }
fn dom_col(d: usize, l: f32, a: f32) -> Color { Color::hsla(d as f32 / NDOM as f32 * 340.0, 0.62, l.clamp(0.0,1.0), a.clamp(0.0,1.0)) }
fn draw_lab(mut giz: Gizmos, sim: Res<Sim>) {
    let night = sim.w.is_night(sim.tick);
    let dim = if night { 0.6 } else { 1.0 };
    // TERRENO POR FREQUÊNCIA: fundo de domínios (cada bioma uma cor) — os domínios se formam no espaço
    let gr = 30usize;
    for gy in 0..gr { for gx in 0..gr { let x=(gx as f32+0.5)/gr as f32; let y=(gy as f32+0.5)/gr as f32;
        gc(&mut giz, wpos(x,y), 2.2, dom_col(domain_at(x,y), 0.5, 0.09*dim)); } }
    giz.rect_2d(Isometry2d::from_translation(Vec2::new(-360.0, 0.0)), Vec2::new(646.0, 646.0), Color::srgba(0.2,0.3,0.5,0.14));
    for nst in &sim.w.nests { let p=wpos(nst.0,nst.1); let rr=nst.2*640.0; gc(&mut giz,p,rr,Color::srgba(0.6,0.52,1.0,0.30)); gc(&mut giz,p,rr*0.6,Color::srgba(0.6,0.52,1.0,0.12)); }
    for wt in &sim.w.water { let p=wpos(wt.0,wt.1); gc(&mut giz,p,3.5,Color::srgb(0.4,0.72,1.0)); gc(&mut giz,p,10.0,Color::srgba(0.29,0.64,1.0,0.22)); gc(&mut giz,p,18.0,Color::srgba(0.29,0.64,1.0,0.07)); }
    // COMIDA colorida pelo DOMÍNIO de conhecimento que ela ensina (comida = frase/bytes)
    for f in &sim.w.food { if f.2 { gc(&mut giz, wpos(f.0,f.1), 1.9, dom_col(domain_at(f.0,f.1), 0.55, dim)); } }
    // TEIA SOCIAL: liga vizinhos próximos; cor = média dos bytes → grupos que "falam igual" viram uma constelação
    let bs = &sim.beings;
    for i in 0..bs.len() { if !bs[i].alive { continue; }
        for j in (i+1)..bs.len() { if !bs[j].alive { continue; }
            let dd = wd(bs[i].x,bs[j].x).powi(2)+wd(bs[i].y,bs[j].y).powi(2);
            if dd < 0.018 { let a=(1.0-dd/0.018)*0.22*dim; let byte=(((bs[i].byte as u16)+(bs[j].byte as u16))/2) as u8;
                giz.line_2d(wpos(bs[i].x,bs[i].y), wpos(bs[j].x,bs[j].y), byte_col(byte,0.55,a)); } } }
    // SERES: glow + corpo colorido pelo BYTE falado + rastro + pulso de fala animado
    for b in bs { if !b.alive { continue; } let p=wpos(b.x,b.y); let e=(b.energy/100.0).clamp(0.0,1.0);
        let know = b.knowledge.iter().sum::<f32>()/NDOM as f32;    // conhecimento total (0..1)
        let st=b.trail.len().saturating_sub(8);
        for k in (st+1)..b.trail.len() { giz.line_2d(wpos(b.trail[k-1][0],b.trail[k-1][1]),wpos(b.trail[k][0],b.trail[k][1]), byte_col(b.byte,0.5,0.06*dim)); }
        if know>0.05 { gc(&mut giz, p, 5.0+know*9.0, Color::srgba(1.0,0.95,0.75,know*0.16*dim)); }  // AURA de conhecimento (dourada, cresce c/ a mente)
        gc(&mut giz, p, 3.0+e*4.0, byte_col(b.byte, 0.42, 0.14*dim));   // glow do byte
        gc(&mut giz, p, 1.6+e*2.2, byte_col(b.byte, 0.32+e*0.35, dim)); // corpo
        if b.neigh>0.0 && b.byte!=0 { let ph=((sim.tick.wrapping_add(b.id))%24) as f32/24.0; gc(&mut giz,p,4.0+ph*11.0, byte_col(b.byte,0.62,(1.0-ph)*0.16*dim)); }
    }
    for pr in &sim.w.preds { let p=wpos(pr.0,pr.1); gc(&mut giz,p,4.0,Color::srgb(1.0,0.3,0.36)); gc(&mut giz,p,9.0,Color::srgba(1.0,0.2,0.3,0.25)); gc(&mut giz,p,16.0,Color::srgba(1.0,0.15,0.25,0.09)); }
    // selecionado: destaque branco + rota forte + CÉREBRO
    if let Some(b) = sim.sel.and_then(|id| sim.beings.iter().find(|b| b.id==id)) {
        let p = wpos(b.x, b.y);
        gc(&mut giz, p, 10.0, Color::WHITE); gc(&mut giz, p, 14.0, Color::srgba(1.0,1.0,1.0,0.3));
        for k in 1..b.trail.len() { giz.line_2d(wpos(b.trail[k-1][0],b.trail[k-1][1]),wpos(b.trail[k][0],b.trail[k][1]),Color::srgba(1.0,1.0,1.0,0.4)); }
        draw_brain(&mut giz, b, &sim.g);
    }
}

// desenha o CÉREBRO COMPLETO do ser (metade direita): sensores→mem→instinto→memória→pensamento→ação+world-model
fn draw_brain(giz: &mut Gizmos, b: &Being, _g: &[f32]) {
    let w = b.brain.genome();
    let mut o = 0; let take = |n: usize, o: &mut usize| { let s = w[*o..*o+n].to_vec(); *o += n; s };
    let l1 = take(NIN*NH1,&mut o); let _l2b_off = NH1*NH2; let l2 = take(NH1*NH2,&mut o); let _l2bias = take(NH2,&mut o);
    let act = take(NH2*NOUT,&mut o); let _actb = take(NOUT,&mut o); let wm = take(NH2*NIN,&mut o);
    let cS=BLBL[0].0; let cM=BLBL[1].0; let cL1=BLBL[2].0; let cR=BLBL[3].0; let cL2=BLBL[4].0; let cO=BLBL[5].0; let cW=BLBL[6].0;
    let pS:Vec<Vec2>=(0..NIN).map(|i| npos(cS,i,NIN)).collect();
    let pM:Vec<Vec2>=(0..NIN).map(|i| npos(cM,i,NIN)).collect();
    let pL1:Vec<Vec2>=(0..NH1).map(|i| npos(cL1,i,NH1)).collect();
    let pR:Vec<Vec2>=(0..NH1).map(|i| npos(cR,i,NH1)).collect();
    let pL2:Vec<Vec2>=(0..NH2).map(|i| npos(cL2,i,NH2)).collect();
    let pO:Vec<Vec2>=(0..NOUT).map(|i| npos(cO,i,NOUT)).collect();
    let pP:Vec<Vec2>=(0..NIN).map(|i| npos(cW,i,NIN)).collect();
    let mx=|a:&[f32]| a.iter().fold(1e-6f32,|m,v| m.max(v.abs()));
    let (mL1,mL2,mA,mW)=(mx(&l1),mx(&l2),mx(&act),mx(&wm));
    let syn=|giz:&mut Gizmos,a:&[Vec2],b:&[Vec2],wt:&[f32],ci:usize,ro:usize,m:f32|{
        for j in 0..ro { for i in 0..ci { let ww=wt[j*ci+i]; if ww.abs()<m*0.55 {continue;}
            let col=if ww>=0.0 {Color::srgba(0.2,0.82,0.65,(ww.abs()/m*0.4).min(0.38))} else {Color::srgba(1.0,0.36,0.42,(ww.abs()/m*0.4).min(0.38))};
            giz.line_2d(a[i],b[j],col); } } };
    // sinapses (por baixo)
    syn(giz,&pM,&pL1,&l1,NIN,NH1,mL1); syn(giz,&pR,&pL2,&l2,NH1,NH2,mL2);
    syn(giz,&pL2,&pO,&act,NH2,NOUT,mA); syn(giz,&pL2,&pP,&wm,NH2,NIN,mW);
    // links de memória CfC (tracejado)
    for i in 0..NIN { giz.line_2d(pS[i],pM[i],Color::srgba(0.29,0.78,1.0,0.22)); }
    for i in 0..NH1 { giz.line_2d(pL1[i],pR[i],Color::srgba(0.29,0.78,1.0,0.22)); }
    // nós (ativações)
    let cog=&b.cog;
    let node=|giz:&mut Gizmos,p:Vec2,r:f32,v:f32|{ let a=v.abs().min(1.0);
        let glow=if v>=0.0 {Color::srgba(0.2,0.82,0.65,a*0.22)} else {Color::srgba(1.0,0.71,0.33,a*0.22)};
        if a>0.35 { gc(giz,p,r+2.0+a*4.0,glow); } gc(giz,p,r,actf(v)); };
    for i in 0..NIN { node(giz,pS[i],4.0,b.senses[i]); }
    for i in 0..NIN { node(giz,pM[i],4.0,cog.mem[i]); }
    for i in 0..NH1 { node(giz,pL1[i],4.5,cog.h1[i]); }
    for i in 0..NH1 { node(giz,pR[i],4.5,cog.rec[i]); }
    for i in 0..NH2 { node(giz,pL2[i],5.0,cog.h2[i]); }
    for i in 0..NOUT { node(giz,pO[i],6.0,cog.out[i]); }
    gc(giz,pO[b.last_act],10.0,act_col(b.last_act));   // anel na AÇÃO escolhida
    for i in 0..NIN { node(giz,pP[i],3.5,cog.pred[i]); }
}

fn hud_lab(sim: Res<Sim>,
    mut hq: Query<&mut Text, (With<Hud>, Without<SpeedTag>, Without<PlayTag>)>,
    mut sq: Query<&mut Text, (With<SpeedTag>, Without<Hud>, Without<PlayTag>)>,
    mut pq: Query<&mut Text, (With<PlayTag>, Without<Hud>, Without<SpeedTag>)>) {
    if let Ok(mut t) = sq.single_mut() { *t = Text::new(format!("veloc {:.1}/s", sim.speed)); }
    if let Ok(mut t) = pq.single_mut() { *t = Text::new(if sim.paused { "RODAR" } else { "PAUSAR" }); }
    let bar = |v: f32, mx: f32| { let n=16usize; let f=((v/mx)*n as f32).round().clamp(0.0,n as f32) as usize; format!("[{}{}]","#".repeat(f)," ".repeat(n-f)) };
    // MÉTRICAS DE INTELIGÊNCIA COLETIVA (validação): agrupamento, vocabulário-byte, sobrevivência
    let np = sim.beings.len().max(1) as f32;
    let avg_neigh = sim.beings.iter().map(|b| b.neigh).sum::<f32>() / np;
    let avg_age = sim.beings.iter().map(|b| b.age).sum::<u32>() as f32 / np;
    let mut seen = [false; 256]; let mut vocab = 0; for b in &sim.beings { if !seen[b.byte as usize] { seen[b.byte as usize]=true; vocab+=1; } }
    let mut s = format!("tick {} · {} · pop {}      COLETIVO →  grupo medio {:.1}  ·  vocabulario {} bytes  ·  idade media {:.0}",
        sim.tick, if sim.w.is_night(sim.tick) {"NOITE"} else {"DIA"}, sim.beings.len(), avg_neigh, vocab, avg_age);
    match sim.sel.and_then(|id| sim.beings.iter().find(|b| b.id==id)) {
        Some(b) => {
            s += &format!("        SER #{} · idade {} · geracao {} · ACAO: {}\n", b.id, b.age, b.gener, ACT_NAMES[b.last_act].to_uppercase());
            s += &format!("energia {} {:>3.0}    hidratacao {} {:>3.0}    vigor {} {:>3.0}    saude {} {:>3.0}\n",
                bar(b.energy,100.0),b.energy, bar(b.hydra,100.0),b.hydra, bar(b.stamina,100.0),b.stamina, bar(b.health,100.0),b.health);
            s += &format!("dopamina {}  cortisol {}  curiosidade {}  fadiga {}\n", bar(b.dopa,1.0),bar(b.corti,1.0),bar(b.curio,1.0),bar(b.adeno,1.0));
            let tot: u32 = b.act_cnt.iter().sum::<u32>().max(1);
            s += &format!("historico de acoes:  ");
            for i in 0..5 { s += &format!("{} {:.0}%   ", ACT_NAMES[i], b.act_cnt[i] as f32/tot as f32*100.0); }
            s += &format!("   ·   FALA byte {:03} ({:08b}) · ouve {} vizinhos", b.byte, b.byte, b.neigh as u32);
            s += &format!("\nCONHECIMENTO (comeu {} frases):  ", b.learned);
            for k in 0..NDOM { let fill=((b.knowledge[k]*4.0).round() as usize).min(4); s += &format!("{} [{}{}]  ", DOM_NAMES[k], "#".repeat(fill), "-".repeat(4-fill)); }
        }
        None => { s += "        — clique numa bolinha do mundo pra inspecionar o cerebro do ser —"; }
    }
    if let Ok(mut t) = hq.single_mut() { *t = Text::new(s); }
}

// ---- carrega/salva genoma (formato simples f32 LE; e i8 quantizado estilo Universe) ----
fn save_genome(path: &str, g: &[f32]) { let mut buf = Vec::new(); for &x in g { buf.extend_from_slice(&x.to_le_bytes()); }
    std::fs::write(path, buf).ok(); }
fn load_genome(path: &str) -> Option<Vec<f32>> { let b = std::fs::read(path).ok()?; if b.len() < GENOME*4 { return None; }
    Some((0..GENOME).map(|i| f32::from_le_bytes([b[i*4],b[i*4+1],b[i*4+2],b[i*4+3]])).collect()) }
fn random_genome(r: &mut Rng) -> Vec<f32> { (0..GENOME).map(|_| r.g() * 0.5).collect() }

// ================== EVOLUÇÃO (esquema real do Universe) ==================
fn eval_pop(genomes: &[Vec<f32>], seeds: &[u64]) -> Vec<f64> {
    // avalia cada genoma em VÁRIOS mundos (média) → seleção robusta, não sortuda
    let mut fit = vec![0.0f64; genomes.len()];
    for &seed in seeds {
        let mut r = Rng(seed);
        let mut w = World::new(80, 6, 4, 4, &mut r);
        let mut beings: Vec<Being> = genomes.iter().enumerate()
            .map(|(i, g)| Being::from_genome(i as u32, r.f(), r.f(), 0, 0, g, &mut r)).collect();
        for t in 0..1800u32 { step_world(&mut beings, &mut w, t, &mut r, false, false); if beings.iter().all(|b| !b.alive) { break; } }
        for b in &beings { fit[b.id as usize] += b.fitness(); }
    }
    for f in fit.iter_mut() { *f /= seeds.len() as f64; }
    fit
}
fn evolve(gens: u32) {
    let mut r = Rng(20260707);
    let (pop, elite) = (60usize, 15usize);
    let mut genomes: Vec<Vec<f32>> = (0..pop).map(|_| random_genome(&mut r)).collect();
    let mut champ = genomes[0].clone(); let mut champ_fit = -1.0;
    println!("IARA-VIDA evolve — pop {} · genoma {} pesos · mundo multi-necessidade (comida/água/ninho/predador/dia-noite)\n", pop, GENOME);
    for gn in 0..gens {
        let fits = eval_pop(&genomes, &[100 + gn as u64, 555 + gn as u64, 909 + gn as u64, 222 + gn as u64, 777 + gn as u64]);
        let mut idx: Vec<usize> = (0..pop).collect(); idx.sort_by(|&a, &b| fits[b].partial_cmp(&fits[a]).unwrap());
        if fits[idx[0]] > champ_fit { champ_fit = fits[idx[0]]; champ = genomes[idx[0]].clone(); }
        let mean: f64 = fits.iter().sum::<f64>() / pop as f64;
        if gn % 5 == 0 || gn == gens - 1 { println!("  gen {:>2}  melhor {:>7.0}  média {:>7.0}", gn, fits[idx[0]], mean); }
        let elites: Vec<Vec<f32>> = idx[..elite].iter().map(|&i| genomes[i].clone()).collect();
        let mut next = elites.clone();
        let rate = (0.09 * 0.97f32.powi(gn as i32) * 2.0).min(0.25);
        while next.len() < pop {
            let tour = |r: &mut Rng| { let mut b = usize::MAX; for _ in 0..3 { let c = (r.f()*elite as f32) as usize; if c<b { b=c; } } b };
            let (p1, p2) = (&elites[tour(&mut r)], &elites[tour(&mut r)]);
            let mut child = vec![0.0; GENOME]; let bs = 48; let mut i = 0;
            while i < GENOME { let end = (i+bs).min(GENOME); let src = if r.f()<0.5 { p1 } else { p2 };
                child[i..end].copy_from_slice(&src[i..end]); i = end; }
            for v in child.iter_mut() { if r.f() < rate { *v = (*v + r.g()*0.05).clamp(-3.0, 3.0); } }
            next.push(child);
        }
        genomes = next;
    }
    save_genome("vida_champion.dna", &champ);
    println!("\ncampeão fitness {:.0} → salvo vida_champion.dna ({} pesos f32)", champ_fit, GENOME);
    // relatório do campeão (métricas de inteligência)
    report_being(&champ);
}
fn report_being(g: &[f32]) {
    let cap = 3000u32; let seeds = [7u64, 42, 314, 2718, 1618];
    let mut lifes = Vec::new(); let mut acts = [0u32; NOUT];
    let (mut eat_h, mut drink_t, mut nest, mut pacc, mut tt) = (0.0f64, 0.0f64, 0u32, 0.0f64, 0.0f64);
    for &sd in &seeds {
        let mut r = Rng(sd);
        let mut w = World::new(80, 6, 4, 4, &mut r);
        let mut b = vec![Being::from_genome(0, 0.5, 0.5, 0, 0, g, &mut r)];
        for t in 0..cap { step_world(&mut b, &mut w, t, &mut r, false, false); if !b[0].alive { break; } }
        let bb = &b[0]; lifes.push(bb.age);
        for i in 0..NOUT { acts[i] += bb.act_cnt[i]; }
        let tk = bb.age.max(1) as f64; tt += tk;
        eat_h += bb.eat_hungry; drink_t += bb.drink_thirsty; nest += bb.nest_ticks; pacc += (1.0 - bb.perr_sum/tk).max(0.0);
    }
    let tot: u32 = acts.iter().sum::<u32>().max(1);
    let ent: f64 = acts.iter().map(|&c| { let p = c as f64/tot as f64; if p>0.0 {-p*p.ln()} else {0.0} }).sum();
    let mean_life = lifes.iter().sum::<u32>() as f64 / seeds.len() as f64;
    println!("\n=== MENTE DO CAMPEÃO ({} mundos com predadores + dia/noite, até {} ticks) ===", seeds.len(), cap);
    println!("  vida média {:.0} ticks · {:?} · {} sobreviveu ao teto", mean_life, lifes, lifes.iter().filter(|&&l| l>=cap-1).count());
    print!("  repertório: "); for i in 0..NOUT { print!("{} {:.0}% · ", ACT_NAMES[i], acts[i] as f64/tot as f64*100.0); }
    println!("\n  entropia-ações {:.2} · comer-c/fome {:.3} · beber-c/sede {:.3} · ninho {:.0}% · acurácia-previsão {:.2}",
        ent, eat_h/tt, drink_t/tt, nest as f64/tt*100.0, pacc/seeds.len() as f64);
}

// ================== UM TICK DO MUNDO ==================
fn step_world(beings: &mut Vec<Being>, w: &mut World, t: u32, r: &mut Rng, allow_repro: bool, remove_dead: bool) {
    let night = w.is_night(t);
    for f in w.food.iter_mut() { if !f.2 { f.3 -= 1.0; if f.3 <= 0.0 { f.2 = true; f.0 = r.f(); f.1 = r.f(); } } }
    // predadores caçam (mais rápidos à noite)
    let pspeed = if night { 0.010 } else { 0.006 };
    for pi in 0..w.preds.len() { let (px, py) = w.preds[pi]; let mut best = 9.0; let (mut dx, mut dy) = (0.0, 0.0);
        for b in beings.iter() { if b.alive { let (a, c) = (wd(b.x, px), wd(b.y, py)); let d = a*a+c*c; if d < best { best=d; dx=a; dy=c; } } }
        let n = best.sqrt().max(1e-3); w.preds[pi] = ((px + dx/n*pspeed).rem_euclid(1.0), (py + dy/n*pspeed).rem_euclid(1.0)); }
    // senso coletivo: densidade + alinhamento local, e sinal (quem viu predador vocaliza)
    let poss: Vec<(f32,f32,f32,f32,bool)> = beings.iter().map(|b| (b.x,b.y,b.head.cos(),b.head.sin(),b.alive)).collect();
    let prev_bytes: Vec<u8> = beings.iter().map(|b| b.byte).collect();   // byte que cada ser FALAVA (tick anterior)
    let n = beings.len();
    let mut newborns: Vec<Being> = Vec::new();
    let mut nextid = beings.iter().map(|b| b.id).max().unwrap_or(0) + 1;
    for i in 0..n {
        if !beings[i].alive { continue; }
        // vizinhança: densidade + alinhamento + BYTE OUVIDO (média ponderada dos bits dos vizinhos)
        let (mut dens, mut ax, mut ay, mut wsum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut heard = [0.0f32; NMSG];
        for j in 0..n { if j!=i && poss[j].4 { let dd = wd(beings[i].x,poss[j].0).powi(2)+wd(beings[i].y,poss[j].1).powi(2);
            if dd < 0.025 { let wgt = 1.0 - dd/0.025; dens += 1.0; ax += poss[j].2; ay += poss[j].3; wsum += wgt;
                for k in 0..NMSG { if (prev_bytes[j] >> k) & 1 == 1 { heard[k] += wgt; } else { heard[k] -= wgt; } } } } }
        if wsum > 0.0 { for k in 0..NMSG { heard[k] = (heard[k]/wsum).clamp(-1.0,1.0); } }
        let align = if dens>0.0 { (ax*beings[i].head.cos()+ay*beings[i].head.sin())/dens } else { 0.0 };
        let s = sense(&beings[i], w, t, dens*0.15, align, &heard);
        // surpresa (world-model): erro entre previsão anterior e sensação atual
        let perr: f32 = (0..NIN).map(|k| (beings[i].prev_pred[k] - s[k]).abs()).sum::<f32>() / NIN as f32;
        beings[i].curio = 0.9*beings[i].curio + 0.1*perr;
        let (dopa, corti) = (beings[i].dopa, beings[i].corti);
        let cog = beings[i].brain.think(&s, dopa, corti);
        // AÇÃO
        let turn = cog.out[0]; let thrust = cog.out[1];
        let (eat, drink, rest) = (cog.out[2], cog.out[3], cog.out[4]);
        // BYTE que o ser FALA agora (8 bits de mensagem)
        let mut byte = 0u8; for k in 0..NMSG { if cog.out[5+k] > 0.5 { byte |= 1 << k; } }
        let b = &mut beings[i];
        b.byte = byte; b.neigh = dens; b.neigh_sum += dens as f64;
        // consumir (comer/beber/descansar) REDUZ locomoção → o ser tem que ESCOLHER viajar vs consumir
        let consum = eat.max(drink).max(rest);
        let mthr = thrust * (1.0 - 0.65 * consum);
        b.head = (b.head + turn*3.0*DT).rem_euclid(std::f32::consts::TAU);
        b.x = (b.x + mthr*0.5*DT*b.head.cos()).rem_euclid(1.0);
        b.y = (b.y + mthr*0.5*DT*b.head.sin()).rem_euclid(1.0);
        let in_nest = w.nests.iter().any(|&(nx,ny,rr)| wd(b.x,nx).powi(2)+wd(b.y,ny).powi(2) < rr*rr);
        // metabolismo REAL (parado morre): energia ~0.3/tick + esforço; à noite fora do ninho drena mais
        let mut ecost = w.metab*DT + 6.0*mthr*DT*(1.0 - 0.45*b.knowledge[5]);   // NAVEGAÇÃO: sabe atravessar → gasta menos
        if night && !in_nest { ecost += w.night_drain*DT*(1.0 - 0.7*b.knowledge[3]); }   // ABRIGO: conhecimento sobrevive à noite
        b.energy -= ecost; b.hydra -= (2.0 + 3.0*mthr)*DT; b.stamina -= (1.2*mthr + if night && !in_nest {0.8} else {0.2})*DT;
        let hungry = b.energy < 55.0; let thirsty = b.hydra < 55.0;
        let (mut ate, mut drank, mut rested) = (false, false, false);
        if eat > 0.5 { for f in w.food.iter_mut() { if f.2 && wd(b.x,f.0).powi(2)+wd(b.y,f.1).powi(2) < 0.03*0.03 {
            let d = domain_at(f.0, f.1);                    // comida carrega o CONHECIMENTO do domínio onde nasceu
            f.2 = false; f.3 = w.regrow; b.energy = (b.energy+22.0).min(100.0); b.dopa = (b.dopa+0.6).min(1.0);
            if LEARN.load(Ordering::Relaxed) { b.knowledge[d] = (b.knowledge[d] + 0.16).min(1.0); b.learned += 1; }   // COMER = APRENDER
            ate = true; if hungry { b.eat_hungry += 1.0; } break; } } }
        if drink > 0.5 && field(b.x,b.y,&w.water,0.05) > 0.4 { b.hydra = (b.hydra+11.0).min(100.0); b.dopa = (b.dopa+0.3).min(1.0); drank = true; if thirsty { b.drink_thirsty += 1.0; } }
        if rest > 0.5 && in_nest { b.stamina = (b.stamina+5.0).min(100.0); b.health = (b.health+1.2).min(100.0); rested = true; }
        if in_nest { b.nest_ticks += 1; }
        let fear = field(b.x,b.y,&w.preds,0.10);
        b.corti = (0.92*b.corti + 0.4*fear).clamp(0.0, 1.0);
        b.adeno = (b.adeno + 0.35*mthr*DT - if rested {1.0*DT} else {0.0}).clamp(0.0, 1.0);
        b.dopa *= 0.90;
        // PREDAÇÃO com DILUIÇÃO: predador confunde no grupo — pega o ISOLADO (seleção por AGRUPAMENTO)
        if fear > 0.85 { for &(px,py) in &w.preds { if wd(b.x,px).powi(2)+wd(b.y,py).powi(2) < 0.022*0.022 {
            if r.f() < 1.0/(1.0 + 0.7*b.neigh) { b.health -= 34.0; b.corti = 1.0; } } } }
        // ethograma pelo que REALMENTE aconteceu (bits de fala não são "ação")
        let la = if ate {2} else if drank {3} else if rested {4} else {0};
        b.last_act = la; b.act_cnt[la] += 1;
        if allow_repro {   // rota + histórico só nos modos ao vivo (não pesa a evolução)
            b.trail.push([b.x, b.y]); let l = b.trail.len(); if l > 60 { b.trail.drain(0..l-60); }
            b.act_hist.push(la as u8); let h = b.act_hist.len(); if h > 160 { b.act_hist.drain(0..h-160); }
        }
        // acumuladores
        b.sat_sum += b.satisfaction() as f64; b.perr_sum += perr as f64;
        b.age += 1; b.prev_pred = cog.pred; b.cog = cog; b.senses = s;
        // morte
        if b.energy <= 0.0 || b.hydra <= 0.0 || b.health <= 0.0 { b.alive = false; }
        // reprodução (a-life mode)
        if allow_repro && b.energy > 82.0 && b.stamina > 40.0 && b.age > 120 && n + newborns.len() < 260 {
            b.energy -= 40.0;
            let mut cg = b.brain.genome();
            for v in cg.iter_mut() { if r.f() < 0.12 { *v = (*v + r.g()*0.08).clamp(-3.0, 3.0); } }
            let mut child = Being::from_genome(nextid, (b.x+0.01).rem_euclid(1.0), b.y, b.gener+1, t, &cg, r);
            child.brain.reset(); nextid += 1; newborns.push(child);
        }
    }
    if remove_dead { beings.retain(|b| b.alive); }
    for c in newborns { beings.push(c); }
}

// ================== WATCH (ASCII, pra mim validar) ==================
fn watch() {
    let mut r = Rng(1);
    let g = load_genome("vida_champion.dna").unwrap_or_else(|| { println!("(sem vida_champion.dna, usando aleatório)"); random_genome(&mut r) });
    let mut w = World::new(90, 6, 4, 5, &mut r);
    let mut beings: Vec<Being> = (0..60).map(|i| { let mut cg = g.clone();
        for v in cg.iter_mut() { if r.f()<0.1 { *v += r.g()*0.1; } } Being::from_genome(i, r.f(), r.f(), 0, 0, &cg, &mut r) }).collect();
    let (gw, gh) = (64usize, 24usize);
    let mut tick = 0u32;
    for frame in 0..16 {
        for _ in 0..300 { step_world(&mut beings, &mut w, tick, &mut r, true, true); tick += 1; if beings.is_empty() { break; } }
        let mut grid = vec![vec![b' '; gw]; gh];
        for &(x,y,a,_) in &w.food { if a { grid[(y*gh as f32) as usize % gh][(x*gw as f32) as usize % gw] = b'.'; } }
        for &(x,y) in &w.water { grid[(y*gh as f32) as usize % gh][(x*gw as f32) as usize % gw] = b'~'; }
        for &(x,y,_) in &w.nests { grid[(y*gh as f32) as usize % gh][(x*gw as f32) as usize % gw] = b'#'; }
        for b in &beings { let c = match b.last_act { 2=>b'e',3=>b'd',4=>b'r',5=>b's',_=>b'o' };
            grid[(b.y*gh as f32) as usize % gh][(b.x*gw as f32) as usize % gw] = c; }
        for &(x,y) in &w.preds { grid[(y*gh as f32) as usize % gh][(x*gw as f32) as usize % gw] = b'X'; }
        let night = w.is_night(tick);
        println!("frame {:>2} tick {:>5} pop {:>3} {}  (o=anda e=come d=bebe r=descansa s=sinaliza X=predador ~=água #=ninho)",
            frame, tick, beings.len(), if night {"NOITE"} else {"dia"});
        println!("+{}+", "-".repeat(gw));
        for row in &grid { println!("|{}|", String::from_utf8_lossy(row)); }
        if beings.is_empty() { println!("espécie extinta"); break; }
    }
}

// ================== RECORD (trace JSON p/ a UI) ==================
fn record() {
    let mut r = Rng(2);
    let g = load_genome("vida_champion.dna").unwrap_or_else(|| { println!("(sem vida_champion.dna — evolua antes; usando aleatório)"); random_genome(&mut r) });
    let mut w = World::new(95, 6, 4, 5, &mut r);
    let mut beings: Vec<Being> = (0..70).map(|i| { let mut cg = g.clone();
        for v in cg.iter_mut() { if r.f()<0.1 { *v += r.g()*0.12; } } Being::from_genome(i, r.f(), r.f(), 0, 0, &cg, &mut r) }).collect();
    let frames_n = 1000u32; let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("\"dims\":{{\"nin\":{},\"nh1\":{},\"nh2\":{},\"nout\":{}}},\n", NIN, NH1, NH2, NOUT));
    out.push_str(&format!("\"sense_names\":{},\n", jstrs(&SENSE_NAMES)));
    out.push_str(&format!("\"act_names\":{},\n", jstrs(&ACT_NAMES)));
    out.push_str(&format!("\"water\":{},\n", jpairs(&w.water)));
    out.push_str(&format!("\"nests\":{},\n", jtrip(&w.nests)));
    out.push_str(&format!("\"day_len\":{},\n", w.day_len));
    out.push_str(&format!("\"genome\":{},\n", jflt(&g)));   // pesos do campeão → UI desenha sinapses reais
    out.push_str("\"frames\":[\n");
    let mut tick = 0u32;
    for fr in 0..frames_n {
        for _ in 0..2 { step_world(&mut beings, &mut w, tick, &mut r, true, true); tick += 1; }
        if beings.is_empty() { break; }
        // protagonista = o ser vivo mais velho (estável; a UI segue o mesmo cérebro)
        let fi = (0..beings.len()).max_by_key(|&i| beings[i].age).unwrap();
        let focal = &beings[fi];
        let mut fs = String::from("{");
        fs.push_str(&format!("\"t\":{},\"night\":{},", tick, if w.is_night(tick) {1} else {0}));
        // população (flat, arredondado)
        fs.push_str("\"bx\":["); for (k,b) in beings.iter().enumerate() { if k>0 {fs.push(',');} fs.push_str(&r3(b.x)); } fs.push_str("],");
        fs.push_str("\"by\":["); for (k,b) in beings.iter().enumerate() { if k>0 {fs.push(',');} fs.push_str(&r3(b.y)); } fs.push_str("],");
        fs.push_str("\"be\":["); for (k,b) in beings.iter().enumerate() { if k>0 {fs.push(',');} fs.push_str(&format!("{}",(b.energy as i32).max(0))); } fs.push_str("],");
        fs.push_str("\"ba\":["); for (k,b) in beings.iter().enumerate() { if k>0 {fs.push(',');} fs.push_str(&format!("{}",b.last_act)); } fs.push_str("],");
        fs.push_str(&format!("\"fx\":{},", jflt(&w.food.iter().filter(|f|f.2).map(|f|f.0).collect::<Vec<_>>())));
        fs.push_str(&format!("\"fy\":{},", jflt(&w.food.iter().filter(|f|f.2).map(|f|f.1).collect::<Vec<_>>())));
        fs.push_str(&format!("\"px\":{},", jflt(&w.preds.iter().map(|p|p.0).collect::<Vec<_>>())));
        fs.push_str(&format!("\"py\":{},", jflt(&w.preds.iter().map(|p|p.1).collect::<Vec<_>>())));
        // cérebro focal completo
        fs.push_str(&format!("\"focal\":{{\"id\":{},\"x\":{},\"y\":{},\"act\":{},", focal.id, r3(focal.x), r3(focal.y), focal.last_act));
        fs.push_str(&format!("\"needs\":[{},{},{},{}],", ri(focal.energy), ri(focal.hydra), ri(focal.stamina), ri(focal.health)));
        fs.push_str(&format!("\"mod\":[{},{},{},{}],", r3(focal.dopa), r3(focal.corti), r3(focal.curio), r3(focal.adeno)));
        fs.push_str(&format!("\"senses\":{},", jarr(&focal.senses)));
        fs.push_str(&format!("\"mem\":{},\"h1\":{},\"rec\":{},\"h2\":{},\"out\":{},\"pred\":{}",
            jarr(&focal.cog.mem), jarr(&focal.cog.h1), jarr(&focal.cog.rec), jarr(&focal.cog.h2), jarr(&focal.cog.out), jarr(&focal.cog.pred)));
        fs.push_str("}}");
        if fr>0 { out.push_str(",\n"); } out.push_str(&fs);
    }
    out.push_str("\n]}\n");
    std::fs::create_dir_all("vida").ok();
    std::fs::write("vida/trace.json", &out).ok();
    std::fs::write("vida/trace.js", format!("window.VIDA_TRACE={};", out)).ok(); // carrega via file:// na UI
    println!("gravado vida/trace.js + trace.json ({} KB, {} frames, focal glass-box com pesos)", out.len()/1024, frames_n);
}

// helpers JSON
fn r3(x: f32) -> String { format!("{:.3}", x) }
fn ri(x: f32) -> String { format!("{}", x as i32) }
fn jarr(a: &[f32]) -> String { let mut s = String::from("["); for (i,x) in a.iter().enumerate() { if i>0 {s.push(',');} s.push_str(&r3(*x)); } s.push(']'); s }
fn jflt(a: &[f32]) -> String { let mut s = String::from("["); for (i,x) in a.iter().enumerate() { if i>0 {s.push(',');} s.push_str(&r3(*x)); } s.push(']'); s }
fn jpairs(a: &[(f32,f32)]) -> String { let mut s=String::from("["); for (i,p) in a.iter().enumerate(){ if i>0{s.push(',');} s.push_str(&format!("[{},{}]",r3(p.0),r3(p.1))); } s.push(']'); s }
fn jtrip(a: &[(f32,f32,f32)]) -> String { let mut s=String::from("["); for (i,p) in a.iter().enumerate(){ if i>0{s.push(',');} s.push_str(&format!("[{},{},{}]",r3(p.0),r3(p.1),r3(p.2))); } s.push(']'); s }
fn jstrs(a: &[&str]) -> String { let mut s=String::from("["); for (i,x) in a.iter().enumerate(){ if i>0{s.push(',');} s.push_str(&format!("\"{}\"",x)); } s.push(']'); s }

// ================== SERVE (HTTP std-only; UI dirige o Rust real) — stub inicial ==================
fn serve() {
    let mut r = Rng(3);
    let g = load_genome("vida_champion.dna").unwrap_or_else(|| random_genome(&mut r));
    let listener = std::net::TcpListener::bind("127.0.0.1:8099").expect("bind 8099");
    println!("IARA-VIDA serve :8099 (endpoints /reset /step /state) — UI ao vivo");
    let mut w = World::new(95, 6, 4, 5, &mut r);
    let mut beings: Vec<Being> = (0..70).map(|i| Being::from_genome(i, r.f(), r.f(), 0, 0, &g, &mut r)).collect();
    let mut tick = 0u32;
    for stream in listener.incoming() {
        let mut s = match stream { Ok(s) => s, Err(_) => continue };
        let mut buf = [0u8; 1024]; let nread = s.read(&mut buf).unwrap_or(0);
        let req = String::from_utf8_lossy(&buf[..nread]);
        let path = req.split_whitespace().nth(1).unwrap_or("/");
        if path.starts_with("/step") { for _ in 0..3 { step_world(&mut beings, &mut w, tick, &mut r, true, true); tick += 1; } }
        else if path.starts_with("/reset") { let mut rr = Rng(tick as u64 + 9); w = World::new(95,6,4,5,&mut rr);
            beings = (0..70).map(|i| Being::from_genome(i, rr.f(), rr.f(), 0, 0, &g, &mut rr)).collect(); tick = 0; }
        let body = state_json(&beings, &w, tick);
        let resp = format!("HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
        s.write_all(resp.as_bytes()).ok();
    }
}
fn state_json(beings: &[Being], w: &World, t: u32) -> String {
    let mut o = String::from("{");
    o.push_str(&format!("\"t\":{},\"night\":{},\"pop\":{},", t, if w.is_night(t){1}else{0}, beings.len()));
    o.push_str("\"bx\":["); for (k,b) in beings.iter().enumerate(){ if k>0{o.push(',');} o.push_str(&r3(b.x)); } o.push_str("],");
    o.push_str("\"by\":["); for (k,b) in beings.iter().enumerate(){ if k>0{o.push(',');} o.push_str(&r3(b.y)); } o.push_str("]}");
    o
}
