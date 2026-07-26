//! CARCAÇA IARA no Universe (headless) — a ARQUITETURA da IARA (não os pesos) num ser leve e replicável.
//! Órgãos: NEUROMODULAÇÃO causal (dopa/corti/adeno) + FLORESTA associativa (situação→ação, força de sinapse) +
//! órgão PREDITIVO (forward model) + ROTEADOR (reflexo/deliberação). Agnóstico: nasce sem saber nada e DESCOBRE
//! o laço ação↔percepção vivendo (bater dói → cortisol → grava evitar). Rust puro, sem Bevy → milhares.
//! Uso: cargo run --release --bin carcaca -- [--curva | --bateria | --milhares]
use std::collections::HashMap;
const ARENA: f32 = 40.0;
const NOBST: usize = 7;
const NACT: usize = 4; // 0=acelera 1=freia 2=vira_esq 3=vira_dir
fn clamp(x: f32, a: f32, b: f32) -> f32 { if x < a { a } else if x > b { b } else { x } }

struct Rng(u64);
impl Rng { fn new(s: u64) -> Self { Rng(s ^ 0x9E3779B97F4A7C15) }
    fn u(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f32 { (self.u() >> 40) as f32 / (1u64 << 24) as f32 }
    fn r(&mut self, a: f32, b: f32) -> f32 { a + (b - a) * self.f() } }

// ===================== A CARCAÇA: os órgãos IARA =====================
struct Carcass {
    dopa: f32, corti: f32, adeno: f32,                 // NEUROMODULAÇÃO
    forest: HashMap<(i32, i32), [f32; NACT]>,          // FLORESTA: situação(dist_b,vel_b) → força da sinapse por ação
    fwd: HashMap<(i32, i32, usize), (f32, u32)>,       // órgão PREDITIVO: (sit,ação) → (Δdist previsto, n)
    neuro: bool, plast: bool, pred: bool,
    last: Option<((i32, i32), usize)>,
}
impl Carcass {
    fn new(neuro: bool, plast: bool, pred: bool) -> Self {
        Carcass { dopa: 0.0, corti: 0.0, adeno: 0.0, forest: HashMap::new(), fwd: HashMap::new(), neuro, plast, pred, last: None }
    }
    fn plast_coef(&self) -> f32 {                      // PLASTICIDADE causal: dopamina/dor abrem, fadiga fecha
        if self.plast { 0.15 + 0.6 * self.corti + 0.3 * self.dopa - 0.1 * self.adeno } else { 0.12 }
    }
    fn decide(&mut self, sit: (i32, i32), dist: f32, rng: &mut Rng) -> usize {
        let base = *self.forest.get(&sit).unwrap_or(&[0.0; NACT]);
        let mut score = base;
        if self.pred {                                 // ROTEADOR System 2: se o preditivo prevê colisão, pune (esp. acelerar)
            for a in 0..NACT {
                let dp = self.fwd.get(&(sit.0, sit.1, a)).map(|x| x.0).unwrap_or(0.0);
                if dist + dp < 2.5 && a == 0 { score[a] -= 2.5; }
            }
        }
        if self.neuro && dist < 3.0 { score[1] += 1.5 * self.corti; }  // reflexo: cortisol → frear quando perto
        let m = score.iter().cloned().fold(f32::MIN, f32::max);
        let ex: Vec<f32> = score.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = ex.iter().sum(); let r = rng.f() * z; let mut acc = 0.0; let mut a_ = NACT - 1;
        for a in 0..NACT { acc += ex[a]; if r <= acc { a_ = a; break; } }
        self.last = Some((sit, a_)); a_
    }
    fn learn(&mut self, reward: f32, d0: f32, d1: f32) {
        if let Some((sit, act)) = self.last {
            if self.pred {                             // forward model (auto-supervisionado): "andar move a cena"
                let e = self.fwd.entry((sit.0, sit.1, act)).or_insert((0.0, 0));
                e.0 = (e.0 * e.1 as f32 + (d1 - d0)) / (e.1 + 1) as f32; e.1 = (e.1 + 1).min(50);
            }
            if self.plast {                            // FLORESTA: fortalece/enfraquece a sinapse gated pela neuromod
                let c = self.plast_coef();
                let row = self.forest.entry(sit).or_insert([0.0; NACT]);
                row[act] = clamp(row[act] + c * reward, -3.0, 3.0);
            }
        }
    }
    fn decay(&mut self, moving: bool) {
        self.dopa *= 0.85; self.corti *= 0.8;
        self.adeno = clamp(self.adeno + if moving { 0.01 } else { -0.02 }, 0.0, 1.0);
    }
}

// ===================== o corpo + o mundo (arena com paredes) =====================
struct Obst { x: f32, y: f32, r: f32 }
struct World { obst: Vec<Obst> }
impl World {
    fn new(seed: u64) -> Self {
        let mut r = Rng::new(seed);
        let obst = (0..NOBST).map(|_| Obst { x: r.r(6.0, ARENA - 6.0), y: r.r(6.0, ARENA - 6.0), r: r.r(2.0, 3.5) }).collect();
        World { obst }
    }
    fn blocked(&self, x: f32, y: f32) -> bool {
        x < 0.0 || x > ARENA || y < 0.0 || y > ARENA || self.obst.iter().any(|o| (x - o.x).powi(2) + (y - o.y).powi(2) < o.r * o.r)
    }
}
struct Being { x: f32, y: f32, head: f32, vel: f32, health: f32, colis: u32, passos: u32, dist_tot: f32, brain: Carcass }
impl Being {
    fn new(seed: u64, w: &World, neuro: bool, plast: bool, pred: bool) -> Self {
        let mut r = Rng::new(seed ^ 0xABCD);
        let (mut x, mut y) = (r.r(5.0, ARENA - 5.0), r.r(5.0, ARENA - 5.0));
        for _ in 0..50 { if !w.obst.iter().any(|o| (x - o.x).powi(2) + (y - o.y).powi(2) < (o.r + 3.0).powi(2)) { break; } x = r.r(5.0, ARENA - 5.0); y = r.r(5.0, ARENA - 5.0); }
        Being { x, y, head: r.r(0.0, 6.28), vel: 0.3, health: 1.0, colis: 0, passos: 0, dist_tot: 0.0, brain: Carcass::new(neuro, plast, pred) }
    }
    fn sense(&self, w: &World) -> f32 {                // distância à parede/obstáculo À FRENTE
        let (dx, dy) = (self.head.cos(), self.head.sin());
        let mut d = ARENA;
        let mut r = 0.6;
        while r < 24.0 { let (px, py) = (self.x + dx * r, self.y + dy * r); if w.blocked(px, py) { d = r; break; } r += 0.6; }
        d
    }
    fn db(d: f32) -> i32 { ((d / 3.0) as i32).min(4) }
    fn vb(v: f32) -> i32 { ((v / 0.4) as i32).min(2) }
    fn step(&mut self, w: &World, rng: &mut Rng) {
        self.passos += 1;
        let d0 = self.sense(w); let sit = (Self::db(d0), Self::vb(self.vel));
        let a = self.brain.decide(sit, d0, rng);
        match a { 0 => self.vel = clamp(self.vel + 0.15, 0.1, 1.2), 1 => self.vel = clamp(self.vel - 0.2, 0.0, 1.2), 2 => self.head += 0.5, _ => self.head -= 0.5 }
        let (nx, ny) = (self.x + self.head.cos() * self.vel, self.y + self.head.sin() * self.vel);
        let mut reward = 0.0;
        if w.blocked(nx, ny) {                          // BATER DÓI → cortisol dispara → aprende a evitar
            self.colis += 1; self.health = (self.health - 0.03).max(0.0); self.brain.corti = (self.brain.corti + 0.8).min(1.0); reward = -1.0;
            self.x = clamp(self.x - self.head.cos() * 1.6, 1.0, ARENA - 1.0); self.y = clamp(self.y - self.head.sin() * 1.6, 1.0, ARENA - 1.0);
            self.vel = 0.1; self.head += rng.r(2.4, 3.9);
        } else {
            self.x = clamp(nx, 0.0, ARENA); self.y = clamp(ny, 0.0, ARENA); self.dist_tot += self.vel;
            if self.vel > 0.15 { self.brain.dopa = (self.brain.dopa + 0.03).min(1.0); reward = 0.15; }
        }
        let d1 = self.sense(w);
        self.brain.learn(reward, d0, d1);
        self.brain.decay(self.vel > 0.1);
        self.health = (self.health + 0.0015).min(1.0);
    }
}

fn live(neuro: bool, plast: bool, pred: bool, ticks: u32, seed: u64) -> (u32, f32, HashMap<(i32, i32), [f32; NACT]>) {
    let w = World::new(seed); let mut b = Being::new(seed, &w, neuro, plast, pred); let mut rng = Rng::new(seed ^ 0x1234);
    for _ in 0..ticks { b.step(&w, &mut rng); }
    (b.colis, b.dist_tot, b.brain.forest)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.iter().find(|a| a.starts_with("--")).map(|s| s.as_str()).unwrap_or("--curva");
    match mode {
        "--milhares" => {
            let t0 = std::time::Instant::now(); let n = 1000u64;
            for i in 0..n { live(true, true, true, 300, i); }
            println!("{} seres × 300 passos em {:.1}s — carcaça LEVE em Rust, replicável em milhares (sem pesos)", n, t0.elapsed().as_secs_f32());
        }
        "--bateria" => {
            println!("=== BATERIA (8 sementes, 3000 passos) — a carcaça aprende a navegar? ===\n");
            for (nome, (ne, pl, pr)) in [("CARCAÇA COMPLETA", (true, true, true)), ("SEM plasticidade", (true, false, true)), ("SEM neuromod", (false, true, true)), ("SEM preditivo", (true, true, false))] {
                let (mut col, mut dist) = (0u32, 0.0f32);
                for sd in 0..8 { let (c, d, _) = live(ne, pl, pr, 3000, sd); col += c; dist += d; }
                println!("  {:22} colisões {:4} · distância {:5.0} · dist/colisão {:5.1}", nome, col / 8, dist / 8.0, dist / (col.max(1)) as f32);
            }
        }
        _ => {
            println!("=== 1 ser vivendo 3000 passos — DESCOBRE o laço ação↔percepção sozinho ===\n");
            let (col, dist, forest) = live(true, true, true, 3000, 3);
            println!("colisões {} · distância {:.0}\n", col, dist);
            println!("MAPA SENSÓRIO-MOTOR aprendido (situação → força da sinapse [acel,freia,viraE,viraD]):");
            let mut keys: Vec<_> = forest.keys().cloned().collect(); keys.sort();
            for k in keys { let r = forest[&k]; println!("  dist_b={} vel_b={} → [{:>5.1},{:>5.1},{:>5.1},{:>5.1}]", k.0, k.1, r[0], r[1], r[2], r[3]); }
            println!("\n→ perto+rápido deve dar tudo negativo (evita); longe deve dar acelerar positivo. Emergiu sozinho.");
        }
    }
}
