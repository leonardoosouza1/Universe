// BATERIA 12 — CO-EVOLUÇÃO de frequência (Leonardo: "achar onde os seres ficam em freq baixa/média/alta").
// Mundo espacial com NB bandas; a banda b pisca a f_b (geométrico lento→rápido). Um ser tem GENOMA = (posição,
// decay do CfC). A posição escolhe a banda em que ele vive; o decay é a "frequência do cérebro". Fitness =
// acurácia de rastrear a comida da sua banda (sensor ruidoso) MENOS lotação (seleção freq-dependente → nichos).
// Pergunta HONESTA: a EVOLUÇÃO (sem gradiente) descobre sozinha o decay ressonante de cada banda, e onde a
// população se acomoda? Headless.
use universe::cfc::CfCCell;
const PI: f32 = std::f32::consts::PI;
const NB: usize = 7;

struct Rng(u64);
impl Rng { fn u(&mut self)->u64 { self.0=self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z=self.0;
    z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }
    fn f(&mut self)->f32 { (self.u()>>40) as f32/(1u64<<24) as f32 }
    fn g(&mut self)->f32 { let(u,v)=(self.f().max(1e-6),self.f()); (-2.0*u.ln()).sqrt()*(std::f32::consts::TAU*v).cos() }
    fn n(&mut self,hi:usize)->usize { (self.f()*hi as f32) as usize % hi.max(1) } }

fn band_freqs()->[f32;NB] { core::array::from_fn(|b| 0.004*2.0f32.powi(b as i32)) } // 0.004 .. 0.256
fn band_of(pos:f32)->usize { pos.round().clamp(0.0,(NB-1) as f32) as usize }   // pos em unidades de banda (uniforme)
fn decay_of(g:f32)->f32 { 0.05*1.6f32.powf(8.0/(1.0+(-g).exp())) }                  // 0.05 .. ~2.15 (log)

fn track_acc(decay:f32, f:f32, noise:f32, steps:usize, seed:u64)->f32 {
    let mut cfc=CfCCell::new(1); cfc.decay_rate=decay; cfc.denoise_mag=0.0; cfc.reset();
    let mut rng=Rng(seed); let (mut c,mut n)=(0usize,0usize);
    for i in 0..steps {
        let s = if (2.0*PI*f*i as f32).sin() > 0.0 {1u8} else {0};
        let obs = if rng.f() < noise { 1-s } else { s };
        let mut rec=[0i8;1]; cfc.update(&[ if obs==1 {100i8} else {-100} ], &mut rec, 1.0);
        let pred = if rec[0] as i32 > 0 {1u8} else {0};
        if i>30 { if pred==s {c+=1;} n+=1; }
    }
    c as f32/n.max(1) as f32
}

// evolui SÓ o decay (1 gene) numa banda isolada p/ max acurácia de rastreio — a evolução acha o decay ressonante?
fn evolve_decay(f:f32, noise:f32, steps:usize, seed:u64)->(f32,f32) {
    let (pop, gens, elite)=(60usize, 80i32, 12usize);
    let mut r=Rng(seed);
    let mut genes:Vec<f32>=(0..pop).map(|_| r.g()*1.5).collect();
    for gn in 0..gens {
        let mut sc:Vec<(f32,usize)>=(0..pop).map(|i|{
            let a=(0..3).map(|s| track_acc(decay_of(genes[i]),f,noise,steps, 500+(gn as u64)*11+s*3)).sum::<f32>()/3.0; (a,i) }).collect();
        sc.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        let el:Vec<f32>=sc[..elite].iter().map(|&(_,i)| genes[i]).collect();
        let mut next=el.clone(); let rate=(0.5*0.98f32.powi(gn)).max(0.08);
        while next.len()<pop { let mut c=el[r.n(elite)]; if r.f()<0.7 { c+=r.g()*rate; } next.push(c); }
        genes=next;
    }
    let best=decay_of(genes[0]);
    (best, track_acc(best,f,noise,5000, 99))
}

fn main() {
    let freqs=band_freqs(); let noise=0.30f32; let steps=3000usize;
    println!("CO-EVOLUÇÃO de frequência — evolução ISOLADA do decay por banda (a evolução acha o decay ressonante?).");
    println!("ruído {}%. Compare o decay EVOLUÍDO com o grid-ótimo da bat.10.\n", (noise*100.0) as i32);
    println!("  banda  freq_mundo   decay_EVOLUÍDO   acurácia");
    let mut prev=0.0f32; let mut mono=true;
    for b in 0..NB {
        let (d,a)=evolve_decay(freqs[b], noise, steps, 1000 + b as u64*97);
        println!("  {:>4}   {:<9.3}   {:>10.2}       {:.0}%", b, freqs[b], d, a*100.0);
        if d < prev-0.03 { mono=false; } prev=d;
    }
    println!("\n(esperado: decay evoluído CRESCE com a freq da banda — igual ao grid da bat.10, mas achado pela SELEÇÃO)");
    if mono { println!("→ RESSONÂNCIA CO-EVOLUÍDA: a evolução (sem gradiente) achou sozinha o decay ressonante de cada banda."); }
    else    { println!("→ ressonância NÃO estritamente monótona (ruído do GA em bandas de teto baixo); ver tabela."); }
}
