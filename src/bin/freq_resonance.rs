// BATERIA 10 — RESSONÂNCIA DE FREQUÊNCIA (Leonardo: "mapa em níveis de frequência, onde os seres ficam").
// Mundo: a comida PISCA no tempo a uma frequência f (lento→rápido). O ser só vê uma observação RUIDOSA do
// estado (come? sim/não). O cérebro é um CfC de 1 canal cuja decay_rate é a "frequência do cérebro": decay
// baixo = integra lento (bom p/ piscar lento), decay alto = responde rápido (bom p/ piscar rápido).
// Pergunta HONESTA: o decay ÓTIMO (o sobrevivente) desloca com a frequência do mundo? Isso É a ressonância.
// Sem readout aprendido: o CfC denoisa e a gente limiariza — evidência limpa e determinística. Headless.
use universe::cfc::CfCCell;

const PI: f32 = std::f32::consts::PI;
struct Rng(u64);
impl Rng { fn u(&mut self)->u64 { self.0=self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z=self.0;
    z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }
    fn f(&mut self)->f32 { (self.u()>>40) as f32/(1u64<<24) as f32 } }

// acurácia de recuperar o estado LIMPO da comida a partir da observação ruidosa, com um CfC de dado decay
fn track_acc(decay:f32, f:f32, noise:f32, steps:usize, seed:u64)->f32 {
    let mut cfc=CfCCell::new(1); cfc.decay_rate=decay; cfc.denoise_mag=0.0; cfc.reset();
    let mut rng=Rng(seed); let (mut correct,mut n)=(0usize,0usize);
    for i in 0..steps {
        let s = if (2.0*PI*f*i as f32).sin() > 0.0 {1u8} else {0};     // comida presente?
        let obs = if rng.f() < noise { 1-s } else { s };               // sensor ruidoso
        let inp=[ if obs==1 {100i8} else {-100} ];
        let mut rec=[0i8;1]; cfc.update(&inp,&mut rec,1.0);            // CfC integra/denoisa
        let pred = if rec[0] as i32 > 0 {1u8} else {0};
        if i>30 { if pred==s {correct+=1;} n+=1; }                     // warmup do estado
    }
    correct as f32/n.max(1) as f32
}

fn main() {
    let noise=0.30f32;                                                 // 30% dos sensores mentem
    let steps=6000usize;
    let decays:Vec<f32>=(0..16).map(|k| 0.05*1.5f32.powi(k)).collect(); // grade log de "frequências do cérebro"
    let freqs:Vec<f32>=vec![0.004, 0.008, 0.016, 0.031, 0.062, 0.125, 0.25]; // mundo: lento → rápido
    println!("RESSONÂNCIA — comida pisca a freq f; cérebro CfC com decay d; ruído {}%. acurácia de rastrear o estado limpo.", (noise*100.0) as i32);
    println!("(cada linha: uma frequência do MUNDO; marcamos * o decay do SOBREVIVENTE = melhor acurácia)\n");
    print!("  world f \\ brain decay:");
    for d in &decays { print!("{:>6.2}", d); } println!();
    let mut peaks=vec![];
    for &f in &freqs {
        // média de algumas sementes p/ tirar o ruído
        let accs:Vec<f32>=decays.iter().map(|&d| {
            let s:f32=(0..4).map(|sd| track_acc(d,f,noise,steps, 1000+sd*7)).sum::<f32>()/4.0; s }).collect();
        let (bi,_)=accs.iter().enumerate().fold((0,0.0f32),|(bi,bv),(i,&v)| if v>bv {(i,v)} else {(bi,bv)});
        peaks.push((f, decays[bi], accs[bi]));
        print!("  f={:<6.3}          ", f);
        for (i,a) in accs.iter().enumerate() {
            if i==bi { print!("{:>5.0}*", a*100.0); } else { print!("{:>6.0}", a*100.0); }
        }
        println!();
    }
    println!("\n=== SOBREVIVENTE por frequência do mundo (decay ótimo) ===");
    let mut mono=true; let mut prev=0.0f32;
    for (f,d,a) in &peaks {
        println!("  mundo f={:<6.3}  →  cérebro decay ótimo {:<6.2}  (acurácia {:.0}%)", f, d, a*100.0);
        if *d < prev - 1e-6 { mono=false; } prev=*d;
    }
    if mono { println!("\n→ RESSONÂNCIA VALIDADA: o decay do sobrevivente CRESCE monotonicamente com a frequência do mundo."); }
    else    { println!("\n→ tendência de ressonância presente mas NÃO estritamente monótona (ver tabela)."); }
}
