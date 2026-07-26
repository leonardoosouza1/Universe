// PONTE SURVIVAL↔INTELIGÊNCIA (Leonardo 2026-07-08): um mundo onde COMER = PREVER BYTE.
// O ser recebe um fluxo de bytes (texto PT) e prevê o próximo (bit a bit). Previsão boa → GANHA energia;
// ruim → morre de fome. A EVOLUÇÃO seleciona quem prevê melhor — sem gradiente, sem "loss": só a FOME.
// Pergunta: a sobrevivência SOZINHA faz emergir um preditor de bytes (inteligência)? Mede bpb vs unigrama.
// Cérebros REAIS do Universe (BitLinear + CfC + DenseLayer). Input = os 8 BITS do byte (a tese bit viva). Headless.
use universe::cfc::CfCCell;
use universe::layers::BitLinear;
use universe::dense_layer::DenseLayer;
use std::io::Read;

const H: usize = 24;                       // neurônios ocultos (tiny; dá pra crescer)
const GENOME: usize = 8*H + H*H + H*8 + 8 + 1;   // l_in(8×H) + REC(H×H) + out(H×8)+bias(8) + decay do CfC
const LN2: f32 = std::f32::consts::LN_2;

struct Rng(u64);
impl Rng { fn u(&mut self)->u64 { self.0=self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z=self.0;
    z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }
    fn f(&mut self)->f32 { (self.u()>>40) as f32/(1u64<<24) as f32 }
    fn g(&mut self)->f32 { let(u,v)=(self.f().max(1e-6),self.f()); (-2.0*u.ln()).sqrt()*(std::f32::consts::TAU*v).cos() }
    fn n(&mut self,hi:usize)->usize { (self.f()*hi as f32) as usize % hi.max(1) } }

struct Brain { lin: BitLinear, rec_w: Vec<f32>, prev: Vec<f32>, cfc: CfCCell, out: DenseLayer }
impl Brain {
    fn from_genome(g:&[f32])->Brain {
        let mut o=0;
        let mut lin=BitLinear::new(8,H); lin.weights=g[o..o+8*H].to_vec(); o+=8*H;
        let rec_w=g[o..o+H*H].to_vec(); o+=H*H;                       // RECORRÊNCIA H×H (hidden→hidden)
        let mut out=DenseLayer::new(H,8); out.weights=g[o..o+H*8].to_vec(); o+=H*8; out.bias=g[o..o+8].to_vec(); o+=8;
        let mut cfc=CfCCell::new(H); cfc.decay_rate=0.3+3.0/(1.0+(-g[o]).exp()); cfc.denoise_mag=0.0; cfc.reset();
        Brain{lin, rec_w, prev: vec![0.0;H], cfc, out}
    }
    // byte atual → prevê os 8 bits do próximo. Agora com RECORRÊNCIA real: h = tanh(l_in + W_rec·h_anterior)
    fn predict(&mut self, b:u8)->[f32;8] {
        let bits:[i8;8]=core::array::from_fn(|k| if (b>>k)&1==1 {100} else {-100});
        let lo=self.lin.forward(&bits);
        let mut h=vec![0.0f32;H];
        for j in 0..H { let mut a=lo[j]; for i in 0..H { a += self.rec_w[j*H+i]*self.prev[i]; } h[j]=a.tanh(); }
        self.prev=h.clone();
        let hi:Vec<i8>=h.iter().map(|&x| (x*127.0).clamp(-127.0,127.0) as i8).collect();
        let mut rec=vec![0i8;H]; self.cfc.update(&hi,&mut rec,0.1);
        let recf:Vec<f32>=rec.iter().map(|&x| x as f32/127.0).collect();
        let o=self.out.forward(&recf);
        core::array::from_fn(|k| 1.0/(1.0+(-o[k]).exp()))
    }
}
fn rand_genome(r:&mut Rng)->Vec<f32>{ (0..GENOME).map(|_| r.g()*0.6).collect() }

// avalia um cérebro num trecho: energia sobe com previsão boa, cai com a fome. Retorna (bytes sobrevividos, bpb do trecho).
fn eval(g:&[f32], text:&[u8], start:usize, len:usize, metab:f32) -> (usize, f64) {
    let mut brain=Brain::from_genome(g);
    let (mut energy, mut bits_sum, mut n)=(26.0f32, 0.0f64, 0usize);
    for i in 0..len-1 {
        let (b,nb)=(text[start+i], text[start+i+1]);
        let p=brain.predict(b);
        let (mut acc, mut ll)=(0.0f32, 0.0f32);
        for k in 0..8 { let a=((nb>>k)&1) as f32; let pk=p[k].clamp(1e-4,1.0-1e-4);
            acc += 1.0-(a-pk).abs(); ll += -(a*pk.ln()+(1.0-a)*(1.0-pk).ln())/LN2; }
        bits_sum += ll as f64; n+=1;
        energy += (acc/8.0 - 0.52)*3.0 - metab;         // COMER = prever bem; FOME = metab
        if energy <= 0.0 { return (i+1, bits_sum/n as f64); }   // morreu de fome
        if energy > 60.0 { energy = 60.0; }
    }
    (len, bits_sum/n as f64)
}
// bpb puro (sem energia) num trecho — a "inteligência" medida
fn bpb(g:&[f32], text:&[u8], start:usize, len:usize)->f64 {
    let mut brain=Brain::from_genome(g); let (mut s,mut n)=(0.0f64,0usize);
    for i in 0..len-1 { let (b,nb)=(text[start+i],text[start+i+1]); let p=brain.predict(b);
        for k in 0..8 { let a=((nb>>k)&1) as f32; let pk=p[k].clamp(1e-4,1.0-1e-4);
            s += (-(a*pk.ln()+(1.0-a)*(1.0-pk).ln())/LN2) as f64; } n+=1; }
    s/n as f64
}

fn main() {
    // carrega um pedaço do corpus PT
    let path="/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt";
    let mut buf=vec![0u8; 6_000_000];
    { let mut f=std::fs::File::open(path).expect("corpus"); f.read_exact(&mut buf).expect("read"); }
    let text=&buf;
    let train_end=5_000_000usize;   // treina no começo, mede no fim (held-out)
    // BASELINE unigrama por bit: prever cada bit na sua marginal (sem contexto)
    let mut ones=[0u64;8]; for &b in &text[..train_end] { for k in 0..8 { ones[k]+=((b>>k)&1) as u64; } }
    let ntr=train_end as f64; let mut uni=0.0f64;
    for k in 0..8 { let p=(ones[k] as f64/ntr).clamp(1e-6,1.0-1e-6); uni += -(p*p.log2()+(1.0-p)*(1.0-p).log2()); }
    println!("PONTE survival↔inteligência — a FOME faz emergir um preditor de bytes?");
    println!("corpus PT {} bytes · cérebro tiny 8→CfC({})→8 · genoma {} pesos", text.len(), H, GENOME);
    println!("baseline UNIGRAMA (prever cada bit na marginal, SEM contexto): {:.3} bpb\n", uni);

    // GA: fitness = bytes sobrevividos (energia vem da previsão). Sem gradiente. Só sobrevivência.
    let mut r=Rng(20260708);
    let (pop, gens, elite, chunk)=(80usize, 120i32, 16usize, 1500usize);
    let metab=0.02f32;
    let base_bpb={ let mut s=0.0; for _ in 0..8 { let g=rand_genome(&mut r); s+=bpb(&g,text,train_end+r.n(500_000),4000); } s/8.0 };
    let mut genomes:Vec<Vec<f32>>=(0..pop).map(|_| rand_genome(&mut r)).collect();
    let mut champ=genomes[0].clone(); let mut champ_fit=0.0;
    for gn in 0..gens {
        let seeds:[usize;3]=[r.n(train_end-chunk), r.n(train_end-chunk), r.n(train_end-chunk)];
        let mut fits:Vec<(f64,usize)>=(0..pop).map(|i|{
            let f=seeds.iter().map(|&s| eval(&genomes[i],text,s,chunk,metab).0 as f64).sum::<f64>()/3.0; (f,i) }).collect();
        fits.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        if fits[0].0>champ_fit { champ_fit=fits[0].0; champ=genomes[fits[0].1].clone(); }
        if gn%15==0 || gn==gens-1 {
            let mbpb=bpb(&champ,text,train_end+123, 8000);
            println!("  gen {:>3}  sobrevive {:>5.0}/{} bytes  ·  bpb held-out {:.3}  (unigrama {:.3})", gn, fits[0].0, chunk, mbpb, uni);
        }
        let ord:Vec<Vec<f32>>=fits.iter().map(|&(_,i)| genomes[i].clone()).collect();
        let el:Vec<Vec<f32>>=ord[..elite].to_vec(); let mut next=el.clone();
        let rate=(0.10*0.97f32.powi(gn)*2.0).min(0.25);
        while next.len()<pop {
            let tour=|r:&mut Rng| { let mut b=usize::MAX; for _ in 0..3 { let c=r.n(elite); if c<b {b=c;} } b };
            let (p1,p2)=(&el[tour(&mut r)],&el[tour(&mut r)]);
            let mut ch=vec![0.0;GENOME]; let mut i=0; while i<GENOME { let e=(i+32).min(GENOME); let src=if r.f()<0.5 {p1}else{p2};
                ch[i..e].copy_from_slice(&src[i..e]); i=e; }
            for v in ch.iter_mut() { if r.f()<rate { *v=(*v+r.g()*0.06).clamp(-3.0,3.0); } }
            next.push(ch);
        }
        genomes=next;
    }
    let final_bpb=bpb(&champ,text,train_end+50_000, 20000);
    println!("\n=== VEREDITO ===");
    println!("  aleatório (não-evoluído): {:.3} bpb", base_bpb);
    println!("  UNIGRAMA (sem contexto):  {:.3} bpb", uni);
    println!("  CAMPEÃO (evoluído p/ SOBREVIVER): {:.3} bpb", final_bpb);
    if final_bpb < uni - 0.02 {
        println!("\n→ PONTE VALIDADA: a FOME sozinha (sem gradiente) fez o ser prever bytes MELHOR que a unigrama");
        println!("  = usa CONTEXTO = inteligência EMERGIU da sobrevivência. ({:.3} < {:.3} bpb)", final_bpb, uni);
    } else {
        println!("\n→ ainda NÃO bateu a unigrama ({:.3} vs {:.3}) — precisa de mais evolução / cérebro maior / recorrência.", final_bpb, uni);
    }
}
