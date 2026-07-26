// PONTE v2 — INSTINTO (genoma) + APRENDIZADO DE VIDA (Leonardo 2026-07-08).
// Baterias 1/3 mostraram: neuroevolução SOZINHA não bate a unigrama (o GA não otimiza muitos pesos sem gradiente;
// recorrência só aproximou o H=8 a 6.83 vs 6.03). Hipótese: o gargalo é o OTIMIZADOR, não a arquitetura.
// Aqui o ser tem DUAS fontes de saber: (1) o genoma evolui o substrato (l_in + recorrência + CfC + decay + lr),
// (2) DURANTE A VIDA ele aprende ONLINE por uma regra-delta local na camada de saída — como aprender ao viver/ler.
// Mede bpb PREQUENCIAL (prevê o byte t só com o passado, depois aprende) vs unigrama fixa E unigrama online.
// Cérebros REAIS do Universe. Input = 8 bits do byte. Headless.
use universe::cfc::CfCCell;
use universe::layers::BitLinear;
use universe::dense_layer::DenseLayer;
use std::io::Read;

const H: usize = 8;
const GENOME: usize = 8*H + H*H + H*8 + 8 + 2;   // l_in + REC(H×H) + out(H×8)+bias(8) + decay + lr
const LN2: f32 = std::f32::consts::LN_2;

struct Rng(u64);
impl Rng { fn u(&mut self)->u64 { self.0=self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z=self.0;
    z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }
    fn f(&mut self)->f32 { (self.u()>>40) as f32/(1u64<<24) as f32 }
    fn g(&mut self)->f32 { let(u,v)=(self.f().max(1e-6),self.f()); (-2.0*u.ln()).sqrt()*(std::f32::consts::TAU*v).cos() }
    fn n(&mut self,hi:usize)->usize { (self.f()*hi as f32) as usize % hi.max(1) } }

struct Brain { lin: BitLinear, rec_w: Vec<f32>, prev: Vec<f32>, cfc: CfCCell, out: DenseLayer, lr: f32 }
impl Brain {
    fn from_genome(g:&[f32])->Brain {
        let mut o=0;
        let mut lin=BitLinear::new(8,H); lin.weights=g[o..o+8*H].to_vec(); o+=8*H;
        let rec_w=g[o..o+H*H].to_vec(); o+=H*H;
        let mut out=DenseLayer::new(H,8); out.weights=g[o..o+H*8].to_vec(); o+=H*8; out.bias=g[o..o+8].to_vec(); o+=8;
        let mut cfc=CfCCell::new(H); cfc.decay_rate=0.3+3.0/(1.0+(-g[o]).exp()); cfc.denoise_mag=0.0; cfc.reset(); o+=1;
        let lr=0.06/(1.0+(-g[o]).exp());                                  // taxa de aprendizado de vida (0..0.06)
        Brain{lin, rec_w, prev: vec![0.0;H], cfc, out, lr}
    }
    // devolve (probs dos 8 bits, feature recf usada pela saída) — recf alimenta a regra-delta
    fn step(&mut self, b:u8)->([f32;8], Vec<f32>) {
        let bits:[i8;8]=core::array::from_fn(|k| if (b>>k)&1==1 {100} else {-100});
        let lo=self.lin.forward(&bits);
        let mut h=vec![0.0f32;H];
        for j in 0..H { let mut a=lo[j]; for i in 0..H { a += self.rec_w[j*H+i]*self.prev[i]; } h[j]=a.tanh(); }
        self.prev=h.clone();
        let hi:Vec<i8>=h.iter().map(|&x| (x*127.0).clamp(-127.0,127.0) as i8).collect();
        let mut rec=vec![0i8;H]; self.cfc.update(&hi,&mut rec,0.1);
        let recf:Vec<f32>=rec.iter().map(|&x| x as f32/127.0).collect();
        let o=self.out.forward(&recf);
        (core::array::from_fn(|k| 1.0/(1.0+(-o[k]).exp())), recf)
    }
    // APRENDER de vida: regra-delta local na saída (regressão logística online sobre a feature recorrente)
    fn learn(&mut self, probs:&[f32;8], recf:&[f32], next:u8) {
        for k in 0..8 { let target=((next>>k)&1) as f32; let err=target-probs[k];
            let d=self.lr*err;
            for j in 0..H { self.out.weights[k*H+j] += d*recf[j]; }
            self.out.bias[k] += d;
        }
    }
}
fn rand_genome(r:&mut Rng)->Vec<f32>{ (0..GENOME).map(|_| r.g()*0.6).collect() }

// energia sobe com previsão boa (após aprender), cai com fome. O ser aprende ONLINE enquanto vive.
fn eval(g:&[f32], text:&[u8], start:usize, len:usize, metab:f32) -> f64 {
    let mut brain=Brain::from_genome(g);
    let mut energy=26.0f32;
    for i in 0..len-1 {
        let (b,nb)=(text[start+i], text[start+i+1]);
        let (p,recf)=brain.step(b);
        let mut acc=0.0f32;
        for k in 0..8 { let a=((nb>>k)&1) as f32; let pk=p[k].clamp(1e-4,1.0-1e-4); acc += 1.0-(a-pk).abs(); }
        brain.learn(&p,&recf,nb);                         // vive → aprende
        energy += (acc/8.0 - 0.52)*3.0 - metab;
        if energy <= 0.0 { return (i+1) as f64; }
        if energy > 60.0 { energy = 60.0; }
    }
    len as f64
}
// bpb PREQUENCIAL: prevê byte t só com o passado, mede -log2 P, DEPOIS aprende com o byte real (código online honesto)
fn bpb_preq(g:&[f32], text:&[u8], start:usize, len:usize)->f64 {
    let mut brain=Brain::from_genome(g); let (mut s,mut n)=(0.0f64,0usize);
    for i in 0..len-1 { let (b,nb)=(text[start+i],text[start+i+1]); let (p,recf)=brain.step(b);
        for k in 0..8 { let a=((nb>>k)&1) as f32; let pk=p[k].clamp(1e-4,1.0-1e-4);
            s += (-(a*pk.ln()+(1.0-a)*(1.0-pk).ln())/LN2) as f64; } n+=1;
        brain.learn(&p,&recf,nb);
    }
    s/n as f64
}
// unigrama ONLINE (prequencial): frequência adaptativa por bit — baseline justo p/ aprendizado de vida
fn unigram_online(text:&[u8], start:usize, len:usize)->f64 {
    let (mut ones,mut tot)=([1.0f64;8],2.0f64); let (mut s,mut n)=(0.0f64,0usize);
    for i in 0..len { let b=text[start+i];
        for k in 0..8 { let p=(ones[k]/tot).clamp(1e-6,1.0-1e-6); let a=((b>>k)&1) as f64;
            s += -(a*p.log2()+(1.0-a)*(1.0-p).log2()); ones[k]+=a; }
        tot+=1.0; n+=1;
    }
    s/n as f64
}
// bigrama ONLINE (TETO order-1): prevê bits do byte t condicionado no byte t-1 (tabela adaptativa 256×8).
// É o quanto dá pra ganhar usando 1 byte de contexto — o alvo real se o ser "usa" a memória do byte anterior.
fn bigram_online(text:&[u8], start:usize, len:usize)->f64 {
    let mut ones=vec![1.0f64;256*8]; let mut tot=vec![2.0f64;256]; let (mut s,mut n)=(0.0f64,0usize);
    for i in 1..len { let prev=text[start+i-1] as usize; let b=text[start+i];
        for k in 0..8 { let p=(ones[prev*8+k]/tot[prev]).clamp(1e-6,1.0-1e-6); let a=((b>>k)&1) as f64;
            s += -(a*p.log2()+(1.0-a)*(1.0-p).log2()); ones[prev*8+k]+=a; }
        tot[prev]+=1.0; n+=1;
    }
    s/n as f64
}

fn main() {
    let path="/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt";
    let mut buf=vec![0u8; 6_000_000];
    { let mut f=std::fs::File::open(path).expect("corpus"); f.read_exact(&mut buf).expect("read"); }
    let text=&buf;
    let train_end=5_000_000usize;
    let mut ones=[0u64;8]; for &b in &text[..train_end] { for k in 0..8 { ones[k]+=((b>>k)&1) as u64; } }
    let ntr=train_end as f64; let mut uni=0.0f64;
    for k in 0..8 { let p=(ones[k] as f64/ntr).clamp(1e-6,1.0-1e-6); uni += -(p*p.log2()+(1.0-p)*(1.0-p).log2()); }
    let uni_online=unigram_online(text,train_end+50_000,20000);
    let bi_online=bigram_online(text,train_end+50_000,20000);
    let args:Vec<String>=std::env::args().collect();
    let seed=args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20260708u64);
    let gens=args.get(2).and_then(|s| s.parse().ok()).unwrap_or(120i32);
    println!("PONTE v2 — INSTINTO(genoma) + VIDA(delta-rule online). H={} genoma={} seed={} gens={}", H, GENOME, seed, gens);
    println!("PISO unigrama FIXA {:.3} · unigrama ONLINE {:.3}  |  TETO order-1 bigrama ONLINE {:.3} bpb\n", uni, uni_online, bi_online);

    let mut r=Rng(seed);
    let (pop, elite, chunk)=(80usize, 16usize, 1500usize);
    let metab=0.02f32;
    let base_bpb={ let mut s=0.0; for _ in 0..8 { let g=rand_genome(&mut r); s+=bpb_preq(&g,text,train_end+r.n(500_000),4000); } s/8.0 };
    let mut genomes:Vec<Vec<f32>>=(0..pop).map(|_| rand_genome(&mut r)).collect();
    let mut champ=genomes[0].clone(); let mut champ_fit=0.0;
    for gn in 0..gens {
        let seeds:[usize;3]=[r.n(train_end-chunk), r.n(train_end-chunk), r.n(train_end-chunk)];
        let mut fits:Vec<(f64,usize)>=(0..pop).map(|i|{
            let f=seeds.iter().map(|&s| eval(&genomes[i],text,s,chunk,metab)).sum::<f64>()/3.0; (f,i) }).collect();
        fits.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        if fits[0].0>champ_fit { champ_fit=fits[0].0; champ=genomes[fits[0].1].clone(); }
        if gn%15==0 || gn==gens-1 {
            let mbpb=bpb_preq(&champ,text,train_end+123, 8000);
            println!("  gen {:>3}  sobrevive {:>5.0}/{}  ·  bpb preq {:.3}  (uni fix {:.3} / on {:.3})", gn, fits[0].0, chunk, mbpb, uni, uni_online);
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
    // final = MÉDIA sobre 4 janelas held-out distintas (mata falso-positivo de janela sortuda)
    let wins=[50_000usize, 300_000, 600_000, 900_000];
    let finals:Vec<f64>=wins.iter().map(|&w| bpb_preq(&champ,text,train_end+w, 20000)).collect();
    let final_bpb=finals.iter().sum::<f64>()/finals.len() as f64;
    println!("\n=== VEREDITO (seed {}) ===", seed);
    println!("  aleatório (não-evoluído, mas APRENDE de vida): {:.3} bpb", base_bpb);
    println!("  PISO unigrama fixa {:.3} · online {:.3}  |  TETO bigrama online {:.3}", uni, uni_online, bi_online);
    println!("  CAMPEÃO (instinto+vida) por janela: {:?}", finals.iter().map(|x|(x*1000.0).round()/1000.0).collect::<Vec<_>>());
    println!("  CAMPEÃO média 4 janelas: {:.3} bpb", final_bpb);
    let best=uni.min(uni_online);
    if final_bpb < best - 0.005 {
        let frac=(best-final_bpb)/(best-bi_online).max(1e-9);
        println!("\n→ BATEU o piso unigrama ({:.3} < {:.3}); usa {:.0}% do contexto disponível (teto bigrama {:.3})", final_bpb, best, frac*100.0, bi_online);
    } else {
        println!("\n→ NÃO bateu o piso ({:.3} vs {:.3}) — só empata a marginal, não usa contexto.", final_bpb, best);
    }
}
