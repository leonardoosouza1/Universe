// BATERIA: INTELIGÊNCIA A PARTIR DE SOBREVIVÊNCIA — CÉREBRO = ÓRGÃOS (Leonardo 2026-07-09).
// Teoria: o cérebro do ser é feito de ÓRGÃOS combináveis. Instinto = BitLinear; Memória = CfC (0, 1, 2... órgãos
// empilhados); Motor = DenseLayer. (Próximo órgão: conhecimento importado = ByteBrain-como-sensor — ver HANDOFF.)
// Varre TAMANHO do cérebro (H) × Nº de ÓRGÃOS de MEMÓRIA (CfC) × NICHO (gap de cegueira). ≥50 validações.
// Objetivo: até onde dá pra ENXUGAR a inteligência? Qual o MENOR cérebro que sobrevive bem?
// Mundo forrageio headless (come planta=+energia, predador caça, fome mata). Órgãos REAIS do Universe.
use universe::cfc::CfCCell;
use universe::layers::BitLinear;
use universe::dense_layer::DenseLayer;

const NIN: usize = 7;
const TAU: f32 = std::f32::consts::TAU;

#[derive(Clone, Copy)]
struct Cfg { h: usize, mem: usize }   // h = tamanho; mem = nº de órgãos CfC empilhados
fn gsize(c: Cfg) -> usize { NIN*c.h + c.h*2 + 2 + c.mem }
fn params(c: Cfg) -> usize { NIN*c.h + c.h*2 + 2 + c.mem }   // pesos totais (proxy de "custo" do cérebro)

struct Rng(u64);
impl Rng { fn u(&mut self)->u64 { self.0=self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z=self.0;
    z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }
    fn f(&mut self)->f32 { (self.u()>>40) as f32/(1u64<<24) as f32 }
    fn g(&mut self)->f32 { let(u,v)=(self.f().max(1e-6),self.f()); (-2.0*u.ln()).sqrt()*(TAU*v).cos() }
    fn n(&mut self,hi:usize)->usize { (self.f()*hi as f32) as usize % hi.max(1) } }

struct Brain { lin: BitLinear, cfcs: Vec<CfCCell>, out: DenseLayer, h: usize }
impl Brain {
    fn from_genome(g:&[f32], c:Cfg)->Brain {
        let h=c.h; let mut o=0;
        let mut lin=BitLinear::new(NIN,h); lin.weights=g[o..o+NIN*h].to_vec(); o+=NIN*h;
        let mut out=DenseLayer::new(h,2); out.weights=g[o..o+h*2].to_vec(); o+=h*2; out.bias=g[o..o+2].to_vec(); o+=2;
        let mut cfcs=Vec::new();
        for _ in 0..c.mem { let mut cf=CfCCell::new(h); cf.decay_rate=0.3+3.0/(1.0+(-g[o]).exp()); cf.denoise_mag=0.0; cf.reset(); o+=1; cfcs.push(cf); }
        Brain{lin,cfcs,out,h}
    }
    fn act(&mut self,s:&[i8;NIN])->[f32;2]{
        let mut h:Vec<f32>=self.lin.forward(s).iter().map(|x|x.tanh()).collect();
        for cf in self.cfcs.iter_mut(){
            let hi:Vec<i8>=h.iter().map(|&x|(x*127.0).clamp(-127.0,127.0) as i8).collect();
            let mut rec=vec![0i8;self.h]; cf.update(&hi,&mut rec,0.2);
            h=rec.iter().map(|&x|x as f32/127.0).collect();
        }
        let o=self.out.forward(&h); [o[0],o[1]]
    }
}
fn rand_genome(r:&mut Rng, c:Cfg)->Vec<f32>{ (0..gsize(c)).map(|_| r.g()*0.6).collect() }

fn eval(g:&[f32], c:Cfg, gap:i32, seed:u64)->f64 {
    let mut rng=Rng(seed);
    let size=100.0f32; let nfood=12usize;
    let mut foods:Vec<(f32,f32)>=(0..nfood).map(|_| (rng.f()*size, rng.f()*size)).collect();
    let (mut cx,mut cy)=(size*0.5,size*0.5);
    let mut heading=rng.f()*TAU; let mut energy=100.0f32;
    let (mut px,mut py)=(rng.f()*size, rng.f()*size);
    let mut brain=Brain::from_genome(g,c);
    let mut eaten=0u32; let max_ticks=1500i32;
    for tick in 0..max_ticks {
        let (mut fd,mut fdx,mut fdy)=(1e9f32,0.0,0.0);
        for &(fx,fy) in &foods { let d=((fx-cx).powi(2)+(fy-cy).powi(2)).sqrt().max(1e-3); if d<fd {fd=d; fdx=(fx-cx)/d; fdy=(fy-cy)/d;} }
        let blind=gap>1 && (tick%gap!=0);
        let (sfdx,sfdy,sfd)= if blind {(0.0,0.0,0.0)} else {(fdx,fdy,1.0-(fd/size).min(1.0))};
        let pd=((px-cx).powi(2)+(py-cy).powi(2)).sqrt().max(1e-3);
        let (pdx,pdy)=((px-cx)/pd,(py-cy)/pd); let pnear=1.0-(pd/size).min(1.0);
        let s:[i8;NIN]=[(sfdx*100.0) as i8,(sfdy*100.0) as i8,(sfd*100.0) as i8,(pdx*100.0) as i8,(pdy*100.0) as i8,(pnear*100.0) as i8,((energy/100.0*2.0-1.0)*100.0) as i8];
        let a=brain.act(&s);
        heading += a[0].clamp(-1.0,1.0)*0.6;
        let speed=(a[1].clamp(-1.0,1.0)*0.5+0.5)*2.5;
        cx=(cx+heading.cos()*speed).clamp(0.0,size); cy=(cy+heading.sin()*speed).clamp(0.0,size);
        for fpt in foods.iter_mut(){ let d=((fpt.0-cx).powi(2)+(fpt.1-cy).powi(2)).sqrt(); if d<3.0 {energy=(energy+35.0).min(100.0); eaten+=1; *fpt=(rng.f()*size,rng.f()*size);} }
        let ph=(cy-py).atan2(cx-px); px+=ph.cos()*2.0; py+=ph.sin()*2.0;
        if pd<3.0 { return tick as f64+eaten as f64*20.0; }
        energy-=0.6; if energy<=0.0 { return tick as f64+eaten as f64*20.0; }
    }
    max_ticks as f64+eaten as f64*20.0
}
fn fit_avg(g:&[f32], c:Cfg, gap:i32, seeds:&[u64])->f64 { seeds.iter().map(|&s| eval(g,c,gap,s)).sum::<f64>()/seeds.len() as f64 }

fn evolve(c:Cfg, gap:i32, master:u64)->Vec<f32> {
    let gs=gsize(c); let mut r=Rng(master);
    let (pop,gens,elite)=(50usize,45i32,10usize);
    let mut genomes:Vec<Vec<f32>>=(0..pop).map(|_| rand_genome(&mut r,c)).collect();
    let mut champ=genomes[0].clone(); let mut cf=-1.0;
    for gn in 0..gens {
        let seeds:[u64;3]=[r.u(),r.u(),r.u()];
        let mut fits:Vec<(f64,usize)>=(0..pop).map(|i| (fit_avg(&genomes[i],c,gap,&seeds),i)).collect();
        fits.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
        if fits[0].0>cf { cf=fits[0].0; champ=genomes[fits[0].1].clone(); }
        let ord:Vec<Vec<f32>>=fits.iter().map(|&(_,i)| genomes[i].clone()).collect();
        let el:Vec<Vec<f32>>=ord[..elite].to_vec(); let mut next=el.clone();
        let rate=(0.10*0.97f32.powi(gn)*2.0).min(0.25);
        while next.len()<pop {
            let tour=|r:&mut Rng| { let mut b=usize::MAX; for _ in 0..3 { let cc=r.n(elite); if cc<b {b=cc;} } b };
            let (p1,p2)=(&el[tour(&mut r)],&el[tour(&mut r)]);
            let mut ch=vec![0.0;gs]; let mut i=0; while i<gs { let e=(i+16).min(gs); let src=if r.f()<0.5 {p1}else{p2}; ch[i..e].copy_from_slice(&src[i..e]); i=e; }
            for v in ch.iter_mut() { if r.f()<rate { *v=(*v+r.g()*0.08).clamp(-3.0,3.0); } }
            next.push(ch);
        }
        genomes=next;
    }
    champ
}

fn main() {
    let test:Vec<u64>=(0..12).map(|k| 777_000+k*13).collect();
    let hs=[4usize,6,12,24,48,96];
    let mems=[0usize,1,2];
    let gaps=[1i32,4,8];
    println!("=== INTELIGÊNCIA DA SOBREVIVÊNCIA — CÉREBRO=ÓRGÃOS (BitLinear+CfC×N+DenseLayer) ===");
    println!("Varre H(tamanho) × órgãos-CfC × gap(nicho). Cada célula: melhor de 2 GAs, held-out {} ambs. aleatório~20.", test.len());
    println!("Total de validações = {} configs.\n", hs.len()*mems.len()*gaps.len());
    println!("{:>3} {:>4} {:>5} {:>7} | {:>8}", "H", "mem", "par", "gap", "fitness");
    let mut rows:Vec<(usize,usize,usize,i32,f64)>=vec![];
    for &h in &hs { for &m in &mems { for &gap in &gaps {
        let c=Cfg{h,mem:m};
        let best=(0..2).map(|k| fit_avg(&evolve(c,gap,20260709+ (h*131+m*17) as u64 + (gap as u64)*7 + k*1009), c,gap,&test)).fold(f64::MIN,f64::max);
        rows.push((h,m,params(c),gap,best));
        println!("{:>3} {:>4} {:>5} {:>7} | {:>8.0}", h, m, params(c), gap, best);
    }}}
    // análise: MENOR cérebro (menos params) que chega a >=80% do melhor fitness geral (por gap)
    println!("\n=== ATÉ ONDE DÁ PRA ENXUGAR? (menor cérebro com >=80% do melhor, por nicho) ===");
    for &gap in &gaps {
        let g:Vec<_>=rows.iter().filter(|r| r.3==gap).collect();
        let bestf=g.iter().map(|r| r.4).fold(0.0f64,f64::max);
        let thr=bestf*0.8;
        let mut cheapest:Option<&(usize,usize,usize,i32,f64)>=None;
        for r in &g { if r.4>=thr { match cheapest { Some(c) if c.2<=r.2 => {}, _=>cheapest=Some(r) } } }
        if let Some(r)=cheapest {
            println!("  gap={:<2}: melhor {:.0} · MÍNIMO viável = H{} mem{} ({} params) com fitness {:.0} (>=80%)", gap, bestf, r.0, r.1, r.2, r.4);
        }
    }
    println!("\n=== FIM ===");
}
