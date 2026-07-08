// IARA-VIDA 3D — ecossistema de sobrevivência na ENGINE do Universe (Bevy 3D, como o mundo de corrida).
// Terreno grande · comida (=conhecimento) · arbustos pra esconder · água · HERBÍVOROS (presas) e PREDADORES.
// Clique num ser → a câmera segue e o painel mostra o cérebro/energia/conhecimento. Reproduz = EVOLUÇÃO ao vivo.
// Reusa os cérebros reais (CfCCell + BitLinear + DenseLayer). Rodar: HSA_OVERRIDE_GFX_VERSION=10.3.0 ./target/release/vida3d
use bevy::prelude::*;
use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::tonemapping::Tonemapping;
use universe::cfc::CfCCell;
use universe::layers::BitLinear;
use universe::dense_layer::DenseLayer;

const WORLD: f32 = 260.0;         // lado do mundo (unidades) — mapa grande
const HALF: f32 = WORLD * 0.5;
const NIN: usize = 14;            // sensores: comida(2) água(2) ameaça(3) parceiro(2) fome/energia(2) escondido bias(2)
const NH: usize = 12;
const NOUT: usize = 3;            // virar, acelerar, [comer/atacar implícito por contato]
const DT: f32 = 0.12;

// ---------------- RNG ----------------
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15); let mut z=self.0;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB); z^(z>>31) }
    fn f(&mut self)->f32 { (self.u()>>40) as f32/(1u64<<24) as f32 }
    fn r(&mut self,a:f32,b:f32)->f32 { a+(b-a)*self.f() }
    fn g(&mut self)->f32 { let(u,v)=(self.f().max(1e-6),self.f()); (-2.0*u.ln()).sqrt()*(std::f32::consts::TAU*v).cos() }
}
fn q8(x:f32)->i8 { (x.clamp(-1.0,1.0)*127.0) as i8 }

// ---------------- CÉREBRO (camadas reais do Universe) ----------------
const GENOME: usize = NIN*NH + NH*NOUT + NOUT + 1;
struct Brain { l1: BitLinear, cfc: CfCCell, out: DenseLayer }
impl Brain {
    fn from_genome(g:&[f32])->Brain {
        let mut o=0;
        let mut l1=BitLinear::new(NIN,NH); l1.weights=g[o..o+NIN*NH].to_vec(); o+=NIN*NH;
        let mut out=DenseLayer::new(NH,NOUT); out.weights=g[o..o+NH*NOUT].to_vec(); o+=NH*NOUT; out.bias=g[o..o+NOUT].to_vec(); o+=NOUT;
        let mut cfc=CfCCell::new(NH); cfc.decay_rate=0.4+3.0/(1.0+(-g[o]).exp()); cfc.reset();
        Brain{l1,cfc,out}
    }
    fn think(&mut self, s:&[f32;NIN])->[f32;NOUT] {
        let si:Vec<i8>=s.iter().map(|&x| q8(x)).collect();
        let h1:Vec<f32>=self.l1.forward(&si).iter().map(|x| x.tanh()).collect();
        let hi:Vec<i8>=h1.iter().map(|&x| q8(x)).collect();
        let mut rec=vec![0i8;NH]; self.cfc.update(&hi,&mut rec,DT);
        let recf:Vec<f32>=rec.iter().map(|&x| x as f32/127.0).collect();
        let o=self.out.forward(&recf);
        [o[0].tanh(), 1.0/(1.0+(-o[1]).exp()), o[2].tanh()]
    }
    fn genome(&self)->Vec<f32> { let mut g=Vec::with_capacity(GENOME);
        g.extend_from_slice(&self.l1.weights); g.extend_from_slice(&self.out.weights); g.extend_from_slice(&self.out.bias);
        g.push(0.0); g }
}
fn rand_genome(r:&mut Rng)->Vec<f32>{ (0..GENOME).map(|_| r.g()*0.6).collect() }

// ---------------- COMPONENTES ----------------
#[derive(Clone,Copy,PartialEq)] enum Kind { Herb, Pred }
#[derive(Component)]
struct Creature { kind:Kind, brain:Brain, head:f32, energy:f32, age:u32, gener:u32,
    clan:u8, hidden:bool, knowledge:f32, sensed:[f32;NIN] }
#[derive(Component)] struct Food { dom:u8 }
#[derive(Component)] struct Bush;
#[derive(Component)] struct Water;
#[derive(Component)] struct FollowCam { yaw:f32, pitch:f32, dist:f32, target:Vec3 }
#[derive(Component)] struct Hud;

#[derive(Resource)]
struct Sim { rng:Rng, tick:u32, generation:u32, sel:Option<Entity>, paused:bool, speed:f32,
    born_h:u32, born_p:u32, died:u32,
    m_herb:Handle<Mesh>, m_pred:Handle<Mesh>, m_food:Handle<Mesh>,
    mat_food:[Handle<StandardMaterial>;6], mat_clan:Vec<Handle<StandardMaterial>>, mat_pred:Handle<StandardMaterial> }

fn dom_at(x:f32,z:f32)->u8 { use std::f32::consts::TAU;
    let (mut best,mut bd)=(f32::MIN,0u8);
    for k in 0..6 { let f=(1.0+k as f32*1.2)/WORLD; let ph=k as f32*2.1;
        let v=(TAU*f*x+ph).sin()+(TAU*f*z+ph*1.7).cos(); if v>best {best=v; bd=k as u8;} } bd }

fn main() {
    println!("IARA-VIDA 3D — clique num ser (câmera segue) · WASD move câmera · roda=zoom · Espaço=pausa · ↑↓=veloc");
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin{ primary_window:Some(Window{
            title:"IARA-VIDA · ecossistema vivo (Universe)".into(), resolution:(1400.0,820.0).into(), ..default() }), ..default() }))
        .insert_resource(ClearColor(Color::srgb(0.05,0.07,0.10)))
        .insert_resource(AmbientLight{ brightness:260.0, color:Color::srgb(0.7,0.8,0.95), ..default() })
        .add_systems(Startup, setup)
        .add_systems(Update, (sim_step, camera_ctrl, click_select, hud))
        .run();
}

fn setup(mut c:Commands, mut meshes:ResMut<Assets<Mesh>>, mut mats:ResMut<Assets<StandardMaterial>>) {
    let mut rng = Rng(20260708);
    // câmera 3D estilo engine (hdr + bloom) + luz
    c.spawn((Camera3d::default(), Camera{hdr:true,..default()}, Tonemapping::AcesFitted, Bloom::NATURAL,
        Transform::from_xyz(0.0, 140.0, 200.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy::pbr::DistanceFog{ color:Color::srgb(0.5,0.62,0.78), falloff:bevy::pbr::FogFalloff::Linear{start:250.0,end:900.0}, ..default() },
        FollowCam{ yaw:0.0, pitch:0.9, dist:230.0, target:Vec3::ZERO }));
    c.spawn((DirectionalLight{ illuminance:22000.0, shadows_enabled:true, color:Color::srgb(1.0,0.96,0.9), ..default() },
        Transform::from_xyz(180.0,320.0,120.0).looking_at(Vec3::ZERO, Vec3::Y)));
    // chão: campo de domínios por frequência (grade de placas coloridas)
    let tile = meshes.add(Cuboid::new(WORLD/16.0, 0.4, WORLD/16.0));
    for gz in 0..16 { for gx in 0..16 {
        let x=-HALF+(gx as f32+0.5)*WORLD/16.0; let z=-HALF+(gz as f32+0.5)*WORLD/16.0;
        let d=dom_at(x,z); let col=dom_color(d, 0.20);
        c.spawn((Mesh3d(tile.clone()), MeshMaterial3d(mats.add(StandardMaterial{ base_color:col, perceptual_roughness:0.95, ..default() })),
            Transform::from_xyz(x,-0.2,z))); } }
    // meshes/materiais reutilizados
    let m_herb=meshes.add(Capsule3d::new(1.6,2.6));
    let m_pred=meshes.add(Capsule3d::new(2.4,4.2));
    let m_food=meshes.add(Sphere::new(1.3));
    let m_bush=meshes.add(Sphere::new(4.5));
    let m_water=meshes.add(Cylinder::new(9.0,0.5));
    let mat_food:[Handle<StandardMaterial>;6]=core::array::from_fn(|d| mats.add(StandardMaterial{ base_color:dom_color(d as u8,0.9), emissive:dom_color(d as u8,0.9).to_linear()*0.4, ..default() }));
    let mat_clan:Vec<Handle<StandardMaterial>>=(0..8).map(|k| mats.add(StandardMaterial{ base_color:clan_color(k), perceptual_roughness:0.5, ..default() })).collect();
    let mat_pred=mats.add(StandardMaterial{ base_color:Color::srgb(0.9,0.15,0.2), emissive:LinearRgba::new(0.5,0.05,0.08,1.0), perceptual_roughness:0.4, ..default() });
    let mat_bush=mats.add(StandardMaterial{ base_color:Color::srgb(0.10,0.28,0.12), perceptual_roughness:1.0, ..default() });
    let mat_water=mats.add(StandardMaterial{ base_color:Color::srgb(0.15,0.4,0.75), emissive:LinearRgba::new(0.02,0.1,0.25,1.0), perceptual_roughness:0.2, metallic:0.1, ..default() });
    // arbustos (esconderijo) + água
    for _ in 0..26 { let (x,z)=(rng.r(-HALF,HALF),rng.r(-HALF,HALF));
        c.spawn((Mesh3d(m_bush.clone()), MeshMaterial3d(mat_bush.clone()), Transform::from_xyz(x,2.0,z), Bush)); }
    for _ in 0..7 { let (x,z)=(rng.r(-HALF,HALF),rng.r(-HALF,HALF));
        c.spawn((Mesh3d(m_water.clone()), MeshMaterial3d(mat_water.clone()), Transform::from_xyz(x,0.1,z), Water)); }
    // comida (=conhecimento), colorida pelo domínio
    for _ in 0..70 { let (x,z)=(rng.r(-HALF,HALF),rng.r(-HALF,HALF)); let d=dom_at(x,z);
        c.spawn((Mesh3d(m_food.clone()), MeshMaterial3d(mat_food[d as usize].clone()), Transform::from_xyz(x,1.3,z), Food{dom:d})); }
    // criaturas: herbívoros (presas) + predadores
    for i in 0..48 {
        let herb=i<40; let (x,z)=(rng.r(-HALF,HALF),rng.r(-HALF,HALF));
        let g=rand_genome(&mut rng); let clan=(rng.f()*8.0) as u8;
        let mesh = if herb {m_herb.clone()} else {m_pred.clone()};
        let mat = if herb {mat_clan[clan as usize % 8].clone()} else {mat_pred.clone()};
        c.spawn((Mesh3d(mesh), MeshMaterial3d(mat), Transform::from_xyz(x, if herb {2.2} else {3.0}, z),
            Creature{ kind:if herb {Kind::Herb} else {Kind::Pred}, brain:Brain::from_genome(&g),
                head:rng.r(0.0,6.28), energy:70.0, age:0, gener:0, clan, hidden:false, knowledge:0.0, sensed:[0.0;NIN] }));
    }
    // HUD
    c.spawn((Text::new(""), TextFont{font_size:15.0,..default()}, TextColor(Color::srgb(0.85,0.92,1.0)),
        Node{ position_type:PositionType::Absolute, left:Val::Px(14.0), top:Val::Px(10.0), ..default() }, Hud));
    c.spawn((Text::new("clique num ser · WASD/setas move câmera · Q/E gira · roda=zoom · Espaço=pausa"),
        TextFont{font_size:12.0,..default()}, TextColor(Color::srgb(0.5,0.6,0.75)),
        Node{ position_type:PositionType::Absolute, left:Val::Px(14.0), bottom:Val::Px(10.0), ..default() }));
    c.insert_resource(Sim{ rng, tick:0, generation:0, sel:None, paused:false, speed:1.0,
        born_h:0, born_p:0, died:0, m_herb, m_pred, m_food, mat_food, mat_clan, mat_pred });
}

fn dom_color(d:u8,l:f32)->Color { Color::hsl(d as f32/6.0*340.0, 0.6, l) }
fn clan_color(k:u8)->Color { Color::hsl(k as f32/8.0*360.0, 0.65, 0.55) }

// ---------------- ECOSSISTEMA (1 passo por frame; speed escala o DT) ----------------
fn sim_step(
    mut c: Commands, mut sim: ResMut<Sim>,
    mut creatures: Query<(Entity, &mut Transform, &mut Creature), Without<Food>>,
    foods: Query<(Entity, &Transform), (With<Food>, Without<Creature>)>,
    bushes: Query<&Transform, (With<Bush>, Without<Creature>, Without<Food>)>,
    waters: Query<&Transform, (With<Water>, Without<Creature>, Without<Food>)>,
) {
    if sim.paused { return; }
    let dt = DT * sim.speed;
    let snap: Vec<(Entity,f32,f32,Kind,u8,bool)> = creatures.iter().map(|(e,t,cr)| (e,t.translation.x,t.translation.z,cr.kind,cr.clan,cr.hidden)).collect();
    let fsnap: Vec<(Entity,f32,f32)> = foods.iter().map(|(e,t)| (e,t.translation.x,t.translation.z)).collect();
    let bsnap: Vec<(f32,f32)> = bushes.iter().map(|t| (t.translation.x,t.translation.z)).collect();
    let wsnap: Vec<(f32,f32)> = waters.iter().map(|t| (t.translation.x,t.translation.z)).collect();
    let ncre = snap.len();
    let mut eaten: Vec<Entity> = Vec::new();
    let mut deaths: std::collections::HashSet<Entity> = std::collections::HashSet::new();
    let mut births: Vec<(Kind,f32,f32,f32,Vec<f32>,u8,u32)> = Vec::new();
    for (e, mut tr, mut cr) in creatures.iter_mut() {
        let (x,z) = (tr.translation.x, tr.translation.z);
        cr.hidden = cr.kind==Kind::Herb && bsnap.iter().any(|&(bx,bz)| (x-bx).powi(2)+(z-bz).powi(2) < 25.0);
        let near = |list:&[(f32,f32)]| { let (mut bd,mut dx,mut dz)=(1e9f32,0.0,0.0); for &(gx,gz) in list { let d=(x-gx).powi(2)+(z-gz).powi(2); if d<bd {bd=d;dx=gx-x;dz=gz-z;} } (bd,dx,dz) };
        let (fd,fx,fz)={ let (mut bd,mut dx,mut dz)=(1e9f32,0.0,0.0); for &(_,gx,gz) in &fsnap { let d=(x-gx).powi(2)+(z-gz).powi(2); if d<bd {bd=d;dx=gx-x;dz=gz-z;} } (bd,dx,dz) };
        let (_wd,wx,wz)=near(&wsnap);
        let (mut td,mut tx,mut tz)=(1e9f32,0.0,0.0); let (mut pd,mut px,mut pz)=(1e9f32,0.0,0.0);
        for &(oe,ox,oz,okind,oclan,ohidden) in &snap { if oe==e {continue;} let d=(x-ox).powi(2)+(z-oz).powi(2);
            let threat = match cr.kind { Kind::Herb => okind==Kind::Pred, Kind::Pred => okind==Kind::Herb && !ohidden };
            if threat && d<td {td=d;tx=ox-x;tz=oz-z;}
            if okind==cr.kind && oclan==cr.clan && d<pd {pd=d;px=ox-x;pz=oz-z;} }
        let nz2 = |dx:f32,dz:f32,d:f32| { let n=d.sqrt().max(1e-3); (dx/n,dz/n) };
        let (nfx,nfz)=nz2(fx,fz,fd); let (nwx,nwz)=nz2(wx,wz,_wd); let (ntx,ntz)=nz2(tx,tz,td); let (npx,npz)=nz2(px,pz,pd);
        let mut s=[0.0f32;NIN];
        s[0]=nfx; s[1]=nfz; s[2]=nwx; s[3]=nwz; s[4]=ntx; s[5]=ntz; s[6]=(-td/1600.0).exp();
        s[7]=npx; s[8]=npz; s[9]=1.0-cr.energy/100.0; s[10]=cr.knowledge; s[11]=if cr.hidden {1.0} else {0.0};
        s[12]=if cr.kind==Kind::Pred {1.0} else {-1.0}; s[13]=1.0;
        cr.sensed=s;
        let o=cr.brain.think(&s); let (turn,thrust)=(o[0],o[1]);
        cr.head += turn*2.2*dt;
        let sp = if cr.kind==Kind::Pred {30.0} else {24.0};
        let nx=(x+thrust*sp*dt*cr.head.sin()).clamp(-HALF,HALF);
        let nzz=(z+thrust*sp*dt*cr.head.cos()).clamp(-HALF,HALF);
        tr.translation.x=nx; tr.translation.z=nzz; tr.rotation=Quat::from_rotation_y(-cr.head);
        cr.energy -= (2.6 + 5.0*thrust)*dt; cr.age+=1;
        match cr.kind {
            Kind::Herb => {
                for &(fe,gx,gz) in &fsnap { if (nx-gx).powi(2)+(nzz-gz).powi(2) < 12.0 && !eaten.contains(&fe) {
                    eaten.push(fe); cr.energy=(cr.energy+30.0).min(120.0); cr.knowledge=(cr.knowledge+0.12).min(1.0); break; } }
                if !cr.hidden && td < 16.0 { deaths.insert(e); }
            }
            Kind::Pred => { for &(_,ox,oz,okind,_,ohidden) in &snap { if okind==Kind::Herb && !ohidden && (nx-ox).powi(2)+(nzz-oz).powi(2) < 20.0 { cr.energy=(cr.energy+24.0).min(120.0); break; } } }
        }
        let repro_e = if cr.kind==Kind::Pred {96.0} else {88.0};
        if cr.energy>repro_e && cr.age>60 { cr.energy-=42.0; births.push((cr.kind,nx,nzz,cr.head,cr.brain.genome(),cr.clan,cr.gener+1)); }
        if cr.energy<=0.0 { deaths.insert(e); }
    }
    // aplica: comida comida respawna noutro lugar
    for fe in &eaten { c.entity(*fe).despawn(); let (rx,rz)=(sim.rng.r(-HALF,HALF),sim.rng.r(-HALF,HALF)); let d=dom_at(rx,rz);
        let mf=sim.m_food.clone(); let mm=sim.mat_food[d as usize].clone();
        c.spawn((Mesh3d(mf), MeshMaterial3d(mm), Transform::from_xyz(rx,1.3,rz), Food{dom:d})); }
    // nascimentos (evolução): filho = pai mutado
    let mut added=0usize;
    for (kind,x,z,head,pg,clan,gener) in births { if ncre+added>=170 {break;}
        let mut cg=pg; for v in cg.iter_mut() { if sim.rng.f()<0.15 { *v=(*v+sim.rng.g()*0.15).clamp(-3.0,3.0); } }
        let (mesh,mat,y)= match kind { Kind::Herb=>(sim.m_herb.clone(),sim.mat_clan[clan as usize%8].clone(),2.2), Kind::Pred=>(sim.m_pred.clone(),sim.mat_pred.clone(),3.0) };
        c.spawn((Mesh3d(mesh), MeshMaterial3d(mat), Transform::from_xyz((x+3.0).clamp(-HALF,HALF),y,z),
            Creature{ kind, brain:Brain::from_genome(&cg), head, energy:48.0, age:0, gener, clan, hidden:false, knowledge:0.0, sensed:[0.0;NIN] }));
        if kind==Kind::Herb {sim.born_h+=1;} else {sim.born_p+=1;} added+=1;
    }
    for e in deaths { c.entity(e).despawn(); sim.died+=1; if sim.sel==Some(e) {sim.sel=None;} }
    sim.tick+=1;
}

// ---------------- CÂMERA (segue o selecionado · WASD move · roda=zoom) ----------------
fn camera_ctrl(time:Res<Time>, keys:Res<ButtonInput<KeyCode>>, mut wheel:EventReader<bevy::input::mouse::MouseWheel>,
    mut sim:ResMut<Sim>, mut cam:Query<(&mut Transform,&mut FollowCam)>,
    creatures:Query<&Transform,(With<Creature>,Without<FollowCam>)>) {
    if keys.just_pressed(KeyCode::Space) { sim.paused=!sim.paused; }
    if keys.just_pressed(KeyCode::ArrowUp) { sim.speed=(sim.speed*1.4).min(6.0); }
    if keys.just_pressed(KeyCode::ArrowDown) { sim.speed=(sim.speed/1.4).max(0.2); }
    let Ok((mut tr,mut fc))=cam.single_mut() else {return};
    let dt=time.delta_secs();
    if let Some(sel)=sim.sel { if let Ok(ct)=creatures.get(sel) { fc.target=fc.target.lerp(ct.translation, 0.12); } }
    let pan=110.0*dt*(fc.dist/230.0);
    if keys.pressed(KeyCode::KeyW) { fc.target.x-=fc.yaw.sin()*pan; fc.target.z-=fc.yaw.cos()*pan; }
    if keys.pressed(KeyCode::KeyS) { fc.target.x+=fc.yaw.sin()*pan; fc.target.z+=fc.yaw.cos()*pan; }
    if keys.pressed(KeyCode::KeyA) { fc.target.x-=fc.yaw.cos()*pan; fc.target.z+=fc.yaw.sin()*pan; }
    if keys.pressed(KeyCode::KeyD) { fc.target.x+=fc.yaw.cos()*pan; fc.target.z-=fc.yaw.sin()*pan; }
    if keys.pressed(KeyCode::KeyQ) { fc.yaw-=1.4*dt; } if keys.pressed(KeyCode::KeyE) { fc.yaw+=1.4*dt; }
    for ev in wheel.read() { fc.dist=(fc.dist - ev.y*16.0).clamp(35.0,520.0); }
    fc.target.x=fc.target.x.clamp(-HALF,HALF); fc.target.z=fc.target.z.clamp(-HALF,HALF);
    let (sy,cy)=(fc.yaw.sin(),fc.yaw.cos()); let sp2=fc.pitch.sin();
    let off=Vec3::new(sy*sp2, fc.pitch.cos(), cy*sp2)*fc.dist;
    *tr = Transform::from_translation(fc.target+off).looking_at(fc.target, Vec3::Y);
}

// ---------------- CLIQUE = selecionar (raycast p/ o plano dos seres) ----------------
fn click_select(mouse:Res<ButtonInput<MouseButton>>, windows:Query<&Window>,
    cams:Query<(&Camera,&GlobalTransform)>, creatures:Query<(Entity,&Transform),With<Creature>>, mut sim:ResMut<Sim>) {
    if !mouse.just_pressed(MouseButton::Left) { return; }
    let (Ok(win),Ok((cam,ct)))=(windows.single(),cams.single()) else {return};
    let Some(cur)=win.cursor_position() else {return};
    let Ok(ray)=cam.viewport_to_world(ct,cur) else {return};
    let d=ray.direction.as_vec3(); if d.y.abs()<1e-4 {return;}
    let t=(2.5-ray.origin.y)/d.y; if t<0.0 {return;}
    let hit=ray.origin+d*t;
    let (mut best,mut bid)=(1e9f32,None);
    for (e,tr) in &creatures { let dd=(tr.translation.x-hit.x).powi(2)+(tr.translation.z-hit.z).powi(2); if dd<best {best=dd;bid=Some(e);} }
    if best<120.0 { sim.sel=bid; }
}

// ---------------- HUD ----------------
fn hud(sim:Res<Sim>, creatures:Query<(Entity,&Creature)>, mut q:Query<&mut Text,With<Hud>>) {
    let (mut nh,mut np,mut gsum)=(0u32,0u32,0u64);
    for (_,cr) in &creatures { if cr.kind==Kind::Herb {nh+=1} else {np+=1}; gsum+=cr.gener as u64; }
    let tot=(nh+np).max(1);
    let mut s=format!("IARA-VIDA 3D · HERBIVOROS {}  PREDADORES {}  ·  geracao media {:.1}  ·  nasceram H{}/P{}  mortes {}  ·  veloc {:.1}{}",
        nh,np, gsum as f32/tot as f32, sim.born_h, sim.born_p, sim.died, sim.speed, if sim.paused {"  [PAUSA]"} else {""});
    if let Some(sel)=sim.sel { if let Some((_,cr))=creatures.iter().find(|(e,_)| *e==sel) {
        let kind=if cr.kind==Kind::Pred {"PREDADOR"} else {"HERBIVORO"};
        let bar=|v:f32,mx:f32|{ let n=14usize; let f=((v/mx)*n as f32).round().clamp(0.0,n as f32) as usize; format!("[{}{}]","#".repeat(f)," ".repeat(n-f)) };
        s+=&format!("\n\nSELECIONADO: {} · clã {} · geração {} · idade {}{}", kind, cr.clan, cr.gener, cr.age, if cr.hidden {" · ESCONDIDO no arbusto" } else {""});
        s+=&format!("\nenergia {} {:>3.0}    conhecimento {} {:.2}", bar(cr.energy,120.0), cr.energy, bar(cr.knowledge,1.0), cr.knowledge);
    } } else { s+="\n\nclique num ser pra a câmera seguir e ver o cérebro/energia/conhecimento dele."; }
    if let Ok(mut t)=q.single_mut() { *t=Text::new(s); }
}
