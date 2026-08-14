//! Painel HTTP tempo-real — REFACTOR_PLAN.md §E5 / §3, ativado por `--painel`.
//!
//! Servidor HTTP minimalista (`std::net::TcpListener`, sem framework) numa thread própria
//! dentro do processo `gpu_evolve`. Existe para o Leonardo VER a corrida enquanto ela roda —
//! sem isto a única forma de acompanhar é ler `stdout` linha a linha. Estado compartilhado via
//! `Arc<Mutex<..>>`; um cliente por vez é suficiente (é diagnóstico, não produção).
//!
//! # Custo quando desligado
//!
//! Sem `--painel` nenhuma thread é lançada, nenhum `Arc`/`Mutex` é alocado — `main.rs` só
//! constrói `Option<PainelHandle>` e todo ponto de atualização é `if let Some(p) = &painel`.
//!
//! # Dois níveis de detalhe da ativação (E4)
//!
//! `PainelState` carrega só os 4 números resumo de L1 (`dead_pct_l1` etc, exatamente os campos
//! do REFACTOR_PLAN.md §3.2) — é o que `/api/status` devolve, consumido pelo indicador
//! numérico do topo do painel. `/api/activation` devolve `ActivationSnapshot`, que tem as DUAS
//! camadas (L1 e L2) com histograma de 32 bins cada — é o que a tabela de alarmes e o
//! histograma do painel consomem. `update_activation()` escreve nos dois de uma vez, para que
//! nunca divirjam.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// Estado "de geração" + "de tick" — tudo que `/api/status` devolve. Campos batem com
/// REFACTOR_PLAN.md §3.2, na mesma ordem.
#[derive(Default, Serialize, Clone)]
pub struct PainelState {
    // --- status da geração (atualizado no fim de cada geração) ---
    // `gen` é palavra reservada desde a edition 2024 (sintaxe de generators) — campo
    // chama `geracao` por isso, não por preferência de nome.
    pub geracao: u32,
    pub mediana_m: f64,
    pub elite_m: f64,
    pub mortos_pct: f32,
    /// Progresso médio / progresso do piloto sensorial. `gpu_evolve` não calcula essa
    /// razão em nenhum ponto do laço de produção hoje (é `replay_mapa.rs::F58` quem faz
    /// isso, OFFLINE, sobre um `.replay` já gravado) — fica em 0.0 até algum caller real
    /// alimentar `update_obediencia()`. "se disponível" do enunciado: não está.
    pub obediencia_pct: f32,
    /// Fração da população que já cruzou CADA setor, em ordem (`setores[s]` = fração com
    /// `checkpoint > s`). Não é "fitness média por setor" ao pé da letra — decidimos por
    /// esta curva de progresso porque é o que `checkpoint: Vec<u32>` já contém sem
    /// recomputar nada, e responde à pergunta que a corrida real interessa: ONDE a
    /// população emperra. Documentado aqui para não virar leitura errada do nome do campo.
    pub setores: Vec<f32>,
    /// OLS slope (m/gen) da mediana até esta geração — `universe::gpu::evolution::linear_slope`
    /// sobre o histórico acumulado (0.0 com menos de 2 gerações).
    pub slope: f64,

    // --- snapshot de tick (atualizado a cada N ticks, default N=64) ---
    pub tick: u64,
    pub vivos: u32,
    pub vel_mediana: f32,
    pub life_mediana: f32,
    pub fuel_mediana: f32,

    // --- ativação (atualizado no fim de cada geração, só com --ativacao) ---
    pub activation_alarms: Vec<String>,
    pub dead_pct_l1: f32,
    pub saturated_pct_l1: f32,
    pub spectral_radius: f32,
    pub snr_i8: f32,

    /// Traçado da pista em coordenadas de mundo (x, z) — enviado uma vez, em `set_track()`,
    /// e reaproveitado o resto da corrida: o mundo não muda entre gerações. Vai junto de
    /// `/api/status` (não um endpoint à parte) porque o painel HTML já lê esse endpoint a
    /// 1 Hz e o traçado é pequeno (poucos KB) — replicar noutra rota só duplicaria o poll.
    pub waypoints: Vec<[f32; 2]>,
}

/// Um ser, num instante — o que `/api/beings` devolve.
#[derive(Default, Serialize, Clone)]
pub struct BeingSnapshot {
    pub id: usize,
    pub x: f32,
    pub z: f32,
    pub vel: f32,
    /// Vida normalizada em [0, 1] (`life_left / life_cap()`), não ticks brutos — o HTML
    /// colore o ponto direto por este valor sem precisar saber o teto.
    pub life: f32,
    pub vivo: bool,
}

/// Uma camada (L1 ou L2) dentro de `/api/activation`.
#[derive(Default, Serialize, Clone)]
pub struct LayerSnapshot {
    pub name: String,
    pub dead_pct: f32,
    pub saturated_pct: f32,
    pub spectral_radius: f32,
    pub effective_rank: usize,
    pub snr_i8: f32,
    /// 32 bins, mesmo agregado de `analysis::activation::LayerStats::histogram`.
    pub histogram: Vec<u32>,
    /// Ativacao INSTANTANEA de cada neuronio do ser observado, em [-1, 1].
    ///
    /// Leonardo: *"o cerebro.html nao pode ser estatico — gostaria de ver o cerebro em
    /// funcionamento, como ele fica ativo na pista, o que ativa."* Os agregados acima
    /// (morto/saturado/histograma) contam a HISTORIA da camada; este vetor e o AGORA:
    /// quais neuronios estao disparando neste tick, enquanto o ser dirige. E o que
    /// transforma o painel de relatorio em observacao.
    pub live: Vec<f32>,
    /// Ativacao media por neuronio ao longo da geracao — separa "disparou agora" de
    /// "dispara sempre". Neuronio com media alta e variancia zero e um vies, nao um sensor.
    pub mean_per_neuron: Vec<f32>,
}

/// O que `/api/activation` devolve — mais detalhe que os 4 campos resumo de `PainelState`.
#[derive(Default, Serialize, Clone)]
pub struct ActivationSnapshot {
    pub layers: Vec<LayerSnapshot>,
    pub alarms: Vec<String>,
}

/// HTML embutido no binário — zero I/O em runtime, zero CDN (canvas + JS puro).
const HTML: &str = include_str!("painel.html");

/// Alça do painel: o que `main.rs`/`context.rs` chamam para publicar dados. Lançar
/// (`launch`) só faz sentido com `--painel`; sem a flag `main.rs` nunca constrói isto.
pub struct PainelHandle {
    state: Arc<Mutex<PainelState>>,
    beings: Arc<Mutex<Vec<BeingSnapshot>>>,
    activation: Arc<Mutex<ActivationSnapshot>>,
}

impl PainelHandle {
    /// Sobe o `TcpListener` numa thread separada e devolve a alça imediatamente — não
    /// bloqueia `main()`. Se a porta já estiver em uso, avisa em stderr e desliga o painel
    /// (o resto do processo continua rodando: um diagnóstico não pode derrubar a corrida).
    pub fn launch(port: u16) -> Self {
        let state = Arc::new(Mutex::new(PainelState::default()));
        let beings = Arc::new(Mutex::new(Vec::<BeingSnapshot>::new()));
        let activation = Arc::new(Mutex::new(ActivationSnapshot::default()));

        let state_l = Arc::clone(&state);
        let beings_l = Arc::clone(&beings);
        let activation_l = Arc::clone(&activation);

        std::thread::spawn(move || {
            let listener = match TcpListener::bind(("0.0.0.0", port)) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[painel] ERRO ao abrir a porta {port}: {e} — painel DESLIGADO, \
                               a corrida continua sem ele.");
                    return;
                }
            };
            println!("[painel] servindo em http://localhost:{port}  (Ctrl+C não derruba só o painel — mata o processo inteiro)");
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => serve_one(stream, &state_l, &beings_l, &activation_l),
                    Err(_) => continue, // conexão que caiu no meio do accept — ignora e segue.
                }
            }
        });

        Self { state, beings, activation }
    }

    /// Traçado da pista — chamar UMA vez, logo depois de `VoxelWorld::new()`, antes do
    /// laço de gerações.
    pub fn set_track(&self, waypoints_world: &[bevy::math::Vec2]) {
        let mut st = self.state.lock().unwrap();
        st.waypoints = waypoints_world.iter().map(|w| [w.x, w.y]).collect();
    }

    /// Fim de geração: os números que hoje já viram `println!("gen ...")` em `main.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_gen(
        &self,
        geracao: u32,
        mediana_m: f64,
        elite_m: f64,
        mortos_pct: f32,
        obediencia_pct: f32,
        setores: Vec<f32>,
        slope: f64,
    ) {
        let mut st = self.state.lock().unwrap();
        st.geracao = geracao;
        st.mediana_m = mediana_m;
        st.elite_m = elite_m;
        st.mortos_pct = mortos_pct;
        st.obediencia_pct = obediencia_pct;
        st.setores = setores;
        st.slope = slope;
    }

    /// Ativação L1/L2 (E4, só chamado quando `--ativacao` está ligado). Alimenta tanto o
    /// resumo em `PainelState` (L1) quanto o detalhe completo de `/api/activation` (L1+L2).
    pub fn update_activation(&self, layers: Vec<LayerSnapshot>, alarms: Vec<String>) {
        if let Some(l1) = layers.iter().find(|l| l.name == "L1") {
            let mut st = self.state.lock().unwrap();
            st.dead_pct_l1 = l1.dead_pct;
            st.saturated_pct_l1 = l1.saturated_pct;
            st.spectral_radius = l1.spectral_radius;
            st.snr_i8 = l1.snr_i8;
            st.activation_alarms = alarms.clone();
        }
        let mut act = self.activation.lock().unwrap();
        act.layers = layers;
        act.alarms = alarms;
    }

    /// A cada N ticks (`main.rs` decide N, default 64): posições + velocidade + vida de
    /// TODOS os seres, mais os agregados medianos que vão para `PainelState`.
    pub fn update_tick(&self, tick: u64, snaps: Vec<BeingSnapshot>, fuel_mediana: f32) {
        let vivos = snaps.iter().filter(|b| b.vivo).count() as u32;
        let vel_mediana = mediana(snaps.iter().filter(|b| b.vivo).map(|b| b.vel));
        let life_mediana = mediana(snaps.iter().filter(|b| b.vivo).map(|b| b.life));
        {
            let mut st = self.state.lock().unwrap();
            st.tick = tick;
            st.vivos = vivos;
            st.vel_mediana = vel_mediana;
            st.life_mediana = life_mediana;
            st.fuel_mediana = fuel_mediana;
        }
        *self.beings.lock().unwrap() = snaps;
    }
}

/// Mediana de um iterador de f32 — população vazia (ninguém vivo) devolve 0.0 em vez de
/// pânico no `unwrap()` do `sort_by`.
fn mediana(xs: impl Iterator<Item = f32>) -> f32 {
    let mut v: Vec<f32> = xs.collect();
    if v.is_empty() { return 0.0; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Extrai o path da primeira linha da requisição HTTP (`GET /api/status HTTP/1.1`).
/// Parse mínimo de propósito — este servidor só precisa distinguir 4 rotas fixas.
fn extract_path(req: &str) -> &str {
    req.lines().next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/")
}

/// Atende UMA conexão e fecha. HTTP/1.1 mínimo: status line + Content-Type + Content-Length
/// + corpo. `Connection: close` explícito — sem keep-alive não há por que o cliente esperar
/// outra resposta na mesma conexão, e fechar cedo evita conexões penduradas acumulando.
fn serve_one(
    mut stream: TcpStream,
    state: &Arc<Mutex<PainelState>>,
    beings: &Arc<Mutex<Vec<BeingSnapshot>>>,
    activation: &Arc<Mutex<ActivationSnapshot>>,
) {
    let mut buf = [0u8; 1024];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = extract_path(&req);

    let (status, ct, body): (&str, &str, String) = match path {
        "/" => ("200 OK", "text/html; charset=utf-8", HTML.to_string()),
        "/api/status" => {
            let st = state.lock().unwrap().clone();
            ("200 OK", "application/json", serde_json::to_string(&st).unwrap_or_else(|_| "{}".into()))
        }
        "/api/activation" => {
            let act = activation.lock().unwrap().clone();
            ("200 OK", "application/json", serde_json::to_string(&act).unwrap_or_else(|_| "{}".into()))
        }
        "/api/beings" => {
            let b = beings.lock().unwrap().clone();
            ("200 OK", "application/json", serde_json::to_string(&b).unwrap_or_else(|_| "[]".into()))
        }
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    };

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}
