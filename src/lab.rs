use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct LabParams {
    // Metadados
    pub experiment_name: String,
    pub variant_id: u32,

    // Axiom policy
    pub axiom_wall_avoidance: f32,
    pub axiom_dark_center: f32,
    pub axiom_whisker: f32,
    pub axiom_racing_line: f32,
    pub axiom_decay_divisor: f32,
    pub axiom_floor: f32,

    // Fitness weights
    pub k_smooth: f32,
    pub k_speed: f32,
    pub k_novel: f32,
    pub k_stab: f32,
    pub k_phi: f32,
    pub k_aesth: f32,
    pub lap_bonus: f32,
    pub survival_bonus: f32,
    pub death_penalty: f32,

    /// Via do cérebro a DESLIGAR neste processo (ver `neuro.rs`). Vazio = linha de base.
    /// `#[serde(default)]` porque os specs antigos não têm o campo e têm de seguir carregando.
    #[serde(default)]
    pub via_ablada: String,

    /// Como os neurônios MOTORES nascem (ver `neuro::Nascimento`):
    /// vazio/"primado" = canais escolhidos à mão · "esparso" = mesma quantidade de conexões
    /// com canais SORTEADOS (independente) · "denso" = aleatório sem esparsidade (piso).
    #[serde(default)]
    pub nascimento: String,

    /// Até onde a linha mais distante da retina enxerga, em metros. 0 = padrão (33,3 m).
    /// Acima de ~28 m/s a frenagem exigida passa desses 33,3 m e o carro fica cego para o
    /// ponto de frenagem — ver `visao.rs`.
    #[serde(default)]
    pub visao_alcance_m: f32,
    /// Abertura horizontal da retina em graus. 0 = padrão (120°).
    #[serde(default)]
    pub visao_fov_deg: f32,

    /// Fração da população que nasce na faixa de mutação DESTRUTIVA (σ 0,08–0,20).
    /// 0 = padrão (0,50, ou seja slots 30-59 de 60).
    ///
    /// Medido: com a fração padrão a MEDIANA da população é **zero metros** — metade não sai
    /// do lugar, e `wall_mean = 0,00s` confirma que não é batida, é mutação destrutiva.
    /// A população efetiva é 30, não 60.
    #[serde(default)]
    pub fracao_exploradora: f32,

    /// Se `true`, o bônus de volta é CONTÍNUO (voltas fracionárias) em vez de contagem
    /// inteira. 0/ausente = comportamento antigo, para poder comparar.
    ///
    /// # O degrau dentro da fitness
    ///
    /// `lap_bonus = total_laps as f32 * 50000` usa a contagem INTEIRA, enquanto `prog` chega
    /// a ~5800 m. O bônus domina o progresso por ~9×, então dentro de um mesmo balde de volta
    /// a seleção enxerga quase nada:
    ///
    /// | carro | fitness | contra o de 3,01 |
    /// |---|---|---|
    /// | 3,01 voltas | 155.000 | — |
    /// | 3,99 voltas | 156.600 | **+1%** |
    /// | 4,00 voltas | 206.600 | +32% |
    ///
    /// Ou seja: **ir de 3,0 para 3,99 voltas quase não é premiado**, e é exatamente aí que a
    /// população estava (melhor carro em 3,72 de um teto físico de 4,36). O degrau da métrica
    /// que escondeu as voltas ([[feedback_metrica_de_degrau]]) também estava DENTRO do sinal
    /// de seleção, cegando a evolução na faixa onde ela precisava enxergar.
    ///
    /// Contínuo: `lap_bonus * (prog / comprimento_da_volta)`. Mesma escala, mesmo incentivo a
    /// fechar voltas, sem o degrau.
    #[serde(default)]
    pub lap_bonus_continuo: bool,

    /// Se `true`, o limiar de nicho é ADAPTATIVO (mediana das distâncias do pool) em vez do
    /// fixo `NICHE_DISTANCE = 8`. Ausente/false = comportamento antigo, para comparar.
    ///
    /// Medido depois de consertar o fingerprint: a distância típica entre elites é 2,4–5,3 e
    /// o máximo 8–10 — o limiar fixo de 8 é 1,5–3× maior que a típica e engole tudo num
    /// cluster só, em toda geração de todo treino. A calibração histórica ("v14: tentou 10,
    /// piorou") é nula: foi feita quando o fingerprint saturava e toda distância era 0.
    #[serde(default)]
    pub nicho_adaptativo: bool,

    /// Liga os kernels de percepção (e futuramente cérebro/física) na GPU.
    ///
    /// Quando `true`, o kernel `percepcao.wgsl` é encadeado APÓS a retina GPU já existente.
    /// O buffer de retina física permanece em VRAM — nenhum round-trip CPU→GPU nesse caminho.
    /// Os canais temporais (motion, approach_rate, EMA, racing-line, etc.) continuam sendo
    /// escritos pela CPU POR CIMA do resultado do kernel, exatamente como documentado em
    /// `gpu/perception.rs::CANAIS_NA_CPU`.
    ///
    /// Default `false` para preservar comportamento antigo em todos os experimentos já
    /// rodados. Ative passando `"gpu_core": true` no params.json da corrida.
    #[serde(default)]
    pub gpu_core: bool,

    /// MODO VIDA: parar de matar por lentidão e deixar o combustível ser o relógio.
    ///
    /// Autópsia com a retina em GPU, 2 gerações de 60 seres: **`BATIDA: 0`**. Ninguém morre
    /// batendo. Morrem de `SpeedBleed` (43 e 57 de 60) — sangram por andar devagar, 60 HP/s
    /// depois de 2,8 s abaixo de 4 m/s. É execução por lentidão, não seleção por perigo.
    ///
    /// Leonardo: *"podemos parar de matar os modelos e deixar eles livres na pista; eles
    /// podem bater e perder vida, mas para morrer levaria muito tempo. Teriam necessidade de
    /// combustível, então precisam caçar — o que seriam as voltas"*.
    ///
    /// Com a flag ligada:
    ///   * SpeedBleed cai de 60 para 2 HP/s — leva ~2,5 min parado para matar, não 5 s;
    ///   * morte por estagnação e por raspar parede deixam de matar e viram DANO;
    ///   * combustível vira o relógio real e a volta reabastece (era a cada 3 voltas).
    ///
    /// A vida passa a acabar por não caçar, que é a pressão que a tese quer.
    #[serde(default)]
    pub modo_vida: bool,

    // Survival thresholds
    pub speed_bleed_damage: f32,
    pub stagnation_secs: f32,
    pub wall_t_threshold: i8,
}

impl Default for LabParams {
    fn default() -> Self {
        Self {
            experiment_name: "default".to_string(),
            variant_id: 0,
            axiom_wall_avoidance: 0.18,
            axiom_dark_center: 0.45,
            axiom_whisker: 0.15,
            axiom_racing_line: 0.26,
            axiom_decay_divisor: 50.0,
            axiom_floor: 0.35,
            k_smooth: 100.0,
            k_speed: 0.5,
            k_novel: 50.0,
            k_stab: 200.0,
            k_phi: 300.0,
            k_aesth: 200.0,
            lap_bonus: 50000.0,
            survival_bonus: 3000.0,
            death_penalty: 2000.0,
            via_ablada: String::new(),
            nascimento: String::new(),
            visao_alcance_m: 0.0,
            visao_fov_deg: 0.0,
            fracao_exploradora: 0.0,
            lap_bonus_continuo: false,
            nicho_adaptativo: false,
            gpu_core: false,
            modo_vida: false,
            speed_bleed_damage: 15.0,
            stagnation_secs: 15.0,
            wall_t_threshold: 40,
        }
    }
}

impl LabParams {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

use bevy::prelude::Resource;
