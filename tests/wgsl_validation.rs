//! Validação estática de shaders WGSL modificados na FASE CANAIS (2026-08-14),
//! SEM GPU — `naga::front::wgsl::parse_str` + `naga::valid::Validator` rodam
//! inteiramente na CPU (parse + type-check), a mesma verificação que
//! `wgpu::Device::create_shader_module` faria internamente, mas sem precisar de
//! um device real.
//!
//! # Por que este teste existe
//!
//! Os kernels `.wgsl` do projeto são embutidos via `include_str!` como texto puro
//! — `cargo build`/`cargo test` NUNCA verificam a sintaxe ou os tipos do WGSL, só
//! que a string compila como `&str` em Rust. Um erro de sintaxe WGSL (parêntese
//! faltando, tipo errado, canal com nome duplicado) só apareceria em RUNTIME, na
//! primeira vez que `device.create_shader_module()` rodasse — ou seja, só numa
//! GPU real. A regra do projeto nesta sessão foi "não rode nada na GPU: há 2
//! baterias de currículo ocupando-a", então este teste é o único jeito de saber,
//! com `cargo test`, se os shaders editados (`perception_temporal.wgsl`,
//! `perception_motion.wgsl` — ch101, 109, 114-117, FASE CANAIS) ao menos
//! PARSEIAM e TYPE-CHECAM antes de alguém rodar `gpu_evolve` de verdade.
//!
//! O que isto NÃO prova: que a LÓGICA está certa (semântica dos canais,
//! normalizadores calibrados) — só que o shader é WGSL válido. Validação de
//! comportamento (σ_temporal/σ_entre via `--auditar-visao`) continua pendente,
//! documentada nos comentários dos próprios shaders.

fn validar_wgsl(nome: &str, fonte: &str) {
    let module = naga::front::wgsl::parse_str(fonte)
        .unwrap_or_else(|e| panic!("{nome}: falhou o PARSE de WGSL:\n{}", e.emit_to_string(fonte)));

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("{nome}: falhou a VALIDAÇÃO (type-check) de WGSL:\n{}", e.emit_to_string(fonte)));
}

/// Editado na FASE CANAIS: ch101 (curvatura 260m, reusa `curv_at_d`) e ch109
/// (derivada real da racing line, antes hardcoded 0.0).
#[test]
fn perception_temporal_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/perception_temporal.wgsl");
    validar_wgsl("perception_temporal.wgsl", fonte);
}

/// Editado na FASE CANAIS: ch114-117 (approach hemisférico + fóvea), nova função
/// `write_channel_signed`, `workgroup_size` de 10 para 11 threads.
#[test]
fn perception_motion_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/perception_motion.wgsl");
    validar_wgsl("perception_motion.wgsl", fonte);
}

/// FASE CANAIS (2026-08-14): inicialmente só comentários (header "NÃO portado"
/// atualizado para refletir ch114-117 saindo da lista de "só motor.rs").
///
/// FASE SENSORES, degrau 1 (2026-08-14, mudança substantiva): FOV do sweep de
/// parede/curvatura (BLOCO C/D/E) e alcances/lookaheads (BLOCO J/K) deixaram de ser
/// `const` de compilação e viraram genes herdáveis por ser, lidos de um buffer novo
/// (`sensors`, binding 7 de `layout_spatial`) — ver `gpu::evolution::N_SENSOR_GENES`/
/// `PerceptionGpu::upload_sensors`. Mudança estrutural (binding novo, bind group
/// layout com 1 entrada a mais, leitura por-ser no início de `main()`) — exatamente o
/// tipo de edição em que um erro de sintaxe/tipo só apareceria numa GPU real.
#[test]
fn perception_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/perception.wgsl");
    validar_wgsl("perception.wgsl", fonte);
}

/// Editado em P1.5 (`docs/PLANO_EVOLUCAO.md`): `Params` ganhou `out_gain` +
/// `norm_l2_by_std` (+ padding pra manter o uniform em múltiplo de 16 bytes), a
/// ETAPA 2 foi reestruturada para separar o matmul de L2 (guardado em LDS,
/// `w_l2r[48]` novo) do ganho/fadiga/tanh — com um `workgroupBarrier()` novo em
/// controle uniforme no meio — para permitir a normalização opcional por
/// desvio-padrão. Mudança estrutural (barrier novo, array LDS novo, `if`
/// aninhado calculando `l2_std`), não só troca de literal — é exatamente o tipo
/// de edição em que um erro de sintaxe/tipo só apareceria numa GPU real; este
/// teste pega isso sem precisar de uma.
#[test]
fn brain_wgsl_e_valido() {
    let fonte = include_str!("../src/gpu/brain.wgsl");
    validar_wgsl("brain.wgsl", fonte);
}
