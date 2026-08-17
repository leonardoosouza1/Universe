//! Teste de paridade CPU × GPU do CfC de percepção (`retina_digest.wgsl` vs
//! `iara_cfc::CfcCell::step`, via `RetinaPerception`).
//!
//! # Por que `#[ignore]`
//!
//! Este projeto não roda compute real de GPU dentro de `cargo test` (grep por
//! `RetinaGpu::new`/`PerceptionGpu::new` dentro de `#[cfg(test)]` no resto do crate
//! não acha nenhum) — os kernels são validados por binários manuais (`gpu_validate`,
//! `pista_check`). Este teste seria o primeiro a abrir um `wgpu::Device` dentro de
//! `cargo test`, o que quebraria em máquinas/CI sem GPU. Fica `#[ignore]` para não
//! travar `cargo test --release --lib`; roda explicitamente com:
//!
//! ```text
//! HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
//!   cargo test --release --lib -- --ignored retina_digest_parity
//! ```
//!
//! # O que ele prova
//!
//! Roda `RetinaDigestGpu` por N ticks com retinas sintéticas (variadas, incluindo
//! parede nas colunas centrais) e compara, tick a tick:
//! - ch192-207 (canal do CfC) contra a mesma fórmula calculada na CPU a partir do
//!   estado hidden ANTERIOR (preserva a defasagem de 1 tick documentada no kernel);
//! - `min_wall_ahead`/`edge_row6`/`edge_row7` contra a lógica que existia em
//!   `context.rs` antes desta migração (reproduzida aqui só como oráculo de teste —
//!   não é uma segunda cópia de produção, é o "antes" contra o "depois").
//!
//! O estado recorrente h em si (`pipeline.state.h` da CPU vs `cfc_state` da GPU) é
//! validado INDIRETAMENTE: qualquer divergência na recorrência aparece nos canais do
//! tick SEGUINTE, porque `h_new` desta iteração vira o "h antigo" lido no próximo
//! dispatch — um teste de vários ticks encadeados pega divergência acumulada que um
//! teste de tick único não pegaria.

use super::RetinaDigestGpu;
use crate::config::iara::vision::{RETINA_COLS, RETINA_TOTAL};
use crate::gpu::perception::WAVE_U32S;
use crate::gpu::{Pixel, legacy_encoding};
use crate::perception::RetinaPerception;

const N_BEINGS: usize = 2;
const N_TICKS: usize = 4;

/// Gera uma retina sintética determinística (packed `dist_mm<<8|material`), variando
/// por `(being, tick)` para exercer o CfC com entradas diferentes tick a tick — sem
/// isso, `h` convergiria trivialmente e o teste não pegaria erro de recorrência.
/// Inclui parede (material=2) nas colunas centrais [42,84) em alguns pixels para
/// exercer o caminho de `min_wall_ahead`.
fn synthetic_retina(being: usize, tick: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(RETINA_TOTAL);
    let seed = (being as u32)
        .wrapping_mul(97)
        .wrapping_add((tick as u32).wrapping_mul(131));
    for j in 0..RETINA_TOTAL {
        let jx = j as u32;
        let h = jx
            .wrapping_mul(2654435761)
            .wrapping_add(seed)
            .wrapping_mul(0x9E3779B1);
        let dist_mm = (h % 60_000) as u32; // 0..60 m
        let col = j % RETINA_COLS;
        // Força parede nas colunas centrais em ~1/4 dos pixels dessa faixa, perto
        // (< 20 m), para que min_wall_ahead tenha um valor real para comparar.
        let material: u32 = if col >= RETINA_COLS / 3 && col < 2 * RETINA_COLS / 3 && (h % 4) == 0 {
            2
        } else {
            h % 4
        };
        let dist_mm = if material == 2 {
            dist_mm % 20_000
        } else {
            dist_mm
        };
        out.push((dist_mm << 8) | material);
    }
    out
}

/// Oráculo de teste: reproduz a lógica que `context.rs::tick()` fazia ANTES desta
/// migração (min_wall_ahead + edge_row6/7 a partir da retina física crua). Não é
/// código de produção — existe só para o teste ter um "antes" independente do WGSL
/// para comparar contra o "depois".
fn cpu_wall_and_edges(retina_packed: &[u32]) -> (f32, f32, f32) {
    let mut min_wall_ahead = f32::MAX;
    let mut retina_i8 = vec![0i8; RETINA_TOTAL];
    for (j, &packed) in retina_packed.iter().enumerate() {
        let dist_m = (packed >> 8) as f32 / 1000.0;
        let material = (packed & 0xFF) as u8;
        if material == 2 {
            let col = j % RETINA_COLS;
            if col >= RETINA_COLS / 3 && col < 2 * RETINA_COLS / 3 {
                min_wall_ahead = min_wall_ahead.min(dist_m);
            }
        }
        retina_i8[j] = legacy_encoding(Pixel { dist_m, material });
    }
    let mut edge = [0f32; 2];
    for (k, &row_idx) in [6usize, 7usize].iter().enumerate() {
        let edge_sum: i32 = (0..RETINA_COLS - 1)
            .map(|col| {
                let a = retina_i8[row_idx * RETINA_COLS + col] as i32;
                let b = retina_i8[row_idx * RETINA_COLS + col + 1] as i32;
                (a - b).abs()
            })
            .sum();
        edge[k] = (edge_sum as f32 / 6000.0).clamp(0.0, 1.0);
    }
    (min_wall_ahead, edge[0], edge[1])
}

/// Oráculo de teste: mesma fórmula de conversão hidden→canal copiada em
/// `retina_digest.wgsl` (aproximação x/(1+|x|), NÃO tanh exato — ver doc do kernel).
fn channel_from_hidden(h: &[f32]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (i, &hi) in h.iter().enumerate() {
        let abs_h = hi.abs();
        let tanh_h = if abs_h > 10.0 {
            hi.signum() * 0.999
        } else {
            hi / (1.0 + abs_h)
        };
        let val = (tanh_h * 127.0 * 0.3).clamp(-127.0, 127.0);
        out[i] = val as i8;
    }
    out
}

async fn open_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("retina-digest-parity-test"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .ok()
}

#[test]
#[ignore = "abre GPU real — rodar manualmente: cargo test --release --lib -- --ignored retina_digest_parity"]
fn retina_digest_bate_com_cpu_ao_longo_de_varios_ticks() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[retina_digest_parity] sem GPU disponível — teste pulado");
        return;
    };

    let digest = RetinaDigestGpu::new(&device, &queue, N_BEINGS);

    let buf_retina = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-retina"),
        size: (N_BEINGS * RETINA_TOTAL * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // P1.8: `output` (buf_output_shared) continua um parâmetro obrigatório de
    // `bind_and_dispatch` — o bind group layout é compartilhado com
    // `perception.rs` — mas o kernel NÃO escreve mais em ch192-207 dele (achado
    // "ch192-207 aliasa pro PRÓXIMO ser", corrigido: os 16 valores agora vão
    // para `extra_out[3..19]`, ver comparação abaixo). `buf_output` só precisa
    // existir e ser do tamanho certo pro bind group não falhar — não há mais
    // leitura de corrupção pra fazer aqui, então nenhum readback dele é
    // necessário neste teste.
    let bytes_output = (N_BEINGS * WAVE_U32S * 4) as u64;
    let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-output"),
        size: bytes_output,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Oráculo CPU: mesma seed fixa que RetinaDigestGpu::new() usa internamente para
    // subir os pesos — os dois lados partem dos MESMOS pesos de CfC.
    let mut cpu_percept: Vec<RetinaPerception> =
        (0..N_BEINGS).map(|_| RetinaPerception::new()).collect();

    for tick in 0..N_TICKS {
        let mut retina_data = Vec::with_capacity(N_BEINGS * RETINA_TOTAL);
        let mut retinas_per_being = Vec::with_capacity(N_BEINGS);
        for being in 0..N_BEINGS {
            let r = synthetic_retina(being, tick);
            retina_data.extend_from_slice(&r);
            retinas_per_being.push(r);
        }
        queue.write_buffer(&buf_retina, 0, bytemuck::cast_slice(&retina_data));

        // Canal esperado ANTES de avançar o CfC — o kernel escreve com o h "velho".
        let expected_channels: Vec<[i8; 16]> = cpu_percept
            .iter()
            .map(|p| channel_from_hidden(&p.pipeline.state.h))
            .collect();

        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&buf_output, 0, Some(bytes_output));
        let bytes_extra = digest.bind_and_dispatch(
            &mut enc,
            &device,
            &queue,
            N_BEINGS,
            &buf_retina,
            &buf_output,
        );
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // ── Lê extra_out: min_wall_ahead/edge_row6/edge_row7 (slots 0-2) e o
        // estado CfC squashed "ch192-207" (slots 3-18, achado P1.8 — deixou de
        // aliasar em `buf_output` e passou a viver aqui, espaço próprio por ser) ──
        let extra = digest.read_after_submit(&device, N_BEINGS, bytes_extra);
        for being in 0..N_BEINGS {
            let (want_wall, want_e6, want_e7) = cpu_wall_and_edges(&retinas_per_being[being]);
            let row = extra[being];
            let (got_wall, got_e6, got_e7) = (row[0], row[1], row[2]);
            // min_wall_ahead: se nenhuma parede qualificou, os dois lados usam
            // sentinelas GRANDES mas DIFERENTES (f32::MAX vs 1e9) — só comparamos
            // o valor exato quando uma parede real foi encontrada.
            if want_wall < 1.0e8 {
                assert!(
                    (got_wall - want_wall).abs() < 0.01,
                    "tick={tick} being={being} min_wall_ahead: GPU={got_wall} CPU={want_wall}"
                );
            } else {
                assert!(
                    got_wall > 1.0e8,
                    "tick={tick} being={being} esperava sentinela, GPU={got_wall}"
                );
            }
            assert!(
                (got_e6 - want_e6).abs() < 0.01,
                "tick={tick} being={being} edge_row6: GPU={got_e6} CPU={want_e6}"
            );
            assert!(
                (got_e7 - want_e7).abs() < 0.01,
                "tick={tick} being={being} edge_row7: GPU={got_e7} CPU={want_e7}"
            );
            for ci in 0..16usize {
                let got = row[3 + ci];
                let want = expected_channels[being][ci] as f32;
                assert!(
                    (got - want).abs() <= 1.0,
                    "tick={tick} being={being} ch192+{ci} (agora extra_out[{}]): GPU={got} CPU={want} (h_old={:?})",
                    3 + ci,
                    cpu_percept[being].pipeline.state.h
                );
            }
        }

        // ── Avança o CfC da CPU para o próximo tick (mesma retina que a GPU viu) ──
        for being in 0..N_BEINGS {
            let mut retina_i8 = vec![0i8; RETINA_TOTAL];
            for (j, &packed) in retinas_per_being[being].iter().enumerate() {
                let dist_m = (packed >> 8) as f32 / 1000.0;
                let material = (packed & 0xFF) as u8;
                retina_i8[j] = legacy_encoding(Pixel { dist_m, material });
            }
            cpu_percept[being].perceive(&retina_i8);
        }
    }
}
