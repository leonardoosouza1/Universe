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
use crate::gpu::{legacy_encoding, Pixel};
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
    let seed = (being as u32).wrapping_mul(97).wrapping_add((tick as u32).wrapping_mul(131));
    for j in 0..RETINA_TOTAL {
        let jx = j as u32;
        let h = jx.wrapping_mul(2654435761).wrapping_add(seed).wrapping_mul(0x9E3779B1);
        let dist_mm = (h % 60_000) as u32; // 0..60 m
        let col = j % RETINA_COLS;
        // Força parede nas colunas centrais em ~1/4 dos pixels dessa faixa, perto
        // (< 20 m), para que min_wall_ahead tenha um valor real para comparar.
        let material: u32 = if col >= RETINA_COLS / 3 && col < 2 * RETINA_COLS / 3 && (h % 4) == 0 {
            2
        } else {
            h % 4
        };
        let dist_mm = if material == 2 { dist_mm % 20_000 } else { dist_mm };
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
        let tanh_h = if abs_h > 10.0 { hi.signum() * 0.999 } else { hi / (1.0 + abs_h) };
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
    // ACHADO (pré-existente, fora do escopo desta migração — ver relatório): o layout
    // de `output` é `WAVE_U32S=48` words POR SER, mas write_channel(being, 192+i, ...)
    // usa `ch/4` sem levar isso em conta — o índice de palavra fica `being*48 + 48..51`,
    // que é o INÍCIO do bloco do PRÓXIMO ser, não um espaço reservado deste ser. Ou
    // seja: ch192-207 hoje ALIASA para o canal `i` (0-15, FAR/NEAR pool) do ser
    // SEGUINTE — já acontecia em `perception_temporal.wgsl` antes desta migração,
    // `retina_digest.wgsl` reproduz o MESMO endereçamento (mesma fórmula de
    // `write_channel`) porque o mandato desta tarefa é preservar o comportamento
    // atual bit a bit, não consertar um bug de layout descoberto de passagem.
    // Alocamos `CAP_PAD` seres a mais só para o overflow do ÚLTIMO ser usado no teste
    // cair numa área não lida (mesmo truque que `PerceptionGpu::new()` já faz com
    // `capacity.max(64)` em produção) — SEM isso o índice do último ser sairia do
    // buffer alocado neste teste pequeno (N_BEINGS=2) e o comportamento de
    // clamping de bounds-check do driver seria implementation-defined.
    const CAP_PAD: usize = 2;
    let bytes_output = ((N_BEINGS + CAP_PAD) * WAVE_U32S * 4) as u64;
    let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-output"),
        size: bytes_output,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-output-readback"),
        size: bytes_output,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Oráculo CPU: mesma seed fixa que RetinaDigestGpu::new() usa internamente para
    // subir os pesos — os dois lados partem dos MESMOS pesos de CfC.
    let mut cpu_percept: Vec<RetinaPerception> = (0..N_BEINGS).map(|_| RetinaPerception::new()).collect();

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
        let bytes_extra =
            digest.bind_and_dispatch(&mut enc, &device, &queue, N_BEINGS, &buf_retina, &buf_output);
        enc.copy_buffer_to_buffer(&buf_output, 0, &buf_output_readback, 0, bytes_output);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // ── Lê canais 192-207 ──────────────────────────────────────────────────
        let slice = buf_output_readback.slice(0..bytes_output);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&data);
        for being in 0..N_BEINGS {
            // ch192-207 de `being` ALIASA para o canal i (0-15) de `being+1` — ver o
            // comentário grande acima de `bytes_output`. Lemos daí, não do bloco de
            // `being`, porque é ONDE o byte realmente aparece (mesmo endereçamento
            // que já existia em `perception_temporal.wgsl`).
            let base = (being + 1) * WAVE_U32S;
            for ci in 0..16usize {
                let ch = ci; // aliasado: canal i do próximo ser, não 192+i deste ser
                let word = words[base + ch / 4];
                let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
                let got = byte as i8;
                let want = expected_channels[being][ci];
                assert!(
                    (got as i32 - want as i32).abs() <= 1,
                    "tick={tick} being={being} ch192+{ci} (aliasado p/ ch{ci} do ser {}): GPU={got} CPU={want} (h_old={:?})",
                    being + 1,
                    cpu_percept[being].pipeline.state.h
                );
            }
        }
        drop(data);
        buf_output_readback.unmap();

        // ── Lê extra_out (min_wall_ahead, edge_row6, edge_row7) ─────────────────
        let extra = digest.read_after_submit(&device, N_BEINGS, bytes_extra);
        for being in 0..N_BEINGS {
            let (want_wall, want_e6, want_e7) = cpu_wall_and_edges(&retinas_per_being[being]);
            let [got_wall, got_e6, got_e7] = extra[being];
            // min_wall_ahead: se nenhuma parede qualificou, os dois lados usam
            // sentinelas GRANDES mas DIFERENTES (f32::MAX vs 1e9) — só comparamos
            // o valor exato quando uma parede real foi encontrada.
            if want_wall < 1.0e8 {
                assert!(
                    (got_wall - want_wall).abs() < 0.01,
                    "tick={tick} being={being} min_wall_ahead: GPU={got_wall} CPU={want_wall}"
                );
            } else {
                assert!(got_wall > 1.0e8, "tick={tick} being={being} esperava sentinela, GPU={got_wall}");
            }
            assert!(
                (got_e6 - want_e6).abs() < 0.01,
                "tick={tick} being={being} edge_row6: GPU={got_e6} CPU={want_e6}"
            );
            assert!(
                (got_e7 - want_e7).abs() < 0.01,
                "tick={tick} being={being} edge_row7: GPU={got_e7} CPU={want_e7}"
            );
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
