# GPU Roadmap — Universe (ROCm 7, gfx1031)

> Pesquisa completa no Obsidian: `Universe/PESQUISA-ROCM-RS-2026-04-24.md`
> Hardware: RX 6750 XT (Navi22 / gfx1031 / RDNA2), ROCm 7.x
> Baseline: branch `race-world`, BitLinear CPU via `faer` + `rayon`

## Tese

Não migrar pra GPU antes de medir CPU. Não copiar rocm-rs — **abstrair** via trait.
Kernel custom HIPRTC > wrapper crate (1 kernel só, controle total, sem drift vs ROCm 7).

## Baseline medido (2026-04-24, release, RX 6750 XT host CPU)

`cargo bench --bench bitlinear`:

| Bench | Tempo | Throughput |
|---|---|---|
| `fast_brain_128x64/forward` | 1.72 µs | 4.77 Gelem/s |
| `slow_brain_32x8/forward` | 93 ns | 2.75 Gelem/s |
| `swarm_80cars_tick/sequential` | 139 µs | 4.72 Gelem/s |

**Projeção: 80 cars × 60Hz = 8.3 ms/s single-thread**. Com `rayon` cross-car, bem menos.
**BitLinear não é gargalo do simulador atual.** Profiling real primeiro (`motor.rs`
physics step + retina raycast) antes de qualquer GPU spike.

### Reveredito

- Tier 1.1 (packed 2-bit): ganho marginal sobre 1.72µs. Adia.
- Tier 2 (GPU HIPRTC): não justificado enquanto swarm ≤200 e neural <10% do frame.
- Foco atual → **profile `motor.rs`**. Identificar o verdadeiro hot path.

## Tier 1 — CPU-side (esta semana, low-risk)

1. `benches/bitlinear.rs` criterion — baseline sólido ANTES de tocar código.
2. Packed ternary 2 bits/peso em buffer runtime-only. Cache 8192B → 2048B.
   Não muda DNA v3. Método `BitLinear::pack_runtime()` pós-quantize/load.
3. SIMD explícito via `pulp` (faer já depende). Transpor weights row-major output,
   dot-product i8x32.
4. Validar determinismo: output bit-exato vs baseline em 10 DNAs aleatórios.
5. Se ganho < 3× → descarta, mantém atual.

## Tier 2 — GPU spike (próxima semana, branch isolada)

Branch: `gpu-engine-spike`. **Não mexer em race-world.**

6. Kernel BitLinear via HIPRTC (runtime compile). Wave64, `__reduce_add_sync`
   (ROCm 7), nontemporal loads nos pesos.
7. Batch cross-carro: 80 inputs i8[128] como matriz → 1 kernel launch/tick
   (não 80).
8. HIP Graphs p/ amortizar overhead (60Hz × 80 cars = 4800 launches/s sem graph).
9. `hipHostMallocUncached` p/ copy de população DNA → VRAM no spawn.
10. Pinned memory + async stream: sobrepõe physics CPU com forward GPU.

## Tier 3 — arquitetura (longo prazo)

11. CfC/LNN no slow_brain via CubeCL (autodiff, multi-backend).
12. Retina raycast (68 rays × 80 cars) em wgpu compute dispatch.
13. Night Dojo offline: consolidação de `pain_memories` em weight priors via GPU.

## Trait de abstração

```rust
// src/engine/brain.rs (nova)
pub trait BrainForward {
    fn forward_i8(&self, input: &[i8]) -> Vec<f32>;
    fn myelinate(&mut self, factor: f32);
    fn quantize(&mut self, latent: &[f32]);
}

impl BrainForward for BitLinear { /* atual */ }
pub struct BitLinearPacked;      // Tier 1
#[cfg(feature = "gpu")]
pub struct BitLinearGpu;         // Tier 2
```

`SovereignEngine::instinct_layer: Box<dyn BrainForward>`. DNA v3 idêntico.

## Riscos

- **Determinismo quebra com GPU**: reduction tree ≠ CPU acc ordem. CPU continua oracle p/ replay; GPU só treino.
- **ROCm 7 vs rocm-rs (target 6.3)**: forkar é dívida. Ir direto em `cubecl-hip-sys` ou HIPRTC raw.
- **gfx1031 ≠ MI350**: sem FP4/FP6/FP8 MX. Ficar em INT8 ternário (`dp4a` / `__builtin_amdgcn_sdot4`).
- **Sem HIP Graphs = inviável**: 4800 launches/s × ~1.5μs = 7ms/s puro overhead.

## Checklist (ações pendentes)

- [ ] `benches/bitlinear.rs` com criterion
- [ ] Feature flag `fast_forward` em `Cargo.toml`
- [ ] Trait `BrainForward` esqueleto em `src/engine/brain.rs`
- [ ] `BitLinearPacked` impl + tests de equivalência
- [ ] Branch `gpu-engine-spike`, PoC HIPRTC isolado
- [ ] Comparativo `forward/s`: baseline / Tier1 / Tier2
