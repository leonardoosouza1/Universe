# Removidos da árvore ativa

Nada foi apagado. Cada item aqui saiu de `src/` com a evidência que justifica, e volta com
um `mv` se a evidência estiver errada.

## Binários

### `bin/texto_dna.rs` (47 linhas, última alteração 2026-05-02)
Sem nenhum chamador. `grep -rl texto_dna` no projeto inteiro devolve **só o próprio
arquivo**. A `ARQUITETURA.md` já registrava: *"Caller em produção: NINGUÉM exceto
texto_dna.rs"*. Usava o caminho de módulo legado `evolution::dna` enquanto todo o resto
migrou para `dna_genesis`.

### `bin/gpu_check.rs` (132 linhas)
Supersedido por `gpu_validate` (1504 linhas). Mesmo propósito — provar equivalência da
retina em GPU — mas `gpu_validate` cobre os cinco kernels, inclusive a retina. `gpu_check`
não estava declarado em `[[bin]]`, não é citado em nenhum script, nenhum doc e nenhum
código.

## Avaliados e MANTIDOS (a auditoria errou; registrado para não repetir a análise)

- **`lab.rs` / `universe-lab`** — um agente classificou como morto alegando que a flag
  `--lab-params` não existiria. Ela existe e está implementada em `src/main.rs` (5 sítios:
  parsing do argumento, leitura do JSON, `experiment_name`, `via_ablada`). É ferramenta viva.
- **`gpu_loop.rs`** — classificado como supersedido por `gpu_evolve`/`gpu_report`, com
  ressalva do próprio auditor de que pode ser a FONTE canônica da lógica de tick que os
  outros copiam. Mantido: remover a origem de uma cópia é pior que ter a cópia.
- **`cortex_evolve` / `cortex_test` / `cortex_map`** — linha de pesquisa da folha cortical,
  independente do `SovereignEngine`. Diferente não é morto.

---

## Código morto removido de dentro de arquivos vivos

| onde | o que era | evidência |
|---|---|---|
| `motor.rs` | `sensory_compact[32]` | 10 canais preenchidos a cada tick, **zero leituras** no projeto inteiro |
| `motor.rs` | cálculo de dano de batida | 10 linhas terminando em `// racer.hp -= dmg;` comentado desde a v7 |
| `pipeline.rs` | `K_SMOOTH`, `K_SPEED`, `K_NOVEL`, `K_STAB`, `K_PHI`, `K_AESTH` | cada constante aparecia UMA vez: a própria definição |
| `Cargo.toml` | 13 blocos `[[bin]]` | redundantes — o Cargo descobre `src/bin/*.rs` sozinho |

## Bugs encontrados durante a limpeza (não eram "só arrumação")

1. **`gpu_evolve.rs:665` — quarta cópia da codificação da retina, DIVERGIDA.** Parede em
   rampa linear onde `world.rs`, `gpu/mod.rs` e `perception.wgsl` usam raiz. A 15 m dava 91
   contra 76: o CfC de percepção recebia a parede mais LONGE do que ela está, justo no curto
   alcance que decide se o ser bate. Agora chama `gpu::legacy_encoding`.
2. **`gpu_validate.rs:569` — o validador tinha o gabarito velho.** Usava a fadiga de saída
   assimétrica, substituída horas antes em `engine.rs` e `brain.wgsl`. Um validador com
   referência desatualizada é pior que nenhum: aprova a divergência que deveria pegar.
3. **`layers.rs:86` — quantização truncava para zero.** `(w*63.5) as i8` encolhe toda
   magnitude (0,5 → 0,48819). Num laço que salva e recarrega o campeão a cada geração, é
   decaimento acumulado, não ruído simétrico. Agora arredonda.
4. **Testes legados de DNA montavam arquivo com a constante errada.** Usavam
   `DNA_FAST_SIZE` (21408, cresceu com a L1) para escrever formatos v2/v3, definidos com
   8192. Falhavam por TAMANHO. O leitor sempre esteve certo.
5. **Testes do Sentinel com o contrato antigo.** O limiar de derrapagem subiu de propósito
   (0,4→0,65) e os testes não acompanharam. Além de corrigir, ficou um teste novo fixando a
   fronteira da permissão de derrapagem, que não tinha cobertura nenhuma.

## Estado da suíte

Antes: **5 falhas permanentes** (as mesmas havia meses). Depois: **157 passam, 0 falham.**
Uma suíte vermelha é uma suíte que ninguém lê — foi por isso que a fadiga que eu mesmo
adicionei nesta sessão quebrou dois contratos do CfC sem ninguém notar.
