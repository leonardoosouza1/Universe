---
name: experimento-universe
description: Protocolo obrigatório antes de rodar ou reportar qualquer bateria evolutiva no Universe. Use SEMPRE que for medir efeito de uma mudança no cérebro, na fitness, no corpo ou no ambiente — e antes de anunciar qualquer resultado numérico sobre comportamento dos seres.
---

# Protocolo de experimento — Universe

Esta skill existe porque cada passo abaixo corresponde a um erro que **já foi cometido** neste
projeto e custou horas de GPU e conclusões invertidas. Os detalhes estão em `CLAUDE.md`
(seção "Gaps recorrentes"). Aqui está a ordem executável.

Se o pedido for "meça se X ajuda", não pular nenhum passo. Se algum passo for pulado
deliberadamente, **escrever no relatório qual e por quê**.

## 1. Teto da métrica, ANTES de rodar

Calcular o valor máximo que a métrica pode atingir com aquela configuração e escrever o
número no relatório.

```
distância máxima possível = min(
    ticks / 64 × velocidade_alcançável,
    vida_disponível / 64 × velocidade_alcançável
)
equilíbrio de vida (m/s) = 64 / life_earn      # abaixo disso o ser sangra vida
```

Uma volta são 2928 m. A 15 m/s isso pede ~12.500 ticks **de vida**, não de simulação.

*Erro real:* `--ticks 1200` produziu "muro em 256 m" que era o teto de 290 m dos próprios
ticks. Zero ou 100% cravado = métrica no limite, não modelo falhando.

## 2. Divergência antes da bateria

Rodar 2–3 gerações nas duas condições com a **mesma semente**. Os números têm de ser
**diferentes**. Se forem idênticos, o mecanismo não chegou ao kernel — parar e achar por quê.

*Erro real:* números idênticos apareceram 8+ vezes nesta sessão e **nunca** foram
coincidência (binário velho, caminho de código diferente, edição que não aplicou, mudança só
de exibição sem tocar a seleção, stride errado).

## 3. Normalização: o que mais muda junto?

Ao tornar um parâmetro heterogêneo ou variá-lo, verificar analiticamente o que mais se move
com ele e neutralizar. Escrever a conta no comentário do código.

*Erro quase cometido:* dar τ por canal sem normalizar o ganho DC faria o canal rápido sair
**80× mais fraco** — a evolução o descartaria por ser mudo, não por ser rápido.

## 4. Bateria pareada

- **≥3 sementes**, mesma semente nos dois braços, `--same-start`
- **≥30 gerações**
- Rodar em paralelo, no máximo 4 processos concorrentes na GPU
- Sempre `nice -n 19`, sempre `env -u DISPLAY`, sempre `HSA_OVERRIDE_GFX_VERSION=10.3.0`

```bash
C="--seres 48 --gens 30 --ticks 12000 --fitness centro --width 28 --same-start --life-cap 6000"
env -u DISPLAY ./target/release/gpu_evolve $C --seed $S <FLAG> --out "$D" > "$D/run.log" 2>&1
```

## 5. Estatística

- **Média das últimas N gerações** (N=5). NUNCA o melhor de todas.
- Seguir o **tamanho do efeito** conforme n cresce — não σ. Duas condições independentes com
  a mesma inclinação valem mais que um σ alto numa só.
- Reportar quantas sementes concordam no **sentido** (3/4, 1/4…).
- Média de razões ≠ razão de médias. Para grandezas com denominador que quase zera
  (obediência à curva, eficiência), usar **quociente de somas**.

*Erro real:* média de `ω_real/ω_exigido` por quadro deu +1,14; a projeção agregada correta
deu +0,045. Fator 25 de diferença, e o sinal da conclusão mudou.

## 6. Sobre QUEM é a estatística

Declarar explicitamente. Excluir **mortos** de qualquer média de comportamento — o simulador
trava `throttle=0, freio=1, volante=0` em quem morre (`gpu_evolve.rs:843`).

*Erro real:* "eles rastejam a 1,7 m/s" repetido por vinte mensagens. Vivos, faziam 79 km/h;
a média estava diluída por ~2.000 ticks de cadáver.

## 7. Verificação visual — TODAS as gerações

```bash
./target/release/replay_mapa --dir <pasta da corrida> --width 28 --saida /tmp/replay_srv
# depois: http://localhost:8127/mapa.html  (servidor "replay" do .claude/launch.json)
```

Olhar **no mínimo** a primeira, o meio e a última geração. O viewer abre na **geração 0**.

*Erro real:* medi a geração 0 e reportei como estado final. A conclusão era o inverso: gen 0
tinha giro aleatório (sentido certo 55% = acaso), gen 29 acertava o lado em 83%.

## 8. Relatório

Todo número acompanhado do comando que o produz. Toda afirmação sobre comportamento
acompanhada da medida. Sem medida, escrever **"não medi"** — nunca a descrição.

Antes de anunciar causa, testar se o efeito é grande o bastante para importar: a fadiga de
saída era real (dose-resposta monotônica, 3/4 sementes) e valia **5 m em 266** — 2%. Foi
apresentada como "o freio" e não era.

## Ferramentas do repositório

| ferramenta | responde |
|---|---|
| `replay_mapa --dir` | todas as gerações, população, mortes, curvatura, ω exigido × real |
| `replay_obs` | 23 métricas por ser, agrupamento por estilo, onde cada um parou |
| `replay_diag` | um ser, quadro a quadro, contra a geometria |
| `scripts/watchdog.sh` | corrida travada (pergunta se o ARQUIVO cresceu, não se o processo vive) |
| `cargo test --release --bin gpu_evolve` | ida-e-volta do genoma no disco |
