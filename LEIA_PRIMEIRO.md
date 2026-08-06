# Universe · race-world-v3 — o que foi medido, consertado e construído

Sessão de 2026-07-30/31. Ponto de partida: *"como fica a alocação de neurônios, onde salva X
neurônios, o que eles fazem"*. O que apareceu no caminho foi mais interessante que a pergunta.

---

## Comece por aqui

```bash
cargo build --release

./target/release/pistas                    # dificuldade REAL de cada pista
./target/release/pistas --curriculo        # gera uma rampa de dificuldade medida
./target/release/dump_dna CHAMPION.dna     # o que tem dentro do .dna
./target/release/sinapses CHAMPION.dna --mapa      # onde o cérebro escuta cada sentido
./target/release/sinapses CHAMPION.dna --priming   # o que o humano escreveu vs o que há

scripts/rodar.sh <rótulo> --headless --gens 3      # roda ISOLADO (nunca rode o binário direto)
scripts/rodar_tudo.sh                              # as quatro baterias, em sequência
```

> ⚠️ **Nunca rode `./target/release/universe` da raiz.** Ele sobrescreve `CHAMPION.dna` no
> diretório de trabalho, e um teste de 1 geração apaga o campeão treinado. Aconteceu duas
> vezes nesta sessão. Use `scripts/rodar.sh`, que isola em `/tmp`. Há cópia em `dna_backup/`.

---

## O resultado principal

**Alocação medida por ablação, 5 repetições, campeão treinado:**

| via ablada | net_max | custo | uso do portão (cx/rf/sn) |
|---|---|---|---|
| base | 3198 ± 25 | — | 0,370 0,342 0,288 |
| **córtex** | 170 ± 82 | **94,7 ± 2,6%** | 0,363 0,348 0,290 |
| reflexo | 3061 ± 23 | 4,3 ± 1,0% | 0,378 0,351 0,271 |
| lenta (modulador) | 3112 ± 21 | 2,7 ± 1,0% | 0,362 0,355 0,283 |
| marcha | 3177 ± 18 | 0,7 ± 0,9% | 0,376 0,351 0,273 |
| sentinela (suave) | 3140 ± 17 | 1,8 ± 0,9% | 0,372 0,347 0,281 |

> O portão dá **37%** ao córtex e **34%** ao reflexo. O córtex carrega **94,7%**, o reflexo
> **4,3%**. **Descolamento de 22×** entre o que o portão diz e o que importa.

E o peso do portão mal se move mesmo quando a via está desligada — ele segue dando peso a
quem não está dizendo nada. **Nenhuma estatística de peso, atenção ou portão diz onde o
cérebro guarda o que importa. Só ablação.**

O mapa de pesos confirma pelo outro lado: peso por canal 9,5–10,7 nos seis grupos
sensoriais, e a fatia da visão (35,4%) é exatamente sua fração de canais (80/223).
**Alocação indistinguível de uniforme.**

---

## Três bugs encontrados medindo

**1. O currículo destruía a população.** Dois mecanismos independentes — clima e largura de
pista — avançavam pela distância do **campeão**. Um carro que fez 1,93 voltas no seco levava
os 60 ao nível 3, onde a aderência cai a 0,33, a frenagem exigida vai a **130,9 m** e o
sensor de curvatura mais longo do carro alcança **120 m**. Cegueira física, não dificuldade.

Trajetória antes: `34% → 10% → 1%` de voltas, mortalidade `81% → 93% → 100%`.
Consertado: avança pela **fração da população** (≥30% sobe, <5% **recua**), com histerese e
no máximo um degrau por geração. 11 testes.

**2. Reabastecimento que matava.** A cada 3ª volta, `racer.fuel = 1.0` — num tanque cuja
escala de nascimento era **1000**. Despejo de 1000× disfarçado de tanque cheio, deixando
~166 s de vida. Explica `3+v=0` em todas as gerações medidas: quem era bom o bastante para
chegar na 3ª volta era sentenciado por isso.

**3. Colisão de canais sensoriais.** Três escritores gravam nos mesmos canais no mesmo frame
(`motor.rs` 561, 579, 676, 1413) e o último vence. **Metade do leque de sensores de parede é
calculada e jogada fora todo frame**, e o comentário de layout descreve o que se pretendia,
não o que acontece. Corrigido no dicionário de `neuro.rs`; consertar na fonte recupera 8
canais reais mas muda índices e obriga a retreinar.

---

## O que a física diz sobre dificuldade

Não é largura: alargar de 8 para 24 m muda a distância em 27% e a **mortalidade** de 58/60
para 33/60 — largura compra margem de erro, não velocidade.

É a **distância de frenagem**, que decide qual sonda de curvatura ainda chega a tempo
(o carro tem quatro: 20, 40, 70 e 120 m):

| teto de velocidade | exige a de 40 m | a de 70 m | a de 120 m |
|---|---|---|---|
| 30 m/s | **impossível** | impossível | impossível |
| 40 m/s | curva até 76 m | até 16 m | impossível |
| 45 m/s | até 119 m | até 59 m | impossível |

**Abaixo de ~35 m/s nenhuma geometria consegue exigir sequer a sonda de 40 m** — as sondas
longas do carro são peso morto. A 2ª marcha chega a 33 m/s, **a 3ª a 45 m/s**:

> Marcha não acrescenta percepção. Ela **ATIVA a percepção que o carro já tem e não consegue
> usar.**

E os rótulos de dificuldade das pistas estavam invertidos: "Nürburgring EXTREME" tem raio
mínimo 147 m e demanda **zero** (nem precisa frear); "Silverstone VERY HARD" exige 16 m. O
campo `complexity` do `TrackProfile` **nunca é lido por ninguém**.

---

## Documentos

| arquivo | o que traz |
|---|---|
| [ALOCACAO_NEURAL.md](ALOCACAO_NEURAL.md) | a bateria de ablação, o que cada sinapse diz, as armadilhas do formato |
| [PROJETO_DE_MUNDO.md](PROJETO_DE_MUNDO.md) | como acrescentar corpo e ambiente sem quebrar o experimento |
| `src/neuro.rs` | vias, sentidos, priming, modos de nascimento — com o porquê de cada um |
| `src/pista_gen.rs` | geração de pista por dificuldade **medida**, com amostragem por rejeição |

## Duas leis de projeto, medidas

> **Toda ação protetora precisa custar numa moeda DIFERENTE daquela contra a qual protege.**
> Senão uma política degenerada (sempre X, nunca X) domina e os sensores viram decoração.

> **O ambiente nunca pode exigir mais antecipação do que os sensores do corpo alcançam.**
> Além disso não é dificuldade, é cegueira — e ali nada se aprende.
