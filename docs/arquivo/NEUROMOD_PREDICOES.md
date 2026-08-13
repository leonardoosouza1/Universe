# NEUROMOD_PREDICOES — Predições testáveis para o Universe

**Gerado em:** 2026-08-03  
**Fonte:** `neuromod/docs/01` a `08`, `SPEC.md`, `README.md`  
**Contexto do Universe:** seres com cérebro neural numa pista, tick rodando na GPU (5 kernels). Folha cortical em `src/gpu/cortex_sheet.rs`: 1024 unidades, aferência gaussiana sem rótulo de região, Hebb com normalização subtrativa, banco de τ log-espaçado de 1,6 ms a 318 ms, ondas espontâneas.

---

## O que já está respondido (não re-testar)

Os docs 06 e 07 são projetos de experimento, **ainda não executados**. Mas o ecossistema ao redor do Universe já mediu vários pontos cobertos pelos documentos:

| Achado | Onde medido | Status |
|---|---|---|
| U invertido do ganho (β): pico em β≈1, β=8 custa −90% | exp06 (neuromod) | ✅ medido |
| Modulação multiplicativa bate aditiva 16/16 | Experimentos Universe (memória project_neuromod_gatilho) | ✅ medido |
| Ganho obedece U invertido com pico 0,5–1,0 → +13 a +16% | Universe | ✅ medido |
| Gatilho CEGO custa −21,43% | Universe | ✅ medido |
| Escalas de tempo variadas são selecionadas: 40× vs 5,3× | Universe (project_banco_de_tau) | ✅ medido |
| TROCAR τ com o regime não paga (+0,11%) | Universe (project_banco_de_tau) | ✅ medido |
| Teto G calculável antes de simular | exp08 (neuromod) | ✅ analítico |
| Memória é inútil em MDP totalmente observável | exp03 (neuromod) + teorema | ✅ provado |
| Tese do corpo: teto G só é alto com estados internos lentos acoplados; corr(ganho, teto) = +0,970 | Universe (project_tese_corpo_confirmada) | ✅ medido |

**Esses pontos não precisam de novos experimentos.** O que está em aberto são as predições abaixo.

---

## Predições — ordenadas: mais testável e mais barato primeiro

---

### P1 — Fronteira visual/auditiva emerge onde a correlação cruzada cai

**O que o documento afirma**

> Doc 08, §4: "A fronteira entre 'visual' e 'auditivo' não é desenhada. Ela é onde a correlação cruzada cai." A regra hebbiana com normalização subtrativa (`Δw_ik = η·(a_i·a_k − α·w_ik·Σ_j a_j·a_k)`) faz entradas que disparam juntas ficarem juntas e descorrelacionadas se separarem. Isso reproduz, sem instrução explícita, colunas de dominância ocular, mapas retinotópicos e fronteiras entre áreas (doc 08, §4).

**A predição operacional no Universe**

A folha cortical (`cortex_sheet.rs`) já tem aferência gaussiana sem rótulo de região e Hebb com normalização subtrativa. Os 223 canais de entrada (`N_INPUTS = 223`) contêm sinais de modalidades diferentes (retina, propriocepção, velocidade, etc.). Após suficiente aprendizado hebbiano:

- As unidades da folha devem **se dividir em territórios** correspondentes às modalidades dominantes da entrada.
- A fronteira entre dois territórios (ex.: unidades dominadas por retina vs. unidades dominadas por velocidade/propriocepção) deve coincidir com a região onde a correlação cruzada entre os pesos de aferência das duas populações cai abruptamente.
- O teste específico: computar a matriz de correlação entre os vetores de peso `w_{i,*}` e `w_{j,*}` (cada linha = perfil de aferência de uma unidade). A correlação intra-território deve ser alta; cruzando a fronteira, deve cair em degrau. A fronteira emerge, não é hardcoded.

**Como medir**

1. Após N_ticks de simulação (por exemplo, 5.000 ticks com pelo menos 30 seres), fazer readback do buffer `buf_afw` (pesos hebbians): `f32[cap × 1024 × 223]`.
2. Para cada ser, calcular a matriz de correlação 1024×1024 dos vetores de peso (cada coluna `w_{i,:}` tem 223 entradas).
3. Aplicar clustering hierárquico nos vetores de peso. Plotar a correlação cruzada média entre clusters contra a distância espacial entre unidades na folha 32×32.
4. **Predição numérica:** a correlação intra-cluster deve exceder 0,6; a correlação cruzada entre clusters de modalidades distintas deve ficar abaixo de 0,2. A fronteira deve ser visível como queda abrupta no mapa 32×32.
5. Controle: embaralhar os rótulos dos canais de entrada antes de treinar. Com entrada completamente randômica, nenhum cluster deve emergir (correlação uniforme).

Arquivo a criar: `tools/cortex_map.py` — lê o binário de `buf_afw`, computa correlação, plota heatmap 32×32 colorido por cluster dominante.

**O que refutaria**

- Correlação cruzada entre territórios tão alta quanto intra-território → a folha não segrega, Hebb sozinho é insuficiente nesta escala de tempo.
- Clusters emergem mas **não correspondem** às modalidades de entrada → a organização é por vizinhança aleatória, não por estrutura de correlação.
- Embaralhar a entrada produz clusters idênticos → os clusters refletem a inicialização, não o aprendizado.

---

### P2 — Banco de τ deve cobrir a faixa prevista pela fórmula G

**O que o documento afirma**

> Doc 06, §3 (fórmula do ganho G): para um mundo com regime oculto de contraste forte (ruído alternando 100×), os τ ótimos por regime são τ* ∈ [1,1 · 4,2] (razão 3,8×). Doc 06, §5.3 calibra o acoplamento a partir dessa faixa: `a₁·m_máx = 3,8 − 1 ≈ 2,8`.
>
> Doc 05, §6 (exp05): evolução de N=2 neurônios reinventou o par SA/RA (τ=1,5 e τ=23,4, spread 16×) sem instrução. O banco log-espaçado que emerge sozinho é a mesma estrutura do banco de τ do doc 01 §3 e das bandas de frequência do doc 08 §5.3.
>
> Doc 08, §5.3: razão média entre bandas vizinhas ≈ 2,3×, espaçamento logarítmico. "Esta é a mesma estrutura que emergiu sozinha no exp05."

**A predição operacional no Universe**

O banco de τ da `CortexSheet` já está fixo e log-espaçado de 1,6 ms a 318 ms (razão total 200×). O que precisa ser testado é se a **seleção** favorece seres cujo banco de τ ativo (ou seja, cujas unidades têm maior atividade média) cobre uma faixa específica relacionada ao ritmo do ambiente.

A predição: ao analisar quais unidades ficam ativas durante a navegação (ativação média acima de um limiar ao longo de muitos ticks), os τ dessas unidades ativas devem ter **spread maior** que o τ de unidades silenciosas. Especificamente, o spread dos τ ativos deve superar 10× (valor observado no Universe, 40× nos campeões, contra 5,3× de um controle aleatório).

**Como medir**

1. Após simulação estável, fazer readback de `buf_act` a cada 100 ticks por pelo menos 500 ticks.
2. Calcular a ativação média temporal de cada uma das 1024 unidades.
3. Separar unidades em "ativas" (ativação média > 0,05) e "silenciosas" (< 0,01).
4. Calcular `spread_ativo = τ_max_ativo / τ_min_ativo` e `spread_silencioso = τ_max_silencioso / τ_min_silencioso`.
5. **Predição numérica:** `spread_ativo > 10×` e `spread_ativo >> spread_silencioso`. O banco ativo deve ser aproximadamente log-espaçado, não concentrado numa faixa.

Arquivo: adicionar ao método `read_act_stats` ou criar `tools/tau_spectrum.py`.

**O que refutaria**

- Unidades ativas têm τ concentrado numa faixa estreita (ex.: todas < 10 ms) → a folha usa só uma escala de tempo, estrutura de filtros não está sendo explorada.
- Spread das unidades ativas é igual ao das silenciosas → a atividade é uniformemente distribuída pelos τ, sem hierarquia.
- Spread cresce mas não correlaciona com desempenho individual dos seres → a diversidade de τ não tem papel funcional.

---

### P3 — Pressão trófica auto-dimensiona a folha sem especificar N

**O que o documento afirma**

> Doc 08, §6: "Aproximadamente 50% dos neurônios gerados morrem. A sobrevivência depende de competição por fatores tróficos limitados produzidos pelo alvo: `quota_i = trófico_total · atividade_i / Σ_j atividade_j`". Com trófico **fixo**, a poda não se adapta ao ambiente (variação de 68,6% a 71,9% de sobrevivência, praticamente idêntica entre ambientes pobres e ricos). Com trófico **proporcional à demanda do alvo**, a população se ajusta: 6,8% em ambiente pobre, 79,8% em ambiente rico (doc 08, §6.1). "A população se ajusta ao tamanho do problema sem você especificar nada."
>
> Doc 08, predição D5: "Com trófico fixo, a população não se adapta à demanda; com trófico proporcional, adapta." D5 já tem evidência numérica do doc 08, mas não foi validada na folha do Universe.
>
> Doc 02, §4.3 (princípio pressão/tensão): "Pressão só produz estrutura quando existe tensão — uma força oposta que torna custosa a perda daquilo que está sendo penalizado. Pressão sem tensão produz colapso, não organização." (Validado nos exp00 vs exp02.)

**A predição operacional no Universe**

A folha atual (`CortexSheet`) tem 1024 unidades fixas por ser, sem poda. A predição é: se introduzir um mecanismo de poda onde unidades com atividade abaixo de um limiar "morrem" (seus pesos são zerados e não são atualizados), e o limiar for proporcional à atividade total da folha (poda competitiva), então:

- O número efetivo de unidades ativas deve convergir para um valor proporcional à **riqueza do ambiente sensorial**, não à capacidade inicial.
- Em pista simples (pouco estímulo variado): poucas unidades ativas, concentradas em poucos τ.
- Em pista complexa (estímulo variado, curvas, obstáculos): mais unidades ativas, spread de τ maior.

**Como medir**

1. Condição A: pista simples (reta longa, pouco estímulo sensorial variado).
2. Condição B: pista complexa (curvas, variação de velocidade, estímulos diversos).
3. Em ambas, contar o número de unidades com ativação média > 0,05 após 2.000 ticks.
4. **Predição numérica:** condição B deve ter pelo menos 2× mais unidades ativas que condição A.
5. Controle de poda fixa: com limiar absoluto (não competitivo), a diferença entre condições deve ser menor.

**O que refutaria**

- Número de unidades ativas é idêntico em ambas as condições → a poda competitiva não está captando a riqueza sensorial.
- Poda sem tensão (poda com limiar absoluto baixo) colapsa todas as unidades para o mínimo → confirma o princípio pressão/tensão, mas de forma negativa.
- Número de unidades ativas não correlaciona com desempenho na pista → a poda não tem papel funcional.

---

### P4 — Neuromodulação só paga quando há regime oculto de contraste alto

**O que o documento afirma**

> Doc 06, §2.1 (teorema): "Com m constante, a célula modulada é apenas outro LTC com parâmetros diferentes. Reparametrização pura, zero ganho expressivo. Neuromodulação só adiciona poder quando m **varia**, e especificamente quando os parâmetros ótimos (τ*, β*) dependem de um **contexto oculto** inferível apenas pela história."
>
> Doc 06, §3.3 (valores de G calculados): estacionário G=0,0%, ruído 2× G=1,7%, ruído 10× G=12,5%, ruído 100× G=25,6%. Quando ruído E persistência mudam juntos, G=1,2% (os efeitos se cancelam). "Um mundo 'mais complexo' pode exigir menos modulação."
>
> Doc 07, §4: para os regimes do corpo mínimo — predador ativo vs ausente (G=25,6%), dia/noite (G=19,1%), estação de comida (G=12,9%), dano cicatrizando vs infeccionando (G=9,4%), mundo estacionário (G=0,0%).
>
> Doc 03, §3: canal `gain` (análogo de noradrenalina) encurta τ e aumenta β. Medido no Universe: modulação multiplicativa bate aditiva 16/16; ganho com U invertido; gatilho CEGO custa −21,43%.

**A predição operacional no Universe**

A pista do Universe já tem regime alternante (luz do dia, velocidade dos seres, presença de outros). A predição é:

- A neuromodulação (canal `gain`) só agrega performance mensurável se o ambiente tiver **contraste de regime alto** (pelo menos 10× de diferença na qualidade do sinal sensorial entre regimes).
- Em ambiente homogêneo (pista constante, sem perturbações), modular não deve agregar mais que 2% sobre um neurônio com τ fixo ótimo — e possivelmente piorar, pois o gatilho pode capturar ruído.
- O ganho medido deve ficar abaixo do G previsto pela fórmula `(J_fixo − J_adapt)/J_fixo`. Com contraste de 10× no Universe, G teórico ≈ 12,5%.

**Como medir**

1. Calcular G analiticamente para o ambiente da pista atual: medir a variância do sinal sensorial em dois regimes representativos (ex.: reta vs. curva fechada) e aplicar a fórmula `P(K)` do doc 06 §3.1.
2. Comparar fitness de seres com modulação ativa vs. ablados (m≡0) na mesma pista.
3. **Predição numérica:** ganho medido deve ser menor que G calculado (teto teórico) e maior que 0 se o contraste for alto. Se o contraste for baixo (regime quase estacionário), ganho ≤ 2%.
4. Controle: pista completamente aleatória onde nenhuma memória ajuda → modulação deve empatar ou perder.

**O que refutaria**

- Modulação ganha >G calculado → a fórmula do §3.1 está errada (teto teórico violado); investigar.
- Modulação ganha em ambiente estacionário (G=0%) → o modulador é uma reparametrização disfarçada (teorema §2.1).
- Ganho com gatilho cego (modulação aleatória) similar ao gatilho interoceptivo → o gatilho não está inferindo o regime.

---

### P5 — Período crítico: ACh alta deixa o mapa maleável, queda de ACh o trava

**O que o documento afirma**

> Doc 08, §8: "O período crítico não é um relógio separado. É plasticidade modulada: `η(t) = η₀ · m_ACh(t) · f(maturação_inibitória)`. ACh alta (novidade, atenção) → plasticidade alta → mapa maleável. Inibição madura + ACh baixa → plasticidade baixa → mapa consolidado. Isto fornece o botão de 'quando travar'. Sem fechamento, o mapa nunca estabiliza e o sistema esquece continuamente."
>
> Doc 03, canal `sens_rec` (acetilcolina): controla "razão entrada sensorial ÷ recorrência — confiar no mundo vs na memória". τ_limpeza = 60 passos, τ_repor = 600 passos (SPEC, §3).
>
> Doc 08, §5.4: "Acetilcolina empurra para gamma [bandas rápidas]; estados de sono para delta. O canal sens_rec não muda só a razão entrada/recorrência — muda em qual escala de tempo a rede opera."

**A predição operacional no Universe**

A folha cortical atual tem taxa de aprendizado Hebb fixa. A predição é: se introduzir modulação da taxa Hebb pelo nível de "novidade" do ambiente (análogo de ACh — alta quando há eventos inesperados, baixa quando o ambiente é repetitivo):

- O mapa deve se organizar **mais rapidamente** nas primeiras fases (novidade alta → η alto).
- Após convergência, **congelar o mapa** (η → 0 quando novidade cai abaixo de um limiar) deve **preservar o desempenho** em vez de degradá-lo com esquecimento contínuo.
- A diferença de desempenho entre "mapa congelado após aprendizado" e "mapa plasticidade constante" deve ser positiva após 5.000 ticks (o segundo sofre interferência contínua).

**Como medir**

1. Treinar duas condições por 10.000 ticks:
   - Condição A: η fixo (Hebb contínuo, como está agora).
   - Condição B: η proporcional à surpresa média da folha (variância das ativações entre ticks consecutivos), com limiar de congelamento quando a variância cai abaixo de 0,001.
2. Medir:
   - Taxa de organização: quantos ticks até a correlação intra-cluster superar 0,6 (P1).
   - Estabilidade: desvio-padrão do desempenho motor nos últimos 2.000 ticks.
3. **Predição numérica:** condição B deve atingir correlação intra-cluster 0,6 em menos ticks que A, e ter desvio-padrão de desempenho menor no regime estável.

**O que refutaria**

- Congelar o mapa degrada o desempenho imediatamente → o ambiente do Universe está mudando rápido demais para consolidação ser útil.
- Η variável e η fixo produzem organização idêntica → a modulação da plasticidade não tem papel funcional na escala de tempo da simulação.
- Mapa congelado cedo tem desempenho melhor que mapa congelado tarde → o período crítico ótimo é muito curto; a folha satura antes da exploração adequada.

---

### P6 — Alometria: folhas maiores devem concentrar mais unidades ativas no "córtex" (camadas tardias / τ longos)

**O que o documento afirma**

> Doc 08, §1: expoentes alométricos (mamíferos). Córtex: b≈1,35 (hiperalométrico). Cerebelo: b≈1,00 (isométrico). Resto: b≈0,70 (hipoalométrico). "Dobrar o total multiplica o córtex por 2,55 e o resto por 1,62. A composição muda sozinha, na direção de mais córtex, sem nenhum gene específico para córtex."
>
> Doc 08, §1.2: "Córtex e cerebelo mantêm razão de ~1:4 em número de neurônios através de quase toda a evolução dos mamíferos — mesmo com o córtex ocupando ~80% da massa. Restrição de desenvolvimento profunda, útil como teste de sanidade: se um modelo produz razões muito fora disso, algo está errado."
>
> Doc 08, §2 (cronograma): "O que se forma por último é o que mais cresce." Unidades com τ longo (processamento integrativo / delta) são análogas ao córtex; τ curto (gamma, reflexo imediato) são análogas ao tronco.

**A predição operacional no Universe**

Na folha de 1024 unidades, τ longos (próximos a 318 ms) são análogos ao córtex (processamento de longa escala de tempo). τ curtos (1,6 ms) são análogos ao tronco/reflexo. Se escalar a folha (1024 → 2048 → 4096 unidades), a proporção de unidades ativas com τ longo deve crescer **mais rápido que linearmente** — expoente > 1,0 — enquanto a proporção de unidades com τ curto cresce sub-linearmente.

Teste de sanidade quantitativo concreto: na configuração atual (1024 unidades), o número de unidades ativas com τ > 100 ms (análogo de córtex) deve ser aproximadamente 4× o número de unidades ativas com τ < 10 ms (análogo de cerebelo). Essa razão ~1:4 deve se manter ao escalar.

**Como medir**

1. Com a folha de 1024 unidades, após estabilização: contar unidades com τ > 100 ms ativas e com τ < 10 ms ativas. Calcular razão.
2. (Quando/se escalar para 2048 unidades): repetir. Verificar se a proporção de unidades com τ longo cresceu mais que a de τ curto.
3. **Predição numérica (sanidade):** razão τ_longo/τ_curto ativas ≈ 4. Se razão estiver muito fora (ex.: <1 ou >16), o banco de τ atual ou o regime de entrada está mal calibrado.

**O que refutaria**

- Razão τ_longo/τ_curto ativas = 1 → sem hierarquia temporal; a folha não está usando as bandas de forma diferenciada.
- Ao escalar, a proporção de τ longos **não** cresce mais rápido → expoente alométrico ≈ 1 para ambas; não há emergência de hierarquia.
- Razão consistentemente > 16 → τ curtos estão todos silenciosos; a folha não está usando reflexo rápido, só integração lenta.

---

### P7 — Entrada de baixa frequência poda o território em vez de capturá-lo (STDP)

**O que o documento afirma**

> Doc 08, §5.2: "A plasticidade depende de frequência (corrige §4). Hebb simples é incompleto — o sinal da mudança depende da frequência: baixa frequência (~1 Hz) → depressão de longo prazo (enfraquece); alta frequência (~50–100 Hz) → potenciação (fortalece). Implicação de projeto: não basta 'entrada rica'. A estatística temporal fina determina se as sinapses fortalecem ou apagam. Entrada rica mas lenta demais pode podar o território em vez de capturá-lo."
>
> Doc 08, predição D3: "Entrada rica mas de baixa frequência poda o território em vez de capturá-lo. Refuta se: captura igual em qualquer frequência → STDP irrelevante no modelo."

**A predição operacional no Universe**

A Hebb atual da `CortexSheet` usa normalização subtrativa mas não tem dependência explícita de frequência (não é STDP completo). A predição é: se dois canais de entrada têm a mesma variância mas frequências dominantes distintas — um rápido (ticks consecutivos muito diferentes) e um lento (ticks consecutivos quase idênticos) — o canal lento deve **capturar menos território da folha** (menos unidades com peso maior nele) do que o canal rápido de mesma variância.

Concretamente nos 223 canais do Universe: canais de retina (que variam a cada tick conforme o ser move) devem dominar mais unidades que canais de propriocepção lenta (que variam em escalas de segundos), **mesmo que ambos tenham variância total similar**.

**Como medir**

1. Após 5.000 ticks, para cada canal de entrada (0 a 222), calcular:
   - A variância total ao longo do tempo.
   - A frequência dominante (ou equivalentemente: a autocorrelação a 1 tick vs. a 10 ticks).
   - O número de unidades da folha para as quais esse canal tem o maior peso de aferência.
2. Plotar: (frequência dominante do canal) × (número de unidades dominadas).
3. **Predição numérica:** correlação positiva entre frequência do canal e território capturado, mesmo controlando pela variância total. Canais com autocorrelação >0,9 a 1 tick (muito lento) devem dominar menos unidades que canais com autocorrelação <0,3 a 1 tick.

**O que refutaria**

- Canais lentos e rápidos capturam territórios idênticos controlando pela variância → STDP/dependência de frequência não é relevante na escala da simulação.
- Canais lentos dominam mais unidades → contradição com D3; investigar se o τ longo das unidades está favorecendo entradas lentas de forma não-STDP.

---

### P8 — Ondas espontâneas: removê-las degrada os mapas iniciais (D4)

**O que o documento afirma**

> Doc 08, §5.5: "A retina gera ondas de atividade correlacionada antes de o animal abrir os olhos, e elas já esculpem os mapas. Consequência de projeto: é necessário um gerador rítmico endógeno na fase de aferência, senão a segregação não tem o que correlacionar no início do desenvolvimento." Predição D4: "Remover ondas espontâneas degrada os mapas iniciais. Refuta se: mapas idênticos → a fase 2 não depende de atividade endógena."

**A predição operacional no Universe**

O kernel de aferência (`cortex_sheet.wgsl`) já menciona ondas espontâneas na documentação do `CortexSheet`. A predição é: **nos primeiros ticks de vida de um ser** (antes que a entrada sensorial real seja rica), a atividade espontânea da folha deve produzir correlações espaciais nos pesos hebbians que **pré-estruturam** a organização posterior.

Comparação: ser com ondas espontâneas ativas desde o tick 0 vs. ser cujas ondas espontâneas são zeradas nos primeiros 500 ticks. Após 2.000 ticks com entrada real normal:

- O ser com ondas espontâneas deve ter mapa mais organizado (correlação intra-cluster maior).
- O ser sem ondas espontâneas iniciais deve ter mapa mais aleatório e demorar mais para organizar.

**Como medir**

1. Criar variante do kernel de aferência que suprime as ondas espontâneas para `tick < 500`.
2. Rodar ambas as condições por 2.000 ticks com 10 seres cada.
3. Comparar `read_act_stats` (std das ativações — proxy de heterogeneidade organizada) aos ticks 500 e 2.000.
4. **Predição numérica:** std da condição com ondas espontâneas deve ser > 20% maior que a sem ondas ao tick 500; ao tick 2.000 a diferença pode diminuir mas ainda ser detectável.

**O que refutaria**

- std idêntico em ambas as condições aos ticks 500 e 2.000 → a fase de aferência no Universe não depende de atividade endógena; a entrada sensorial real desde o tick 1 é suficiente.
- Condição sem ondas espontâneas organiza mais rápido → as ondas espontâneas estão interferindo com a entrada real, não pré-estruturando.

---

## Contradições identificadas entre documentos e medições

### Contradição C1: τ_P ótimo (teoria vs medição)

**Doc 02, §6.1 e doc 06, §2.1** derivam que, para estações ocultas de duração média T, `τ_P ótimo = T/(2q) = duração_média / 2`. Com duração média de 150 passos → τ_P previsto ≈ 75.

**Medido no exp04** (doc 02, §6.2): mediana de τ_P = 2,1. Ordem de magnitude errada.

**Diagnóstico do doc**: o agente observava a energia (variável de ação, não sensor), que já codifica a informação em poucos passos, tornando o canal lento redundante.

**Implicação para o Universe**: se os seres observam sua própria velocidade e posição (variáveis de ação), a memória crônica sobre regimes de pista **não vai emergir**. Para que emerja, a variável de regime deve ser **oculta** e detectável apenas por acúmulo longo. Isso ainda não foi testado na pista.

### Contradição C2: N grande piora vs. alometria sugere mais é melhor

**Doc 05, exp05** (doc neuromod): N≥10 piorou o desempenho — capacidade sem pressão que a organize é passivo. Os τ colapsam em 1–5.

**Doc 08, §1 (alometria)**: mais neurônios → mais córtex proporcionalmente → mais inteligência (extrapolando a alometria).

**Resolução**: o doc 08 explica que escalar sem mecanismo de poda (trófico) não produz alometria — produz colapso dos τ (C2 é o mesmo fenômeno do exp05, não uma contradição real). A alometria só emerge quando o suprimento trófico escala com o uso do alvo. No Universe, adicionar unidades à folha **sem** poda competitiva deve reproduzir o colapso do exp05.

### Contradição C3: Multiplicativo sobre estado h é proibido vs. já implementado

**Doc 06, §4 (invariante I1)**: "Estado `h`: nunca tocado diretamente. Somar `m` a `h` transformaria o modulador em mais uma entrada de dado — exatamente o que a §2.1 proíbe."

O Universe já mediu que modulação multiplicativa bate aditiva 16/16. Mas é preciso verificar se a implementação atual toca `h` diretamente ou modifica `τ` e `β` (correto). Se o canal de gain atual soma ao estado em vez de dividir τ, o ganho medido (+13 a +16%) está subestimado por estar usando o mecanismo errado.

**O que verificar**: no código WGSL da folha cortical ou do brain.wgsl, confirmar que a modulação de gain implementa `τ ← τ/(1 + a₁·m)` e `β ← β₀·(1 + a₂·m)`, e não `h ← h + f(m)`.

---

## Resumo das 5 predições mais fortes e mais baratas de testar

| Rank | Predição | Custo | Risco de refutar |
|---|---|---|---|
| **1** | **P1: Fronteira emerge onde correlação cruzada cai** | Leitura de `buf_afw` + script Python de correlação. Nenhuma mudança de código. | Alto — é o teste central da folha cortical equipotente |
| **2** | **P2: Banco de τ ativo tem spread > 10× nos seres funcionais** | Leitura de `buf_act` + `tau_bank()`. Já existe `read_act_stats`. | Médio — se spread estiver concentrado, a folha não está usando filtros em cascata |
| **3** | **P4: Neuromodulação só paga com contraste de regime alto** | Calcular G analítico da pista, comparar com ganho medido (ablação m≡0 vs m vivo). | Alto — confirma ou refuta a fórmula G no contexto real do Universe |
| **4** | **P6: Razão τ_longo/τ_curto ativas ≈ 4 (teste de sanidade alométrico)** | Cálculo imediato sobre `read_act_stats` + `tau_bank()`. Uma linha de código. | Baixo custo, alta informatividade — se a razão estiver muito fora de 4, algo está errado na calibração |
| **5** | **C3: Verificar que a modulação é multiplicativa sobre τ/β, não aditiva sobre h** | Inspeção de código WGSL. Sem executar nada. | Zero custo computacional, máxima importância arquitetural |

---

## Guia de execução (sem build, sem GPU pesada)

Para as predições P1, P2, P6 e C3 (as mais baratas):

```bash
# P1, P2, P6 — análise offline após dump de buffers
# 1. Adicionar ao loop de simulação: a cada 1000 ticks, salvar buf_act e buf_afw em binário
# 2. Ler com:
python3 tools/cortex_map.py  # a criar — lê buf_afw, plota correlação 32×32
python3 tools/tau_spectrum.py  # a criar — lê buf_act, computa spread dos τ ativos

# C3 — inspeção imediata
grep -n "m_gain\|tau.*mod\|beta.*mod\|gain.*mul" Universe/src/gpu/*.wgsl Universe/src/gpu/brain.wgsl
# Verificar: deve aparecer divisão de tau, não soma a h
```

Para P4 (ablação de modulação):
```bash
# Modificar kernel para aceitar flag "ablate_modulation" via uniform
# Rodar com e sem ablação, comparar fitness médio dos seres após 5000 ticks
```

---

*As ressalvas do doc 08 §12 valem aqui: os expoentes alométricos (~1,35 córtex, ~1,00 cerebelo, ~0,70 resto) e a razão 1:4 foram citados de memória nos documentos fonte. Verificar em Herculano-Houzel (2009) e Finlay & Darlington (1995) antes de usar esses números como critério de falha em experimentos formais.*
