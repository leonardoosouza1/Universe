# Investigation: previsão do shadow no notebook

Status: OPEN — avaliação exploratória, sem mudança de pesos ou runtime.
Data: 2026-09-27 UTC. O pedido do dono é fazer o projeto avançar também no
cérebro e comprovar resultados nos dados reais, sem chamar observação de
aprendizado. O modelo aprovado continua `shadow-20260926-220823`, que prevê;
o Brain vanilla controla os aldeões.

## Acceptance and budget

Esta execução não definiu antecipadamente um limiar de aceitação nem isolou
uma falha do runtime. O teste limitado mede dois dias fechados de um episódio
novo, com um dia anterior para aquecer o estado GRU e um seguinte para rótulos
futuros. O próximo experimento deve pré-registrar dias/vilas, referência de
imobilidade/persistência, cobertura mínima e critério antes de modificar ou
treinar um candidato. Treino/promoção e GPU não foram executados.

## Reproduction identity and executed path

- Universe checkout `74d12e1`, limpo na inspeção. O runtime CPU canônico está
  em `crates/vila-shadow-cpu/`; o binário avaliado veio da imagem do painel
  do release de notebook (`sha256:13ff149f25c3b06a839ef292c5478ecac59a0a77e52e5edda8cf9fa5e4c623d0`).
- Pesos CPU `cpu-model.json` SHA256
  `8b4dda31b57b5ebbded9d5ac8a9ab667bf503460d171d66b068d24b1bb777ff7`;
  `metrics.json` SHA256
  `c858c6a9355243c3442bef8abf6e2efff9dee79193f3ac1e5d1f7e0b2b519c42`.
- Episódio `A-6784215452709343147-20260927-063347`, plugin
  `0.1.1-notebook`, dias avaliados 12 e 20. Cada dia usa o anterior como
  aquecimento e o seguinte para alvos. Os arquivos `keyframes`, `events` e
  `server` gzip já estavam fechados; nenhum arquivo ativo foi avaliado.
- Caminho executado: keyframe Paper → 23 features com relógio do mundo e
  profissão por linha → runtime Rust aprovado com estado por aldeão → saídas
  atividade/deslocamento/sono → rótulos futuros do mesmo aldeão. Frames em
  torno de resgate (1.200 ticks antes, 200 depois) e horizontes com lacunas
  foram excluídos. O controlador do aldeão nunca consumiu essas saídas.
- Script/recibos:
  `/home/leonardo/projetos/IA/m/vila-viva-server/analysis/eval_shadow_closed.py`,
  `/home/leonardo/.local/state/vila-viva/validation/shadow-current-day12-0707.json`
  e `shadow-current-day20-0707.json` no mesmo diretório. Comandos usaram
  `nice -n 10 python3 analysis/eval_shadow_closed.py ... --eval-day 12` e
  `--eval-day 20`, com `--brain-root` no acervo aprovado e `--binary` na cópia
  local da imagem. Cada análise durou ~17 s; os dois testes do mascaramento/
  continuidade passaram.

## Observed comparison

| Dias | Aldeões | Atividade válida | Acurácia GRU | Persistência | Deslocamento válido | MSE GRU | MSE imóvel | Sono |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 20 | 22.180 | 0,9564 | 0,7919 | 22.950 | 0,5621 | 0,5798 | 0,9274 |
| 20 | 20 | 23.250 | 0,9548 | 0,8025 | 23.525 | 0,5688 | 0,5837 | 0,9435 |
| Agregado | 20 em cada dia | 45.430 | 0,9556 | 0,7974 | 46.475 | 0,5655 | 0,5818 | 0,9356 |

O deslocamento superou a referência imóvel por apenas ~2,8% de MSE nesses
dois dias escolhidos. O `shift_realtime` histórico do `metrics.json` aprovado
registrava MSE 0,5337 contra 0,2404 da referência imóvel: o sentido da
comparação muda conforme o domínio. Não há evidência suficiente para alterar
pesos ou afirmar generalização. Foram mascarados 1.430 frames dos dois dias
avaliados; a cobertura é de um episódio e uma semente, sem intervalo de
confiança, classes individuais ou análise de profissão/saúde.

## Gate verdicts and handoff

| Gate | Verdict | Evidence / limit |
|---|---|---|
| Build/schema | PASS for existing release | Binário e pesos aprovados executaram; nenhuma mudança numérica foi compilada. |
| Mechanism | NOT MEASURED | Não houve mudança de equação, feature ou peso a isolar. |
| Behavior | NOT MEASURED | O shadow prevê; não atuou nos aldeões. |
| Evolution | NOT MEASURED | Nenhum treino, seleção ou promoção. |
| Transfer/regression | PARTIAL | Dois dias atuais e referência simples; falta domínio/seed independente. |
| UI | NOT MEASURED | Nenhum replay visual foi alegado. |

Próximo passo: congelar antecipadamente janelas adicionais e medir erro por
classe, por movimento real versus repouso, por saúde/profissão e em episódios
anteriores; só então formular uma intervenção no treino/arquitetura. Qualquer
mudança no runtime CPU deve manter paridade com o release `vila-viva-server` e
revalidar Torch/Rust, sem tocar no cérebro GPU de táxi sem o fluxo GPU próprio.
