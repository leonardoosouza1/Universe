# Piloto por Fórmula — spec de execução (handoff da sessão forja, 2026-08-15)

## Contexto em 5 linhas

Na sessão-forja validamos hoje: pesos de rede CALCULADOS por forma fechada
(`W* = (ΦᵀΦ+λI)⁻¹ΦᵀY`, ridge, sem gradiente) resolvem física suave com erro
0.002–2%, sabem ABSTER-SE fora do domínio (100% de acerto), e "aluno supera
professor ruidoso" (9×). Módulo pronto: `/media/llm-data/forja-2026-08-15/forja.py`
(numpy puro). Agora o alvo é o Universe: **destilar o campeão num piloto
calculado em segundos** a partir do replay dele.

## A tese a testar

O campeão custou ~40 gerações de GPU. O replay dele contém pares (sentiu →
fez) a cada tick. Se um controlador resolvido por ridge sobre esses pares
dirigir ≥70% da distância do campeão, **evolução vira refino**: todo campeão
futuro é destilável, e a geração-0 seguinte já nasce sabendo dirigir.
Se falhar, o MODO da falha diz quanto o estado interno (CfC) importa — número
que nunca medimos por esse eixo.

## O que a sessão Universe precisa produzir (passo 1 — dump)

Uma corrida de observação do campeão atual (o de ~6951 m ou mais novo),
gravando POR TICK, em dois arquivos:

```
/media/llm-data/universe-runs/piloto-formula/X.npy   # float32 [T, C_in]
/media/llm-data/universe-runs/piloto-formula/Y.npy   # float32 [T, C_out]
/media/llm-data/universe-runs/piloto-formula/meta.json
```

- `X[t]` = vetor sensorial COMPLETO que o cérebro recebeu no tick t (retina,
  velocidade, sondas, tudo que entra — os ~192 canais; usar os 142 auditados
  como vivos se preferir, mas documentar quais no meta).
- `Y[t]` = saída MOTORA que o cérebro produziu no tick t (volante, acelerador,
  freio — o que existir), ANTES de qualquer clamp/pós-processamento se
  possível; se só houver pós-clamp, ok, anotar no meta.
- `meta.json`: nomes dos canais na ordem, tick rate, id/checkpoint do campeão,
  seed/pista, posição de largada, versão do binário.

Requisitos:
1. **Largada ÚNICA e fixa** (lição já paga: fitness media a largada — o dump
   e a validação usam a MESMA largada do campeão).
2. Corrida longa: ≥20k ticks (mais = melhor; 1 corrida basta pra v1).
3. Se barato, gravar TAMBÉM uma corrida de um indivíduo mediano (controle:
   o clone do campeão deve bater o mediano; se não bater, o clone não capturou
   nada).

## O solve (passo 2 — qualquer sessão roda, CPU, <10 s)

```python
import numpy as np, json
X = np.load(".../X.npy").astype(np.float64)   # [T, C_in]
Y = np.load(".../Y.npy").astype(np.float64)   # [T, C_out]

# banco de tau (memoria barata; o cerebro original tem CfC):
def ema(X, a):
    E = np.empty_like(X); h = X[0].copy()
    for t in range(len(X)):
        h = a*h + (1-a)*X[t]; E[t] = h
    return E
feats = np.hstack([X, ema(X, 0.8), ema(X, 0.97)])   # [T, 3*C_in]

mu, sd = feats.mean(0), feats.std(0) + 1e-9
Z = (feats - mu) / sd
ym, ys = Y.mean(0), Y.std(0) + 1e-9
Yn = (Y - ym) / ys

cut = int(len(Z) * 0.85)                      # held-out honesto
best = None
A = Z[:cut].T @ Z[:cut]; B = Z[:cut].T @ Yn[:cut]
for lam in (1e-2, 1e-1, 1e0, 1e1):
    W = np.linalg.solve(A + lam*np.eye(Z.shape[1]), B)
    e = float(np.abs(Z[cut:] @ W - Yn[cut:]).mean())
    if best is None or e < best[0]: best = (e, lam, W)
err_ho, lam, W = best
print("erro held-out por acao:", err_ho, "lam:", lam)

json.dump({"W": W.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
           "y_mu": ym.tolist(), "y_sd": ys.tolist(),
           "alphas": [0.8, 0.97], "lam": lam, "err_holdout": err_ho},
          open(".../piloto.json", "w"))
```

## O piloto no runtime (passo 3 — lado Rust)

Implementar `LinearPilot` que lê `piloto.json` e, por tick:

```
z    = concat(x, ema08, ema097)               # 2 estados EMA persistem entre ticks
z    = (z - mu) / sd
y    = (z @ W) * y_sd + y_mu                  # UMA matvec [3*C_in x C_out]
```

Estado: só os 2 vetores EMA (`h = a*h + (1-a)*x`). Custo por tick: ~1
matvec pequena — desprezível.

**v2 com abstenção (se v1 andar):** guardar 512 linhas de Z do treino como
referência; se `min_dist(z, refs)` > limiar (quantil 0.99 do treino), estado
desconhecido → fallback (frear / reduzir marcha). É o "sabe que não sabe".

## Validação (passo 4)

Mesma pista, mesma largada, headless:

| corrida | esperado |
|---|---|
| campeão original | ~6951 m (referência) |
| clone linear (só X, sem EMA) | mede a parte reativa |
| clone + banco de τ (0.8, 0.97) | mede quanto memória barata recupera |
| indivíduo mediano | o clone TEM que bater isto |

Reportar: metros de cada um, held-out do solve, e ONDE o clone se perde
(tick/trecho da pista) se se perder.

## Armadilhas conhecidas (pagas em outras frentes)

- **Rajada/deriva**: erro pequeno leva a estado nunca visitado → erro cresce
  → clone se perde. É esperado; por isso a métrica é metros na pista, não só
  erro por tick. Erro por tick baixo + pista ruim = deriva (documentar, é
  resultado válido).
- **Volante cravado**: o campeão atual tem vício de círculo (74% virando à
  direita). O clone vai herdar. Não é bug do clone.
- **Normalização**: sensores têm escalas muito diferentes — o (feats-mu)/sd é
  OBRIGATÓRIO, e o piloto Rust precisa usar mu/sd DO ARQUIVO, não recalcular.
- **Ordem dos canais**: a ordem em X.npy e no runtime tem que ser idêntica —
  conferir com 3 ticks de sanidade (dump de z no Rust vs Python).
- **Orçamento de hardware (regra nova global)**: nada acima de 80% de
  VRAM/RAM/threads; este experimento é todo leve, mas a corrida de validação
  segue a regra.

## Leitura dos desfechos

- **≥70% do campeão**: destilação funciona → usar clone como gen-0 das
  próximas evoluções (semear população com mutações do W) e ler os pesos
  (importância por sensor de graça — comparar com ablação por sentido).
- **Anda mas se perde**: mapear os trechos → abstenção+fallback na v2 →
  fronteira exata do que exige o cérebro completo.
- **Não anda**: CfC/estado interno é essencial mesmo com banco de τ →
  medir de novo com mais αs (0.5/0.9/0.99) antes de concluir; se persistir,
  é o número honesto de quanto o líquido importa (responde pergunta aberta
  do saci também).

Resultados → `/media/llm-data/universe-runs/piloto-formula/RESULTADO.md` e
me avisem na sessão-forja que eu cruzo com a bateria A/B/C.
