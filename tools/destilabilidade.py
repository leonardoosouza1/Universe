#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
destilabilidade.py — indice D de destilabilidade da politica de um ser do Universe.

PERGUNTA QUE ESTA FERRAMENTA RESPONDE
-------------------------------------
Quanto da politica do campeao (o par sentiu->fez, tick a tick) e extraivel por
FORMA FECHADA, ou seja, por um controlador LINEAR resolvido por ridge? Se quase
tudo e extraivel, a evolucao esta refinando algo que 10 s de CPU resolvem, e o
MUNDO e facil demais. Se nao e, o cerebro esta fazendo algo que formula nao pega.

    D alto  (~1)  -> mundo facil: apertar a escada de dificuldade
    D caindo      -> o degrau ligado tornou o mundo mais exigente (dose certa)
    D baixo       -> politica genuinamente nao-linear/contextual: evoluir vale a GPU

DEFINICAO DE D (imprimida tambem na saida)
------------------------------------------
Para cada acao a in {acelerador, volante, freio}, no HELD-OUT (ultimos 15%
TEMPORAIS da corrida — nunca sorteio aleatorio: serie a 64 Hz vaza vizinho):

    R2_a = 1 - SQE_a / SQT_a
    SQE_a = sum_t (y_at - yhat_at)^2        (erro do ridge)
    SQT_a = sum_t (y_at - media_holdout_a)^2 (variancia do proprio held-out)

    D = sum_a Var_a * R2_a / sum_a Var_a          com Var_a = SQT_a / n_holdout

Ou seja: D e a media dos R2 PONDERADA pela variancia de cada acao nas unidades
CRUAS de intent. Identidade util: D == 1 - (sum_a SQE_a) / (sum_a SQT_a), o R2
agregado. Acao que quase nao varia quase nao pesa (nao adianta acertar o freio
que nunca e pisado). D e recortado em [0,1] para leitura; o valor bruto tambem
sai no JSON (pode ser negativo se o linear for pior que a media).

R2 e invariante a reescala linear do alvo, entao o "espaco normalizado" muda o
erro absoluto mas nao muda R2 — por isso as duas coisas sao reportadas: o erro
normalizado (comparavel com o piloto.json / RESULTADO.md) e o R2 (legivel).

Sobre o erro normalizado: RESULTADO.md do piloto-por-formula reporta MAE/y_sd
(nao RMSE). Conferido em 2026-08-16 refazendo o ajuste sobre X.npy/Y.npy com
lam=10: MAE/y_sd = 0.180 / 0.157 / 0.455 (media 0.264) contra os 0.177 / 0.156
/ 0.453 (media 0.262) da tabela — bate dentro de ~1%, o resto e diferenca de
onde se tira mu/sd. Por isso a tabela abaixo imprime MAE e RMSE lado a lado:
MAE e o numero comparavel com a sessao do piloto, RMSE e o que se relaciona com
o R2 (R2 = 1 - RMSE_norm^2 quando a normalizacao usa o sd do held-out).

ENTRADA
-------
    python tools/destilabilidade.py <obs.f32> [<obs2.f32> ...] [--json saida.json]

obs.f32 vem da flag --observar do gpu_evolve. Linhas de 394 float32:
  [0]tick [1]vel [2]rpm [3]gear [4]temp [5]thr [6]steer [7]brk [8]vida [9]joy
  [10..106) canais 0-95 pos-CfC · [106..202) canais 96-191 pos-KAN
  [202..298) l1_raw · [298..394) intent (as 3 primeiras sao as acoes usadas)
Logo X = colunas [10:202] (192 canais, exatamente como o L1 recebe) e
      Y = colunas [298:301] (intent cru pos-tanh, pre-clamps).
Verificado contra o dump do piloto-por-formula: obs[:,10:202] == X.npy e
obs[:,298:301] == Y.npy com diff maximo 0.0 (obs-m23-re-nova, 2026-08-16).

SANIDADES OBRIGATORIAS (item (d) do spec)
-----------------------------------------
 1. preditor CONSTANTE (media do treino): R2 deve dar ~0. Se der bem acima de
    0, a media do treino e do held-out divergem muito (deriva de regime).
 2. X EMBARALHADO no tempo no held-out (mesmo W, linhas de X permutadas com
    semente fixa): deve DESTRUIR o R2. Se nao destruir, o alvo e trivial (quase
    constante) e D nao significa nada -> imprime ALARME.
 3. PERSISTENCIA (y_t = y_{t-1}), o preditor mais burro possivel numa serie de
    64 Hz. Nao e baseline do spec, mas e o contraste honesto: se a persistencia
    ganha do ridge, o R2 alto e da suavidade do sinal, nao da politica -> ALARME.

D_SUPRA — o D descontado da suavidade (leia junto com D)
--------------------------------------------------------
A 64 Hz a acao quase nao muda entre ticks, entao QUALQUER preditor decente pega
R2 alto de graca. Medido nas 3 corridas de 2026-08-16, a persistencia bate o
ridge em todas (D_pers 0.973/0.992/0.931 contra D 0.927/0.980/0.928). Logo D
sozinho mistura "a politica e linear" com "o sinal e liso". Por isso sai tambem:

    D_supra = 1 - soma(SQE_ridge) / soma(SQE_persistencia)

= quanto do que a mera continuidade NAO explica o linear ainda pega. Positivo =
o linear sabe algo sobre a politica alem de "continue fazendo o que fazia";
negativo = nao sabe. Ressalva de honestidade: a persistencia enxerga y_{t-1}
real, coisa que um controlador em malha fechada nao tem — ela nao e um piloto
concorrente, e so a regua da trivialidade do alvo. Por isso D_supra negativo NAO
contradiz o clone linear ter dirigido 7156 m; contradiz apenas ler R2=0.93 como
"93% da politica e linear".

D_SEM_REALIMENTACAO — o D do MUNDO (o numero para a escada)
-----------------------------------------------------------
Tres canais de entrada sao, na pratica, a acao ANTERIOR do proprio ser:
ch120 Memory0Cfc, ch121 SteerMem, ch122 BrakeMem. Medido: corr(ch121_t,
volante_{t-1}) = +0.988 (obs-m23-re-nova) e +0.978 (obs-campeao). Com eles na
entrada, "a politica e linear" se confunde com "a acao e continua" — o ridge
copia o proprio passado. A ferramenta detecta sozinha qualquer canal com
|corr(ch_t, y_{t-1})| >= 0.90, reajusta o ridge SEM eles, e reporta
D_sem_realimentacao. Para medir a DIFICULDADE DO MUNDO (o uso na escada), este
e o numero a acompanhar; D cheio e o numero a citar quando se fala do piloto
por formula, porque o piloto tem esses canais na entrada de verdade.

Sem dependencias alem de numpy. Nada de GPU: puro CPU.
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Layout do obs.f32
# ---------------------------------------------------------------------------
LARGURA_LINHA = 394
COL_X = (10, 202)      # 192 canais como o L1 recebe
COL_Y = (298, 301)     # intent cru: acelerador, volante, freio
ACOES = ["acelerador", "volante", "freio"]

# Familias de canal (inicio inclusivo, fim exclusivo) — faixas do spec da sessao.
# Os nomes entre parenteses foram conferidos em piloto-formula/meta.json["canais"].
# A faixa 123-126 NAO estava na lista do spec; entra explicita para que nenhum
# canal seja silenciosamente descartado da quebra por bloco.
FAMILIAS = [
    ("retina",            0,  16),
    ("parede dist",      16,  32),
    ("parede curv",      32,  40),
    ("estrada perto/longe", 40, 56),
    ("material+fuga",    56,  64),
    ("CORPO",            64,  76),
    ("proximidade",      76,  96),
    ("derivados",        96, 120),
    ("memoria",         120, 123),   # Memory0Cfc, SteerMem, BrakeMem
    ("whisker/spin/stress", 123, 126),  # WhiskerNorm, Spinning, CompStress
    ("dopamina",        126, 128),   # JoyTick, JoyStreak
    ("FOVEA",           128, 144),
    ("PERIFERICA",      144, 160),
    ("HORIZONTE",       160, 192),
]

FRACAO_HOLDOUT = 0.15
GRADE_LAMBDA = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
SEMENTE_EMBARALHA = 20260816


# ---------------------------------------------------------------------------
# Nucleo numerico
# ---------------------------------------------------------------------------
COL_VEL = 1
COL_VIDA = 8


def carregar(caminho):
    """Le obs.f32 e devolve (X [n,192], Y [n,3], obs [n,394])."""
    bruto = np.fromfile(caminho, dtype=np.float32)
    if bruto.size == 0:
        raise ValueError("arquivo vazio: %s" % caminho)
    if bruto.size % LARGURA_LINHA != 0:
        raise ValueError(
            "%s tem %d float32, nao multiplo de %d (formato obs.f32)"
            % (caminho, bruto.size, LARGURA_LINHA)
        )
    obs = bruto.reshape(-1, LARGURA_LINHA)
    X = obs[:, COL_X[0]:COL_X[1]].astype(np.float64)
    Y = obs[:, COL_Y[0]:COL_Y[1]].astype(np.float64)
    return X, Y, obs


def regime(obs, corte):
    """Compara o trecho de treino com o de held-out. Existe para pegar a armadilha
    real: se o ser MORREU ou encalhou antes do fim, o held-out vira um trecho
    constante, SQT vai a ~0 e o R2 explode (medido: R2=-11763 em obs-campeao-rpm,
    com vel media 0.11 m/s e 98.5% dos ticks parados no held-out). Nesse caso D nao
    mede destilabilidade nenhuma — mede que o alvo sumiu."""
    vel = obs[:, COL_VEL]
    vida = obs[:, COL_VIDA]
    return {
        "vel_media_treino": float(vel[:corte].mean()),
        "vel_media_holdout": float(vel[corte:].mean()),
        "frac_parado_treino": float((vel[:corte] < 0.5).mean()),
        "frac_parado_holdout": float((vel[corte:] < 0.5).mean()),
        "vida_final": float(vida[-1]),
    }


def zscore_treino(X, mu, sd):
    return (X - mu) / sd


def diagnostico_canais(X):
    """Canais mortos (constantes na corrida inteira) e canais BIT-IDENTICOS entre si.

    Regra do projeto: numero identico onde deveria divergir = alarme. Duas colunas
    de sensor byte-a-byte iguais significam sensor duplicado ou fiacao errada, e
    inflam artificialmente a leitura por familia (a familia copiada 'preve' tanto
    quanto a original). Detecta por hash dos bytes da coluna: O(n_canais).
    """
    mortos = [int(c) for c in np.where(X.std(axis=0) < 1e-12)[0]]
    baldes = {}
    for c in range(X.shape[1]):
        col = np.ascontiguousarray(X[:, c])
        baldes.setdefault(col.tobytes(), []).append(int(c))
    grupos_mortos, grupos_vivos = [], []
    for _, cols in baldes.items():
        if len(cols) < 2:
            continue
        (grupos_mortos if cols[0] in mortos else grupos_vivos).append(cols)
    grupos_vivos.sort()
    grupos_mortos.sort()
    return {"mortos": mortos, "dup_vivos": grupos_vivos, "dup_mortos": grupos_mortos}


def detectar_realimentacao(X, Y, limiar=0.90):
    """Acha canais de entrada que sao, na pratica, a ACAO ANTERIOR do proprio ser.

    Motivo: ch120-122 (Memory0Cfc / SteerMem / BrakeMem) sao realimentacao do
    comando. Medido em 2026-08-16: corr(ch121_t, volante_{t-1}) = +0.988 em
    obs-m23-re-nova e +0.978 em obs-campeao. Um ridge com esses canais na entrada
    esta em boa parte copiando a ultima acao — e "a politica e linear" vira
    "a acao e continua". Quem quer medir a DIFICULDADE DO MUNDO tem de olhar o D
    sem eles. Devolve a lista de canais com |corr(ch_t, y_{t-1})| >= limiar.
    """
    achados = []
    Yprev = Y[:-1]
    Xat = X[1:]
    for c in range(X.shape[1]):
        col = Xat[:, c]
        if col.std() < 1e-12:
            continue
        for j in range(Y.shape[1]):
            if Yprev[:, j].std() < 1e-12:
                continue
            r = float(np.corrcoef(col, Yprev[:, j])[0, 1])
            if abs(r) >= limiar:
                achados.append({"canal": int(c), "acao": ACOES[j], "corr": r,
                                "familia": nome_familia(c)})
                break
    return achados


def nome_familia(c):
    for nome, ini, fim in FAMILIAS:
        if ini <= c < fim:
            return nome
    return "?"


def resolver_ridge(Xz, Yc, lam):
    """W = (XzT Xz + lam I)^-1 XzT Yc. Xz ja centrado/escalado, Yc ja centrado."""
    d = Xz.shape[1]
    A = Xz.T @ Xz + lam * np.eye(d)
    B = Xz.T @ Yc
    return np.linalg.solve(A, B)


def escolher_lambda(Xz_tr, Yc_tr):
    """Escolhe lambda por split TEMPORAL interno (ultimos 15% do treino).

    Nunca usa o held-out final — senao o numero reportado seria otimista.
    Criterio: menor erro medio (RMSE normalizado pelo sd do sub-treino).
    """
    n = Xz_tr.shape[0]
    corte = int(round(n * (1.0 - FRACAO_HOLDOUT)))
    corte = max(min(corte, n - 1), 1)
    Xa, Xb = Xz_tr[:corte], Xz_tr[corte:]
    Ya, Yb = Yc_tr[:corte], Yc_tr[corte:]
    if Xb.shape[0] < 10:  # corrida curta demais para split interno
        return 10.0, []
    sd_a = Ya.std(axis=0)
    sd_a[sd_a < 1e-12] = 1.0
    trilha = []
    melhor, melhor_lam = np.inf, GRADE_LAMBDA[0]
    for lam in GRADE_LAMBDA:
        W = resolver_ridge(Xa, Ya, lam)
        pred = Xb @ W
        err = float(np.mean(np.sqrt(np.mean(((pred - Yb) / sd_a) ** 2, axis=0))))
        trilha.append({"lam": lam, "erro_interno": err})
        if err < melhor:
            melhor, melhor_lam = err, lam
    return melhor_lam, trilha


def r2_por_acao(Y_true, Y_pred):
    """R2 = 1 - SQE/SQT usando a media do PROPRIO held-out em SQT."""
    sqe = np.sum((Y_true - Y_pred) ** 2, axis=0)
    sqt = np.sum((Y_true - Y_true.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(sqt > 1e-18, 1.0 - sqe / np.maximum(sqt, 1e-18), 0.0)
    return r2, sqe, sqt


def indice_D(sqe, sqt):
    """D = sum Var_a R2_a / sum Var_a == 1 - sum SQE / sum SQT (R2 agregado)."""
    tot_t = float(np.sum(sqt))
    if tot_t <= 1e-18:
        return 0.0, 0.0
    d_bruto = 1.0 - float(np.sum(sqe)) / tot_t
    return float(np.clip(d_bruto, 0.0, 1.0)), d_bruto


def ajustar_e_avaliar(X, Y, colunas=None, lam=None):
    """Ajusta ridge no treino (85% iniciais) e avalia no held-out (15% finais).

    Devolve dicionario com erro normalizado, R2 por acao, D e o necessario para
    as sanidades. Toda estatistica de normalizacao vem SO do treino.
    """
    n = X.shape[0]
    corte = int(round(n * (1.0 - FRACAO_HOLDOUT)))
    corte = max(min(corte, n - 1), 1)

    Xs = X if colunas is None else X[:, colunas]
    X_tr, X_ho = Xs[:corte], Xs[corte:]
    Y_tr, Y_ho = Y[:corte], Y[corte:]

    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    n_mortos = int(np.sum(sd < 1e-8))
    sd = np.where(sd < 1e-8, 1.0, sd)

    y_mu = Y_tr.mean(axis=0)
    y_sd = Y_tr.std(axis=0)
    y_sd_seguro = np.where(y_sd < 1e-8, 1.0, y_sd)

    Xz_tr = zscore_treino(X_tr, mu, sd)
    Xz_ho = zscore_treino(X_ho, mu, sd)
    Yc_tr = Y_tr - y_mu

    if lam is None:
        lam, trilha = escolher_lambda(Xz_tr, Yc_tr)
    else:
        trilha = []

    W = resolver_ridge(Xz_tr, Yc_tr, lam)
    pred_ho = Xz_ho @ W + y_mu

    # (a) erro no espaco normalizado. MAE/y_sd e a metrica do RESULTADO.md do piloto;
    # RMSE/y_sd e a que conversa com o R2.
    err_norm = np.sqrt(np.mean(((pred_ho - Y_ho) / y_sd_seguro) ** 2, axis=0))
    mae_norm = np.mean(np.abs((pred_ho - Y_ho) / y_sd_seguro), axis=0)

    # (b)(c) R2 e D
    r2, sqe, sqt = r2_por_acao(Y_ho, pred_ho)
    D, D_bruto = indice_D(sqe, sqt)

    var_ho = sqt / max(Y_ho.shape[0], 1)
    razao_sd = np.sqrt(var_ho) / y_sd_seguro  # held-out ainda varia como o treino?

    return {
        "razao_sd_holdout_treino": razao_sd,
        "y_sd_treino": y_sd,
        "n_ticks": int(n),
        "n_treino": int(corte),
        "n_holdout": int(n - corte),
        "n_features": int(Xs.shape[1]),
        "canais_constantes_no_treino": n_mortos,
        "lam": float(lam),
        "trilha_lambda": trilha,
        "erro_norm": err_norm,
        "erro_norm_medio": float(np.mean(err_norm)),
        "mae_norm": mae_norm,
        "mae_norm_medio": float(np.mean(mae_norm)),
        "r2": r2,
        "sqe": sqe,
        "sqt": sqt,
        "var_holdout": var_ho,
        "D": D,
        "D_bruto": D_bruto,
        "W": W,
        "mu": mu, "sd": sd, "y_mu": y_mu, "y_sd": y_sd,
        "corte": corte,
        "pred_ho": pred_ho,
        "Y_ho": Y_ho,
        "Xz_ho": Xz_ho,
    }


def sanidades(base, Y, reg):
    """(d) baselines: constante, X embaralhado no tempo, persistencia."""
    Y_ho = base["Y_ho"]
    n_ho = Y_ho.shape[0]

    # 1. preditor constante = media do TREINO
    pred_const = np.repeat(base["y_mu"][None, :], n_ho, axis=0)
    r2_const, _, _ = r2_por_acao(Y_ho, pred_const)

    # 2. mesmo W, mas X do held-out embaralhado no tempo (semente fixa)
    rng = np.random.default_rng(SEMENTE_EMBARALHA)
    ordem = rng.permutation(n_ho)
    pred_emb = base["Xz_ho"][ordem] @ base["W"] + base["y_mu"]
    r2_emb, sqe_e, sqt_e = r2_por_acao(Y_ho, pred_emb)
    D_emb, _ = indice_D(sqe_e, sqt_e)

    # 3. persistencia y_t = y_{t-1} (usa o ultimo tick do treino como semente)
    Y_prev = np.vstack([Y[base["corte"] - 1:base["corte"]], Y_ho[:-1]])
    r2_pers, sqe_p, sqt_p = r2_por_acao(Y_ho, Y_prev)
    D_pers, _ = indice_D(sqe_p, sqt_p)

    # D_supra: quanto o ridge pega ALEM do que a continuidade ja da (skill score)
    sqe_ridge = np.sum((Y_ho - base["pred_ho"]) ** 2)
    D_supra = float(1.0 - sqe_ridge / max(float(np.sum(sqe_p)), 1e-18))

    alarmes = []

    # ARMADILHA PRINCIPAL: ser morto/encalhado no trecho de teste.
    degenerado = (float(np.min(base["razao_sd_holdout_treino"])) < 0.10
                  or (reg["frac_parado_holdout"] > 0.90 and reg["frac_parado_treino"] < 0.60))
    if degenerado:
        alarmes.append(
            "ALARME: HELD-OUT DEGENERADO — o ser esta parado/morto no trecho de teste "
            "(vel media %.2f m/s contra %.2f no treino; %.1f%% dos ticks parados contra "
            "%.1f%%; vida final %.1f). A acao vira quase constante, SQT vai a zero e o R2 "
            "explode. D NAO e comparavel com o de outras corridas: reveja o dump ou corte "
            "a corrida antes da morte."
            % (reg["vel_media_holdout"], reg["vel_media_treino"],
               100.0 * reg["frac_parado_holdout"], 100.0 * reg["frac_parado_treino"],
               reg["vida_final"])
        )

    if float(np.max(r2_emb)) > 0.20 or D_emb > 0.20:
        alarmes.append(
            "ALARME: X embaralhado no tempo NAO destruiu o R2 (D_embaralhado=%.3f). "
            "O alvo e quase trivial (acao quase constante) — D nao significa nada nesta corrida."
            % D_emb
        )
    if D_pers >= base["D"]:
        alarmes.append(
            "ALARME: persistencia (y_t = y_{t-1}) empata ou ganha do ridge "
            "(D_persistencia=%.3f >= D=%.3f; D_supra=%+.3f). Boa parte do R2 vem da "
            "SUAVIDADE do sinal a 64 Hz, nao da politica: use D_supra para comparar "
            "corridas, e D so como numero absoluto." % (D_pers, base["D"], D_supra)
        )
    if float(np.max(np.abs(r2_const))) > 0.10:
        alarmes.append(
            "ATENCAO: preditor constante deu R2 = %s (deveria ser ~0). Media do treino e do "
            "held-out divergem: ha deriva de regime dentro da corrida."
            % np.array2string(r2_const, precision=3)
        )

    return {
        "r2_constante": r2_const,
        "r2_embaralhado": r2_emb, "D_embaralhado": D_emb,
        "r2_persistencia": r2_pers, "D_persistencia": D_pers,
        "D_supra": D_supra,
        "degenerado": bool(degenerado),
        "alarmes": alarmes,
    }


def quebra_por_familia(X, Y, lam):
    """(e) qual familia de canais SOZINHA ja preve a acao. Mesmo lam do modelo cheio
    seria injusto (menos features pedem menos regularizacao), entao cada familia
    escolhe o proprio lambda pelo mesmo split interno."""
    linhas = []
    for nome, ini, fim in FAMILIAS:
        cols = np.arange(ini, min(fim, X.shape[1]))
        if cols.size == 0:
            continue
        res = ajustar_e_avaliar(X, Y, colunas=cols, lam=None)
        linhas.append({
            "familia": nome, "faixa": "%d-%d" % (ini, fim), "n_canais": int(cols.size),
            "lam": res["lam"],
            "r2": res["r2"], "D": res["D"], "erro_norm": res["erro_norm"],
        })
    linhas.sort(key=lambda r: -r["D"])
    return linhas


# ---------------------------------------------------------------------------
# Saida
# ---------------------------------------------------------------------------
def barra(v, largura=20):
    v = float(np.clip(v, 0.0, 1.0))
    cheio = int(round(v * largura))
    return "#" * cheio + "." * (largura - cheio)


def leitura_de_D(D):
    if D >= 0.90:
        return ("MUITO ALTO", "a politica e praticamente uma formula: o mundo NAO exige cerebro. "
                              "Apertar a escada de dificuldade.")
    if D >= 0.75:
        return ("ALTO", "a evolucao esta refinando algo quase linear. Mundo facil; ha folga para "
                        "subir um degrau.")
    if D >= 0.50:
        return ("MEDIO", "metade da politica escapa da forma fechada. Dose do degrau plausivel — "
                         "acompanhar a serie de D entre geracoes.")
    if D >= 0.25:
        return ("BAIXO", "o cerebro faz algo que formula nao pega. Aqui evoluir paga o custo de GPU.")
    return ("MUITO BAIXO", "politica dominada por nao-linearidade/contexto (ou corrida degenerada — "
                           "conferir as sanidades antes de comemorar).")


def imprimir(caminho, base, san, familias, diag, reg):
    print("=" * 78)
    print("ARQUIVO: %s" % caminho)
    print("=" * 78)
    print("ticks=%d  treino=%d (85%% iniciais)  held-out=%d (15%% FINAIS, temporal)"
          % (base["n_ticks"], base["n_treino"], base["n_holdout"]))
    print("features=%d canais (obs[:,10:202])   canais constantes no treino=%d   lambda=%g"
          % (base["n_features"], base["canais_constantes_no_treino"], base["lam"]))
    print("regime: vel media %.2f (treino) -> %.2f m/s (held-out) | parado %.1f%% -> %.1f%% "
          "| vida final %.1f"
          % (reg["vel_media_treino"], reg["vel_media_holdout"],
             100.0 * reg["frac_parado_treino"], 100.0 * reg["frac_parado_holdout"],
             reg["vida_final"]))

    if diag["dup_vivos"]:
        n_ch = sum(len(g) - 1 for g in diag["dup_vivos"])
        print()
        print("  >>> ALARME DE CANAIS: %d grupo(s) de canais VIVOS bit-a-bit IDENTICOS "
              "(%d canais redundantes)." % (len(diag["dup_vivos"]), n_ch))
        for g in diag["dup_vivos"][:12]:
            print("      ch%s  (familias: %s)"
                  % ("=".join(str(c) for c in g),
                     ", ".join(sorted(set(nome_familia(c) for c in g)))))
        if len(diag["dup_vivos"]) > 12:
            print("      ... e mais %d grupo(s)." % (len(diag["dup_vivos"]) - 12))
        print("      Sensor duplicado infla a quebra por familia: a copia 'preve' tanto")
        print("      quanto a original. Nao e coincidencia — investigar a fiacao.")
    if diag["mortos"]:
        print("  canais MORTOS (constantes a corrida inteira, %d): %s"
              % (len(diag["mortos"]), ",".join("ch%d" % c for c in diag["mortos"][:24])))
    print()

    print("(a)(b) RIDGE LINEAR — held-out")
    print("  %-12s %9s %9s %9s %10s %11s"
          % ("acao", "MAE_norm", "RMSE_norm", "R2", "var_ho", "peso em D"))
    peso = base["var_holdout"] / max(float(np.sum(base["var_holdout"])), 1e-18)
    for i, a in enumerate(ACOES):
        print("  %-12s %9.4f %9.4f %9.4f %10.5f %10.1f%%"
              % (a, base["mae_norm"][i], base["erro_norm"][i], base["r2"][i],
                 base["var_holdout"][i], 100.0 * peso[i]))
    print("  %-12s %9.4f %9.4f" % ("MEDIA", base["mae_norm_medio"], base["erro_norm_medio"]))
    print("  (erro dividido pelo desvio-padrao da acao no treino. MAE_norm e a metrica")
    print("   do RESULTADO.md/piloto.json; R2 = 1 - SQE/SQT, SQT na media do held-out)")
    print()

    print("(c) INDICE DE DESTILABILIDADE")
    print("  D = soma_a Var_a * R2_a / soma_a Var_a   ==   1 - soma(SQE) / soma(SQT)")
    print("      (media dos R2 ponderada pela variancia CRUA de cada acao no held-out)")
    if san["degenerado"]:
        rot, txt = "INVALIDO", ("held-out degenerado — o ser morreu/encalhou antes do fim. "
                                "Este D nao mede destilabilidade, ver sanidade abaixo.")
    else:
        rot, txt = leitura_de_D(base["D"])
    print()
    print("  D = %.4f  [%s]   %s" % (base["D"], barra(base["D"]), rot))
    if abs(base["D_bruto"] - base["D"]) > 1e-9:
        print("  (D bruto sem recorte = %.4f — linear pior que a media em alguma acao)"
              % base["D_bruto"])
    print("  media simples dos R2 (sem ponderar) = %.4f" % float(np.mean(base["r2"])))
    print("  leitura: %s" % txt)
    print()
    if san["canais_realimentacao"]:
        print("  D_sem_realimentacao = %.4f   (D do MUNDO: %.4f - %.4f = %.4f de queda)"
              % (san["D_sem_realimentacao"], base["D"], san["D_sem_realimentacao"],
                 base["D"] - san["D_sem_realimentacao"]))
        print("      canais removidos por serem a ACAO ANTERIOR do proprio ser:")
        for f in san["realimentacao"]:
            print("        ch%-3d %-12s corr com %-9s anterior = %+.4f"
                  % (f["canal"], f["familia"], f["acao"], f["corr"]))
        print("      R2 sem eles: %s"
              % "  ".join("%s %.3f" % (ACOES[i], san["r2_sem_realimentacao"][i])
                          for i in range(len(ACOES))))
        print()
    print("  D_supra = 1 - SQE_ridge / SQE_persistencia = %+.4f" % san["D_supra"])
    print("      quanto o linear pega ALEM da mera continuidade (y_t = y_{t-1}).")
    if san["D_supra"] <= 0:
        print("      NEGATIVO: o mapa reativo linear e PIOR que repetir a ultima acao —")
        print("      o D alto acima vem em grande parte da suavidade do sinal a 64 Hz.")
    print()

    print("(d) SANIDADE — o numero acima so vale se estas passarem")
    print("  %-34s %8s %8s %8s %8s" % ("preditor", "acel", "volante", "freio", "D"))
    print("  %-34s %8.4f %8.4f %8.4f %8s"
          % ("constante (media do treino)", san["r2_constante"][0], san["r2_constante"][1],
             san["r2_constante"][2], "0.000"))
    print("  %-34s %8.4f %8.4f %8.4f %8.3f"
          % ("ridge com X embaralhado no tempo", san["r2_embaralhado"][0],
             san["r2_embaralhado"][1], san["r2_embaralhado"][2], san["D_embaralhado"]))
    print("  %-34s %8.4f %8.4f %8.4f %8.3f"
          % ("persistencia y_t = y_{t-1}", san["r2_persistencia"][0],
             san["r2_persistencia"][1], san["r2_persistencia"][2], san["D_persistencia"]))
    print("  %-34s %8.4f %8.4f %8.4f %8.3f"
          % ("RIDGE (o medido)", base["r2"][0], base["r2"][1], base["r2"][2], base["D"]))
    if san["alarmes"]:
        print()
        for a in san["alarmes"]:
            print("  >>> " + a)
    else:
        print("  ok: constante ~0 e embaralhado destruiu o R2 — o alvo nao e trivial.")
    print()

    print("(e) QUEBRA POR FAMILIA DE CANAIS — cada familia SOZINHA (ordenado por D)")
    print("  %-22s %7s %5s %7s %7s %7s %7s  %s"
          % ("familia", "faixa", "n", "acel", "volante", "freio", "D", "barra"))
    for l in familias:
        print("  %-22s %7s %5d %7.3f %7.3f %7.3f %7.3f  %s"
              % (l["familia"], l["faixa"], l["n_canais"], l["r2"][0], l["r2"][1],
                 l["r2"][2], l["D"], barra(l["D"], 16)))
    print("  (soma dos D das familias > D total e esperado: os canais sao redundantes)")
    print()


def para_json(caminho, base, san, familias, diag, reg):
    return {
        "arquivo": os.path.abspath(caminho),
        "definicao_D": ("D = soma_a Var_a*R2_a / soma_a Var_a == 1 - soma(SQE)/soma(SQT); "
                        "held-out = ultimos 15% TEMPORAIS; Var_a = variancia crua da acao a "
                        "no held-out"),
        "n_ticks": base["n_ticks"], "n_treino": base["n_treino"],
        "n_holdout": base["n_holdout"], "n_features": base["n_features"],
        "canais_constantes_no_treino": base["canais_constantes_no_treino"],
        "lambda": base["lam"],
        "acoes": ACOES,
        "erro_norm_rmse": base["erro_norm"].tolist(),
        "erro_norm_rmse_medio": base["erro_norm_medio"],
        "erro_norm_mae": base["mae_norm"].tolist(),
        "erro_norm_mae_medio": base["mae_norm_medio"],
        "diagnostico_canais": {
            "mortos": diag["mortos"],
            "grupos_identicos_vivos": diag["dup_vivos"],
            "grupos_identicos_mortos": diag["dup_mortos"],
        },
        "r2": base["r2"].tolist(),
        "var_holdout": base["var_holdout"].tolist(),
        "D": base["D"], "D_bruto": base["D_bruto"],
        "r2_medio_simples": float(np.mean(base["r2"])),
        "leitura": "INVALIDO" if san["degenerado"] else leitura_de_D(base["D"])[0],
        "regime": reg,
        "razao_sd_holdout_treino": base["razao_sd_holdout_treino"].tolist(),
        "sanidade": {
            "r2_constante": san["r2_constante"].tolist(),
            "r2_embaralhado": san["r2_embaralhado"].tolist(),
            "D_embaralhado": san["D_embaralhado"],
            "r2_persistencia": san["r2_persistencia"].tolist(),
            "D_persistencia": san["D_persistencia"],
            "alarmes": san["alarmes"],
        },
        "D_supra": san["D_supra"],
        "D_sem_realimentacao": san["D_sem_realimentacao"],
        "canais_realimentacao": san["canais_realimentacao"],
        "realimentacao": san["realimentacao"],
        "familias": [
            {"familia": l["familia"], "faixa": l["faixa"], "n_canais": l["n_canais"],
             "lambda": l["lam"], "r2": l["r2"].tolist(), "D": l["D"],
             "erro_norm": l["erro_norm"].tolist()}
            for l in familias
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Indice de destilabilidade D: fracao da politica do ser que um "
                    "controlador linear (ridge, forma fechada) reproduz.")
    ap.add_argument("obs", nargs="+", help="um ou mais caminhos de obs.f32")
    ap.add_argument("--json", default=None, help="grava o resultado completo em JSON")
    ap.add_argument("--lam", type=float, default=None,
                    help="fixa o lambda do ridge (padrao: escolhido por split temporal interno)")
    ap.add_argument("--sem-familias", action="store_true",
                    help="pula a quebra por familia de canais (item e)")
    args = ap.parse_args(argv)

    saida = []
    falhas = []
    for caminho in args.obs:
        try:
            X, Y, obs = carregar(caminho)
        except Exception as e:  # arquivo torto nao derruba os demais
            print("ERRO em %s: %s" % (caminho, e), file=sys.stderr)
            falhas.append({"arquivo": caminho, "erro": str(e)})
            continue
        base = ajustar_e_avaliar(X, Y, lam=args.lam)
        reg = regime(obs, base["corte"])
        san = sanidades(base, Y, reg)
        diag = diagnostico_canais(X)

        # D sem os canais que realimentam a acao anterior (o D "do mundo")
        fb = detectar_realimentacao(X, Y)
        canais_fb = sorted(set(f["canal"] for f in fb))
        if canais_fb:
            restantes = np.array([c for c in range(X.shape[1]) if c not in set(canais_fb)])
            base_sf = ajustar_e_avaliar(X, Y, colunas=restantes, lam=args.lam)
            san["realimentacao"] = fb
            san["canais_realimentacao"] = canais_fb
            san["D_sem_realimentacao"] = base_sf["D"]
            san["r2_sem_realimentacao"] = base_sf["r2"]
        else:
            san["realimentacao"] = []
            san["canais_realimentacao"] = []
            san["D_sem_realimentacao"] = base["D"]
            san["r2_sem_realimentacao"] = base["r2"]
        fam = [] if args.sem_familias else quebra_por_familia(X, Y, base["lam"])
        imprimir(caminho, base, san, fam, diag, reg)
        saida.append(para_json(caminho, base, san, fam, diag, reg))

    if len(saida) > 1:
        print("=" * 78)
        print("RESUMO — D por corrida (maior = mundo mais destilavel = mais facil)")
        print("=" * 78)
        print("  %-40s %7s %8s %9s %10s"
              % ("arquivo", "D", "D_s/real", "D_supra", "leitura"))
        for s in sorted(saida, key=lambda s: -s["D"]):
            nome = s["arquivo"]
            if len(nome) > 40:
                nome = "..." + nome[-37:]
            print("  %-40s %7.4f %8.4f %+9.3f %10s"
                  % (nome, s["D"], s["D_sem_realimentacao"], s["D_supra"], s["leitura"]))
        print("  D_s/real = D sem os canais que realimentam a acao anterior (o D do MUNDO)")
        print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"corridas": saida, "falhas": falhas}, f, indent=1, ensure_ascii=False)
        print("JSON gravado em %s" % os.path.abspath(args.json))

    return 0 if saida else 1


if __name__ == "__main__":
    sys.exit(main())
