# Neuromodulação que ajusta o cérebro E o corpo — projeto antes do código

Leonardo: *"podemos evoluir depois para neuromodulação que ajusta tanto o cérebro e o corpo,
como foco e concentração... mas só depois de passar nos testes e dar voltas completas,
mostrando que evoluiu"*.

Este documento é o projeto. Ele existe antes do código porque a série EXP10 já pagou caro
para descobrir **três formas de a neuromodulação não render nada**, e todas as três são
fáceis de repetir por distração.

---

## O que já sabemos, e que restringe o projeto

| achado | onde | o que impõe aqui |
|---|---|---|
| **gatilho cego custa −21,43%** | EXP10 passo 6 | o modulador TEM de ver o regime. Modulador que dispara por relógio ou por ruído é pior que não ter |
| **multiplicativo bate aditivo 16/16** | A3 | modular é `τ ← τ₀·exp(u)`, `alcance ← alcance₀·exp(u)`. Nunca somar |
| **U invertido no ganho** | Neuromod gatilho vs escala | ganho 0,5–1,0 rende +13 a +16%; ganho 8,0 rende −26,7%. O modulador obedece à mesma lei que ele modula |
| **teto G só é alto se houver CORPO** | Tese do corpo (confirmada, corr +0,970) | sem estados internos lentos acoplados, G≈0 e nenhuma implementação salva |
| **lei da moeda cruzada** | EXP10, 4 reestruturações | a ação protetora tem de custar em moeda DIFERENTE da que ela protege, senão vira decoração |
| **ablação com co-adaptação superestima** | Ablação e co-adaptação (+1735%) | medir a contribuição do modulador NÃO pode ser zerando `u` num cérebro co-adaptado |
| **G não traduz para fitness** | G não traduz (24,20% → +0,11%) | calcular G ANTES, mas não prometer o número: G mede estimativa, fitness mede decisão |

> A lição que atravessa todas: **calcular o teto antes de simular.** Foi o que fez a sessão
> inteira render, e foi o que faltava quando `≥2 voltas` ficou 15 gerações em 0,0%.

---

## O corpo já existe (e é por isso que aqui G pode não ser zero)

`racer.rs` já carrega estados internos lentos acoplados à computação:

| estado | o que é | por que serve de aferente |
|---|---|---|
| `fuel` | tanque, drenos de 1/s, 30/s e 60/s | escassez com dinâmica lenta |
| `hp` | dano acumulado, cicatriza devagar | histórico, não instante |
| `tire_temp` | carga térmica 0–1 | integra o estilo de pilotagem |
| `engine_power` | degradação mecânica 0,1–1 | irreversível dentro da corrida |
| `joy_streak` | tempo contínuo em flow (cap 5 s) | zera em "dor" — é um estado de humor |
| `proximity_sustain` | segundos raspando lateral | alerta SUSTENTADO, não evento |
| `slow_time` | segundos abaixo de 3 m/s | pré-morte |

Isto é exatamente a configuração que a Tese do Corpo prevê como tendo teto alto: **múltiplos
estados internos lentos, acoplados, com ótimos que variam com eles.** Nos experimentos onde
G deu 0 (MDP) ou 4,89% (mercado), o corpo não existia ou não estava acoplado.

**Predição registrada antes de medir:** o teto G aqui é maior que os 4,89% do mercado, porque
o número de estados corporais acoplados é maior. Se der ≈0, a Tese do Corpo leva um golpe
sério e isso vale mais que o experimento dar certo.

---

## O que a neuromodulação vai controlar

Dois alvos, um no cérebro e um no corpo — como Leonardo pediu.

### 1. FOCO (corpo/sensor): trocar ângulo por distância

`visao.rs` já expõe `alcance_m` e `fov_deg` em tempo de execução. Focar é:

```
fov     ← fov₀     · exp(−k·u)      estreita
alcance ← alcance₀ · exp(+k·u)      alonga
```

**Por que isto respeita a lei da moeda cruzada:** focar compra ANTECIPAÇÃO (ver a curva a
tempo de frear, o gargalo real medido: acima de 28 m/s o carro fica cego para o ponto de
frenagem) e custa CONSCIÊNCIA LATERAL (as paredes, que é outra moeda inteiramente — e
`proximity_sustain` mede a conta chegando). Sem esse custo cruzado, focar sempre seria
gratuito e o canal viraria decoração, como já aconteceu 4 vezes no EXP10.

O total de raios é constante: estreitar o FOV com as mesmas 32 colunas AUMENTA a resolução
angular. Foco é literalmente mais nitidez onde importa, menos campo. Não é um bônus.

### 2. CONCENTRAÇÃO (cérebro): mexer nas constantes de tempo

```
τ ← τ₀ · exp(u)
```

Já medido no Universe: **ter escalas de tempo variadas é selecionado** (espalhamento de τ
40× nos campeões contra 5,3× no controle aleatório), mas **TROCAR τ com o regime não pagou
(+0,11%)** naquele desenho. A diferença aqui é o corpo: lá o τ trocava com um regime externo
oculto; aqui troca com estado INTERNO, que é a condição que a Tese do Corpo diz ser
necessária.

---

## O aferente: o que o modulador enxerga

Gatilho cego custa −21,43%, então `u` é função dos estados corporais, não do relógio:

```
u = tanh( w · [fuel_norm, hp_norm, tire_temp, engine_power, joy_streak, proximity_sustain, v] )
```

`w` é EVOLUÍDO, não escrito à mão — é a diferença entre o priming (31,8%) e o córtex (96,5%).
Ganho inicial em 0,7, dentro da faixa 0,5–1,0 que rendeu +13 a +16%. **Nunca 8,0.**

---

## Como medir sem repetir os erros

1. **Não medir por ablação de `u` num cérebro co-adaptado.** `τ = τ₀·exp(u)` co-adapta: zerar
   `u` deixa τ ERRADO, não ausente, e o número mede desarranjo (+1735% num caso).
   Medir com **modulador CEGO** (mesma arquitetura, mesmo custo, aferente embaralhado) —
   controle que já mostrou o efeito real ser −21,43% quando cego.
2. **Calcular G antes.** Se o teto fechado der <1%, não vale simular: foi o que a linha do
   mercado ensinou gastando muito para descobrir.
3. **Reportar mediana ao lado de max**, e **voltas fracionárias ao lado da contagem inteira.**
   As três roupas do mesmo erro nesta sessão foram todas métricas escondendo a distribuição.
4. **Registrar a predição antes de rodar.** Foi o que salvou o câmbio (teto 0,36% previsto,
   0,7 ± 0,9% medido) de virar história.

---

## Portão

Leonardo condicionou: *"só depois de passar nos testes e dar voltas completas, mostrando que
evoluiu"*.

- ✅ testes passando
- ✅ voltas completas — **28% da população dá 2 voltas**, melhor carro 2,89 voltas
  (o `0,0%` anterior era o orçamento de 120 s, não o cérebro: `scripts/teto_voltas_exp.sh`)
- ⏳ **mostrando que evoluiu** — `scripts/treino_voltas.sh` com o relógio certo. É o que falta.

Enquanto o portão não fecha, este documento fica projeto. Implementar antes de medir a
evolução seria justamente pular o passo que o Leonardo pediu.
