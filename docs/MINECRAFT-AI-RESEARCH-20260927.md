# Pesquisa de IA no Minecraft — continuação no notebook

**27/09/2026 — pacote documental, sem alteração de runtime ou deploy.**
Este arquivo é o ponto de entrada para Codex/Claude analisar a pesquisa junto do
código e dos episódios atuais no notebook. O pacote está nos dois repositórios;
os arquivos de pesquisa e o relatório de origem são idênticos em ambos.

## Leia nesta ordem

1. [Síntese, diagnóstico e bateria proposta](research/MINECRAFT-AI-SYNTHESIS-20260927.md).
2. [Métodos de aprendizado, versões e requisitos](research/MINECRAFT-AI-LEARNING-20260927.md).
3. [Bots, LLMs, bibliotecas de habilidades e coordenação](research/MINECRAFT-AI-SYMBOLIC-20260927.md).
4. [Relatório do notebook usado como evidência](research/VILA-VIVA-REPORT-20260927.md).

A síntese prevalece sobre sugestões preliminares dos levantamentos. Ela separa
fatos reportados, leitura estática das bibliotecas e hipóteses ainda não testadas.
Há links para artigos, repositórios, documentação Paper e código histórico do
Citizens compatível com a dependência Minecraft 1.21.4.

## Repositórios e responsabilidades

| Repositório privado | Branch desta publicação | Papel |
|---|---|---|
| [Universe](https://github.com/leonardoosouza1/Universe/tree/organism/school-20260922) | `organism/school-20260922` | Runtime/política canônica e investigação científica |
| [vila-viva-server](https://github.com/leonardoosouza1/vila-viva-server/tree/main) | `main` | Paper/plugin, painel, deploy e cópias dos runtimes usados no notebook |

O relatório cita `Universe 06eec1e` e `vila-viva-server ba0e492`, feitos no notebook
e ainda não publicados no momento daquele relatório. Esses commits **não foram
inspecionados no desktop**. O pacote documental não substitui suas alterações.
As referências locais do desktop antes desta publicação eram `74d12e1` e
`1e8c8bc`, respectivamente; confirme o estado efetivo antes de qualquer integração.

## Atualização preservando o trabalho local

No diretório correto de cada clone, confira primeiro `git status --short --branch`.
Se houver alterações não commitadas, preserve/revise esse trabalho antes do pull;
não use reset, clean ou force-push para fazer a documentação entrar.

Com a árvore limpa e na branch correspondente, os comandos abaixo permitem merge
com os commits locais do notebook, sem exigir fast-forward:

```bash
# No clone Universe, na branch organism/school-20260922:
git pull --no-rebase origin organism/school-20260922

# No clone vila-viva-server, na branch main:
git pull --no-rebase origin main
```

Se houver conflito, preserve a implementação atual e integre os documentos.
O README recebeu apenas um link de navegação. O pacote não altera defaults de
TPS, modelos aprovados, dados, serviços ou configuração.

## Pedido para o agente que vai continuar

> Leia este pacote e os AGENTS.md locais. Confira os commits, configurações e
> episódios reais do notebook. Reconstrua um comando de aldeão de ponta a ponta:
> observação → política → caminho aplicado → processamento vanilla → velocidade
> → deslocamento → resultado. Determine o primeiro elo divergente antes de editar.
> Verifique se a hipótese de concorrência com o Brain é sustentada pelo trace;
> não a trate como causa já provada. Avalie os gates M0/M1 da síntese antes de
> propor mais treino. Entregue diagnóstico com arquivo/símbolo, UUID/ticks,
> medições, hipóteses rivais e o menor próximo teste discriminante.

Primeiro alvo técnico: uma ação curta que mova o aldeão até um alvo livre e devolva
controle à rotina com segurança. `moveTo=true` não certifica chegada. A GRU shadow
prevê; não há evidência neste pacote de política aprendida superior ao vanilla.

A CPU está autorizada para o runtime Minecraft do notebook; isso não autoriza
executar o legado taxi/organism na CPU. O relatório registra **80 TPS/power-saver**
como preferência operacional posterior: não restaurar automaticamente o antigo
padrão 20/400 TPS apenas porque algum handoff anterior o menciona. Verificar a
preferência vigente e o estado real antes de qualquer operação.

Não há nesta publicação nova execução da bateria, vídeo de sucesso, benchmark de
hardware ou promoção de pesos. Links externos são referências técnicas; versões,
permissões e efeitos precisam ser validados antes de adotar suas implementações.
