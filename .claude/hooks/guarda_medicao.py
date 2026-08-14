#!/usr/bin/env python3
"""Guarda mecânica dos modos de falha documentados no CLAUDE.md deste projeto.

Não substitui julgamento — só mata os erros que são PURA MECÂNICA e que já custaram
caro aqui: substituição de texto que não aplica (produziu uma bateria inteira falsa e um
"25,29σ" inexistente) e campo novo no genoma sem atualizar o serializador (o campeão
perdia 1091 pesos do cérebro lento em silêncio).

Bloqueia com código 2; a mensagem no stderr volta como feedback.
"""
import json, re, sys

def main():
    try:
        ev = json.load(sys.stdin)
    except Exception:
        sys.exit(0)                       # entrada ilegível não é motivo para travar o fluxo

    nome = ev.get("tool_name", "")
    ent  = ev.get("tool_input", {}) or {}
    avisos = []

    if nome == "Bash":
        cmd = ent.get("command", "") or ""

        # POLÍTICA DE ARMAZENAMENTO (2026-08-13): dado pesado NUNCA no disco do sistema.
        # O disco chegou a 340 MB livres em 2026-08-11 e matou baterias em andamento; /tmp
        # ainda por cima some no reboot (perdeu um snapshot de refatoração). Só `--out` e
        # redirecionamentos de saída são barrados — ler de /tmp continua livre.
        if re.search(r"--out\s+/tmp/", cmd) or re.search(r"--saida\s+/tmp/", cmd):
            avisos.append(
                "ARMAZENAMENTO: `--out /tmp/...` é proibido neste projeto.\n"
                "  Saída de bateria/replay vai para /media/llm-data/universe-runs/<nome>/.\n"
                "  Ver CLAUDE.md, seção POLÍTICA DE ARMAZENAMENTO.")

        # Bateria longa sem declarar o teto da métrica (Gap 4). Não bloqueia corrida curta:
        # o alvo é a bateria que roda meia hora para descobrir que bateu no teto de ticks —
        # aconteceu com --ticks 1200 ("muro em 256 m" era o teto de 290 m) e de novo em
        # 2026-08-13 (mediana 3.189 m era exatamente 20.000 ticks ÷ 64 Hz × 10,2 m/s).
        m = re.search(r"--gens\s+(\d+)", cmd)
        if m and int(m.group(1)) >= 20 and "gpu_evolve" in cmd:
            if not re.search(r"--ticks\s+(\d+)", cmd):
                avisos.append(
                    "GAP 4 (CLAUDE.md): bateria de {} gerações sem `--ticks` explícito.\n"
                    "  Calcule o teto ANTES: metros_max = ticks/64 × velocidade_alcançável.\n"
                    "  Zero ou valor cravado = métrica no limite, não modelo falhando."
                    .format(m.group(1)))

        # Gap 6: str.replace sem verificar que casou.
        if re.search(r"\.replace\s*\(", cmd) and "python" in cmd:
            if not re.search(r"\bassert\b", cmd):
                avisos.append(
                    "GAP 6 (CLAUDE.md): `.replace(` num script Python SEM `assert`.\n"
                    "  Substituição que não casa imprime sucesso e não muda nada. Já produziu\n"
                    "  aqui uma bateria com 3 sementes idênticas e um 25,29σ falso.\n"
                    "  Conserto: `assert s.count(old) == 1` antes de cada replace.")

    if nome in ("Edit", "Write", "MultiEdit"):
        caminho = ent.get("file_path", "") or ""
        novo    = ent.get("new_string", "") or ent.get("content", "") or ""
        # DUPLICAÇÃO (2026-08-13): o projeto tem histórico de equação copiada que DIVERGIU
        # em silêncio — a codificação da retina existia em 4 lugares e um deles usava rampa
        # linear onde os outros usam raiz, entregando parede mais longe do que ela está.
        # Mesma família: CfC de percepção copiado em 3 binários, `make_physics_state` em 3,
        # `uniform_entry` em 14. Quando um arquivo novo reimplementa algo que já existe,
        # avisar antes de a cópia nascer.
        for termo, canonico in (
            ("fn legacy_encoding", "src/gpu/mod.rs (única fonte da codificação da retina)"),
            ("struct CfcState",    "crate iara-cfc (equação canônica do CfC de percepção)"),
            ("fn participation_ratio", "src/math/linalg.rs"),
            ("fn spectral_radius",     "src/math/linalg.rs"),
            ("fn make_physics_state",  "src/gpu/physics.rs"),
        ):
            if termo in novo and canonico.split(" ")[0] not in caminho:
                avisos.append(
                    "DUPLICAÇÃO: `{}` já existe em {}.\n"
                    "  Importe do canônico em vez de reimplementar. Cópia que diverge em\n"
                    "  silêncio já custou dias de investigação neste projeto."
                    .format(termo, canonico))

        # COMENTÁRIO MENTIROSO: a onda sensorial tem 192 canais desde que foi reduzida, mas
        # comentários com 223 sobreviveram e enganaram duas auditorias, que quase
        # reportaram um "bug crítico de stride" inexistente. Comentário desatualizado neste
        # repositório não é ruído: é armadilha.
        if re.search(r"\b223\b", novo) and caminho.endswith((".rs", ".wgsl")):
            avisos.append(
                "COMENTÁRIO SUSPEITO: `223` é o tamanho ANTIGO da onda sensorial (hoje 192).\n"
                "  Se for comentário, está mentindo. Se for constante, está errada.\n"
                "  Confira com config.rs::SENSORY_WAVE_SIZE antes de gravar.")

        # Gap 7: campo novo no Genome sem tocar no disco.
        if caminho.endswith("gpu/evolution.rs") and "pub struct Genome" in novo:
            avisos.append(
                "GAP 7 (CLAUDE.md): mexeu na struct `Genome`.\n"
                "  `save_genome`/`load_genome` em src/bin/gpu_evolve.rs precisam do campo novo,\n"
                "  senão o campeão o perde em silêncio a cada --carregar.\n"
                "  Rodar depois: cargo test --release --bin gpu_evolve")

    if avisos:
        print("\n\n".join(avisos), file=sys.stderr)
        sys.exit(2)
    sys.exit(0)

main()
