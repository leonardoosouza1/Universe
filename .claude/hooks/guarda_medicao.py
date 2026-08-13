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
