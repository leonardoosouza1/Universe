//! Ferramenta de análise de impacto: dado um símbolo, lista TUDO que depende dele
//! consultando o grafo de código em `.code-review-graph/graph.db`.
//!
//! # Uso
//!   cargo run --bin impact -- <nome_do_simbolo> [--depth N] [--graph /caminho/graph.db]
//!
//! # Exemplos
//!   cargo run --bin impact -- save_champion_dna
//!   cargo run --bin impact -- MorphoGenes --depth 3
//!   cargo run --bin impact -- DNA_FAST_SIZE --depth 2
//!
//! # Como funciona
//! Delega as queries ao `python3` (que tem sqlite3 built-in), sem depender de
//! `rusqlite` como dependência Cargo. O BFS transitivo é feito em Rust.
//!
//! # Pré-requisito
//! python3 no PATH (padrão no Ubuntu 22+).
//!
//! # Sobre o graph.db
//! O banco pode estar desatualizado (nomes antigos como `cerebro.rs` → `brain.rs`).
//! Use como ponto de partida; confirme no código com grep.

use std::env;
use std::process::{self, Command};
use std::collections::{HashMap, HashSet, VecDeque};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Uso: impact <nome_do_simbolo> [--depth N] [--graph /caminho/graph.db]");
        eprintln!("Exemplos:");
        eprintln!("  impact save_champion_dna");
        eprintln!("  impact MorphoGenes --depth 3");
        eprintln!("  impact DNA_FAST_SIZE");
        process::exit(1);
    }

    let symbol_needle = &args[1];
    let mut max_depth: usize = 3;
    let mut db_path = String::from(
        "/home/leonardo/projects/LLM/.code-review-graph/graph.db"
    );

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--depth" if i + 1 < args.len() => {
                max_depth = args[i + 1].parse().unwrap_or(3);
                i += 2;
            }
            "--graph" if i + 1 < args.len() => {
                db_path = args[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("Argumento ignorado: {}", other);
                i += 1;
            }
        }
    }

    // Verifica python3
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("Erro: python3 não encontrado no PATH.");
        process::exit(2);
    }

    println!("=== ANÁLISE DE IMPACTO: '{}' ===", symbol_needle);
    println!("Banco: {}", db_path);
    println!("Profundidade máxima: {}", max_depth);
    println!();

    // 1. Encontra nós correspondentes.
    let seed_nodes = py_query(
        &db_path,
        &format!(
            "SELECT qualified_name, COALESCE(file_path,''), COALESCE(kind,'?') \
             FROM nodes \
             WHERE qualified_name LIKE '%{n}%' OR name LIKE '%{n}%' \
             ORDER BY qualified_name LIMIT 50",
            n = sql_escape(symbol_needle)
        ),
    );

    if seed_nodes.is_empty() {
        println!("Nenhum nó encontrado com '{}' no grafo.", symbol_needle);
        println!("Dica: graph.db pode estar desatualizado.");
        println!("Use: grep -rn '{}' src/ --include='*.rs'", symbol_needle);
        return;
    }

    println!("Nós correspondentes a '{}':", symbol_needle);
    for row in &seed_nodes {
        let parts: Vec<&str> = row.splitn(3, '\t').collect();
        if parts.len() == 3 {
            println!("  [{}] {} ({})", parts[2].trim(), parts[0], abbrev_path(parts[1]));
        }
    }
    println!();

    // 2. BFS nos dependentes transitivos.
    let seed_qnames: Vec<String> = seed_nodes.iter()
        .filter_map(|r| r.split('\t').next().map(|s| s.to_string()))
        .collect();

    let dependents = bfs_dependents(&db_path, &seed_qnames, max_depth);

    if dependents.is_empty() {
        println!("Nenhum dependente encontrado.");
        println!("O símbolo pode ser leaf ou o graph.db está desatualizado.");
        return;
    }

    // 3. Agrupa por tipo de aresta.
    println!("Quem é afetado se '{}' mudar:", symbol_needle);
    println!("{:-<70}", "");

    let mut by_kind: HashMap<String, Vec<(String, String, usize)>> = HashMap::new();
    for (qname, file, kind, depth) in &dependents {
        by_kind.entry(kind.clone()).or_default()
               .push((qname.clone(), file.clone(), *depth));
    }

    let kind_order = [
        "CALLS", "REFERENCES", "TESTED_BY", "CONTAINS", "IMPORTS_FROM", "INHERITS",
    ];
    let mut shown: Vec<String> = kind_order.iter()
        .map(|s| s.to_string())
        .filter(|k| by_kind.contains_key(k))
        .collect();
    for k in by_kind.keys() {
        if !shown.contains(k) { shown.push(k.clone()); }
    }

    for kind in &shown {
        let items = &by_kind[kind];
        println!("\n  ── {} ({} dependentes) ──", kind, items.len());
        let mut sorted = items.clone();
        sorted.sort_by_key(|(_, _, d)| *d);
        for (qname, file, depth) in &sorted {
            println!("    [d{}] {} ({})", depth, short_qname(qname), abbrev_path(file));
        }
    }

    println!();
    println!("Total de dependentes únicos: {}", dependents.len());
    println!();
    println!("Legenda:");
    println!("  [d1] depende DIRETAMENTE do símbolo alvo");
    println!("  [d2] depende de algo que depende do alvo");
    println!("  [dN] alcance transitivo N");
    println!();
    println!("AVISO: graph.db gerado estaticamente — pode estar desatualizado.");
    println!("Use 'grep -rn <simbolo> src/ --include=*.rs' para confirmar.");
}

// ── BFS transitivo ────────────────────────────────────────────────────────────

fn bfs_dependents(
    db: &str,
    seeds: &[String],
    max_depth: usize,
) -> Vec<(String, String, String, usize)> {
    let mut visited: HashSet<String> = seeds.iter().cloned().collect();
    let mut queue: VecDeque<(String, usize)> =
        seeds.iter().map(|q| (q.clone(), 0usize)).collect();
    let mut result: Vec<(String, String, String, usize)> = Vec::new();

    while let Some((target, depth)) = queue.pop_front() {
        if depth >= max_depth { continue; }

        // Quem aponta para `target`?
        let sql = format!(
            "SELECT e.source_qualified, COALESCE(n.file_path,''), e.kind \
             FROM edges e \
             LEFT JOIN nodes n ON n.qualified_name = e.source_qualified \
             WHERE e.target_qualified = '{t}' OR e.target_qualified LIKE '{t}%' \
             LIMIT 300",
            t = sql_escape(&target),
        );
        let rows = py_query(db, &sql);

        for row in rows {
            let parts: Vec<&str> = row.splitn(3, '\t').collect();
            if parts.len() < 3 { continue; }
            let src = parts[0].to_string();
            let file = parts[1].to_string();
            let kind = parts[2].trim().to_string();

            if !visited.contains(&src) {
                visited.insert(src.clone());
                result.push((src.clone(), file, kind, depth + 1));
                queue.push_back((src, depth + 1));
            }
        }
    }

    result
}

// ── Python sqlite3 bridge ─────────────────────────────────────────────────────

/// Executa SQL via python3 (sqlite3 built-in) e retorna linhas separadas por '\t'.
fn py_query(db: &str, sql: &str) -> Vec<String> {
    // Python script inline para fazer a query.
    let script = format!(
        r#"
import sqlite3, sys
try:
    c = sqlite3.connect('file:{}?mode=ro', uri=True)
    for row in c.execute({}):
        print('\t'.join(str(x) if x is not None else '' for x in row))
    c.close()
except Exception as e:
    sys.stderr.write(str(e) + '\n')
"#,
        db,
        python_str_literal(sql),
    );

    let out = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output();

    match out {
        Ok(o) => {
            if !o.stderr.is_empty() {
                let err = String::from_utf8_lossy(&o.stderr);
                eprintln!("Aviso py_query: {}", err.trim());
            }
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect()
        }
        Err(e) => {
            eprintln!("Erro ao executar python3: {}", e);
            vec![]
        }
    }
}

/// Escapa aspas simples para SQL (dobra-as).
fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

/// Converte string Rust para literal Python (aspas triplas para segurança).
fn python_str_literal(s: &str) -> String {
    // Usa aspas triplas para evitar problemas com newlines e aspas simples no SQL.
    format!("\"\"\"{}\"\"\"", s.replace("\"\"\"", "\\\"\\\"\\\""))
}

// ── Helpers de formatação ─────────────────────────────────────────────────────

fn abbrev_path(path: &str) -> String {
    if path.contains("/Universe/src/") {
        path.split("/Universe/src/")
            .last()
            .map(|s| format!("src/{}", s))
            .unwrap_or_else(|| path.to_string())
    } else if path.len() > 60 {
        format!("...{}", &path[path.len() - 57..])
    } else {
        path.to_string()
    }
}

fn short_qname(qname: &str) -> &str {
    if let Some(pos) = qname.rfind('/') {
        &qname[pos + 1..]
    } else {
        qname
    }
}
