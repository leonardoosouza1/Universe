//! Impact-analysis tool: given a symbol, lists EVERYTHING that depends on it
//! by querying the code graph in `.code-review-graph/graph.db`.
//!
//! # Usage
//!   cargo run --bin impact -- <symbol_name> [--depth N] [--graph /path/graph.db]
//!
//! # Examples
//!   cargo run --bin impact -- save_champion_dna
//!   cargo run --bin impact -- MorphoGenes --depth 3
//!   cargo run --bin impact -- DNA_FAST_SIZE --depth 2
//!
//! # How it works
//! Delegates the queries to `python3` (which has sqlite3 built-in), without depending on
//! `rusqlite` as a Cargo dependency. The transitive BFS is done in Rust.
//!
//! # Prerequisite
//! python3 on PATH (default on Ubuntu 22+).
//!
//! # About graph.db
//! The database may be outdated (old names like `cerebro.rs` → `brain.rs`).
//! Use it as a starting point; confirm in the code with grep.

use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::process::{self, Command};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: impact <symbol_name> [--depth N] [--graph /path/graph.db]");
        eprintln!("Examples:");
        eprintln!("  impact save_champion_dna");
        eprintln!("  impact MorphoGenes --depth 3");
        eprintln!("  impact DNA_FAST_SIZE");
        process::exit(1);
    }

    let symbol_needle = &args[1];
    let mut max_depth: usize = 3;
    let mut db_path = String::from("/home/leonardo/projects/LLM/.code-review-graph/graph.db");

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
                eprintln!("Argument ignored: {}", other);
                i += 1;
            }
        }
    }

    // Checks python3
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("Error: python3 not found on PATH.");
        process::exit(2);
    }

    println!("=== IMPACT ANALYSIS: '{}' ===", symbol_needle);
    println!("Database: {}", db_path);
    println!("Maximum depth: {}", max_depth);
    println!();

    // 1. Finds matching nodes.
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
        println!("No node found with '{}' in the graph.", symbol_needle);
        println!("Hint: graph.db may be outdated.");
        println!("Use: grep -rn '{}' src/ --include='*.rs'", symbol_needle);
        return;
    }

    println!("Nodes matching '{}':", symbol_needle);
    for row in &seed_nodes {
        let parts: Vec<&str> = row.splitn(3, '\t').collect();
        if parts.len() == 3 {
            println!(
                "  [{}] {} ({})",
                parts[2].trim(),
                parts[0],
                abbrev_path(parts[1])
            );
        }
    }
    println!();

    // 2. BFS over the transitive dependents.
    let seed_qnames: Vec<String> = seed_nodes
        .iter()
        .filter_map(|r| r.split('\t').next().map(|s| s.to_string()))
        .collect();

    let dependents = bfs_dependents(&db_path, &seed_qnames, max_depth);

    if dependents.is_empty() {
        println!("No dependents found.");
        println!("The symbol may be a leaf, or graph.db is outdated.");
        return;
    }

    // 3. Groups by edge kind.
    println!("Who is affected if '{}' changes:", symbol_needle);
    println!("{:-<70}", "");

    let mut by_kind: HashMap<String, Vec<(String, String, usize)>> = HashMap::new();
    for (qname, file, kind, depth) in &dependents {
        by_kind
            .entry(kind.clone())
            .or_default()
            .push((qname.clone(), file.clone(), *depth));
    }

    let kind_order = [
        "CALLS",
        "REFERENCES",
        "TESTED_BY",
        "CONTAINS",
        "IMPORTS_FROM",
        "INHERITS",
    ];
    let mut shown: Vec<String> = kind_order
        .iter()
        .map(|s| s.to_string())
        .filter(|k| by_kind.contains_key(k))
        .collect();
    for k in by_kind.keys() {
        if !shown.contains(k) {
            shown.push(k.clone());
        }
    }

    for kind in &shown {
        let items = &by_kind[kind];
        println!("\n  ── {} ({} dependents) ──", kind, items.len());
        let mut sorted = items.clone();
        sorted.sort_by_key(|(_, _, d)| *d);
        for (qname, file, depth) in &sorted {
            println!(
                "    [d{}] {} ({})",
                depth,
                short_qname(qname),
                abbrev_path(file)
            );
        }
    }

    println!();
    println!("Total unique dependents: {}", dependents.len());
    println!();
    println!("Legend:");
    println!("  [d1] depends DIRECTLY on the target symbol");
    println!("  [d2] depends on something that depends on the target");
    println!("  [dN] transitive reach N");
    println!();
    println!("WARNING: graph.db is generated statically — it may be outdated.");
    println!("Use 'grep -rn <symbol> src/ --include=*.rs' to confirm.");
}

// ── Transitive BFS ────────────────────────────────────────────────────────────

fn bfs_dependents(
    db: &str,
    seeds: &[String],
    max_depth: usize,
) -> Vec<(String, String, String, usize)> {
    let mut visited: HashSet<String> = seeds.iter().cloned().collect();
    let mut queue: VecDeque<(String, usize)> = seeds.iter().map(|q| (q.clone(), 0usize)).collect();
    let mut result: Vec<(String, String, String, usize)> = Vec::new();

    while let Some((target, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        // Who points to `target`?
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
            if parts.len() < 3 {
                continue;
            }
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

/// Runs SQL via python3 (sqlite3 built-in) and returns rows separated by '\t'.
fn py_query(db: &str, sql: &str) -> Vec<String> {
    // Inline Python script to run the query.
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

    let out = Command::new("python3").arg("-c").arg(&script).output();

    match out {
        Ok(o) => {
            if !o.stderr.is_empty() {
                let err = String::from_utf8_lossy(&o.stderr);
                eprintln!("py_query warning: {}", err.trim());
            }
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect()
        }
        Err(e) => {
            eprintln!("Error running python3: {}", e);
            vec![]
        }
    }
}

/// Escapes single quotes for SQL (doubles them).
fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

/// Converts a Rust string into a Python literal (triple quotes for safety).
fn python_str_literal(s: &str) -> String {
    // Uses triple quotes to avoid problems with newlines and single quotes in the SQL.
    format!("\"\"\"{}\"\"\"", s.replace("\"\"\"", "\\\"\\\"\\\""))
}

// ── Formatting helpers ────────────────────────────────────────────────────────

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
