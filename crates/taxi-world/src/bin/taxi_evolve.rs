//! Thin CLI entry point. The whole implementation lives in the library
//! (`taxi_world::taxi_evolve_cli`) — see that module's doc comment for why: a
//! `tests/*.rs` integration test can only link the crate's LIBRARY target, not
//! another binary target, and this task's acceptance test (`tests/
//! evolve_smoke.rs`) needs to run `taxi_evolve` in-process, not as a
//! subprocess.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match taxi_world::taxi_evolve_cli::run(&args) {
        Ok(saida) => eprintln!("gravado {}", saida.display()),
        Err(e) => {
            eprintln!("taxi_evolve: {e}");
            std::process::exit(1);
        }
    }
}
