use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};

const INPUTS: usize = 23;
const HIDDEN: usize = 64;
const MAX_LINE_BYTES: usize = 1_000_000;
const FEATURES: [&str; INPUTS] = [
    "rx",
    "ry",
    "rz",
    "vx",
    "vy",
    "vz",
    "yaw_s",
    "yaw_c",
    "health",
    "food",
    "on_ground",
    "in_water",
    "baby",
    "sleeping",
    "light_sky",
    "light_block",
    "indoors",
    "tod_s",
    "tod_c",
    "p_farmer",
    "p_none",
    "p_nitwit",
    "p_other",
];

#[derive(Deserialize)]
struct Model {
    format: String,
    schema: String,
    contract: Contract,
    architecture: Architecture,
    weights: Weights,
}

#[derive(Deserialize)]
struct Contract {
    features: Vec<String>,
    classes: Vec<String>,
    world_clock: bool,
    profession: String,
    frame_ticks: usize,
}

#[derive(Deserialize)]
struct Architecture {
    input_dim: usize,
    hidden_dim: usize,
    move_dim: usize,
    sleep_dim: usize,
}

#[derive(Deserialize)]
struct Weights {
    inp_weight: Vec<Vec<f32>>,
    inp_bias: Vec<f32>,
    gru_weight_ih: Vec<Vec<f32>>,
    gru_weight_hh: Vec<Vec<f32>>,
    gru_bias_ih: Vec<f32>,
    gru_bias_hh: Vec<f32>,
    act_weight: Vec<Vec<f32>>,
    act_bias: Vec<f32>,
    move_weight: Vec<Vec<f32>>,
    move_bias: Vec<f32>,
    sleep_weight: Vec<Vec<f32>>,
    sleep_bias: Vec<f32>,
}

#[derive(Deserialize)]
struct Request {
    inputs: Vec<f32>,
    hidden: Option<Vec<f32>>,
}

#[derive(Serialize)]
struct Response {
    embed: Vec<f32>,
    hidden: Vec<f32>,
    activity: Vec<f32>,
    r#move: Vec<f32>,
    sleep: f32,
}

fn finite(x: f32) -> bool {
    x.is_finite()
}
fn vector(v: &[f32], n: usize, name: &str) -> Result<(), String> {
    if v.len() != n {
        return Err(format!("{name} length must be {n}, got {}", v.len()));
    }
    if !v.iter().all(|x| finite(*x)) {
        return Err(format!("{name} contains a non-finite value"));
    }
    Ok(())
}
fn matrix(m: &[Vec<f32>], rows: usize, cols: usize, name: &str) -> Result<(), String> {
    if m.len() != rows {
        return Err(format!("{name} rows must be {rows}, got {}", m.len()));
    }
    for row in m {
        vector(row, cols, name)?;
    }
    Ok(())
}
fn validate(model: &Model) -> Result<(), String> {
    if model.format != "vila-shadow-cpu/v1" || model.schema != "vila-shadow-cpu/v1" {
        return Err("unsupported model format or schema".into());
    }
    let c = &model.contract;
    if c.features.len() != INPUTS
        || !c
            .features
            .iter()
            .map(String::as_str)
            .eq(FEATURES.iter().copied())
    {
        return Err("model feature order does not match the Vila Viva contract".into());
    }
    if c.classes.is_empty()
        || c.classes.len() > 128
        || c.classes.iter().any(|x| x.is_empty())
        || c.classes.iter().collect::<HashSet<_>>().len() != c.classes.len()
    {
        return Err("classes must contain 1..128 nonempty values".into());
    }
    if !c.world_clock || c.profession != "per_row" || c.frame_ticks != 20 {
        return Err(
            "model contract requires world_clock=true, profession=per_row, frame_ticks=20".into(),
        );
    }
    let a = &model.architecture;
    if a.input_dim != INPUTS || a.hidden_dim != HIDDEN || a.move_dim != 2 || a.sleep_dim != 1 {
        return Err("unsupported model dimensions".into());
    }
    let w = &model.weights;
    let classes = c.classes.len();
    matrix(&w.inp_weight, HIDDEN, INPUTS, "inp_weight")?;
    vector(&w.inp_bias, HIDDEN, "inp_bias")?;
    matrix(&w.gru_weight_ih, 3 * HIDDEN, HIDDEN, "gru_weight_ih")?;
    matrix(&w.gru_weight_hh, 3 * HIDDEN, HIDDEN, "gru_weight_hh")?;
    vector(&w.gru_bias_ih, 3 * HIDDEN, "gru_bias_ih")?;
    vector(&w.gru_bias_hh, 3 * HIDDEN, "gru_bias_hh")?;
    matrix(&w.act_weight, classes, HIDDEN, "act_weight")?;
    vector(&w.act_bias, classes, "act_bias")?;
    matrix(&w.move_weight, 2, HIDDEN, "move_weight")?;
    vector(&w.move_bias, 2, "move_bias")?;
    matrix(&w.sleep_weight, 1, HIDDEN, "sleep_weight")?;
    vector(&w.sleep_bias, 1, "sleep_bias")?;
    Ok(())
}
fn dot(row: &[f32], input: &[f32], bias: f32) -> f32 {
    row.iter().zip(input).fold(bias, |s, (a, b)| s + a * b)
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
fn linear(weight: &[Vec<f32>], bias: &[f32], input: &[f32]) -> Vec<f32> {
    weight
        .iter()
        .zip(bias)
        .map(|(r, b)| dot(r, input, *b))
        .collect()
}
fn infer(model: &Model, request: Request) -> Result<Response, String> {
    vector(&request.inputs, INPUTS, "inputs")?;
    let previous = request.hidden.unwrap_or_else(|| vec![0.0; HIDDEN]);
    vector(&previous, HIDDEN, "hidden")?;
    let w = &model.weights;
    let embed: Vec<f32> = linear(&w.inp_weight, &w.inp_bias, &request.inputs)
        .into_iter()
        .map(f32::tanh)
        .collect();
    let mut reset = vec![0.0; HIDDEN];
    let mut update = vec![0.0; HIDDEN];
    let mut candidate = vec![0.0; HIDDEN];
    for i in 0..HIDDEN {
        reset[i] = sigmoid(
            dot(&w.gru_weight_ih[i], &embed, w.gru_bias_ih[i])
                + dot(&w.gru_weight_hh[i], &previous, w.gru_bias_hh[i]),
        );
        update[i] = sigmoid(
            dot(
                &w.gru_weight_ih[HIDDEN + i],
                &embed,
                w.gru_bias_ih[HIDDEN + i],
            ) + dot(
                &w.gru_weight_hh[HIDDEN + i],
                &previous,
                w.gru_bias_hh[HIDDEN + i],
            ),
        );
        candidate[i] = (dot(
            &w.gru_weight_ih[2 * HIDDEN + i],
            &embed,
            w.gru_bias_ih[2 * HIDDEN + i],
        ) + reset[i]
            * dot(
                &w.gru_weight_hh[2 * HIDDEN + i],
                &previous,
                w.gru_bias_hh[2 * HIDDEN + i],
            ))
        .tanh();
    }
    let hidden: Vec<f32> = (0..HIDDEN)
        .map(|i| (1.0 - update[i]) * candidate[i] + update[i] * previous[i])
        .collect();
    let logits = linear(&w.act_weight, &w.act_bias, &hidden);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (*x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let activity: Vec<f32> = exps.iter().map(|x| x / sum).collect();
    let move_out = linear(&w.move_weight, &w.move_bias, &hidden);
    let sleep = sigmoid(dot(&w.sleep_weight[0], &hidden, w.sleep_bias[0]));
    if !embed
        .iter()
        .chain(&hidden)
        .chain(&activity)
        .chain(&move_out)
        .all(|x| finite(*x))
        || !finite(sleep)
    {
        return Err("inference produced a non-finite value".into());
    }
    Ok(Response {
        embed,
        hidden,
        activity,
        r#move: move_out,
        sleep,
    })
}
fn write_json<T: Serialize>(out: &mut impl Write, value: &T) {
    let _ = serde_json::to_writer(&mut *out, value);
    let _ = writeln!(out);
    let _ = out.flush();
}
fn main() {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: vila-shadow-cpu MODEL_JSON");
            std::process::exit(2);
        }
    };
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read model {path}: {e}");
        std::process::exit(2)
    });
    let model: Model = serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("invalid model JSON: {e}");
        std::process::exit(2)
    });
    if let Err(e) = validate(&model) {
        eprintln!("invalid model: {e}");
        std::process::exit(2);
    }
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(v) => v,
            Err(e) => {
                write_json(
                    &mut stdout,
                    &serde_json::json!({"error": format!("stdin read error: {e}")}),
                );
                continue;
            }
        };
        if line.len() > MAX_LINE_BYTES {
            write_json(
                &mut stdout,
                &serde_json::json!({"error": "request exceeds 1000000 bytes"}),
            );
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => infer(&model, request).map_err(|e| e),
            Err(e) => Err(format!("invalid request JSON: {e}")),
        };
        match response {
            Ok(v) => write_json(&mut stdout, &v),
            Err(e) => write_json(&mut stdout, &serde_json::json!({"error": e})),
        }
    }
}
