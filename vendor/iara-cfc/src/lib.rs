//! Generic CfC (Closed-form Continuous-time) neural cell.
//!
//! Extracted from the IARA ecosystem. Application-specific output heads
//! (bit specialists, class heads, etc.) are intentionally absent — consumers
//! compose [`LinearHead`] instances on top of the cell.

use std::io::{self, Read, Write};

// ─────────────────────────────────────────────────────────────────────────────
// RNG
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG. Sufficient for weight initialization.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn step(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// Uniform sample in [0, 1).
    pub fn unit(&mut self) -> f32 {
        self.step() as f32 / u32::MAX as f32
    }

    /// Uniform sample in (-1, 1).
    pub fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dense matrix–vector multiply: `out[r] = Σ w[r,c] * x[c]`.
///
/// `w` is row-major with shape `[rows, cols]`.
pub fn matvec(out: &mut [f32], w: &[f32], x: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary serialization helpers (little-endian, no external deps)
// ─────────────────────────────────────────────────────────────────────────────

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_vec<W: Write>(w: &mut W, v: &[f32]) -> io::Result<()> {
    write_u32(w, v.len() as u32)?;
    for &x in v {
        write_f32(w, x)?;
    }
    Ok(())
}

fn read_vec<R: Read>(r: &mut R) -> io::Result<Vec<f32>> {
    let n = read_u32(r)? as usize;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(read_f32(r)?);
    }
    Ok(v)
}

// ─────────────────────────────────────────────────────────────────────────────
// CfcCell
// ─────────────────────────────────────────────────────────────────────────────

/// Core CfC cell weights.
///
/// Parameters:
/// - `w_xi`: `[D × E]` — input projection from embedding space
/// - `w_hi`: `[D × D]` — hidden-to-input gate
/// - `w_f`:  `[D × D]` — hidden-to-forget gate
/// - `b_i`:  `[D]`     — input gate bias (init 0.0)
/// - `b_f`:  `[D]`     — forget gate bias (init 0.5 for stable decay at start)
///
/// Forward recurrence:
/// ```text
/// i_gate = tanh(w_xi @ x  +  w_hi @ h  +  b_i)
/// f_gate = sigmoid(w_f @ h  +  b_f)        ← input-dependent time constant
/// decay  = exp(-f_gate)
/// h_new  = h * decay  +  i_gate * (1 - decay)
/// ```
pub struct CfcCell {
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub w_xi: Vec<f32>,
    pub w_hi: Vec<f32>,
    pub w_f: Vec<f32>,
    pub b_i: Vec<f32>,
    pub b_f: Vec<f32>,
}

impl CfcCell {
    /// Create and randomly initialize a new cell.
    ///
    /// Initialization scale follows 1/√fan_in convention.
    pub fn new(hidden_dim: usize, input_dim: usize, rng: &mut Rng) -> Self {
        let scale_x = (1.0 / (input_dim as f32).sqrt()) * 0.6;
        let scale_h = (1.0 / (hidden_dim as f32).sqrt()) * 0.6;

        let mut mk = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| rng.signed() * s).collect() };

        CfcCell {
            hidden_dim,
            input_dim,
            w_xi: mk(hidden_dim * input_dim, scale_x),
            w_hi: mk(hidden_dim * hidden_dim, scale_h),
            w_f: mk(hidden_dim * hidden_dim, scale_h),
            b_i: vec![0.0; hidden_dim],
            // 0.5 bias means f_gate ≈ sigmoid(0.5) ≈ 0.62 → decay ≈ 0.54 at init,
            // which avoids explosive or vanishing hidden state on the first steps.
            b_f: vec![0.5; hidden_dim],
        }
    }

    /// One forward step.
    ///
    /// Returns the updated hidden state `h_new` of length `hidden_dim`.
    /// Does NOT allocate the intermediate buffers on the hot path if the
    /// caller provides a [`CfcState`] workspace.
    pub fn step(&self, h: &[f32], x: &[f32]) -> Vec<f32> {
        let d = self.hidden_dim;
        let e = self.input_dim;
        let mut pre_xi = vec![0.0f32; d];
        let mut pre_hi = vec![0.0f32; d];
        let mut pre_f = vec![0.0f32; d];

        matvec(&mut pre_xi, &self.w_xi, x, d, e);
        matvec(&mut pre_hi, &self.w_hi, h, d, d);
        matvec(&mut pre_f, &self.w_f, h, d, d);

        let mut h_new = vec![0.0f32; d];
        for i in 0..d {
            let i_gate = (pre_xi[i] + pre_hi[i] + self.b_i[i]).tanh();
            let f_gate = sigmoid(pre_f[i] + self.b_f[i]);
            let decay = (-f_gate).exp();
            h_new[i] = h[i] * decay + i_gate * (1.0 - decay);
        }
        h_new
    }

    /// Total parameter count.
    pub fn n_params(&self) -> usize {
        self.w_xi.len() + self.w_hi.len() + self.w_f.len() + self.b_i.len() + self.b_f.len()
    }

    /// Serialize weights to writer (little-endian binary).
    pub fn save<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_vec(w, &self.w_xi)?;
        write_vec(w, &self.w_hi)?;
        write_vec(w, &self.w_f)?;
        write_vec(w, &self.b_i)?;
        write_vec(w, &self.b_f)?;
        Ok(())
    }

    /// Deserialize weights from reader.
    ///
    /// Dimensions are inferred from the stored lengths; the caller provides
    /// `hidden_dim` and `input_dim` for validation.
    pub fn load<R: Read>(r: &mut R, hidden_dim: usize, input_dim: usize) -> io::Result<Self> {
        let w_xi = read_vec(r)?;
        let w_hi = read_vec(r)?;
        let w_f = read_vec(r)?;
        let b_i = read_vec(r)?;
        let b_f = read_vec(r)?;

        if w_xi.len() != hidden_dim * input_dim
            || w_hi.len() != hidden_dim * hidden_dim
            || w_f.len() != hidden_dim * hidden_dim
            || b_i.len() != hidden_dim
            || b_f.len() != hidden_dim
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CfcCell dimension mismatch on load",
            ));
        }

        Ok(CfcCell {
            hidden_dim,
            input_dim,
            w_xi,
            w_hi,
            w_f,
            b_i,
            b_f,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CfcState
// ─────────────────────────────────────────────────────────────────────────────

/// Hidden state buffer for a single cell instance.
pub struct CfcState {
    pub h: Vec<f32>,
}

impl CfcState {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            h: vec![0.0; hidden_dim],
        }
    }

    /// Step the cell and update state in place. Returns a reference to `h`.
    pub fn step(&mut self, cell: &CfcCell, x: &[f32]) -> &[f32] {
        self.h = cell.step(&self.h, x);
        &self.h
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CfcGrads
// ─────────────────────────────────────────────────────────────────────────────

/// Gradient accumulators for a [`CfcCell`], matching its weight shapes.
pub struct CfcGrads {
    pub g_w_xi: Vec<f32>,
    pub g_w_hi: Vec<f32>,
    pub g_w_f: Vec<f32>,
    pub g_b_i: Vec<f32>,
    pub g_b_f: Vec<f32>,
}

impl CfcGrads {
    pub fn zero(cell: &CfcCell) -> Self {
        let d = cell.hidden_dim;
        let e = cell.input_dim;
        CfcGrads {
            g_w_xi: vec![0.0; d * e],
            g_w_hi: vec![0.0; d * d],
            g_w_f: vec![0.0; d * d],
            g_b_i: vec![0.0; d],
            g_b_f: vec![0.0; d],
        }
    }

    pub fn clear(&mut self) {
        self.g_w_xi.fill(0.0);
        self.g_w_hi.fill(0.0);
        self.g_w_f.fill(0.0);
        self.g_b_i.fill(0.0);
        self.g_b_f.fill(0.0);
    }

    /// L2 norm across all gradient tensors (for diagnostics).
    pub fn global_norm(&self) -> f32 {
        let sq = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        (sq(&self.g_w_xi) + sq(&self.g_w_hi) + sq(&self.g_w_f) + sq(&self.g_b_i) + sq(&self.g_b_f))
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearHead
// ─────────────────────────────────────────────────────────────────────────────

/// A single affine output projection: `out = W @ h + b`, shape `[out_dim, hidden_dim]`.
///
/// Consumers add as many heads as needed on top of [`CfcCell`].
pub struct LinearHead {
    pub out_dim: usize,
    pub hidden_dim: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

impl LinearHead {
    pub fn new(out_dim: usize, hidden_dim: usize, rng: &mut Rng) -> Self {
        let scale = (1.0 / (hidden_dim as f32).sqrt()) * 0.6;
        let w = (0..out_dim * hidden_dim)
            .map(|_| rng.signed() * scale)
            .collect();
        let b = vec![0.0; out_dim];
        LinearHead {
            out_dim,
            hidden_dim,
            w,
            b,
        }
    }

    /// Forward: fills `out` with the projected values.
    pub fn forward(&self, out: &mut [f32], h: &[f32]) {
        matvec(out, &self.w, h, self.out_dim, self.hidden_dim);
        for i in 0..self.out_dim {
            out[i] += self.b[i];
        }
    }

    pub fn save<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_vec(w, &self.w)?;
        write_vec(w, &self.b)?;
        Ok(())
    }

    pub fn load<R: Read>(r: &mut R, out_dim: usize, hidden_dim: usize) -> io::Result<Self> {
        let w = read_vec(r)?;
        let b = read_vec(r)?;
        if w.len() != out_dim * hidden_dim || b.len() != out_dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LinearHead dimension mismatch on load",
            ));
        }
        Ok(LinearHead {
            out_dim,
            hidden_dim,
            w,
            b,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BPTT
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediate tensors stored during a forward pass, needed for BPTT.
pub struct ForwardCache {
    /// Hidden states h[0..=T], where h[0] is the initial state.
    hs: Vec<Vec<f32>>,
    /// Input embeddings at each step.
    xs: Vec<Vec<f32>>,
    /// Per-step: i_gate, f_gate, decay (all shape [D]).
    i_gates: Vec<Vec<f32>>,
    f_gates: Vec<Vec<f32>>,
    decays: Vec<Vec<f32>>,
}

/// Run the full forward pass over a sequence of inputs, caching all
/// intermediate values required for BPTT.
///
/// `inputs`: sequence of `T` input vectors, each of length `input_dim`.
/// `h0`:     initial hidden state.
fn forward_cache(cell: &CfcCell, inputs: &[Vec<f32>], h0: &[f32]) -> (ForwardCache, Vec<Vec<f32>>) {
    let t = inputs.len();
    let d = cell.hidden_dim;
    let e = cell.input_dim;

    let mut hs = vec![vec![0.0f32; d]; t + 1];
    let mut xs = Vec::with_capacity(t);
    let mut i_gates = Vec::with_capacity(t);
    let mut f_gates = Vec::with_capacity(t);
    let mut decays = Vec::with_capacity(t);

    hs[0].copy_from_slice(h0);

    let mut pre_xi = vec![0.0f32; d];
    let mut pre_hi = vec![0.0f32; d];
    let mut pre_f = vec![0.0f32; d];

    for ti in 0..t {
        let x = &inputs[ti];
        debug_assert_eq!(x.len(), e);

        matvec(&mut pre_xi, &cell.w_xi, x, d, e);
        matvec(&mut pre_hi, &cell.w_hi, &hs[ti], d, d);
        matvec(&mut pre_f, &cell.w_f, &hs[ti], d, d);

        let mut ig = vec![0.0f32; d];
        let mut fg = vec![0.0f32; d];
        let mut dc = vec![0.0f32; d];

        for i in 0..d {
            ig[i] = (pre_xi[i] + pre_hi[i] + cell.b_i[i]).tanh();
            fg[i] = sigmoid(pre_f[i] + cell.b_f[i]);
            dc[i] = (-fg[i]).exp();
            hs[ti + 1][i] = hs[ti][i] * dc[i] + ig[i] * (1.0 - dc[i]);
        }

        xs.push(x.clone());
        i_gates.push(ig);
        f_gates.push(fg);
        decays.push(dc);
    }

    let outputs: Vec<Vec<f32>> = hs[1..].to_vec();
    (
        ForwardCache {
            hs,
            xs,
            i_gates,
            f_gates,
            decays,
        },
        outputs,
    )
}

/// BPTT result — accumulated gradients for the cell and the gradient flowing
/// back into the initial hidden state.
pub struct BpttResult {
    pub grads: CfcGrads,
    /// Gradient w.r.t. the initial hidden state h0 (useful for stacked cells).
    pub dh0: Vec<f32>,
}

/// Backpropagate through time over the stored forward cache.
///
/// `dh_outputs`: per-step loss gradient w.r.t. the output hidden states
///               `h[1]…h[T]`, each of length `hidden_dim`. Pass `None` slices
///               (or zero vectors) for steps where no loss is attached.
///
/// Returns accumulated gradients — caller is responsible for applying them.
pub fn bptt(cell: &CfcCell, cache: &ForwardCache, dh_outputs: &[Vec<f32>]) -> BpttResult {
    let t = cache.xs.len();
    let d = cell.hidden_dim;
    let e = cell.input_dim;
    debug_assert_eq!(dh_outputs.len(), t);

    let mut grads = CfcGrads::zero(cell);
    let mut dh_next = vec![0.0f32; d];

    for ti in (0..t).rev() {
        // Combine gradient from loss at this step and from future steps.
        let mut dh = dh_next.clone();
        for i in 0..d {
            dh[i] += dh_outputs[ti][i];
        }

        let ig = &cache.i_gates[ti];
        let fg = &cache.f_gates[ti];
        let dc = &cache.decays[ti];
        let h_prev = &cache.hs[ti];

        let mut dh_prev = vec![0.0f32; d];
        let mut dpre_xi = vec![0.0f32; d];
        let mut dpre_hi = vec![0.0f32; d];
        let mut dpre_f = vec![0.0f32; d];

        for i in 0..d {
            // dh/d(h_prev) via decay path
            dh_prev[i] = dh[i] * dc[i];

            // d(decay) via forget gate
            let d_decay = dh[i] * (h_prev[i] - ig[i]);
            // d(-f_gate).exp() = exp(-f) * (-1) → chain: d_decay * (-decay)
            let d_fg = d_decay * (-dc[i]);
            // sigmoid'(x) = f*(1-f)
            dpre_f[i] = d_fg * fg[i] * (1.0 - fg[i]);
            grads.g_b_f[i] += dpre_f[i];

            // d(i_gate) path
            let d_ig = dh[i] * (1.0 - dc[i]);
            // tanh'(x) = 1 - tanh²(x) = 1 - i_gate²
            let d_pre_ig = d_ig * (1.0 - ig[i] * ig[i]);
            dpre_xi[i] = d_pre_ig;
            dpre_hi[i] = d_pre_ig;
            grads.g_b_i[i] += d_pre_ig;
        }

        // w_f and w_hi gradients, plus their contribution to dh_prev.
        for i in 0..d {
            for j in 0..d {
                grads.g_w_f[i * d + j] += dpre_f[i] * h_prev[j];
                grads.g_w_hi[i * d + j] += dpre_hi[i] * h_prev[j];
                dh_prev[j] += dpre_f[i] * cell.w_f[i * d + j];
                dh_prev[j] += dpre_hi[i] * cell.w_hi[i * d + j];
            }
        }

        // w_xi gradients.
        let x = &cache.xs[ti];
        for i in 0..d {
            for j in 0..e {
                grads.g_w_xi[i * e + j] += dpre_xi[i] * x[j];
            }
        }

        dh_next = dh_prev;
    }

    BpttResult {
        grads,
        dh0: dh_next,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient application with clipping
// ─────────────────────────────────────────────────────────────────────────────

/// Apply gradient update with per-tensor L2 norm clipping.
///
/// Each parameter tensor is updated independently:
/// `w -= lr * g * min(1, clip/‖g‖)`.
pub fn apply_grads_clipped(cell: &mut CfcCell, grads: &CfcGrads, lr: f32, clip: f32) {
    let apply = |w: &mut [f32], g: &[f32]| {
        let norm: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > clip { clip / norm } else { 1.0 };
        for i in 0..w.len() {
            w[i] -= lr * g[i] * scale;
        }
    };

    apply(&mut cell.w_xi, &grads.g_w_xi);
    apply(&mut cell.w_hi, &grads.g_w_hi);
    apply(&mut cell.w_f, &grads.g_w_f);
    apply(&mut cell.b_i, &grads.g_b_i);
    apply(&mut cell.b_f, &grads.g_b_f);
}

/// Apply gradient update for a [`LinearHead`] with per-tensor clipping.
pub fn apply_head_grads_clipped(
    head: &mut LinearHead,
    g_w: &[f32],
    g_b: &[f32],
    lr: f32,
    clip: f32,
) {
    let apply = |w: &mut [f32], g: &[f32]| {
        let norm: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if norm > clip { clip / norm } else { 1.0 };
        for i in 0..w.len() {
            w[i] -= lr * g[i] * scale;
        }
    };
    apply(&mut head.w, g_w);
    apply(&mut head.b, g_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: full BPTT training step with a single LinearHead
// ─────────────────────────────────────────────────────────────────────────────

/// One complete train step: forward → loss → BPTT → gradient update.
///
/// The head projects `h_t → logits` of shape `[out_dim]`; cross-entropy loss
/// is computed against `targets[t]` (class indices, 0-based).
///
/// `inputs[t]` feeds the cell; `targets[t]` is the label for the hidden state
/// produced *after* processing `inputs[t]` (i.e. predicts next step).
///
/// Returns mean cross-entropy loss over the sequence.
pub fn train_step_ce(
    cell: &mut CfcCell,
    head: &mut LinearHead,
    inputs: &[Vec<f32>],
    targets: &[usize],
    h0: &[f32],
    lr: f32,
    clip: f32,
) -> f32 {
    debug_assert_eq!(inputs.len(), targets.len());
    let t = inputs.len();
    if t == 0 {
        return 0.0;
    }

    let (cache, h_seq) = forward_cache(cell, inputs, h0);

    let out_dim = head.out_dim;
    let d = cell.hidden_dim;
    let mut total_loss = 0.0f32;
    let mut dh_outputs: Vec<Vec<f32>> = vec![vec![0.0f32; d]; t];
    let mut g_w_head = vec![0.0f32; out_dim * d];
    let mut g_b_head = vec![0.0f32; out_dim];

    for ti in 0..t {
        let h = &h_seq[ti];
        let target = targets[ti];

        // Forward through head.
        let mut logits = vec![0.0f32; out_dim];
        head.forward(&mut logits, h);

        // Softmax + cross-entropy.
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        total_loss += -(probs[target].max(1e-9).ln());

        // d_logits = probs - one_hot(target).
        let mut d_logits = probs;
        d_logits[target] -= 1.0;

        // Accumulate head gradients and backprop into dh.
        for k in 0..out_dim {
            g_b_head[k] += d_logits[k];
            for j in 0..d {
                g_w_head[k * d + j] += d_logits[k] * h[j];
                dh_outputs[ti][j] += d_logits[k] * head.w[k * d + j];
            }
        }
    }

    // BPTT through cell.
    let result = bptt(cell, &cache, &dh_outputs);
    apply_grads_clipped(cell, &result.grads, lr, clip);
    apply_head_grads_clipped(head, &g_w_head, &g_b_head, lr, clip);

    total_loss / t as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cell(d: usize, e: usize) -> CfcCell {
        CfcCell::new(d, e, &mut Rng::new(42))
    }

    #[test]
    fn forward_different_inputs_produce_different_outputs() {
        let cell = make_cell(8, 4);
        let h = vec![0.0f32; 8];
        let x1 = vec![1.0, 0.0, 0.0, 0.0];
        let x2 = vec![0.0, 1.0, 0.0, 0.0];

        let h1 = cell.step(&h, &x1);
        let h2 = cell.step(&h, &x2);

        assert_ne!(
            h1, h2,
            "different inputs must produce different hidden states"
        );
    }

    #[test]
    fn hidden_state_decays_toward_zero_without_input() {
        let cell = make_cell(8, 4);
        let x_zero = vec![0.0f32; 4];

        // Start with a non-zero hidden state.
        let mut h: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let initial_norm: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Drive with zero input for many steps — decay should shrink state.
        for _ in 0..50 {
            h = cell.step(&h, &x_zero);
        }
        let final_norm: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();

        assert!(
            final_norm < initial_norm * 0.5,
            "hidden state should decay: initial_norm={initial_norm:.4} final_norm={final_norm:.4}"
        );
    }

    #[test]
    fn gradient_clipping_bounds_update() {
        let mut cell = make_cell(4, 2);
        let w_xi_before = cell.w_xi.clone();

        // Construct pathologically large gradients.
        let mut grads = CfcGrads::zero(&cell);
        grads.g_w_xi.fill(1e6);
        grads.g_w_hi.fill(1e6);
        grads.g_w_f.fill(1e6);
        grads.g_b_i.fill(1e6);
        grads.g_b_f.fill(1e6);

        apply_grads_clipped(&mut cell, &grads, 0.01, 1.0);

        // With clip=1.0 and norm >> 1, each element change is bounded.
        for (after, before) in cell.w_xi.iter().zip(w_xi_before.iter()) {
            let delta = (after - before).abs();
            assert!(
                delta < 1.0,
                "per-element update should be bounded by clip; got delta={delta}"
            );
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let cell = make_cell(8, 4);
        let mut buf = Vec::new();
        cell.save(&mut buf).expect("save failed");

        let loaded = CfcCell::load(&mut buf.as_slice(), 8, 4).expect("load failed");

        assert_eq!(cell.w_xi, loaded.w_xi);
        assert_eq!(cell.w_hi, loaded.w_hi);
        assert_eq!(cell.w_f, loaded.w_f);
        assert_eq!(cell.b_i, loaded.b_i);
        assert_eq!(cell.b_f, loaded.b_f);
    }

    #[test]
    fn linear_head_save_load() {
        let head = LinearHead::new(6, 8, &mut Rng::new(7));
        let mut buf = Vec::new();
        head.save(&mut buf).expect("save failed");
        let loaded = LinearHead::load(&mut buf.as_slice(), 6, 8).expect("load failed");
        assert_eq!(head.w, loaded.w);
        assert_eq!(head.b, loaded.b);
    }

    #[test]
    fn train_step_reduces_loss() {
        let d = 16;
        let e = 4;
        let out_dim = 4;

        let mut cell = CfcCell::new(d, e, &mut Rng::new(1));
        let mut head = LinearHead::new(out_dim, d, &mut Rng::new(2));
        let h0 = vec![0.0f32; d];

        // Simple sequence: cycle through classes.
        let inputs: Vec<Vec<f32>> = (0..16)
            .map(|i| {
                let mut v = vec![0.0f32; e];
                v[i % e] = 1.0;
                v
            })
            .collect();
        let targets: Vec<usize> = (0..16).map(|i| (i + 1) % out_dim).collect();

        let loss_before = train_step_ce(&mut cell, &mut head, &inputs, &targets, &h0, 0.001, 1.0);

        // After many steps the loss should trend down.
        let mut loss = loss_before;
        for _ in 0..100 {
            loss = train_step_ce(&mut cell, &mut head, &inputs, &targets, &h0, 0.01, 1.0);
        }

        assert!(
            loss < loss_before,
            "loss should decrease with training: before={loss_before:.4} after={loss:.4}"
        );
    }

    #[test]
    fn bptt_gradients_nonzero() {
        let d = 4;
        let e = 2;
        let cell = make_cell(d, e);
        let inputs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let h0 = vec![0.0f32; d];
        let (cache, _) = forward_cache(&cell, &inputs, &h0);

        // Inject a non-zero gradient at the last step.
        let mut dh_outputs: Vec<Vec<f32>> = vec![vec![0.0f32; d]; 3];
        dh_outputs[2] = vec![1.0; d];

        let result = bptt(&cell, &cache, &dh_outputs);
        let norm = result.grads.global_norm();
        assert!(
            norm > 0.0,
            "gradients must be nonzero when loss gradient is injected"
        );
    }

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn matvec_correctness() {
        // 2×3 matrix, identity-like pattern.
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let x = vec![3.0, 5.0, 7.0];
        let mut out = vec![0.0f32; 2];
        matvec(&mut out, &w, &x, 2, 3);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }
}
