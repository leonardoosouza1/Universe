# Evolving compositions of mathematical functions

The proposed chain of mathematical functions is a useful controller abstraction.
It fits the body/brain separation without requiring a new world or physics engine.
It does not automatically supply learning: the representation, update rule and
evaluation environment are separate design choices.

## Corrections to the supplied explanation

- Function composition and computational graphs describe neural computation well.
  A recurrent brain has explicit state and feedback across time; a single static
  feed-forward chain is only one special case. The full body/world interaction
  includes contacts, discrete events and sampled signals.
- Differentiating supported numerical operations is different from accepting any
  arbitrary Python function. Vector backpropagation applies Jacobian products;
  JAX exposes function transformations for this purpose. See the
  [JAX automatic differentiation documentation](https://docs.jax.dev/en/latest/automatic-differentiation.html).
- The example containing only `DenseLayer.forward(x) = x * w` collapses to
  `output = input * product(weights)`, regardless of depth. It cannot demonstrate
  the representational benefit of nonlinear or recurrent composition.
- Symbolic regression searches expressions against data/targets; genetic
  programming can instead search controllers scored through environment
  interaction. Universe generally has trajectories and consequences, not a known
  correct motor target for every observation. Exact formula recovery is not
  guaranteed, especially with noise or inadequate observations.
- Complexity limits, domain restrictions and constant optimization are explicit
  algorithm choices. PySR exposes operator/nesting/complexity constraints and
  dimensional penalties; BFGS is not a universal requirement for every constant
  at every generation. See the [PySR API](https://ai.damtp.cam.ac.uk/pysr/v2.0.0/api).
- Protected division/logarithms change the function being searched. Their units,
  limits and behavior near singularities need tests. Returning a convenient
  constant on invalid input can itself become an exploitable policy.

## Fit to Universe

Use the stateful controller contract

`(memory_next, action) = F(graph, parameters; observation, memory)`.

The genotype encodes a bounded graph, constants, allowed connection changes and
memory time scales. Per-being working state contains current memory and neuron
activity. Inherited parameters, lifetime state and any within-lifetime plasticity
must remain separate. The body still defines observations, action units and
physical consequences. Changing the body requires an explicit sensory/motor
adapter, not merely matching array lengths.

A first candidate should be a small recurrent mathematical controller beside the
existing Fly controller. Restrict the grammar to bounded arithmetic, saturation,
gating and explicit memory updates. Search structure between generations on the
CPU if cheap; execute compiled/batched numerical operations from GPU buffers
during each lifetime. Software module boundaries do not imply a kernel launch or
CPU/GPU transfer for every mathematical operator.

Keep operator count and state size fixed initially. Compare against a constant
controller and a small recurrent baseline using exactly the same observations,
actions, tracks and compute/evaluation budgets. Train and hold-out seeds/tracks
must be separate. Assess progress together with measured contacts/proximity,
resource use and numerical validity. The current battery shows why: reducing gain
improved progress while worsening sampled overlap.

The first goal is evidence that a small evolved composition generalizes to a
held-out control task. It is not a claim of general intelligence, autonomous
biological development or a reason to replace the validated kernels wholesale.
