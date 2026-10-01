# Lab 1 — Neural Models: Learning, Depth, Activations and Output Layers

Two redundant binary sensors, `x1` and `x2`. The device must raise a
disagreement warning exactly when one sensor is active and the other is not —
that is, XOR. The point of a four-row dataset is that the representation and
the learning signal stay small enough to inspect completely.

| File | Contents |
|---|---|
| `xor_binary.py` | Tasks 3–4: the 2–2–1 binary network, backpropagation check, symmetry experiment, activation experiment |
| `xor_threeclass.py` | Task 5: the same hidden layer with three logits and cross-entropy |
| `prompts.md` | The exact LLM prompts, and the seven corrections made before accepting the generated code |
| `results_binary.txt` | Captured output of `xor_binary.py` |
| `results_threeclass.txt` | Captured output of `xor_threeclass.py` |

```bash
python xor_binary.py
python xor_threeclass.py
```

Run with PyTorch 2.14.1 on CPU, seed 0, Adam(lr=0.05), 5000 full-batch steps.

---

## Task 1 — The problem, before any code

**Input space** `X = {0,1}^2`, four points. **Output space** `Y = {0,1}`, where
1 means *disagreement warning*.

| `x1` | `x2` | `y` |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The four points in the plane, labelled by class:

```
 x2
  1 |  (0,1)=1          (1,1)=0
    |
  0 |  (0,0)=0          (1,0)=1
    +---------------------------- x1
         0                 1
```

**Why one straight boundary cannot separate the classes.** The two positive
points `(0,1)` and `(1,0)` lie at opposite ends of one diagonal; the two
negative points `(0,0)` and `(1,1)` lie at opposite ends of the other. Any line
`w1*x1 + w2*x2 + b = 0` splits the plane into two half-planes, and each diagonal
pair has the other pair's midpoint between it — both diagonals cross at
`(0.5, 0.5)`, so no half-plane can contain both positives while excluding both
negatives. Formally, a linear decision rule would need `w2 + b > 0` and
`w1 + b > 0` (both positives fire) together with `b < 0` and `w1 + w2 + b < 0`
(both negatives do not); adding the first two inequalities gives
`w1 + w2 + 2b > 0`, which contradicts the last two, since those give
`w1 + w2 + 2b < 0`.

**Prediction for a single affine map plus sigmoid.** It cannot reach 4/4. The
best it can do is sacrifice one corner. The run confirms this exactly: the
symmetric network in Part C2 — which is effectively linear — converges to a
final loss of **0.477402**, and the analytic optimum of a one-line model is

```
(1/4) [ 0 + ln(3/2) + ln(3/2) + ln 3 ] = 0.477386
```

i.e. it gets one point right and assigns probability 2/3 to the other three.
The measured number matching the closed form to four decimal places is the
strongest evidence in this lab that the failure is representational, not an
optimisation accident.

**Think about it — what scientific claim do four points let us test?** That
parameter count and representational kind are different things. A model can
have any number of parameters and still be the wrong *kind* of function. XOR is
the minimal counterexample: adding affine-only depth adds parameters without
adding representable functions, because a composition of affine maps is affine.

---

## Task 2 — Model design and validation criteria

```
2 inputs  ->  2 hidden units (nonlinear)  ->  1 logit
loss      :  binary cross-entropy on the logit (BCEWithLogitsLoss)
optimiser :  full-batch Adam, lr = 0.05, 5000 steps
```

**1. Why the hidden nonlinearity is scientifically necessary.** Without it the
network computes `W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)`, a single
affine map however many layers are stacked. The nonlinearity is what lets the
two hidden units carve the input space into regions that are not half-planes, so
the output can read a nonlinearly separable combination. Depth without
nonlinearity buys nothing here.

**2. Why sigmoid + binary cross-entropy is a sensible output pairing.** The
target is one yes/no answer, so the natural output is a single probability:
sigmoid maps a real logit to `(0,1)`. Binary cross-entropy is the negative
log-likelihood of a Bernoulli variable, so minimising it *is* maximum-likelihood
estimation under the model the sigmoid defines. The pairing also cancels well:
the logit gradient reduces to `p - y`, with no factor of `sigma'(z)` left to
vanish. In PyTorch, `BCEWithLogitsLoss` computes the pair in one stable step
rather than taking `log` of an already-saturated sigmoid.

**3. What will count as successful learning.** Four checks, fixed in advance:

- the loss falls by orders of magnitude from its initial value, not merely a
  little;
- all four thresholded predictions are correct (4/4);
- the gradients are nonzero where learning is still needed, and autograd agrees
  with an independent numerical derivative;
- the behaviour repeats across several seeds, so the result is not one lucky
  draw of initial weights.

---

## Task 4, Part A — Basic learning check

| | value |
|---|---|
| initial loss | 0.715159 |
| final loss | 2.378083e-05 |

| `x1` | `x2` | target | probability | predicted |
|---|---|---|---|---|
| 0 | 0 | 0 | 0.000017 | 0 |
| 0 | 1 | 1 | 0.999971 | 1 |
| 1 | 0 | 1 | 0.999965 | 1 |
| 1 | 1 | 0 | 0.000014 | 0 |

**4/4 correct.** The loss fell by a factor of about 30,000 and every probability
sits within `3e-5` of its target, so the network is not merely better than
chance — it has essentially interpolated the truth table.

## Task 4, Part B — Backpropagation check

The first-layer gradient after one forward/backward pass on the untrained
network:

```
model[0].weight.grad  =  dL/dW1   (shape (2, 2)):
tensor([[ 0.0005,  0.0006],
        [-0.0426, -0.0448]])
```

`parameter.grad[j][i]` is `dL/dW1[j][i]`, the partial derivative of the scalar
loss with respect to the weight joining input `i` to hidden unit `j`. It is
accumulated by reverse-mode automatic differentiation: the chain rule applied
backwards from the scalar loss through the output layer and the hidden
activation, which is exactly the backpropagation recursion from the lecture.
It is also the quantity the optimiser subtracts a scaled version of.

**Why the gradient is the average of the example-wise gradients.** The loss is a
mean, `L = (1/4) * sum_n L_n`, and differentiation is linear, so
`dL/dW1 = (1/4) * sum_n dL_n/dW1`. Measured rather than assumed: the four
example-wise gradients were computed separately and averaged, and

```
max |batch gradient - mean of example gradients| = 0.000e+00
```

**Independent check that autograd is right at all.** Central differences on
every entry of `W1`, run on a `float64` copy of the network with `h = 1e-6`:

| entry | autograd | finite difference | &#124;diff&#124; |
|---|---|---|---|
| `W1[0][0]` |  0.000505399 |  0.000505399 | 5.37e-11 |
| `W1[0][1]` |  0.000606949 |  0.000606949 | 2.99e-12 |
| `W1[1][0]` | -0.042569033 | -0.042569033 | 1.04e-11 |
| `W1[1][1]` | -0.044772913 | -0.044772913 | 2.10e-11 |

This check initially *failed*, in float32 with `h = 1e-4`, reporting
disagreements around `4e-4` — the same size as the gradient entries. The
gradient was correct; the test was not. See `prompts.md`, correction 1.

## Task 4, Part C — Symmetry experiment

**C1, every weight and bias zeroed** (as the lab specifies). The two rows of
`W1` stay identical — in fact both stay exactly `[0.0, 0.0]` at every step and
after 5000 steps. Final loss **0.693147**, probabilities `[0.5, 0.5, 0.5, 0.5]`,
2/4 correct (and only because ties at exactly 0.5 threshold upward).

With `W2 = 0`, the derivative arriving at the hidden layer is zero, so
`dL/dW1 = 0` identically and the hidden layer never moves. Only the output bias
can learn, and the best a constant output achieves on a balanced XOR table is
`p = 0.5` everywhere, giving `ln 2 = 0.693147` — which is the number observed.

**C2, the two hidden units initialised identically but not to zero.** This is
the symmetry problem proper, and it was added because C1 alone demonstrates the
wrong mechanism.

```
step 0:  row0 = [-0.005294, 0.379323]    row1 = [-0.005294, 0.379323]
step 4:  row0 = [ 0.166789, 0.305036]    row1 = [ 0.166789, 0.305036]
final :  row0 = [ 6.717868, 6.719142]    row1 = [ 6.717868, 6.719142]
rows still identical: True
```

Final loss **0.477402**, 3/4 correct. Here the gradients are nonzero yet the two
rows move in lockstep for ever: identical hidden units compute the same output,
therefore receive the same upstream derivative, therefore get the same update.
The 2–2–1 network collapses to a 1-hidden-unit network, whose boundary is a
single line, and it lands precisely on the analytic one-line optimum computed in
Task 1. Random initialisation is what breaks the symmetry; C1 is the degenerate
special case where the gradient vanishes as well.

## Task 4, Part D — Activation experiment

Same seed, same architecture, same optimiser; only the hidden activation
changes. Gradient norm measured at step 10, before the loss collapses, so it
reflects the activation's derivative rather than proximity to the optimum.

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇<sub>W1</sub>L‖₂ |
|---|---|---|---|
| Sigmoid | 4.774e-01 | no (3/4) | 0.001227 |
| Tanh | 2.378e-05 | **yes** | 0.009383 |
| ReLU | 6.931e-01 | no (2/4) | 0.000000 |

Repeating over five seeds, because one seed cannot separate the activation from
the draw of initial weights:

| Hidden activation | Solved 4/4 | Median final loss | Best final loss |
|---|---|---|---|
| Sigmoid | 4/5 | 7.720e-05 | 7.361e-05 |
| Tanh | 4/5 | 3.160e-05 | 2.378e-05 |
| ReLU | 1/5 | 6.931e-01 | 8.867e-06 |

**Interpretation.** On this problem tanh had the largest early gradient
(0.0094) and solved the task at seed 0; sigmoid's early gradient was about 8×
smaller, consistent with its derivative being bounded by 1/4 against tanh's 1,
and at this step budget seed 0 left it stranded on the one-line solution at
0.4774. But the five-seed sweep reverses that reading: sigmoid solved 4 of 5
seeds, so seed 0 was unrepresentative and the single-run table would have
supported a false conclusion. ReLU's zero early gradient is a different failure
entirely, and the sweep shows it is the fragile choice here (1/5), while its
*best* seed gave the lowest loss of any run (8.9e-06). Four data points and two
hidden units cannot establish that any activation is universally best; what they
show is that with so few units, ReLU's dead-unit risk is high, and that
conclusions drawn from one seed are unreliable.

**Think about it — telling a saturated sigmoid from a dead ReLU.** Both produce
a small gradient by different mechanisms, and the pre-activations separate them.
Measured at step 10:

| Activation | mean `f'(z)` | fraction with `f'(z)` exactly 0 |
|---|---|---|
| sigmoid | 0.244881 | 0.00 |
| tanh | 0.828233 | 0.00 |
| ReLU | 0.000000 | 1.00 |

For sigmoid every `f'(z)` is small but strictly positive — gradient still flows,
only slowly, and a unit can recover. For ReLU at this seed every `z` is negative
across all four inputs (`z` ranged from `-0.13` to `-1.29`), so `f'(z)` is
exactly zero, no gradient flows, and the units are dead with no route back. The
diagnostic is to print `z` alongside `f'(z)`: saturation shows as `|z|` large
with `f'(z)` small-but-positive, death shows as `f'(z)` identically zero.

---

## Task 5 — Three-class extension

Classes: 0 = both inactive `(0,0)`, 1 = sensors disagree `(0,1)`/`(1,0)`,
2 = both active `(1,1)`. Only the output layer and loss changed: two tanh hidden
units feeding three logits, trained with cross-entropy.

**Predictions recorded before running** (all four confirmed):

1. the final weight matrix is `W2` of shape `(3, 2)`, plus a bias of shape
   `(3,)` — measured `(3, 2)` and `(3,)`;
2. three logits per example, so the logit tensor is `(4, 3)` — measured `(4, 3)`;
3. softmax probabilities sum to one because each `exp(z_k)` is divided by the
   sum of all `exp(z_j)`, a denominator that is exactly their total;
4. the logit gradient has the form `p - y`.

Initial loss 1.059108 (against `ln 3 = 1.098612` for an uninformative model),
final loss 1.090759e-05.

| `x1` | `x2` | target | P(class 0) | P(class 1) | P(class 2) | predicted |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.999986 | 0.000014 | 0.000000 | 0 |
| 0 | 1 | 1 | 0.000007 | 0.999991 | 0.000002 | 1 |
| 1 | 0 | 1 | 0.000007 | 0.999992 | 0.000002 | 1 |
| 1 | 1 | 2 | 0.000000 | 0.000013 | 0.999987 | 2 |

**4/4 correct.** For the input `(0,1)` the softmax vector is
`[6.6342e-06, 0.9999914169, 1.9422e-06]`, summing to `1.0000000000`
(`|sum - 1| = 0`); all four row sums are 1 to within `6e-08`.

**Why the logit gradient is `p - y`, verified not assumed.** For cross-entropy
on a softmax output, `dL_n/dz_k = p_k - y_k`: the log-normaliser contributes
`p_k` and the selected log-probability contributes `-y_k`. Comparing autograd's
`dL/dz` against `(p - y)/4` computed by hand from the softmax probabilities and
one-hot targets:

```
max |autograd - (p - y)/4| = 1.364e-12
```

The division by 4 is only the mean reduction over the batch. This cancellation
is why softmax and cross-entropy are chosen together: no activation derivative
survives to attenuate the signal.

**Optional diagnostic — shift invariance.** Adding 100 to all three logits
leaves the probabilities unchanged to `5.275e-11`, because
`exp(z_k + c) / sum_j exp(z_j + c)` cancels the common factor `exp(c)`. The
cancellation is exact in mathematics but not in floating point: a naive
`exp(z)/sum(exp(z))` on `z = [1000, 1001, 1002]` returns `[nan, nan, nan]`,
because `exp(1000)` overflows before anything can cancel, while
`torch.softmax` returns `[0.090031, 0.244728, 0.665241]`. Stable
implementations subtract `max_j z_j` first; by the same cancellation the result
is mathematically identical, but the largest exponent becomes `exp(0) = 1`, so
nothing overflows.

**Think about it — what survives scaling to a real vocabulary?** Unchanged: the
softmax normalisation, the cross-entropy objective, and the gradient `p - y`.
Changed: the output matrix becomes `(vocabulary × hidden)` and dominates the
parameter count; the normalising sum over classes becomes the expensive part of
every step; targets are stored as indices rather than one-hot rows; and the
context is no longer two binary inputs but a long sequence, which requires an
architecture that can represent it.

---

## Reflection questions

**1. What did the XOR experiment demonstrate about depth versus nonlinearity?**
That they are not substitutes. Stacking affine layers composes to a single
affine map, so affine-only depth adds parameters without adding representable
functions and still cannot express XOR. Nonlinearity is the ingredient that
changes the class of functions available; depth then changes how efficiently
that class can be composed. The 2–2–1 network with tanh reached 2.4e-05, while
the effectively-linear symmetric network in C2 stalled at the provable one-line
optimum of 0.4774.

**2. What showed that backpropagation supplied a useful learning signal rather
than merely a nonzero gradient?** A nonzero gradient only says the loss is
locally sensitive to the weights; it does not say the direction is right. The
useful-signal evidence was: the loss fell from 0.715 to 2.4e-05, all four
predictions crossed to the correct side of the threshold, autograd's gradient
matched an independent finite-difference derivative to 5.4e-11 (so the direction
was the true descent direction and not an artefact), and the result reproduced
across 4 of 5 seeds. Part C is the useful contrast: in C2 the gradient is
nonzero and training proceeds for 5000 steps, yet no useful representation can
emerge, because the gradient is the same for both units.

**3. Why did identical/zero initialisation prevent distinct features?** Two
hidden units with identical incoming weights compute identical outputs for every
input. Backpropagation sends each unit a derivative that depends only on its own
output and its outgoing weight, so identical units receive identical gradients
and identical updates, and the equality is preserved at every step — as observed,
the rows were still identical after 5000 steps. There is nothing in the
mathematics to break the tie, which is precisely why symmetry breaking has to be
supplied by the initialisation. In the all-zero case it is worse: `W2 = 0` makes
`dL/dW1` identically zero, so the hidden layer cannot move at all.

**4. How did the activation affect the observed gradient — science versus
engineering?** *Scientific explanation:* the backpropagated signal is multiplied
by `f'(z)` at the hidden layer. For sigmoid, `f'(z) <= 1/4`; for tanh,
`f'(z) <= 1`; for ReLU, `f'(z)` is 1 on active units and exactly 0 on inactive
ones. So sigmoid systematically attenuates the signal and ReLU either passes it
intact or blocks it entirely. *Engineering observation:* the early gradient norm
was 0.0094 for tanh against 0.0012 for sigmoid, roughly the factor of 4–8 the
derivative bounds predict, and 0.0000 for ReLU because all its units were
inactive at that seed. The distinction matters: the first statement is a claim
about the mechanism and holds generally; the second is a measurement on one
4-point problem with two hidden units, and the five-seed sweep showed it is not
even stable across seeds.

**5. Why must the output layer and loss be chosen together?** Because the pair
defines what the model's output *means* and determines whether the gradient is
well-behaved. Sigmoid + BCE is the Bernoulli likelihood; softmax +
cross-entropy is the categorical likelihood. Both pairings cancel to a logit
gradient of `p - y`, so no activation derivative remains to vanish. Mismatch it
— sigmoid with squared error, say — and the gradient acquires a `sigma'(z)`
factor that goes to zero exactly when the model is most confidently wrong, which
is when the largest correction is needed. Three independent sigmoid outputs would
also not be mutually exclusive classes, so the probabilities would not sum to 1
and `argmax` would not be a valid decision rule.

**6. Where did the LLM help, and where was human verification essential?**
*Helped:* writing the PyTorch boilerplate — the training loop, the
`retain_grad()` call needed to expose `dL/dz`, the softmax-overflow
demonstration, and the table formatting — quickly and mostly correctly.
*Essential human verification:* the finite-difference check was written in
float32 with `h = 1e-4` and reported disagreements of about `4e-4`, i.e. it
could not have detected a wrong gradient; the zero-initialisation experiment
demonstrated a vanished gradient rather than the symmetry problem, so the C2
variant had to be added; and the single-seed activation table supported a
conclusion about sigmoid that the five-seed sweep contradicts. All three are
cases where the code ran, printed plausible numbers, and was answering the wrong
question. See `prompts.md` for the full list of seven corrections.

**7. Which tests would survive scaling, and which would not?** *Keep:* tracking
initial versus final loss; held-out accuracy in place of 4/4 interpolation;
gradient-norm monitoring per layer; the activation/pre-activation statistics from
Part D, which are cheap and catch dead or saturated units; repeated runs across
seeds; and invariance checks like softmax normalisation, which are O(1) per step
and catch real implementation bugs. *Drop:* exhaustive finite-difference gradient
checking, which costs two forward passes per parameter and is hopeless at scale —
it would be replaced by a spot check on a random subset of parameters, or by
`torch.autograd.gradcheck` on a small module in isolation. Also impossible at
scale: enumerating the entire input space, which is the one thing that makes this
four-row experiment fully conclusive.
