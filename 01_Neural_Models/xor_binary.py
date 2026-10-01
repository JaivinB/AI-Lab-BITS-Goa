"""
Lab 1, Tasks 3 and 4 - binary XOR with a 2-2-1 network.

Scenario: two redundant binary safety sensors.  A disagreement warning must be
raised exactly when one sensor is active and the other is inactive, i.e. XOR.

Design fixed in Task 2 and NOT changed by any experiment below:

    2 inputs  ->  2 hidden units (nonlinear)  ->  1 logit
    loss      :  binary cross-entropy applied to the logit
                 (BCEWithLogitsLoss = sigmoid + BCE, computed stably)
    optimiser :  full-batch gradient-based optimisation over all four examples

Everything printed here is evidence for one of the Task 4 checks:

    Part A  basic learning check  - initial/final loss, four probabilities,
                                    thresholded labels, 4/4 correctness
    Part B  backpropagation check - parameter.grad as dL/dW1, a numerical proof
                                    that the mean-loss gradient is the mean of
                                    the example-wise gradients, and a
                                    finite-difference check of autograd
    Part C  symmetry experiment   - all-zero init, and identical-nonzero init
    Part D  activation experiment - sigmoid vs tanh vs ReLU, with the
                                    pre-activation diagnostic that separates a
                                    saturated sigmoid from a dead ReLU
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# The complete dataset: the whole truth table, four examples, no split.
# X in {0,1}^2, Y in {0,1}.  Full-batch training therefore sees everything.
# ---------------------------------------------------------------------------
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}

SEED = 0
STEPS = 5000
LR = 0.05
EARLY_STEP = 10          # the "early training step" used for the gradient norm
SWEEP_SEEDS = (0, 1, 2, 3, 4)


def build_model(activation):
    """2-2-1 network returning a single logit.  Architecture never varies."""
    return nn.Sequential(
        nn.Linear(2, 2),
        ACTIVATIONS[activation](),
        nn.Linear(2, 1),
    )


def train(model, steps=STEPS, lr=LR, trace_rows=False, capture_early=False):
    """Full-batch training.

    Returns a dict of evidence: initial/final loss, the first-layer gradient
    norm at EARLY_STEP, the hidden pre-activations at EARLY_STEP, and
    optionally a trace of the first-layer weight rows over the first steps.
    """
    loss_fn = nn.BCEWithLogitsLoss()             # mean over the four examples
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    evidence = {"trace": []}

    for step in range(steps):
        optimiser.zero_grad()
        logits = model(X)                        # forward pass
        loss = loss_fn(logits, Y)                # scalar loss
        loss.backward()                          # reverse-mode AD: fills .grad

        if step == 0:
            evidence["initial_loss"] = loss.item()
        if step == EARLY_STEP:
            evidence["early_grad_norm"] = model[0].weight.grad.norm().item()
            if capture_early:
                with torch.no_grad():
                    evidence["early_preactivations"] = model[0](X).clone()
        if trace_rows and step < 5:
            evidence["trace"].append(
                (step, model[0].weight.detach().clone().tolist()))

        optimiser.step()                         # parameters change here

    with torch.no_grad():
        evidence["final_loss"] = loss_fn(model(X), Y).item()
    return evidence


def predictions(model):
    """Probabilities, thresholded labels and the number correct out of four."""
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X)).squeeze(1)
    labels = (probabilities >= 0.5).float()
    correct = int((labels == Y.squeeze(1)).sum().item())
    return probabilities, labels, correct


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Part A - basic learning check
# ---------------------------------------------------------------------------
def part_a():
    banner("PART A - basic learning check (hidden activation: tanh, random init)")
    torch.manual_seed(SEED)
    model = build_model("tanh")
    evidence = train(model)
    probabilities, labels, correct = predictions(model)

    print("Initial loss : %.6f" % evidence["initial_loss"])
    print("Final loss   : %.6e" % evidence["final_loss"])
    print()
    print(" x1  x2  target   probability   predicted")
    for (x1, x2), target, p, label in zip(X.tolist(), Y.squeeze(1).tolist(),
                                          probabilities.tolist(), labels.tolist()):
        print("  %d   %d     %d       %.6f        %d" % (x1, x2, target, p, label))
    print()
    print("Correct: %d/4   ->   %s" % (correct, "PASS" if correct == 4 else "FAIL"))


# ---------------------------------------------------------------------------
# Part B - backpropagation check
# ---------------------------------------------------------------------------
def part_b():
    banner("PART B - backpropagation check")
    torch.manual_seed(SEED)
    model = build_model("tanh")
    loss_fn = nn.BCEWithLogitsLoss()

    # One forward/backward pass on the freshly initialised network.
    model.zero_grad()
    batch_loss = loss_fn(model(X), Y)
    batch_loss.backward()
    batch_grad = model[0].weight.grad.clone()

    print("Loss on the untrained network: %.6f" % batch_loss.item())
    print()
    print("model[0].weight.grad  =  dL/dW1   (shape %s):" % (tuple(batch_grad.shape),))
    print(batch_grad)
    print()
    print("Entry [j][i] is the partial derivative of the scalar loss with")
    print("respect to the weight joining input i to hidden unit j.  It is the")
    print("quantity the optimiser subtracts (scaled) from that weight.")
    print()

    # The loss is a MEAN over four examples, so dL/dW1 must be the mean of the
    # four example-wise gradients.  Verify numerically rather than assert it.
    per_example = []
    for index in range(4):
        model.zero_grad()
        loss_fn(model(X[index:index + 1]), Y[index:index + 1]).backward()
        per_example.append(model[0].weight.grad.clone())
    mean_of_examples = torch.stack(per_example).mean(dim=0)

    print("Mean of the four example-wise dL/dW1 tensors:")
    print(mean_of_examples)
    print()
    print("max |batch gradient - mean of example gradients| = %.3e"
          % (batch_grad - mean_of_examples).abs().max().item())
    print("-> the mean loss differentiates to the mean of the example gradients,")
    print("   because differentiation is linear and L = (1/4) * sum_n L_n.")
    print()

    # Independent check that autograd computes the right derivative at all:
    # central finite differences on every entry of W1.  Done in float64,
    # because in float32 the differencing noise (~1e-4) is as large as the
    # gradient entries themselves and the check cannot discriminate.
    print("Finite-difference check of autograd on W1")
    print("(central differences, h=1e-6, float64 to control rounding):")
    double_model = build_model("tanh").double()
    double_model.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
    Xd, Yd = X.double(), Y.double()
    double_model.zero_grad()
    loss_fn(double_model(Xd), Yd).backward()
    analytic_grad = double_model[0].weight.grad.clone()

    h = 1e-6
    worst = 0.0
    with torch.no_grad():
        for j in range(2):
            for i in range(2):
                original = double_model[0].weight[j, i].item()
                double_model[0].weight[j, i] = original + h
                plus = loss_fn(double_model(Xd), Yd).item()
                double_model[0].weight[j, i] = original - h
                minus = loss_fn(double_model(Xd), Yd).item()
                double_model[0].weight[j, i] = original
                numeric = (plus - minus) / (2 * h)
                analytic = analytic_grad[j, i].item()
                worst = max(worst, abs(numeric - analytic))
                print("  W1[%d][%d]: autograd % .9f   finite-diff % .9f   |diff| %.2e"
                      % (j, i, analytic, numeric, abs(numeric - analytic)))
    print("  worst disagreement: %.2e  ->  %s"
          % (worst, "autograd agrees with the numerical derivative"
             if worst < 1e-7 else "DISAGREEMENT"))


# ---------------------------------------------------------------------------
# Part C - symmetry experiment
# ---------------------------------------------------------------------------
def _symmetry_run(title, initialise, explanation):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)
    torch.manual_seed(SEED)
    model = build_model("tanh")
    initialise(model)
    evidence = train(model, trace_rows=True)

    print("First-layer weight matrix over the first five steps:")
    for step, weight in evidence["trace"]:
        print("  step %d:  row0 = %s" % (step, [round(v, 6) for v in weight[0]]))
        print("           row1 = %s" % ([round(v, 6) for v in weight[1]],))
    final_weight = model[0].weight.detach()
    print()
    print("After %d steps:" % STEPS)
    print("  row0 = %s" % [round(v, 6) for v in final_weight[0].tolist()])
    print("  row1 = %s" % [round(v, 6) for v in final_weight[1].tolist()])
    print("  rows still identical: %s" % torch.allclose(final_weight[0], final_weight[1]))
    print()
    probabilities, labels, correct = predictions(model)
    print("Final loss   : %.6f" % evidence["final_loss"])
    print("Probabilities: %s" % [round(p, 6) for p in probabilities.tolist()])
    print("Predicted    : %s   (targets [0, 1, 1, 0])" % [int(v) for v in labels.tolist()])
    print("Correct      : %d/4" % correct)
    print()
    print(explanation)


def part_c():
    banner("PART C - symmetry experiment")

    def zero_everything(model):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

    def identical_nonzero_rows(model):
        # Keep the random draw, then force both hidden units to be copies of
        # each other.  The gradients are now nonzero but equal, which is the
        # symmetry mechanism itself rather than a dead network.
        with torch.no_grad():
            model[0].weight[1] = model[0].weight[0]
            model[0].bias[1] = model[0].bias[0]
            model[2].weight[0, 1] = model[2].weight[0, 0]

    _symmetry_run(
        "C1: every weight and bias set to zero (as the lab specifies)",
        zero_everything,
        "The two rows stay identical - in fact both stay exactly zero.  With\n"
        "W2 = 0 the signal reaching the hidden layer is zero, so dL/dW1 = 0 and\n"
        "the hidden layer never moves at all.  Only the output bias can learn,\n"
        "and the best a constant output can do on a balanced XOR table is\n"
        "p = 0.5 everywhere, i.e. loss ln 2 = 0.693147.",
    )

    _symmetry_run(
        "C2: the two hidden units initialised identically but NOT to zero",
        identical_nonzero_rows,
        "Here the gradients are nonzero, yet the two rows remain identical at\n"
        "every step.  Identical hidden units compute the same output, so they\n"
        "receive the same upstream derivative and are updated by the same\n"
        "amount for ever.  The 2-2-1 network therefore behaves as a 1-hidden-\n"
        "unit network, whose decision boundary is a single line - and one line\n"
        "cannot separate XOR.  This is the symmetry problem proper; C1 is the\n"
        "degenerate special case in which the gradient also vanishes.\n"
        "Random initialisation is what breaks the symmetry.",
    )


# ---------------------------------------------------------------------------
# Part D - activation experiment
# ---------------------------------------------------------------------------
def _derivative_of(name, preactivations):
    """f'(z) for the hidden activation, used for the saturation diagnostic."""
    if name == "sigmoid":
        s = torch.sigmoid(preactivations)
        return s * (1 - s)
    if name == "tanh":
        return 1 - torch.tanh(preactivations) ** 2
    return (preactivations > 0).float()          # relu


def part_d():
    banner("PART D - activation experiment "
           "(identical seed, only the hidden activation changes)")
    rows = []
    for name in ("sigmoid", "tanh", "relu"):
        torch.manual_seed(SEED)                  # same initial weights each time
        model = build_model(name)
        evidence = train(model, capture_early=True)
        probabilities, _, correct = predictions(model)
        rows.append((name, evidence, correct,
                     [round(p, 4) for p in probabilities.tolist()]))

    header = "%-10s %14s %13s %22s" % ("activation", "final loss",
                                       "4/4 correct?", "early ||dL/dW1||_2")
    print(header)
    print("-" * len(header))
    for name, evidence, correct, _ in rows:
        print("%-10s %14.3e %13s %22.6f"
              % (name, evidence["final_loss"],
                 "yes" if correct == 4 else "no (%d/4)" % correct,
                 evidence["early_grad_norm"]))
    print()
    for name, _, _, probabilities in rows:
        print("  %-8s probabilities: %s" % (name, probabilities))

    # The "Think About It": a small gradient can come from a saturated sigmoid
    # (derivative near zero but positive) or from a negative ReLU input
    # (derivative exactly zero).  Inspecting the pre-activations separates them.
    print()
    print("Saturation diagnostic at step %d - hidden pre-activations z and f'(z):"
          % EARLY_STEP)
    for name, evidence, _, _ in rows:
        z = evidence["early_preactivations"]
        derivative = _derivative_of(name, z)
        print("  %s:" % name)
        print("    z      (4 examples x 2 units) = %s"
              % [[round(v, 4) for v in row] for row in z.tolist()])
        print("    f'(z)                          = %s"
              % [[round(v, 4) for v in row] for row in derivative.tolist()])
        print("    mean f'(z) = %.6f   fraction with f'(z) exactly 0 = %.2f"
              % (derivative.mean().item(), (derivative == 0).float().mean().item()))
    print()
    print("  Sigmoid: f'(z) is small but strictly positive everywhere - the unit")
    print("  is saturating, and gradient still flows, only slowly.")
    print("  ReLU:    f'(z) is exactly 0 wherever z <= 0 - those units are dead")
    print("  and no gradient flows through them at all.  The two mechanisms are")
    print("  distinguishable precisely by looking at the pre-activations.")

    print()
    print("The gradient norm is measured at step %d, i.e. before the loss has"
          % EARLY_STEP)
    print("collapsed, so it reflects the activation's derivative rather than")
    print("proximity to the optimum.")
    return rows


def part_d_sweep():
    """Repeated-run behaviour: one seed is not evidence about an activation."""
    banner("PART D (continued) - repeated runs over %d seeds" % len(SWEEP_SEEDS))
    print("Task 2 listed 'repeated-run behaviour' as a validation check, and a")
    print("single seed cannot separate the activation from the draw of initial")
    print("weights.  Same architecture, same optimiser, seeds %s."
          % (list(SWEEP_SEEDS),))
    print()
    header = "%-10s %14s %18s %16s" % ("activation", "solved 4/4",
                                       "median final loss", "best final loss")
    print(header)
    print("-" * len(header))
    summary = []
    for name in ("sigmoid", "tanh", "relu"):
        losses, solved = [], 0
        for seed in SWEEP_SEEDS:
            torch.manual_seed(seed)
            model = build_model(name)
            evidence = train(model)
            _, _, correct = predictions(model)
            losses.append(evidence["final_loss"])
            solved += (correct == 4)
        losses.sort()
        median = losses[len(losses) // 2]
        summary.append((name, solved, median, losses[0]))
        print("%-10s %14s %18.3e %16.3e"
              % (name, "%d/%d" % (solved, len(SWEEP_SEEDS)), median, losses[0]))
    print()
    print("Read this as a statement about this 4-point problem with 2 hidden")
    print("units and this optimiser budget - not as a ranking of activations.")
    return summary


if __name__ == "__main__":
    print("torch", torch.__version__, "| seed", SEED, "| steps", STEPS,
          "| optimiser Adam(lr=%.2f)" % LR)
    part_a()
    part_b()
    part_c()
    part_d()
    part_d_sweep()
    print()
