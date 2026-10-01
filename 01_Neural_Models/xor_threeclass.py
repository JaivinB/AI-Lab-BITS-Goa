"""
Lab 1, Task 5 - the same sensor input, now as a three-class decision.

    Class 0 : both sensors inactive     (0,0)
    Class 1 : sensors disagree          (0,1) or (1,0)
    Class 2 : both sensors active       (1,1)

Only the output layer and the loss change relative to xor_binary.py:

    2 inputs -> 2 hidden units (tanh) -> 3 logits,  cross-entropy loss

The hidden representation stays deliberately small so that the comparison with
the binary experiment is about the output design and nothing else.

Predictions recorded BEFORE running the code (Task 5 asks for these first):

    1. the final weight matrix is W2 with shape (3, 2) - three output units,
       each reading the two hidden units - plus a bias of shape (3,);
    2. three logits per example, so the logit tensor is (4, 3);
    3. softmax probabilities sum to one because softmax divides each exp(z_k)
       by the sum of all exp(z_j), so the components share a common
       denominator that is exactly their total;
    4. the logit gradient is p - y because for cross-entropy on a softmax
       output, dL_n/dz_k = p_k - y_k; the derivative of the log-normaliser
       contributes p_k and the derivative of the selected log-probability
       contributes -y_k.

Everything below is the verification of those four predictions.
"""

import torch
import torch.nn as nn

X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
TARGETS = torch.tensor([0, 1, 1, 2])             # class indices, not one-hot

CLASS_MEANING = {0: "both inactive", 1: "sensors disagree", 2: "both active"}

SEED = 0
STEPS = 5000
LR = 0.05


def build_model():
    """Identical body to the binary model; only the output width changes."""
    return nn.Sequential(
        nn.Linear(2, 2),
        nn.Tanh(),
        nn.Linear(2, 3),                         # three logits instead of one
    )


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def train(model):
    loss_fn = nn.CrossEntropyLoss()              # softmax + NLL, computed stably
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    initial_loss = None
    for step in range(STEPS):
        optimiser.zero_grad()
        loss = loss_fn(model(X), TARGETS)
        loss.backward()
        if step == 0:
            initial_loss = loss.item()
        optimiser.step()
    with torch.no_grad():
        final_loss = loss_fn(model(X), TARGETS).item()
    return initial_loss, final_loss


def main():
    print("torch", torch.__version__, "| seed", SEED, "| steps", STEPS,
          "| optimiser Adam(lr=%.2f)" % LR)

    torch.manual_seed(SEED)
    model = build_model()

    # ---------------------------------------------------------------- shapes
    banner("PREDICTION 1 and 2 - shapes of the new output layer")
    output_layer = model[2]
    print("W2 shape            : %s   (predicted (3, 2))" % (tuple(output_layer.weight.shape),))
    print("b2 shape            : %s      (predicted (3,))" % (tuple(output_layer.bias.shape),))
    with torch.no_grad():
        logits = model(X)
    print("logit tensor shape  : %s   (4 examples x 3 logits)" % (tuple(logits.shape),))
    print("logits per example  : %d      (predicted 3)" % logits.shape[1])

    # --------------------------------------------------------------- training
    initial_loss, final_loss = train(model)
    banner("TRAINING")
    print("Initial loss : %.6f   (ln 3 = %.6f if the model were uninformative)"
          % (initial_loss, torch.log(torch.tensor(3.0)).item()))
    print("Final loss   : %.6e" % final_loss)

    # ------------------------------------------------------- predicted classes
    banner("RESULTS - predicted class probabilities for all four inputs")
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
    predicted = probabilities.argmax(dim=1)

    print(" x1  x2   target        P(class 0)  P(class 1)  P(class 2)   predicted")
    for index in range(4):
        x1, x2 = X[index].tolist()
        p = probabilities[index].tolist()
        print("  %d   %d      %d       %10.6f  %10.6f  %10.6f        %d"
              % (x1, x2, TARGETS[index], p[0], p[1], p[2], predicted[index]))
    correct = int((predicted == TARGETS).sum().item())
    print()
    print("Correct: %d/4   ->   %s" % (correct, "PASS" if correct == 4 else "FAIL"))
    for index in range(4):
        print("  (%d,%d) -> class %d (%s)"
              % (X[index][0], X[index][1], predicted[index],
                 CLASS_MEANING[int(predicted[index])]))

    # ------------------------------------------- PREDICTION 3: normalisation
    banner("PREDICTION 3 - a softmax vector sums to one")
    example = 1                                  # the input (0,1)
    vector = probabilities[example]
    print("Input (%d,%d), softmax probability vector:" % tuple(X[example].tolist()))
    print("  %s" % [float("%.10f" % v) for v in vector.tolist()])
    print("  sum = %.10f" % vector.sum().item())
    print("  |sum - 1| = %.3e" % abs(vector.sum().item() - 1.0))
    print()
    print("Row sums for all four examples: %s"
          % [round(v, 10) for v in probabilities.sum(dim=1).tolist()])

    # ------------------------------------------ PREDICTION 4: p - y gradient
    banner("PREDICTION 4 - the logit gradient equals (p - y) / N")
    model.zero_grad()
    logits = model(X)
    logits.retain_grad()                         # expose dL/dz directly
    loss = nn.CrossEntropyLoss()(logits, TARGETS)
    loss.backward()
    autograd_logit_grad = logits.grad.clone()

    one_hot = torch.zeros(4, 3)
    one_hot[torch.arange(4), TARGETS] = 1.0
    with torch.no_grad():
        p_minus_y = (torch.softmax(logits, dim=1) - one_hot) / 4   # mean loss -> /N

    print("autograd dL/dz:")
    print(autograd_logit_grad)
    print()
    print("(p - y) / 4 computed by hand:")
    print(p_minus_y)
    print()
    print("max |autograd - (p - y)/4| = %.3e"
          % (autograd_logit_grad - p_minus_y).abs().max().item())
    print("-> the closed-form softmax/cross-entropy logit gradient is confirmed.")
    print("   The division by 4 is only the mean reduction over the batch.")

    # ---------------------------------------- optional shift-invariance check
    banner("OPTIONAL DIAGNOSTIC - adding a constant to every logit")
    with torch.no_grad():
        logits = model(X)
        shifted = logits + 100.0
        base_probabilities = torch.softmax(logits, dim=1)
        shifted_probabilities = torch.softmax(shifted, dim=1)
    print("max |softmax(z) - softmax(z + 100)| = %.3e"
          % (base_probabilities - shifted_probabilities).abs().max().item())
    print()
    print("The probabilities are unchanged apart from floating-point roundoff,")
    print("because exp(z_k + c) / sum_j exp(z_j + c) cancels the factor exp(c)")
    print("in the numerator and the denominator.")
    print()
    print("A naive implementation nonetheless breaks, because exp() of a large")
    print("logit overflows before the cancellation can happen:")
    large = torch.tensor([[1000.0, 1001.0, 1002.0]])
    naive = torch.exp(large) / torch.exp(large).sum()
    stable = torch.softmax(large, dim=1)
    print("  naive exp(z)/sum(exp(z)) on z = %s  ->  %s"
          % (large.tolist()[0], naive.tolist()[0]))
    print("  torch.softmax on the same z                      ->  %s"
          % [round(v, 6) for v in stable.tolist()[0]])
    print()
    print("Stable implementations subtract max_j z_j first.  That shift leaves")
    print("the result mathematically identical, by the cancellation above, but")
    print("makes the largest exponent exp(0) = 1, so nothing overflows.")

    # ------------------------------------- what carries over to a real LM
    banner("SCALING REMARK (Task 5 'Think About It')")
    print("Treating next-token prediction as classification over a vocabulary")
    print("leaves the following unchanged as the number of classes grows from 3")
    print("to tens of thousands: the softmax normalisation, the cross-entropy")
    print("loss, and the gradient p - y.  What changes is everything around it -")
    print("the output matrix becomes (vocabulary x hidden) and dominates the")
    print("parameter count, the normalising sum over classes becomes the")
    print("expensive part of each step, the targets become sparse one-hot rows")
    print("stored as indices, and the context is no longer two binary inputs but")
    print("a long sequence requiring an architecture that can represent it.")
    print()


if __name__ == "__main__":
    main()
