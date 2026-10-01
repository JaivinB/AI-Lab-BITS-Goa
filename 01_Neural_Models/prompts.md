# LLM prompts used in Lab 1

The assistant used was Claude. The prompts below are the exact ones issued, in
order. The design in Task 2 was fixed on paper first; the LLM was asked only to
translate that design into PyTorch, never to choose the architecture or the task.

## Prompt 1 - first implementation (Task 3)

> Generate minimal PyTorch code for the following model and dataset. Do not
> change the architecture or task.
>
> Dataset: the four XOR examples exactly, (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
> Model: 2 inputs -> 2 hidden units -> 1 output. Hidden activation tanh. The
> output must be a raw logit trained with BCEWithLogitsLoss, not an explicit
> sigmoid followed by BCELoss. Random weight initialisation. Full-batch
> training for a few thousand CPU steps.
>
> After training, report the final loss, all four probabilities, the thresholded
> labels, and one parameter-gradient tensor obtained after backward(). Set a
> random seed for reproducibility and explain each test in one sentence.

## Prompt 2 - the symmetry experiment (Task 4, Part C)

> Take the same script and add a variant in which every weight and bias is set
> to zero before training, leaving the architecture unchanged. Print the two
> rows of the first-layer weight matrix at several early steps and after
> training, and report whether they stay identical.

## Prompt 3 - the activation sweep (Task 4, Part D)

> Add an experiment that trains the randomly initialised network three times,
> changing only the hidden activation: sigmoid, tanh, ReLU. Use the same seed
> for all three so the initial weights match. For each run record the final
> loss, whether all four examples are classified correctly, and the Euclidean
> norm of the first-layer gradient at an early training step (step 10).

## Prompt 4 - distinguishing a small gradient from a dead unit

> For each activation, also capture the hidden pre-activations z at step 10 and
> print f'(z) alongside them, so that a saturated sigmoid (f'(z) small but
> positive) can be told apart from a negative ReLU input (f'(z) exactly zero).

## Prompt 5 - the three-class extension (Task 5)

> Modify only the output layer and the loss of the previous code. Keep 2 inputs
> and 2 tanh hidden units. Replace the single logit with three logits and train
> with multiclass cross-entropy, where class 0 is (0,0), class 1 is (0,1) and
> (1,0), and class 2 is (1,1).
>
> Report the predicted class probabilities for all four inputs, print the
> softmax vector for one example and verify numerically that it sums to
> approximately 1, and add a diagnostic that adds 100 to all three logits before
> softmax and checks the probability vector is unchanged.

## Prompt 6 - verifying the closed form rather than trusting it

> Add a check that compares autograd's dL/dz on the logits against (p - y)/N
> computed by hand from the softmax probabilities and the one-hot targets, and
> report the maximum absolute difference.

## Changes made to the generated code before accepting it

Seven corrections were needed. The first four mattered scientifically.

1. **The finite-difference check was wrong in float32.** The generated check ran
   in the default dtype with `h = 1e-4` and reported disagreements of about
   `4e-4` - the same order as the gradient entries themselves, so the check
   could not discriminate between a correct and an incorrect gradient. It was
   rewritten to run on a `float64` copy of the network with `h = 1e-6`, which
   brought the worst disagreement to `5.4e-11`. A test that cannot fail for the
   right reason is worse than no test.

2. **The zero-initialisation experiment demonstrated the wrong mechanism.** With
   every parameter zeroed, `W2 = 0`, so no signal reaches the hidden layer and
   `dL/dW1` is identically zero: the rows stay identical only because nothing
   moves at all. That is a vanished gradient, not the symmetry problem. A second
   variant (`C2`) was added in which the two hidden units are initialised
   identically but to nonzero values, so the gradients are nonzero and *equal*.
   This is the mechanism the lab actually asks about.

3. **One seed was being presented as evidence about activations.** The generated
   sweep ran each activation once. At seed 0, ReLU's two hidden units happen to
   be dead for all four inputs, so the comparison said more about the draw of
   initial weights than about ReLU. A five-seed sweep was added, which reverses
   the apparent conclusion for sigmoid (3/4 at seed 0, but 4/5 seeds solved).

4. **The output used `nn.Sigmoid()` followed by `BCELoss`** in the first reply,
   despite the prompt asking for logits. It was changed to a bare `nn.Linear`
   output with `BCEWithLogitsLoss`, which is the numerically stable pairing.

5. A `print` in the activation summary contained a `%d` placeholder with no
   argument, so it printed the literal text `step %d`. Fixed.

6. The generated code reported only the final loss. The initial loss was added,
   since "learning occurred" is a claim about a change, not about one number.

7. `logits.retain_grad()` had to be added before `backward()` to expose `dL/dz`;
   without it the gradient on a non-leaf tensor is discarded and the `p - y`
   check cannot be performed at all.
