# LLM prompts used in Lab 5

The assistant used was Claude. The probabilistic model — the variables, the
dependencies, the distribution being estimated — was specified before any
prompt was issued. This is Approach B from Part XV of the lab sheet, not
Approach A.

## Prompt 1 — the first-order model (Part V)

> Write a simple Python implementation of a first-order autoregressive language
> model. The model should:
> 1. take a list of tokenised sentences as training data;
> 2. count transitions between consecutive tokens;
> 3. construct the conditional distribution `P(Xt | X_{t-1})`;
> 4. display the probabilities for a specified previous token;
> 5. predict the most probable next token;
> 6. generate a sentence by repeatedly sampling the next token;
> 7. stop when the `<END>` token is generated.
>
> Do not use a machine-learning library or a pretrained language model. Use
> ordinary Python data structures and random sampling.

## Prompt 2 — the normalisation test (Part VII)

> Add a test that, for every context the model has seen, sums the conditional
> probabilities over all successors and reports the deviation from 1. Print one
> line per context and a worst-case figure, and make it say PASS or FAIL.

## Prompt 3 — greedy against sampling (Part X)

> Add two generation modes: Mode A always takes `argmax P(w | context)`, Mode B
> samples from `P(w | context)`. Generate five sentences with each and report
> how many are distinct.

## Prompt 4 — the second-order model (Part XII)

> Modify the existing first-order autoregressive model into a second-order
> model. The model should estimate `P(Xt | X_{t-2}, X_{t-1})`. Represent the
> model using counts of observed triples and use these counts to construct
> conditional probability distributions. Do not replace the model with a neural
> network or a pretrained language model.

## Prompt 5 — the comparison (Part XIII)

> Compare the first- and second-order models on: number of distinct
> parameters, number of zero-probability contexts, diversity of generated
> sentences, and how many generated sentences appear verbatim in the training
> corpus. Also evaluate the probability of a given sentence under each model by
> the chain rule.

## Changes made to the generated code before accepting it

1. **The second-order model was generated as a separate copy of the class.**
   Asked to "modify the first-order model into a second-order model", the LLM
   duplicated the whole class with `(w1, w2)` keys instead of `w` keys — two
   bodies of near-identical code that would then drift apart. Since the only
   thing that changes is the *length of the conditioning context*, which is
   precisely what the Bayesian network structure changes, it was refactored into
   one `NgramModel(order)` class. The first-order model is `order=1` and the
   second-order model is `order=2`.

2. **Only one `<START>` was prepended, for both orders.** With a second-order
   model this leaves the first real token with a context of length 1, so
   `P(X1 | <START>)` is estimated in a different table from every other
   conditional and the tuple lengths become inconsistent. The padding is now
   `order` copies of `<START>`, which gives the first token a full-length
   context and removes the special case.

3. **Sampling iterated over an unordered dictionary.** The generated
   `_sample` walked `distribution.items()` in insertion order, so the mapping
   from the random number to the token depended on the order the corpus happened
   to be read in. Two runs with the same seed could then disagree after any
   change to the data. Iteration is now over `sorted(...)`, which makes the
   sampler reproducible.

4. **Generation could not terminate, and said nothing about it.** The generated
   loop was `while True` with no cap. With the first-order model and greedy
   decoding this never returns: `the → cat → sat → on → the → ...` is a cycle
   that never reaches `<END>`. A `max_length` cap was added *and* the function
   now returns whether the cap was hit, which is what turned a hang into the
   most interesting measurement in the lab (Question 10).

5. **An unseen context returned a uniform distribution.** The generated
   `distribution()` fell back to "assume every token equally likely" when the
   context was absent. That is a smoothing decision, silently invented, and it
   would have made the 106 unseen second-order contexts look like they carried
   information. It now returns `{}`, so the model's inability to predict there
   is visible rather than papered over. See Question 7.

6. The parameter count returned the number of stored CPT entries. That is the
   memory footprint, not the number of free parameters: a row with `k`
   successors has `k - 1` free probabilities, because it must sum to one. Both
   figures are now reported separately, since they tell different stories.
