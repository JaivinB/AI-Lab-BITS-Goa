# Lab 5 — Bayesian Networks and Autoregressive Language Models

```
Probability -> Bayesian Network -> Autoregressive Model -> Language Generation
```

A first- and second-order n-gram model built by counting, treated explicitly as
a Bayesian network over the sequence variables, and tested against the
probabilistic properties it is supposed to satisfy.

| File | Contents |
|---|---|
| `ngram_lm.py` | `NgramModel(order)` — counting, CPTs, prediction, greedy and sampling generation, normalisation test, chain-rule evaluation |
| `generated_sentences.txt` | Twenty sampled sentences from each model |
| `prompts.md` | The exact LLM prompts and the six corrections made before accepting the code |
| `results.txt` | Captured program output |

```bash
python ngram_lm.py
```

Pure Python 3 — no machine-learning library, no pretrained model. Counts in
dictionaries, probabilities by division, generation by `random.random()`.

**Corpus** (6 sentences, lower-cased, padded with `<START>`/`<END>`):

```
the cat sat on the mat        the dog ran to the park
the cat sat on the rug        the cat ran to the park
the dog sat on the mat        the dog sat on the rug
```

Vocabulary `|V| = 11`: `the, cat, dog, sat, ran, on, to, mat, rug, park, <END>`.
`<START>` can appear in a context but is never generated; `<END>` is generated
but never appears in a context.

---

## Part I — Question 1: why is the chain-rule decomposition useful for generating text?

Because it converts one intractable problem into a sequence of tractable ones.
Modelling `P(x1, ..., xT)` directly would need a probability for every possible
sentence — `|V|^T` numbers, which is not estimable and not even storable.
The factorisation

```
P(x1..xT) = P(x1) * prod_t P(xt | x1..x_{t-1})
```

is exact, not an approximation, and it replaces that single enormous
distribution with `T` distributions over just `|V|` outcomes each. Three
consequences matter for generation. It is *constructive*: it gives a procedure —
sample `x1`, condition on it, sample `x2`, and so on — so generating a sentence
is just `T` draws from small distributions. It is *incremental*: text can be
produced left to right, without deciding the length first. And it *localises
learning*: every token of every sentence becomes a training example for
"predict the next token given the context", so one short corpus supplies many
observations.

## Part II — Question 2: what independence assumption does `X1 → X2 → X3 → ...` make?

That each token is conditionally independent of everything before its immediate
predecessor, given that predecessor:

```
P(Xt | X1, ..., X_{t-1}) = P(Xt | X_{t-1})
```

equivalently `Xt ⊥ X1..X_{t-2} | X_{t-1}` — the first-order Markov property. The
chain rule itself is exact; this assumption is where the modelling error enters,
and it is what makes the network a *chain* rather than a fully connected graph.
The second-order model weakens it to
`P(Xt | X1..X_{t-1}) = P(Xt | X_{t-2}, X_{t-1})`, giving `X_{t-2} → Xt ← X_{t-1}`.

The corpus shows the assumption failing concretely. The token `the` occurs both
sentence-initially and mid-sentence, and what may follow differs completely —
but a first-order model has one row for `the` and must blend both uses, which is
exactly why it generates `the mat` as a complete sentence (see Part IX).

---

## Part IV — Question 3: the conditional probability table

`P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)`

| Context | Distribution | Zero-probability successors |
|---|---|---|
| `the` (12 transitions) | `cat` 3/12 = 0.250, `dog` 3/12 = 0.250, `mat` 2/12 = 0.167, `park` 2/12 = 0.167, `rug` 2/12 = 0.167 | 6: `<END>`, `on`, `ran`, `sat`, `the`, `to` |
| `cat` (3) | `sat` 2/3 = 0.667, `ran` 1/3 = 0.333 | 9 |
| `dog` (3) | `sat` 2/3 = 0.667, `ran` 1/3 = 0.333 | 9 |
| `sat` (4) | `on` 4/4 = 1.000 | 10 |
| `ran` (2) | `to` 2/2 = 1.000 | 10 |
| `<START>` (6) | `the` 6/6 = 1.000 | 10 |

**Zero-probability transitions.** Most of the table is zero: of the 121 cells a
complete `11 × 11` CPT would have, only 17 are non-zero — 86% of the table is
unobserved. Some zeros are *linguistically* right (`P(the | the) = 0`), but the
model cannot tell those apart from zeros that are merely *unobserved*: `cat` is
never followed by `<END>` in six sentences, so `P(<END> | cat) = 0` and the model
asserts that a sentence can never end on `cat` — an accident of the corpus
promoted to a certainty. Note also that `sat` and `ran` have probability 1.0 on
their single successor, which is maximum-likelihood estimation being
overconfident on counts of 4 and 2.

---

## Part VI — Inspecting the generated code

**Question 4 — where are the transition counts stored?** In
`NgramModel.counts`, a `defaultdict(lambda: defaultdict(int))` mapping a context
*tuple* to a dictionary of `{next token: count}`. `counts[("the",)]["cat"]` is
`C(the, cat) = 3`. It is a sparse representation: only observed pairs are stored,
which is why 17 entries hold a table with 121 cells.

**Question 5 — where is `P(Xt | X_{t-1})` computed?** In
`NgramModel.distribution`, and nowhere else. It fetches the count row, sums it,
and divides. Probabilities are computed on demand rather than cached, so there
is exactly one place where normalisation can be wrong — which is what makes the
Part VII test meaningful.

**Question 6 — how does the program choose the next word?** Both, selectably.
`predict()` returns `argmax_w P(w | context)` (Mode A, greedy); `_sample()`
draws from the distribution by inverse-CDF sampling (Mode B). **The difference:**
greedy is deterministic and returns the single most probable continuation, so it
collapses the distribution to its mode and discards everything the model knows
about alternatives — identical output every run. Sampling is stochastic and
visits each successor with its own probability, so it reproduces the model's
distribution rather than just its peak. Greedy answers "what is most likely
next?"; sampling answers "what does this distribution look like?". Only sampling
is *generation* in the probabilistic sense of Part IX, `Xt ~ P(Xt | X_{t-1})`.

**Question 7 — what if a word has no observed transition?** `distribution()`
returns `{}`, generation stops and reports the sentence as truncated. The
generated code originally fell back on a uniform distribution over the
vocabulary — a smoothing decision, silently invented, which would have made the
model look as though it had information where it has none. Returning `{}` keeps
the failure visible. This matters here: the second-order model has **106 of 121
possible contexts with no distribution at all**. A real system needs smoothing
(add-one, or backing off to the lower-order model), and that is a deliberate
modelling choice, not a default.

---

## Part VII — Testing the probability model

For every context `w`, `sum_v P(v | w)` must equal 1. All 11 first-order
contexts and all 15 second-order contexts:

```
First-order  : worst deviation from 1.0: 0.000e+00  ->  PASS
Second-order : worst deviation from 1.0: 0.000e+00  ->  PASS
```

A second, independent check: the chain rule evaluated factor by factor on a
training sentence, which tests that the CPTs are the ones the Bayesian network
specifies and not merely rows that happen to sum to 1.

```
P(the | <START>) = 1.0000      P(on | sat)  = 1.0000
P(cat | the)     = 0.2500      P(the | on)  = 1.0000
P(sat | cat)     = 0.6667      P(mat | the) = 0.1667
                                product     = 0.027778
```

**Question 8 — if one of the totals were 0.87, what would that tell you?** That
the implementation is broken, not that the model is a poor fit — a normalised
conditional distribution sums to 1 by construction, whatever the data. The
likely causes are all arithmetic: dividing by a global token count instead of the
row total `sum_k C(w_i, w_k)`; building the denominator from a stale copy of the
counts; dropping a successor from the row while leaving it in the total (0.87
being short, not over, suggests exactly this); or double-counting the denominator
by including `<START>` positions. It is the ideal kind of test, because the
correct answer is known *a priori* from the specification and does not depend on
the corpus — which is what Part VII means by testing a probabilistic invariant.

---

## Part VIII — Predicting the next word

| Context | `argmax P(w | context)` | P | Full distribution |
|---|---|---|---|
| `the` | `cat` — **tie with `dog`** | 0.250 | cat 0.250, dog 0.250, mat 0.167, park 0.167, rug 0.167 |
| `cat` | `sat` | 0.667 | sat 0.667, ran 0.333 |
| `dog` | `sat` | 0.667 | sat 0.667, ran 0.333 |
| `sat` | `on` | 1.000 | on 1.000 |
| `ran` | `to` | 1.000 | to 1.000 |
| `on` | `the` | 1.000 | the 1.000 |
| `to` | `the` | 1.000 | the 1.000 |

The tie on `the` is reported rather than hidden: `argmax` has to break it
arbitrarily, and the choice silently determines what greedy decoding produces.

**Question 9 — are the most probable predictions what you would personally
expect?** Partly, and the mismatches are the informative part. `P(on | sat) = 1`
and `P(to | ran) = 1` match intuition but are far too confident — they rest on 4
and 2 observations, and a human knows a cat can sit *still* or sit *down*. More
revealing, `P(cat | the) = P(dog | the) = 0.25` while `P(mat | the) = 0.167`: the
model finds `the cat` and `the mat` nearly equally plausible, whereas a human
reader strongly expects a *subject* after a sentence-initial `the` and a *noun
object* after `on the`. The model cannot make that distinction because it does
not know which `the` it is looking at — the exact failure of the first-order
Markov assumption.

What this shows about the difference between a probability model and human
linguistic expectation: the model reports *corpus frequency* under a fixed,
deliberately impoverished conditioning context. Human expectation draws on
syntax, position, meaning and world knowledge, and is not a frequency estimate at
all. Where they agree, it is because frequency is a good proxy for what the
corpus happens to contain — not because the model has acquired the knowledge.
Scaling the context (as the second-order model does, and as a transformer does
much further) narrows the gap without changing the nature of the quantity being
estimated.

---

## Parts IX and X — Generation: greedy against sampling

### First-order model

**Mode A (greedy), 5 sentences** — all five identical, all five truncated at the
length cap:

```
the cat sat on the cat sat on the cat sat on the cat sat on the cat ...
distinct sentences: 1 of 5      hit the length cap: 5 of 5
```

**Greedy decoding in the first-order model never terminates.** The argmaxes form
a cycle: `the → cat` (0.25, after breaking the tie) `→ sat` (0.667) `→ on` (1.0)
`→ the` (1.0) `→ cat → ...`. `<END>` is never the most probable successor of any
token on the cycle, so it is never emitted. Without the length cap the program
hangs — this was a real bug in the generated code (`prompts.md`, correction 4),
and turning the hang into a reported measurement is what made the behaviour
visible.

**Mode B (sampling), 5 sentences:**

```
1. the rug                                    4. the rug
2. the dog sat on the cat ran to the park     5. the mat
3. the park
distinct sentences: 4 of 5      hit the length cap: 0 of 5
```

**Twenty sampled sentences:** 12 distinct of 20; **2 of 20 appear verbatim in the
corpus**; none truncated. Examples of the failure modes:

- `the mat` — a complete two-word "sentence", because `P(mat | the) = 0.167` and
  `P(<END> | mat) = 1.0`. Nothing stops it: the model has no notion of a sentence
  needing a verb.
- `the dog sat on the cat ran to the dog sat on the dog ran to the cat sat on the rug` —
  the chain re-enters `the` and keeps going, splicing clauses together.

**Question 10 — which mode produces more variation, and why?** Sampling, by a
wide margin: 12 distinct sentences of 20 against 1 of 5 for greedy. Greedy is a
deterministic function of the model, so with a fixed tie-break it has exactly
one output — all variation is discarded the moment the distribution is replaced
by its mode. Sampling draws each successor with its own probability, so every
non-degenerate context is a branch point; the number of distinct outputs grows
multiplicatively with the number of such contexts along the path. The trade is
visible in the same numbers: greedy guarantees locally most-probable tokens and
here produces an infinite loop, while sampling terminates naturally — because
`<END>` has non-zero probability and so eventually gets drawn — at the cost of
sometimes producing `the mat`. Diversity and coherence pull against each other,
and neither mode is simply better.

### Second-order model

**Mode A (greedy), 5 sentences** — identical, but now it **terminates**:

```
the cat sat on the mat
distinct sentences: 1 of 5      hit the length cap: 0 of 5
```

The extra token of context breaks the cycle. `on the` is a different context
from `<START> the`, so `P(mat | on the) = 0.5` and then
`P(<END> | the mat) = 1.0`. The first-order model could not distinguish those
two uses of `the` and so could never reach `<END>` greedily. This is the
clearest single result in the lab: **adding one token of context changed greedy
decoding from non-terminating to correct.**

**Mode B (sampling), twenty sentences:** 6 distinct of 20, and **20 of 20 appear
verbatim in the training corpus** — the 6 distinct outputs are exactly the 6
training sentences. The second-order model has, on this corpus, memorised it.

---

## Part XI — Question 11: how the second-order model differs

**1. Graph structure.** The chain `X_{t-1} → Xt` becomes a converging
connection `X_{t-2} → Xt ← X_{t-1}`: each variable now has two parents, and the
factorisation for four tokens becomes
`P(X1) P(X2|X1) P(X3|X1,X2) P(X4|X2,X3)`.

**2. Conditional probability table.** The table is indexed by *pairs*, so it has
`|contexts|² ` rows instead of `|contexts|`: 121 possible rows against 11. Each
row is still a distribution over `|V| = 11` successors, so the complete table
grows from 121 cells to 1331.

**3. Context available for prediction.** Two tokens instead of one, which lets
the model separate uses of a word that the first-order model must merge. Measured
effect: `P(cat | the) = 0.25` first-order, but `P(cat | <START> the) = 0.50` and
`P(mat | on the) = 0.50` second-order — the two senses of `the` are now in
different rows.

**4. Data needed.** Far more, and this is the cost. Measured on this corpus:

| | order 1 | order 2 |
|---|---|---|
| contexts observed | 11 | 15 |
| contexts possible | 11 | 121 |
| **contexts with no distribution** | **0** | **106** |
| non-zero CPT entries stored | 17 | 19 |
| free parameters estimated | 6 | 4 |
| full CPT size | 121 | 1331 |

The second-order model has no opinion whatsoever about 106 of 121 contexts —
88% of its table is unreachable from six sentences, where the first-order model
covers every context it could encounter. Note the inversion in the last two
rows: order 2 stores *more* entries but estimates *fewer free* parameters (4
against 6), because most of its rows have a single observed successor and so are
deterministic. That is overfitting made numerical: more table, less information
per row.

---

## Part XIII — Comparing the two models

| Measure | order 1 | order 2 |
|---|---|---|
| Distinct sentences in 20 sampled | 12 | 6 |
| Appearing verbatim in the corpus | 2 | **20** |
| Hit the length cap | 0 | 0 |
| Greedy decoding terminates | **no** | yes |
| `P(the cat sat on the mat)` | 0.0278 | 0.1667 |
| `P(the cat ran to the park)` | 0.0139 | 0.1667 |
| `P(the mat sat on the cat)` | 0.0000 | 0.0000 |

**Qualitative coherence.** Order 1 produces ungrammatical fragments (`the mat`,
`the park`) and run-on splices (`the dog sat on the cat ran to the dog sat on
...`). Order 2 produces only well-formed sentences — because it produces only
*training* sentences. Neither model generalises: both assign probability exactly
0 to `the mat sat on the cat`, a perfectly grammatical sentence over the same
vocabulary. Order 1 at least assigns probability to a well-formed
sentence it never saw — `the cat ran to the rug`, which appears twice among its
20 samples and which order 2 can *never* generate, since `P(park | to the) = 1`
leaves `rug` with probability 0 in that context. Order 2's every sample was a
memorised sentence. The apparent improvement in coherence
is, on a corpus this small, indistinguishable from memorisation — which is why
the lab asks for examples rather than a single score.

**Question 12 — why does more context potentially improve prediction, and
simultaneously make the model harder to estimate?** More context improves
prediction because it removes a false independence assumption: tokens *do* depend
on more than their immediate predecessor, and the first-order model's single row
for `the` provably blends two incompatible distributions. Conditioning on two
tokens separates them, and the measured effect is sharper predictions
(`P(cat | <START> the) = 0.50` against `0.25`) and greedy decoding that
terminates.

It is simultaneously harder to estimate because the CPT grows *exponentially* in
the context length while the data stays fixed. With context length `n` the table
has `|V|^n` rows, each needing its own distribution over `|V` successors —
`121 → 1331` cells here for one extra token, and `|V|^3 · |V| = 14641` for one
more. The corpus provides 42 token transitions however the model is indexed, so
every additional token of context spreads the same evidence over `|V|` times as
many rows. The result is the 106 empty contexts, and rows estimated from 1 or 2
observations that assert probability 1.0. This is the bias–variance trade-off
expressed as table size: short contexts give a confidently wrong model, long
contexts an unconfident or memorising one. Practical systems resolve it by
*sharing* statistical strength instead of partitioning it — backing off to
lower-order models, or, in a neural model, replacing the table with a function
whose parameter count does not grow with the number of possible contexts.

---

## Part XV — Question 13: why is Approach B preferable?

Approach A ("write a Python language model for me") delegates the model, not just
the code. Approach B ("implement `P(Xt | X_{t-1})`, estimated from transition
counts, with sampling-based generation") delegates only the implementation, and
it is preferable on all five counts the lab lists.

- **Specifying the intended behaviour** — Approach A has no criterion for
  success, so any program that emits text satisfies it. Approach B names the
  estimator and the generation rule, so "it runs" and "it is right" become
  different questions. Concretely, the prompt's clause "stop when `<END>` is
  generated" is what exposed the non-terminating greedy loop as a defect rather
  than a quirk.
- **Understanding the representation** — Approach B forces you to decide, before
  seeing code, that a state is a context tuple and the parameters are
  normalised counts. Approach A would likely have returned something
  transformer-shaped or an API call, and the Bayesian-network structure the lab
  is about would never have appeared.
- **Validating the generated implementation** — you can only check code against
  a specification you wrote down. Six corrections were needed here, and each was
  found by comparing the code to a clause of the prompt.
- **Testing probabilistic invariants** — `sum_v P(v | w) = 1` is derivable from
  the specification without any data, so it is a test whose correct answer is
  known in advance. Approach A gives you nothing to test against; you would be
  reduced to judging whether the output "looks like English".
- **Distinguishing implementation from model** — the clearest case is correction
  5: the generated code returned a *uniform distribution* for unseen contexts.
  That is a smoothing decision — a change to the model — introduced as an
  implementation convenience. With the model specified separately, it is
  visible as a change of model and rejected; without, it would have silently
  made 106 empty contexts look informative.

---

## Part XX — Question 14: what did thinking of this as a Bayesian network add?

**A factorisation of the joint distribution.** It made explicit that the model's
object is `P(x1..xT)`, and that the whole thing is a product of small
conditionals. That is what allowed the sequence probability to be computed
factor by factor — `0.027778` for `the cat sat on the mat` — and it is why six
sentences can estimate anything at all.

**A representation of dependencies, and a way to reason about independence
assumptions.** The graph *is* the assumption. `X1 → X2 → X3` says
`Xt ⊥ X1..X_{t-2} | X_{t-1}`, and drawing it that way made the model's defect
predictable before it was observed: one row for `the` must merge the
sentence-initial and the post-`on` uses. The second-order network
`X_{t-2} → Xt ← X_{t-1}` is a different, weaker assumption, and changing the
graph is exactly what fixed non-terminating greedy decoding. Without the network
view, "add more context" is a hyperparameter; with it, it is a change to the
dependency structure with a predictable consequence.

**A way to test whether an implementation matches its probabilistic
specification.** This is the most practically valuable one. Because the CPT rows
are distributions, `sum_v P(v | w) = 1` must hold for every context — a test with
a known answer, independent of the data and of any judgement about output
quality. The same view produced the chain-rule check, and it is what classifies
the uniform-distribution fallback as a model change rather than a bug fix.

**A way to understand the effect of increasing context.** The network makes the
cost of context quantitative rather than vague: one extra parent multiplies the
number of CPT rows by `|V|`, so the 121-cell table becomes 1331 and 106 of 121
contexts go unobserved. "More context is better" and "more context needs more
data" stop being competing intuitions and become the same arithmetic.

---

## Reflection on the use of the LLM

The LLM wrote the counting loop, the inverse-CDF sampler, the CPT formatting and
the table layout quickly and mostly correctly, and the second-order version
followed from a one-line prompt. Six corrections were needed (`prompts.md`), of
which three changed what the program *meant* rather than how it was written.

**The one worth examining in detail** — the fifth. The generated
`distribution()` ended with a fallback: if the context had never been observed,
return a uniform distribution over the vocabulary. It is a defensible line of
code, it prevents a crash, and nothing in the output marks it. But it is a
*smoothing* decision — a change to the probabilistic model — introduced as
error handling. Had it stayed, the second-order model's 106 unobserved contexts
would each have reported a confident-looking distribution over 11 tokens, the
normalisation test would have passed on all 121 contexts instead of 15 (every
row summing to 1), and Part XIII's sparsity comparison would have shown no
sparsity at all. The headline finding of the lab would have been quietly erased
by a two-line `if`. It was caught by reading the code against the specification
— "construct the conditional distribution from transition counts" licenses no
fallback — and not by any test, because the behaviour it produced looked
*better* than the correct one.

The other two substantive corrections are of the same kind: the non-terminating
greedy loop (a hang the generated code could not report, because it never
measured termination), and the duplicated second-order class (correct output,
but it hides the fact that order is the single parameter the Bayesian network
structure turns on). In every case the code ran and the numbers were plausible.
What testing establishes is not that a program runs, but that it is estimating
the distribution you specified.

```
P(next token | previous tokens)
```

The machinery used to estimate that probability differs enormously between a
dictionary of counts and a transformer. The probabilistic question does not.
