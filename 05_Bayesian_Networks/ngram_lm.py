"""
Lab 5 - Bayesian networks and autoregressive language models.

The chain rule factorises any joint distribution over a sequence:

    P(x1..xT) = P(x1) * prod_{t=2..T} P(xt | x1..x_{t-1})

An n-gram model truncates the conditioning context to a fixed window, which
turns the factorisation into a Bayesian network with a specific, inspectable
structure:

    order 1:  X1 -> X2 -> X3 -> ...        P(Xt | X_{t-1})
    order 2:  X_{t-2} -> Xt <- X_{t-1}     P(Xt | X_{t-2}, X_{t-1})

Both orders are instances of one class below.  `NgramModel(order=1)` is the
first-order model and `NgramModel(order=2)` the second-order one: the only
difference is the length of the tuple used as the conditioning context, which
is exactly what changes in the Bayesian network.

No machine-learning library and no pretrained model: counts in dictionaries,
probabilities by division, generation by `random.random()`.
"""

import random
from collections import defaultdict

START, END = "<START>", "<END>"

CORPUS = """\
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""


def tokenise(corpus=CORPUS, order=1):
    """Lower-case, split on whitespace, pad with START and END.

    `order` copies of START are prepended, so that the first real token has a
    full-length context and P(X1 | START..) is estimated like any other
    conditional rather than as a special case.
    """
    sentences = []
    for line in corpus.strip().splitlines():
        tokens = line.lower().split()
        if tokens:
            sentences.append([START] * order + tokens + [END])
    return sentences


class NgramModel:
    """A fixed-window autoregressive model estimated by counting.

    counts[context][next_token] is C(context, next_token).  `context` is a
    tuple of the previous `order` tokens, so order 1 gives P(Xt | X_{t-1}) and
    order 2 gives P(Xt | X_{t-2}, X_{t-1}).
    """

    def __init__(self, order=1):
        self.order = order
        self.counts = defaultdict(lambda: defaultdict(int))   # the CPT, as counts
        self.vocabulary = set()

    def train(self, sentences):
        """Count every observed (context -> next token) transition."""
        for tokens in sentences:
            self.vocabulary.update(tokens)
            for index in range(self.order, len(tokens)):
                context = tuple(tokens[index - self.order:index])
                self.counts[context][tokens[index]] += 1
        self.vocabulary.discard(START)      # START is never generated
        return self

    def distribution(self, context):
        """P(Xt | context), normalised from the counts.

        C(w_context, w) / sum_k C(w_context, w_k).  Returns {} for a context
        never seen in training - the model has no opinion at all there, which
        is the n-gram model's central weakness.
        """
        context = self._as_context(context)
        row = self.counts.get(context)
        if not row:
            return {}
        total = sum(row.values())
        return {token: count / total for token, count in row.items()}

    def predict(self, context):
        """argmax_w P(w | context), with ties reported rather than hidden."""
        distribution = self.distribution(context)
        if not distribution:
            return None, []
        best = max(distribution.values())
        tied = sorted(t for t, p in distribution.items() if p == best)
        return tied[0], tied

    def generate(self, mode="sample", max_length=30, rng=random):
        """Generate a sentence, stopping when END is produced.

        mode "sample" draws Xt ~ P(Xt | context); mode "greedy" always takes
        the argmax.  `max_length` is a safety cap, and whether it is reached is
        reported, because greedy generation is not guaranteed to terminate.
        """
        context = tuple([START] * self.order)
        produced = []
        truncated = False
        for _ in range(max_length):
            distribution = self.distribution(context)
            if not distribution:
                truncated = True
                break
            if mode == "greedy":
                token, _tied = self.predict(context)
            else:
                token = self._sample(distribution, rng)
            if token == END:
                break
            produced.append(token)
            context = tuple(list(context[1:]) + [token]) if self.order else ()
        else:
            truncated = True
        return " ".join(produced), truncated

    @staticmethod
    def _sample(distribution, rng):
        """Inverse-CDF sampling from an explicit probability dictionary."""
        threshold = rng.random()
        cumulative = 0.0
        for token, probability in sorted(distribution.items()):
            cumulative += probability
            if threshold < cumulative:
                return token
        return sorted(distribution)[-1]      # floating-point guard

    def _as_context(self, context):
        """Accept a bare string for order 1, a tuple for any order."""
        if isinstance(context, str):
            context = (context,)
        return tuple(context)

    # ----------------------------------------------------------- diagnostics
    def parameter_count(self):
        """Distinct free probabilities actually estimated from the data.

        Each context with k observed successors contributes k - 1 free
        parameters, because the row must sum to one.
        """
        return sum(len(row) - 1 for row in self.counts.values())

    def stored_entries(self):
        """Non-zero CPT entries held in memory."""
        return sum(len(row) for row in self.counts.values())

    def observed_contexts(self):
        return len(self.counts)

    def context_alphabet(self):
        """Tokens that can legitimately appear in a context.

        END cannot: nothing follows the end of a sentence.  START can, and is
        not in the vocabulary, since it is never generated.  Stating this
        explicitly matters, because the count of possible contexts is what the
        order-1 against order-2 comparison turns on.
        """
        return (self.vocabulary - {END}) | {START}

    def possible_contexts(self):
        """|context alphabet|^order - how large a complete CPT would have to
        be if every context were observed."""
        return len(self.context_alphabet()) ** self.order

    def unseen_contexts(self):
        """Contexts for which the model has no distribution at all."""
        return self.possible_contexts() - self.observed_contexts()


# ---------------------------------------------------------------------------
# Tests of the probabilistic specification (Part VII)
# ---------------------------------------------------------------------------
def normalisation_test(model, label):
    """For every context w, sum_v P(v | w) must equal 1."""
    print("%s - sum_v P(v | context) for every observed context:" % label)
    worst = 0.0
    for context in sorted(model.counts):
        total = sum(model.distribution(context).values())
        worst = max(worst, abs(total - 1.0))
        print("  %-28s %.10f" % (" ".join(context), total))
    print("  worst deviation from 1.0: %.3e  ->  %s"
          % (worst, "PASS" if worst < 1e-9 else "FAIL"))
    print()
    return worst


def probability_of_sequence(model, sentence):
    """P(x1..xT) by the chain rule, as the product of the model's factors.

    This is the factorisation the Bayesian network represents, evaluated
    directly - a second check that the CPTs mean what they are supposed to.
    """
    tokens = [START] * model.order + sentence.lower().split() + [END]
    probability = 1.0
    factors = []
    for index in range(model.order, len(tokens)):
        context = tuple(tokens[index - model.order:index])
        distribution = model.distribution(context)
        p = distribution.get(tokens[index], 0.0)
        factors.append(("P(%s | %s)" % (tokens[index], " ".join(context)), p))
        probability *= p
    return probability, factors


def heading(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Part IV / Question 3 - the conditional probability table
# ---------------------------------------------------------------------------
def show_cpt(model, contexts, label):
    heading(label)
    for context in contexts:
        distribution = model.distribution(context)
        name = context if isinstance(context, str) else " ".join(context)
        if not distribution:
            print("P(next | %s): context never observed - no distribution" % name)
            continue
        row = model.counts[model._as_context(context)]
        total = sum(row.values())
        print("P(next | %s)   [%d observed transitions]" % (name, total))
        for token in sorted(distribution, key=lambda t: (-distribution[t], t)):
            print("    %-10s %d/%-3d = %.4f" % (token, row[token], total,
                                                distribution[token]))
        zero = sorted(model.vocabulary - set(distribution))
        print("    zero-probability successors (%d): %s" % (len(zero), zero))
        print()


# ---------------------------------------------------------------------------
# Part VIII - predicting the next word
# ---------------------------------------------------------------------------
def prediction_table(model, contexts, label):
    heading(label)
    print("%-14s %-12s %-8s %s" % ("context", "argmax", "P", "full distribution"))
    print("-" * 70)
    for context in contexts:
        name = context if isinstance(context, str) else " ".join(context)
        distribution = model.distribution(context)
        if not distribution:
            print("%-14s %-12s %-8s %s" % (name, "-", "-", "context unseen"))
            continue
        best, tied = model.predict(context)
        rendered = ", ".join("%s %.3f" % (t, p) for t, p in
                             sorted(distribution.items(),
                                    key=lambda kv: (-kv[1], kv[0])))
        marker = "%s%s" % (best, " (tie: %s)" % ", ".join(tied)
                           if len(tied) > 1 else "")
        print("%-14s %-12s %-8.3f %s" % (name, marker, distribution[best],
                                         rendered))
    print()


# ---------------------------------------------------------------------------
# Parts IX and X - generation, greedy against sampling
# ---------------------------------------------------------------------------
def generate_many(model, count, mode, seed):
    rng = random.Random(seed)
    sentences, truncations = [], 0
    for _ in range(count):
        sentence, truncated = model.generate(mode=mode, rng=rng)
        sentences.append(sentence)
        truncations += truncated
    return sentences, truncations


def generation_report(model, label, seed=1):
    heading(label)

    print("Mode A - greedy: always argmax P(w | context)")
    greedy, truncated = generate_many(model, 5, "greedy", seed)
    for index, sentence in enumerate(greedy, 1):
        print("  %d. %s" % (index, sentence))
    print("  distinct sentences: %d of 5" % len(set(greedy)))
    print("  hit the length cap: %d of 5" % truncated)
    print()

    print("Mode B - sampling: draw w ~ P(w | context)")
    sampled, truncated = generate_many(model, 5, "sample", seed)
    for index, sentence in enumerate(sampled, 1):
        print("  %d. %s" % (index, sentence))
    print("  distinct sentences: %d of 5" % len(set(sampled)))
    print("  hit the length cap: %d of 5" % truncated)
    print()

    twenty, truncated = generate_many(model, 20, "sample", seed + 100)
    print("Twenty sampled sentences (Part IX):")
    for index, sentence in enumerate(twenty, 1):
        print("  %2d. %s" % (index, sentence))
    print("  distinct sentences: %d of 20" % len(set(twenty)))
    print("  hit the length cap: %d of 20" % truncated)
    in_corpus = sum(1 for s in twenty if s in CORPUS.splitlines())
    print("  sentences that appear verbatim in the training corpus: %d of 20"
          % in_corpus)
    print()
    return twenty


# ---------------------------------------------------------------------------
# Part XIII - comparing the two models
# ---------------------------------------------------------------------------
def comparison(first, second, first_sentences, second_sentences):
    heading("PART XIII - COMPARING THE FIRST- AND SECOND-ORDER MODELS")
    rows = [
        ("vocabulary |V|", len(first.vocabulary), len(second.vocabulary)),
        ("contexts observed", first.observed_contexts(),
         second.observed_contexts()),
        ("context alphabet size", len(first.context_alphabet()),
         len(second.context_alphabet())),
        ("contexts possible (alphabet^order)", first.possible_contexts(),
         second.possible_contexts()),
        ("contexts with no distribution", first.unseen_contexts(),
         second.unseen_contexts()),
        ("non-zero CPT entries stored", first.stored_entries(),
         second.stored_entries()),
        ("free parameters estimated", first.parameter_count(),
         second.parameter_count()),
        ("full CPT size (contexts x |V|)",
         first.possible_contexts() * len(first.vocabulary),
         second.possible_contexts() * len(second.vocabulary)),
    ]
    print("%-34s %12s %12s" % ("Measure", "order 1", "order 2"))
    print("-" * 60)
    for name, a, b in rows:
        print("%-34s %12s %12s" % (name, a, b))
    print()

    corpus_lines = CORPUS.splitlines()
    for label, sentences in (("order 1", first_sentences),
                             ("order 2", second_sentences)):
        distinct = len(set(sentences))
        verbatim = sum(1 for s in sentences if s in corpus_lines)
        print("%s: %d distinct of %d generated, %d verbatim from the corpus"
              % (label, distinct, len(sentences), verbatim))
    print()

    print("Sequence probability under each model, by the chain rule:")
    for sentence in ("the cat sat on the mat", "the cat ran to the park"):
        p1, _ = probability_of_sequence(first, sentence)
        p2, _ = probability_of_sequence(second, sentence)
        print("  %-26s order 1: %.6f    order 2: %.6f" % (sentence, p1, p2))
    print()
    print("A sentence that never occurs in the corpus:")
    for sentence in ("the mat sat on the cat",):
        p1, _ = probability_of_sequence(first, sentence)
        p2, _ = probability_of_sequence(second, sentence)
        print("  %-26s order 1: %.6f    order 2: %.6f" % (sentence, p1, p2))
    print()


def main():
    print("Lab 5 - Bayesian networks and autoregressive language models")
    print("Corpus: %d sentences, no ML library, counting only."
          % len(CORPUS.strip().splitlines()))

    first = NgramModel(order=1).train(tokenise(order=1))
    second = NgramModel(order=2).train(tokenise(order=2))

    # Part IV / Question 3
    show_cpt(first, ["the", "cat", "dog", "sat", "ran", START],
             "PART IV - FIRST-ORDER CPT  P(Xt | X_{t-1})")

    # Part VII
    heading("PART VII - TESTING THE PROBABILISTIC SPECIFICATION")
    normalisation_test(first, "First-order model")
    normalisation_test(second, "Second-order model")
    print("The chain rule evaluated on a training sentence, as a second check")
    print("that the factors are the ones the Bayesian network specifies:")
    probability, factors = probability_of_sequence(first, "the cat sat on the mat")
    for name, value in factors:
        print("  %-22s = %.4f" % (name, value))
    print("  product = %.6f" % probability)
    print()

    # Part VIII
    prediction_table(first, ["the", "cat", "dog", "sat", "ran", "on", "to"],
                     "PART VIII - NEXT-WORD PREDICTION, FIRST-ORDER MODEL")

    # Parts IX and X
    first_sentences = generation_report(
        first, "PARTS IX AND X - GENERATION, FIRST-ORDER MODEL")

    # Parts XI and XII
    show_cpt(second, [(START, START), (START, "the"), ("the", "cat"),
                      ("cat", "sat"), ("on", "the"), ("the", "mat")],
             "PART XI - SECOND-ORDER CPT  P(Xt | X_{t-2}, X_{t-1})")
    prediction_table(second, [(START, "the"), ("the", "cat"), ("the", "dog"),
                              ("cat", "sat"), ("cat", "ran"), ("on", "the"),
                              ("the", "park")],
                     "SECOND-ORDER NEXT-WORD PREDICTION")
    second_sentences = generation_report(
        second, "GENERATION, SECOND-ORDER MODEL", seed=1)

    comparison(first, second, first_sentences, second_sentences)

    with open("generated_sentences.txt", "w") as handle:
        handle.write("# Twenty sampled sentences, first-order model\n")
        handle.write("\n".join(first_sentences) + "\n\n")
        handle.write("# Twenty sampled sentences, second-order model\n")
        handle.write("\n".join(second_sentences) + "\n")
    print("Generated sentences written to generated_sentences.txt")


if __name__ == "__main__":
    main()
