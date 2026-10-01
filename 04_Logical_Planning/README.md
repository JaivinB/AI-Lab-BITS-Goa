# Lab 4 — Logical Reasoning for Planning

```
Logic + Search = Planning
```

A warehouse robot must deliver a package from location `a` to location `c`.
Logic decides which actions are *possible*; search decides which sequences to
*try*.

| File | Contents |
|---|---|
| `planner.py` | `Action` (signed preconditions and effects), the applicability test, breadth-first planner, independent plan validator |
| `tests.py` | Tests A, B and C, and the Task 5 comparison of explanation against verification |
| `planner.pl` | Optional extension: Prolog as an independent verifier of the plan's moves |
| `prompts.md` | The exact LLM prompts and the five corrections made before accepting the code |
| `results.txt` | Captured output of `tests.py` |

```bash
python planner.py   # applicability in the initial state, then Test A
python tests.py     # all three tests plus Task 5
swipl planner.pl    # optional Prolog section (see note below)
```

Pure Python 3, no dependencies. SWI-Prolog is optional and is **not installed in
the environment these results were produced in**, so the Prolog section below
gives the clauses and the derivations rather than a captured terminal
transcript; `planner.pl` contains a `demo` predicate that prints true/false for
every query when run with `swipl planner.pl`.

---

## Task 0 — The planning problem

**(a) Initial state** `I = {At(Robot,a), At(Package,a)}`

**(b) Goal** `G = {At(Package,c)}` — note that the goal is a *subset* to be
entailed, not a complete state. The robot's final location is unconstrained.

**(c) and (d) Actions, with preconditions and effects**

| Action | Positive preconditions | Positive effects | Negative effects |
|---|---|---|---|
| `Move(X,Y)` for connected `X,Y` | `At(Robot,X)` | `At(Robot,Y)` | `At(Robot,X)` |
| `PickUp(Package,L)` | `At(Robot,L)`, `At(Package,L)` | `Holding(Package)` | `At(Package,L)` |
| `Drop(Package,L)` | `At(Robot,L)`, `Holding(Package)` | `At(Package,L)` | `Holding(Package)` |

Connectivity is `a–b` and `b–c`, so the `Move` instances are `Move(a,b)`,
`Move(b,a)`, `Move(b,c)`, `Move(c,b)` — ten actions in total across the three
schemas. A closed-world assumption is in force: a proposition absent from the
state is false.

**Which actions are applicable in `I`?** Two:

| | Action | Reason |
|---|---|---|
| ✓ | `Move(a,b)` | `At(Robot,a)` holds |
| ✓ | `PickUp(Package,a)` | `At(Robot,a)` and `At(Package,a)` both hold |

and eight are not, for example `Move(b,c)` (missing `At(Robot,b)`) and
`Drop(Package,a)` (missing `Holding(Package)`).

**Is `PickUp(Package,a)` applicable?** Yes. Its preconditions are
`At(Robot,a)` and `At(Package,a)`, and `I` contains both, so
`I |= Preconditions(PickUp(Package,a))`.

**Is `Drop(Package,c)` applicable?** No, and it fails on *both* of its
preconditions: `At(Robot,c)` is absent from `I` (the robot is at `a`) and
`Holding(Package)` is absent (nothing has been picked up). Being listed among
the available actions is not applicability — the question is always whether the
preconditions hold *in the current state*. That is exactly where logical
reasoning enters planning.

---

## Task 1 — A plan constructed by hand

| State | Facts |
|---|---|
| `S0` | `At(Robot,a)`, `At(Package,a)` |
| `S1` = after `PickUp(Package,a)` | `At(Robot,a)`, `Holding(Package)` |
| `S2` = after `Move(a,b)` | `At(Robot,b)`, `Holding(Package)` |
| `S3` = after `Move(b,c)` | `At(Robot,c)`, `Holding(Package)` |
| `S4` = after `Drop(Package,c)` | `At(Robot,c)`, `At(Package,c)` |

`S4 |= G`, so the plan is
`PickUp(Package,a), Move(a,b), Move(b,c), Drop(Package,c)` — four actions.

**A note on the sequence the laboratory sheet suggests.** The sheet offers
`Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)` as something to
reason about. It is **not valid**: `PickUp(Package,b)` requires
`At(Package,b)`, but `Move` affects only the robot's location, so the package is
still at `a`. The order matters — the package must be picked up *before* the
robot moves. This is the single most useful thing in the lab, because the invalid
sequence reads more naturally than the valid one, and `tests.py` runs both
through the validator to show it (see Task 5).

---

## Task 3 — Testing the generated planner

### Test A — solvable problem

```
Initial state : {At(Package,a), At(Robot,a)}
Goal          : {At(Package,c)}
Plan found    : yes, 4 actions (states expanded: 8)

  S0                {At(Package,a), At(Robot,a)}
  PickUp(Package,a) {At(Robot,a), Holding(Package)}
  Move(a,b)         {At(Robot,b), Holding(Package)}
  Move(b,c)         {At(Robot,c), Holding(Package)}
  Drop(Package,c)   {At(Package,c), At(Robot,c)}
```

Four actions, identical to the hand plan, and every state transition matches the
table in Task 1. The plan was then replayed by `validate_plan`, which checks each
precondition against the state in which that action actually executes: every
precondition satisfied, final state entails the goal, **plan valid**.

### Test B — impossible problem (`PickUp` removed)

Remaining actions: `Move(a,b)`, `Move(b,a)`, `Move(b,c)`, `Move(c,b)`,
`Drop(Package,a)`, `Drop(Package,b)`, `Drop(Package,c)`.

```
Plan found    : NO - no plan found (states expanded: 3)
```

Without `PickUp` the package can never leave `a`, and the planner says so rather
than inventing an action. What makes that termination *sound* rather than lucky:
the reachable state space is finite and visited states are never re-expanded, so
breadth-first search exhausts it and stops. A planner without the visited set
would oscillate `a → b → a → b` for ever and never report failure — which is
exactly the bug the generated code had (see `prompts.md`, correction 4).

### Test C — irrelevant actions

The goal is `At(Package,c)`, not `At(Robot,c)`. Three separate checks:

**1. The goal test itself.** Applied to `{At(Robot,c), At(Package,a)}` — robot
arrived, package did not — `satisfies(state, goal)` returns **False**. Correct.

**2a. A genuinely irrelevant action.** `Patrol(a,c)` sends the robot from `a`
straight to `c`, but carries a *negative* precondition `¬Holding(Package)`, so
it can never help deliver anything.

```
applicable in the initial state: True
plan: PickUp(Package,a), Move(a,b), Move(b,c), Drop(Package,c)
length 4   Patrol used: False
```

The plan is unchanged. `Patrol` is applicable in `I` and does put the robot at
`c`, but its negative precondition means it can never be combined with carrying
the package. This is also the **only** test in the lab that exercises negative
preconditions, which the generated `applicable` silently ignored.

**2b. For contrast, an action that genuinely helps.** `Teleport(Robot,c)` is
identical but without the negative precondition, so it carries a held package:

```
plan: PickUp(Package,a), Teleport(Robot,c), Drop(Package,c)
length 3
state after Teleport: {At(Robot,c), Holding(Package)}
```

The plan shortens to three actions, and `Teleport` appears only *between*
`PickUp` and `Drop`. The state after it does **not** entail `At(Package,c)`;
`Drop` is still required. So even when a robot-only action is useful, the goal is
reached through the package's location, never through the robot's arrival.

**3. Only `Move` actions.** With `PickUp` and `Drop` both removed, the robot can
still reach `c` via `Move(a,b)`, `Move(b,c)` — so a planner that confused
`At(Robot,c)` with `At(Package,c)` would report success. It returns `None`.

| Test | Expected | Result |
|---|---|---|
| A — solvable | 4-action plan, each action valid | 4 actions, validated |
| B — `PickUp` removed | "no plan found" | no plan found, 3 states expanded |
| C.1 — goal test on robot-only state | `False` | `False` |
| C.2a — irrelevant `Patrol` | plan unchanged, `Patrol` unused | 4 actions, unused |
| C.2b — useful `Teleport` | used, but `Drop` still needed | 3 actions, `Drop` present |
| C.3 — `Move` only | no plan | `None` |

---

## Task 4 — Logic and search

**Where each is used.** The logical component is `Action.applicable(state)`: it
decides whether `S |= Preconditions(a)` by checking that every positive
precondition is in the state and no negative precondition is. That is the *only*
place reasoning about truth happens. The search component is
`breadth_first_plan`: it maintains the frontier and the visited set and decides
which applicable action to try next. It consults the logic only through
`applicable`, and never inspects a proposition itself. `Apply(S, a)` sits between
them — the state update is mechanical once applicability has been established.

**Completing the description.** The missing step is the *applicability
decision* — the logical conclusion `S |= Preconditions(a)`:

```
                         Current state
                               |
                   Check action preconditions
                               |
             >>>  S |= Preconditions(a)?  <<<     the logical step:
                  if yes, the action is applicable in S;
                  if no, this action is discarded for this state
                               |
                    Generate successor state
                      S' = (S \ Del(a)) u Add(a)
                               |
                    Search over alternatives
                 (enqueue S' if not already visited)
                               |
                            Goal?
                        S' |= G  ->  return the plan
```

**How they work together, in my own words.** Planning needs two different
questions answered, and neither alone is enough. Logic answers "may I do this
here?" — a question about one state and one action, with a yes/no answer that
follows from the action's definition. It cannot tell you what to do, because it
has no notion of the goal being closer or further away. Search answers "which of
the things I may do should I try, and in what order?" — a question about
sequences, which has no answer at all until something tells it which sequences
are even legal. Logic prunes the branching factor to the applicable actions;
search explores what remains. In Test A, logic reduces ten actions to two in the
initial state, and search then decides to try `PickUp` before `Move` by
enumerating at depth 1 before depth 2. The slogan: **logic determines what is
possible; search determines what to try.**

**The relation to the previous module.** This is the same search machinery as
Lab 3, over a different state space. States are sets of propositions rather than
grid cells, successors are generated by applicable actions rather than by legal
moves, and the goal test is entailment rather than coordinate equality — but the
frontier, the visited set, and the path reconstruction are identical in kind. BFS
is used for the same reason: every action costs 1, so the first plan found is
shortest. An A* version would need a heuristic estimating actions-to-goal, for
example the number of unsatisfied goal propositions.

---

## Task 5 — A generated explanation is not an independent verification

The sheet's suggested sequence
`Move(a,b), PickUp(Package,b), Move(b,c), Drop(Package,c)` reads perfectly
plausibly: go to `b`, pick the package up, carry it to `c`, put it down. Asked to
justify it, an LLM will produce a fluent account of exactly that story. Replayed
against the actual preconditions:

```
step 2: PickUp(Package,b) not applicable - missing ['At(Package,b)'],
        forbidden present []
final state {At(Package,a), At(Robot,b)} does not entail the goal
        {At(Package,c)}
```

Step 2 fails. `PickUp(Package,b)` requires `At(Package,b)`, and the package never
moved — `Move` affects only the robot's location. The correct plan picks up
first, and the validator confirms it: `problems: none — plan valid`, final state
`{At(Package,c), At(Robot,c)}`.

**Which should you trust — the LLM's explanation, or the independently executed
state transitions?** The state transitions. Not because the explanation is
usually wrong, but because it is not *independent*: the plan and the explanation
come from the same source, so an error in the plan tends to be reproduced in the
justification. The two then agree with each other rather than with the domain,
and agreement between an answer and its own defence is worth nothing. The replay
consults the action definitions instead, and is therefore *capable of
disagreeing* — which is the only property that makes a check a check. In this
lab the LLM's explanation happened to be correct; that was established by the
replay, not by reading the explanation.

> A generated explanation is not the same as an independent verification.

---

## Reflection questions

**1. Why specify preconditions and effects before asking an LLM to write the
planner?** Because they are the problem, and everything else is implementation.
Written down first, they become the specification the generated code can be
*checked against* — which is how the four code bugs were found. Written
afterwards, they get reverse-engineered from whatever the code does, and the
specification silently becomes a description of the bugs. Concretely: knowing
that the goal is a set to be *entailed*, not matched, is what exposed the
`state == goal` error; knowing that `PickUp` requires co-location is what makes
the sheet's plausible sequence visibly invalid.

**2. An error that could occur if preconditions were not checked.** The planner
would return `Drop(Package,c)` as a one-action plan from the initial state: its
effect adds `At(Package,c)`, which entails the goal, so an unchecked planner
"solves" the problem immediately. The robot would be at `a`, holding nothing, and
the package would materialise at `c`. Every action's effects are only *licensed*
by its preconditions; drop the check and effects become wishes. Equally, ignoring
*negative* preconditions — the actual bug in the generated code — would let
`Patrol(a,c)` run while holding the package, making a physically meaningless
state reachable.

**3. Why is a plan that "looks reasonable" not necessarily valid?** Because
plausibility is judged against a mental model of the world, and validity against
the formal action definitions, and the two diverge exactly where the formalism is
stricter than intuition. The sheet's sequence is the perfect example: a human
reader fills in "and the robot brings the package along" because that is how
warehouses work, but nothing in `Move`'s effects says so. Validity is also *order
sensitive* in a way that plausibility is not — the same four action types are
valid in one order and invalid in another, and prose does not make the difference
salient.

**4. What did the LLM contribute?** A working implementation of the
state-as-proposition-set design in one pass: the `Action` class, the set algebra
for `apply`, the BFS loop carrying plans alongside states, and the output
formatting. It also produced a correct justification of the final plan when
asked. That is real engineering leverage on the parts where the design was
already fixed.

**5. What had to be verified independently?** Five things, listed in
`prompts.md`. Three of them were invisible to any test that only ran the
warehouse problem: negative preconditions were stored and never checked, effects
were applied add-before-delete instead of delete-before-add, and there was no
visited set. None of these failed on this domain — the actions happen to have no
negative preconditions and no add/delete overlap — so they would have been
inherited silently by the next domain. The fourth, `state == goal` instead of
`goal <= state`, would have reported "no plan found" for a solvable problem. The
validator was written specifically so that the plan's correctness does not depend
on the search being correct.

**6. Where is logical reasoning being used in this laboratory?** In three places.
In `applicable`, which decides `S |= Preconditions(a)` under a closed-world
assumption. In the goal test, which is entailment `S |= G` rather than equality —
the final state contains `At(Robot,c)` as well, and the goal is still satisfied.
And in the optional Prolog extension, where the same connectivity knowledge is
re-expressed as facts and rules and queried by resolution, independently of the
Python program that produced the plan.

**7. How is planning related to the search algorithms of the previous module?**
It *is* search, over a state space defined logically rather than geometrically.
The correspondence is exact: Lab 3's grid cells become sets of propositions,
`successors()` becomes "the applicable actions and their results", and `is_goal`
becomes entailment. What changes is the size and structure of the space — grid
states are enumerable and admit a geometric heuristic like Manhattan distance,
whereas planning states are subsets of propositions and grow exponentially in the
number of propositions, so domain-independent heuristics have to be derived from
the action definitions themselves. Both labs also share the lesson that the
search is only as good as its bookkeeping: without a visited set, Lab 3 revisits
cells and Lab 4 oscillates between locations for ever.

---

## Optional extension — Prolog as a logical verifier

`planner.pl` holds the warehouse knowledge as facts and rules, independently of
the Python planner. SWI-Prolog is not installed in the environment used for the
Python results, so the answers below are derived from the clauses rather than
pasted from a terminal; `?- demo.` prints all of them in one run.

### Task 6 — `can_move`

```prolog
connected(a, b).  connected(b, a).  connected(b, c).  connected(c, b).

can_move(X, Y) :- connected(X, Y).
```

| Query | Answer | Why |
|---|---|---|
| `?- can_move(a, b).` | `true` | `connected(a,b)` is a fact; the rule body unifies with it directly |
| `?- can_move(a, c).` | `false` | no fact `connected(a,c)` exists, and the rule has only that one body goal |

**(a) Why does Prolog return `true` for `can_move(a,b)`?** The goal
`can_move(a,b)` unifies with the rule head `can_move(X,Y)` under
`X = a, Y = b`, leaving the body goal `connected(a,b)`, which matches a fact in
the database. The derivation succeeds, so the query succeeds.

**(b) Why does it not establish `can_move(a,c)`?** The same unification leaves
the goal `connected(a,c)`. No fact matches it and no rule concludes `connected`,
so resolution fails with no alternatives left. Under the closed-world assumption
Prolog reports `false` — meaning "not derivable from this knowledge base", not
"false in the world". `a` and `c` genuinely are connected *via b*; what fails is
the claim that a single move gets from one to the other.

**(c) What is the relationship to `Connected(X,Y) → CanMove(X,Y)`?** The clause
`can_move(X,Y) :- connected(X,Y).` is that implication written as a definite
clause, read right-to-left: if `connected(X,Y)` can be established then conclude
`can_move(X,Y)`. The variables are implicitly universally quantified, so it is
`∀X∀Y (Connected(X,Y) → CanMove(X,Y))`. Prolog uses it in one direction only —
backward chaining from the conclusion to the premise — which is why it can prove
`can_move` from `connected` but would never infer `connected` from `can_move`.

### Task 7 — checking a proposed plan

```prolog
valid_move(X, Y) :- connected(X, Y).

valid_route([_]).
valid_route([X, Y | Rest]) :- valid_move(X, Y), valid_route([Y | Rest]).
```

| Query | Answer |
|---|---|
| `?- valid_move(a, b).` | `true` |
| `?- valid_move(b, c).` | `true` |
| `?- valid_move(a, c).` | `false` |
| `?- valid_route([a, b, c]).` | `true` — the movement skeleton of the Python plan |
| `?- valid_route([a, c]).` | `false` — the challenge case |
| `?- reachable(a, c).` | `true` — but via `b`, not in one move |

**The challenge.** The Python planner produced the movements `a → b → c`, and
Prolog confirms each step independently: `valid_move(a,b)` and `valid_move(b,c)`
both succeed, so `valid_route([a,b,c])` succeeds. Had the planner instead
proposed `Move(a,c)`, Prolog would refuse it — `valid_move(a,c)` fails — because
no fact in the warehouse description supports a direct `a`–`c` move. The
`reachable/2` predicate makes the distinction sharp: `reachable(a,c)` succeeds
while `valid_move(a,c)` fails, so the knowledge base cleanly separates "there is
a route" from "there is a single legal move".

```
Python generates the candidate  ->  Prolog checks it against the facts
```

The verification is independent in the sense that matters: the Prolog clauses
were written from the warehouse description, not from the planner's output, so
they can disagree with it.

### Task 8 — facts, rules, inference

```prolog
wet_road.
slippery     :- wet_road.
reduce_speed :- slippery.
```

`?- reduce_speed.` succeeds. The query `reduce_speed` matches the head of the
third clause, leaving `slippery`; that matches the head of the second, leaving
`wet_road`; that is a fact. The chain of implications:

```
WetRoad  ⇒  (WetRoad → Slippery)  ⇒  (Slippery → ReduceSpeed)  ⇒  ReduceSpeed
```

Prolog works through this backwards, from the conclusion to the fact, and reports
success once it bottoms out. The programmer supplied only facts and rules; the
system performed the inference.

### Prolog reflection

**1. Difference between a fact and a rule.** A fact is a clause with an empty
body — unconditionally true in the knowledge base, like `connected(a,b).`. A rule
has a body of conditions, like `can_move(X,Y) :- connected(X,Y).`, and licenses
its head only when the body can be derived. A fact is the degenerate case of a
rule whose body is always satisfied.

**2. How a query corresponds to entailment.** Asking `?- G.` asks whether `G`
follows from the facts and rules. Prolog attempts to construct a derivation by
backward chaining and unification; success means a proof was found, so `KB ⊨ G`.
Failure means no proof exists *from this knowledge base* — a weaker claim than
`G` being false, which is why `can_move(a,c)` reporting `false` does not mean
`a` and `c` are unconnected.

**3. Why use Prolog to verify a Python plan?** Because the check is then
expressed in a different formalism, written from the domain description rather
than from the plan, and executed by a different engine. A bug in the Python
successor function cannot propagate into it. The Python validator in `planner.py`
is already independent of the *search*, but it shares the `Action` definitions
with the planner; Prolog shares nothing but the warehouse description.

**4. What does an independent verifier add when the plan was LLM-generated?** It
replaces trust with a check that can fail. An LLM's plan and its explanation are
correlated — both are produced by the same process from the same context — so the
explanation cannot serve as evidence. An external checker is uncorrelated with
the generator, which is what makes its agreement informative. It also makes the
failure mode safe: a wrong plan is *rejected* rather than executed, and rejection
is the outcome you want when the generator is fluent and occasionally wrong.

---

```
Understand -> Specify -> Generate -> Execute -> Verify
```

An LLM can assist with the implementation. The resulting agent still has to be
tested, and in this lab four of the five defects found were invisible to any test
that only ran the problem the lab supplied.
