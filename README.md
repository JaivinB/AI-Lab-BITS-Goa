# AI Lab — BITS Goa

Laboratory submissions for the undergraduate Artificial Intelligence course.
Five labs, each one using an LLM as an engineering assistant while the problem
specification, the testing and the interpretation stay with the engineer.

```
AI Science  ->  AI Engineering
```

Every number quoted in a report was produced by the code in that lab's folder
and is reproducible from the committed source.

## The labs

| # | Lab | What it builds | Headline result |
|---|---|---|---|
| [01](01_Neural_Models) | Neural Models | A 2–2–1 network for XOR, plus a three-class softmax extension | Final loss **2.4e-05** (4/4 correct); the symmetric network stalls at **0.4774**, matching the analytic optimum of a one-line model to four decimals |
| [02](02_Agents) | Agents | A goal-based warehouse agent with a BFS decision component | **20-move** shortest path, verified by independent replay; provably optimal against the Manhattan bound |
| [03](03_Search) | Search and A* | A*, BFS and four heuristics over one problem definition | A* and BFS **tie exactly** on the lab's maze (63 states each); goal-directed tie-breaking cuts expansions **89 → 19** on an open floor |
| [04](04_Logical_Planning) | Logical Planning | A STRIPS planner (logic + BFS), with Prolog as an independent verifier | **4-step plan**, validated action by action; the lab sheet's own suggested plan is shown to be **invalid** at step 2 |
| [05](05_Bayesian_Networks) | Bayesian Networks | First- and second-order n-gram models as Bayesian networks | Greedy decoding **never terminates** at order 1 but does at order 2; order 2 leaves **106 of 121** contexts with no distribution |

## Running the code

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python 01_Neural_Models/xor_binary.py
python 01_Neural_Models/xor_threeclass.py
python 02_Agents/warehouse_agent.py
python 03_Search/search_agent.py
python 03_Search/tests_and_experiments.py
python 04_Logical_Planning/planner.py
python 04_Logical_Planning/tests.py
python 05_Bayesian_Networks/ngram_lm.py
```

Each script is self-contained and prints its own evidence. Run from inside the
lab's folder, or with that folder on `PYTHONPATH`, since Labs 3 and 4 import a
sibling module.

## What is in each folder

| File | Purpose |
|---|---|
| `README.md` | The lab report: problem specification, design, results, and answers to every question the lab sheet asks |
| `prompts.md` | The exact prompts given to the LLM, and every correction made to the generated code before accepting it |
| `results*.txt` | Captured program output, so the reports can be checked against the runs |
| `*.py` / `*.pl` | The source |

## Method

Each lab follows the workflow the course sets out:

```
Understand -> Specify -> Generate -> Execute -> Verify
```

The specification was written before prompting in every lab, and the generated
code was read against that specification before being run. Three habits did most
of the work:

- **Verification that does not share code with the thing it verifies.** Labs 2,
  3 and 4 each replay a returned plan against the environment independently of
  the search that produced it. The first generated version in each case called
  the search's own successor function, so it could not have caught a bug.
- **Tests whose answers are known in advance.** Lab 3's warehouse returns 40
  moves and 63 expansions under *every* variant — correct or broken — so it
  cannot distinguish A* from BFS, a good heuristic from `h = 0`, or an
  admissible heuristic from an inadmissible one. The bugs were found on maps
  chosen to be able to fail.
- **Repeating runs before drawing conclusions.** Lab 1's single-seed activation
  table supports a claim about sigmoid that a five-seed sweep contradicts.

`prompts.md` in each folder lists what the LLM got right and what had to be
corrected. Across the five labs, 28 corrections were needed; the ones worth
reading are the cases where the generated code ran, printed plausible numbers,
and was answering the wrong question:

- a finite-difference gradient check written in float32 with `h = 1e-4`, whose
  numerical noise was the same size as the gradients it was checking (Lab 1);
- a Euclidean heuristic implemented as the *squared* distance, inadmissible and
  therefore silently breaking A*'s optimality, while being presented as the
  admissible alternative (Lab 3);
- negative preconditions stored but never checked, invisible because no action
  in the domain used one (Lab 4);
- an unseen n-gram context returning a *uniform* distribution — a smoothing
  decision introduced as error handling, which would have erased the lab's main
  finding about data sparsity (Lab 5).

## Environment

Python 3.12, PyTorch 2.14.1, CPU only. Labs 2–5 need nothing beyond the standard
library. SWI-Prolog is optional and was **not** installed in the environment used
here, so Lab 4's Prolog section gives the clauses and the derivations rather than
a captured transcript; `04_Logical_Planning/planner.pl` has a `demo` predicate
that prints every query's answer when run with `swipl planner.pl`.

## Author

Jaivin Bengani — BITS Pilani, Goa Campus.
