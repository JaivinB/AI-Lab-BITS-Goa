# LLM prompts used in Lab 3

The assistant used was Claude. The search problem (Task 0) and the agent design
(Task 1) were written before any prompt was issued; the LLM was asked to
translate that design into code.

## Prompt 1 — generate A* (Task 2)

> I am implementing a simple goal-based search agent in Python. The environment
> is a grid represented by an ASCII map. The agent starts at `S` and must reach
> `G`. The symbol `#` represents obstacles and `.` represents free cells. The
> agent can move up, down, left or right, and every movement has cost 1.
>
> Implement A* search. Use Manhattan distance as the heuristic:
> `h(n) = |x - xG| + |y - yG|`.
>
> The program should:
> - represent grid positions as states;
> - maintain an appropriate frontier;
> - calculate `g(n)`, `h(n)` and `f(n)`;
> - avoid repeatedly expanding the same state;
> - reconstruct the path when the goal is reached;
> - report the path and its length;
> - report the number of states expanded.
>
> Keep the implementation simple and explain the main components of the code.

## Prompt 2 — BFS over the same problem definition (Task 5)

> Add a breadth-first version of the same agent. It must use the same `Grid`
> class, the same successor function and the same path reconstruction, so that
> the only difference between the two algorithms is the frontier discipline.
> Do not change the warehouse map.

## Prompt 3 — make the heuristic swappable (Task 6)

> Make the heuristic a parameter of the A* function rather than a hard-coded
> call, and add `h(n) = 0`, Euclidean distance, and `2 x` Manhattan distance as
> alternatives. Note in a comment which of these are admissible and which are
> not.

## Prompt 4 — independent verification

> Write a `verify(grid, result)` function that replays a returned plan from the
> start cell, applying the action deltas directly, and checks that every cell on
> the path is passable, that the final cell is the goal, that the path length
> and the action count agree, and that the reported cost equals the number of
> unit-cost actions. It must not call the successor function the search used.

## Prompt 5 — tie-breaking

> On an open grid, A* with Manhattan distance expands as many states as BFS.
> Add an option to break f-ties by preferring the state with the smaller `h(n)`,
> keep the existing insertion-order behaviour as the default, and explain why
> neither rule can affect optimality.

## Changes made to the generated code before accepting it

1. **The closed-set handling was wrong.** The generated A* pushed successors
   without checking whether a state had already been expanded, and popped
   without checking whether the popped state was stale. With a consistent
   heuristic this is only wasteful, but with the inflated `2 x` Manhattan
   heuristic it causes states to be expanded repeatedly and makes the
   "states expanded" figure meaningless — which is the one measurement Task 5
   and Task 6 compare. A `closed` set plus a stale-entry check on pop was added.

2. **The goal test was applied when a state was generated, not when it was
   expanded.** For BFS with unit costs this happens to give the same path, but
   it is wrong in general for A*: a state can be generated with a suboptimal
   `g` and testing it on generation can return a suboptimal path. It was moved
   to the point of expansion for both algorithms, which also makes the expansion
   counts comparable.

3. **`states expanded` was counted at push time.** Renamed and split: `expanded`
   counts states popped and developed, `generated` counts successors produced.
   Comparing BFS and A* on a figure that actually meant "nodes generated" would
   have understated the difference between them.

4. **f-ties were resolved by comparing the states themselves.** The generated
   code pushed `(f, state)`, so Python compared the `(row, col)` tuples when `f`
   tied — a hidden, arbitrary bias towards low row indices. An explicit
   insertion counter was added so the default behaviour is well-defined and
   runs are reproducible.

5. The generated Euclidean heuristic was written as the *squared* distance,
   which is not admissible on this grid — `(dx^2 + dy^2)` exceeds `dx + dy`
   whenever either offset is 2 or more, so it would have silently broken
   optimality while being presented as the admissible alternative. Changed to
   `math.hypot`.

6. The map parser assumed all rows had equal length, so a short trailing line
   raised `IndexError`. Rows are now padded with obstacles.
