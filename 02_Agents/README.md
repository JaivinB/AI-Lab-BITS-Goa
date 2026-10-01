# Lab 2 — Agents: constructing a goal-based agent with an LLM

An autonomous warehouse vehicle must carry packages from the loading bay `S` to
the dispatch area `G` without crossing shelving.

| File | Contents |
|---|---|
| `warehouse_agent.py` | The goal-based agent: environment, state, goal, actions, BFS decision component, plan verifier |
| `prompts.md` | The exact LLM prompts and the four corrections made before accepting the code |
| `results.txt` | Captured program output |

```bash
python warehouse_agent.py
```

Pure Python 3, no dependencies.

---

## Task 1 — Understanding the problem

**1. What is the environment?** A 7×21 grid of cells, each either free (`.`) or
shelving (`#`), with a start `S` at `(1,1)` and a goal `G` at `(1,19)`. It is
fully observable (the map is known in advance), static (shelving does not move),
discrete, and deterministic (an action's outcome is fixed). The border is solid,
so the agent cannot leave the warehouse.

**2. What is the goal of the agent?** To occupy the cell marked `G`. Expressed as
a goal test: `state == goal_position`. It is an explicit, checkable predicate on
the state, which is what makes the agent goal-based.

**3. What actions are available?** Four: Up, Down, Left, Right, each moving the
vehicle one grid square. An action is available in a state only if the
destination cell is inside the grid and is not shelving, so between 1 and 4
actions are legal in any given cell.

**4. What information must the agent maintain to choose its next action?** Its
current position; the map; the explicit goal; and — the part a reflex agent
lacks — search bookkeeping: the frontier of cells awaiting exploration, the set
of cells already reached, and for each reached cell the predecessor and the
action that got there, so a plan can be reconstructed once the goal is found.

**5. Why is this goal-based rather than a simple reflex agent?** A reflex agent
maps the current percept to an action through fixed condition–action rules. Here
the percept — which of the four neighbouring cells are free — is radically
insufficient: the cells `(1,2)` and `(1,4)` look identical locally, but from one
the right move is onward and from the other it is a detour around the shelving at
`(1,6)`. Worse, a reflex rule such as "move towards the goal if possible" walks
into the dead-end corridor and oscillates there for ever, since nothing in the
percept says the corridor is a trap. The goal-based agent instead *projects* the
consequences of action sequences before committing to one, and so can conclude
that a route exists and what it is. The same machinery also lets it conclude
that **no** route exists — which a reflex agent can never determine, because the
absence of a path is not a local property.

**Think about it — if the warehouse doubled in size?** BFS would still be
*correct*: all moves cost 1, so the first time BFS reaches the goal the plan is
of minimum length, regardless of scale. What degrades is cost. The state space is
the number of free cells, so doubling each dimension roughly quadruples both
the cells expanded and the memory held in the frontier and reached set — BFS
stores the whole frontier, and in a grid that frontier grows with the perimeter
of the explored region. The difficulties that arise: memory becomes the binding
constraint before time does; BFS explores in every direction equally, including
directly away from the goal, which is pure waste once the map is large; and
re-planning after any change to the map costs a full search again. The fix is to
inform the search with a heuristic — which is exactly what Lab 3 does with A*
and Manhattan distance on the same kind of map. Beyond that, one would move to
path caching between fixed waypoints, or a hierarchical decomposition into aisles
rather than cells.

---

## Task 2 — Designing the agent

```
                    +-------------------------------+
                    |      WAREHOUSE (environment)  |
                    |  ASCII grid; static; known    |
                    +-------------------------------+
                        |                        ^
            percept:    |                        |  action:
            grid, own   |                        |  Up/Down/Left/Right
            position    v                        |
   +------------------------------------------------------------+
   |                      GOAL-BASED AGENT                      |
   |                                                            |
   |   +-------------+        +---------------------------+     |
   |   |   STATE     |        |  GOAL                     |     |
   |   | (row, col)  |        |  state == position of G   |     |
   |   +-------------+        +---------------------------+     |
   |          |                        |                        |
   |          v                        v                        |
   |   +------------------------------------------------+       |
   |   |        DECISION COMPONENT  (search)            |       |
   |   |  frontier: FIFO queue      -> BFS              |       |
   |   |  reached:  dict state -> (previous, action)     |      |
   |   |  successors(state) via the four actions        |       |
   |   |  stop when goal test succeeds                  |       |
   |   +------------------------------------------------+       |
   |                           |                                |
   |                           v                                |
   |   +------------------------------------------------+       |
   |   |  PLAN: action sequence  ->  executed one step   |      |
   |   |  at a time, each step re-validated              |      |
   |   +------------------------------------------------+       |
   +------------------------------------------------------------+
```

- **Environment** — `Warehouse`: parses the map, answers `passable(cell)` and
  `successors(state)`, and locates `S` and `G`. Immutable during planning.
- **Current state** — the `(row, column)` tuple the vehicle occupies. Position
  alone is a sufficient state because the map is static and the vehicle carries
  nothing that affects movement.
- **Goal** — `state == self.goal`, held explicitly as a field on the agent.
- **Actions** — `{Up, Down, Left, Right}` as `(dr, dc)` deltas, cost 1 each.
- **Decision component** — breadth-first search returning a shortest action
  sequence, then an executor that replays it with independent validation.

**Why BFS.** Every action costs the same, so BFS's expansion order is by
increasing plan length and the first goal reached is optimal. The state space is
one node per free cell — small enough to search exhaustively — and BFS is
complete, so it terminates with "no path" rather than looping when the goal is
unreachable. A cost-aware algorithm such as Uniform Cost Search would reduce to
BFS here; A* would be the right choice if the map were much larger (Lab 3).

---

## Task 3 — Results

Prompt and corrections: `prompts.md`. Full output: `results.txt`.

```
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

| Measure | Value |
|---|---|
| Path found | yes |
| Plan length | **20 moves** |
| States expanded | 59 |
| Start, goal | `(1,1)`, `(1,19)` |
| Plan verified by independent replay | yes |

The plan: `Right Right Right Down Right Right Right Up Right ...` — three cells
east, drop into row 2 to pass under the shelving at `(1,6)`, cross, climb back to
row 1 at `(1,7)` and run east to `G`.

**Is 20 optimal?** The Manhattan distance from `(1,1)` to `(1,19)` is 18, and
any detour around the block at `(1,6)` costs one step down and one step back up,
so 20 is a lower bound as well as the length found. The agent's own check is
independent of this argument: it replays the plan cell by cell, confirms every
cell is passable, and confirms the last cell is `G`.

**1. Did the LLM generate a working program on the first attempt?** It ran and
printed a correct 20-move path on the first attempt. But two of its claims were
not actually being tested: the "verification" step called the same
`successors()` function the search used, so it could not have caught a bug in
the obstacle test, and the program returned a list of cells while describing it
as a plan of actions. It worked; it was not yet validated.

**2. How could the prompt be improved?** By specifying the *structure* and the
*evidence*, not only the behaviour. The initial prompt asked for a path and got
one. The useful additions were: naming the five architectural components so the
code maps onto the goal-based architecture rather than being a bare function;
requiring the plan to be an action sequence; requiring verification that does
not share code with the search; and requiring an unreachable-goal case. A prompt
that asks only for output gets output; a prompt that asks for testable structure
gets something you can check.

**3. What search algorithm did the LLM choose?** Breadth-first search, with a
FIFO `deque` frontier and a dictionary of reached states mapping to
predecessors.

**4. Why did it select that algorithm?** Because it matches the problem's
stated structure, and that structure is heavily represented in its training
data: unit-cost moves on a 4-connected grid is the textbook case where BFS is
both optimal and simple, and "shortest path in a maze" is close to a stock
exercise. It was the right choice for the right reason here — but that is worth
noticing rather than relying on. The LLM selected the algorithm from the shape
of the problem description, not from an analysis of the cost model; had the map
carried varying terrain costs, the same familiarity would likely have produced
the same BFS, which would then have been silently wrong. Verifying the choice
against the cost model remains the engineer's job.
