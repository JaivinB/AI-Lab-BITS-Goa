# LLM prompts used in Lab 2

The assistant used was Claude. The agent design in Task 2 was written first; the
LLM was asked to implement that design, not to invent it.

## Prompt 1 — the specification (Task 3)

> Write a well-documented Python program implementing a goal-based agent for the
> warehouse navigation problem below.
>
> ```
> #####################
> #S....#............G#
> #.##....##########..#
> #....##.............#
> #.######.###.#.###..#
> #........#..........#
> #####################
> ```
>
> `S` is the start, `G` the goal, `#` an obstacle, `.` free space. The vehicle
> moves Up, Down, Left or Right by one grid square.
>
> The program should
> - represent the warehouse as a two-dimensional grid;
> - determine a collision-free path from S to G;
> - avoid all obstacles;
> - print either the path found or a suitable message if no path exists;
> - explain the search algorithm that has been chosen and why it is appropriate.
>
> Structure the code so that the environment, the agent's state, the goal, the
> available actions and the decision-making component are separate and
> identifiable, matching the goal-based agent architecture.

## Prompt 2 — independent verification of the plan

> Add a step that replays the returned plan one action at a time from the start
> cell, checking that every intermediate cell is passable and that the final
> cell is the goal. Raise an error if either check fails. The search and the
> check must not share code, so that a bug in the search is actually caught.

## Prompt 3 — the unreachable case

> Add a second, smaller map in which the goal is sealed behind obstacles, and
> show that the program reports that no path exists rather than looping.

## Changes made to the generated code before accepting it

1. **The first version returned only the list of cells, not the actions.** A
   goal-based agent's plan is a sequence of *actions*; the cells are what you
   get by executing it. The search was changed to store the action taken on each
   edge in `came_from`, so `formulate_plan` returns both the action sequence and
   the states it passes through.

2. **Verification reused the search's own successor function.** As generated,
   the "check" called the same `successors()` the search had used, so any bug in
   `passable()` would have been invisible to it. `execute()` was rewritten to
   apply the action deltas directly and test `passable()` on the resulting cell
   itself.

3. **Ragged map lines would have crashed on indexing.** The generated parser
   assumed every row had equal length. A padding step was added so a short
   trailing line is treated as obstacle rather than raising `IndexError`.

4. The generated code counted a state as "expanded" when it was enqueued. That
   inflates the figure and makes it incomparable with the Lab 3 counts, so the
   counter was moved to the point where a state is actually dequeued and its
   successors generated.
