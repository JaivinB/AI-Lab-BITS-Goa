"""
Lab 3 - Search and A* for the warehouse robot.

The search problem P = (S, A, T, s0, G, c) is realised as follows:

    S   states        (row, column) of the robot on a free cell
    A   actions        Up, Down, Left, Right
    T   transition     T(s, a) = s + delta(a), defined only when the target
                       cell is inside the grid and is not an obstacle
    s0  initial state  the cell marked S
    G   goal           the single cell marked G
    c   cost           1 for every action

Both BFS and A* are implemented over that one problem definition, so a
comparison between them cannot be contaminated by a difference in the problem.
A* uses

    f(n) = g(n) + h(n)

with the heuristic supplied by the caller, which is what makes the heuristic
investigation in Task 6 a change of one argument rather than a change of code.
"""

import heapq
import math
from collections import deque

# The warehouse supplied in the laboratory sheet.
WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

OBSTACLE, START, GOAL = "#", "S", "G"


class Grid:
    """The environment: the ASCII map, plus the start and goal cells."""

    def __init__(self, text):
        self.rows = [list(line) for line in text.strip("\n").splitlines()]
        self.height = len(self.rows)
        self.width = max(len(r) for r in self.rows)
        for row in self.rows:
            row.extend([OBSTACLE] * (self.width - len(row)))
        self.start = self._find(START)
        self.goal = self._find(GOAL)

    def _find(self, symbol):
        for r, row in enumerate(self.rows):
            for c, cell in enumerate(row):
                if cell == symbol:
                    return (r, c)
        raise ValueError("symbol %r not found in map" % symbol)

    def passable(self, cell):
        """An action is invalid exactly when its target cell fails this test."""
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width \
            and self.rows[r][c] != OBSTACLE

    def successors(self, state):
        """T: yield (action, next state, step cost) for the legal actions."""
        r, c = state
        for action, (dr, dc) in ACTIONS.items():
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                yield action, nxt, 1          # c(s, a) = 1 for every action

    def is_goal(self, state):
        return state == self.goal

    def render(self, path=None):
        picture = [row[:] for row in self.rows]
        for r, c in path or []:
            if picture[r][c] == ".":
                picture[r][c] = "*"
        return "\n".join("".join(row) for row in picture)


# ---------------------------------------------------------------------------
# Heuristics.  h(n) estimates the remaining cost from n to the goal.
# ---------------------------------------------------------------------------
def manhattan(state, goal):
    """|x - xG| + |y - yG|.  Admissible here: every move changes exactly one
    coordinate by one at cost 1, so no path can be shorter than this."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def zero(state, goal):
    """h(n) = 0.  Admissible but uninformed; A* degenerates to uniform cost."""
    return 0


def euclidean(state, goal):
    """Straight-line distance.  Admissible but never exceeds Manhattan, so it
    is weaker: it underestimates by more on any diagonal offset."""
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def double_manhattan(state, goal):
    """2 x Manhattan.  INADMISSIBLE: it can exceed the true remaining cost, so
    A* loses its optimality guarantee."""
    return 2 * manhattan(state, goal)


HEURISTICS = {
    "manhattan": manhattan,
    "zero": zero,
    "euclidean": euclidean,
    "2x manhattan": double_manhattan,
}


class Result:
    """What the search reports when it terminates."""

    def __init__(self, found, path=None, actions=None, cost=None, expanded=0,
                 generated=0, max_frontier=0):
        self.found = found
        self.path = path or []
        self.actions = actions or []
        self.cost = cost
        self.expanded = expanded          # states popped and developed
        self.generated = generated        # successor states produced
        self.max_frontier = max_frontier  # peak frontier size (memory)

    @property
    def length(self):
        return len(self.actions)


def _reconstruct(came_from, state):
    """Walk the predecessor links back from the goal to build the plan."""
    path, actions = [state], []
    while came_from[state][0] is not None:
        state, action = came_from[state]
        actions.append(action)
        path.append(state)
    path.reverse()
    actions.reverse()
    return path, actions


def breadth_first_search(grid):
    """BFS: FIFO frontier, no heuristic, no cost arithmetic.

    Blind search - it uses only the structure of the graph, never any estimate
    of how far the goal still is.  With unit costs the first goal reached is
    optimal, because states leave the queue in order of increasing depth.
    """
    frontier = deque([grid.start])
    came_from = {grid.start: (None, None)}
    expanded = generated = 0
    max_frontier = 1

    while frontier:
        state = frontier.popleft()
        if grid.is_goal(state):                       # goal test on expansion
            path, actions = _reconstruct(came_from, state)
            return Result(True, path, actions, len(actions),
                          expanded, generated, max_frontier)
        expanded += 1
        for action, nxt, _cost in grid.successors(state):
            generated += 1
            if nxt not in came_from:                  # never revisit a state
                came_from[nxt] = (state, action)
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))

    return Result(False, expanded=expanded, generated=generated,
                  max_frontier=max_frontier)


def a_star_search(grid, heuristic=manhattan, tie_break="fifo"):
    """A*: priority-queue frontier ordered by f(n) = g(n) + h(n).

    Informed search - the heuristic injects knowledge about the goal's location
    that is not present in the graph structure itself.  `closed` prevents a
    state from being expanded twice; the counter in the queue entry breaks
    f-ties deterministically so runs are reproducible.

    `tie_break` decides what happens among states with equal f, which on a
    uniform-cost grid is a very large set:

        "fifo"        insertion order, i.e. no preference at all
        "prefer_goal" expand the state with the smaller h first, so the search
                      drives towards the goal instead of fanning out along the
                      whole f = f* contour

    Both settings are optimal when h is admissible, because neither ever
    expands a state whose f exceeds f*; they differ only in how many of the
    states with f exactly equal to f* get expanded before the goal comes up.
    """
    counter = 0
    start_h = heuristic(grid.start, grid.goal)
    frontier = [(start_h, start_h if tie_break == "prefer_goal" else 0,
                 counter, grid.start)]               # (f, tie key, counter, state)
    g_score = {grid.start: 0}                        # g(n): cost from start
    came_from = {grid.start: (None, None)}
    closed = set()
    expanded = generated = 0
    max_frontier = 1

    while frontier:
        _f, _tie, _n, state = heapq.heappop(frontier)  # lowest f(n) is expanded
        if state in closed:
            continue
        if grid.is_goal(state):
            path, actions = _reconstruct(came_from, state)
            return Result(True, path, actions, g_score[state],
                          expanded, generated, max_frontier)
        closed.add(state)
        expanded += 1

        for action, nxt, step_cost in grid.successors(state):
            generated += 1
            tentative_g = g_score[state] + step_cost
            if nxt in closed:
                continue
            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                came_from[nxt] = (state, action)
                h = heuristic(nxt, grid.goal)        # h(n)
                counter += 1
                heapq.heappush(frontier,
                               (tentative_g + h,                   # f(n)
                                h if tie_break == "prefer_goal" else 0,
                                counter, nxt))
        max_frontier = max(max_frontier, len(frontier))

    return Result(False, expanded=expanded, generated=generated,
                  max_frontier=max_frontier)


def verify(grid, result):
    """Independent replay of the plan - does not reuse successors().

    Returns a list of problems; an empty list means the plan survived every
    check.  This is what distinguishes 'the program printed a path' from 'the
    path is a valid solution'.
    """
    problems = []
    if not result.found:
        return problems
    if result.path[0] != grid.start:
        problems.append("path does not begin at the start cell")
    if result.path[-1] != grid.goal:
        problems.append("path does not end at the goal cell")
    if len(result.path) != len(result.actions) + 1:
        problems.append("path length and action count disagree")
    for cell in result.path:
        if not grid.passable(cell):
            problems.append("path enters the obstacle at %s" % (cell,))
    state = grid.start
    for action in result.actions:
        dr, dc = ACTIONS[action]
        state = (state[0] + dr, state[1] + dc)
    if state != grid.goal:
        problems.append("replaying the actions ends at %s, not the goal" % (state,))
    if result.cost != len(result.actions):
        problems.append("reported cost %s differs from the number of unit-cost "
                        "actions %d" % (result.cost, len(result.actions)))
    return problems


def describe(label, grid, result, show_map=True):
    """Print the termination report asked for in Task 1."""
    print("-" * 66)
    print(label)
    print("-" * 66)
    if not result.found:
        print("Solution found : NO - the goal is unreachable")
        print("States expanded: %d" % result.expanded)
        print()
        return
    problems = verify(grid, result)
    print("Solution found : yes")
    print("Path length    : %d moves (cost %d)" % (result.length, result.cost))
    print("States expanded: %d   (generated %d, peak frontier %d)"
          % (result.expanded, result.generated, result.max_frontier))
    print("Verification   : %s"
          % ("all checks passed" if not problems else "FAILED: %s" % problems))
    print("Path           : %s" % " ".join(str(c) for c in result.path))
    print("Actions        : %s" % " ".join(result.actions))
    if show_map:
        print()
        print(grid.render(result.path))
    print()


if __name__ == "__main__":
    grid = Grid(WAREHOUSE)
    print("=" * 66)
    print("THE WAREHOUSE")
    print("=" * 66)
    print(grid.render())
    print()
    print("Start %s   Goal %s   Manhattan distance between them: %d"
          % (grid.start, grid.goal, manhattan(grid.start, grid.goal)))
    print()
    describe("A* with Manhattan distance", grid, a_star_search(grid))
    describe("A* with Manhattan distance and goal-directed tie-breaking", grid,
             a_star_search(grid, manhattan, tie_break="prefer_goal"),
             show_map=False)
    describe("BFS (blind search) on the same problem", grid,
             breadth_first_search(grid), show_map=False)
