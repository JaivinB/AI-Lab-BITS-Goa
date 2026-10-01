"""
Lab 2 - Agents: a goal-based agent for the warehouse navigation problem.

The agent is goal-based in the sense of Russell & Norvig: it holds an explicit
representation of its goal, searches over sequences of actions, and commits to
a plan that provably reaches that goal.  A reflex agent, by contrast, would map
the current percept to an action with no notion of "where it is trying to get to"
and would be defeated by the first dead end.

Architecture (Task 2)
---------------------
    Warehouse (environment)   ->  ASCII grid, immutable during planning
    State                     ->  (row, column) of the vehicle
    Goal                      ->  state == position of 'G'
    Actions                   ->  Up / Down / Left / Right, cost 1 each
    Decision component        ->  breadth-first search over the state space,
                                  returning a shortest action sequence
    Execution                 ->  the plan is replayed one action at a time and
                                  each intermediate state is re-validated

Breadth-first search is used because every action has the same cost, so the
first time BFS reaches the goal it has found a minimum-length plan, and the
state space (one cell per free square) is small enough to explore exhaustively.
"""

from collections import deque

# The warehouse supplied in the laboratory sheet.
WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

# Action name -> (row delta, column delta).  One move = one grid square.
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

FREE, OBSTACLE, START, GOAL = ".", "#", "S", "G"


class Warehouse:
    """The environment: an immutable grid plus the start and goal positions."""

    def __init__(self, text):
        self.grid = [list(line) for line in text.strip("\n").splitlines()]
        self.rows = len(self.grid)
        self.cols = max(len(r) for r in self.grid)
        # Pad ragged lines so indexing is always safe.
        for row in self.grid:
            row.extend([OBSTACLE] * (self.cols - len(row)))
        self.start = self._locate(START)
        self.goal = self._locate(GOAL)

    def _locate(self, symbol):
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == symbol:
                    return (r, c)
        raise ValueError("symbol %r not present in the map" % symbol)

    def passable(self, state):
        """True if `state` is inside the warehouse and is not shelving."""
        r, c = state
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.grid[r][c] != OBSTACLE

    def successors(self, state):
        """Yield (action, next_state) for every legal action in `state`."""
        r, c = state
        for action, (dr, dc) in ACTIONS.items():
            nxt = (r + dr, c + dc)
            if self.passable(nxt):
                yield action, nxt

    def render(self, path=None):
        """Return the map with an optional path drawn in as '*'."""
        picture = [row[:] for row in self.grid]
        for state in path or []:
            r, c = state
            if picture[r][c] == FREE:
                picture[r][c] = "*"
        return "\n".join("".join(row) for row in picture)


class GoalBasedAgent:
    """Goal-based agent: state + goal + actions + a search-based decision step."""

    def __init__(self, environment):
        self.env = environment
        self.state = environment.start      # what the agent believes its position is
        self.goal = environment.goal        # the explicit goal
        self.plan = []                      # the action sequence it intends to follow
        self.states_expanded = 0            # bookkeeping, used in the report

    def goal_reached(self, state):
        return state == self.goal

    def formulate_plan(self):
        """Breadth-first search from the current state to the goal.

        Returns (actions, states) or (None, None) when the goal is unreachable.
        The frontier is a FIFO queue, so states are expanded in order of
        increasing plan length; `came_from` lets us rebuild the plan at the end.
        """
        frontier = deque([self.state])
        came_from = {self.state: (None, None)}   # state -> (previous state, action taken)
        self.states_expanded = 0

        while frontier:
            state = frontier.popleft()
            self.states_expanded += 1

            if self.goal_reached(state):
                return self._reconstruct(came_from, state)

            for action, nxt in self.env.successors(state):
                if nxt not in came_from:         # never enqueue a state twice
                    came_from[nxt] = (state, action)
                    frontier.append(nxt)

        return None, None

    @staticmethod
    def _reconstruct(came_from, state):
        actions, states = [], [state]
        while came_from[state][0] is not None:
            state, action = came_from[state]
            actions.append(action)
            states.append(state)
        actions.reverse()
        states.reverse()
        return actions, states

    def execute(self, actions):
        """Replay the plan one action at a time, re-checking every step.

        This is the independent verification of the plan: the search could be
        wrong, but a plan that walks through shelving will be caught here.
        """
        state = self.env.start
        visited = [state]
        for action in actions:
            dr, dc = ACTIONS[action]
            nxt = (state[0] + dr, state[1] + dc)
            if not self.env.passable(nxt):
                raise AssertionError("plan enters an obstacle at %s" % (nxt,))
            state = nxt
            visited.append(state)
        if not self.goal_reached(state):
            raise AssertionError("plan ends at %s, not at the goal %s" % (state, self.goal))
        self.state = state
        return visited


def report(environment, label="Warehouse"):
    """Plan, verify and print the result for one environment."""
    agent = GoalBasedAgent(environment)
    actions, states = agent.formulate_plan()

    print("=" * 62)
    print(label)
    print("=" * 62)
    print(environment.render())
    print()
    print("Start: %s    Goal: %s" % (environment.start, environment.goal))

    if actions is None:
        print("No collision-free path exists from S to G.")
        print("States expanded: %d" % agent.states_expanded)
        print()
        return None

    walked = agent.execute(actions)          # independent replay of the plan
    print("Path found with %d moves (states expanded: %d)."
          % (len(actions), agent.states_expanded))
    print()
    print("Plan:")
    print("  " + " -> ".join(actions))
    print()
    print("Positions visited (row, col):")
    print("  " + " ".join(str(s) for s in walked))
    print()
    print("Path drawn on the map ('*' = route taken):")
    print(environment.render(states))
    print()
    print("Verification: plan replayed step by step, every cell passable, "
          "final cell is G.")
    print()
    return actions


if __name__ == "__main__":
    report(Warehouse(WAREHOUSE), "Task 3 - the warehouse supplied in the laboratory sheet")

    # A goal-based agent must also recognise that a goal is unattainable
    # rather than wandering indefinitely (the reflex agent's failure mode).
    walled_off = """\
#########
#S....###
#####.###
####..#G#
#########"""
    report(Warehouse(walled_off), "Extra check - goal sealed behind shelving")
