"""
Lab 4 - Logical reasoning for planning:  Logic + Search = Planning

A planning problem is (I, A, G).  A state is a set of logical propositions.
Each action carries positive and negative preconditions and positive and
negative effects.

    logic   decides whether  S |= Preconditions(a)   -> what is possible
    search  decides which applicable sequence to try -> what to attempt

The two are deliberately kept in separate functions below, so that the division
of labour is visible in the code and not only in the report:

    Action.applicable(state)   is the entire logical component
    breadth_first_plan(...)    is the entire search component

The warehouse domain: three locations a, b, c, connected a-b and b-c.  The
robot and the package both start at a; the package must end at c.
"""

from collections import deque

# ---------------------------------------------------------------------------
# Propositions are plain strings.  A state is a frozenset of them.
# ---------------------------------------------------------------------------
LOCATIONS = ("a", "b", "c")
CONNECTED = (("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"))


def at(thing, location):
    return "At(%s,%s)" % (thing, location)


def holding(thing):
    return "Holding(%s)" % thing


class Action:
    """A STRIPS-style action: preconditions and effects, both signed."""

    def __init__(self, name, pos_pre=(), neg_pre=(), pos_eff=(), neg_eff=()):
        self.name = name
        self.pos_pre = frozenset(pos_pre)
        self.neg_pre = frozenset(neg_pre)
        self.pos_eff = frozenset(pos_eff)
        self.neg_eff = frozenset(neg_eff)

    def applicable(self, state):
        """THE LOGICAL COMPONENT.  True iff  state |= Preconditions(self).

        Every positive precondition must be present in the state, and no
        negative precondition may be present.  A closed-world assumption is in
        force: a proposition absent from the state is taken to be false.
        """
        return self.pos_pre <= state and not (self.neg_pre & state)

    def apply(self, state):
        """S' = Apply(S, a):  delete the negative effects, then add the
        positive ones.  Deletion first, so an effect that is both deleted and
        added ends up present."""
        return frozenset((state - self.neg_eff) | self.pos_eff)

    def unsatisfied(self, state):
        """Which preconditions fail in `state` - used to explain a refusal."""
        missing = sorted(self.pos_pre - state)
        present = sorted(self.neg_pre & state)
        return missing, present

    def __repr__(self):
        return self.name


def warehouse_actions(include_pickup=True, include_drop=True,
                      include_move=True):
    """The action set.  The flags let a test remove a capability (Test B)."""
    actions = []
    if include_move:
        for source, destination in CONNECTED:
            actions.append(Action(
                name="Move(%s,%s)" % (source, destination),
                pos_pre=[at("Robot", source)],
                pos_eff=[at("Robot", destination)],
                neg_eff=[at("Robot", source)],
            ))
    if include_pickup:
        for location in LOCATIONS:
            actions.append(Action(
                name="PickUp(Package,%s)" % location,
                pos_pre=[at("Robot", location), at("Package", location)],
                pos_eff=[holding("Package")],
                neg_eff=[at("Package", location)],
            ))
    if include_drop:
        for location in LOCATIONS:
            actions.append(Action(
                name="Drop(Package,%s)" % location,
                pos_pre=[at("Robot", location), holding("Package")],
                pos_eff=[at("Package", location)],
                neg_eff=[holding("Package")],
            ))
    return actions


INITIAL = frozenset([at("Robot", "a"), at("Package", "a")])
GOAL = frozenset([at("Package", "c")])


def satisfies(state, goal):
    """The goal test is itself entailment: state |= goal."""
    return goal <= state


def breadth_first_plan(initial, actions, goal, max_states=100000):
    """THE SEARCH COMPONENT.

    Breadth-first search over states.  Logic is consulted only through
    `applicable`; search decides the order in which applicable actions are
    tried.  Returns (plan, states, expanded) or (None, None, expanded) when no
    plan exists.  Terminates on an unsolvable problem because the state space
    is finite and visited states are never re-expanded.
    """
    frontier = deque([(initial, [], [initial])])
    visited = {initial}
    expanded = 0

    while frontier:
        state, plan, trail = frontier.popleft()
        expanded += 1
        if satisfies(state, goal):
            return plan, trail, expanded
        for action in actions:
            if action.applicable(state):              # logic
                successor = action.apply(state)
                if successor not in visited:          # search bookkeeping
                    visited.add(successor)
                    frontier.append(
                        (successor, plan + [action], trail + [successor]))
        if len(visited) > max_states:                 # safety valve
            break

    return None, None, expanded


def show_state(state):
    return "{" + ", ".join(sorted(state)) + "}"


def validate_plan(initial, plan, goal):
    """Independent verification: replay the plan, checking each precondition
    against the state in which the action is actually executed.

    This does not call the search.  A generated explanation that the plan is
    valid is not the same thing as an independent check that it is, which is
    the point of Task 5.
    """
    state = initial
    problems = []
    for step, action in enumerate(plan, start=1):
        if not action.applicable(state):
            missing, present = action.unsatisfied(state)
            problems.append(
                "step %d: %s not applicable - missing %s, forbidden present %s"
                % (step, action.name, missing, present))
            break
        state = action.apply(state)
    if not satisfies(state, goal):
        problems.append("final state %s does not entail the goal %s"
                        % (show_state(state), show_state(goal)))
    return state, problems


def report(label, initial, actions, goal):
    """Run one planning problem and print everything the lab asks to record."""
    print("=" * 70)
    print(label)
    print("=" * 70)
    print("Initial state : %s" % show_state(initial))
    print("Goal          : %s" % show_state(goal))
    print("Actions        : %d available" % len(actions))

    plan, trail, expanded = breadth_first_plan(initial, actions, goal)

    if plan is None:
        print("Plan found    : NO - no plan found (states expanded: %d)" % expanded)
        print()
        print("The planner reports failure rather than inventing an action.")
        print()
        return None

    print("Plan found    : yes, %d actions (states expanded: %d)"
          % (len(plan), expanded))
    print()
    print("Plan and the state after every action:")
    print("  S0            %s" % show_state(trail[0]))
    for index, action in enumerate(plan):
        print("  %-13s %s" % (action.name, show_state(trail[index + 1])))
    print()

    final_state, problems = validate_plan(initial, plan, goal)
    print("Independent replay of the plan:")
    if problems:
        print("  INVALID: %s" % problems)
    else:
        print("  every precondition satisfied in the state where the action")
        print("  executes, and the final state entails the goal -> plan valid")
    print()
    return plan


def applicability_in_initial_state(actions, initial):
    """Task 0: which actions are applicable in I, and why not the others."""
    print("=" * 70)
    print("TASK 0 - which actions are applicable in the initial state?")
    print("=" * 70)
    print("Initial state: %s" % show_state(initial))
    print()
    for action in actions:
        if action.applicable(initial):
            print("  APPLICABLE     %-20s preconditions %s all hold"
                  % (action.name, sorted(action.pos_pre)))
    print()
    for action in actions:
        if not action.applicable(initial):
            missing, _present = action.unsatisfied(initial)
            print("  not applicable %-20s missing %s"
                  % (action.name, missing))
    print()


if __name__ == "__main__":
    actions = warehouse_actions()
    applicability_in_initial_state(actions, INITIAL)
    report("TEST A - the original warehouse problem (solvable)",
           INITIAL, actions, GOAL)
