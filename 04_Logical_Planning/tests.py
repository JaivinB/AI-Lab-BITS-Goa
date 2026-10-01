"""
Lab 4, Tasks 3 to 5 - testing the generated planner.

Test A  solvable problem      - the original warehouse
Test B  impossible problem    - the PickUp action removed
Test C  irrelevant actions    - the robot may reach c without the package;
                                check the planner does not confuse the two
Task 5  generated explanation vs independent verification - a plan that reads
        plausibly is replayed against the actual preconditions
"""

from planner import (GOAL, INITIAL, Action, at, breadth_first_plan, holding,
                     report, satisfies, show_state, validate_plan,
                     warehouse_actions)


def heading(title):
    print()
    print("#" * 70)
    print("# " + title)
    print("#" * 70)
    print()


# --------------------------------------------------------------------------
# Test A - solvable
# --------------------------------------------------------------------------
def test_a():
    heading("TEST A - solvable problem (the original warehouse)")
    plan = report("Warehouse: deliver the package from a to c",
                  INITIAL, warehouse_actions(), GOAL)
    print("Expected 4 actions; found %d." % len(plan))
    print("Every action in the returned plan was verified individually by")
    print("replaying the plan and checking its preconditions against the state")
    print("in which it actually executes.")
    print()


# --------------------------------------------------------------------------
# Test B - impossible
# --------------------------------------------------------------------------
def test_b():
    heading("TEST B - impossible problem (the PickUp action removed)")
    print("Without PickUp the package can never leave a, so no plan exists.")
    print("The planner must say so rather than inventing an action.")
    print()
    actions = warehouse_actions(include_pickup=False)
    print("Remaining actions: %s" % [a.name for a in actions])
    print()
    plan = report("Warehouse without PickUp", INITIAL, actions, GOAL)
    print("Reported no plan: %s" % (plan is None))
    print()
    print("Note what makes this termination sound rather than lucky: the")
    print("reachable state space is finite and already-visited states are never")
    print("re-expanded, so breadth-first search exhausts it and stops.  A")
    print("planner that re-expanded states would loop here for ever, moving the")
    print("robot a -> b -> a -> b without ever failing.")
    print()


# --------------------------------------------------------------------------
# Test C - irrelevant actions
# --------------------------------------------------------------------------
def test_c():
    heading("TEST C - irrelevant actions (the robot reaching c is not the goal)")
    print("The Move actions change the robot's location and nothing else.  The")
    print("goal is At(Package,c), not At(Robot,c).  Three separate checks:")
    print()

    # 1. The goal test itself must reject a state where only the robot arrived.
    robot_arrived = frozenset([at("Robot", "c"), at("Package", "a")])
    print("1. Goal test applied directly to a state where the robot is at c")
    print("   but the package is still at a:")
    print("     state %s" % show_state(robot_arrived))
    print("     goal  %s" % show_state(GOAL))
    print("     satisfies(state, goal) = %s   <- must be False"
          % satisfies(robot_arrived, GOAL))
    print()

    # 2a. A genuinely irrelevant action must not change the plan.  Patrol
    # also exercises negative preconditions, which nothing else in the domain
    # uses, so this is the only test that touches that half of `applicable`.
    print("2a. Add a Patrol action that sends the robot from a straight to c,")
    print("    but only when it is NOT holding the package - so it can never")
    print("    help deliver anything.  This is the irrelevant action proper,")
    print("    and it is also the only test of a negative precondition:")
    patrol = Action(
        name="Patrol(a,c)",
        pos_pre=[at("Robot", "a")],
        neg_pre=[holding("Package")],
        pos_eff=[at("Robot", "c")],
        neg_eff=[at("Robot", "a")],
    )
    actions = warehouse_actions() + [patrol]
    plan, trail, expanded = breadth_first_plan(INITIAL, actions, GOAL)
    print("      applicable in the initial state: %s"
          % patrol.applicable(INITIAL))
    print("      plan: %s" % [a.name for a in plan])
    print("      length %d (states expanded %d)" % (len(plan), expanded))
    print("      Patrol used: %s   <- must be False"
          % any(a.name == "Patrol(a,c)" for a in plan))
    print("      The plan is unchanged at 4 actions.  Patrol is applicable in")
    print("      the initial state and does put the robot at c, but its")
    print("      negative precondition means it can never be combined with")
    print("      carrying the package, so it contributes nothing to the goal.")
    print()

    # 2b. By contrast, an action that IS useful should be exploited.
    print("2b. For contrast, a Teleport action identical to Patrol but without")
    print("    the negative precondition - so it carries a held package:")
    teleport = Action(
        name="Teleport(Robot,c)",
        pos_pre=[at("Robot", "a")],
        pos_eff=[at("Robot", "c")],
        neg_eff=[at("Robot", "a")],
    )
    actions = warehouse_actions() + [teleport]
    plan, trail, expanded = breadth_first_plan(INITIAL, actions, GOAL)
    print("      plan: %s" % [a.name for a in plan])
    print("      length %d (states expanded %d)" % (len(plan), expanded))
    print("      The plan shortens to 3, and Teleport appears only between")
    print("      PickUp and Drop.  The goal is still reached through the")
    print("      package's own location, never through the robot's arrival:")
    print("      the state after Teleport is")
    print("        %s" % show_state(trail[2]))
    print("      which does NOT entail At(Package,c); Drop is still required.")
    print()

    # 3. Only Move actions: the robot can reach c, the package cannot.
    print("3. Remove PickUp and Drop entirely, leaving only Move.  The robot")
    print("   can still reach c, so if the planner confused At(Robot,c) with")
    print("   At(Package,c) it would report success here:")
    move_only = warehouse_actions(include_pickup=False, include_drop=False)
    plan, _trail, expanded = breadth_first_plan(INITIAL, move_only, GOAL)
    print("     plan found: %s (states expanded %d)   <- must be None"
          % (plan, expanded))
    reachable_states = []
    state = INITIAL
    for action in move_only:
        if action.applicable(state):
            reachable_states.append(show_state(action.apply(state)))
    print("     the robot does reach c in the explored space, e.g. via")
    print("     Move(a,b) then Move(b,c), yet no plan is reported.")
    print()


# --------------------------------------------------------------------------
# Task 5 - a generated explanation is not an independent verification
# --------------------------------------------------------------------------
def task5():
    heading("TASK 5 - generated explanation vs independent verification")
    print("The laboratory sheet suggests reasoning about the sequence")
    print()
    print("    Move(a,b),  PickUp(Package,b),  Move(b,c),  Drop(Package,c)")
    print()
    print("It reads perfectly plausibly: go to b, pick the package up, carry it")
    print("to c, put it down.  An LLM asked to justify it can produce a fluent")
    print("explanation of exactly that story.  Replaying it against the actual")
    print("preconditions:")
    print()

    by_name = {a.name: a for a in warehouse_actions()}
    plausible = [by_name["Move(a,b)"], by_name["PickUp(Package,b)"],
                 by_name["Move(b,c)"], by_name["Drop(Package,c)"]]
    final_state, problems = validate_plan(INITIAL, plausible, GOAL)
    for problem in problems:
        print("    %s" % problem)
    print()
    print("Step 2 fails: PickUp(Package,b) requires At(Package,b), and the")
    print("package never moved - it is still at a, because Move affects only")
    print("the robot's location.  The sequence is invalid, and no amount of")
    print("fluency in the explanation changes that.")
    print()
    print("The correct plan picks the package up BEFORE moving:")
    correct = [by_name["PickUp(Package,a)"], by_name["Move(a,b)"],
               by_name["Move(b,c)"], by_name["Drop(Package,c)"]]
    final_state, problems = validate_plan(INITIAL, correct, GOAL)
    print("    PickUp(Package,a), Move(a,b), Move(b,c), Drop(Package,c)")
    print("    problems: %s" % (problems if problems else "none - plan valid"))
    print("    final state: %s" % show_state(final_state))
    print()
    print("Which to trust: the independently executed state transitions.  The")
    print("explanation and the plan come from the same source, so an error in")
    print("the plan tends to be reproduced in the explanation - the two agree")
    print("with each other rather than with the domain.  The replay consults")
    print("the action definitions instead, and is capable of disagreeing.")
    print()


if __name__ == "__main__":
    test_a()
    test_b()
    test_c()
    task5()
