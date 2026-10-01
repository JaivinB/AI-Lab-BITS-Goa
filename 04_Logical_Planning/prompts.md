# LLM prompts used in Lab 4

The assistant used was Claude. The planning problem (Task 0) and the hand plan
(Task 1) were written before any prompt was issued.

## Prompt 1 — the planner (Task 2)

> I want to implement a simple planning agent in Python. Represent a state as a
> set of logical propositions. Each action should contain:
> - a name;
> - positive preconditions;
> - negative preconditions;
> - positive effects;
> - negative effects.
>
> An action is applicable if all of its preconditions are satisfied by the
> current state. When an action is applied:
> 1. remove its negative effects from the state;
> 2. add its positive effects to the state.
>
> Use breadth-first search to find a sequence of actions that achieves a
> specified goal. The program should also:
> - detect when no plan exists;
> - print the resulting sequence of actions;
> - print the states reached after each action.
>
> Explain the implementation and identify any assumptions you make.

## Prompt 2 — the warehouse domain

> Instantiate that planner for this domain. Three locations a, b, c with
> `connected(a,b)`, `connected(b,a)`, `connected(b,c)`, `connected(c,b)`.
> `Move(X,Y)` has precondition `At(Robot,X)`, adds `At(Robot,Y)` and deletes
> `At(Robot,X)`. `PickUp(Package,L)` has preconditions `At(Robot,L)` and
> `At(Package,L)`, adds `Holding(Package)` and deletes `At(Package,L)`.
> `Drop(Package,L)` has preconditions `At(Robot,L)` and `Holding(Package)`,
> adds `At(Package,L)` and deletes `Holding(Package)`. Initial state
> `{At(Robot,a), At(Package,a)}`, goal `{At(Package,c)}`.

## Prompt 3 — keep logic and search separable

> Make sure the applicability test and the search are separate functions, so
> that the logical component and the search component can be pointed at
> individually. The search must consult the logic only through the
> applicability test.

## Prompt 4 — independent plan validation (Task 5)

> Write a `validate_plan(initial, plan, goal)` function that replays a plan
> step by step, checks each action's preconditions against the state in which
> that action actually executes, and reports which precondition failed and at
> which step. It must not call the search.

## Prompt 5 — explain the plan's validity

> For every action in the plan you produced, identify its preconditions and
> show that those preconditions are satisfied in the state in which the action
> is executed.

The explanation returned was fluent and, in this case, correct. It was compared
against the output of `validate_plan`, which is what actually establishes the
claim — see the Task 5 section of the report.

## Prompt 6 — the Prolog verifier (Tasks 6–8)

> Write a SWI-Prolog file containing the warehouse connectivity facts, a rule
> `can_move(X,Y) :- connected(X,Y)`, a rule `valid_move(X,Y)`, and the
> `wet_road`/`slippery`/`reduce_speed` chain. Add a predicate that checks a
> whole list of locations is a valid route, and a `demo` predicate that runs
> every query and prints true or false for each.

## Changes made to the generated code before accepting it

1. **The goal test was equality rather than entailment.** The generated code
   tested `state == goal`, which fails on the warehouse problem: the final state
   is `{At(Package,c), At(Robot,c)}` and the goal is `{At(Package,c)}`. Equality
   would have reported *no plan found* for a solvable problem. It was changed to
   `goal <= state`, i.e. `state |= goal`, which is what the specification
   actually says.

2. **Negative preconditions were accepted but never checked.** `Action.__init__`
   stored `neg_pre`, and `applicable` tested only `pos_pre <= state`. Every
   action in this domain happens to have no negative preconditions, so every
   test passed and the bug was invisible. The check `not (self.neg_pre & state)`
   was added, and Test C 2a was written specifically to exercise it — it is the
   only test in the lab that does.

3. **Effects were applied in the wrong order.** The generated `apply` added the
   positive effects first and then removed the negative ones, so an action with
   a proposition in both sets would delete it. Correct STRIPS semantics is
   delete-then-add. No action here has that overlap, so again nothing failed;
   the order was fixed because the next domain would break it.

4. **Visited-state bookkeeping was missing**, so the search would have revisited
   states and looped indefinitely on Test B (the robot can oscillate
   `a -> b -> a` for ever). A `visited` set was added, which is what makes
   "no plan found" a sound conclusion rather than a timeout.

5. The generated code printed the plan but not the intermediate states, although
   the prompt asked for both. The state trail is now carried along each
   frontier entry and printed beside each action.
