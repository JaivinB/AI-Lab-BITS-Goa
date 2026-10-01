"""
Lab 3, Tasks 3, 5 and 6 - testing the agent and investigating the heuristic.

Task 3 tests the program on cases whose correct answer is known in advance.
Task 5 compares A* with blind search on the unchanged warehouse.
Task 6 investigates what happens when the heuristic is weakened (h = 0,
Euclidean) or inflated beyond admissibility (2 x Manhattan).

A program that prints a plausible path is not a validated algorithm, so every
plan here is replayed by `verify()`, which does not share code with the search.
"""

from search_agent import (WAREHOUSE, HEURISTICS, Grid, a_star_search,
                          breadth_first_search, describe, manhattan, verify)

# --------------------------------------------------------------------------
# Test maps.  Each one has an answer that is known without running the code.
# --------------------------------------------------------------------------
TRIVIAL = """\
#####
#SG##
#####"""

NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""

# Two routes to G: four moves down the left wall, or twenty-two the long way
# round.  A correct shortest-path search must return the four-move plan.
ALTERNATIVE_PATHS = """\
###########
#S........#
#.#######.#
#.#######.#
#.#######.#
#G........#
###########"""

# An open floor with no corridors, so the heuristic has real choices to guide.
OPEN_FLOOR = """\
#################
#S..............#
#...............#
#...............#
#...............#
#...............#
#..............G#
#################"""

# A partly open floor on which 2 x Manhattan demonstrably loses optimality.
# Found by searching random layouts for a case where the inflated heuristic
# returns a longer path than the admissible one.
INADMISSIBILITY_DEMO = """\
###############
#S.......####.#
#.....#.....#.#
#.......###..##
#.....##......#
#.####.#.#....#
#......#.##.#.#
#......#.##..G#
###############"""


def heading(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


# --------------------------------------------------------------------------
# Task 3 - systematic testing
# --------------------------------------------------------------------------
def task3():
    heading("TASK 3 - TESTING THE GENERATED PROGRAM")

    print("Test 1: the warehouse supplied in the laboratory sheet")
    print("Expected: a path exists; length must be at least the Manhattan")
    print("distance of 20 between S and G.")
    print()
    grid = Grid(WAREHOUSE)
    describe("Test 1 - A* on the original warehouse", grid, a_star_search(grid))

    print("Test 2: trivial case - the goal is adjacent to the start")
    print("Expected: exactly one move.")
    print()
    grid = Grid(TRIVIAL)
    result = a_star_search(grid)
    describe("Test 2 - goal adjacent to start", grid, result)
    print("  one-step solution: %s" % (result.length == 1))
    print()

    print("Test 3: no solution - the goal is sealed off")
    print("Expected: termination with failure, not an infinite loop.")
    print()
    grid = Grid(NO_SOLUTION)
    result = a_star_search(grid)
    describe("Test 3 - unreachable goal", grid, result)
    print("  reported failure rather than looping: %s" % (not result.found))
    print()

    print("Test 4: alternative paths - two routes of different length")
    print("Expected: the four-move route down the left wall, not the")
    print("twenty-two-move route the long way round.")
    print()
    grid = Grid(ALTERNATIVE_PATHS)
    result = a_star_search(grid)
    describe("Test 4 - shortest of two available routes", grid, result)
    print("  shortest route returned: %s (length %d, alternative is 22)"
          % (result.length == 4, result.length))
    print()


# --------------------------------------------------------------------------
# Task 5 - A* against blind search
# --------------------------------------------------------------------------
def _comparison_table(label, text):
    grid = Grid(text)
    bfs = breadth_first_search(grid)
    astar = a_star_search(grid, manhattan)
    tuned = a_star_search(grid, manhattan, tie_break="prefer_goal")
    for name, result in (("BFS", bfs), ("A*", astar), ("A* tuned", tuned)):
        problems = verify(grid, result)
        if problems:
            print("  %s FAILED VERIFICATION: %s" % (name, problems))

    print(label)
    print("  %-18s %10s %10s %14s" % ("Measure", "BFS", "A*", "A* + tie-break"))
    print("  " + "-" * 56)
    print("  %-18s %10s %10s %14s" % ("Solution found",
                                      "yes" if bfs.found else "no",
                                      "yes" if astar.found else "no",
                                      "yes" if tuned.found else "no"))
    print("  %-18s %10d %10d %14d" % ("Path length", bfs.length, astar.length,
                                      tuned.length))
    print("  %-18s %10d %10d %14d" % ("States expanded", bfs.expanded,
                                      astar.expanded, tuned.expanded))
    print("  %-18s %10d %10d %14d" % ("Peak frontier", bfs.max_frontier,
                                      astar.max_frontier, tuned.max_frontier))
    print()
    return bfs, astar, tuned


def task5():
    heading("TASK 5 - A* COMPARED WITH BLIND SEARCH (warehouse unchanged)")
    bfs, astar, tuned = _comparison_table(
        "The warehouse from the laboratory sheet:", WAREHOUSE)
    print("  Both find a %d-move path and both expand %d states - the heuristic"
          % (astar.length, astar.expanded))
    print("  bought nothing at all here.  The peak frontier of %d explains why:"
          % astar.max_frontier)
    print("  the map is almost entirely single-width corridor, so at nearly")
    print("  every step there is only one successor to choose from.  A heuristic")
    print("  can only help by reordering choices, and this map offers almost")
    print("  none.  A* is not better merely because it is better informed; it is")
    print("  better only when there is a decision for the information to inform.")
    print()

    print("The same two algorithms on an open floor of comparable size, where")
    print("the search does have choices to make:")
    print()
    bfs, astar, tuned = _comparison_table("Open floor, 15 x 6 free cells:",
                                          OPEN_FLOOR)
    print("  A* with Manhattan distance still expands %d states - exactly the"
          % astar.expanded)
    print("  same as BFS, and almost the whole floor.  This is not a bug and not")
    print("  a bad heuristic.  On a unit-cost grid every cell on any monotone")
    print("  staircase from S to G has f(n) = g(n) + h(n) = %d, the optimal cost"
          % astar.length)
    print("  exactly.  Admissibility guarantees A* never expands a state with")
    print("  f > f*, but it says nothing about the states with f = f*, and here")
    print("  almost every free cell is one of them.  A* is therefore free to")
    print("  expand them in any order, and insertion order fans out like BFS.")
    print()
    print("  Breaking f-ties towards the goal - prefer the smaller h(n) - cuts")
    print("  expansions from %d to %d, a factor of %.1f, with the same %d-move"
          % (astar.expanded, tuned.expanded,
             astar.expanded / tuned.expanded, tuned.length))
    print("  optimal path.  Optimality is untouched, because no state with")
    print("  f > f* is expanded under either rule.")
    print()


# --------------------------------------------------------------------------
# Task 6 - investigating the heuristic
# --------------------------------------------------------------------------
def _heuristic_table(label, text, optimal_length=None):
    grid = Grid(text)
    if optimal_length is None:
        optimal_length = a_star_search(grid, manhattan).length

    print(label)
    print("  %-14s %9s %7s %22s %10s"
          % ("h(n)", "solution", "length", "expanded (fifo/tied)", "optimal?"))
    print("  " + "-" * 68)
    for name in ("manhattan", "zero", "euclidean", "2x manhattan"):
        plain = a_star_search(grid, HEURISTICS[name])
        tuned = a_star_search(grid, HEURISTICS[name], tie_break="prefer_goal")
        for result in (plain, tuned):
            problems = verify(grid, result)
            if problems:
                print("  %s FAILED VERIFICATION: %s" % (name, problems))
        print("  %-14s %9s %7d %22s %10s"
              % (name,
                 "yes" if plain.found else "no",
                 plain.length,
                 "%d / %d" % (plain.expanded, tuned.expanded),
                 "yes" if plain.length == optimal_length else
                 "NO (+%d)" % (plain.length - optimal_length)))
    print()


def task6():
    heading("TASK 6 - INVESTIGATING THE HEURISTIC")
    print("Why Manhattan distance is appropriate here: the robot moves only")
    print("horizontally and vertically, one cell per move, at cost 1.  Reaching")
    print("the goal therefore requires at least |dx| + |dy| moves, so Manhattan")
    print("distance never overestimates the remaining cost - it is admissible -")
    print("and it is exact on an empty floor.  Obstacles can only make the true")
    print("cost larger, never smaller, so admissibility survives them.")
    print()
    print("Each row reports expansions under both tie-breaking rules, because on")
    print("a unit-cost grid the tie rule matters as much as the heuristic.")
    print()

    _heuristic_table("Warehouse from the laboratory sheet:", WAREHOUSE)
    print("  Corridors flatten every difference: all four heuristics return the")
    print("  same 40-move path and expand the same 63 states, and tie-breaking")
    print("  changes nothing either.  Even h = 0 - no information at all - costs")
    print("  nothing on this map, because there is essentially one route to")
    print("  find.  No experiment on this warehouse alone could distinguish a")
    print("  good heuristic from a useless one.")
    print()

    _heuristic_table("Open floor (same four heuristics):", OPEN_FLOOR)
    print("  With room to choose, the heuristics separate - but only once ties")
    print("  are broken towards the goal:")
    print("    h = 0        89 / 89  - degenerates to uniform-cost search; it")
    print("                 carries no information, so it cannot even break a")
    print("                 tie, and the tuned column is identical.")
    print("    Euclidean    89 / 71  - admissible, but on 4-connected moves it")
    print("                 underestimates by more than Manhattan does, so it")
    print("                 is the weaker of the two admissible heuristics.")
    print("    Manhattan    89 / 19  - exact on an open floor; with goal-directed")
    print("                 tie-breaking it expands only the 19 cells of the")
    print("                 path itself.")
    print("    2 x Manhattan 19 / 19 - fewest expansions of all, and it needs no")
    print("                 tie-breaking, because inflating h destroys the ties.")
    print("  The ranking by expansions follows the ranking by how much the")
    print("  heuristic knows - which is the answer to 'what information does the")
    print("  algorithm use to decide where to search next?'")
    print()

    heading("TASK 6 (continued) - WHAT AN INADMISSIBLE HEURISTIC COSTS")
    print("2 x Manhattan can exceed the true remaining cost, so A* loses its")
    print("optimality guarantee.  On both maps above the returned path stayed")
    print("optimal anyway, which is itself the warning: a broken heuristic can")
    print("look perfectly fine on the maps you happen to test.  On a layout with")
    print("genuinely competing routes it does fail:")
    print()
    _heuristic_table("Partly open floor with competing routes:",
                     INADMISSIBILITY_DEMO)
    print("  2 x Manhattan expands fewest - 39 states against 55 - and returns a")
    print("  22-move path where the optimum is 18.  That is the trade exactly:")
    print("  inflating h buys search effort with solution quality.  Weighted A*")
    print("  with weight w returns a path no worse than w times optimal, and")
    print("  22 <= 2 x 18 respects that bound.  Too optimistic a heuristic")
    print("  (h = 0) is safe but slow; too aggressive a heuristic is fast but")
    print("  can be wrong, and the error is silent - the program still prints a")
    print("  valid path, just not a shortest one.  Only a test that knows the")
    print("  true optimum detects it.")
    print()


if __name__ == "__main__":
    task3()
    task5()
    task6()
