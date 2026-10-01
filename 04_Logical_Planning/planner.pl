% Lab 4, optional extension - Prolog as an independent verifier.
%
% The Python planner in planner.py GENERATES a candidate plan.  This file
% CHECKS the moves in that plan against a logical description of the
% warehouse, written separately as facts and rules.  The architecture is
%
%     generate (Python)  ->  independent verification (Prolog)
%
% Run with:   swipl planner.pl
% then enter the queries marked below, or run them all with:  ?- demo.

% ---------------------------------------------------------------------------
% Task 6 - facts describing the warehouse
% ---------------------------------------------------------------------------
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% A rule: Connected(X,Y) -> CanMove(X,Y).
can_move(X, Y) :-
    connected(X, Y).

% ?- can_move(a, b).      expected: true
% ?- can_move(a, c).      expected: false

% ---------------------------------------------------------------------------
% Task 7 - checking a proposed plan
% ---------------------------------------------------------------------------
valid_move(X, Y) :-
    connected(X, Y).

% ?- valid_move(a, b).    expected: true
% ?- valid_move(b, c).    expected: true
% ?- valid_move(a, c).    expected: false  (the challenge case)

% A whole sequence of moves is valid when every consecutive step is.  This
% checks the movement skeleton of the plan the Python planner produced.
valid_route([_]).
valid_route([X, Y | Rest]) :-
    valid_move(X, Y),
    valid_route([Y | Rest]).

% ?- valid_route([a, b, c]).   expected: true   (the plan's movements)
% ?- valid_route([a, c]).      expected: false  (the proposed Move(a,c))

% Reachability, for contrast: a *route* from X to Y may exist even when no
% single move does.  This is the distinction the can_move queries expose.
reachable(X, X).
reachable(X, Y) :-
    connected(X, Z),
    reachable(Z, Y).

% ?- reachable(a, c).     expected: true   - but via b, not in one move

% ---------------------------------------------------------------------------
% Task 8 - facts, rules, inference
% ---------------------------------------------------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% ?- reduce_speed.        expected: true

% ---------------------------------------------------------------------------
% Run every query in one go.
% ---------------------------------------------------------------------------
check(Goal) :-
    (   call(Goal)
    ->  format("~w.~30|true~n", [Goal])
    ;   format("~w.~30|false~n", [Goal])
    ).

demo :-
    format("Task 6 - can_move~n"),
    check(can_move(a, b)),
    check(can_move(a, c)),
    format("~nTask 7 - valid_move and the proposed plan~n"),
    check(valid_move(a, b)),
    check(valid_move(b, c)),
    check(valid_move(a, c)),
    check(valid_route([a, b, c])),
    check(valid_route([a, c])),
    check(reachable(a, c)),
    format("~nTask 8 - facts + rules + inference~n"),
    check(reduce_speed).

:- initialization(demo).
