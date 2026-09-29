# Solver experiments (retired)

Alternatives to `g_solve.py` (the MILP that draws the app's minimal-W zigzag for a word g),
tried September 2026.  None replaced it; they are kept because they independently confirm
its answers and produced two lasting results (below).

Run them with the venv, which has OR-Tools (Homebrew's Python refuses pip installs):

    PYTHONPATH=.:attic/solver_experiments .venv/bin/python -c "import g_walk as G; print(G.solve_min([1,2],3))"

## What was tried

| File / function | Model | Verdict |
|---|---|---|
| `g_cpsat.solve_min` | g_solve's node/segment model in CP-SAT, per W, lazy pair constraints, `linearization_level=2` | 35-40% faster than g_solve in total, but rare 10-13 s outliers |
| `g_cpsat.solve_min_varw` | W as a variable, minimised in one solve (q bounded to -6..6, tier 3 only on height overlap) | needs 8 workers; then totals about equal to g_solve, best worst case (1.3 s) |
| `g_walk.feasible` | the knot as a closed lattice walk: one AllDifferent per half-column replaces tiers 2 and 3; crossings found by pairwise height tests | slow to construct; superseded |
| `g_walk.feasible_rank` | the same walk with a rank variable per step (a crossing is a rank change) | simplest model, 15-20x slower than g_solve; can switch the no-wiggle rule off |

All four agree with g_solve on W and verdict for all 1177 benchmark knots (727 at L=3..9
|g|<=10, 150 at L=5 |g|=12, 100 at L=6 |g|=11, 200 random words at L=8..11 |g| 12..18).

## Lessons

- What made these problems fast was a strong continuous relaxation (the MILP, or CP-SAT
  with its LP turned up), adding pair constraints lazily, and for CP-SAT's variable-W
  model the parallel portfolio -- not SAT-style search on its own.
- CP-SAT's modulo is truncated (sign of the dividend): the remainder variable must range
  over (-W, W).
- In the walk model, rank variables were worth 50-125x over pairwise height comparisons.

## Results that outlived the experiments

- **Wiggles.**  g_solve's W is minimal only among drawings without wiggles (runs between
  bights with no crossing).  With wiggles allowed, 4 of the 727 library knots get
  narrower (all L=5): `1 1 1 2 1 3 2 4 4 4` 10->8, `1 1 1 2 3 2 3 4 4 4` 12->8,
  `1 1 1 2 3 2 4 3 4 4` 10->8, `1 1 2 1 3 2 4 3 4 4` 8->6.  Recorded in g_solve's caveats.
- **Symmetric drawings.**  `feasible_rank(extra=...)` showed that the two self-flip words
  whose g_solve drawing was not flip-symmetric do have symmetric drawings at the same W;
  g_solve now finds them itself (`solve_symmetric`).
- **Rows of bights.**  `feasible_rank(min_heights=True)` minimises distinct bight heights:
  504 of 727 knots can use fewer at a larger W.  Dropped as a criterion (the differences
  are small wiggles the rendering largely hides).
