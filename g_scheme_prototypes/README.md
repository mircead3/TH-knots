# g→zigzag direct-constructor prototypes (WIP)

Scratchpad prototypes preserving the state of the g-scheme constructor work
(2026-09-15/16). **Not wired into the app or `construct4.py`** — experimental.
Run with `PYTHONPATH=<repo root>` (they import `construct3/construct4/pb4/braids/gcatalog`).

## The problem
Turn a braid step-word `g` (1-cycle perm) into a minimal-W valid ±45° zigzag
diagram, as a *direct construction* (few build attempts), replacing the
search-heavy brute/place-and-bend hybrid in `construct4.construct_best`.

## Files (build on each other)
- **`direct2.py`** — core helpers: `build_at` (x-placement → y via `solve_components`
  with **conditional same-gen edges** → route ≤1 bend → `align_core`), `rots`,
  `solve_components`. The y-model fix lives here.
- **`cc2.py`** — **current best constructor**: commutation-reachable clasp-centering
  + bounded slack DFS. `construct_A(g,L)`. **381/392 (~97%)** of the L=3..5 library
  at minimal W, fast.
- **`c_construct.py`** — earlier "option C" (centering candidates ± slack, no
  commutation reachability). Recovered only 16/82 failures. Superseded by cc2.
- **`band.py`** — abandoned from-scratch "band layout" rewrite (did worse than cc2;
  don't pursue — patch cc2 instead).

## Key facts (see memory `project_g_direct_constructor`)
- **y is deterministic** given x: straight-segment equalities + same-gen "crisscross"
  edges (Δy=0 only when no adjacent σ_{i±1} intervenes). No y-search.
- **Placement rule** (all hand-verified): each generator's crossings form a clasp/band
  at its y-level (dx=2); commuting bands share x-columns, adjacent bands go sequential;
  a generator commuting with a clasp **centers** over it, lifted on the diagonal — and
  the **lift scales linearly with clasp length** (so it's O(1) computable, not searched).
  Commutation matters for placement even though greedy is commutation-invariant.
- **≤1 bend between consecutive crossings is COMPLETE** — 2+ bends would create a
  crossing-free (bare) run, which `valid_diagram` rejects. No multi-bend router needed.

## Known gap / next step
cc2 finds non-minimal W on very deep clasps (e.g. σ2⁷: W=16 vs true W=14) because it
*searches* for the roamer lift instead of *computing* it. Focused patch: (a) set the
roamer y-lift directly from clasp width; (b) make `align_core` reach the wide closure.
