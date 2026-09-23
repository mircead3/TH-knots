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

## `unified.py` — the full rule-based pipeline (2026-09-19)

**`unified.construct(g, L)`** implements the complete placement theory derived from
hand-solved examples. Candidate set = **distinct greedy layouts over the
rotation+commutation class** (a class of thousands of words collapses to 18–48
distinct layouts), times a small per-band x-shift and roamer repositioning.
y is deterministic (`solve_components`); **W is emergent** (align tries W upward,
narrow endpoint window first then wide); no slack DFS, no Wtarget.

Why the commutation class matters: rotations alone **cannot** produce a layout where
a band is *split around* a roamer (e.g. "separate σ4³ to encompass σ3"), which several
knots require. `layouts()` generates it as bounded alternating rounds
(rotate → commutation-close, ×3); 2 rounds is the minimum that captures band splits.

**Result: solves 11/11 of the knots cc2 cannot do**, three matching hand solutions
exactly — including the σ2⁷ deep clasp at its true minimal **W=14** (cc2 only reached 16).

### Recommended use: HYBRID
`unified` is ~100s/knot and at a tight budget fails easy knots cc2 does instantly, so
it should **not** replace cc2. Use **cc2 as the fast path (381/392) and `unified` as a
fallback on its failures (11/11) → 392/392**.

### Perf note before any port
`pb4.trace` + `pb4.verify` run for *every* candidate endpoint combo (~1–2 ms each), and
one build can do 12 W-values × ~1500 combos. Cache/short-circuit `verify` before this
becomes production code. (`layouts()` is NOT the bottleneck — 0.1–0.3 s.)

## Known gap / next step
cc2 finds non-minimal W on very deep clasps (e.g. σ2⁷: W=16 vs true W=14) because it
*searches* for the roamer lift instead of *computing* it. Focused patch: (a) set the
roamer y-lift directly from clasp width; (b) make `align_core` reach the wide closure.
