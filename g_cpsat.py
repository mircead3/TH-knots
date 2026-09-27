"""g -> minimal-W zigzag with CP-SAT (OR-Tools) -- an EXPERIMENT beside g_solve.

Same formulation as g_solve.feasible, one-to-one, so the two can be compared knot by
knot: nodes (crossings, then bights) carry x in [0, W) and y; each segment e = (a, b, s)
has a wrap count m_e in {0, 1} and advance k_e = x_b - x_a + m_e*W >= 1, with
y_b - y_a = s*k_e and sum(m) = L.  What differs is how the either/or conditions are
stated.  g_solve's MILP needs big-M rows and 0/1 switches, which its continuous
relaxation prunes badly; CP-SAT takes them natively and learns from conflicts:

  tier 2  same-slope e, f: e below f, OR f below e, OR different helix
          ((h_e - h_f) mod W != 0, h = y_a - s*x_a) -- three enforcement literals and a
          native modulo, at least one literal true.
  tier 3  rising e, falling f: meetings at 2Y = D + q*W must all avoid the interior of
          the y-overlap [lo, hi] -- lo/hi by native max/min, q an integer, exactly the
          inequalities g_solve uses (with the same cl/ch relaxations, tier3_pairs).

All tiers go in at once (no escalation, no lazy loop): the point of the experiment is to
see whether CP-SAT copes with the full model directly.  Accepted solutions must still
pass g_solve's post-check and full-period rule; the model only proposes.

Run with the experiment's venv (OR-Tools is not in the system Python):
    PYTHONPATH=. .venv/bin/python -c "import g_cpsat as C; print(C.solve_min([1,2,3,4],5))"
"""
from ortools.sat.python import cp_model

import g_solve as GS


def feasible(g, L, W, workers=1, time_limit=None, pairs2=None, pairs3=None):
    """Tiers 1-3 at this W in one CP-SAT model.  -> solution dict (g_solve.feasible's
    shape), None if proven infeasible, or 'unknown' if the time limit hit first.
    pairs2 / pairs3: restrict tier 2's (e, f) same-slope pairs / tier 3's (e rising,
    f falling) pairs to these sets; None = all.  Relaxations, for the lazy loop."""
    n = len(g)
    nn, nb, segs, parent, bkind = GS.segments(g, L)
    ns = len(segs)
    ymax = max(60, (W * L + 1) // 2)          # never excludes a diagram (see g_solve)
    M = cp_model.CpModel()
    X = [M.NewIntVar(0, W - 1, f'x{i}') for i in range(nn)]
    Y = [M.NewIntVar(-ymax, ymax, f'y{i}') for i in range(nn)]
    Mw = [M.NewIntVar(0, 1, f'm{e}') for e in range(ns)]

    # --- tier 1: the equations
    for e, (a, b, s) in enumerate(segs):
        k = X[b] - X[a] + W * Mw[e]
        M.Add(k >= 1)
        M.Add(Y[b] - Y[a] == s * k)
    M.Add(sum(Mw) == L)
    M.Add(X[0] == 0); M.Add(Y[0] == 0)        # gauge

    def span(e):
        """(node at the LOW end, node at the HIGH end) of segment e."""
        a, b, s = segs[e]
        return (a, b) if s == 1 else (b, a)

    # --- tier 2: no same-slope overlap
    hb = 2 * ymax + 2 * W                      # bound on |h_e - h_f|
    for e in range(ns):
        for f in range(e + 1, ns):
            s = segs[e][2]
            if segs[f][2] != s: continue
            if pairs2 is not None and (e, f) not in pairs2: continue
            le, he = span(e); lf, hf = span(f)
            bA = M.NewBoolVar(''); bB = M.NewBoolVar(''); bC = M.NewBoolVar('')
            M.Add(Y[he] <= Y[lf]).OnlyEnforceIf(bA)
            M.Add(Y[hf] <= Y[le]).OnlyEnforceIf(bB)
            ae, af = segs[e][0], segs[f][0]
            diff = M.NewIntVar(-hb, hb, '')
            M.Add(diff == (Y[ae] - s * X[ae]) - (Y[af] - s * X[af]))
            # CP-SAT's modulo is TRUNCATED (sign of the dividend: -5 mod 8 = -5), so r
            # must range over (-W, W); declaring it [0, W) forced diff >= 0 for every
            # pair and made every non-trivial W infeasible.  Only r == 0 matters, and
            # that is the same under either convention.
            r = M.NewIntVar(-(W - 1), W - 1, '')
            M.AddModuloEquality(r, diff, W)
            M.Add(r != 0).OnlyEnforceIf(bC)
            M.AddBoolOr([bA, bB, bC])

    # --- tier 3: no transversal crossing
    qb = (4 * (ymax + W)) // W + 4
    for (e, f, cl, ch) in GS.tier3_pairs(n, segs, bkind):
        if pairs3 is not None and (e, f) not in pairs3: continue
        A, Bn, _ = segs[e]; C, Dn, _ = segs[f]
        lo = M.NewIntVar(-ymax, ymax, ''); hi = M.NewIntVar(-ymax, ymax, '')
        M.AddMaxEquality(lo, [Y[A], Y[Dn]])
        M.AddMinEquality(hi, [Y[Bn], Y[C]])
        q = M.NewIntVar(-qb, qb, '')
        D = X[C] + Y[C] - X[A] + Y[A]
        M.Add(D + W * q <= 2 * (lo + cl) - 2)
        M.Add(D + W * q + W >= 2 * (hi - ch) + 2)

    S = cp_model.CpSolver()
    S.parameters.num_workers = workers
    # Use CP-SAT's LP relaxation fully.  At the default level it barely does, and then
    # plain tier-1 equations are refuted by search instead of by the relaxation: 12-14s
    # on some L=8-10 words (W=6 infeasible, or a round with a few pairs), 0.1-0.9s here.
    S.parameters.linearization_level = 2
    if time_limit: S.parameters.max_time_in_seconds = time_limit
    st = S.Solve(M)
    if st == cp_model.INFEASIBLE: return None
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return 'unknown'
    v = lambda var: int(S.Value(var))
    return dict(W=W, X=[v(X[i]) for i in range(n)], Y=[v(Y[i]) for i in range(n)],
                BX=[v(X[i]) for i in range(n, nn)], BY=[v(Y[i]) for i in range(n, nn)],
                M=[v(Mw[e]) for e in range(ns)], segs=segs, parent=parent, bkind=bkind,
                nb=nb, tier=3)


def feasible_lazy(g, L, W, workers=1, time_limit=None):
    """Tier 1, then add only the tier-2/3 pairs the solution violates, re-solve, until
    none are violated (g_solve's lazy tiers, with CP-SAT solving each round).
    -> solution dict, None (infeasible: sound, a subset of pairs is a relaxation), or
    'unknown'."""
    p2, p3 = set(), set()
    while True:
        r = feasible(g, L, W, workers, time_limit, pairs2=p2, pairs3=p3)
        if not isinstance(r, dict): return r
        a = set(GS.overlap_violations(r)) - p2
        c = set(GS.crossing_violations(r)) - p3
        if not a and not c: return r
        p2 |= a; p3 |= c


def solve_min(g, L, Wmax=60, workers=1, time_limit=None, lazy=True):
    """Smallest W with a verified diagram, scanning W upward like g_solve.solve_min.
    -> (solution, verdict): 'minimal' (every smaller even W proven infeasible),
    'achieved' (some smaller W was feasible but its drawing failed the post-check, or
    timed out), 'not-a-knot', or 'none'."""
    if GS.perm_cycles([abs(x) for x in g], L) != 1:
        return None, 'not-a-knot'
    W = max(2, GS.wlb(g)); W += W % 2
    proven = True
    ok = lambda r: not GS.check(g, L, r) and GS.full_period(g, L, r)
    while W <= Wmax:
        r = feasible_lazy(g, L, W, workers, time_limit) if lazy else None
        if isinstance(r, dict) and ok(r):
            return r, ('minimal' if proven else 'achieved')
        if lazy and r is None:                 # a relaxation is infeasible: W ruled out
            W += 2; continue
        # the full model: not lazy, or the lazy drawing failed the post-check
        r = feasible(g, L, W, workers=workers, time_limit=time_limit)
        if isinstance(r, dict):
            if ok(r):
                return r, ('minimal' if proven else 'achieved')
            proven = False                     # a tier-clean drawing the post-check rejects
        elif r == 'unknown':
            proven = False
        W += 2
    return None, 'none'
