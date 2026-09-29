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


# Tier 3 only matters for a rising/falling pair that OVERLAPS in height (lo <= hi):
# segments at disjoint heights cannot meet.  Imposing it on every pair, as the MILP does,
# forces q to absorb their height difference, so q needed a range ~ ymax/W.  Enforced
# only on overlap, q is local: 2(Y - y_A) = S + q*W with Y - y_A inside the rising
# segment (< 2W, since m <= 1), x_C - x_A in (-W, W) and 0 <= y_C - y_A < 4W, so
# |q| <= 5.  QB = 6 for margin.  That turns q*W into an "or" over 13 cases.
QB = 6

def _overlap(M, lo, hi):
    ov = M.NewBoolVar('')
    M.Add(lo <= hi).OnlyEnforceIf(ov); M.Add(lo > hi).OnlyEnforceIf(ov.Not())
    return ov


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
    for (e, f, cl, ch) in GS.tier3_pairs(n, segs, bkind):
        if pairs3 is not None and (e, f) not in pairs3: continue
        A, Bn, _ = segs[e]; C, Dn, _ = segs[f]
        lo = M.NewIntVar(-ymax, ymax, ''); hi = M.NewIntVar(-ymax, ymax, '')
        M.AddMaxEquality(lo, [Y[A], Y[Dn]])
        M.AddMinEquality(hi, [Y[Bn], Y[C]])
        ov = _overlap(M, lo, hi)
        q = M.NewIntVar(-QB, QB, '')
        D = X[C] + Y[C] - X[A] + Y[A]
        M.Add(D + W * q <= 2 * (lo + cl) - 2).OnlyEnforceIf(ov)
        M.Add(D + W * q + W >= 2 * (hi - ch) + 2).OnlyEnforceIf(ov)

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


def solve_min(g, L, Wmax=60, workers=1, time_limit=None, lazy=True, Wstart=None):
    """Smallest W with a verified diagram, scanning W upward like g_solve.solve_min.
    -> (solution, verdict): 'minimal' (every smaller even W proven infeasible),
    'achieved' (some smaller W was feasible but its drawing failed the post-check, or
    timed out), 'not-a-knot', or 'none'."""
    if GS.perm_cycles([abs(x) for x in g], L) != 1:
        return None, 'not-a-knot'
    W = max(2, GS.wlb(g), Wstart or 0); W += W % 2
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


# ---------------------------------------------------------------- W as a variable

class _Trace(cp_model.CpSolverSolutionCallback):
    """Records (seconds, W, best bound) at every improving solution."""
    def __init__(self):
        super().__init__(); self.points = []
    def on_solution_callback(self):
        self.points.append((self.WallTime(), int(self.ObjectiveValue()),
                            int(self.BestObjectiveBound())))


def feasible_varw(g, L, Wlo, Whi, workers=1, time_limit=None, pairs2=None, pairs3=None,
                  hint=None, stats=None):
    """The same model with W a VARIABLE in [Wlo, Whi] (even), minimised.  The MILP must
    fix W because of the products m*W and q*W; CP-SAT takes them directly: m*W as a
    product switched on by m, the helix test as a variable modulus, q*W as a product.
    -> (solution dict, proven_optimal) or (None, True) if infeasible for every W in
    range, or (None, False) on a time limit.  hint: a previous solution, to warm-start
    the next lazy round."""
    n = len(g)
    nn, nb, segs, parent, bkind = GS.segments(g, L)
    ns = len(segs)
    ymax = max(60, (Whi * L + 1) // 2)
    M = cp_model.CpModel()
    h = M.NewIntVar((Wlo + 1) // 2, Whi // 2, 'h')
    W = M.NewIntVar(Wlo, Whi, 'W'); M.Add(W == 2 * h)          # W even, as g_solve scans
    X = [M.NewIntVar(0, Whi - 1, f'x{i}') for i in range(nn)]
    Y = [M.NewIntVar(-ymax, ymax, f'y{i}') for i in range(nn)]
    Mw = [M.NewBoolVar(f'm{e}') for e in range(ns)]
    for i in range(nn):
        M.Add(X[i] <= W - 1)
        M.Add(2 * Y[i] <= L * W + 1); M.Add(-2 * Y[i] <= L * W + 1)   # |y| <= W*L/2
    for e, (a, b, s_) in enumerate(segs):
        p = M.NewIntVar(0, Whi, '')                                  # p = m_e * W
        M.Add(p == W).OnlyEnforceIf(Mw[e]); M.Add(p == 0).OnlyEnforceIf(Mw[e].Not())
        k = X[b] - X[a] + p
        M.Add(k >= 1)
        M.Add(Y[b] - Y[a] == s_ * k)
    M.Add(sum(Mw) == L)
    M.Add(X[0] == 0); M.Add(Y[0] == 0)

    def span(e):
        a, b, s_ = segs[e]
        return (a, b) if s_ == 1 else (b, a)

    hb = 2 * ymax + 2 * Whi
    for e in range(ns):
        for f in range(e + 1, ns):
            s_ = segs[e][2]
            if segs[f][2] != s_: continue
            if pairs2 is not None and (e, f) not in pairs2: continue
            le, he = span(e); lf, hf = span(f)
            bA = M.NewBoolVar(''); bB = M.NewBoolVar(''); bC = M.NewBoolVar('')
            M.Add(Y[he] <= Y[lf]).OnlyEnforceIf(bA)
            M.Add(Y[hf] <= Y[le]).OnlyEnforceIf(bB)
            ae, af = segs[e][0], segs[f][0]
            diff = M.NewIntVar(-hb, hb, '')
            M.Add(diff == (Y[ae] - s_ * X[ae]) - (Y[af] - s_ * X[af]))
            r = M.NewIntVar(-(Whi - 1), Whi - 1, '')                  # truncated modulo
            M.AddModuloEquality(r, diff, W)
            M.Add(r != 0).OnlyEnforceIf(bC)
            M.AddBoolOr([bA, bB, bC])

    for (e, f, cl, ch) in GS.tier3_pairs(n, segs, bkind):
        if pairs3 is not None and (e, f) not in pairs3: continue
        A, Bn, _ = segs[e]; C, Dn, _ = segs[f]
        lo = M.NewIntVar(-ymax, ymax, ''); hi = M.NewIntVar(-ymax, ymax, '')
        M.AddMaxEquality(lo, [Y[A], Y[Dn]])
        M.AddMinEquality(hi, [Y[Bn], Y[C]])
        ov = _overlap(M, lo, hi)
        q = M.NewIntVar(-QB, QB, '')
        wq = M.NewIntVar(-QB * Whi, QB * Whi, '')
        M.AddMultiplicationEquality(wq, [W, q])      # q has 2*QB+1 values: a small "or"
        D = X[C] + Y[C] - X[A] + Y[A]
        M.Add(D + wq <= 2 * (lo + cl) - 2).OnlyEnforceIf(ov)
        M.Add(D + wq + W >= 2 * (hi - ch) + 2).OnlyEnforceIf(ov)

    M.Minimize(W)
    if hint:
        M.AddHint(W, hint['W'])
        for i in range(n): M.AddHint(X[i], hint['X'][i]); M.AddHint(Y[i], hint['Y'][i])
        for j in range(nb): M.AddHint(X[n + j], hint['BX'][j]); M.AddHint(Y[n + j], hint['BY'][j])
        for e in range(ns): M.AddHint(Mw[e], hint['M'][e])
    S = cp_model.CpSolver()
    S.parameters.num_workers = workers
    S.parameters.linearization_level = 2
    if time_limit: S.parameters.max_time_in_seconds = time_limit
    tr = _Trace() if stats is not None else None
    st = S.Solve(M, tr) if tr else S.Solve(M)
    if stats is not None:          # optional diagnostics: see the trace in the notes
        stats.update(time=S.WallTime(), conflicts=S.NumConflicts(), branches=S.NumBranches(),
                     improvements=tr.points, status=S.StatusName(st),
                     vars=len(M.Proto().variables), constraints=len(M.Proto().constraints))
    if st == cp_model.INFEASIBLE: return None, True
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return None, False
    v = lambda var: int(S.Value(var))
    Wv = v(W)
    return dict(W=Wv, X=[v(X[i]) for i in range(n)], Y=[v(Y[i]) for i in range(n)],
                BX=[v(X[i]) for i in range(n, nn)], BY=[v(Y[i]) for i in range(n, nn)],
                M=[v(Mw[e]) for e in range(ns)], segs=segs, parent=parent, bkind=bkind,
                nb=nb, tier=3), st == cp_model.OPTIMAL


def solve_min_varw(g, L, Wmax=60, workers=1, time_limit=None):
    """Minimal W in ONE optimisation per lazy round instead of a scan over W.

    Each round minimises W over a relaxation (a subset of the pairs), so its optimum is
    a LOWER bound on the true minimum.  When a round's optimal drawing violates no pair
    and passes the post-check, it is feasible for the full problem at that bound, hence
    minimal.  If it is clean of pairs but fails the post-check, fall back to the per-W
    scan (solve_min) from that bound.  -> (solution, verdict) as solve_min."""
    if GS.perm_cycles([abs(x) for x in g], L) != 1:
        return None, 'not-a-knot'
    Wlo = max(2, GS.wlb(g)); Wlo += Wlo % 2
    p2, p3 = set(), set(); hint = None
    while True:
        r, opt = feasible_varw(g, L, Wlo, Wmax, workers, time_limit, p2, p3, hint)
        if r is None:
            return (None, 'none') if opt else solve_min(g, L, Wmax, workers, time_limit)
        if not opt:                            # timed out mid-optimisation: be safe
            return solve_min(g, L, Wmax, workers, time_limit, Wstart=Wlo)
        Wlo = r['W']                           # the relaxation's optimum: a lower bound
        a = set(GS.overlap_violations(r)) - p2
        c = set(GS.crossing_violations(r)) - p3
        if a or c:
            p2 |= a; p3 |= c; hint = r; continue
        if not GS.check(g, L, r) and GS.full_period(g, L, r):
            return r, 'minimal'
        return solve_min(g, L, Wmax, workers, time_limit, Wstart=Wlo)
