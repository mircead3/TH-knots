"""g -> minimal-W zigzag as a closed WALK on the lattice, solved by CP-SAT -- an experiment.

The knot at width W is one closed walk of N = W*L unit steps, each one column to the
right and one unit up or down; step t is at column t mod W, so every column is visited
exactly L times (the strands).  Unknowns: the heights y_0..y_{N-1}.  Constraints:

  half-column rule   the L steps through each half-column are at different heights
                     (their sums y_t + y_{t+1} differ).  This alone forbids overlapping
                     steps, crossings between columns, and any meeting at a column other
                     than a transversal CROSSING (both passes straight through) or a
                     TANGENCY (a peak on a valley).  It replaces g_solve's tiers 2 and 3.
  letters            each letter of g is placed at one column; at every (column x,
                     generator i) the crossings the walk makes -- rising pass meeting a
                     falling one, with i-1 visits below -- must equal the letters placed
                     there.  An unwanted crossing has no letter to account for it.
  order              letters of adjacent generators keep g's cyclic order (exactly one
                     wrap per adjacent-pair projection): the walk's word IS g on the
                     cylinder, letter by letter.
  no idle runs       (rule on, the default) every run between consecutive bights contains
                     a crossing.  Equivalent to g_solve's bight rules: 0 or 1 bight between
                     consecutive crossings.  Off: extra wiggles allowed.

No segments, no pairs, no tiers, no lazy loop.  The result is checked by g_solve's
run-based checks (strict_ok, full period).  Run with the experiment's venv:
    PYTHONPATH=. .venv/bin/python -c "import g_walk as G; print(G.solve_min([1,2],3))"
"""
from ortools.sat.python import cp_model

import g_solve as GS
import pb4, braids as b


def feasible(g, L, W, no_idle_runs=True, workers=1, time_limit=None):
    """A closed walk of width W drawing g, or None (infeasible) / 'unknown' (timeout).
    -> dict(W, y = heights of the walk, runs = zigzag run lengths)."""
    n = len(g); N = W * L
    Hm = N // 2 + 1
    M = cp_model.CpModel()
    y = [M.NewIntVar(-Hm, Hm, f'y{t}') for t in range(N)]
    up = [M.NewBoolVar(f'u{t}') for t in range(N)]          # step t goes up
    for t in range(N):
        nx = y[(t + 1) % N]
        M.Add(nx == y[t] + 1).OnlyEnforceIf(up[t])
        M.Add(nx == y[t] - 1).OnlyEnforceIf(up[t].Not())
    M.Add(y[0] == 0)

    # half-column rule: s_t = y_t + y_{t+1} = 2 y_t + 2 up_t - 1
    s = []
    for t in range(N):
        v = M.NewIntVar(-2 * Hm - 1, 2 * Hm + 1, '')
        M.Add(v == 2 * y[t] + 2 * up[t] - 1)
        s.append(v)
    for x in range(W):
        M.AddAllDifferent([s[t] for t in range(x, N, W)])

    # visits: in-slope up[t-1], out-slope up[t]
    straight = []; bight = []
    for t in range(N):
        a, c = up[(t - 1) % N], up[t]
        st = M.NewBoolVar('')                                 # a == c
        M.Add(a == c).OnlyEnforceIf(st); M.Add(a != c).OnlyEnforceIf(st.Not())
        straight.append(st); bight.append(st.Not())
    rising = []
    for t in range(N):                                       # straight and going up
        r = M.NewBoolVar('')
        M.AddBoolAnd([straight[t], up[t]]).OnlyEnforceIf(r)
        M.AddBoolOr([straight[t].Not(), up[t].Not()]).OnlyEnforceIf(r.Not())
        rising.append(r)

    # per column: meetings, crossings, generator (= 1 + visits below)
    cross = [None] * N          # visit t is part of a crossing
    Z = {}                      # (x, i) -> list of bools "rising visit of a sigma_i at x"
    for x in range(W):
        vis = list(range(x, N, W))
        eq = {}; lt = {}
        for a in vis:
            for c in vis:
                if a == c: continue
                if (c, a) in eq: eq[(a, c)] = eq[(c, a)]
                else:
                    e = M.NewBoolVar('')
                    M.Add(y[a] == y[c]).OnlyEnforceIf(e); M.Add(y[a] != y[c]).OnlyEnforceIf(e.Not())
                    eq[(a, c)] = e
                l_ = M.NewBoolVar('')
                M.Add(y[c] < y[a]).OnlyEnforceIf(l_); M.Add(y[c] >= y[a]).OnlyEnforceIf(l_.Not())
                lt[(c, a)] = l_                                   # c below a
        for a in vis:
            meets = M.NewBoolVar('')                                  # a shares its point
            others = [eq[(a, c)] for c in vis if c != a]
            M.AddBoolOr(others).OnlyEnforceIf(meets)
            for o in others: M.AddImplication(o, meets)
            cr = M.NewBoolVar('')                                     # a is in a crossing
            M.AddBoolAnd([meets, straight[a]]).OnlyEnforceIf(cr)
            M.AddBoolOr([meets.Not(), straight[a].Not()]).OnlyEnforceIf(cr.Not())
            cross[a] = cr
            below = sum(lt[(c, a)] for c in vis if c != a)
            for i in range(1, L):
                z = M.NewBoolVar('')                                  # rising visit of sigma_i
                M.AddBoolAnd([cr, rising[a]]).OnlyEnforceIf(z)
                M.Add(below == i - 1).OnlyEnforceIf(z)
                # converse: a crossing's rising visit with i-1 below IS z
                bi = M.NewBoolVar('')
                M.Add(below == i - 1).OnlyEnforceIf(bi); M.Add(below != i - 1).OnlyEnforceIf(bi.Not())
                M.AddBoolOr([cr.Not(), rising[a].Not(), bi.Not(), z])
                Z.setdefault((x, i), []).append(z)

    # letters: each at one column; crossings at (x, i) == letters placed at (x, i)
    P = [[M.NewBoolVar(f'p{t}_{x}') for x in range(W)] for t in range(n)]
    for t in range(n): M.AddExactlyOne(P[t])
    col = [sum(x * P[t][x] for x in range(W)) for t in range(n)]
    for x in range(W):
        for i in range(1, L):
            M.Add(sum(Z.get((x, i), [])) == sum(P[t][x] for t in range(n) if g[t] == i))
    # gauge: letter 0 at column 0, and the walk starts on its rising pass
    M.Add(P[0][0] == 1)
    M.Add(rising[0] == 1); M.Add(cross[0] == 1)
    # order: along each adjacent-pair projection (cyclic), columns increase, one wrap
    for a in range(1, L - 1):
        seq = [t for t in range(n) if g[t] in (a, a + 1)]
        wraps = []
        for k in range(len(seq)):
            t1, t2 = seq[k], seq[(k + 1) % len(seq)]
            w = M.NewBoolVar('')
            M.Add(col[t1] < col[t2]).OnlyEnforceIf(w.Not())
            M.Add(col[t1] > col[t2]).OnlyEnforceIf(w)
            wraps.append(w)
        M.Add(sum(wraps) == 1)
    if L == 2:                                  # one generator: its letters in order
        seq = list(range(n)); wraps = []
        for k in range(n):
            t1, t2 = seq[k], seq[(k + 1) % n]
            w = M.NewBoolVar('')
            M.Add(col[t1] < col[t2]).OnlyEnforceIf(w.Not()); M.Add(col[t1] > col[t2]).OnlyEnforceIf(w)
            wraps.append(w)
        M.Add(sum(wraps) == 1)

    # no idle runs: every run between consecutive bights contains a crossing.
    # seen[t] = "a crossing since the last bight, up to and including visit t".
    if no_idle_runs:
        seen = [M.NewBoolVar('') for _ in range(N)]
        for t in range(N):
            prev = seen[(t - 1) % N]
            # seen[t] <-> cross[t] or (prev and not bight[t])
            M.AddBoolOr([cross[t], prev, seen[t].Not()])
            M.AddBoolOr([cross[t], bight[t].Not(), seen[t].Not()])
            M.AddImplication(cross[t], seen[t])
            M.AddBoolOr([prev.Not(), bight[t], seen[t]])
            M.AddImplication(bight[t], prev)        # the run ending at this bight had one

    S = cp_model.CpSolver()
    S.parameters.num_workers = workers
    S.parameters.linearization_level = 2
    if time_limit: S.parameters.max_time_in_seconds = time_limit
    st = S.Solve(M)
    if st == cp_model.INFEASIBLE: return None
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return 'unknown'
    ys = [S.Value(v) for v in y]
    return dict(W=W, y=ys, runs=walk_runs(ys))


def walk_runs(ys):
    """Heights of a closed walk -> zigzag run lengths starting with an UP run (the form
    g_solve.to_runs returns and the app's setKnotFromRuns reads)."""
    N = len(ys)
    d = [1 if ys[(t + 1) % N] > ys[t] else -1 for t in range(N)]
    # start at a valley: step t0 goes up, step t0-1 went down
    t0 = next(t for t in range(N) if d[t] == 1 and d[t - 1] == -1)
    d = d[t0:] + d[:t0]
    runs = []; k = 1
    for t in range(1, N):
        if d[t] == d[t - 1]: k += 1
        else: runs.append(k); k = 1
    runs.append(k)
    return runs


def verified(g, L, runs):
    """g_solve's run-based checks: the drawing is g's knot at 3 coprime B, full period."""
    if not GS.strict_ok(g, L, runs): return False
    vr = pb4.verify(runs, L, b.smallest_coprime_b(L))
    return vr is not None and len(vr) == len(g)


def solve_min(g, L, Wmax=60, no_idle_runs=True, workers=1, time_limit=None):
    """Scan even W upward from g_solve's lower bound. -> (result, verdict) as g_solve:
    'minimal' when every smaller W was proven infeasible and this one verified."""
    if GS.perm_cycles([abs(x) for x in g], L) != 1:
        return None, 'not-a-knot'
    W = max(2, GS.wlb(g)); W += W % 2
    proven = True
    while W <= Wmax:
        r = feasible_rank(g, L, W, no_idle_runs, workers, time_limit, redundant=False)
        if isinstance(r, dict):
            if verified(g, L, r['runs']):
                return r, ('minimal' if proven else 'achieved')
            proven = False                        # a walk the checks reject: keep going
        elif r == 'unknown':
            proven = False
        W += 2
    return None, 'none'


# ---------------------------------------------------------------- rank formulation

def feasible_rank(g, L, W, no_idle_runs=True, workers=1, time_limit=None, redundant=False):
    """The same walk model with crossings read from RANKS instead of pairwise height tests.

    r_t in 0..L-1 is step t's rank among the L steps on its half-column (bottom = 0),
    tied to the heights by pairwise order (a permutation per half-column).  A visit at
    column x sits between its two steps t-1 and t, and its rank changes exactly when it
    crosses: r_t = r_{t-1} + 1 is the RISING pass of sigma_{r_t} (ranks r_t - 1 -> r_t,
    i.e. generator i = r_t), -1 the falling pass, 0 no crossing (plain pass or tangency).
    redundant: add constraints that follow from the rest but guide the search -- exactly
    |g| rising crossings, and (with no_idle_runs) exactly g_solve's number of bights."""
    n = len(g); N = W * L
    Hm = N // 2 + 1
    M = cp_model.CpModel()
    y = [M.NewIntVar(-Hm, Hm, f'y{t}') for t in range(N)]
    up = [M.NewBoolVar(f'u{t}') for t in range(N)]
    for t in range(N):
        nx = y[(t + 1) % N]
        M.Add(nx == y[t] + 1).OnlyEnforceIf(up[t])
        M.Add(nx == y[t] - 1).OnlyEnforceIf(up[t].Not())
    M.Add(y[0] == 0)
    s = []
    for t in range(N):
        v = M.NewIntVar(-2 * Hm - 1, 2 * Hm + 1, '')
        M.Add(v == 2 * y[t] + 2 * up[t] - 1)
        s.append(v)
    r = [M.NewIntVar(0, L - 1, f'r{t}') for t in range(N)]
    for x in range(W):
        steps = list(range(x, N, W))
        M.AddAllDifferent([s[t] for t in steps])
        M.AddAllDifferent([r[t] for t in steps])          # ranks: a permutation
        for i, a in enumerate(steps):
            for c in steps[i + 1:]:
                o = M.NewBoolVar('')                        # step a below step c
                M.Add(s[a] < s[c]).OnlyEnforceIf(o); M.Add(s[a] > s[c]).OnlyEnforceIf(o.Not())
                M.Add(r[a] < r[c]).OnlyEnforceIf(o); M.Add(r[a] > r[c]).OnlyEnforceIf(o.Not())

    # visit t (between steps t-1 and t): rank change dr in {-1, 0, 1}
    rise = {}; cross = []; bight = []
    for t in range(N):
        p = (t - 1) % N
        M.Add(r[t] - r[p] <= 1); M.Add(r[t] - r[p] >= -1)
        rs = M.NewBoolVar(''); fl = M.NewBoolVar('')
        M.Add(r[t] == r[p] + 1).OnlyEnforceIf(rs); M.Add(r[t] != r[p] + 1).OnlyEnforceIf(rs.Not())
        M.Add(r[t] == r[p] - 1).OnlyEnforceIf(fl); M.Add(r[t] != r[p] - 1).OnlyEnforceIf(fl.Not())
        # a rising crossing is straight up, a falling one straight down
        M.AddImplication(rs, up[p]); M.AddImplication(rs, up[t])
        M.AddImplication(fl, up[p].Not()); M.AddImplication(fl, up[t].Not())
        cr = M.NewBoolVar(''); M.AddBoolOr([rs, fl]).OnlyEnforceIf(cr)
        M.AddImplication(rs, cr); M.AddImplication(fl, cr)
        cross.append(cr)
        bt = M.NewBoolVar('')                                   # direction changes here
        M.Add(up[p] != up[t]).OnlyEnforceIf(bt); M.Add(up[p] == up[t]).OnlyEnforceIf(bt.Not())
        bight.append(bt)
        for i in range(1, L):                                   # rising pass of sigma_i
            z = M.NewBoolVar('')
            M.AddBoolAnd([rs]).OnlyEnforceIf(z); M.Add(r[t] == i).OnlyEnforceIf(z)
            ri = M.NewBoolVar('')
            M.Add(r[t] == i).OnlyEnforceIf(ri); M.Add(r[t] != i).OnlyEnforceIf(ri.Not())
            M.AddBoolOr([rs.Not(), ri.Not(), z])
            rise[(t, i)] = z

    P = [[M.NewBoolVar(f'p{t}_{x}') for x in range(W)] for t in range(n)]
    for t in range(n): M.AddExactlyOne(P[t])
    col = [sum(x * P[t][x] for x in range(W)) for t in range(n)]
    for x in range(W):
        for i in range(1, L):
            M.Add(sum(rise[(t, i)] for t in range(x, N, W))
                  == sum(P[k][x] for k in range(n) if g[k] == i))
    M.Add(P[0][0] == 1)
    M.Add(rise[(0, g[0])] == 1)                                 # walk starts on letter 0
    gens = list(range(1, L - 1)) if L > 2 else [None]
    for a in gens:
        seq = [k for k in range(n) if a is None or g[k] in (a, a + 1)]
        wraps = []
        for k in range(len(seq)):
            t1, t2 = seq[k], seq[(k + 1) % len(seq)]
            w = M.NewBoolVar('')
            M.Add(col[t1] < col[t2]).OnlyEnforceIf(w.Not()); M.Add(col[t1] > col[t2]).OnlyEnforceIf(w)
            wraps.append(w)
        M.Add(sum(wraps) == 1)

    if no_idle_runs:
        seen = [M.NewBoolVar('') for _ in range(N)]
        for t in range(N):
            prev = seen[(t - 1) % N]
            M.AddBoolOr([cross[t], prev, seen[t].Not()])
            M.AddBoolOr([cross[t], bight[t].Not(), seen[t].Not()])
            M.AddImplication(cross[t], seen[t])
            M.AddBoolOr([prev.Not(), bight[t], seen[t]])
            M.AddImplication(bight[t], prev)
    if redundant:
        M.Add(sum(rise.values()) == n)                          # exactly |g| crossings
        M.Add(sum(cross) == 2 * n)
        if no_idle_runs:
            M.Add(sum(bight) == GS.segments(g, L)[1])           # g_solve's bight count

    S = cp_model.CpSolver()
    S.parameters.num_workers = workers
    S.parameters.linearization_level = 2
    if time_limit: S.parameters.max_time_in_seconds = time_limit
    st = S.Solve(M)
    if st == cp_model.INFEASIBLE: return None
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return 'unknown'
    ys = [S.Value(v) for v in y]
    return dict(W=W, y=ys, runs=walk_runs(ys))
