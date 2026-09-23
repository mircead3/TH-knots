"""g -> zigzag by solving the diagram's linear system.  Exact, deterministic, minimal W.

THE FORMULATION (Mircea's, 2026-09-22)
--------------------------------------
A diagram is a graph drawn on a cylinder of circumference W.  Each generator of g is a
degree-4 node.  For  g = ... s_i h s_j ...  there is an edge between s_i and s_j iff
|i-j| <= 1 and h contains neither s_i nor s_j; it is straight UP if j=i+1, straight DOWN
if j=i-1, and a BEND if i=j.  For a bend, ... s_i h s_i ..., the bend is UP if h contains
neither s_i nor s_{i+1}, DOWN if h contains neither s_i nor s_{i-1}.

Each edge carries its own x-ADVANCE k >= 1 as a VARIABLE, and

    y_j = y_i + s*k          s = +1 (up) / -1 (down)
    x_j = x_i + k   (mod W)

The sign belongs to the edge, not the traversal: going from i to i+1 forward (up) is the
same edge as i+1 to i backward (down), so reversing flips k and s together.

Keeping every x in [0, W) turns the congruence into  k = x_j - x_i + m*W  with m >= 0 an
integer WINDING count.  That is the only nonlinearity (m*W), so W is fixed per iteration
and scanned upward from the band bound.  Bights are nodes too, so every edge is straight:
a bend A~>B becomes A -> bight -> B.

CUT: summing k over all segments gives the closed curve's total x-advance, PERIOD = W*L.
x telescopes to 0 around the loop (each crossing is a head twice and a tail twice, each
bight once each), so  sum(m) = L  exactly.

WHAT THE SOLVER GUARANTEES
--------------------------
Three tiers of necessary conditions, escalating only when a tier's solution fails the
post-check.  Ruling a W out is valid from any tier, since every constraint is necessary
for a valid diagram -- a weaker tier rules out fewer W's, never one wrongly.

  tier 1  the equations, k >= 1, x in [0,W), sum(m) = L, m <= 1

On top of the tiers, solve_min requires the diagram's braid word to have period |g| (see
full_period): "minimal W" means minimal among diagrams whose repeated tile really is g's,
since a shorter-period tile repeated B times renders a DIFFERENT knot (h^k at B = h at
kB).  Only knots whose minimal diagram is internally symmetric are affected -- one in the
L<=5, |g|<=10 range, where W goes 4 -> 6.
  tier 2  + NO SAME-SLOPE OVERLAP.  Same-slope segments share a helix iff y - s*x agrees
          mod W, and on a helix y determines the point, so they overlap iff their
          y-intervals overlap in more than a point.  This subsumes coincident steps, a
          bight landing mid-segment, coincident crossings and peak-peak/valley-valley
          bights, while ALLOWING a peak/valley tangency (y-intervals meeting at exactly
          one endpoint) -- which is legal: the curve touches itself from opposite sides
          without crossing, and the heuristic's own valid diagrams contain these.
  tier 3  + NO TRANSVERSAL CROSSING.  An opposite-slope pair meets at heights
          2Y = D + q*W, spaced W/2 apart.  Requiring one meeting below the y-overlap and
          the next above it excludes all of them, in two linear inequalities with one
          integer q -- no binaries beyond lo = max / hi = min.  A meeting is LEGAL only
          where it sits at a node of BOTH segments (a designated crossing, or a
          peak/valley tangency); since e rises A->Bn and f falls C->Dn, such a meeting
          can only be at lo or at hi, so exactly those two boundaries are relaxed when a
          compatible endpoint could occupy them.  No pair is ever skipped: a compatible
          pair of nodes *may* coincide, it need not, and skipping on that let a
          transversal crossing through.

Measured on the full glen<=10 library (509 braid-key classes): 509/509 solved, every
diagram post-verified AND independently confirmed by Gauss sequence, and W PROVEN MINIMAL
for all 509 -- in 28s total.  505 knots are answered at tier 1 and 4 at tier 2; tier 3 is
what rules out the last smaller W for the one knot that needs it
(g=[1,1,1,2,1,3,2,4,4,4], where the equations alone stay feasible at W=8 but no diagram
exists there).  17 knots come out STRICTLY BETTER than the previous rule-based hybrid
(best: W=20 -> 4); none worse.

RESIDUAL CAVEATS, stated rather than buried:
  * m <= 1 is an assumption.  Tested: no smaller W becomes feasible even at m <= 3.
  * |y| <= ymax is a window, not a proof.
  * The edge rules above are load-bearing: a missing edge under-constrains, a spurious
    one over-constrains and would make infeasibility claims wrong.  509 Gauss-verified
    diagrams support them in the realizability direction.
  * construct_brute (construct4.py) remains the independent oracle: exhaustive over run
    sequences, it separately proved no diagram exists at W<=8 for g=[2,1,3,2,1,1,1,4,4,4]
    and reproduced the W=10 zigzag, agreeing with tier 3.  Keep it for that purpose -- it
    is the only exhaustive check here.
  * to_runs may return a zigzag of SHORTER period than |g| when the diagram is more
    symmetric than the word (e.g. g=[1,2,1,3,2,4,3,4] -> runs [5,5,5,5], braid word
    [2,4,1,3]).  Cross-checks against pb4.braid_key must therefore compare at MATCHED
    periods; comparing a period-4 diagram's key with an 8-letter word's key fails even
    though the knots are identical.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import Counter, defaultdict
from itertools import product

import braids as b
import pb4
from enum_g import perm_cycles, canon_rot


# ---------------------------------------------------------------- structure

def edges(g, L):
    """Forward edges. -> [(t, t2, kind)] with kind in {up, down, bendup, benddown}.

    Each crossing sigma_i owns slots i-1 and i.  Along a slot, the forward edge runs to
    the first subsequent generator touching that slot (cyclically):
        slot i    touched by {i, i+1} -> sigma_{i+1}: UP     sigma_i: BEND UP
        slot i-1  touched by {i-1, i} -> sigma_{i-1}: DOWN   sigma_i: BEND DOWN
    That is 2 edges per crossing, 2n edges, 4n ends -- the complete edge set.
    """
    n = len(g); out = []
    for t in range(n):
        i = abs(g[t])
        for lo, up, dn in ((i, 'bendup', 'up'), (i - 1, 'benddown', 'down')):
            if lo == i:
                targets = {i} | ({i + 1} if i + 1 <= L - 1 else set())
            else:
                targets = {i} | ({i - 1} if i - 1 >= 1 else set())
            for step in range(1, n + 1):
                t2 = (t + step) % n
                if abs(g[t2]) in targets:
                    out.append((t, t2, up if abs(g[t2]) == i else dn))
                    break
    return out


def segments(g, L):
    """Straight segments over nodes = crossings 0..n-1 then bights n..n+nb-1.

    -> (nn, nb, segs, parent, bkind) with segs = [(a, b, slope)],
    parent[e] in {straight, bend}, bkind[j] = +1 PEAK / -1 VALLEY.
    """
    n = len(g); E = edges(g, L)
    segs = []; parent = []; bkind = []; bj = 0
    nb = sum(1 for e in E if e[2].startswith('bend'))
    for (t, t2, kind) in E:
        if kind in ('up', 'down'):
            segs.append((t, t2, 1 if kind == 'up' else -1)); parent.append('straight')
        else:
            s = 1 if kind == 'bendup' else -1
            u = n + bj; bj += 1; bkind.append(s)
            segs.append((t, u, s)); parent.append('bend')
            segs.append((u, t2, -s)); parent.append('bend')
    return n + nb, nb, segs, parent, bkind


def wlb(g):
    """Lower bound on W: twice the largest band (verified tight in most cases)."""
    return 2 * max(Counter(abs(v) for v in g).values())


# ---------------------------------------------------------------- the model

def feasible(g, L, W, tier=1, ymax=60, mmax=1, seed=None):
    """Feasibility of the tier-`tier` system at this fixed W.  -> solution dict or None."""
    n = len(g)
    nn, nb, segs, parent, bkind = segments(g, L)
    ns = len(segs)

    NX, NY, NM = 0, nn, 2 * nn
    nv = 2 * nn + ns
    rows_eq = []; b_eq = []; rows_ub = []; b_ub = []

    def yhi_ylo(e):
        """(node at the LOW y end, node at the HIGH y end) of segment e."""
        a, bb, s = segs[e]
        return (a, bb) if s == 1 else (bb, a)

    # --- equations
    for e, (a, bn, s) in enumerate(segs):
        r = np.zeros(nv)                       # y_b - y_a = s*(x_b - x_a + m*W)
        r[NY + bn] += 1; r[NY + a] -= 1
        r[NX + bn] -= s; r[NX + a] += s
        r[NM + e] -= s * W
        rows_eq.append(r); b_eq.append(0.0)
        r = np.zeros(nv)                       # k >= 1
        r[NX + bn] -= 1; r[NX + a] += 1; r[NM + e] -= W
        rows_ub.append(r); b_ub.append(-1.0)
    r = np.zeros(nv); r[NM:NM + ns] = 1.0      # sum(m) = L
    rows_eq.append(r); b_eq.append(float(L))
    r = np.zeros(nv); r[NX] = 1.0; rows_eq.append(r); b_eq.append(0.0)   # gauge
    r = np.zeros(nv); r[NY] = 1.0; rows_eq.append(r); b_eq.append(0.0)

    extra_lo = []; extra_hi = []; extra_int = []

    def grow(k):
        """Append k new variables, padding the rows built so far."""
        nonlocal nv, rows_eq, rows_ub
        base = nv; nv += k
        rows_eq = [np.concatenate([r, np.zeros(nv - len(r))]) for r in rows_eq]
        rows_ub = [np.concatenate([r, np.zeros(nv - len(r))]) for r in rows_ub]
        return base

    # --- tier 2: no same-slope overlap
    if tier >= 2:
        same = [(e, f) for e in range(ns) for f in range(e + 1, ns)
                if segs[e][2] == segs[f][2]]
        NQ = grow(len(same) + 3 * len(same))
        NB = NQ + len(same)
        Q = (2 * ymax + 2 * W) // W + 2
        M = 4 * (ymax + W) + Q * W + 10
        for pi, (e, f) in enumerate(same):
            sl = segs[e][2]
            loe, hie = yhi_ylo(e); lof, hif = yhi_ylo(f)
            bA, bB, bC = NB + 3 * pi, NB + 3 * pi + 1, NB + 3 * pi + 2
            for (hh, ll, bb) in ((hie, lof, bA), (hif, loe, bB)):
                r = np.zeros(nv)               # y_hh <= y_ll   when bb = 1
                r[NY + hh] += 1; r[NY + ll] -= 1; r[bb] += M
                rows_ub.append(r); b_ub.append(float(M))
            ae, af = segs[e][0], segs[f][0]    # (C) different helix: h = y_a - s*x_a
            for sign, off in ((-1, M - 1), (1, M + W - 1)):
                r = np.zeros(nv)
                r[NY + ae] += sign; r[NX + ae] -= sign * sl
                r[NY + af] -= sign; r[NX + af] += sign * sl
                r[NQ + pi] += sign * W * -1
                r[bC] += M
                rows_ub.append(r); b_ub.append(float(off))
            r = np.zeros(nv); r[bA] = r[bB] = r[bC] = -1.0
            rows_ub.append(r); b_ub.append(-1.0)
        extra_int.append((NQ, NB, -Q, Q)); extra_int.append((NB, nv, 0, 1))

    # --- tier 3: no transversal crossing
    if tier >= 3:
        # A meeting is LEGAL iff it sits at a node of BOTH segments -- either the same
        # node (a designated crossing) or a peak/valley pair (a tangency).  Both are
        # decidable statically, so the exclusion interval is closed or half-open per
        # pair.  e rises A->Bn, f falls C->Dn, so y_A<y_Bn and y_Dn<y_C, hence
        def compat(u, v):
            if u == v: return True
            if u < n or v < n: return False            # a crossing coincides with nothing
            return bkind[u - n] != bkind[v - n]        # peak + valley = tangency
        # NEVER skip a pair.  compat() says two nodes *could* coincide, not that they do
        # -- skipping on it dropped a pair whose peak and valley ends sat 3 apart, which
        # is how a transversal crossing slipped through.  Instead: e rises A->Bn and f
        # falls C->Dn, so y_A <= lo and y_Bn >= hi; a legal meeting can therefore only
        # sit at lo (if y_A is a compatible endpoint) or at hi (if y_Bn is).  Relax just
        # those two boundaries.
        opp = []
        for e in range(ns):
            if segs[e][2] != 1: continue
            for f in range(ns):
                if segs[f][2] != -1: continue
                A, Bn, _ = segs[e]; C, Dn, _ = segs[f]
                cl = 1 if (compat(A, C) or compat(A, Dn)) else 0
                ch = 1 if (compat(Bn, C) or compat(Bn, Dn)) else 0
                opp.append((e, f, cl, ch))
        NT = grow(4 * len(opp))                # per pair: q, lo, hi, 2 binaries -> 5
        NT2 = grow(2 * len(opp))
        M = 6 * (ymax + W) + 20
        for pi, (e, f, cl, ch) in enumerate(opp):
            qv, lov, hiv = NT + 3 * pi, NT + 3 * pi + 1, NT + 3 * pi + 2
            zlo, zhi = NT2 + 2 * pi, NT2 + 2 * pi + 1
            A, Bn, _ = segs[e]                 # e rises from A to Bn
            C, Dn, _ = segs[f]                 # f falls from C to Dn
            # lo = max(y_A, y_Dn)
            for u, z in ((A, zlo), (Dn, None)):
                r = np.zeros(nv); r[NY + u] += 1; r[lov] -= 1
                rows_ub.append(r); b_ub.append(0.0)          # lo >= y_u
            r = np.zeros(nv); r[lov] += 1; r[NY + A] -= 1; r[zlo] += M
            rows_ub.append(r); b_ub.append(float(M))         # lo <= y_A  if zlo=1
            r = np.zeros(nv); r[lov] += 1; r[NY + Dn] -= 1; r[zlo] -= M
            rows_ub.append(r); b_ub.append(0.0)              # lo <= y_Dn if zlo=0
            # hi = min(y_Bn, y_C)
            for u in (Bn, C):
                r = np.zeros(nv); r[hiv] += 1; r[NY + u] -= 1
                rows_ub.append(r); b_ub.append(0.0)          # hi <= y_u
            r = np.zeros(nv); r[NY + Bn] += 1; r[hiv] -= 1; r[zhi] -= M
            rows_ub.append(r); b_ub.append(0.0)              # hi >= y_Bn if zhi=0
            r = np.zeros(nv); r[NY + C] += 1; r[hiv] -= 1; r[zhi] += M
            rows_ub.append(r); b_ub.append(float(M))         # hi >= y_C   if zhi=1
            # D = x_C + y_C - x_A + y_A ;  meetings at 2Y = D + q*W
            def Drow(r, sign):
                r[NX + C] += sign; r[NY + C] += sign
                r[NX + A] -= sign; r[NY + A] += sign
            # Exclude meetings strictly INTERIOR to both segments.  Endpoint meetings
            # are the legitimate ones: a designated crossing, or a peak/valley TANGENCY
            # (where the two bights are DIFFERENT nodes, so a "shares a node" exemption
            # would miss it and the constraint would reject valid diagrams).
            r = np.zeros(nv)                   # D + q*W <= 2*(lo + cl) - 2
            Drow(r, 1); r[qv] += W; r[lov] -= 2
            rows_ub.append(r); b_ub.append(2.0 * cl - 2.0)
            r = np.zeros(nv)                   # D + (q+1)*W >= 2*(hi - ch) + 2
            Drow(r, -1); r[qv] -= W; r[hiv] += 2
            rows_ub.append(r); b_ub.append(float(W) + 2.0 * ch - 2.0)
        extra_int.append((NT, NT2, -(4 * (ymax + W)) // max(W, 1) - 4,
                          (4 * (ymax + W)) // max(W, 1) + 4))
        extra_int.append((NT2, nv, 0, 1))

    lo = np.empty(nv); hi = np.empty(nv)
    lo[NX:NX + nn] = 0; hi[NX:NX + nn] = W - 1
    lo[NY:NY + nn] = -ymax; hi[NY:NY + nn] = ymax
    lo[NM:NM + ns] = 0; hi[NM:NM + ns] = mmax
    for (a, z, l, h) in extra_int:
        lo[a:z] = l; hi[a:z] = h
    if tier >= 3:                              # lo/hi vars are heights, widen them
        for pi in range(len(opp)):
            for off in (1, 2):
                lo[NT + 3 * pi + off] = -ymax - 2; hi[NT + 3 * pi + off] = ymax + 2

    Aeq = np.array(rows_eq); Aub = np.array(rows_ub)
    cons = [LinearConstraint(Aeq, b_eq, b_eq),
            LinearConstraint(Aub, -np.inf, b_ub)]
    c = np.zeros(nv)
    if seed is not None:        # DIAGNOSTIC ONLY: walk to a different vertex.
        c = np.random.default_rng(seed).integers(-9, 10, size=nv).astype(float)
    res = milp(c=c, constraints=cons, bounds=Bounds(lo, hi),
               integrality=np.ones(nv))
    if not res.success:
        return None
    v = np.round(res.x).astype(int)
    return dict(W=W, X=list(v[NX:NX + n]), Y=list(v[NY:NY + n]),
                BX=list(v[NX + n:NX + nn]), BY=list(v[NY + n:NY + nn]),
                M=list(v[NM:NM + ns]), segs=segs, parent=parent, bkind=bkind, nb=nb,
                tier=tier)


def full_period(g, L, sol):
    """True iff the diagram's braid word has period |g|, i.e. the tile that gets
    repeated B times really is g's tile and not a shorter block repeated.

    This matters because a diagram may be MORE symmetric than the word: for
    g=[1,2,1,3,2,4,3,4] the minimal diagram is W=4 with runs [5,5,5,5], whose word is
    [2,4,1,3] -- period 4.  Repeated B times that renders the |g|=4 knot at 2B, not the
    |g|=8 knot at B (h^k at B = h at kB).  Requiring full period gives W=6, [7,7,8,8].
    NOTE: this is the one place the solve path needs pb4.
    """
    runs = to_runs(g, L, sol)
    if not runs: return False
    vr = pb4.verify(runs, L, b.smallest_coprime_b(L))
    return vr is not None and len(vr) == len(g)


def solve_min(g, L, Wmax=60, max_tier=3, require_full_period=True):
    """Smallest W with a VERIFIED diagram.  Deterministic -- no sampling.

    -> (solution, verdict) with verdict:
       'minimal'  every smaller even W proven infeasible AND this W exhibited
       'achieved' some smaller W stayed feasible with no diagram found (upper bound only)

    max_tier caps the escalation, for experiments only -- DO NOT lower it in callers.
    Tier 3 does more than certify minimality: for some rotations of g, tiers 1-2 return
    only unfaithful solutions at the minimal W and tier 3 is what finds the diagram
    (g=[2,1,3,2,1,1,1,4,4,4] gives W=14 at max_tier=2 versus 10 at 3).  The returned W is
    a property of the knot, so it must not depend on the rotation -- capping breaks that.
    """
    W = max(2, wlb(g)); W += W % 2
    proven = True
    while W <= Wmax:
        got = None
        for tier in range(1, max_tier + 1):
            r = feasible(g, L, W, tier=tier)
            if r is None:
                got = 'ruled-out'; break       # this W is impossible, at any tier
            if not check(g, L, r) and not (require_full_period
                                           and not full_period(g, L, r)):
                got = r; break
        if got is None:
            proven = False                     # feasible here, no diagram found
        elif got != 'ruled-out':
            return got, ('minimal' if proven else 'achieved')
        W += 2
    return None, 'none'


# ---------------------------------------------------------------- verification

def render(g, L, sol):
    """-> (steps, crossings, bights, W); steps = [(p, q, slope, seg)] on the cylinder."""
    W = sol['W']; n = len(g)
    P = [(int(sol['X'][t]), int(sol['Y'][t])) for t in range(n)]
    B = [(int(sol['BX'][j]), int(sol['BY'][j])) for j in range(len(sol['BX']))]
    node = P + B; steps = []
    for e, (a, bb, s) in enumerate(sol['segs']):
        xa, ya = node[a]; xb, _ = node[bb]
        k = xb - xa + int(sol['M'][e]) * W
        for j in range(k):
            steps.append((((xa + j) % W, ya + s * j),
                          ((xa + j + 1) % W, ya + s * (j + 1)), s, e))
    return steps, P, B, W


def trace(g, L, sol):
    """Walk the closed curve segment by segment: straight through a crossing, reversing
    at a bight.  (A step-based walk takes the wrong branch at a tangency.)"""
    n = len(g); segs = sol['segs']
    outs = defaultdict(list)
    for e, (a, bb, s) in enumerate(segs): outs[a].append(e)
    order = []; e = 0; seen = set()
    while e not in seen:
        seen.add(e); order.append(e)
        _a, bb, s = segs[e]
        want = s if bb < n else -s
        nxt = [f for f in outs[bb] if segs[f][2] == want]
        if not nxt: return None
        e = nxt[0]
    return order


def tangencies(g, L, sol, W):
    """Points carrying exactly one peak and one valley bight -- a legal self-touch."""
    bk = sol['bkind']; at = defaultdict(list)
    for j in range(len(bk)):
        at[(int(sol['BX'][j]) % W, int(sol['BY'][j]))].append(bk[j])
    return {pt for pt, o in at.items() if len(o) == 2 and set(o) == {1, -1}}


def check(g, L, sol):
    """[] iff the only self-intersections are designated crossings and tangencies."""
    steps, P, B, W = render(g, L, sol)
    errs = []
    if len(steps) != W * L:
        errs.append(f'length {len(steps)} != W*L = {W * L}')
    deg = defaultdict(int); onstep = Counter()
    for (p, q, s, e) in steps:
        deg[p] += 1; deg[q] += 1; onstep[(p, q)] += 1
    if any(v > 1 for v in onstep.values()):
        errs.append(f'{sum(1 for v in onstep.values() if v > 1)} coincident steps')
    ok4 = {(x % W, y) for (x, y) in P} | tangencies(g, L, sol, W)
    bad4 = [pt for pt, d in deg.items() if d == 4 and pt not in ok4]
    if bad4:
        errs.append(f'{len(bad4)} unintended crossings e.g. {sorted(bad4)[:3]}')
    if any(d not in (2, 4) for d in deg.values()):
        errs.append('points of bad degree')
    miss = [pt for pt in {(x % W, y) for (x, y) in P} if deg.get(pt, 0) != 4]
    if miss:
        errs.append(f'{len(miss)} designated crossings not degree 4')
    ups = {(p[0], p[1]) for (p, q, s, e) in steps if s == 1}
    dns = {(p[0], p[1]) for (p, q, s, e) in steps if s == -1}
    if any((c, y + 1) in dns for (c, y) in ups):
        errs.append('mid-cell crossing')
    order = trace(g, L, sol)
    if order is None:
        errs.append('curve dead-ends')
    elif len(order) != len(sol['segs']):
        errs.append(f'loop covers {len(order)}/{len(sol["segs"])} segments')
    return errs


def gauss_ok(g, L, sol):
    """Independent knot-identity check: the crossing sequence along the drawn curve must
    match the one the closed braid word predicts."""
    n = len(g); slot = 0; want = []
    for _ in range(L):
        for t in range(n):
            i = abs(g[t])
            if slot == i - 1: want.append(t); slot = i
            elif slot == i:   want.append(t); slot = i - 1
    order = trace(g, L, sol)
    if order is None or len(order) != len(sol['segs']): return False
    got = [sol['segs'][e][1] for e in order if sol['segs'][e][1] < n]
    if len(got) != len(want): return False
    d = want + want
    return (any(d[i:i + len(got)] == got for i in range(len(want)))
            or any(d[i:i + len(got)] == got[::-1] for i in range(len(want))))


# ---------------------------------------------------------------- output / library

def strict_ok(g, L, runs, nB=3):
    """THE knot-identity check: the DIAGRAM, closed at B bights, must be g's knot at the
    same B -- tested at the first nB values of B coprime to L.

    Nothing is repeated or padded to force agreement.  This is stricter than gauss_ok,
    which compares crossing-visit order within the tile and therefore PASSES a diagram
    whose tile is a repeated shorter block: such a tile closed B times gives the shorter
    word closed at a multiple of B, i.e. a different knot (h^k at B = h at kB).  That is
    exactly how a wrong-knot diagram (g=[1,2,1,3,2,4,3,4], W=4, runs [5,5,5,5]) once got
    through.  B must be COPRIME to L or the closure is a link, not a knot.
    """
    from math import gcd
    Bs = [x for x in range(2, 14) if gcd(x, L) == 1][:nB]
    tgt_g = [abs(z) for z in g]
    for B in Bs:
        vr = pb4.verify(runs, L, B)
        if vr is None: return False
        if pb4.braid_key(list(vr), L, B) != pb4.braid_key(tgt_g, L, B): return False
    return True


def to_runs(g, L, sol):
    """Zigzag run lengths -- the form the app consumes.  Runs are delimited by bights,
    so a run is a maximal same-slope stretch of the traversal."""
    n = len(g); order = trace(g, L, sol)
    if order is None: return None
    W = sol['W']; node = ([(int(sol['X'][t]), int(sol['Y'][t])) for t in range(n)]
                          + [(int(sol['BX'][j]), int(sol['BY'][j]))
                             for j in range(len(sol['BX']))])
    klen = []
    for e in order:
        a, bb, s = sol['segs'][e]
        klen.append((node[bb][0] - node[a][0] + int(sol['M'][e]) * W, s))
    runs = []; cur = 0; slope = klen[0][1]
    for k, s in klen:
        if s != slope: runs.append(cur); cur = 0; slope = s
        cur += k
    runs.append(cur)
    # the traversal may start mid-run; fold the last stretch into the first if same slope
    if len(runs) > 1 and klen[0][1] == klen[-1][1]:
        runs[0] += runs.pop()
    return runs


def library(glen=10, Ls=(3, 4, 5)):
    """Distinct knots as braid-key classes, one representative word each."""
    def is_power(w):
        m = len(w)
        return any(m % d == 0 and w == w[:d] * (m // d) for d in range(1, m))
    classes = {}
    for L in Ls:
        Bc = b.smallest_coprime_b(L); seen = set()
        for ell in range(L - 1, glen + 1):
            for w in product(range(1, L), repeat=ell):
                cw = canon_rot(list(w))
                if cw in seen: continue
                seen.add(cw); gg = list(cw)
                if perm_cycles(gg, L) != 1 or is_power(gg): continue
                k = pb4.braid_key(gg, L, Bc)
                if k is None: continue
                cur = classes.get((L, k))
                if cur is None or (len(gg), gg) < (len(cur), cur):
                    classes[(L, k)] = gg
    return [(L, classes[(L, k)]) for (L, k) in classes]


def solve(g, L, max_tier=3, require_full_period=True):
    """Convenience: -> dict(W, runs, verdict) or None."""
    sol, verdict = solve_min(g, L, max_tier=max_tier,
                             require_full_period=require_full_period)
    if sol is None: return None
    return dict(W=sol['W'], runs=to_runs(g, L, sol), verdict=verdict,
                tier=sol['tier'], sol=sol)
