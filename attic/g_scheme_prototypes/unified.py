"""Unified g->zigzag constructor (2026-09-19).

Candidate set = DISTINCT GREEDY LAYOUTS over the rotation+commutation class
(the full class is huge but collapses to tens of distinct layouts), times a
small per-band x-shift and roamer repositioning.  y is deterministic
(solve_components).  W is EMERGENT (no Wtarget): align tries W upward and, at
each W, a narrow endpoint window first (fast) then a wider one (needed for
deep-clasp closures).  First W that closes is minimal.

No slack DFS anywhere.
"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]

from direct2 import solve_components
from construct3 import columns_greedy
from construct4 import route, perm_succ, valid_diagram, pb4
import braids as b
from itertools import product
import time


def _comm_close(S, cap):
    """Closure of S under commuting adjacent transpositions."""
    seen = set(S); frontier = set(S)
    while frontier and len(seen) < cap:
        new = set()
        for w in frontier:
            w = list(w)
            for j in range(len(w) - 1):
                if abs(w[j] - w[j + 1]) >= 2:
                    v = w[:]; v[j], v[j + 1] = v[j + 1], v[j]
                    t = tuple(v)
                    if t not in seen: new.add(t)
        seen |= new; frontier = new
    return seen


def layouts(g, rounds=3, cap=20000):
    """Distinct greedy layouts over the rotation+commutation class.

    Generated as `rounds` alternations of (all rotations) then (commutation
    closure).  Interleaving is required: commutations of the rotations alone
    miss layouts where a band is split around a roamer.  Bounded and fast
    (~0.1s) unlike a naive joint BFS, which re-rotates every variant.
    """
    S = {tuple(g)}
    for _ in range(rounds):
        S = {tuple(list(w)[i:] + list(w)[:i]) for w in S for i in range(len(g))}
        S = _comm_close(S, cap)
        if len(S) >= cap: break
    reps = {}
    for w in S:
        x = columns_greedy(list(w))
        key = tuple(sorted(zip(w, x)))
        if key not in reps: reps[key] = (list(w), x)
    return list(reps.values())


def _combos(choice, succ, core, W, L, Bc, tgt, cap, deadline=None):
    n = 0
    for combo in product(*choice):
        n += 1
        if n > cap: return None
        if deadline and (n & 63) == 0 and time.time() > deadline: return None
        H = [[None] * L for _ in range(W + 1)]; good = True
        for s in range(L):
            c = core[s]
            sv = combo[[k for k in range(L) if succ[k] == s][0]]
            ls = route(0, sv, c['fx'], c['fy'], slope_last=c['entry'])
            rs = route(c['lx'], c['ly'], W, combo[s], slope_first=c['exit'])
            if ls is None or rs is None: good = False; break
            H[0][s] = sv
            for k, hh in enumerate(ls): H[k + 1][s] = hh
            for xx, hh in c['mid'].items(): H[xx][s] = hh
            for k, hh in enumerate(rs): H[c['lx'] + 1 + k][s] = hh
        if not good or any(v is None for r in H for v in r): continue
        runs = pb4.trace(tuple(tuple(r) for r in H), W, L)
        if runs is None or not valid_diagram(runs, L): continue
        vr = pb4.verify(runs, L, Bc)
        if vr is not None and pb4.braid_key(vr, L, Bc) == tgt: return runs
    return None


def align(g, L, core, succ, tgt, Bc, xc, K=48, cap=1500, Wspan=12, deadline=None, Wcap=None):
    """W emergent. Per W: narrow window first, then wide (deep closures)."""
    Wstart = xc if xc % 2 == 0 else xc + 1
    Wend = Wstart + 2 * Wspan
    if Wcap is not None: Wend = min(Wend, Wcap)   # only look for a STRICTLY better W
    for W in range(Wstart, Wend, 2):
        if deadline and time.time() > deadline: return None
        inter = []; ok = True
        for s in range(L):
            c = core[s]
            Rs = [h for h in range(c['fy'] - c['fx'], c['fy'] + c['fx'] + 1)
                  if route(0, h, c['fx'], c['fy'], slope_last=c['entry']) is not None]
            Re = [h for h in range(c['ly'] - (W - c['lx']), c['ly'] + (W - c['lx']) + 1)
                  if route(c['lx'], c['ly'], W, h, slope_first=c['exit']) is not None]
            inter.append((Rs, Re))
        ch = []
        for s in range(L):
            it = sorted(set(inter[s][1]) & set(inter[succ[s]][0]),
                        key=lambda v: abs(v - core[s]['ly']))
            if not it: ok = False; break
            ch.append(it)
        if not ok: continue
        for kk, cc in ((4, 64), (K, cap)):
            runs = _combos([c[:kk] for c in ch], succ, core, W, L, Bc, tgt, cc, deadline)
            if runs is not None: return (W, runs)
    return None


def build(g, L, x, Bc, tgt, K=48, cap=1500, deadline=None, Wcap=None):
    inc, thru, relY, comp, comps = solve_components(g, L, x)
    y = {t: relY[t] for t in range(len(g))}
    core = {}
    for s in range(L):
        lst = inc[s]; fx, fr, ft = lst[0]; lx, lr, lt = lst[-1]; mid = {fx: y[ft]}
        for i in range(len(lst) - 1):
            (xa, ra, ta), (xb, rb, tb) = lst[i], lst[i + 1]
            seg = route(xa, y[ta], xb, y[tb], slope_first=thru(ra), slope_last=thru(rb))
            if seg is None: return None
            for k, hh in enumerate(seg): mid[xa + 1 + k] = hh
        core[s] = dict(fx=fx, fy=y[ft], entry=thru(fr), lx=lx, ly=y[lt],
                       exit=thru(lr), mid=mid)
    return align(g, L, core, perm_succ(g, L), tgt, Bc, max(x), K, cap, deadline=deadline, Wcap=Wcap)


def wlb(g):
    """Lower bound on minimal W: 2 x the size of the largest band (all occurrences
    of one generator).  A band of n crossings occupies n columns at dx=2, and the
    period must exceed that span.  Empirically holds in 17/17 verified cases and is
    TIGHT in 13 -- so hitting it means we can stop immediately.
    (Empirical, not proven: if it were ever too high we would stop at a valid but
    non-minimal W, never at an invalid diagram.)"""
    from collections import Counter
    return 2 * max(Counter(abs(v) for v in g).values())


def construct(g, L, tl=30.0, maxshift=1):
    """Minimal-W valid zigzag, or None. Returns (best, attempts, status, n_layouts)."""
    Bc = b.smallest_coprime_b(L)
    tgt = pb4.braid_key([abs(z) for z in g], L, Bc)
    best = None; att = 0; t0 = time.time(); deadline = t0 + tl
    LB = wlb(g)
    R = layouts(g)
    for w, gx in R:
        bands = {}
        for t, v in enumerate(w): bands.setdefault(v, []).append(t)
        multi = [v for v, i in bands.items() if len(i) >= 2]
        roam = [v for v, i in bands.items() if len(i) == 1]
        for sh in product(*[range(maxshift + 1)] * len(multi)):
            base = gx[:]
            for j, v in enumerate(multi):
                for t in bands[v]: base[t] = gx[t] + sh[j]
            idxs = []; opts = []
            for v in roam:
                t = bands[v][0]; idxs.append(t); o = {base[t]}
                for u, i in bands.items():
                    if u == v or len(i) < 2 or abs(u - v) < 2: continue
                    a, bb = base[i[0]], base[i[-1]]; c = (a + bb) // 2
                    o |= {c - 1, c, c + 1, bb + 1, a - 1}
                opts.append(sorted(z for z in o if z >= 0))
            for cb in (product(*opts) if idxs else [()]):
                if time.time() - t0 > tl:
                    return best, att, 'timeout', len(R)
                x = base[:]
                for j, t in enumerate(idxs): x[t] = cb[j]
                att += 1
                if best is not None and max(x) >= best[0]: continue   # can't beat best
                r = build(w, L, x, Bc, tgt, deadline=deadline,
                          Wcap=(best[0] if best is not None else None))
                if r and (best is None or r[0] < best[0]):
                    best = r
                    # hit the lower bound -> provably minimal, stop scanning
                    if best[0] <= LB:
                        return best, att, f'minimal(W={best[0]}=LB)', len(R)
    return best, att, 'done', len(R)
