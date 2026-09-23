"""Constraint-propagation placer (2026-09-21).

Replaces "choose x freely, then try to route" with "place crossings incrementally,
with candidates derived from the strands' own +/-45 geometry, rejecting as soon as a
segment cannot route".

Key facts this relies on:
  * The low/high ROLE of each strand at each crossing is fixed by slot-tracking in the
    word -- it does NOT depend on x.  So every strand's departure slope (thru(role)) and
    arrival slope are known before any placement.
  * Between two consecutive crossings of one strand, with departure slope sa and arrival
    slope sb over dx columns:
        sa == sb  ->  straight:  dy = sa*dx                       (y forced by x)
        sa != sb  ->  exactly one bend: dy in {-dx+2 ... dx-2} step 2
    (>=2 bends is impossible: it would create a run with no crossing inside, which
    valid_diagram rejects -- established earlier.)
  * A crossing's position must satisfy BOTH its strands' constraints simultaneously,
    which is what prunes the search.
"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]

from construct4 import route, perm_succ, valid_diagram, pb4
import braids as b
import time


def roles(g, L):
    """Per crossing: (low_strand, high_strand).  Fixed by the word, independent of x."""
    slots = list(range(L)); out = []
    for gi in g:
        lo, hi = slots[gi - 1], slots[gi]
        out.append((lo, hi))
        slots[gi - 1], slots[gi] = slots[gi], slots[gi - 1]
    return out


def strand_seq(g, L):
    """Per strand: ordered list of (crossing_index, slope) it participates in.
    slope = +1 if it is the 'low' strand there, -1 if 'high' (the through-slope)."""
    rl = roles(g, L)
    seq = {s: [] for s in range(L)}
    for t, (lo, hi) in enumerate(rl):
        seq[lo].append((t, +1))
        seq[hi].append((t, -1))
    return seq


def _reach(ya, sa, sb, dx):
    """y-values reachable at distance dx given departure slope sa, arrival slope sb."""
    if dx <= 0: return []
    if sa == sb:
        return [ya + sa * dx]                      # straight only
    lo, hi = ya - dx + 2, ya + dx - 2              # exactly one bend
    return list(range(lo, hi + 1, 2))


def candidates(t, g, L, seq, pos, xcap):
    """Feasible (x,y) for crossing t given already-placed crossings `pos`."""
    rl = roles(g, L)
    lo_s, hi_s = rl[t]
    cons = []
    for s, slope_here in ((lo_s, +1), (hi_s, -1)):
        prev = None
        for (tt, sl) in seq[s]:
            if tt == t: break
            if tt in pos: prev = (tt, sl)
        if prev is None:
            cons.append(None)                      # unconstrained (first appearance)
        else:
            tp, sp = prev
            xp, yp = pos[tp]
            cons.append((xp, yp, sp, slope_here))
    out = []
    xs = range(0, xcap + 1)
    for x in xs:
        cand_y = None; ok = True
        for c in cons:
            if c is None: continue
            xp, yp, sp, sh = c
            ys = _reach(yp, sp, sh, x - xp)
            if not ys: ok = False; break
            if cand_y is None: cand_y = set(ys)
            else: cand_y &= set(ys)
            if not cand_y: ok = False; break
        if not ok: continue
        if cand_y is None:
            continue                                # fully free: handled by seeding
        for y in sorted(cand_y): out.append((x, y))
    # POSET SPACING: crossings that share a slot cannot be arbitrarily close.
    # same generator -> |dx| >= 2 ; adjacent generator -> |dx| >= 1 ; commuting -> free.
    def spacing_ok(x):
        for u, (xu, yu) in pos.items():
            d = abs(abs(g[u]) - abs(g[t]))
            if d == 0 and abs(x - xu) < 2: return False
            if d == 1 and abs(x - xu) < 1: return False
        return True
    out = [(x, y) for (x, y) in out if spacing_ok(x)]
    return out


def solve(g, L, xcap=16, tl=10.0, seed_span=6, node_cap=300000):
    """DFS over crossings in word order with geometry-derived candidates."""
    seq = strand_seq(g, L)
    n = len(g); t0 = time.time()
    Bc = b.smallest_coprime_b(L); tgt = pb4.braid_key([abs(z) for z in g], L, Bc)
    best = [None]; nodes = [0]

    def rec(t, pos):
        if time.time() - t0 > tl: return
        if t == n:
            X = [pos[i][0] for i in range(n)]; Y = [pos[i][1] for i in range(n)]
            r = _finish(g, L, X, Y, Bc, tgt)
            if r and (best[0] is None or r[0] < best[0][0]): best[0] = r
            return
        # MRV: place the MOST-CONSTRAINED unplaced crossing next, not word order.
        # (Several crossings collapse to a single candidate once neighbours are set.)
        best_t, best_cs = None, None
        for u in range(n):
            if u in pos: continue
            cs = candidates(u, g, L, seq, pos, xcap)
            if not cs:
                lvl = 2 * abs(g[u]) - 1
                cs = [(x, y) for x in range(0, 3)
                             for y in (lvl - 2, lvl - 1, lvl, lvl + 1, lvl + 2) if y >= 0]
            if best_cs is None or len(cs) < len(best_cs):
                best_t, best_cs = u, cs
                if len(cs) <= 1: break           # can't do better
        if best_cs is None: return
        for (x, y) in best_cs:
            nodes[0] += 1
            if nodes[0] > node_cap or time.time() - t0 > tl: return
            pos[best_t] = (x, y)
            rec(t + 1, pos)
            del pos[best_t]

    rec(0, {})
    return best[0], nodes[0]


def _finish(g, L, X, Y, Bc, tgt):
    """Build the core from a complete placement and close it (W emergent)."""
    from construct3 import incidences
    from unified import align
    try:
        inc, es, cr = incidences(g, L, X)
    except Exception:
        return None
    thru = lambda r: 1 if r == 'low' else -1
    core = {}
    for s in range(L):
        lst = inc[s]
        fx, fr, ft = lst[0]; lx, lr, lt = lst[-1]; mid = {fx: Y[ft]}
        for i in range(len(lst) - 1):
            (xa, ra, ta), (xb, rb, tb) = lst[i], lst[i + 1]
            seg = route(xa, Y[ta], xb, Y[tb], slope_first=thru(ra), slope_last=thru(rb))
            if seg is None: return None
            for k, hh in enumerate(seg): mid[xa + 1 + k] = hh
        core[s] = dict(fx=fx, fy=Y[ft], entry=thru(fr), lx=lx, ly=Y[lt],
                       exit=thru(lr), mid=mid)
    return align(g, L, core, perm_succ(g, L), tgt, Bc, max(X))
