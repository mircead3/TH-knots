"""g-based knot enumeration + construction for the app.

A knot's step-word g (a 1-cycle primitive braid word over sigma_1..sigma_{L-1})
replaces the old C/a/b/c/d parametrization. This module:
  - enumerate_gs(L, glen): distinct knots at (L, |g|=glen), deduped by knot
  - build(L, g): MINIMAL-W zigzag run-sequence for rendering (via g_solve)
Dedup is by the canonical Gauss code (braid_key); a cheap word-level canonical
form (rotation + reversal + index-reflection) pre-filters obvious duplicates.
"""
import braids as b, enum_g as e, pb4
import g_solve as GS
from itertools import product

def minperiod(w):
    n=len(w)
    for p in range(1,n+1):
        if n%p==0 and all(w[i]==w[(i+p)%n] for i in range(n)): return w[:p]
    return w

def g_canon(g, L):
    """Cheap canonical word form folding rotation, reversal, index-reflection."""
    n=len(g)
    forms=[tuple(g), tuple(reversed(g)),
           tuple(L-i for i in g), tuple(reversed([L-i for i in g]))]
    best=None
    for f in forms:
        for r in range(n):
            rot=f[r:]+f[:r]
            if best is None or rot<best: best=rot
    return best

def enumerate_gs(L, glen, cap=4_000_000):
    """Primitive 1-cycle step-words of length glen on L strands, one per knot.
       Returns sorted list of canonical g's (each a list)."""
    Bc=b.smallest_coprime_b(L)
    seen_canon=set(); seen_key=set(); out=[]; n=0
    for w in product(range(1,L), repeat=glen):
        n+=1
        if n>cap: break
        g=list(w)
        if minperiod(g)!=g: continue                 # only primitive steps
        if e.perm_cycles(g,L)!=1: continue           # all strands same (single cycle)
        c=g_canon(g,L)
        if c in seen_canon: continue
        seen_canon.add(c)
        key=pb4.braid_key(g,L,Bc)                     # authoritative knot dedup
        if key is None or key in seen_key: continue
        seen_key.add(key)
        out.append(list(c))
    out.sort()
    return out

def build(L, g):
    """Minimal-W zigzag for step-word g, by solving the diagram's linear system.

    g_solve returns a PROVEN-minimal W (see g_solve.__doc__), and every diagram is
    post-verified for faithfulness and independently confirmed by its Gauss sequence
    before it reaches the app -- if either check fails we return None so the caller
    skips the knot rather than rendering something wrong.
    """
    g=list(g)
    # Full escalation (max_tier=3) is REQUIRED, not just for the minimality label:
    # for some rotations of g, tiers 1-2 return only unfaithful solutions at the
    # minimal W and tier 3 is what finds the diagram.  Capping at 2 gave W=14 instead
    # of 10 for g=[2,1,3,2,1,1,1,4,4,4].  Latency belongs in a cache, not here.
    out=GS.solve(g, L)
    if out is None or not out['runs']: return None
    sol=out['sol']
    # Gate the renderer on the STRICT knot-identity check, not gauss_ok: gauss_ok passes
    # a diagram whose tile is a repeated shorter block, which closes to a DIFFERENT knot.
    if GS.check(g, L, sol): return None
    if not GS.gauss_ok(g, L, sol): return None
    if not GS.strict_ok(g, L, out['runs']): return None
    return {'g':g, 'L':L, 'runs':out['runs'], 'W':out['W'],
            'C':len(out['runs'])//2, 'rotation':g,
            'verdict':out['verdict'], 'tier':out['tier']}

def enumerate_built(L, glen, cap=4_000_000):
    """Enumeration with construction: each entry has g, W, C, runs.
       (Symmetry is computed app-side from the rendered diagram.)"""
    out=[]
    for g in enumerate_gs(L, glen, cap):
        bd=build(L, g)
        if bd is not None: out.append(bd)
    return out
