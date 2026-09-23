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

CLASSCAP = 5000     # words of a rotation+commutation class to explore before
                    # giving up; see is_knot_power

def is_knot_power(g, cap=CLASSCAP):
    """Is g's KNOT a repetition h^k, whether or not the WORD is one?

    Literal periodicity (minperiod(g) != g) is not invariant under the moves that
    preserve the knot -- cyclic rotation and commutation of distant generators
    (sigma_i sigma_j = sigma_j sigma_i for |i-j| >= 2).  So a non-periodic word can
    sit in the same class as a periodic one: [1,2,1,3,2,4,3,4] has no proper period
    yet lies in the class of [2,4,1,3]^2, and only 24 of that class's 136 words are
    literally periodic.  Testing one representative therefore decides nothing.

    So walk the rotation+commutation class and stop at the first literal power.
    Cheap in practice (0.6s for the whole 509-knot library; the two real hits are
    found after 4 and 23 words) because it early-exits rather than enumerating.

    Such g's add no knot the library lacks: g at B bights is h at k*B, so they are
    reachable at |h| with more bights.  Keeping them would also leave the only class
    whose minimal W the solver cannot certify internally (the full-period rule is a
    post-check, not a constraint), needing construct_brute to settle.

    Returns True / False / None, where None means UNDECIDED: the class exceeded cap.
    It used to raise there, on the grounds that silently returning False would keep an
    h^k knot.  But once the enumeration stopped refusing large levels, raising refused
    them instead -- and the cost of keeping such a knot is mild: the full-period rule
    still gives it a correct diagram, it is only a family duplicate that dedup missed.
    So the caller keeps an undecided knot and the dedup is best-effort past the cap
    (classes exceed it around |g| 15-18 for L = 5-7; L=3 never, since nothing commutes
    there and the class is just the |g| rotations).

    A polynomial replacement was attempted -- "the trace is fixed by rotation by a proper
    divisor", tested via the projection lemma -- and is WRONG: commutation acts on adjacent
    positions and rotation changes which positions are adjacent, so linear trace
    equivalence does not transfer to cyclic traces.  It returned False for both known
    knot-powers.  A correct version needs cyclic trace equivalence, i.e. commutation across
    the seam, which is what this search already does.
    """
    from collections import deque
    start=tuple(g); n=len(g)
    if minperiod(list(start))!=list(start): return True
    seen={start}; q=deque([start])
    while q:
        w=list(q.popleft())
        cands=[tuple(w[1:]+w[:1])]                      # rotation
        for i in range(n):                              # commutation (cyclic)
            j=(i+1)%n
            if abs(w[i]-w[j])>=2:
                v=w[:]; v[i],v[j]=v[j],v[i]; cands.append(tuple(v))
        for v in cands:
            if v in seen: continue
            if minperiod(list(v))!=list(v): return True
            seen.add(v); q.append(v)
            if len(seen)>cap: return None          # undecided: class too large
    return False

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

MAXKNOTS = None     # no limit: the scan streams and is interruptible, so a long
                    # level costs nothing -- you browse what has arrived and any
                    # change of L or |g| abandons it.  A cap would only truncate
                    # the answer to bound a time that is no longer a problem.

def enumerate_gs_ex(L, glen, maxknots=MAXKNOTS):
    """Distinct knots at (L, |g|=glen), one canonical g each.  -> (gs, truncated)

    Runs to completion by default (maxknots=None).  maxknots remains for callers that
    genuinely want a prefix; the UI does not, because the scan streams and is
    interruptible, so there is no time to bound.

    Any prefix is well defined and deterministic.  Scanning words lexicographically, the
    first member of a commutation/rotation class you meet is its lex-smallest member,
    which IS its canonical form -- so knots are emitted in sorted order and stopping
    early yields exactly the maxknots lexicographically smallest ones.  (That is also
    why the old out.sort() was a no-op.)  Indices therefore never shift, which is what
    makes streaming to the UI safe.

    History, so the reasoning is not relitigated: this began as a cap on (L-1)**glen that
    REFUSED whole levels -- the wrong quantity, since knot counts are modest (L=6: 1, 7,
    71, 778 at |g|=5,7,9,11) while the candidate space grows 25x per step, so levels with
    a few hundred browsable knots were rejected for the brute force behind them.  Before
    that it silently truncated mid-scan with no indication at all.  Streaming plus
    interruption removes the need for any limit on the answer.

    NO LIMIT on the level.  There used to be one on the candidate space ((L-1)**glen),
    which was pointless: the scan walks words lexicographically from (1,1,...,1) and
    canonical knots turn up immediately, so the size of the space says nothing about how
    fast the first knots arrive.  Measured: L=5 |g|=14 has 268 million words yet yields
    knot #1 in 0.02s and #20 in 1.1s.  A limit there refused levels whose first screenful
    was instant.  Runaway CPU on a level nobody is watching is the server's problem to
    solve (it stops an idle scan), not a reason to refuse the level.
    """
    out=[]
    for g in iter_gs(L, glen, maxknots):
        out.append(g)
    return out, (maxknots is not None and len(out)>=maxknots)


def _trace_sig(g, L):
    """A complete invariant of g's COMMUTATION class (projection lemma): the letter
    counts, plus the subsequence on each non-commuting pair of generators.

    Used to memoise braid_key.  g_canon folds rotation and reflection but not
    commutation, so a level ends up with far more canonical forms than knots -- L=7
    |g|=10 has 79,992 canonical forms and 109 knots.  Words with the same signature are
    the same trace, hence the same closure, hence the same key: 16x fewer calls to the
    most expensive step in the scan.
    """
    from collections import Counter
    return (tuple(sorted(Counter(g).items())),
            tuple(tuple(x for x in g if x == a or x == a + 1) for a in range(1, L - 1)))


def _candidate_words(L, glen):
    """Yield, in lex order, only the words that could possibly be a canonical g.

    Two facts prune the space hard, and both are exact -- no knot is lost:

      * it starts with 1.  A single L-cycle word must contain every generator, so 1 is
        its minimum, and a rotation-minimal word starts at its minimum.  Fixing the
        first letter removes (L-2)/(L-1) of the space on its own.
      * it contains EVERY generator 1..L-1.  So a prefix with more generators still
        missing than positions remaining can never be completed: prune it rather than
        enumerate its whole subtree.

    Before this, iter_gs walked all (L-1)**glen words.  At L=7 |g|=10 that is 60.5M, of
    which only the first 10.1M can hold a canonical form -- the other 50M were scanned
    finding nothing, which is why the count sat unchanged for minutes at the end of a
    level while the scan ground on.
    """
    need0 = frozenset(range(1, L))
    w = [0] * glen
    def rec(i, missing):
        if glen - i < len(missing): return          # cannot still fit the missing ones
        if i == glen:
            yield list(w); return
        for v in range(1, L):                       # ascending keeps the output lex-ordered
            w[i] = v
            yield from rec(i + 1, missing - {v} if v in missing else missing)
    w[0] = 1
    yield from rec(1, need0 - {1})


def iter_gs(L, glen, maxknots=MAXKNOTS):
    """Yield the knots one at a time, in sorted order.

    A generator so a caller can show the first knot immediately and abandon the scan
    partway -- the UI only ever browses from index 1 upward, so waiting for the final
    count before displaying anything is wasted time.  elastic/server.py drives this from
    a worker thread and stops it when L or |g| changes.
    """
    Bc=b.smallest_coprime_b(L)
    seen_canon=set(); seen_key=set(); sig_key={}; found=0
    for g in _candidate_words(L, glen):
        # ORDER MATTERS: perm_cycles is O(n) and discards ~75%, minperiod is O(n^2).
        # The cheap rotation test (a canonical word starts at its own minimum) prunes
        # ~94% more before g_canon, which is the other expensive stage.
        if e.perm_cycles(g,L)!=1: continue
        if g[0]!=min(g): continue
        if minperiod(g)!=g: continue
        c=g_canon(g,L)
        if c in seen_canon: continue
        seen_canon.add(c)
        sg=_trace_sig(c, L)                           # same trace => same knot
        if sg in sig_key: key=sig_key[sg]
        else: key=sig_key[sg]=pb4.braid_key(g,L,Bc)   # authoritative knot dedup
        if key is None or key in seen_key: continue
        seen_key.add(key)
        # EXPENSIVE, so last: drop knot-level powers (h^k knots whose WORD is not one).
        # None = undecided (class too large to decide): keep it.  Dropping would hide a
        # real knot; keeping at worst duplicates a lower-|g| family at a different B.
        if is_knot_power(list(c)) is True: continue
        yield list(c)
        found+=1
        if maxknots is not None and found>=maxknots: return

def enumerate_gs(L, glen, maxknots=MAXKNOTS):
    """Just the list; see enumerate_gs_ex for the truncation flag."""
    return enumerate_gs_ex(L, glen, maxknots)[0]

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

def enumerate_built(L, glen, maxknots=MAXKNOTS):
    """Enumeration with construction: each entry has g, W, C, runs.
       (Symmetry is computed app-side from the rendered diagram.)

    Shares enumerate_gs's output bound so the two cannot disagree about a level."""
    out=[]
    for g in enumerate_gs(L, glen, maxknots):
        bd=build(L, g)
        if bd is not None: out.append(bd)
    return out
