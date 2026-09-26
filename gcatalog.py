"""g-based knot enumeration + construction for the app.

A knot's step-word g (a 1-cycle primitive braid word over sigma_1..sigma_{L-1})
replaces the old C/a/b/c/d parametrization. This module:
  - enumerate_gs(L, glen): distinct knots at (L, |g|=glen), one per cylinder diagram
  - build(L, g): MINIMAL-W zigzag run-sequence for rendering (via g_solve)
Dedup is by cylinder_key: two g's are the same entry iff they draw the same crossings on
the cylinder, up to its symmetries (rotation with distant crossings sliding past each
other, turning it over, reversing direction).  It is B-free.  Two different cylinder
diagrams that are the same knot on the sphere are deliberately KEPT as separate entries.
A cheap word-level canonical form (rotation + reversal + index-reflection) pre-filters.
"""
import g_solve as GS
from itertools import product

def minperiod(w):
    n=len(w)
    for p in range(1,n+1):
        if n%p==0 and all(w[i]==w[(i+p)%n] for i in range(n)): return w[:p]
    return w

def power_ks(g, L):
    """The k > 1 for which g's knot COULD be a k-th power h^k.

    If g's knot is h^k, every word in its rotation+commutation class is a rearrangement
    of h^k, so k divides each generator's count; and perm(h)^k is a single L-cycle,
    which needs gcd(k, L) = 1 (an L-cycle to the k-th power splits into gcd(k, L)
    cycles).  Both hold for literal powers too.  Empty for almost every g, and then g is
    neither a literal nor a knot-level power -- no search needed.  Same for every word
    of g's class (and of its reversal/relabelling), so compute it once per class.
    """
    from math import gcd
    counts={}
    for x in g: counts[abs(x)]=counts.get(abs(x),0)+1
    c=0
    for v in counts.values(): c=gcd(c,v)
    return [k for k in range(2,c+1) if c%k==0 and gcd(k,L)==1]

def is_literal_power(w, ks):
    """Is the word w, as written, h^k for some k in ks?  (Rotating by |w|/k gives w back.)
    Only the admissible ks can occur (see power_ks), so this replaces a full minperiod."""
    n=len(w)
    return any(w[n//k:]+w[:n//k]==w for k in ks)

CLASSCAP = 5000     # words of a rotation+commutation class to explore before
                    # giving up; see is_knot_power

def is_knot_power(g, cap=CLASSCAP, ks=None):
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
    (Only once power_ks leaves some k open.)  ks: power_ks(g, L) if the caller has it.
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
    # Exact shortcut, no search: see power_ks.  With no admissible k, g is not a power.
    # This decides almost every call at once (L=5 |g|=12: 2531 of 2714; L=4 |g|=11,
    # L=6 |g|=11, L=7 |g|=10: all of them), including the classes over cap that used to
    # come back undecided.  L is max(g)+1 because g contains every generator.
    if ks is None: ks=power_ks(g, max(abs(x) for x in g)+1)
    if not ks: return False
    if is_literal_power(start, ks): return True
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
            if is_literal_power(v, ks): return True
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

def is_canonical(g, L):
    """g == g_canon(g, L), decided without computing g_canon: stop at the first of the
    4|g| rotated forms that beats g.  Only rotations that START WITH 1 can: g starts with
    1 (it is a candidate), so a rotation starting higher is larger at once.  In g and
    reversed g those start at a 1; in the relabelled forms (i -> L-i) at an L-1.  Most
    words are rejected by the first or second rotation tried."""
    if g[0] != 1: return False
    n = len(g); t = list(g)
    r = t[::-1]; rel = [L - x for x in t]; relr = rel[::-1]
    for f, lead in ((t, 1), (r, 1), (rel, 1), (relr, 1)):
        for i in range(n):
            if f[i] == lead and f[i:] + f[:i] < t: return False
    return True

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

    Used to memoise cylinder_key.  g_canon folds rotation and reflection but not
    commutation, so a level ends up with far more canonical forms than knots -- L=7
    |g|=10 has 79,992 canonical forms and 109 knots.  Words with the same signature are
    the same trace, hence the same closure, hence the same key: 16x fewer key
    computations.
    """
    from collections import Counter
    return (tuple(sorted(Counter(g).items())),
            tuple(tuple(x for x in g if x == a or x == a + 1) for a in range(1, L - 1)))


def _trace_key(w):
    """Canonical form of the CYCLIC word w up to commutation of letters whose indices
    differ by >= 2 -- i.e. of its crossings drawn on the cylinder, where they may slide
    past each other and around the axis.  (w must use consecutive generators, as every g
    does: 1..L-1.)

    That cylinder picture is fixed by, for each adjacent pair (i, i+1), the cyclic order
    in which their letters interleave, with the occurrences of each letter matched up
    between the pairs (i-1, i) and (i, i+1).  Canonical form: choose which occurrence of
    the lowest generator comes first (s); cut the (lo, lo+1) projection just before it;
    the first lo+1 after that cut is where the (lo+1, lo+2) projection is cut; and so
    on up the chain.  Take the smallest resulting tuple over all s.  O(n_lo * |w|).
    Checked against brute force (the full rotation+commutation class): see gcatalog tests.
    """
    gens = sorted(set(w))
    n = len(w)
    if len(gens) == 1:                                   # one generator: just rotation
        return min(tuple(w[r:] + w[:r]) for r in range(n))
    projs = [[x for x in w if x == a or x == a + 1] for a in gens[:-1]]
    first = [[j for j, x in enumerate(p) if x == a] for p, a in zip(projs, gens)]
    best = None
    for s in range(len(first[0])):
        parts = []; occ = s                              # occurrence of letter a to cut at
        for p, pos, a in zip(projs, first, gens):
            cut = pos[occ]
            rot = p[cut:] + p[:cut]
            parts.append(tuple(rot))
            # the first a+1 after the cut, as an occurrence index of a+1: the number of
            # a+1's before the cut in p (mod their count)
            before = sum(1 for x in p[:cut] if x == a + 1)
            occ = before % sum(1 for x in p if x == a + 1)
        t = tuple(parts)
        if best is None or t < best: best = t
    return best

def cylinder_key(g, L):
    """Identity of g's DIAGRAM ON THE CYLINDER: equal iff the two words draw the same
    crossings on the cylinder up to its symmetries -- rotation around the axis with
    distant crossings sliding past each other (_trace_key), turning it upside down
    (relabel i -> L-i), and reversing direction (reversed word).  Mirror is implicit: an
    unsigned word carries no handedness.

    This is what the enumeration dedups by.  It is B-free (equal diagrams stay equal
    repeated B times) and decides nothing about the knot beyond the cylinder: two
    different cylinder diagrams that happen to be the same knot on the sphere (e.g. by a
    flype) are deliberately kept as two entries.  It replaced braid_key (Gauss code of
    g^Bc on the sphere), which merged exactly the same classes on every level checked.
    """
    r = [L - x for x in g]
    return min(_trace_key(f) for f in (list(g), g[::-1], r, r[::-1]))


def _candidate_words(L, glen):
    """Yield, in lex order, exactly the single-L-cycle words that start with 1.

    Two facts prune the space hard, and both are exact -- no knot is lost:

      * it starts with 1.  A single L-cycle word must contain every generator, so 1 is
        its minimum, and a rotation-minimal word starts at its minimum.  Fixing the
        first letter removes (L-2)/(L-1) of the space on its own.
      * each letter is a transposition, which changes the permutation's cycle count by
        exactly +-1: it merges two cycles if its two strands lie in different ones and
        splits one if they share it.  So a prefix whose permutation has k cycles needs
        at least k-1 more letters to reach one cycle; with fewer left, prune the whole
        subtree.  (Parity is automatic: iter_gs is only asked for |g| = L-1 mod 2.)
        This subsumes the older "every generator still missing must fit" rule -- m
        missing generators cut the strands into at least m+1 blocks, so k-1 >= m.

      * no letter directly follows one at least 2 larger ("3 1", "4 2", ...).  Such a
        pair commutes, and swapping it gives a smaller word of the same cylinder class.
        The word iter_gs lists for a class is the lexicographically smallest word of the
        WHOLE class (rotations, commutations, reversal, relabelling) -- it is its own
        canonical form and is met first -- and that word cannot contain such a pair.  So
        every listed word is still generated, in the same order; only commutation
        variants that would be discarded at the key are cut, at the prefix.

    Every leaf therefore IS a single cycle, and iter_gs no longer re-checks it.  Before
    the cycle bound, 70-80% of the leaves were multi-cycle words discarded one by one
    (L=7 |g|=10: 2.74M leaves -> 0.51M; L=5 |g|=12: 3.67M -> 1.14M).
    """
    if glen < L - 1: return
    perm = list(range(L))          # strand at each position after the prefix
    w = [0] * glen
    def same_cycle(i, j):          # are positions i and j in one cycle of perm?
        k = perm[i]
        while k != i:
            if k == j: return True
            k = perm[k]
        return False
    def rec(i, cycles):
        if glen - i < cycles - 1: return            # cannot merge down to one cycle
        if i == glen:
            yield list(w); return
        # v >= w[i-1] - 1: never a letter right after one at least 2 larger (see the
        # docstring).  Ascending keeps the output lex-ordered.
        for v in range(max(1, w[i - 1] - 1), L):
            w[i] = v
            d = 1 if same_cycle(v - 1, v) else -1   # split or merge
            perm[v - 1], perm[v] = perm[v], perm[v - 1]
            yield from rec(i + 1, cycles + d)
            perm[v - 1], perm[v] = perm[v], perm[v - 1]
    w[0] = 1
    perm[0], perm[1] = perm[1], perm[0]
    yield from rec(1, L - 1)


# Big levels are scanned in lexicographic order, so their first knots all open with long
# runs of 1s and look alike -- and at a big level those first few are all anyone browses.
# Above this size iter_gs first SAMPLES random words for variety, then runs the full scan
# for completeness.  Size = (L-1)**(|g|-1), the unpruned candidate count; 10**6 puts the
# switch at L=5 |g|=12, L=6 |g|=11, L=7 |g|=10, L=8 |g|=9, L=9 |g|=8 -- where lists run
# to hundreds or thousands of knots.  (It was chosen when those scans took 10s+; they now
# take a few seconds, but the point is variety in the first few, not speed.)
SAMPLE_ABOVE = 10**6
SAMPLE_KNOTS = 200       # knots to take by sampling before switching to the full scan
SAMPLE_STALL = 200       # ...or stop sampling after this many valid words in a row add nothing

def _single_cycle(g, L):
    perm = list(range(L))
    for v in g: perm[v - 1], perm[v] = perm[v], perm[v - 1]
    k = perm[0]; n = 1
    while k != 0: k = perm[k]; n += 1
    return n == L

def _sampled_words(L, glen):
    """Random single-L-cycle words, forever.  Seeded by (L, |g|), so a level's order --
    and hence what "#9" means -- is the same on every run."""
    import random
    rng = random.Random(L * 100003 + glen)
    while True:
        g = [rng.randint(1, L - 1) for _ in range(glen)]
        if _single_cycle(g, L): yield g

def iter_gs(L, glen, maxknots=MAXKNOTS, sample=None):
    """Yield the knots one at a time.

    A generator so a caller can show the first knot immediately and abandon the scan
    partway -- the UI only ever browses from index 1 upward, so waiting for the final
    count before displaying anything is wasted time.  elastic/server.py drives this from
    a worker thread and stops it when L or |g| changes.

    ORDER.  Small levels: sorted (lexicographic by canonical word).  Big levels (see
    SAMPLE_ABOVE; sample=True/False forces it): first up to SAMPLE_KNOTS knots found by
    random sampling, then the full lexicographic scan, which skips every knot already
    yielded.  Either way the set of knots is the same and complete, and knots are only
    ever APPENDED, so an index never changes meaning while the list grows.  One
    difference: a knot found by sampling is shown in the canonical spelling of whichever
    class was met first -- still a spelling of the same knot, not always the smallest.
    """
    seen_canon=set(); seen_key=set(); sig_key={}
    def admit(g, canonical=False):
        """The canonical word if g is a NEW knot, else None.  g: a single-L-cycle word;
        canonical=True says g is already its own g_canon form (the full scan checks that
        with is_canonical, far cheaper than computing g_canon).  Everything here runs
        once per g_canon CLASS, not per word: periodicity and the power test are the same
        for every member of a class."""
        if canonical:
            c=tuple(g)
            if c in seen_canon: return None           # met while sampling
        else:
            c=g_canon(g,L)
            if c in seen_canon: return None
            seen_canon.add(c)
        ks=power_ks(c, L)                             # usually empty: then no power tests
        if ks and is_literal_power(c, ks): return None   # literal h^k: h is listed at |g|/k
        sg=_trace_sig(c, L)                           # same linear trace => same key
        if sg in sig_key: key=sig_key[sg]
        else: key=sig_key[sg]=cylinder_key(c,L)       # the dedup: same cylinder diagram
        if key is None or key in seen_key: return None
        seen_key.add(key)
        # EXPENSIVE, so last: drop knot-level powers (h^k knots whose WORD is not one).
        # None = undecided (class too large to decide): keep it.  Dropping would hide a
        # real knot; keeping at worst duplicates a lower-|g| family at a different B.
        if ks and is_knot_power(list(c), ks=ks) is True: return None
        return list(c)
    found=0
    if sample is None: sample = (L - 1) ** (glen - 1) > SAMPLE_ABOVE
    if sample and glen >= L - 1 and (glen - (L - 1)) % 2 == 0:
        stall=0
        for g in _sampled_words(L, glen):
            c=admit(g)
            if c is None:
                stall+=1
                if stall>=SAMPLE_STALL: break
                continue
            stall=0
            yield c
            found+=1
            if maxknots is not None and found>=maxknots: return
            if found>=SAMPLE_KNOTS: break
    # The full scan.  Candidates come in lex order and every class's canonical form is
    # itself a candidate (it starts with 1 and is a single cycle), so a class is met
    # exactly once as "g is its own canonical form" -- no g_canon, and no set of the
    # hundreds of thousands of forms seen (only the sampled ones, to skip them).
    for g in _candidate_words(L, glen):
        if not is_canonical(g, L): continue
        c=admit(g, canonical=True)
        if c is None: continue
        yield c
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
