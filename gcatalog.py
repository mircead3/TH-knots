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

def enumerate_gs_ex(L, glen, maxknots=MAXKNOTS, cmax=None):
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
    for g in iter_gs(L, glen, maxknots, cmax=cmax):
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


def bights_of(g, L):
    """Number of bights of g's drawings (without wiggles), from the word alone.

    A bight is where the curve switches between rising and falling, and without wiggles
    that happens exactly between consecutive crossings of different direction.  Passing
    sigma_v, the strand at position v-1 moves UP to v and the one at v moves DOWN.  Follow
    the single curve from position 0 through g, L times (g is one L-cycle, so it returns),
    record U/D at each crossing, and count direction changes cyclically.  C = bights/2.
    Equals g_solve.segments(g, L)[1]; O(L*|g|)."""
    pos = 0; seq = []
    for _ in range(L):
        for v in g:
            v = abs(v)
            if pos == v - 1: seq.append(1); pos = v
            elif pos == v: seq.append(-1); pos = v - 1
    return sum(1 for i in range(len(seq)) if seq[i] != seq[i - 1])


def _candidate_words(L, glen, cmax=None):
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
        (The full Anisimov-Knuth test -- no letter slides left past a larger one over
        any run of letters it commutes with -- was tried 2026-09-28: exact, but it cut
        no extra words at the levels measured and cost ~13%, so it is not used.)
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
    # C pruning (cmax): each strand's crossings within this one pass through g keep their
    # directions whatever letters follow, so direction changes between them are bights
    # of the final drawing -- a lower bound that only grows along the prefix.  Bights
    # across the seam between passes need the whole word; bights_of settles them at the
    # leaf.  lastdir[s] = direction of strand s's latest crossing in the prefix.
    def same_cycle(i, j):          # are positions i and j in one cycle of perm?
        k = perm[i]
        while k != i:
            if k == j: return True
            k = perm[k]
        return False
    w[0] = 1
    perm[0], perm[1] = perm[1], perm[0]

    if cmax is None:
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
        yield from rec(1, L - 1)
        return

    # C pruning.  Each strand's crossings within this one pass through g keep their
    # directions whatever letters follow, so direction changes between them are bights
    # of the final drawing.  The SEAM adds more: the curve is the L strand pieces of one
    # pass, joined end to end in an order only the complete word fixes.  A piece is UU,
    # DU, UD or DD by its (first, last) direction so far; wherever one piece's last
    # direction differs from the next one's first, the rest of that strand or the seam
    # must hold a bight.  In ANY cyclic order the directions must balance: at least
    # |#UD - #DU| such joins, and at least 2 if all pieces are constant but both UU and DD
    # occur.  Changes inside pieces + that minimum bounds the final count; bights_of
    # settles it exactly at the leaf.  Kept cheap: plain ints, piece type index
    # k = 2*(first up) + (last up), i.e. DD 0, DU 1, UD 2, UU 3.
    bmax = 2 * cmax
    lastdir = [0] * L; firstdir = [0] * L; cnt = [0, 0, 0, 0]
    lastdir[0], lastdir[1] = 1, -1                  # sigma_1: strand 0 up, strand 1 down
    firstdir[0], firstdir[1] = 1, -1
    cnt[3] += 1; cnt[0] += 1
    def rec(i, cycles, changes):
        if i == glen:
            if bights_of(w, L) <= bmax: yield list(w)
            return
        for v in range(max(1, w[i - 1] - 1), L):
            d = 1 if same_cycle(v - 1, v) else -1
            if glen - i - 1 < cycles + d - 1: continue  # cannot merge down to one cycle
            lo, hi = perm[v - 1], perm[v]               # lo moves up, hi moves down
            plo, phi = lastdir[lo], lastdir[hi]
            flo, fhi = firstdir[lo], firstdir[hi]
            ch = changes + (plo == -1) + (phi == 1)
            nflo = flo or 1; nfhi = fhi or -1
            if flo: cnt[2 * (flo > 0) + (plo > 0)] -= 1
            if fhi: cnt[2 * (fhi > 0) + (phi > 0)] -= 1
            klo = 2 * (nflo > 0) + 1; khi = 2 * (nfhi > 0)
            cnt[klo] += 1; cnt[khi] += 1
            ud, du = cnt[2], cnt[1]
            seam = abs(ud - du) if (ud or du) else (2 if cnt[3] and cnt[0] else 0)
            if ch + seam <= bmax:
                w[i] = v
                firstdir[lo], firstdir[hi] = nflo, nfhi
                lastdir[lo], lastdir[hi] = 1, -1
                perm[v - 1], perm[v] = hi, lo
                yield from rec(i + 1, cycles + d, ch)
                perm[v - 1], perm[v] = lo, hi
                lastdir[lo], lastdir[hi] = plo, phi
                firstdir[lo], firstdir[hi] = flo, fhi
            cnt[klo] -= 1; cnt[khi] -= 1
            if flo: cnt[2 * (flo > 0) + (plo > 0)] += 1
            if fhi: cnt[2 * (fhi > 0) + (phi > 0)] += 1
    yield from rec(1, L - 1, 0)


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

def iter_gs(L, glen, maxknots=MAXKNOTS, sample=None, cmax=None):
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
            # A word over the C limit counts as a miss: with a tight cmax almost every
            # random word is one, and skipping them without counting never ended sampling.
            c=None if cmax is not None and bights_of(g, L) > 2 * cmax else admit(g)
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
    for g in _candidate_words(L, glen, cmax):
        if not is_canonical(g, L): continue
        c=admit(g, canonical=True)
        if c is None: continue
        yield c
        found+=1
        if maxknots is not None and found>=maxknots: return

# ---------------------------------------------------------------- small C, any |g|
#
# The word search cannot reach small-C knots at large |g| (C=3 at L=16 runs to |g|=43):
# for small C almost every complete word fails the bight count only at the end.  So
# for small C, generate DRAWINGS instead, the way C mode did, but at every width:
# zigzags with exactly 2C runs, validated on the lattice, their words read off.
#
# COMPLETENESS rests on an observation, not a proof: every knot checked has a drawing
# of width W <= 2C (727 library knots; wherever the exhaustive word search finishes --
# C<=3 up to L=10 |g|<=17, C=4 up to L=8 |g|<=13 -- this generator finds exactly its
# knots; W up to 2C+2 finds nothing more for C=2,3 at L=4..11 and C=4 at L=4..8).  |g| <= C*(L-1) IS proven: along a run the curve's
# rank moves one way, so a run passes at most L-1 crossings, and each crossing is passed
# twice.

def _compositions(n, k, lo=2):
    """k-tuples of ints >= lo summing to n.  lo=2: a run of length 1 goes from bight to
    bight with no interior lattice point, so it can hold no crossing -- an idle run, i.e.
    a wiggle, which the bight count below would reject anyway."""
    if k == 1:
        if n >= lo: yield (n,)
        return
    for a in range(lo, n - lo * (k - 1) + 1):
        for rest in _compositions(n - a, k - 1, lo): yield (a,) + rest


def word_from_runs(runs, L, W):
    """The tile word of the zigzag with these run lengths (up first) at width W, or None
    if it is not a valid drawing.  The walk takes one column per step (column = t mod W).
    VALID: on every half-column the L steps are at different heights (sums y_t+y_{t+1}),
    which rules out overlapping steps and crossings between lattice points and leaves
    only transversal crossings and peak/valley tangencies at columns.  WORD: each step's
    rank on its half-column; at a column a rank rising by one is the rising pass of
    sigma_(new rank).  Letters of one column commute, so their order is immaterial."""
    N = W * L
    y = [0]
    for i, r in enumerate(runs):
        d = 1 if i % 2 == 0 else -1
        for _ in range(r): y.append(y[-1] + d)
    if len(y) != N + 1 or y[-1] != 0: return None
    y.pop()
    rank = [0] * N
    for x in range(W):
        steps = range(x, N, W)
        sums = [y[t] + y[(t + 1) % N] for t in steps]
        if len(set(sums)) != L: return None
        for r, t in enumerate(sorted(steps, key=lambda t: y[t] + y[(t + 1) % N])):
            rank[t] = r
    cols = [[] for _ in range(W)]
    for t in range(N):
        if rank[t] - rank[t - 1] == 1: cols[t % W].append(rank[t])
    return [v for c in cols for v in sorted(c)]


def _valid_walks(L, C, W):
    """Run sequences (u1,d1,...,uC,dC) of VALID drawings at width W, generated
    incrementally: the walk is laid down one step at a time, keeping the half-column
    heights (sums y_t + y_{t+1}) already in use, and a run stops growing at its first
    collision -- every longer run collides at the same step, so the whole branch goes.
    Runs are >= 2 (a length-1 run holds no crossing), up- and down-runs each total W*L/2,
    and u1 is the largest up-run (a necessary condition for the canonical rotation; the
    caller applies the full test)."""
    N = W * L
    if N % 2: return
    half = N // 2
    used = [set() for _ in range(W)]
    runs = [0] * (2 * C)
    def rec(r, t, y, up_left, dn_left):
        if r == 2 * C:
            if t == N and y == 0: yield list(runs)
            return
        up = r % 2 == 0
        n_same = (2 * C - r + 1) // 2                  # runs of this kind left, incl. this
        budget = up_left if up else dn_left
        if n_same == 1: kmin = kmax = budget           # the last one takes what is left
        else: kmin, kmax = 2, budget - 2 * (n_same - 1)
        if up and r > 0: kmax = min(kmax, runs[0])
        if kmin > kmax: return
        d = 1 if up else -1
        added = []; tt, yy = t, y
        for k in range(1, kmax + 1):
            x = tt % W; s = 2 * yy + d
            if s in used[x]: break                     # collision: no longer run can work
            used[x].add(s); added.append((x, s))
            tt += 1; yy += d
            if k >= kmin:
                runs[r] = k
                if up: yield from rec(r + 1, tt, yy, up_left - k, dn_left)
                else: yield from rec(r + 1, tt, yy, up_left, dn_left - k)
        for x, s in added: used[x].discard(s)
    yield from rec(0, 0, 0, half, half)


def iter_small_c(L, C, Wmax=None):
    """Knots with exactly C bight pairs, any |g|, from drawings with 2C runs and width
    W <= Wmax (default 2C; see the COMPLETENESS note above).  Yields canonical words,
    each knot once (cylinder key), knot-level powers dropped as in iter_gs.
    C=4 at L=11: 16133 knots in ~3 min (the word search could not finish |g| <= 22)."""
    Wmax = Wmax or 2 * C
    seen = set()
    for W in range(1, Wmax + 1):
        for runs in _valid_walks(L, C, W):
            # the same drawing starting at another valley: keep the largest rotation
            if any(runs[2 * k:] + runs[:2 * k] > runs for k in range(1, C)): continue
            g = word_from_runs(runs, L, W)
            if not g or len(set(g)) < L - 1: continue      # must use every generator
            if bights_of(g, L) != 2 * C: continue          # wiggles: a smaller C
            key = cylinder_key(g, L)
            if key in seen: continue
            seen.add(key)
            c = g_canon(g, L)
            ks = power_ks(c, L)
            if ks and (is_literal_power(c, ks) or is_knot_power(list(c), ks=ks) is True):
                continue
            yield list(c)


# ---------------------------------------------------------------- filtered browsing

SMALL_C = 4        # up to this C, knots come from the drawing generator (any |g|)
_SMALL_C_DONE = {}  # (L, C) -> iter_small_c's complete output: it yields every |g| at once,
                    # so stepping |g| under a small-C filter must not regenerate it

def word_self_flip(g, L):
    """Is the braid symmetric under the flip (relabel i -> L-i: the drawing upside down),
    up to rotation + commutation, allowing reversal?  A property of the WORD.  The flip
    button tests g_solve's particular drawing instead; they disagree on 2 of the 727
    library knots, whose words are symmetric though that drawing is not."""
    r = [L - abs(x) for x in g]
    k = _trace_key(r)
    return k == _trace_key([abs(x) for x in g]) or k == _trace_key([abs(x) for x in g][::-1])


def braid_amphichiral(g, L):
    """Braid chirality, two-valued (as in the app's braid view).  With the parity signs
    of these alternating diagrams, the mirror word (relabel i -> L-i, negate) follows the
    same sign pattern only for ODD L, where it then IS the flip: amphichiral <=> self-flip.
    For even L the mirror breaks the pattern, so every braid is chiral (odd writhe).
    Library check: 54 amphichiral = the 54 odd-L self-flip knots."""
    return L % 2 == 1 and word_self_flip(g, L)


def iter_filtered(L, gmin, gmax, cmin=None, cmax=None, chiral=None, selfflip=None):
    """Knots at L with gmin <= |g| <= gmax and cmin <= C <= cmax, optionally only chiral /
    amphichiral (chiral=True/False) and self-flip or not (selfflip=True/False).
    The C range is SPLIT by source: C <= SMALL_C comes from the drawing generator
    (iter_small_c), which reaches any |g| -- its knots come out by width, not by |g|; the
    rest (C > SMALL_C) from the word search level by level, C-pruned when cmax is set and
    never beyond cmax*(L-1), the proven bound.  Streams canonical words, each knot once
    (the two parts cannot overlap: they have different C)."""
    def keep(g, clo):
        if not gmin <= len(g) <= gmax: return False
        if clo is not None and bights_of(g, L) < 2 * clo: return False
        if chiral is not None and braid_amphichiral(g, L) == chiral: return False
        if selfflip is not None and word_self_flip(g, L) != selfflip: return False
        return True
    lo = max(1, cmin or 1)
    # 1. small C: the generator, one C at a time (memoised per (L, C): every |g| at once)
    for C in range(lo, min(SMALL_C, cmax if cmax is not None else SMALL_C) + 1):
        if C * (L - 1) < gmin: continue
        done = _SMALL_C_DONE.get((L, C))
        if done is None:
            found = []
            for g in iter_small_c(L, C):
                found.append(g)
                if keep(g, None): yield g
            _SMALL_C_DONE[(L, C)] = found            # only once complete (not abandoned)
        else:
            for g in done:
                if keep(g, None): yield g
    # 2. larger C: the word search, keeping only C > SMALL_C (and >= cmin)
    if cmax is not None and cmax <= SMALL_C: return
    clo = max(lo, SMALL_C + 1)
    top = gmax if cmax is None else min(gmax, cmax * (L - 1))
    for gl in range(max(gmin, L - 1), top + 1):
        if gl % 2 != (L - 1) % 2: continue
        for g in iter_gs(L, gl, cmax=cmax):
            if keep(g, clo): yield g


def enumerate_gs(L, glen, maxknots=MAXKNOTS, cmax=None):
    """Just the list; see enumerate_gs_ex for the truncation flag.  cmax: only knots
    with C <= cmax (C = bights_of / 2), pruned during the search, not filtered after."""
    return enumerate_gs_ex(L, glen, maxknots, cmax)[0]

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
    runs, W, verdict, tier = out['runs'], out['W'], out['verdict'], out['tier']
    # A self-flip WORD gets a self-flip DRAWING, even at a larger W (the user's preference:
    # symmetry over width).  g_solve returns one of possibly several minimal drawings and it
    # need not be symmetric -- 2 of the 727 library knots -- so if not, ask for one with
    # run-length symmetry constraints (g_solve.solve_symmetric), from W upward.  If none
    # turns up within its range, keep the asymmetric drawing.
    if word_self_flip(g, L) and not GS.flip_symmetric(runs):
        found = GS.solve_symmetric(g, L, W)
        if found is not None:
            ssol, sverdict = found
            sruns = GS.to_runs(g, L, ssol)
            if (sruns and not GS.check(g, L, ssol) and GS.gauss_ok(g, L, ssol)
                    and GS.strict_ok(g, L, sruns)):
                runs, W, tier = sruns, ssol['W'], ssol['tier']
                verdict = 'symmetric' if W > out['W'] else verdict
    return {'g':g, 'L':L, 'runs':runs, 'W':W,
            'C':len(runs)//2, 'rotation':g,
            'verdict':verdict, 'tier':tier}

def enumerate_built(L, glen, maxknots=MAXKNOTS):
    """Enumeration with construction: each entry has g, W, C, runs.
       (Symmetry is computed app-side from the rendered diagram.)

    Shares enumerate_gs's output bound so the two cannot disagree about a level."""
    out=[]
    for g in enumerate_gs(L, glen, maxknots):
        bd=build(L, g)
        if bd is not None: out.append(bd)
    return out
