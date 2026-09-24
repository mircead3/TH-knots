#!/usr/bin/env python3
"""Regression tests for g_solve.  Run:  python3 test_g_solve.py [--full]

Fast mode (default, ~30s) checks the specific knots that caught real bugs.
--full (~60s) additionally sweeps the whole library.

Every claim made about g_solve should be reproducible from here; if a claim is not
tested below, treat it as unverified.
"""
import sys, os, time
from math import gcd
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import g_solve as GS, gcatalog as GC, pb4, braids as b
from construct4 import construct_brute

fails = []
def ok(cond, msg):
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond: fails.append(msg)

# (g, L, expected W) -- the five Mircea hand-solved, plus C=1
HAND = [([2,3,3,4,1,1,1,2,1,3],5,8), ([2,1,3,2,1,1,1,4,4,4],5,10),
        ([2,3,1,1,1,2,4,4,4,3],5,12), ([4,2,3,2,4,3,1,1,1,4],5,10),
        ([1,2,4,1,3,2,4,1,3,4],5,8),  ([1,2,3,4],5,2)]

print('1. hand-solved knots: minimal W, faithful, strict knot identity')
for g,L,W in HAND:
    out = GS.solve(g,L)
    ok(out is not None and out['W']==W, 'W=%d for g=%s' % (W,g))
    if out:
        ok(not GS.check(g,L,out['sol']), '  faithful  %s' % (g,))
        ok(GS.strict_ok(g,L,out['runs']), '  strict_ok %s' % (g,))
        ok(out['verdict']=='minimal', '  verdict minimal %s' % (g,))

print('2. tier 3 is load-bearing: it rules out W=8 for the knot that needs it')
g,L = [1,1,1,2,1,3,2,4,4,4], 5
ok(GS.feasible(g,L,8,tier=2) is not None, 'W=8 feasible at tier 2 (so tier 3 is needed)')
ok(GS.feasible(g,L,8,tier=3) is None,     'W=8 INFEASIBLE at tier 3')
ok(GS.solve(g,L)['W']==10,                'answer is W=10')

print('3. a short-period diagram is a DIFFERENT knot (strict_ok must reject it)')
g,L = [1,2,1,3,2,4,3,4], 5
ok(not GS.strict_ok(g,L,[5,5,5,5]),   'rejects the W=4 symmetric diagram')
ok(not GS.strict_ok(g,L,[5]*8),       'rejects the W=8 doubled diagram')
ok(GS.strict_ok(g,L,[7,7,8,8]),       'accepts the W=6 full-period diagram')

print('4. knot-level powers are dropped from enumeration')
ok([1,2,1,3,2,4,3,4] not in GC.enumerate_gs(5,8),   'L=5 |g|=8 drops [1,2,1,3,2,4,3,4]')
ok([1,2,1,3,2,1,3,2,3] not in GC.enumerate_gs(4,9), 'L=4 |g|=9 drops [1,2,1,3,2,1,3,2,3]')
ok(GC.is_knot_power([2,4,1,3,2,4,1,3]),             'literal power detected')
ok(GC.is_knot_power([1,2,1,3,2,4,3,4]),             'knot-level power detected')
ok(not GC.is_knot_power([2,1,3,2,1,1,1,4,4,4]),     'a genuine knot is not flagged')
# A class too large to explore is UNDECIDED (None), not False and not an exception.
# It raised at one point, which was right while the enumeration also refused big levels;
# once that stopped, raising refused them instead.  None means the caller keeps the knot:
# at worst a family duplicate, never a wrong diagram.
undec = GC.is_knot_power([1,2,3,4,5,6,1,2,3,4,5,6,2,1,4,3,6,5], cap=500)
ok(undec is None, 'a class over the cap returns None (undecided), neither False nor raise')
ok(GC.is_knot_power([1,2,1,3,2,4,3,4], cap=200) is True,
   'and a small cap still decides the real knot-powers (found after 4 and 23 words)')

print('4b. enumeration streams, runs to completion, and never silently truncates')
gs,tr = GC.enumerate_gs_ex(5,10)
ok(len(gs)==317 and tr is False, 'L=5 |g|=10 -> 317 knots, complete')
ok(GC.MAXKNOTS is None,          'no knot cap (streaming + interruption removes the need)')
import itertools as _it
ok(list(_it.islice(GC.iter_gs(5,10),10))==gs[:10],
   'iter_gs yields a PREFIX of the full list -- indices never shift as it streams')
ok(gs==sorted(gs),               'knots arrive in sorted order (first class member IS canonical)')
gs2,tr2 = GC.enumerate_gs_ex(5,8,maxknots=10)
ok(len(gs2)==10 and tr2 is True, 'an explicit maxknots still works and reports truncated')
# No level is refused any more, and the size of the candidate space says nothing about
# how fast the first knots arrive -- that is what replaced the old backstop.  Bounded in
# time on purpose: an unbounded call here is what turned this suite into a 2-billion-word
# scan when the backstop went away.
# The guarantee is about the FIRST knot: the scan walks words lexicographically from
# (1,1,...,1) and canonical knots turn up at once, so no level needs refusing.  Later
# knots CAN be far apart at extreme levels (L=9 |g|=16 needs ~44s to reach three), which
# is why this asserts one knot, not three.
for L,gl,space in ((5,20,4**20),(7,20,6**20),(9,16,8**16)):
    t0=time.time(); first=next(GC.iter_gs(L,gl)); el=time.time()-t0
    ok(len(first)==gl and el<5,
       'L=%d |g|=%2d (%s words): first knot in %.2fs, not refused'
       %(L,gl,format(space,','),el))

print('4c. a g that is not a single L-cycle is rejected at once, not after scanning W')
for g,L in (([1,1],3), ([2]*7,3), ([1,2,1,2],4)):
    t0=time.time(); sol,verdict = GS.solve_min(g,L); el=time.time()-t0
    ok(sol is None and verdict=='not-a-knot' and el<0.05,
       'L=%d g=%s -> not-a-knot in %.3fs (no W scan)'%(L,g,el))
ok(GS.solve([1,2,3,4],5)['W']==2, 'and a valid single-cycle g is unaffected')

print('5. minimal W is rotation-invariant (it is a property of the knot)')
for g,L in [([2,1,3,2,1,1,1,4,4,4],5), ([2,3,1,1,1,2,4,4,4,3],5)]:
    Ws = {GS.solve(g[r:]+g[:r], L)['W'] for r in range(len(g))}
    ok(len(Ws)==1, 'all %d rotations of %s give W=%s' % (len(g),g,Ws))

print('6. construct_brute oracle agrees (exhaustive, independent of the model)')
st,_ = construct_brute([2,1,3,2,1,1,1,4,4,4],5,Wmax=8,time_limit=300.0)
ok(st=='FAIL', 'nothing at W<=8 for [2,1,3,2,1,1,1,4,4,4] (so W=10 is minimal)')

print('7. the app path returns verified diagrams only')
bd = GC.build(5,[2,1,3,2,1,1,1,4,4,4])
ok(bd is not None and bd['W']==10 and bd['runs'], 'gcatalog.build W=10')

if '--full' in sys.argv:
    print('8. FULL library sweep')
    t0=time.time(); n=0; bad=0; verdicts={}
    for L in (3,4,5):
        glen=L-1
        while glen<=10:
            if glen%2!=(L-1)%2: glen+=1; continue
            for g in GC.enumerate_gs(L,glen):
                n+=1; d=GC.build(L,g)
                if d is None or not GS.strict_ok(g,L,d['runs']): bad+=1
                else: verdicts[d['verdict']]=verdicts.get(d['verdict'],0)+1
            glen+=2
    ok(n==507, 'library size 507 (got %d)'%n)
    ok(bad==0, 'every knot builds and passes strict_ok (%d bad)'%bad)
    ok(verdicts=={'minimal':507}, "all verdicts 'minimal' (got %s)"%verdicts)
    print('  (%.0fs)'%(time.time()-t0))

print()
print('FAILURES: %d' % len(fails))
for f in fails: print('   ', f)
sys.exit(1 if fails else 0)
