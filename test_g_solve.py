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
try:
    GC.is_knot_power([1,2,3,4,5,6,1,2,3,4,5,6,2,1,4,3,6,5], cap=500)
    ok(False, 'cap overflow must RAISE, not return False')
except RuntimeError:
    ok(True,  'cap overflow raises rather than silently keeping an h^k knot')

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
