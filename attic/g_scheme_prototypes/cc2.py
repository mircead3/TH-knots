"""Option A/C v2: commutation-reachable clasp-centering + bounded slack."""
import sys, os
_here=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[_here, os.path.dirname(_here)]
from direct2 import build_at, rots
from construct3 import columns_greedy
from itertools import product
import time

def clasps(g):
    n=len(g); runs=[]; i=0
    while i<n:
        j=i
        while j+1<n and g[j+1]==g[i]: j+=1
        if j>i: runs.append((i,j,g[i]))
        i=j+1
    return runs

def reach(g):
    n=len(g); rs=clasps(g); cand={}
    for t in range(n):
        s=g[t]
        for (a,b,gen) in rs:
            if a<=t<=b or abs(s-gen)<2: continue
            lo,hi=(t+1,a-1) if t<a else (b+1,t-1)
            if all(abs(g[k]-s)>=2 or abs(g[k]-gen)>=2 for k in range(lo,hi+1)):
                cand.setdefault(t,[]).append((a,b))
    return cand

def sched(g,slack,forced):
    n=len(g); early=[0]*n
    for t in range(n):
        for u in range(t):
            d=abs(g[u]-g[t])
            if d<=1: early[t]=max(early[t],early[u]+(2 if d==0 else 1))
    out=[]; x=[0]*n
    def rec(t):
        if t==n: out.append(x[:]); return
        if t in forced: x[t]=forced[t]; rec(t+1); return
        lo=0
        for u in range(t):
            d=abs(g[u]-g[t])
            if d<=1: lo=max(lo,x[u]+(2 if d==0 else 1))
        for xv in range(lo, early[t]+slack+1): x[t]=xv; rec(t+1)
    rec(0); return out

def construct_A(g,L,maxslack=3,tl=12.0):
    best=None; att=0; t0=time.time()
    for S in range(maxslack+1):
        for r in rots(g):
            gx=columns_greedy(r); cand=reach(r); idxs=list(cand.keys())
            opts=[]
            for t in idxs:
                o=[gx[t]]
                for (a,b) in cand[t]:
                    c=(gx[a]+gx[b])//2
                    far=gx[b]+1 if t<a else gx[a]-1
                    o+=[c-1,c,c+1,far]
                opts.append(sorted(set(o)))
            for combo in (product(*opts) if idxs else [()]):
                forced={idxs[j]:combo[j] for j in range(len(idxs))}
                for x in sched(r,S,forced):
                    if time.time()-t0>tl: return best,att,'timeout'
                    att+=1
                    res=build_at(r,L,x,range(0,1))
                    if res and (best is None or res[0]<best[0]): best=res
        if best: return best,att,f'slack={S}'
    return best,att,'fail'

if __name__=='__main__':
    import re
    fails=[]
    for line in open('/private/tmp/claude-501/-Users-mircea-claude-knots/74b4c845-ee6c-4de0-a816-0ceec42f8c5f/scratchpad/sweep.out'):
        m=re.search(r'FAIL: \((\d+), \[([\d, ]+)\]',line)
        if m: fails.append((int(m.group(1)),[int(x) for x in m.group(2).split(',')]))
    print(f"testing v2 on {len(fails)} prior failures",flush=True)
    rec=0; still=[]; atts=[]; t0=time.time()
    for i,(L,g) in enumerate(fails):
        r,att,st=construct_A(g,L); atts.append(att)
        if r: rec+=1
        else: still.append((L,g))
        if i%15==0: print(f"...{i}/{len(fails)} recovered={rec} still={len(still)} t={time.time()-t0:.0f}s",flush=True)
    atts.sort()
    print(f"DONE v2: RECOVERED {rec}/{len(fails)}  still={len(still)}  median_att={atts[len(atts)//2]} time={time.time()-t0:.0f}s")
    for s in still[:30]: print("  STILL:",s)
