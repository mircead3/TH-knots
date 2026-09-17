"""Option C: hybrid constructor = clasp-centering candidates x bounded slack DFS.
Centering supplies the big diagonal slide so the slack search stays shallow."""
import sys, os
_here=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[_here, os.path.dirname(_here)]
from direct2 import build_at, rots
from construct3 import columns_greedy
from itertools import product
import time

def clasp_roamers(g):
    """maximal same-gen clasps (len>=2) + roamer indices (in the commuting block
       adjacent to a clasp) -> the clasp x-range they may center over."""
    n=len(g); runs=[]; i=0
    while i<n:
        j=i
        while j+1<n and g[j+1]==g[i]: j+=1
        if j>i: runs.append((i,j,g[i]))
        i=j+1
    roam={}
    for (a,b,gen) in runs:
        k=a-1
        while k>=0 and abs(g[k]-gen)>=2: roam.setdefault(k,(a,b)); k-=1
        k=b+1
        while k<n and abs(g[k]-gen)>=2: roam.setdefault(k,(a,b)); k+=1
    return roam

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

def construct_C(g,L,maxslack=4,tl=12.0):
    best=None; att=0; t0=time.time()
    for S in range(maxslack+1):
        for r in rots(g):
            gx=columns_greedy(r); roam=clasp_roamers(r); idxs=list(roam.keys())
            opts=[]
            for t in idxs:
                a,b=roam[t]; c=(gx[a]+gx[b])//2
                opts.append([gx[t], c-1, c, c+1])   # edge + center-1/center/center+1
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
    import re, construct4 as C
    fails=[]
    for line in open('/private/tmp/claude-501/-Users-mircea-claude-knots/74b4c845-ee6c-4de0-a816-0ceec42f8c5f/scratchpad/sweep.out'):
        m=re.search(r'FAIL: \((\d+), \[([\d, ]+)\]',line)
        if m: fails.append((int(m.group(1)),[int(x) for x in m.group(2).split(',')]))
    print(f"testing C on {len(fails)} previous failures (tl=12s each)",flush=True)
    rec=0; still=0; atts=[]; t0=time.time()
    for i,(L,g) in enumerate(fails):
        r,att,st=construct_C(g,L); atts.append(att)
        if r: rec+=1
        else: still+=1
        if i%15==0: print(f"...{i}/{len(fails)} recovered={rec} still={still} elapsed={time.time()-t0:.0f}s",flush=True)
    atts.sort()
    print(f"DONE: RECOVERED {rec}/{len(fails)}  still-failing {still}  median_att={atts[len(atts)//2]} time={time.time()-t0:.0f}s")
