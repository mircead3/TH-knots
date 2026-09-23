import construct4 as C, braids as b, gcatalog
from construct3 import columns_greedy, incidences
from construct4 import route, align_core, perm_succ
from collections import defaultdict
from itertools import product, combinations
import time

def valid_samegen_edges(g):
    """same-gen Δy=0 edges, only when no adjacent generator intervenes."""
    edges=[]; last={}
    for t,v in enumerate(g):
        k=abs(v)
        if k in last:
            between=[abs(g[j]) for j in range(last[k]+1,t)]
            if not any(abs(bk-k)==1 for bk in between):
                edges.append((last[k],t))
        last[k]=t
    return edges

def solve_components(g,L,x):
    """y determined by straight-seg equalities + valid same-gen edges; returns
       inc,thru, relY (within comp), comp map, list of component ids."""
    inc,es,cr=incidences(g,L,x); thru=lambda r:1 if r=='low' else -1
    adj=defaultdict(list)
    for s,lst in inc.items():
        for i in range(len(lst)-1):
            (xa,ra,ta),(xb,rb,tb)=lst[i],lst[i+1]
            if thru(ra)==thru(rb):
                sl=thru(ra); adj[ta].append((tb,sl*(xb-xa))); adj[tb].append((ta,-sl*(xb-xa)))
    for (a,c) in valid_samegen_edges(g):
        adj[a].append((c,0)); adj[c].append((a,0))
    n=len(g); relY={}; comp={}; nc=0
    for st in range(n):
        if st in comp: continue
        nc+=1; comp[st]=nc; relY[st]=0; stk=[st]
        while stk:
            a=stk.pop()
            for (bn,dy) in adj[a]:
                if bn not in comp: comp[bn]=relY  # placeholder
                if bn not in relY: comp[bn]=nc; relY[bn]=relY[a]+dy; stk.append(bn)
    # fix comp map (bug guard)
    comp={}; nc=0; relY={}
    for st in range(n):
        if st in comp: continue
        nc+=1; comp[st]=nc; relY[st]=0; stk=[st]
        while stk:
            a=stk.pop()
            for (bn,dy) in adj[a]:
                if bn not in comp: comp[bn]=nc; relY[bn]=relY[a]+dy; stk.append(bn)
    return inc,thru,relY,comp,sorted(set(comp.values()))

def build_at(g,L,x,offranges):
    inc,thru,relY,comp,comps=solve_components(g,L,x)
    Bc=b.smallest_coprime_b(L); tgt=C.pb4.braid_key([abs(z) for z in g],L,Bc)
    anchor=comp[0]; others=[c for c in comps if c!=anchor]
    ranges=[offranges]*len(others) if others else []
    best=None
    for offs in (product(*ranges) if others else [()]):
        off={anchor:0}; off.update(dict(zip(others,offs)))
        y={t:relY[t]+off[comp[t]] for t in range(len(g))}
        core={}; ok=True
        for s in range(L):
            lst=inc[s]; fx,fr,ft=lst[0]; lx,lr,lt=lst[-1]; mid={fx:y[ft]}
            for i in range(len(lst)-1):
                (xa,ra,ta),(xb,rb,tb)=lst[i],lst[i+1]
                seg=route(xa,y[ta],xb,y[tb],slope_first=thru(ra),slope_last=thru(rb))
                if seg is None: ok=False;break
                for k,hh in enumerate(seg): mid[xa+1+k]=hh
            if not ok: break
            core[s]=dict(fx=fx,fy=y[ft],entry=thru(fr),lx=lx,ly=y[lt],exit=thru(lr),mid=mid)
        if not ok: continue
        res=align_core(g,L,core,perm_succ(g,L),tgt,Bc,max(x))
        if res[0]=='OK':
            W=sum(res[1]['runs'])//L
            if best is None or W<best[0]: best=(W,res[1]['runs'])
    return best

def direct(g,L,maxk=3,maxmoved=None,offr=range(0,1),cap=8000,tl=4.0):
    """diagonal slides: pick a subset of crossings, push each x by k>=1 (y follows via solve).
       Search by increasing total slide budget so minimal W comes first."""
    gx=columns_greedy(g); n=len(g)
    if maxmoved is None: maxmoved=n
    t0=time.time(); att=0; best=None
    # search by number of moved crossings, then by slide amounts
    for nm in range(0,maxmoved+1):
        for moved in combinations(range(n),nm):
            for ks in product(range(1,maxk+1),repeat=nm):
                if time.time()-t0>tl or att>cap: return best,att,('timeout' if best is None else 'ok')
                x=gx[:]
                for j,t in enumerate(moved): x[t]=gx[t]+ks[j]
                att+=1
                r=build_at(g,L,x,offr)
                if r and (best is None or r[0]<best[0]): best=r
        if best: return best,att,'ok'
    return best,att,('fail' if best is None else 'ok')

def rots(g):
    seen=set();out=[]
    for i in range(len(g)):
        r=tuple(g[i:]+g[:i])
        if r not in seen: seen.add(r); out.append(list(r))
    return out

def construct(g,L,**kw):
    best=None; tot=0
    for r in rots(g):
        res,att,st=direct(r,L,**kw)
        tot+=att
        if res and (best is None or res[0]<best[0]): best=res
        if best and best[0]==2: break  # W=2 is global min, stop
    return best,tot

if __name__=='__main__':
    tests=[("hand4",[2,1,3,2,1,3,1],4),("L5a",[2,4,3,1,3,3],5),
           ("L5b",[2,4,4,1,3,4],5),("L5c",[3,2,4,3,1,4,2,4],5),
           ("s1^3",[1,3,1,2,4,1],5),("deep t2",[4,1,3,4,2,1,3,2],5)]
    for name,g,L in tests:
        r,att=construct(g,L,maxk=3,maxmoved=6)
        print(f"{name:9s} {('W'+str(r[0])+' '+str(r[1])) if r else 'FAIL':24s} attempts={att}")
