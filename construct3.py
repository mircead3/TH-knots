import braids as b, enum_g as e, pb4
from collections import defaultdict

def columns_greedy(g):
    """x-position of each crossing via poset: dependent (|gen diff|<=1) preds push right;
       same gen -> +2, adjacent gen -> +1; commuting (>=2) can share x."""
    x=[0]*len(g)
    for t in range(len(g)):
        best=0
        for u in range(t):
            d=abs(g[u]-g[t])
            if d<=1:
                best=max(best, x[u]+(2 if d==0 else 1))
        x[t]=best
    return x

def incidences(g,L,x):
    """Return per-strand ordered [(x,y_placeholder,role,t)] and end-permutation slots."""
    slots=list(range(L))
    inc=defaultdict(list)   # strand -> list of (x, role, t)
    cross=[]                # t -> (lowstrand, highstrand, x)
    for t,gi in enumerate(g):
        low=slots[gi-1]; high=slots[gi]
        inc[low].append((x[t],'low',t)); inc[high].append((x[t],'high',t))
        cross.append((low,high,x[t]))
        slots[gi-1],slots[gi]=slots[gi],slots[gi-1]
    for s in inc: inc[s].sort()
    return inc,slots,cross

def solve_y(g,L,x):
    """Solve crossing y-values from straight-segment equalities. Returns dict t->y or None."""
    inc,endslots,cross=incidences(g,L,x)
    thru=lambda r: 1 if r=='low' else -1
    # equality edges between crossings sharing a straight strand segment
    adj=defaultdict(list)
    for s,lst in inc.items():
        for i in range(len(lst)-1):
            (xa,ra,ta),(xb,rb,tb)=lst[i],lst[i+1]
            if thru(ra)==thru(rb):            # straight segment -> equality
                slope=thru(ra)
                adj[ta].append((tb, slope*(xb-xa)))
                adj[tb].append((ta,-slope*(xb-xa)))
    y={}
    for start in range(len(g)):
        if start in y: continue
        y[start]=0; stack=[start]
        while stack:
            a=stack.pop()
            for (bnode,dy) in adj[a]:
                if bnode in y:
                    if y[bnode]!=y[a]+dy: return None,None,None,None  # conflict
                else:
                    y[bnode]=y[a]+dy; stack.append(bnode)
    return y,inc,endslots,cross

def build_H(g,L,x):
    y,inc,endslots,cross=solve_y(g,L,x)
    if y is None: return None
    thru=lambda r: 1 if r=='low' else -1
    xc=max(x) if x else 0
    # per strand build height over x=0..xc
    H=[[None]*L for _ in range(xc+1)]
    for s in range(L):
        lst=inc[s]
        if not lst:  # strand with no crossing: shouldn't happen for connected weave
            return None
        pts=[(cx, y[t]) for (cx,role,t) in lst]  # crossing points
        # left extension
        x0,y0=pts[0]; s0=thru(lst[0][1])
        cur=[]
        for xx in range(0,x0): cur.append((xx, y0 - s0*(x0-xx)))
        cur.append((x0,y0))
        # between crossings
        for i in range(len(lst)-1):
            (xa,ra,ta)=lst[i]; (xb,rb,tb)=lst[i+1]
            ya=y[ta]; yb=y[tb]; sa=thru(ra); sb=thru(rb)
            span=xb-xa
            if sa==sb:
                for k in range(1,span+1): cur.append((xa+k, ya+sa*k))
            else:
                # one turn: first slope sa then sb; solve turn position
                # ya + sa*u + sb*(span-u) = yb  => u*(sa-sb)= yb-ya - sb*span
                denom=(sa-sb)
                u=(yb-ya - sb*span)//denom
                if not (0<=u<=span) or ya+sa*u+sb*(span-u)!=yb: return None
                for k in range(1,u+1): cur.append((xa+k, ya+sa*k))
                for k in range(1,span-u+1): cur.append((xa+u+k, ya+sa*u+sb*k))
        # right extension to xc
        xk,yk=pts[-1]; sk=thru(lst[-1][1])
        for xx in range(xk+1,xc+1): cur.append((xx, yk+sk*(xx-xk)))
        for (xx,hh) in cur:
            if 0<=xx<=xc: H[xx][s]=hh
    if any(v is None for row in H for v in row): return None
    return [tuple(r) for r in H], endslots, xc

def align_and_verify(g,L):
    Bc=b.smallest_coprime_b(L); tgt=pb4.braid_key([abs(z) for z in g],L,Bc)
    x=columns_greedy(g)
    built=build_H(g,L,x)
    if built is None: return ('FAIL build',x)
    base,endslots,xc=built
    order0=sorted(range(L),key=lambda s:(base[0][s],s))
    targets={}
    for k in range(L): targets[endslots[k]]=base[0][order0[k]]
    Wstart=xc if xc%2==0 else xc+1
    for W in range(Wstart, Wstart+2*10, 2):
        ok=True; tails={}
        for st in range(L):
            hc=base[xc][st]; tg=targets[st]; span=W-xc; d=tg-hc
            if abs(d)>span or (span-d)%2!=0: ok=False;break
            up=(span+d)//2; dn=span-up; path=[]; cur=hc
            for _ in range(up): cur+=1;path.append(cur)
            for _ in range(dn): cur-=1;path.append(cur)
            tails[st]=path
        if not ok: continue
        HH=[tuple(c) for c in base]+[tuple(tails[st][j] for st in range(L)) for j in range(W-xc)]
        runs=pb4.trace(tuple(HH),W,L)
        if runs is None: continue
        vr=pb4.verify(runs,L,Bc)
        if vr is not None and pb4.braid_key(vr,L,Bc)==tgt:
            return ('OK',{'runs':runs,'W':W})
    return ('FAIL align',{'xc':xc,'x':x})

def spacing(ga,gb):
    d=abs(ga-gb); return 0 if d>1 else (2 if d==0 else 1)

def ranges(g,slack=0):
    n=len(g)
    early=[0]*n
    for t in range(n):
        for u in range(t):
            if abs(g[u]-g[t])<=1: early[t]=max(early[t],early[u]+spacing(g[u],g[t]))
    rc=[0]*n
    for t in range(n-1,-1,-1):
        for v in range(t+1,n):
            if abs(g[v]-g[t])<=1: rc[t]=max(rc[t],rc[v]+spacing(g[t],g[v]))
    total=max((early[t]+rc[t] for t in range(n)),default=0)+slack
    late=[total-rc[t] for t in range(n)]
    return early,late

def search_placements(g,L,slack=2,cap=4000):
    early,late=ranges(g,slack)
    n=len(g); results=[]; cnt=[0]
    xarr=[None]*n
    def dfs(t):
        if cnt[0]>cap: return
        if t==n:
            cnt[0]+=1
            built=build_H(g,L,list(xarr))
            if built is None: return
            res=align_and_verify_x(g,L,list(xarr),built)
            if res[0]=='OK': results.append((res[1]['W'],list(xarr),res[1]['runs']))
            return
        lo=early[t]
        for u in range(t):
            if abs(g[u]-g[t])<=1: lo=max(lo,xarr[u]+spacing(g[u],g[t]))
        for xv in range(lo,late[t]+1):
            xarr[t]=xv; dfs(t+1); xarr[t]=None
            if len(results)>40: return
    dfs(0)
    if not results: return None
    results.sort()
    return results[0]

def align_and_verify_x(g,L,x,built):
    Bc=b.smallest_coprime_b(L); tgt=pb4.braid_key([abs(z) for z in g],L,Bc)
    base,endslots,xc=built
    order0=sorted(range(L),key=lambda s:(base[0][s],s))
    targets={}
    for k in range(L): targets[endslots[k]]=base[0][order0[k]]
    Wstart=xc if xc%2==0 else xc+1
    for W in range(Wstart, Wstart+2*10, 2):
        ok=True; tails={}
        for st in range(L):
            hc=base[xc][st]; tg=targets[st]; span=W-xc; d=tg-hc
            if abs(d)>span or (span-d)%2!=0: ok=False;break
            up=(span+d)//2; dn=span-up; path=[]; cur=hc
            for _ in range(up): cur+=1;path.append(cur)
            for _ in range(dn): cur-=1;path.append(cur)
            tails[st]=path
        if not ok: continue
        HH=[tuple(c) for c in base]+[tuple(tails[st][j] for st in range(L)) for j in range(W-xc)]
        runs=pb4.trace(tuple(HH),W,L)
        if runs is None: continue
        vr=pb4.verify(runs,L,Bc)
        if vr is not None and pb4.braid_key(vr,L,Bc)==tgt:
            return ('OK',{'runs':runs,'W':W})
    return ('FAIL',None)

def construct_best(g,L):
    best=None
    for r in range(len(g)):
        gr=g[r:]+g[:r]
        res=search_placements(gr,L)
        if res is not None:
            W,xarr,runs=res
            if best is None or W<best[0]: best=(W,gr,runs)
    return ('OK',{'rotation':best[1],'runs':best[2],'W':best[0]}) if best else ('FAIL all',None)

if __name__=='__main__':
    tests=[([1,2],3),([2,1],3),([1,2,3],4),([1,2,3,4],5),([3,1,2],4)]
    for g,L in tests:
        print(f"g={g} L={L}: {construct_best(g,L)}")
