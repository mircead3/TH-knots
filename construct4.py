import braids as b, enum_g as e, pb4, time
from construct3 import columns_greedy, ranges, spacing, incidences, solve_y
from itertools import product
from collections import Counter

def valid_diagram(runs, L):
    """Reject degenerate multi-strand diagrams: triple points, or two of the L
       shifted strands overlapping along a segment (same height + same slope, or
       coincident at two consecutive columns).  The forward-map verify misses these."""
    if sum(runs[0::2]) != sum(runs[1::2]): return False   # strand must close (net zero)
    PER = sum(runs); W = PER // L
    h = []; s = []; y = 0
    for i, rn in enumerate(runs):
        sl = 1 if i % 2 == 0 else -1
        for _ in range(rn):
            h.append(y); s.append(sl); y += sl
    cross = set()
    for x in range(PER):
        col = [h[(x + k*W) % PER] for k in range(L)]
        if any(v >= 3 for v in Counter(col).values()): return False   # triple point
        for i in range(L):
            for j in range(i+1, L):
                if col[i] == col[j]:                                  # coincide at x
                    if s[(x + i*W) % PER] == s[(x + j*W) % PER]: return False  # same slope
                    if h[(x+1 + i*W) % PER] == h[(x+1 + j*W) % PER]: return False  # coincident edge
                    cross.add((x, col[i]))
    # every run (bight-to-bight segment) must carry >=1 crossing STRICTLY in its
    # interior (crossings at the bight endpoints don't count; a length-1 run has
    # no interior point, so it can never be validly crossed -> rejected).
    xacc = 0
    for rn in runs:
        if not any(((xacc + t) % PER, h[(xacc + t) % PER]) in cross for t in range(1, rn)):
            return False
        xacc += rn
    return True

def route(x0,y0,x1,y1,slope_first=None,slope_last=None):
    """heights at x0+1..x1 via <=1 bend, matching slope constraints; or None."""
    span=x1-x0
    if span<0: return None
    if span==0: return [] if y0==y1 else None
    if abs(y1-y0)>span or (span-abs(y1-y0))%2!=0: return None
    for sA in (1,-1):
        for u in range(span,-1,-1):        # prefer straight (u=span) first
            sB_needed = None
            rem=span-u
            # end: y0+sA*u+sB*rem=y1
            if rem==0:
                if y0+sA*u!=y1: continue
                fs=sA; ls=sA; hs=[]; cur=y0
                for _ in range(u): cur+=sA; hs.append(cur)
            else:
                if (y1-y0-sA*u)%rem!=0: continue
                sB=(y1-y0-sA*u)//rem
                if sB not in (1,-1): continue
                fs= sA if u>0 else sB; ls=sB
                hs=[]; cur=y0
                for _ in range(u): cur+=sA; hs.append(cur)
                for _ in range(rem): cur+=sB; hs.append(cur)
            if slope_first is not None and fs!=slope_first: continue
            if slope_last is not None and ls!=slope_last: continue
            return hs
    return None

def build_core(g,L,x):
    y,inc,endslots,cross=solve_y(g,L,x)
    if y is None: return None
    thru=lambda r:1 if r=='low' else -1
    core={}
    for s in range(L):
        lst=inc[s]
        if not lst: return None
        fx,fr,ft=lst[0]; lx,lr,lt=lst[-1]
        # middle heights fx..lx
        mid={fx:y[ft]}
        for i in range(len(lst)-1):
            (xa,ra,ta),(xb,rb,tb)=lst[i],lst[i+1]
            ya,yb=y[ta],y[tb]; sa,sb=thru(ra),thru(rb)
            seg=route(xa,ya,xb,yb,slope_first=sa,slope_last=sb)
            if seg is None: return None
            for k,hh in enumerate(seg): mid[xa+1+k]=hh
        core[s]=dict(fx=fx,fy=y[ft],entry=thru(fr),lx=lx,ly=y[lt],exit=thru(lr),mid=mid)
    # permutation successor: strand ending in slot k -> strand starting in slot k
    startslot={}
    order0=sorted(range(L),key=lambda s:(core[s]['mid'][core[s]['fx']] if core[s]['fx']==0 else None) if False else 0)
    return core,endslots,cross

def perm_succ(g,L):
    slots=list(range(L))
    for gi in g: slots[gi-1],slots[gi]=slots[gi],slots[gi-1]
    endslot={slots[k]:k for k in range(L)}        # strand -> end slot
    # start: identity slot k has strand k
    succ={}
    for s in range(L): succ[s]=endslot[s]          # strand s ends in slot endslot[s]=k -> continues as strand k
    return succ

def align_core(g,L,core,succ,tgt,Bc,xc,Wcap=None,Wfix=None):
    if Wfix is not None:
        Wrange=[Wfix] if Wfix>=xc else []
    else:
        Wstart=xc if xc%2==0 else xc+1
        Wend = Wstart+2*8 if Wcap is None else min(Wstart+2*8, Wcap)
        Wrange=range(Wstart, Wend, 2)
    for W in Wrange:
        # reachable starts/ends per strand
        Rs={}; Re={}
        for s in range(L):
            c=core[s]
            Rs[s]=[h for h in range(c['fy']-c['fx'], c['fy']+c['fx']+1)
                   if route(0,h,c['fx'],c['fy'],slope_last=c['entry']) is not None]
            Re[s]=[h for h in range(c['ly']-(W-c['lx']), c['ly']+(W-c['lx'])+1)
                   if route(c['lx'],c['ly'],W,h,slope_first=c['exit']) is not None]
        # link values: for each strand s, END[s]=START[succ[s]] in Re[s] cap Rs[succ]
        choicelists=[]
        okW=True
        for s in range(L):
            inter=sorted(set(Re[s]) & set(Rs[succ[s]]), key=lambda v:abs(v-core[s]['ly']))
            if not inter: okW=False; break
            choicelists.append((s,inter[:4]))
        if not okW: continue
        # bounded product over choices, verify
        for combo in product(*[cl[1] for cl in choicelists]):
            endval={choicelists[i][0]:combo[i] for i in range(L)}
            # build full H
            H=[[None]*L for _ in range(W+1)]
            good=True
            for s in range(L):
                c=core[s]
                left=route(0, None, c['fx'], c['fy'], slope_last=c['entry']) if False else None
                startval=endval[[k for k in range(L) if succ[k]==s][0]]  # START[s]=END[pred]
                lseg=route(0,startval,c['fx'],c['fy'],slope_last=c['entry'])
                if lseg is None: good=False;break
                rseg=route(c['lx'],c['ly'],W,endval[s],slope_first=c['exit'])
                if rseg is None: good=False;break
                H[0][s]=startval
                for k,hh in enumerate(lseg): H[k+1][s]=hh
                for xx,hh in c['mid'].items(): H[xx][s]=hh
                for k,hh in enumerate(rseg): H[c['lx']+1+k][s]=hh
            if not good: continue
            if any(v is None for row in H for v in row): continue
            HH=[tuple(r) for r in H]
            runs=pb4.trace(tuple(HH),W,L)
            if runs is None: continue
            if not valid_diagram(runs,L): continue     # cheap reject BEFORE the costly forward map
            vr=pb4.verify(runs,L,Bc)
            if vr is not None and pb4.braid_key(vr,L,Bc)==tgt:
                return ('OK',{'runs':runs,'W':W})
    return ('FAIL align',{'xc':xc})

def construct_one(g,L,slack=2,cap=2000,bound=None):
    """Search x-assignments in poset ranges; build_core + two-sided align; keep smallest W.
       bound=[best_W_so_far] shared mutable for pruning; final W>=max(x)."""
    Bc=b.smallest_coprime_b(L); tgt=pb4.braid_key([abs(z) for z in g],L,Bc)
    succ=perm_succ(g,L)
    early,late=ranges(g,slack); n=len(g)
    if bound is None: bound=[10**9]
    best=[None]; cnt=[0]
    def dfs(t,x,pmax):
        if cnt[0]>cap: return
        if pmax>=bound[0]: return                     # final W>=pmax; can't beat global best
        if t==n:
            cnt[0]+=1
            r=build_core(g,L,x)
            if r is None: return
            core,_,_=r
            res=align_core(g,L,core,succ,tgt,Bc,pmax,Wcap=bound[0])
            if res[0]=='OK':
                W=res[1]['W']
                if best[0] is None or W<best[0][0]: best[0]=(W,res[1]['runs'])
                if W<bound[0]: bound[0]=W
            return
        lo=early[t]
        for u in range(t):
            if abs(g[u]-g[t])<=1: lo=max(lo,x[u]+spacing(g[u],g[t]))
        for xv in range(lo,late[t]+1):
            dfs(t+1,x+[xv],max(pmax,xv))
    dfs(0,[],0)
    if best[0] is None: return ('FAIL',None)
    return ('OK',{'runs':best[0][1],'W':best[0][0]})

def try_at_W(g,L,Wtarget,cap=20000,deadline=None):
    """Search placements with all crossing x<=Wtarget; align exactly at Wtarget. First hit."""
    Bc=b.smallest_coprime_b(L); tgt=pb4.braid_key([abs(z) for z in g],L,Bc)
    succ=perm_succ(g,L)
    early,late=ranges(g,slack=Wtarget); n=len(g)     # allow spreading up to Wtarget
    late=[min(l,Wtarget) for l in late]
    hit=[None]; cnt=[0]; stop=[False]
    def dfs(t,x,pmax):
        if hit[0] is not None or cnt[0]>cap or stop[0]: return
        if pmax>Wtarget: return
        if t==n:
            cnt[0]+=1
            if deadline is not None and (cnt[0] & 1023)==0 and time.time()>deadline:
                stop[0]=True; return
            r=build_core(g,L,x)
            if r is None: return
            core,_,_=r
            res=align_core(g,L,core,succ,tgt,Bc,pmax,Wfix=Wtarget)
            if res[0]=='OK': hit[0]=res[1]['runs']
            return
        lo=early[t]
        for u in range(t):
            if abs(g[u]-g[t])<=1: lo=max(lo,x[u]+spacing(g[u],g[t]))
        for xv in range(lo,late[t]+1):
            dfs(t+1,x+[xv],max(pmax,xv))
            if hit[0] is not None: return
    dfs(0,[],0)
    return hit[0]

def construct_pb(g,L,Wmax=40,time_limit=8.0):
    """Place-and-bend search (fast, scales to large L; reports actual W = sum/L)."""
    deadline=time.time()+time_limit
    rots=[g[r:]+g[:r] for r in range(len(g))]
    for Wt in range(2,Wmax+1,2):
        for gr in rots:
            if time.time()>deadline: return ('FAIL timeout',None)
            runs=try_at_W(gr,L,Wt,deadline=deadline)
            if runs is not None:
                return ('OK',{'rotation':gr,'runs':runs,'W':sum(runs)//L})
    return ('FAIL all',None)

def _comps(total,k,lo=2):
    """Compositions of `total` into k ordered parts, each >= lo."""
    if k==1:
        if total>=lo: yield (total,)
        return
    for first in range(lo, total-lo*(k-1)+1):
        for rest in _comps(total-first,k-1,lo): yield (first,)+rest

def construct_brute(g,L,Wmax=16,time_limit=6.0):
    """Exhaustive over run-sequences by TRUE W (=sum/L). Up-runs and down-runs are
       enumerated as separate compositions of PER/2 (so sum(ups)=sum(downs): the
       strand closes) then interleaved. valid_diagram then verify. Minimal W first,
       thread 1 up. Complete for small L; grows with PER."""
    Bc=b.smallest_coprime_b(L); tgt=pb4.braid_key([abs(z) for z in g],L,Bc)
    deadline=time.time()+time_limit
    for W in range(2,Wmax+1,2):
        PER=W*L
        if PER%2: continue
        half=PER//2
        for t in range(1, half//2+1):        # t up-runs and t down-runs, each >= 2
            for ups in _comps(half,t,2):
                for downs in _comps(half,t,2):
                    if time.time()>deadline: return ('FAIL timeout',None)
                    runs=[]
                    for u,d in zip(ups,downs): runs += (u,d)
                    if not valid_diagram(runs,L): continue
                    try: vr=pb4.verify(runs,L,Bc)
                    except Exception: continue
                    if vr is not None and pb4.braid_key(vr,L,Bc)==tgt:
                        return ('OK',{'runs':runs,'W':W})
    return ('FAIL',None)

def construct_best(g,L,time_limit=3.0):
    """Minimal-W valid diagram. Brute (exact, fast) for small L; place-and-bend for large L."""
    if L<=5:
        return construct_brute(g,L,Wmax=16,time_limit=time_limit)
    return construct_pb(g,L,time_limit=time_limit)

if __name__=='__main__':
    tests=[([1,2],3),([1,2,3],4),([2,2,1,2],3),([2,3,1,3,3],4),([3,3,2,3,1],4)]
    for g,L in tests:
        print(f"g={g} L={L}: {construct_best(g,L)}")
