import braids as b, enum_g as e
from itertools import product

def braid_key(g,L,Bc):
    W=[abs(x) for x in g]*Bc; seq=e.gauss_code(W,L)
    if len(seq)!=2*len(W): return None
    return b._canon_gauss([(seq[i],i%2) for i in range(len(seq))])
def minperiod(w):
    N=len(w)
    for P in range(1,N+1):
        if N%P==0 and all(w[i]==w[(i+P)%N] for i in range(N)): return w[:P]
    return w
def verify(runs,L,Bc):
    ups=runs[0::2];downs=runs[1::2]
    if len(ups)!=len(downs): return None
    t=len(ups);ZIG=[];x=0;y=0
    for i in range(t):
        ZIG.append((x,y));x+=ups[i];y+=ups[i];ZIG.append((x,y));x+=downs[i];y-=downs[i]
    PER=x
    if PER==0 or (PER*Bc)%L: return None
    mod=PER*Bc//L;bP=b.bight_path(ZIG,PER,2*t,mod,Bc)
    total=sum(abs(bP[(i+1)%len(bP)][1]-bP[i][1]) for i in range(len(bP)))
    if total%mod or total//mod!=L: return None
    crs,cum=b.crossings(bP,mod);word,_=b.braid_word(bP,mod,crs,cum,L)
    if b.num_components(word,L)!=1: return None
    return [abs(z) for z in minperiod(word)]

# ---- COLUMN MODEL ----
# Build column height-table H[0..W][s] for L strands. +-1 steps between columns.
# A coincidence (two strands equal height at a column) = a crossing.
# Constraint: multiset(H[W]) == multiset(H[0]) so tiles join.
# The multiset of crossings over one tile (interior + one boundary) must realize g.
# We VERIFY by tracing across tiles -> runs -> forward map == g.

def trace(H,W,L):
    # H: list length W+1, each a tuple of L heights (strand-indexed within a tile)
    # Follow physical strand across tiles. Return the up/down run sequence or None.
    # Build seq of heights by walking; strand continues to next tile via start-height match,
    # keeping slope through coincidences.
    start_h=H[0]
    # map start height -> list of (strand, first_slope)
    from collections import defaultdict
    starts=defaultdict(list)
    for s in range(L):
        slope0=H[1][s]-H[0][s]
        starts[H[0][s]].append((s,slope0))
    heights=[]  # full height walk
    s=0; guard=0
    arriving_slope=None
    while True:
        # walk strand s across this tile: append H[0..W-1][s]
        for x in range(W):
            heights.append(H[x][s])
        endh=H[W][s]
        arriving_slope=H[W][s]-H[W-1][s]
        # choose next strand starting at endh, continuing slope (through crossing) if possible
        cands=starts.get(endh)
        if not cands: return None
        if len(cands)==1:
            s=cands[0][0]
        else:
            # pick one whose first slope == arriving_slope (pass straight through crossing)
            same=[c for c in cands if c[1]==arriving_slope]
            if len(same)==1: s=same[0][0]
            elif len(same)>1: return None
            else:
                other=[c for c in cands if c[1]==-arriving_slope]
                if len(other)==1: s=other[0][0]
                else: return None
        guard+=1
        if guard>4*L+2: return None
        if s==0:
            break
    # heights is a cyclic +-1 walk; convert to runs
    n=len(heights)
    diffs=[heights[(i+1)%n]-heights[i] for i in range(n)]
    if any(d==0 or abs(d)!=1 for d in diffs): return None
    runs=[];cur=0;sign=0
    for d in diffs:
        sg=1 if d>0 else -1
        if sg==sign: cur+=1
        else:
            if sign!=0: runs.append(cur)
            cur=1;sign=sg
    runs.append(cur)
    # merge wrap if first and last same sign
    if len(runs)>=2 and sign==(1 if diffs[0]>0 else -1):
        runs[0]+=runs.pop()
    return runs

def build_and_check(g,L,Bc,W,hmax,limit=None):
    if limit is None: limit=[2_000_000]
    tgt=braid_key(g,L,Bc); K=len(g)
    # DFS: choose H[0] (heights, coincidences allowed), then each next col +-1 per strand.
    # prune by crossing count <= something; accept when multiset matches and trace verifies.
    found=[]
    def col_coincidences(col):
        # number of equal-height pairs
        from collections import Counter
        c=Counter(col); return sum(v*(v-1)//2 for v in c.values())
    # start heights: search assignments in [0,hmax], allow duplicates, canonical (min=0)
    # limit strands' start spread
    def dfs(x,H,ncross):
        if limit[0]<=0: return
        limit[0]-=1
        col=H[-1]
        if x>W:
            from collections import Counter
            if Counter(col)!=Counter(H[0]): return
            # crossings per tile = interior coincidences (x=1..W-1) + boundary (x=0==x=W counted once)
            r=trace(tuple(H),W,L)
            if r is None: return
            vr=verify(r,L,Bc)
            if vr is not None and braid_key(vr,L,Bc)==tgt:
                found.append(list(r)); 
            return
        # bound: remaining columns can add limited crossings; simple prune on total height
        for moves in product((-1,1),repeat=L):
            nc=tuple(col[s]+moves[s] for s in range(L))
            if any(v<0 or v>hmax for v in nc): continue
            add=0
            from collections import Counter
            cnt=Counter(nc)
            add=sum(v*(v-1)//2 for v in cnt.values())
            if ncross+add>K+2: continue  # allow a little slack (boundary double count)
            H.append(nc); dfs(x+1,H,ncross+add); H.pop()
            if found: return
    # enumerate start columns: L shifted copies of one strand => sorted band, adjacent gaps 0/1/2.
    from collections import Counter
    def gapbands(n):
        if n==1:
            yield (0,); return
        for pre in gapbands(n-1):
            for gp in (0,1,2):
                yield pre+(pre[-1]+gp,)
    for base in gapbands(L):
        c0=Counter(base); b0=sum(v*(v-1)//2 for v in c0.values())
        dfs(1,[base],b0)
        if found: return found[0]
    return None

def construct(g,L,Wmax=8,hmax=8):
    Bc=b.smallest_coprime_b(L)
    for W in range(2,Wmax+1):
        r=build_and_check(g,L,Bc,W,hmax)
        if r is not None: return ('runs',r,'W',W)
    return ('FAILED',)

if __name__=='__main__':
    print('s1s2 L3:',construct([1,2],3,Wmax=4,hmax=6))
