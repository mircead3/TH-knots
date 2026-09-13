#!/usr/bin/env python3
"""Port of index.html's zigzag knot core, plus a braid-word extractor.

Every one of these knots is drawn as slope-(+-1) runs with x strictly
increasing, x periodic mod = 2*C*B.  So the diagram IS a braid with the
circumference x as the time axis; the number of strands is
total_arclength / mod = LEADS.  We read the braid word off directly.
"""
from math import gcd
from fractions import Fraction

# ---- zigzag definitions (setKnotDef* in index.html) ----
def zigzag(C, m, params):
    if C == 1:
        ZIG = [(0, 0), (m, m)]
        return ZIG, 2*m, 2, m, m
    if C == 2:
        a, b = params
        ZIG = [(0,0),(m+a-b,m+a-b),(2*m-2*b,2*a),(3*m-a-b,m+a+b)]
        return ZIG, 4*m, 4, m, m+a+b
    if C == 3:
        a, b, c, d = params
        L = m
        ZIG = [(0,0),(3*L-c-d,3*L-c-d),(3*L+a-c-d,3*L-a-c-d),
               (3*L+a-c,3*L-a-c),(3*L+a+b-c,3*L-a-b-c),(3*L+a+b,3*L-a-b)]
        return ZIG, 6*m, 6, m, max(3*L-c-d,3*L-a-c,3*L-a-b)
    raise ValueError(C)

# ---- bight path (generateBights / the bP loop) ----
def bight_path(ZIG, PER, D, mod, B):
    bP = []
    seen = set()
    for k in range(B):
        for j in range(D):
            x, y = ZIG[j]
            bx = ((x + k*PER) % mod + mod) % mod
            dy = ZIG[(j+1) % D][1] - y
            nd = 1 if dy > 0 else (-1 if dy < 0 else 0)
            key = (bx, y, nd)
            if key in seen:
                return bP
            seen.add(key)
            bP.append((bx, y))
    return bP

# ---- crossings with over/under (buildCrossingList) ----
def crossings(bP, mod):
    n = len(bP)
    seen = {}
    for p in range(n):
        ax, ay = bP[p]
        dxp = bP[(p+1) % n][1] - ay          # >0 : ascending run
        if dxp <= 0:
            continue
        for q in range(n):
            cx, cy = bP[q]
            dxq = cy - bP[(q+1) % n][1]        # >0 : descending run
            if dxq <= 0:
                continue
            diffY = cy - ay
            for t in range(1, dxp):
                u = diffY - t
                if u <= 0 or u >= dxq:
                    continue
                if (ax + t) % mod != (cx + u) % mod:
                    continue
                key = ((ax+t) % mod, ay+t)
                if key not in seen:
                    seen[key] = dict(p=p, t=t, q=q, u=u,
                                     xphys=(ax+t) % mod, y=ay+t)
    lst = list(seen.values())
    cum = [0]
    for i in range(n):
        cum.append(cum[i] + abs(bP[(i+1) % n][1] - bP[i][1]))
    # alternating over/under along the strand
    enc = []
    for i, cr in enumerate(lst):
        enc.append((cum[cr['p']] + cr['t'], i, True))
        enc.append((cum[cr['q']] + cr['u'], i, False))
    enc.sort()
    rank = [0]*len(lst)
    for r, e in enumerate(enc):
        if e[2]:
            rank[e[1]] = r
    over = [rank[i] % 2 == 0 for i in range(len(lst))]      # pIsOver (ascending over)
    # parity flip so first-tied crossing is over for the later strand
    bestSeg, bestT, fio = float('inf'), float('inf'), True
    for i, cr in enumerate(lst):
        laterSeg = max(cr['p'], cr['q'])
        tOnLater = cr['t'] if cr['p'] > cr['q'] else cr['u']
        isOverLater = over[i] if cr['p'] > cr['q'] else (not over[i])
        if laterSeg < bestSeg or (laterSeg == bestSeg and tOnLater < bestT):
            bestSeg, bestT, fio = laterSeg, tOnLater, isOverLater
    if not fio:
        over = [not o for o in over]
    for i, cr in enumerate(lst):
        cr['pIsOver'] = over[i]
        cr['pPos'] = cum[cr['p']] + cr['t']
        cr['qPos'] = cum[cr['q']] + cr['u']
    return lst, cum

# ---- braid word: sweep physical x (one revolution), N = LEADS strands ----
def braid_word(bP, mod, crs, cum, LEADS):
    n = len(bP)
    total = cum[n]
    # y of every strand-arc at physical query X (half-integer avoids vertices)
    def arcs_at(X):
        ys = []
        for p in range(n):
            y0 = bP[p][1]
            dy = bP[(p+1) % n][1] - y0
            L = abs(dy)
            if L == 0:
                continue
            s = 1 if dy > 0 else -1
            x0 = cum[p]
            # unwrapped covers [x0, x0+L]; find every w with x0 <= X + w*mod <= x0+L
            import math
            w_lo = math.floor((x0 - X) / mod)
            w_hi = math.ceil((x0 + L - X) / mod)
            for ww in range(w_lo, w_hi + 1):
                xu = X + ww*mod
                if x0 <= xu <= x0 + L:
                    ys.append(y0 + s*(xu - x0))
        return ys
    word = []
    order = sorted(range(len(crs)), key=lambda i: (crs[i]['xphys'], crs[i]['y']))
    for i in order:
        cr = crs[i]
        X = cr['xphys'] - 0.5
        ys = arcs_at(X)
        y_asc = cr['y'] - 0.5     # ascending arc just before the crossing
        idx = sum(1 for yy in ys if yy < y_asc)   # 0-based rank of lower strand
        gen = idx + 1                              # sigma_gen, 1..N-1
        sign = -1 if cr['pIsOver'] else 1          # asc-over => negative, desc-over => positive
        word.append(sign*gen)
    return word, total

# ---- closure permutation (to count components) ----
def closure_perm(word, N):
    perm = list(range(N))          # perm[track] after applying word
    # apply generators left to right; sigma_i swaps strands at positions i-1,i
    pos = list(range(N))
    for g in word:
        i = abs(g) - 1
        pos[i], pos[i+1] = pos[i+1], pos[i]
    return pos

def num_components(word, N):
    pos = closure_perm(word, N)
    seen = [False]*N
    comps = 0
    for s in range(N):
        if not seen[s]:
            comps += 1
            j = s
            while not seen[j]:
                seen[j] = True
                j = pos[j]
    return comps

# ---- Alexander determinant from the diagram (ground truth) ----
def alexander_matrix(crs):
    n = len(crs)
    inst = []
    for k, cr in enumerate(crs):
        inst.append((cr['pPos'], k, not cr['pIsOver']))
        inst.append((cr['qPos'], k, cr['pIsOver']))
    inst.sort()
    arcIdx = 0
    underArc = [0]*n
    overArc = [0]*n
    for pos, k, isUnder in inst:
        if isUnder:
            underArc[k] = arcIdx
            arcIdx = (arcIdx + 1) % n
        else:
            overArc[k] = arcIdx
    M = [[[0, 0] for _ in range(n)] for _ in range(n)]
    for k in range(n):
        a = underArc[k]; b = (a+1) % n; c = overArc[k]
        if not crs[k]['pIsOver']:   # positive
            M[k][a][0] += 0; M[k][a][1] += -1
            M[k][b][0] += 1
            M[k][c][0] += -1; M[k][c][1] += 1
        else:                        # negative
            M[k][a][0] += 1
            M[k][b][1] += -1
            M[k][c][0] += -1; M[k][c][1] += 1
    return M

def det_at(M, t):
    n = len(M); sz = n-1
    if sz <= 0:
        return 1
    A = [[M[i][j][0] + M[i][j][1]*t for j in range(sz)] for i in range(sz)]
    prev = 1
    for k in range(sz-1):
        if A[k][k] == 0:
            sw = -1
            for i in range(k+1, sz):
                if A[i][k] != 0:
                    sw = i; break
            if sw < 0:
                return 0
            A[k], A[sw] = A[sw], A[k]
            prev = -prev
        for i in range(k+1, sz):
            for j in range(k+1, sz):
                A[i][j] = (A[i][j]*A[k][k] - A[i][k]*A[k][j]) // prev
        prev = A[k][k]
    return A[sz-1][sz-1]

def determinant(crs):
    if len(crs) == 0:
        return 1
    return abs(det_at(alexander_matrix(crs), -1))

# ---- convenience: full analysis of one (C,m,params,B) ----
def analyze(C, m, params, B):
    ZIG, PER, D, LEADS, MAXY = zigzag(C, m, params)
    mod = 2*C*B
    bP = bight_path(ZIG, PER, D, mod, B)
    crs, cum = crossings(bP, mod)
    word, total = braid_word(bP, mod, crs, cum, LEADS)
    comps = num_components(word, LEADS)
    det = determinant(crs) if comps == 1 else None
    return dict(C=C, m=m, params=params, B=B, LEADS=LEADS,
                ncross=len(crs), word=word, strands=LEADS,
                strand_sheets=Fraction(total, mod), comps=comps, det=det)

# ---- valid parameter enumeration (validPairs / computeValidC3) ----
def valid_pairs(m):
    pairs = []
    for a in range(0, m-3+1):
        b = a
        while a + b <= m-3:
            if (m-a-b) % 2 == 1 and not (a == 0 and b == 0):
                pairs.append((a, b))
            b += 1
    return pairs

def compute_valid_c3(m):
    L3 = 3*m
    lst = []
    for a in range(2, L3-4+1):
        if a % 3 == 0:
            continue
        bRem = a % 3
        for b in range(2, L3-a-2+1):
            if b % 3 == (3-bRem) % 3:
                continue
            for c in range(2, min(L3-a-b, a+b-2)+1):
                if c % 3 == 0:
                    continue
                dMax = min(L3-a-c, a+b-c)
                for d in range(2, dMax+1):
                    if d % 3 == 0:
                        continue
                    if (b % 3 == 0 or (c+d) % 3 == 0) and c+d > b:
                        continue
                    if a+b == c+d and b < d:
                        continue
                    if a+c+d == L3 and not ((a > d and b >= c) or (a == d and b > c and a == b)):
                        continue
                    if a+b+c == L3 and a < d:
                        continue
                    seg = [L3-c-d, a, d, b, c, L3-a-b]
                    if c+d > b or 2 not in seg:
                        lst.append((a, b, c, d))
    return lst

def smallest_coprime_b(m):
    B = 2
    while gcd(m, B) != 1:
        B += 1
    return B

# ---- canonical over/under Gauss code (mirror+reversal+rotation folded) ----
def _booth_least_rotation(S):
    N = len(S)
    f = [-1]*(2*N)
    k = 0
    for j in range(1, 2*N):
        sj = S[j % N]
        i = f[j-k-1]
        while i != -1 and sj != S[(k+i+1) % N]:
            if sj < S[(k+i+1) % N]:
                k = j-i-1
            i = f[i]
        if sj != S[(k+i+1) % N]:
            if sj < S[k % N]:
                k = j
            f[j-k] = -1
        else:
            f[j-k] = i+1
    return k

def _canon_gauss(seq):
    N = len(seq)
    def tokens(base):
        partner = [0]*N
        posMap = {}
        for i in range(N):
            lab = base[i][0]
            if lab in posMap:
                jj = posMap[lab]; partner[i] = jj; partner[jj] = i
            else:
                posMap[lab] = i
        return [((partner[i]-i) % N)*2 + base[i][1] for i in range(N)]
    def keyOf(t):
        k = _booth_least_rotation(t)
        return tuple(t[(k+i) % N] for i in range(N))
    rev = list(reversed(seq))
    best = None
    for base in (seq, rev):
        t0 = tokens(base)
        tm = tokens([(lab, o ^ 1) for lab, o in base])
        for t in (t0, tm):
            kk = keyOf(t)
            if best is None or kk < best:
                best = kk
    return best

def gauss_seq(crs):
    inst = []
    for i, cr in enumerate(crs):
        inst.append((cr['pPos'], i, 1 if cr['pIsOver'] else 0))
        inst.append((cr['qPos'], i, 0 if cr['pIsOver'] else 1))
    inst.sort()
    return [(i, o) for _, i, o in inst]

def gauss_key(C, m, params, B):
    ZIG, PER, D, LEADS, MAXY = zigzag(C, m, params)
    mod = 2*C*B
    bP = bight_path(ZIG, PER, D, mod, B)
    crs, cum = crossings(bP, mod)
    return _canon_gauss(gauss_seq(crs))

def dedup_by_knot(C, m, tuples):
    """One representative (a,b,c,d) per distinct knot (mirror folded), via the
    canonical over/under Gauss code at the smallest coprime B (single component)."""
    B = smallest_coprime_b(m)
    groups = {}
    for t in tuples:
        params = t if C != 1 else None
        k = gauss_key(C, m, params, B)
        groups.setdefault(k, []).append(t)
    # representative: lexicographically smallest tuple in each group
    reps = [min(g) for g in groups.values()]
    reps.sort()
    return reps, B


# ---- flip symmetry (app's isSym) ----
def _c3_canon_key(seq):
    def even_rots(a):
        return [(a[0],a[1],a[2],a[3],a[4],a[5]),
                (a[2],a[3],a[4],a[5],a[0],a[1]),
                (a[4],a[5],a[0],a[1],a[2],a[3])]
    rev = (seq[5],seq[4],seq[3],seq[2],seq[1],seq[0])
    return min(','.join(map(str, r)) for r in (even_rots(seq)+even_rots(rev)))

def _c3_seq(a,b,c,d,m):     L3=3*m; return (L3-c-d,a,d,b,c,L3-a-b)
def _c3_flip_seq(a,b,c,d,m):L3=3*m; return (a,d,b,c,L3-a-b,L3-c-d)

def is_symmetric(C, m, params):
    if C == 1:
        return True
    if C == 2:
        a, b = params
        return a == b
    a, b, c, d = params
    return _c3_canon_key(_c3_seq(a,b,c,d,m)) == _c3_canon_key(_c3_flip_seq(a,b,c,d,m))

def generate_csv(path, Lmax=13):
    import csv
    rows = []
    # C=1: L=2..Lmax, at smallest coprime B (minimal knot representative)
    for C in (1, 2, 3):
        Ls = range(2, Lmax+1) if C == 1 else (range(4, Lmax+1) if C == 2 else range(3, Lmax+1))
        for m in Ls:
            if C == 1:
                tuples = [None]
            else:
                raw = valid_pairs(m) if C == 2 else compute_valid_c3(m)
                if not raw:
                    continue
                tuples, _ = dedup_by_knot(C, m, raw)
            B = smallest_coprime_b(m)
            for t in tuples:
                params = t
                r = analyze(C, m, params, B)
                if r['comps'] != 1:
                    continue
                g = (r['ncross'] - r['LEADS'] + 1)//2
                a,b,c,d = (params if C==3 else (params[0],params[1],'','') if C==2 else ('','','',''))
                rows.append(dict(C=C, L=m, a=a, b=b, c=c, d=d, B=B,
                                 strands=r['LEADS'], crossings=r['ncross'],
                                 determinant=r['det'], genus=g,
                                 symmetric=is_symmetric(C,m,params),
                                 braid_word=word_compact(r['word'])))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def word_str(word):
    def one(g):
        i = abs(g)
        return f"s{i}" if g > 0 else f"S{i}"   # s=sigma, S=sigma^-1
    return " ".join(one(g) for g in word)

def word_compact(word):
    # group consecutive identical generators as g^k
    out = []
    i = 0
    while i < len(word):
        j = i
        while j < len(word) and word[j] == word[i]:
            j += 1
        g = word[i]; k = j-i
        base = f"s{abs(g)}" if g > 0 else f"s{abs(g)}^-1"
        out.append(base if k == 1 else f"{base}^{k}" if g>0 else f"s{abs(g)}^-{k}")
        i = j
    return " ".join(out)


if __name__ == '__main__':
    print("=== C=1 verification (analytic: (s1 S2 s3 ...)^B on L strands) ===")
    known = {(2,3):3, (3,2):5, (3,4):45, (3,5):121, (2,5):5, (2,7):7}
    for (L, B), det_known in sorted(known.items()):
        r = analyze(1, L, None, B)
        ok = "OK" if r['det'] == det_known else f"MISMATCH(app-known={det_known})"
        print(f"L={L} B={B}: strands={r['strands']} sheets={r['strand_sheets']} "
              f"cross={r['ncross']} comps={r['comps']} det={r['det']} [{ok}]")
        print(f"   word: {word_str(r['word'])}")
