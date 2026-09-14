from math import gcd
from itertools import product

def perm_cycles(word, L):
    pos = list(range(L))
    for a in word:
        i = a-1; pos[i], pos[i+1] = pos[i+1], pos[i]
    seen=[False]*L; c=0
    for s in range(L):
        if not seen[s]:
            c+=1; k=s
            while not seen[k]: seen[k]=True; k=pos[k]
    return c

def gauss_code(word, L):
    """Gauss sequence (crossing ids in traversal order) of the closed braid."""
    M=len(word); slot=0; k=0; seq=[]
    for _ in range(L*M):
        s=word[k]-1
        if slot==s or slot==s+1:
            seq.append(k); slot = s+1 if slot==s else s
        k=(k+1)%M
    return seq  # length should be 2M for a single component

def interlacement_degrees(seq, M):
    posn={}   # crossing id -> [p,q]
    for idx,c in enumerate(seq): posn.setdefault(c,[]).append(idx)
    deg={}
    N2=len(seq)
    for c,(p,q) in posn.items():
        if p>q: p,q=q,p
        d=0
        for c2,(a,b) in posn.items():
            if c2==c: continue
            inside_a = p<a<q
            inside_b = p<b<q
            if inside_a != inside_b: d+=1
        deg[c]=d
    return deg

def smallest_coprime_ge2(L):
    B=2
    while gcd(L,B)!=1: B+=1
    return B

def canon_rot(word):
    return min(tuple(word[i:]+word[:i]) for i in range(len(word)))

if __name__=="__main__":
    LMAX={3:12,4:10,5:9,6:8}
    for L in range(3,7):
        N=smallest_coprime_ge2(L)
        seen=set()
        n_valid=0; alt_fail=[]; nugatory=[]
        for ell in range(L-1, LMAX[L]+1):
            for w in product(range(1,L), repeat=ell):
                cw=canon_rot(list(w))
                if cw in seen: continue
                seen.add(cw)
                g=list(cw)
                if perm_cycles(g,L)!=1: continue
                n_valid+=1
                word=g*N; M=len(word)
                seq=gauss_code(word,L)
                if len(seq)!=2*M: continue   # not single component (shouldn't happen)
                deg=interlacement_degrees(seq,M)
                if any(d%2==1 for d in deg.values()): alt_fail.append(g)
                if any(d==0 for d in deg.values()): nugatory.append(g)
        print(f"L={L} (closed at N={N}), step-words up to len {LMAX[L]}: {n_valid} with perm=1-cycle")
        print(f"   alternation FAILS (odd interlacement): {len(alt_fail)}"
              + (f"   e.g. {alt_fail[:3]}" if alt_fail else ""))
        print(f"   NUGATORY present (deg 0):            {len(nugatory)}"
              + (f"   e.g. {nugatory[:3]}" if nugatory else ""))
