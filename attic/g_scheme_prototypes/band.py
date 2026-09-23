"""Band-layout constructor v0: DIRECT placement (no search).
Each generator's crossings form a band at its y-level (dx=2); commuting bands
share x-columns, adjacent bands go sequential; single-occurrence 'roamer'
generators center over the band they commute with, lifted on the diagonal."""
import sys, os
_here=os.path.dirname(os.path.abspath(__file__))
sys.path[:0]=[_here, os.path.dirname(_here)]
from direct2 import build_at, rots
import construct4 as C, braids as b

def bands_of(g):
    """generator -> sorted list of its crossing indices."""
    d={}
    for t,gen in enumerate(g): d.setdefault(gen,[]).append(t)
    return d

def band_place(g, L):
    """Return per-crossing (x list) computed directly from band structure.
    Bands (generators appearing >=2x) laid dx=2 on shared/sequential columns;
    roamers (appear 1x) centered over the commuting band they touch."""
    n=len(g); bd=bands_of(g)
    band_gens=[gen for gen,idx in bd.items() if len(idx)>=2]
    roam_gens=[gen for gen,idx in bd.items() if len(idx)==1]
    x=[None]*n
    # order bands left-to-right by first occurrence; commuting consecutive bands share columns
    band_gens_sorted=sorted(band_gens, key=lambda gen: bd[gen][0])
    col=0; group_start=0; prev=None
    band_cols={}   # gen -> starting column
    for gi,gen in enumerate(band_gens_sorted):
        if prev is not None and abs(gen-prev)>=2:
            start=band_cols[prev]        # commuting -> share columns with previous band
        else:
            start=col
        band_cols[gen]=start
        for k,t in enumerate(bd[gen]): x[t]=start+2*k
        col=max(col, start+2*len(bd[gen]))
        prev=gen
    return x, band_cols, col, band_gens_sorted, roam_gens, bd

def roamer_options(g, L):
    """direct clasp placement + per-roamer candidate x-positions {center, far, edge}."""
    x, band_cols, col, bgs, roam_gens, bd = band_place(g, L)
    opts={}   # roamer crossing index -> list of candidate x
    far=col
    for gen in roam_gens:
        t=bd[gen][0]; cands=set()
        for bg in bgs:
            if abs(bg-gen)>=2:
                idx=bd[bg]; xa=x[idx[0]]; xb=x[idx[-1]]
                cands.add((xa+xb)//2)          # center over this band
                cands.add(xb+1)                # far side of this band
        cands.add(far); far+=2                  # park at right edge
        opts[t]=sorted(cands)
    return x, opts

def construct_band(g,L):
    """direct clasp placement + tiny roamer {center,far,edge} choice, per rotation."""
    from itertools import product
    best=None; att=0
    for r in rots(g):
        try: x0,opts=roamer_options(r,L)
        except Exception: continue
        idxs=list(opts.keys())
        for combo in (product(*[opts[t] for t in idxs]) if idxs else [()]):
            x=x0[:]
            for j,t in enumerate(idxs): x[t]=combo[j]
            if any(v is None for v in x): continue
            att+=1
            res=build_at(r,L,x,range(-2,3))
            if res and (best is None or res[0]<best[0]): best=res
    return best, att

if __name__=='__main__':
    tests=[("#1",[1,1,1,1,1,2,3,4],5),("#2",[1,2,2,2,2,2,3,4],5),
           ("#3-3clasp",[1,1,1,2,2,2,3,3,3,4],5),("#4",[1,1,1,2,2,3,2,4],5),
           ("alt",[1,2,1,2,1,2,1,2,3,4],5),("deep7",[1,2,2,2,2,2,2,2,3,4],5),
           ("dt",[4,1,3,4,2,1,3,2],5)]
    for name,g,L in tests:
        r,att=construct_band(g,L)
        print(f"{name:10s} direct-band -> {str(r):40s} att={att}")
