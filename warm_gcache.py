"""Pre-warm the persistent g-cache so no level or knot is ever slow in the UI.
Populates the same file the server reads.  Run once after changing gcatalog/g_solve."""
import sys, os, json, time, hashlib
ROOT='/Users/mircea/claude/knots'
sys.path[:0]=[ROOT]
import gcatalog as GC

def sig():
    h=hashlib.sha256()
    for n in ('gcatalog.py','g_solve.py'):
        h.update(open(os.path.join(ROOT,n),'rb').read())
    return h.hexdigest()[:16]

# Levels to pre-warm.  Keep to the region that is complete AND quick: the cap in
# enumerate_gs refuses anything over 20M candidates, and |g|=12 at L=5 costs ~113s to
# enumerate plus ~2700 builds, so it is opt-in via argv rather than warmed by default.
import sys as _sys
MAXGLEN=int(_sys.argv[1]) if len(_sys.argv)>1 else 10
LEVELS=[(L,gl) for L in (3,4,5) for gl in range(L-1,MAXGLEN+1) if gl%2==(L-1)%2]
enum={}; build={}; t0=time.time()
for L,gl in LEVELS:
    ts=time.time(); gs=GC.enumerate_gs(L,gl); e=time.time()-ts
    enum['%d|%d'%(L,gl)]=gs
    bt=time.time(); slow=[]
    for g in gs:
        s0=time.time(); bd=GC.build(L,g); d=time.time()-s0
        if bd is not None:
            build['%d|%s'%(L,','.join(map(str,g)))]=bd
            if d>1.0: slow.append((round(d,1),g))
    print('  L=%d |g|=%2d  %4d knots   enum %5.1fs  build %5.1fs  slow:%s'
          %(L,gl,len(gs),e,time.time()-bt,slow if slow else '-'),flush=True)
path=os.path.join(ROOT,'gcache.json')
json.dump({'sig':sig(),'enum':enum,'build':build},open(path,'w'))
print('wrote %s  (%d levels, %d diagrams, %.1f KB)  total %.0fs'
      %(path,len(enum),len(build),os.path.getsize(path)/1024,time.time()-t0))
