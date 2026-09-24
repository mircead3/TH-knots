#!/usr/bin/env python3
"""Minimal local HTTP server exposing relax_general() to the browser.

Why this exists: a webpage cannot launch a local program by itself, so the
only way for index.html's "Relax" button to use the actual, validated
elastic.relax model (instead of the browser's simplified JS approximation)
is to POST the current knot to a small server running that model. This is
that server. It is deliberately tiny — stdlib only (http.server), no
Flask/aiohttp/etc — since this machine has no extra Python packages
installed beyond numpy/scipy.

Job model: relaxation can take a minute or more, and the browser wants to
(a) show the shape evolving and (b) on Cancel, keep whatever was achieved
so far. So a POST /relax starts the relaxation in a BACKGROUND THREAD and
returns immediately; the browser then polls /relax/poll for the current
shape and calls /relax/stop to cancel. Only ONE job runs at a time (this
is a single-user local tool and the browser only relaxes one knot at
once), so a single global CURRENT job is enough — no job ids. Starting a
new job cooperatively stops any previous one first.

Usage:
  python3 elastic/server.py [port]      # default port 8731
Leave it running in a terminal, then use the Relax button in index.html
(the button shows a clear error if it can't reach the server).

Endpoints (all POST, all reply application/json, all CORS-open):
  /relax   {"loops": [[[x,y,z],...],...], "r": <wire radius>}
           -> {"ok": true}  (starts the job; loops are in the SAME units as
           r — exactly what index.html's getKnotPointsForRelax() produces)
  /relax/poll  {}
           -> {"running", "loops", "energy", "iter", "converged",
               "stopped", "error"}  (loops = latest in-progress or final
           shape, or null if none yet)
  /relax/stop  {}
           -> {"ok": true}  (asks the running job to stop; its next poll
           will report running=false with the achieved-so-far shape)

See relax_general()'s docstring in relax.py for the physics/scope (bending
+ inextensibility + hard contact; no twist; determinant-verified for a
single closed loop, unverified for links).
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Optional single-threaded BLAS for reproducibility: multi-threaded reductions
# can differ run-to-run at the last bit, which a numerically unstable relaxation
# could amplify into visibly different shapes. Set BEFORE importing numpy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from relax import relax_general

# g-based knot enumeration/construction lives in the repo root (one level up).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gcatalog

# ---- persistent g-catalog cache ------------------------------------------------
# Enumeration and construction are both pure functions of (gcatalog, g_solve), and
# both are slow the first time: enumerate is ~4s at L=5 |g|=10 and ~58s at |g|=12,
# and one knot ([1,1,1,2,1,3,2,4,4,4], L=5) costs ~12s because tier 3 has to prove
# W=8 impossible.  In-memory caching made each cost once per SERVER LIFETIME, which
# still means paying it after every restart.  Persisting to disk makes it once ever.
#
# The file is keyed by a signature over the two source files, so any change to the
# enumeration or the solver invalidates it rather than serving stale results.
_ENUM_CACHE = {}    # (L, glen) -> list of canonical g's
_BUILD_CACHE = {}   # (L, g)    -> built diagram; see _handle_construct
_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'gcache.json')
_CACHE_DIRTY = False

def _cache_signature():
    import hashlib
    h = hashlib.sha256()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ('gcatalog.py', 'g_solve.py'):
        with open(os.path.join(root, name), 'rb') as f: h.update(f.read())
    return h.hexdigest()[:16]

def _cache_load():
    try:
        with open(_CACHE_FILE) as f: d = json.load(f)
    except Exception:
        return
    if d.get('sig') != _cache_signature():
        print('g-cache: signature changed (gcatalog/g_solve edited) -- ignoring stale cache')
        return
    for k, v in d.get('enum', {}).items():
        L, glen = k.split('|'); _ENUM_CACHE[(int(L), int(glen))] = v
    for k, v in d.get('build', {}).items():
        L, gs = k.split('|'); _BUILD_CACHE[(int(L), tuple(int(x) for x in gs.split(',')))] = v
    print('g-cache: loaded %d levels, %d diagrams from %s'
          % (len(_ENUM_CACHE), len(_BUILD_CACHE), os.path.basename(_CACHE_FILE)))

def _cache_save():
    global _CACHE_DIRTY
    if not _CACHE_DIRTY: return
    tmp = _CACHE_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump({'sig': _cache_signature(),
                       'enum': {'%d|%d' % k: v for k, v in _ENUM_CACHE.items()},
                       'build': {'%d|%s' % (k[0], ','.join(map(str, k[1]))): v
                                 for k, v in _BUILD_CACHE.items()}}, f)
        os.replace(tmp, _CACHE_FILE)      # atomic: never leave a half-written cache
        _CACHE_DIRTY = False
    except Exception as e:
        print('g-cache: save failed: %s' % e)

# ---- streaming, interruptible enumeration --------------------------------------
# The UI browses knots from index 1 upward, so it does not need the final count before
# showing anything: the first knot of L=6 |g|=11 arrives in 0.07s where the full scan
# takes 232s.  One job at a time, superseded by generation number, so changing L or |g|
# abandons the previous scan instead of queueing behind it.
_ENUM_JOB = {'gen': 0, 'L': None, 'glen': None, 'gs': [], 'done': True,
             'truncated': False, 'error': None}
_enum_lock = threading.RLock()   # RLock: _enum_start snapshots while holding it

def _enum_worker(gen, L, glen):
    try:
        for g in gcatalog.iter_gs(L, glen):
            with _enum_lock:
                if _ENUM_JOB['gen'] != gen: return      # superseded: drop this scan
                _ENUM_JOB['gs'].append(g)
        with _enum_lock:
            if _ENUM_JOB['gen'] != gen: return
            _ENUM_JOB['done'] = True
            _ENUM_JOB['truncated'] = False        # no knot cap: scans run to completion
            _ENUM_CACHE[(L, glen)] = list(_ENUM_JOB['gs'])   # complete: worth caching
        global _CACHE_DIRTY
        _CACHE_DIRTY = True; _cache_save()
    except Exception as ex:
        with _enum_lock:
            if _ENUM_JOB['gen'] == gen:
                _ENUM_JOB['error'] = str(ex); _ENUM_JOB['done'] = True

def _enum_start(L, glen):
    """Begin (or join) the scan for this level.  -> snapshot dict."""
    with _enum_lock:
        if _ENUM_JOB['L'] == L and _ENUM_JOB['glen'] == glen and _ENUM_JOB['error'] is None:
            return _enum_snapshot()                    # already scanning this level
        _ENUM_JOB.update(gen=_ENUM_JOB['gen'] + 1, L=L, glen=glen, gs=[],
                         done=False, truncated=False, error=None)
        gen = _ENUM_JOB['gen']
    threading.Thread(target=_enum_worker, args=(gen, L, glen), daemon=True).start()
    return _enum_snapshot()

def _enum_snapshot():
    with _enum_lock:
        return {'L': _ENUM_JOB['L'], 'glen': _ENUM_JOB['glen'],
                'gs': list(_ENUM_JOB['gs']), 'count': len(_ENUM_JOB['gs']),
                'done': _ENUM_JOB['done'], 'truncated': _ENUM_JOB['truncated'],
                'error': _ENUM_JOB['error']}

DEFAULT_PORT = 8731
MAX_STEPS = 60000          # safety cap so a pathological request can't hang forever
MAX_TOTAL_POINTS = 4000    # sanity cap on request size (sum of all loops' input points)

# NOTE: relax_continuation (thickness annealing) was tried and removed — the
# real issue for these knots isn't speed but that the browser's cylinder
# initial shape collapses to a balled-up local minimum for thin wire (only
# thick wire's contact holds it open into the flat coil). That needs a better
# (flatter/coil) initial condition, not annealing. This path is stable
# momentum + shape-convergence.
POINTS_PER_DIAM = 3.0
POINT_BUDGET_MIN = 400
POINT_BUDGET_MAX = 3000

def choose_point_budget(loops0, r):
    L = sum(float(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum()) for P in loops0)
    lam = L / (2 * r)
    return int(min(POINT_BUDGET_MAX, max(POINT_BUDGET_MIN, round(POINTS_PER_DIAM * lam))))

# ---- single global job, guarded by a lock ----
# CURRENT holds the state of the one relaxation that is running or last ran.
# The worker thread writes 'loops'/'energy'/'iter' as it progresses (via the
# on_progress callback) and 'running'/'converged'/'stopped'/'error' when it
# finishes; the HTTP handlers read it for /relax/poll and set 'stop' for
# /relax/stop. `gen` is a monotonically increasing job counter so a stale
# worker (from a superseded job) can tell it's no longer the current one and
# avoid clobbering the new job's state.
_lock = threading.Lock()
CURRENT = {
    'gen': 0, 'running': False, 'stop': False,
    'loops': None, 'energy': None, 'iter': 0,
    'converged': False, 'stopped': False, 'error': None,
}

def _run_job(gen, loops0, r):
    """Worker thread body: run relax_continuation with progress + stop hooks
    that write into CURRENT (only while this job is still the current one)."""
    def on_progress(step, energy, loops):
        with _lock:
            if CURRENT['gen'] != gen:
                return
            CURRENT['iter'] = step
            CURRENT['energy'] = float(energy)
            # snapshot the live (mutated-in-place) arrays as plain lists
            CURRENT['loops'] = [np.asarray(P).round(6).tolist() for P in loops]

    def should_stop():
        with _lock:
            return CURRENT['gen'] != gen or CURRENT['stop']

    try:
        res = relax_general(loops0, r=r, steps=MAX_STEPS,
                            point_budget=choose_point_budget(loops0, r),
                            on_progress=on_progress, should_stop=should_stop)
        with _lock:
            if CURRENT['gen'] != gen:
                return
            CURRENT['loops'] = res['loops']
            CURRENT['energy'] = res['E_bend']
            CURRENT['iter'] = res['steps_run']
            CURRENT['converged'] = res['converged_at'] is not None
            CURRENT['stopped'] = res.get('stopped', False)
            CURRENT['running'] = False
    except Exception as e:
        with _lock:
            if CURRENT['gen'] != gen:
                return
            CURRENT['error'] = f'relaxation failed: {e}'
            CURRENT['running'] = False

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # silence default per-request stderr logging

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        # Wide-open CORS: a local, single-user, loopback-only tool loaded
        # from file:// (origin "null"); no credentials are ever sent.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length else b'{}'
        return json.loads(raw) if raw.strip() else {}

    def do_POST(self):
        if self.path == '/relax':
            self._handle_start()
        elif self.path == '/relax/poll':
            self._handle_poll()
        elif self.path == '/relax/stop':
            self._handle_stop()
        elif self.path == '/enumerate':
            self._handle_enumerate()
        elif self.path == '/enumerate/poll':
            self._handle_enumerate_poll()
        elif self.path == '/construct':
            self._handle_construct()
        else:
            self._send_json(404, {'error': f'no such endpoint: {self.path}'})

    def _handle_enumerate(self):
        try:
            body = self._read_body()
            L = int(body['L']); glen = int(body['glen'])
            # The APP owns the usable cap (the leads input's max); this is only a sanity net
            # against a pathological request -- _candidate_words loops over L-1 values per
            # position, so an absurd L would hang rather than answer.
            if not (3 <= L <= 64): raise ValueError('L out of range (3..64)')
            # Same rule: the app decides what |g| is worth offering.  This is only a net
            # against an absurd request (recursion depth in _candidate_words is |g|).
            if not (1 <= glen <= 64): raise ValueError('|g| out of range (1..64)')
        except Exception as e:
            self._send_json(400, {'error': f'invalid request: {e}'}); return
        # parity: a single L-cycle needs glen == (L-1) mod 2
        if glen % 2 != (L - 1) % 2:
            self._send_json(200, {'L': L, 'glen': glen, 'gs': [], 'count': 0}); return
        key = (L, glen)
        if key in _ENUM_CACHE:                     # complete and cached: answer at once
            gs = _ENUM_CACHE[key]
            self._send_json(200, {'L': L, 'glen': glen, 'gs': gs, 'count': len(gs),
                                  'done': True, 'truncated': False})
            return
        snap = _enum_start(L, glen)                # starts, or supersedes another level
        # Give a fast level the chance to finish inside this request rather than making
        # the client poll for something that takes 20ms.
        for _ in range(12):
            if snap['done'] or snap['count']: break
            time.sleep(0.025); snap = _enum_snapshot()
        self._send_json(200, snap)

    def _handle_enumerate_poll(self):
        """Current state of the running scan; the client extends its list from this."""
        self._send_json(200, _enum_snapshot())

    def _handle_construct(self):
        try:
            body = self._read_body()
            L = int(body['L']); g = [int(x) for x in body['g']]
            # The APP owns the usable cap (the leads input's max); this is only a sanity net
            # against a pathological request -- _candidate_words loops over L-1 values per
            # position, so an absurd L would hang rather than answer.
            if not (3 <= L <= 64): raise ValueError('L out of range (3..64)')
            if not g or any(not (1 <= x <= L-1) for x in g):
                raise ValueError('g must be non-empty over 1..L-1')
        except Exception as e:
            self._send_json(400, {'error': f'invalid request: {e}'}); return
        # Cache built diagrams: g_solve is ~0.05s for almost every knot, but proving a
        # smaller W impossible costs ~12s on the rare one that needs tier 3, and the
        # client-side cache does not survive a page reload.
        ckey = (L, tuple(g))
        bd = _BUILD_CACHE.get(ckey)
        if bd is None:
            global _CACHE_DIRTY
            bd = gcatalog.build(L, g)
            if bd is not None:
                _BUILD_CACHE[ckey] = bd; _CACHE_DIRTY = True; _cache_save()
        if bd is None:
            # A g whose permutation is not a single L-cycle has no diagram at any W;
            # g_solve rejects it up front rather than scanning every W.
            self._send_json(422, {'error': 'construction failed', 'g': g, 'L': L,
                                  'short': 'no diagram for this g'}); return
        self._send_json(200, bd)

    def _handle_start(self):
        try:
            body = self._read_body()
        except Exception as e:
            self._send_json(400, {'error': f'bad request body: {e}'})
            return
        try:
            loops0_raw = body['loops']
            r = float(body['r'])
            if not loops0_raw:
                raise ValueError('loops must be a non-empty list')
            loops0 = [np.array(loop, dtype=float) for loop in loops0_raw]
            for loop in loops0:
                if loop.ndim != 2 or loop.shape[1] != 3 or len(loop) < 3:
                    raise ValueError('each loop must be an (N>=3, 3) array of points')
            total_pts = sum(len(loop) for loop in loops0)
            if total_pts > MAX_TOTAL_POINTS:
                raise ValueError(f'{total_pts} input points exceeds the {MAX_TOTAL_POINTS} cap')
            if r <= 0:
                raise ValueError('r must be positive')
        except Exception as e:
            self._send_json(400, {'error': f'invalid request: {e}'})
            return

        # Supersede any running job (its worker sees the gen change and bows
        # out) and start a fresh one.
        with _lock:
            gen = CURRENT['gen'] + 1
            CURRENT.update(gen=gen, running=True, stop=False, loops=None,
                           energy=None, iter=0, converged=False,
                           stopped=False, error=None)
        threading.Thread(target=_run_job, args=(gen, loops0, r), daemon=True).start()
        self._send_json(200, {'ok': True})

    def _handle_poll(self):
        with _lock:
            self._send_json(200, {
                'running': CURRENT['running'],
                'loops': CURRENT['loops'],
                'energy': CURRENT['energy'],
                'iter': CURRENT['iter'],
                'converged': CURRENT['converged'],
                'stopped': CURRENT['stopped'],
                'error': CURRENT['error'],
            })

    def _handle_stop(self):
        with _lock:
            CURRENT['stop'] = True
        self._send_json(200, {'ok': True})

if __name__ == '__main__':
    _cache_load()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'elastic relax server listening on http://127.0.0.1:{port}  (Ctrl-C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping.')
