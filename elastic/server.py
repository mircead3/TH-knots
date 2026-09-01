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
        else:
            self._send_json(404, {'error': f'no such endpoint: {self.path}'})

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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'elastic relax server listening on http://127.0.0.1:{port}  (Ctrl-C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping.')
