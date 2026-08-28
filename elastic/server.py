#!/usr/bin/env python3
"""Minimal local HTTP server exposing relax_general() to the browser.

Why this exists: a webpage cannot launch a local program by itself, so the
only way for index.html's "Relax" button to use the actual, validated
elastic.relax model (instead of the browser's simplified JS approximation)
is to POST the current knot to a small server running that model and wait
for the final shape back. This is that server. It is deliberately tiny —
stdlib only (http.server), no Flask/aiohttp/etc — since this machine has
no extra Python packages installed beyond numpy/scipy, and there is
exactly one endpoint to serve.

Usage:
  python3 elastic/server.py [port]      # default port 8731
Leave it running in a terminal, then use the Relax button in index.html as
normal (the button will show a clear error if it can't reach the server).

Request:  POST /relax
  {"loops": [[[x,y,z], ...], ...], "r": <wire radius>}
  (one array per closed component, in the SAME units as r — this is
  exactly the shape index.html's getKnotPointsForRelax() already produces)

Response: 200 application/json
  {"loops": [[[x,y,z], ...], ...], "energy": <float>, "iter": <int>,
   "converged": <bool>}
  or 4xx/5xx with {"error": "..."} on bad input / relaxation failure.

This computes to convergence (or a step budget) and returns ONLY the final
shape — no progress streaming — matching the "I don't care about seeing it
change, just show me the final shape" use case this was built for. See
relax_general()'s docstring in relax.py for the physics/scope (bending +
inextensibility + hard contact; no twist; determinant-verified for a
single closed loop, unverified for links).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from relax import relax_general

DEFAULT_PORT = 8731
MAX_STEPS = 60000          # safety cap so a pathological request can't hang forever
MAX_TOTAL_POINTS = 4000    # sanity cap on request size (sum of all loops' input points)

class Handler(BaseHTTPRequestHandler):
    # Quiet the default per-request stderr logging; a relax call already
    # prints its own progress if invoked with log=True elsewhere, and the
    # default BaseHTTPRequestHandler log line isn't useful noise here.
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        # Wide-open CORS: this is a local, single-user, loopback-only tool,
        # and the browser page is loaded from file:// (origin "null"), which
        # `*` covers unambiguously (no credentials are ever sent, so the
        # normal caution about `*` + credentials doesn't apply here).
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # CORS preflight (the browser sends this before a cross-origin POST
        # with a JSON content type).
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path != '/relax':
            self._send_json(404, {'error': f'no such endpoint: {self.path}'})
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
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

        try:
            res = relax_general(loops0, r=r, steps=MAX_STEPS)
        except Exception as e:
            self._send_json(500, {'error': f'relaxation failed: {e}'})
            return

        self._send_json(200, {
            'loops': res['loops'],
            'energy': res['E_bend'],
            'iter': res['steps_run'],
            'converged': res['converged_at'] is not None,
        })

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'elastic relax server listening on http://127.0.0.1:{port}  (Ctrl-C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopping.')
