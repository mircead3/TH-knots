#!/usr/bin/env python3
"""Run several knot relaxations in parallel (one process per knot).

relax.py's own --sweep runs a list of (p, b) pairs strictly sequentially
in a single process, which is wasteful on a multi-core machine since each
run_one()/relax() call is independent of every other. This script is the
parallel counterpart: it takes an explicit list of "p,b" pairs on the
command line, builds one task dict per pair carrying every physics
parameter relax() understands, and farms them out across a
ProcessPoolExecutor with `--jobs` worker processes (matching relax.py's
recommended default of 6 for this project's 8-core / 4-performance-core
development machine — see STATUS.md).

Each worker process calls relax() directly (imported inside work(), not
at module level — see the comment there) and writes its own result JSON
to elastic/results/, using the exact same filename convention as
relax.py's run_one() (see that function's docstring) so that both entry
points populate the same results/ directory compatibly and
make_summary.py / snapshot.py don't need to know or care which one
produced a given file.

IMPORTANT (macOS): wrap invocations of this script in `caffeinate -i` if
the run is expected to take more than a couple of minutes — on a laptop,
macOS can suspend background processes (including this one and its
worker pool) when the machine goes to sleep, silently stalling a run for
however long the lid stays closed. This bit development several times
before the habit was adopted; see STATUS.md.

Usage:
  python3 sweep_parallel.py 7,2 7,3 8,3 9,2 9,4 [--r 0.001] [--N 600]
      [--steps 60000] [--seed 1] [--kick-every 0] [--kick-amp 30]
      [--ic weave|torus|coil] [--beta 0.0] [--tw0 0.0] [--snug] [--jobs 6]

All flags after the positional p,b pairs are passed straight through to
relax() and are documented in that file's relax() and module docstrings;
they are NOT re-explained here to avoid the two files' explanations
drifting out of sync — relax.py is the single source of truth for what
each physics parameter means.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def work(task):
    """Executed in a WORKER process (not the main process) for one (p, b)
    task dict. `from relax import relax` is deliberately done here,
    inside the function, rather than at module level: ProcessPoolExecutor
    on macOS uses the 'spawn' start method, which re-imports this whole
    module fresh in each worker without inheriting the parent's already-
    imported modules, so importing relax() lazily here (rather than at
    the top of this file) keeps the import cost paid once per worker
    rather than needing relax.py to be importable before the pool even
    starts — a minor robustness/clarity choice more than a strict
    necessity, but avoids any surprise if this script is ever invoked in
    a context where the working directory isn't already on sys.path
    (hence also the explicit `sys.path.insert(0, HERE)` above, done in
    the main process but inherited into spawned workers' `sys.path` via
    the pickled task closure... actually spawned workers re-run this
    module's top level too, so the HERE-based sys.path insert happens
    again independently in each worker — this is intentional belt-and-
    suspenders, not redundant dead code).

    Returns a one-line human-readable summary string (matching the
    format relax.py's run_one() prints), which the main process prints
    as each future completes — see the `__main__` block below.
    """
    from relax import relax
    p, b = task['p'], task['b']
    res = relax(p, b, r=task['r'], N=task['N'], steps=task['steps'],
                seed=task['seed'], kick_every=task['kick_every'],
                kick_amp=task['kick_amp'], ic=task['ic'],
                beta=task['beta'], tw0=task['tw0'], snug=task['snug'])
    kick = f"_k{task['kick_every']}" if task['kick_every'] else ''
    tag = ('' if task['ic'] == 'weave' else '_' + task['ic']) + \
          (f"_tw{task['tw0']:g}" if task['beta'] > 0 else '') + ('_snug' if task['snug'] else '')
    name = f"th_{p}L{b}B_r{task['r']}_s{task['seed']}{kick}{tag}"
    with open(os.path.join(HERE, 'results', name + '.json'), 'w') as f:
        json.dump(res, f)
    m = res['metrics']
    return (f"{p}L x {b}B{kick}: {res['wall_s']}s steps={res['steps_run']} "
            f"det {res['det_initial']}->{res['det_final']}  "
            f"E/4pi2={res['E_bend_over_4pi2']:.3f}  plan={m['planarity']:.3f}  "
            f"contact={m['contact_frac']:.2f}  Wr={m['writhe']:.2f}  "
            f"kappa=[{m['kappa_min']:.1f},{m['kappa_max']:.1f}]")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', help='p,b pairs like 7,2')
    ap.add_argument('--r', type=float, default=0.001)
    ap.add_argument('--N', type=int, default=600)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--kick-every', type=int, default=0)
    ap.add_argument('--kick-amp', type=float, default=30.0)
    ap.add_argument('--ic', choices=['weave', 'torus', 'coil'], default='weave')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--tw0', type=float, default=0.0)
    ap.add_argument('--snug', action='store_true')
    ap.add_argument('--jobs', type=int, default=6)
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    # Build one independent task dict per requested (p, b) pair. Every
    # field relax() might need is duplicated into each task rather than
    # relying on shared/global state, because each task is pickled and
    # sent to a separate worker process (ProcessPoolExecutor) that does
    # not share memory with the main process or with other workers.
    tasks = []
    for pb in args.pairs:
        p, b = map(int, pb.split(','))
        tasks.append(dict(p=p, b=b, r=args.r, N=args.N, steps=args.steps,
                          seed=args.seed, kick_every=args.kick_every,
                          kick_amp=args.kick_amp, ic=args.ic,
                          beta=args.beta, tw0=args.tw0, snug=args.snug))
    # Fan the tasks out across `--jobs` worker processes and print each
    # summary line as soon as that particular knot finishes (as_completed
    # yields futures in COMPLETION order, not submission order, so faster
    # knots — e.g. fewer leads/bights, or ones that converge quickly —
    # report before slower ones regardless of where they appear in the
    # command line).
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(work, t): t for t in tasks}
        for fu in as_completed(futs):
            print(fu.result(), flush=True)
