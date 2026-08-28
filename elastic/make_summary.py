#!/usr/bin/env python3
"""Bundle elastic/results/*.json into results.js for viewer.html
(a JS global, so the viewer works from file:// without fetch).

relax.py writes one JSON file per run into elastic/results/, each holding
that run's parameters, metrics, and full relaxed point cloud (see
relax()'s docstring for exactly what's in each file). This script:

  1. Loads every result JSON in elastic/results/.
  2. Computes one extra derived quantity per run (`covering()`, the total
     curvature expressed as a covering multiplicity) that relax.py itself
     does not compute, because it's cheap and purely for display/
     classification purposes rather than physics.
  3. Runs a simple heuristic classifier (`classify()`) that labels each
     result as a flat coil, a rope-like skein, or an open 3D shape, for
     quick visual triage in the viewer and in the printed table.
  4. Writes ALL results (points included) as a single JS global
     `window.RESULTS = [...]` into results.js, which viewer.html loads
     via a plain <script src="results.js"> tag — this sidesteps the need
     to serve the directory over HTTP or handle CORS/fetch just to view
     results locally by double-clicking viewer.html (or opening it via
     file://) in a browser; see viewer.html's own comments.
  5. Prints a compact one-line-per-knot text summary to stdout (the same
     information visible in the viewer's per-knot info panel, in a form
     convenient to scan or grep in a terminal).

Run this after any batch of relax.py / sweep_parallel.py runs completes,
before opening viewer.html or looking at snapshot.py's output — it is the
step that actually regenerates results.js from whatever is currently in
results/. There is no incremental mode: it always rebuilds from scratch
by re-scanning the whole results/ directory, which is fine since even
several dozen result files load and process in well under a second.

Usage:
  python3 make_summary.py
"""
import json, glob, math, os
import numpy as np

def covering(points):
    """Total curvature / 2pi ~ covering multiplicity of a coiled state.

    By a discrete analogue of the Fenchel/Fary-Milnor-adjacent fact that
    a closed curve's total curvature (sum of turning angles, or here its
    continuum limit integral kappa ds) is at least 2*pi for a simple
    closed curve and is close to EXACTLY 2*pi*m for a curve that looks
    like an m-times-covered circle (a coil of m nested/stacked loops),
    dividing the sum of all per-vertex turning angles by 2*pi gives a
    number that is close to an integer m precisely when the relaxed shape
    is (approximately) an m-fold coil — which is exactly the situation
    the "flat coil" branch of classify() below wants to detect and label
    with the right multiplicity `m`. For shapes that are NOT simple
    coils (open 3D shapes, rope-like skeins), this number is still well-
    defined but doesn't have as clean an interpretation; classify() only
    relies on it once the other checks (planarity, contact fraction)
    have already suggested "this looks like a coil."

    This duplicates (deliberately) the turning-angle computation inside
    relax.py's curvatures()/bending() rather than importing from there,
    since this script works purely from the JSON's raw 'points' array
    (no access to relax.py's internal `h` edge-length convention is
    needed here — only the unit tangent directions matter, computed
    fresh from consecutive point differences).
    """
    P = np.array(points)
    e = np.roll(P, -1, 0) - P
    le = np.linalg.norm(e, axis=1)
    u = e / le[:, None]
    c = np.clip((u * np.roll(u, 1, 0)).sum(1), -1, 1)
    return float(np.arccos(c).sum() / (2 * np.pi))

def classify(res):
    """Heuristic shape classifier used purely for the viewer's label and
    the printed summary table — NOT used anywhere in the physics
    (relax.py never reads this back), so it is fine for this to be a
    simple set of thresholds tuned by eye against the actual result
    shapes rather than a rigorous geometric definition.

    `mc` = covering() (see above); `mult` rounds it to the nearest
    integer coil count. `excess` compares the shape's actual bending
    energy to the THEORETICAL energy of a perfect (untwisted) m-covered
    circle of the same total length, which is exactly m^2 * 4*pi^2 in
    the E_bend_over_4pi2 units relax.py reports (a bare circle costs
    4*pi^2 = m^2 with m=1; an m-times-covered circle scales quadratically
    since curvature scales linearly with m at fixed total length) — so
    `excess` near 0 means "this really is close to an idealized coil,"
    while a large positive excess means the shape carries a lot of extra
    bending cost beyond the idealized-coil baseline (e.g. because the
    strands within the coil are internally twisted/braided around each
    other rather than lying in clean parallel loops).

    The classification logic, in order:
      - `aspect = gyr[1]/gyr[0]` (second-largest / largest principal
        radius of gyration): if this is small (< 0.45), the shape is
        long and thin along one axis relative to the other two —
        exactly what an elongated, rope-like twisted bundle looks like
        (a "spindle" shape) — labelled 'rope-like skein'. This check
        comes FIRST because it was found (during development, comparing
        against actual rendered snapshots) to be a more reliable
        discriminator for the rope regime than the planarity/contact
        checks below, which can be fooled by a rope-like shape that also
        happens to have moderate planarity or contact fraction.
      - Otherwise, if the shape is nearly flat (planarity < 0.2) and has
        substantial self-contact (contact_frac > 0.5), it's some kind of
        coil: label it a clean 'flat coil (m-covered circle)' if the
        covering multiplicity is close to an integer AND the excess
        energy is small (both conditions are needed — a shape could
        accidentally have mc close to an integer while still being
        energetically far from a clean coil, or vice versa), otherwise
        label it 'flat coil (m rings, internally twisted)' — still
        fundamentally a coil, but with visible extra structure (see
        STATUS.md: several TH(p,b) results are exactly this — an
        m-ring coil where the rings are internally twisted around each
        other rather than lying as clean parallel circles).
      - Anything not caught by the above two branches is left as a
        generic 'open 3D shape' — the catch-all for the genuinely
        three-dimensional, non-coiled, non-rope-like equilibria (see
        STATUS.md's B < L < 2B regime).

    Also stashes `coil_mult` and `E_excess` back onto the `res` dict as a
    side effect (mutates its argument) so main() doesn't need to
    recompute them separately when printing the summary table.
    """
    m = res['metrics']
    mc = res['tot_curv_2pi']
    mult = max(1, round(mc))
    excess = res['E_bend_over_4pi2'] / mult**2 - 1
    res['coil_mult'] = mult
    res['E_excess'] = round(excess, 3)
    g = m['gyr']
    aspect = g[1] / g[0]                       # g2/g1: small = prolate spindle
    if aspect < 0.45:
        return 'rope-like skein (elongated twisted bundle)'
    if m['planarity'] < 0.2 and m['contact_frac'] > 0.5:
        if excess < 0.15 and abs(mc - mult) < 0.25:
            return f'flat coil ({mult}-covered circle)'
        return f'flat coil ({mult} rings, internally twisted)'
    return 'open 3D shape'

def main():
    """Scan elastic/results/*.json, annotate each with `tot_curv_2pi` and
    `class` (mutating the loaded dicts in place, then re-serializing the
    whole augmented set), sort by (leads, bights, seed) for a stable,
    human-friendly ordering in both results.js and the printed table,
    write results.js, and print the one-line-per-knot summary.

    Note that results.js ends up containing the FULL original JSON
    content of every result file (including the complete point cloud)
    PLUS the two derived fields — nothing is stripped or summarized away
    — since viewer.html needs the raw points to actually render the 3D
    tubes; only the printed stdout table below is a true summary.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    runs = []
    for f in sorted(glob.glob(os.path.join(here, 'results', '*.json'))):
        res = json.load(open(f))
        res['tot_curv_2pi'] = covering(res['points'])
        res['class'] = classify(res)
        res['file'] = os.path.basename(f)
        runs.append(res)
    runs.sort(key=lambda r: (r['p'], r['b'], r['seed']))
    with open(os.path.join(here, 'results.js'), 'w') as out:
        out.write('window.RESULTS = ')
        json.dump(runs, out)
        out.write(';\n')
    for r in runs:
        m = r['metrics']
        det = f"{r.get('det_initial')}->{r.get('det_final')}"
        print(f"{r['p']}L x {r['b']}B  det={det:>14}  E/4pi2={r['E_bend_over_4pi2']:7.3f}  "
              f"plan={m['planarity']:.3f}  contact={m['contact_frac']:.2f}  "
              f"Wr={m['writhe']:6.2f}  m={r['tot_curv_2pi']:5.2f}  "
              f"xs={r['E_excess']:6.3f}  -> {r['class']}")

if __name__ == '__main__':
    main()
