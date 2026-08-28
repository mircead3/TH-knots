#!/usr/bin/env python3
"""Elastic-wire relaxation of Turk's head (TH) knots.

THE PHYSICAL QUESTION
----------------------
A Turk's head is a knot tied from a single loop of wire, described by two
integers (p leads, b bights): it winds p times around a cylindrical jig's
axis while its edge scallops b times around the jig's circumference before
closing on itself.  When the wire is stiff, springy, and welded into a
closed loop, it does NOT stay in the shape it was tied in — it relaxes
under its own elastic force to some equilibrium shape.  This file predicts
that equilibrium shape by numerically minimizing an elastic energy subject
to the wire being unstretchable and unable to pass through itself.

See STATUS.md in this directory for the current state of the project: what
has been validated against the user's physical bench experiments, what
still disagrees, and what bench measurements would resolve the remaining
questions.  That file is the entry point for anyone (human or model)
picking this project back up without the full conversation history.

THE ENERGY MODEL
-----------------
The wire is discretized as N vertices P[0..N-1] joined into a closed
polygon (edges P[i] -> P[i+1 mod N]), each edge of the same rest length
h = 1/N (so the closed curve has total length 1 — everything is in units
where the wire's contour length is 1).

Three ingredients enter the total energy:

 1. BENDING.  E_bend = (2/h) * sum_i (1 - cos theta_i), where theta_i is
    the turning angle at vertex i.  This is the standard discretization of
    the continuum bending energy integral kappa^2 ds for an elastic rod
    (Kirchhoff rod theory) — a real wire resists being bent sharply, and
    this is by far the dominant elastic cost for a thin wire.

 2. TWIST (optional, beta > 0).  A welded wire loop carries a fixed
    linking number Lk between the two edges of an imagined ribbon running
    along it; Lk splits into writhe Wr (how the CENTERLINE coils in space)
    and twist Tw (how the material rotates about the centerline), with
    Lk = Wr + Tw always.  As the centerline relaxes, Wr changes, and Tw
    must absorb the difference to keep Lk fixed — the elastic cost of
    that stored twist is E_twist = 4*pi^2*beta*(Lk - Wr)^2 (uniform
    equilibration of twist along the rod; beta = GJ/EI is the torsion-to-
    bending stiffness ratio, ~0.77 for round isotropic wire).  See
    writhe_grad() for how the force from this term is computed.

 3. SELF-CONTACT (hard constraint, not an energy).  The wire has a real
    radius r and must not pass through itself: every pair of non-adjacent
    segments is kept at centerline separation >= 2r by a projection step
    (see project_constraints), not by a soft penalty energy.  This is what
    makes different Turk's head knots settle into topologically distinct
    final shapes instead of all flattening to a bare circle, and it is
    also what the knot-determinant check (knot_det) guards: if contact
    resolution ever failed to keep segments apart, a strand could tunnel
    through another and change the knot type without the simulation
    noticing anything else was wrong.

No gravity, no friction: the wire floats in free space with no dissipative
sliding contact.  This is deliberately the simplest possible model; where
it disagrees with the bench (see STATUS.md) the next hypothesis is usually
"friction locked something during collapse" or "some twist was frozen in
at the weld" — both are physical effects this model omits by design, and
the twist term above is the first partial answer to the second one.

INITIAL CONDITIONS (how the tied knot is represented before relaxing)
------------------------------------------------------------------
Three constructors build the tied-but-unrelaxed curve, selected by --ic
(plus the independent --snug and --ic coil refinements below):

  torus_knot()  "in hand" presentation: p times around one way, b times
                the other, on a torus.  All crossings same-handed.  This
                is what you get tying a torus knot by hand with no jig,
                and is ONLY the same knot as th_weave() when p=2 — for
                p>=3 the two constructions are genuinely different knots
                (see th_weave's docstring).

  th_weave()    the actual alternating Turk's head weave as tied on a
                jig: over/under strictly alternates crossing to crossing.
                Two shapes, chosen by `snug`:
                  snug=False: a tall loose weave (historical first
                    attempt; large radial bumps, order-1 height). Kept for
                    backward comparison; DO NOT trust it alone for p>=3
                    without also checking `snug=True` or `ic='coil'` — the
                    loose weave lets the strands fly apart before contact
                    can organize them, which loses the physical symmetry
                    of the tied knot during relaxation (see STATUS.md).
                  snug=True: the wire AS ACTUALLY TIED on the jig — a
                    narrow band of p wraps sitting side by side in near-
                    contact, over/under bumps of about one wire radius.
                    This preserves symmetry through relaxation much better
                    and is the current default for the "rope" regime
                    (L > 2B; see STATUS.md).

  th_coil()     the user's bench PROTOCOL, not the tied shape: after
                tying on the jig, the wire is taken off, forced open by
                hand into a coil, and the coils are equalized by working
                the wire around before release.  Mathematically this is
                the same knot presented as the closure of a p-strand
                braid word, with the b*(p-1) strand swaps spread evenly
                (hence B-fold rotationally symmetric) around a ring of p
                stacked, near-touching loops.  Use --ic coil to reproduce
                this starting point.

Every constructor is followed by resample_closed() (uniform arclength
reparametrization to exactly N points) and normalize() (rescale total
length to 1, center at the origin) before the physics loop begins.

THE RELAXATION ALGORITHM (see relax() for the full loop)
-----------------------------------------------------
Explicit gradient descent on the bending+twist energy would need a
timestep of order h^4 to stay stable, because bending is a 4th-order
(fourth-derivative-like) operator — utterly impractical at N~600.
Instead:

  - The bending force is passed through a Sobolev (H^2) PRECONDITIONER
    (precondition()) that divides each Fourier mode by 1+(q/q*)^4,
    matching the q^4 growth of the bending Hessian's eigenvalues. This is
    the standard trick (Sobolev gradient flow) for evolving curves under
    bending energy without a tiny explicit timestep.
  - The preconditioned step is then projected onto the TANGENT SPACE of
    the inextensibility constraint (tangent_project()) so it does not
    change any edge length to first order — otherwise the wire would
    stretch or compress instead of just bending.
  - A position-based-dynamics (PBD) style projection (project_edges +
    project_constraints) then restores edge lengths exactly and pushes
    apart any segment pair that has drifted within 2r of another,
    iterating a few times per step (this is a Gauss-Seidel / Jacobi
    relaxation of the two constraint families together, not a single
    linear solve — a few iterations are enough because the correction
    needed per step is tiny).
  - The step size s is adapted (grown on success, shrunk on an energy
    increase) and, near contact, additionally CAPPED so that no vertex
    can move more than about half a wire radius per step — this is the
    critical anti-tunneling safeguard: without it, a large step could
    jump a segment clean through a neighboring one that the discrete
    contact check never sampled in between.
  - Optionally, periodic KICKS (small random smooth perturbations,
    kick_every/kick_amp) emulate the bench vibration/handling that lets a
    real wire escape a shallow local minimum; kicks are applied in small
    tunnel-safe substeps and are REJECTED (reverted) if they would either
    let two segments interpenetrate or change the knot's determinant.

VERIFICATION
------------
Every run computes the knot determinant (a classical, easy-to-compute
knot invariant derived from Fox n-colorings, see knot_det()) of both the
initial and the final curve.  If the two differ, some strand passed
through another during the simulation — a genuine bug, not just a
"different final shape" — and a WARNING is printed.  This is the
project's primary safety net and caught real mistakes during development
(see git history / STATUS.md); never trust a run whose determinant
changed.

USAGE
-----
  python3 relax.py --p 3 --b 2 [--r 0.001] [--N 600] [--steps 60000]
      [--seed 1] [--ic weave|torus|coil] [--snug] [--beta 0.77 --tw0 2]
      [--kick-every 5000 --kick-amp 10]
  python3 relax.py --sweep [--from-idx 0 --to-idx 17]   # batch over SWEEP

Output: one JSON file per run in elastic/results/, containing the final
point cloud, energy history, metrics (curvature range, planarity, contact
fraction, writhe...), and the determinant check.  See make_summary.py to
bundle results into the viewer and snapshot.py for quick PNG previews.
For running many knots at once across CPU cores, use sweep_parallel.py
instead of --sweep (which runs strictly sequentially in one process).
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve_banded

# ---------------------------------------------------------------- geometry
#
# Each function below returns an (N, 3) array of points describing the
# TIED (pre-relaxation) knot in one of the conventions discussed in the
# module docstring.  None of these need to trace out exactly-N points
# with uniform arclength spacing yet — resample_closed() fixes that up
# right afterwards, so the parametrizations here are free to use whatever
# is most natural (uniform in the parameter t, not in arclength).

def torus_knot(p, b, N, R=1.0, a=0.5, seed=1, noise=0.03):
    """(p,b) torus curve — the IN-HAND tying presentation (all crossings
    same-handed).  Equals the TH weave only for p=2.

    Standard parametrization of a curve lying on a torus of major radius R
    (distance from the central axis to the tube's centerline) and minor
    radius a (tube radius): it winds p times around the long way (angle
    phi) while going b times around the short way (angle psi). Because the
    curve's radial distance from the axis is R + a*cos(psi), a strand
    crossing at larger radius is always the "outer" one — so with a single
    consistent sense of a projection, every self-crossing has the SAME
    over/under handedness.  This is what happens if you wind wire by hand,
    always crossing over (or always under) the previous wrap, without
    weaving over-under-over-under.

    seed/noise: a small smooth (few low Fourier modes) random perturbation
    is added purely to break the exact symmetry of the analytic curve, so
    that the relaxation does not get stuck exactly on top of an unstable
    symmetric saddle point. It has no physical meaning beyond that.
    """
    t = np.arange(N) / N
    phi = 2 * np.pi * p * t
    psi = 2 * np.pi * b * t
    rr = R + a * np.cos(psi)
    P = np.stack([rr * np.cos(phi), rr * np.sin(phi), a * np.sin(psi)], axis=1)
    rng = np.random.default_rng(seed)
    for k in range(1, 5):
        P += (noise * a / k) * np.cos(2 * np.pi * k * t[:, None]
                                      + rng.uniform(0, 2 * np.pi, 3)) \
                             * rng.standard_normal(3)
    return P

def th_weave(p, b, N, R=1.0, A=0.5, bump=0.12, seed=1, noise=0.03,
             snug=False, r_wire=0.001):
    """Woven Turk's head: crossing passages along the strand sit at
    t = n/(2bp) for n !≡ 0 (mod p); over/under alternates along t.
    The radial profile eases (cosine) between +-bump at the passages.

    Unlike torus_knot(), here the over/under handedness ALTERNATES from
    one crossing to the next along the strand — this is what an actual
    over-under-over-under Turk's head weave looks like, and for p>=3 it is
    a topologically DIFFERENT knot from the (p,b) torus knot (they agree
    only at p=2, where there is just one strand pair per bight, and
    "alternating" and "always same" coincide trivially). Concretely,
    TH(3,2) here is the figure-eight knot (4_1), not the trefoil that
    torus_knot(3,2) would give; TH(3,4) is 8_18, not the torus knot 8_19.

    How the alternation is built: imagine the strand as it winds p times
    around the cylinder's axis (parametrized by t in [0,1), angle
    2*pi*p*t). It crosses under-or-over some other strand each time its
    angular position passes a multiple of 2*pi/p that ISN'T also a
    multiple of a full revolution boundary — those crossing angles are
    n/p for n not divisible by p; over 2*b*p subdivisions the sorted set
    of such crossing TIMES is `pas`. Walking along the strand, the sign
    of the radial bump (whether this passage rides on the outside=over or
    inside=under of the crossing) simply alternates every time `pas` is
    crossed (`sgn`), which is exactly the alternating-weave rule. Between
    two consecutive passages the radius eases from +bump to -bump (or vice
    versa) along a cosine profile (`u`, `w`) so the curve is smooth and the
    knot's actual radial excursion is confined to a small neighborhood of
    each crossing, matching a physically woven strand riding over then
    under its neighbors.

    snug=True builds the knot AS TIED on a jig: a narrow band (width
    ~ p wire diameters, wraps side by side in near-contact, over/under
    excursion ~ 1 wire radius) instead of a tall loose weave. Concretely
    this rescales R down to 1/(2*pi*p) (so the p wraps, each of
    circumference ~2*pi*R, together sum to about unit total length) and
    shrinks A (the height/spread of the band across its p wraps) and bump
    (the radial over/under excursion at each crossing) down to a few wire
    radii, instead of the historical A=0.5, bump=0.12 which correspond to
    a band much wider than a real snug jig winding. See the module
    docstring for why this matters: the loose (snug=False) weave lets
    strands fly apart and lose symmetry before self-contact can organize
    them; the snug weave starts already touching, like the real jig.
    """
    if snug:
        R = 1.0 / (2 * np.pi * p)      # ring sized so curve length ~ 1
        A = 1.6 * p * r_wire           # band half-height: wrap spacing ~3r
        bump = 1.25 * r_wire
    t = np.arange(N) / N
    n = np.arange(2 * b * p)
    pas = n[(n % p) != 0] / (2.0 * b * p)          # sorted passage times
    sgn = np.where(np.arange(len(pas)) % 2 == 0, 1.0, -1.0)
    # cosine easing between consecutive passages (cyclic)
    idx = np.searchsorted(pas, t, side='right')     # next passage index
    prev = (idx - 1) % len(pas)
    tp = pas[prev]
    tn = pas[idx % len(pas)] + (idx == len(pas))    # wrap
    tt = t + (t < tp)                               # unwrap t below tp
    u = (tt - tp) / (tn - tp)
    w = sgn[prev] * np.cos(np.pi * u)
    rr = R + bump * w
    phi = 2 * np.pi * p * t
    P = np.stack([rr * np.cos(phi), rr * np.sin(phi), A * np.cos(2 * np.pi * b * t)], axis=1)
    # smooth low-frequency perturbation to break symmetry (see torus_knot's
    # docstring for why: purely to avoid landing exactly on an unstable
    # symmetric saddle of the energy landscape)
    rng = np.random.default_rng(seed)
    for k in range(1, 5):
        amp = noise * A / k
        P += amp * np.cos(2 * np.pi * k * t[:, None] + rng.uniform(0, 2 * np.pi, 3)) \
                 * rng.standard_normal(3)
    return P

def th_coil(p, b, N, r_wire=0.001, seed=1, noise=0.02):
    """Hand-equalized coil: the TH knot as its weaving-braid closure
    (sigma_1 sigma_2^-1 ... sigma_{p-1}^±)^b — p stacked rings of radius
    1/(2 pi p), slot spacing 3.2r, with the b(p-1) strand swaps spread
    uniformly (hence B-fold symmetrically) around the circle.  This is the
    user's bench protocol: open the knot into a coil, equalize, release.

    Picture p parallel circular loops ("slots" 0..p-1) stacked along the
    axis like a spring, all lying at the same radius R = 1/(2*pi*p). A
    single continuous strand visits all p slots by making b(p-1) crossing
    "swaps" between adjacent slots as it goes around; each swap moves the
    strand from slot idx to slot idx+1 (or back), and its over/under sign
    alternates by slot-pair parity so that the resulting braid, closed up,
    is the same TH(p,b) knot as th_weave() produces (verified against
    th_weave() by knot determinant in this file's test/validation runs;
    see STATUS.md). The b(p-1) swap "events" are spaced uniformly in
    angle around the ring, which is exactly what makes the resulting shape
    b-fold (and, combined with the two strand-orientation choices, roughly
    2-fold) rotationally symmetric — matching the user's description of
    hand-equalizing the coil before releasing it on the bench.

    Construction, step by step:
      1. `events`: the b(p-1) (angle, slot-pair, sign) triples, evenly
         spaced in angle, describing every crossing swap the braid closure
         makes, in order around the ring.
      2. Walk a single strand through the braid: starting in slot `s=0`,
         `passes[a]` collects, for the a-th time the strand is in some
         slot performing a full trip around the ring, exactly which swap
         events it participates in (only the ones touching its current
         slot). `starts[a]` records which slot it started that lap in.
         This traces out the permutation cycle that must return to slot 0
         after all p laps (the `assert` verifies the braid word actually
         closes up into a single-component knot rather than a link).
      3. For each lap a, the strand's angular position within that lap
         (`tha`, running 0..2*pi) determines a smooth height `za` (which
         slot it currently occupies, blending across each swap event with
         a cosine ease `blend`) and radial bump `ra` (positive/negative by
         wire radius, whichever side is "over" at each swap, via `over`),
         analogous to the crossing bumps in th_weave().
      4. z (height along the coil axis) is `za` recentered and scaled by
         the slot spacing `dz`; rho (radial excursion) is `ra`; final
         Cartesian points follow the same "circle of radius R+rho at
         height z, angle theta" pattern as the other two constructors.

    r_wire sets the physical scale (slot spacing dz = 3.2*r_wire, crossing
    bump delta = 1.25*r_wire) so the geometry is consistent with the wire
    thickness used elsewhere in a given run.
    """
    R = 1.0 / (2 * np.pi * p)
    dz = 3.2 * r_wire
    delta = 1.25 * r_wire
    nlet = b * (p - 1)
    w = 0.45 * 2 * np.pi / nlet          # transition half-width
    events = [(2 * np.pi * (m + 0.5) / nlet, m % (p - 1),
               1 if (m % (p - 1)) % 2 == 0 else -1) for m in range(nlet)]
    # follow the strand through p passes of the braid
    s, passes, starts = 0, [], []
    for a in range(p):
        starts.append(s)
        trs = []
        for (thc, idx, sign) in events:
            if s == idx:
                trs.append((thc, s, idx + 1, sign)); s = idx + 1
            elif s == idx + 1:
                trs.append((thc, s, idx, sign)); s = idx
        passes.append(trs)
    assert s == 0, 'braid closure did not return to start slot'
    t = np.arange(N) / N
    theta = 2 * np.pi * p * t
    a_idx = np.minimum((theta // (2 * np.pi)).astype(int), p - 1)
    th = theta - 2 * np.pi * a_idx
    z = np.empty(N)
    rho = np.zeros(N)
    for a in range(p):
        mask = a_idx == a
        tha = th[mask]
        za = np.full(tha.shape, float(starts[a]))
        ra = np.zeros(tha.shape)
        for (thc, sfrom, sto, sign) in passes[a]:
            u = (tha - thc) / w
            za[u > 1] = sto
            ins = (u >= -1) & (u <= 1)
            blend = 0.5 * (1 - np.cos(np.pi * (u[ins] + 1) / 2))
            za[ins] = sfrom + (sto - sfrom) * blend
            over = (sign > 0) == (sto > sfrom)
            ra[ins] += (delta if over else -delta) * 0.5 * (1 + np.cos(np.pi * u[ins]))
        z[mask] = (za - (p - 1) / 2) * dz
        rho[mask] = ra
    rr = R + rho
    P = np.stack([rr * np.cos(theta), rr * np.sin(theta), z], axis=1)
    rng = np.random.default_rng(seed)
    for k in range(1, 5):
        P += (noise * p * r_wire / k) * np.cos(
            2 * np.pi * k * t[:, None] + rng.uniform(0, 2 * np.pi, 3)) \
            * rng.standard_normal(3)
    return P

def resample_closed(P, N):
    """Reparametrize the closed polyline P by ARCLENGTH and resample to
    exactly N uniformly-spaced points.  All three geometry constructors
    above are uniform in an arbitrary parameter t, not in arclength (a
    torus curve moves faster near some parts than others); the physics
    loop needs every edge to start at (very close to) the same rest
    length h=1/N, since bending() and tangent_project() both assume a
    single scalar rest length h shared by all edges. Uses linear
    interpolation between the two straddling original points found via
    binary search (`searchsorted`) on the cumulative arclength `s`.
    """
    seg = np.roll(P, -1, 0) - P
    d = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    snew = np.linspace(0, L, N, endpoint=False)
    idx = np.clip(np.searchsorted(s, snew, side='right') - 1, 0, len(P) - 1)
    frac = ((snew - s[idx]) / d[idx])[:, None]
    return P[idx] * (1 - frac) + P[(idx + 1) % len(P)] * frac

def normalize(P):
    """Rescale the closed curve to total (polygonal) length 1 and
    recenter it at the centroid.  Every run works in these fixed units
    (contour length 1) so that energies, wire radius r, and step sizes
    are all directly comparable across different (p, b) knots without an
    extra size normalization at the end."""
    d = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    P = P / d.sum()
    return P - P.mean(0)

# ------------------------------------------------------------------ energy

def bending(P, h):
    """Discrete bending energy and its gradient (= -force) with respect
    to every vertex position.

    Return (E_b, F_b). E_b = (2/h) sum(1-cos theta_i).

    At each vertex i, let u = P[i]-P[i-1] and e = P[i+1]-P[i] be the
    incoming and outgoing edge vectors and theta_i the angle between them
    (so cos(theta_i) = u.e/(|u||e|)). For a smooth curve with local radius
    of curvature rho, 1-cos(theta) ~ (h/rho)^2/2 for small turning angles,
    so (2/h)*sum(1-cos theta_i) ~ sum h*kappa_i^2 -> the continuum bending
    energy integral kappa^2 ds as h -> 0. This is the standard discrete
    bending energy used in discrete elastic rod simulations.

    The gradient is derived directly by differentiating c = u.e/(|u||e|)
    with respect to each of the three vertices u and e touch (i-1, i,
    i+1) and assembling the (2/h)*(-dc/dP) contribution from every angle
    theta_i that vertex P[j] participates in — vertex j appears as the
    "outgoing" endpoint of angle j (via `gv`, shifted by +1 with
    `np.roll(..., 1, 0)`) and as the "incoming" endpoint of angle j+1 (via
    `gu`, shifted by -1). Returned F is already the NEGATIVE gradient
    (i.e. a physical force: it points to decrease energy), consistent
    with how it's used directly as a descent direction in relax().
    """
    e = np.roll(P, -1, 0) - P
    le = np.linalg.norm(e, axis=1)
    u, lu = np.roll(e, 1, 0), np.roll(le, 1)
    dot = (u * e).sum(1)
    c = dot / (lu * le)
    E = (2.0 / h) * float(np.sum(1.0 - c))
    gu = e / (lu * le)[:, None] - (c / lu**2)[:, None] * u
    gv = u / (lu * le)[:, None] - (c / le**2)[:, None] * e
    coef = 2.0 / h
    F = coef * (gu - gv)
    F += np.roll(-coef * gu, -1, 0)
    F += np.roll(coef * gv, 1, 0)
    return E, F

def precondition(F, qstar):
    """Sobolev (H^2) preconditioner: divide Fourier modes by 1+(q/q*)^4.

    Bending is (locally) a 4th-order operator: its Hessian's eigenvalues
    grow like q^4 for a Fourier mode of wavenumber q. Plain gradient
    descent on such an energy needs a timestep ~ 1/q_max^4 ~ 1/N^4 to
    stay stable — hopelessly slow. Instead, this rescales the force in
    Fourier space by 1/(1+(q/q*)^4), which is close to 1 for the smooth
    low-frequency modes (where explicit descent would be fine anyway) and
    strongly damps the stiff high-frequency modes that would otherwise
    force a tiny step. q* (set from qstar_mult in relax()) marks the
    frequency scale at which damping kicks in. This turns "stiff gradient
    flow on bending energy" into a well-conditioned descent, at the cost
    of the resulting step no longer being a pure L2 gradient (it's an H^2
    Sobolev gradient) — which is fine, since any descent direction with
    negative dot product against the true gradient still decreases energy
    for a small enough step.
    """
    Fh = np.fft.rfft(F, axis=0)
    q = 2 * np.pi * np.arange(Fh.shape[0])
    Fh /= (1.0 + (q / qstar)**4)[:, None]
    return np.fft.irfft(Fh, n=len(F), axis=0)

def tangent_project(F, P):
    """Remove the force component that changes edge lengths.
    Solves the cyclic-tridiagonal (J J^T) lam = J F via a banded LAPACK
    solve plus Sherman-Morrison for the two cyclic corner entries.

    The inextensibility constraint is: every edge length |P[i+1]-P[i]|
    stays fixed. To first order, moving vertices by a displacement field D
    changes edge i's length by u_i . (D[i+1]-D[i]) where u_i is the unit
    tangent of edge i. We want the closest (in the least-squares sense)
    modification of the proposed step F that satisfies ALL these N linear
    constraints simultaneously — i.e. project F onto the tangent space of
    the (N-dimensional) constraint manifold.  This is a standard Lagrange
    multiplier problem: introduce one multiplier lambda_i per edge and
    solve for the lambda that makes F + (correction from lambda) satisfy
    the constraints; the resulting linear system for lambda is (J J^T)
    lambda = J F, where J is the constraint Jacobian.

    Because each edge's length only involves its two endpoint vertices'
    displacements, and only NEIGHBORING edges share a vertex, J J^T works
    out to be tridiagonal (each lambda_i couples only to lambda_{i-1} and
    lambda_{i+1}) PLUS two corner entries because the chain of edges is a
    CLOSED loop (edge N-1 is adjacent to edge 0). A plain tridiagonal
    solve can't see those corners; the fix is the classic trick for
    cyclic tridiagonal systems: peel off the corners as a rank-1 (here,
    effectively rank-2 handled as two extra scalar corrections `gamma`
    and `beta`) update and apply the Sherman-Morrison formula on top of a
    fast banded (LAPACK `solve_banded`) solve of the plain tridiagonal
    part. This is far faster than a generic sparse solve (this file
    previously used scipy.sparse.linalg.spsolve here; the banded solve is
    the same math, done via a specialized O(N) algorithm instead of a
    general sparse factorization) — worth it since this runs every single
    step of the relaxation.

    `off[i]` is the off-diagonal entry M[i, i+1] = -(u_i . u_{i+1});
    `off[N-1]` is therefore the corner entry connecting edge N-1 back to
    edge 0. `diag` starts at the pure-tridiagonal value 2.0 (each row i of
    J J^T has diagonal 2 since |u_i|^2 + |u_i|^2 = 2 for unit tangents,
    after the standard J J^T assembly) and is adjusted at the two corner
    rows to account for removing the corner terms into the Sherman-
    Morrison update. `y` solves the tridiagonal system with the true
    right-hand side `JF`; `z` solves it with a unit correction vector
    representing the corner coupling; `fac` combines them via the
    Sherman-Morrison formula to recover the exact cyclic solution `lam`.

    Once lambda is known, the tangential correction to F at vertex i is
    +u_i*lambda_i (from edge i, which starts at vertex i) minus
    u_{i-1}*lambda_{i-1} (from edge i-1, which ends at vertex i) — hence
    the final `Ft = F + un*lam[:,None] - roll(un*lam[:,None], 1, 0)`.
    """
    N = len(P)
    e = np.roll(P, -1, 0) - P
    un = e / np.linalg.norm(e, axis=1)[:, None]
    JF = (un * (np.roll(F, -1, 0) - F)).sum(1)
    off = -(un * np.roll(un, -1, 0)).sum(1)   # M[i,i+1]; off[N-1] = corner
    beta = off[N - 1]
    gamma = -2.0
    diag = np.full(N, 2.0)
    diag[0] -= gamma
    diag[N - 1] -= beta * beta / gamma
    ab = np.zeros((3, N))
    ab[0, 1:] = off[:N - 1]
    ab[1, :] = diag
    ab[2, :N - 1] = off[:N - 1]
    rhs = np.zeros((N, 2))
    rhs[:, 0] = JF
    rhs[0, 1] = gamma
    rhs[N - 1, 1] = beta
    yz = solve_banded((1, 1), ab, rhs)
    y, z = yz[:, 0], yz[:, 1]
    fac = (y[0] + (beta / gamma) * y[N - 1]) / (1.0 + z[0] + (beta / gamma) * z[N - 1])
    lam = y - fac * z
    Ft = F + un * lam[:, None]
    Ft -= np.roll(un * lam[:, None], 1, 0)
    return Ft

# --------------------------------------------------- segment-segment contact
#
# The wire has a real radius r; two points of the curve more than one
# arclength-neighborhood apart must never come closer than 2r center-to-
# center, or the (zero-thickness) centerline model would be lying about
# whether the physical wire actually intersects itself.  This section
# finds and resolves any such near-approaches.

def seg_closest(P, pairs):
    """Closest points between segment pairs (i,i+1)x(j,j+1).
    Returns (dist, s, t, n) with n the unit vector from j-seg to i-seg.

    Standard closed-form solution for the minimum distance between two
    line segments in 3D (the same algorithm as, e.g., Ericson's "Real-Time
    Collision Detection" section 5.1.9): parametrize segment i as
    a0 + s*d1 (s in [0,1]) and segment j as b0 + t*d2 (t in [0,1]);
    minimizing |a0+s*d1 - b0-t*d2|^2 over the UNCONSTRAINED (s,t) gives a
    2x2 linear system (the `a,e,f,c,bb,denom` algebra below), whose
    solution is then clamped into [0,1] for each parameter in turn (clamp
    s, recompute the corresponding optimal t from the clamped s, then
    reclamp s from that t) to handle the case where the unconstrained
    optimum falls outside the segments. `n` is returned as a UNIT vector
    (not the raw difference) so that project_constraints() can directly
    use it as a push-apart direction scaled by the desired correction
    magnitude.
    """
    N = len(P)
    i, j = pairs[:, 0], pairs[:, 1]
    a0, a1 = P[i], P[(i + 1) % N]
    b0, b1 = P[j], P[(j + 1) % N]
    d1, d2 = a1 - a0, b1 - b0
    r12 = a0 - b0
    a = (d1 * d1).sum(1); e = (d2 * d2).sum(1)
    f = (d2 * r12).sum(1); c = (d1 * r12).sum(1)
    bb = (d1 * d2).sum(1)
    denom = a * e - bb * bb
    s = np.where(denom > 1e-18 * a * e, (bb * f - c * e) / np.maximum(denom, 1e-30), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = np.clip((bb * s + f) / e, 0.0, 1.0)
    s = np.clip((bb * t - c) / a, 0.0, 1.0)
    pa = a0 + s[:, None] * d1
    pb = b0 + t[:, None] * d2
    n = pa - pb
    dist = np.linalg.norm(n, axis=1)
    n = n / np.maximum(dist, 1e-30)[:, None]
    return dist, s, t, n

def build_pairs(P, cutoff, w):
    """Candidate segment pairs whose midpoints are within cutoff.

    Checking every pair of the N segments for contact would be O(N^2)
    every step, which dominates runtime at N~600-900. Instead a KD-tree
    (`cKDTree`) over segment MIDPOINTS finds the (much smaller) set of
    pairs that could plausibly be close, in roughly O(N log N). `cutoff`
    should exceed the actual contact distance 2r by a safety margin
    ("skin", see relax()) so that this candidate list stays valid for
    several simulation steps before it needs rebuilding — see `acc_disp`
    in relax() for how that margin is tracked and spent.

    `w` excludes pairs of segments that are close along the STRAND
    (within w indices of each other, accounting for the cyclic wrap via
    `N - dd`) — two adjacent or near-adjacent segments of a smooth curve
    are always close in space too (they share a vertex or are one bend
    apart) without that meaning actual self-contact; only sufficiently
    separated arcs of the wire indicate a genuine crossing/touch.
    """
    N = len(P)
    mid = 0.5 * (P + np.roll(P, -1, 0))
    tree = cKDTree(mid)
    pr = tree.query_pairs(r=cutoff, output_type='ndarray')
    if len(pr) == 0:
        return pr.reshape(0, 2)
    dd = np.abs(pr[:, 0] - pr[:, 1])
    dd = np.minimum(dd, N - dd)
    return pr[dd > w]

def pair_dmin(P, pairs):
    """Minimum segment-segment distance among a candidate pair list (inf
    if there are no candidate pairs at all). Used both as a contact-
    safety gauge (how close is the nearest potential self-intersection)
    and to size the anti-tunneling step cap in relax()."""
    if not len(pairs):
        return np.inf
    return float(seg_closest(P, pairs)[0].min())

# -------------------------------------------------------------- constraint

def project_edges(P, h, iters=1):
    """Position-based-dynamics (PBD) restoration of edge lengths to
    exactly h, in place, returning P.

    Each edge (P[j], P[j+1]) that has drifted to length l != h is
    corrected by moving both endpoints symmetrically along the edge
    direction by half the length error, which restores l = h exactly for
    that edge alone. Since adjacent edges share a vertex, fixing one edge
    can slightly perturb its neighbor's length — hence this is done in
    two interleaved passes over EVEN-indexed and ODD-indexed edges
    (`ev`, `od`): edges within one parity class share no vertices, so all
    corrections in a single pass can be applied simultaneously (vectorized
    over numpy arrays) without interfering with each other, and
    alternating the two parities lets the small residual errors between
    passes damp out. `iters` (default 1 in this function; relax() and
    project_constraints() may call it more than once, or interleave it
    with contact resolution) controls how many even/odd sweep pairs to
    run — more iterations converge closer to exact rest length but each
    one costs a bit of time, so it is a knob against per-step cost.
    """
    N = len(P)
    ev = np.arange(0, N, 2)
    od = np.arange(1, N, 2)
    for _ in range(iters):
        for idx in (ev, od):
            j = (idx + 1) % N
            d = P[j] - P[idx]
            l = np.linalg.norm(d, axis=1)
            corr = 0.5 * ((l - h) / l)[:, None] * d
            P[idx] += corr
            P[j] -= corr
    return P

def project_constraints(P, h, pairs, d0, iters=4):
    """Interleave edge-length restoration with hard non-penetration
    (segment-segment, corrections spread to endpoints by barycentric
    weights, Jacobi with 0.5 relaxation).  Narrows to near-active pairs
    after the first distance evaluation.

    Each iteration first calls project_edges() (restore inextensibility)
    and then, for every candidate pair still closer than d0 = 2*r (the
    "active" pairs), pushes the two segments apart along the contact
    normal `n` found by seg_closest(). Because a segment's closest point
    to another segment generally falls partway along it (at parameter s
    or t, not exactly at a vertex), the correction is distributed to the
    segment's two ENDPOINT vertices with barycentric weights (1-s, s) and
    (1-t, t) respectively — this is the standard way to apply a point
    constraint that acts at an interior point of an edge onto the
    vertices that actually parametrize it. The 0.25 factor (rather than a
    full push to exactly resolve the gap) is a Jacobi-style
    under-relaxation: since MULTIPLE overlapping active pairs can push the
    same vertex in different directions within one iteration, an
    under-relaxed step avoids overshooting and needs multiple `iters` to
    converge to a fully separated configuration, in exchange for numerical
    stability with many simultaneous contacts (a full-strength push per
    pair can lead to oscillation when several segments crowd one region,
    e.g. inside a tight rope-like bundle).

    Performance note: the FULL candidate `pairs` list from build_pairs()
    can be large (it includes near-misses that are not currently touching,
    kept around so the list stays valid as the curve moves slightly
    between neighbor-list rebuilds — see the "skin" margin in relax()).
    Recomputing exact segment-segment distances for that whole list every
    one of the `iters` inner iterations would be wasteful, since only the
    pairs that are ACTUALLY near (within 1.15*d0) at the start of this
    call can plausibly become active during it. So on the very first
    inner iteration (`it == 0`) the candidate list is narrowed down to
    just those near pairs, and the (typically much smaller) narrowed list
    is reused for the remaining iterations.
    """
    N = len(P)
    for it in range(iters):
        project_edges(P, h)
        if len(pairs):
            dist, s, t, n = seg_closest(P, pairs)
            if it == 0:
                near = dist < 1.15 * d0
                if not near.any():
                    pairs = pairs[:0]
                    continue
                pairs = pairs[near]
                dist, s, t, n = dist[near], s[near], t[near], n[near]
            act = dist < d0
            if act.any():
                i, j = pairs[act, 0], pairs[act, 1]
                push = (0.25 * (d0 - dist[act]))[:, None] * n[act]
                sa, ta = s[act, None], t[act, None]
                np.add.at(P, i, (1 - sa) * push)
                np.add.at(P, (i + 1) % N, sa * push)
                np.add.at(P, j, -(1 - ta) * push)
                np.add.at(P, (j + 1) % N, -ta * push)
    return P

# ------------------------------------------- knot determinant (Fox coloring)
#
# The determinant of a knot, det(K) = |Delta_K(-1)| (the Alexander
# polynomial evaluated at -1, in absolute value), is a classical
# invariant that is cheap to compute from any diagram of the knot and
# takes different values for almost all small knots of interest here
# (trefoil: 3, figure-eight: 5, 5_1: 5 (coincidence with 4_1, but not
# adjacent in this project's knot list), 8_18: 45, 10_123: 121, ... — see
# STATUS.md and the values printed/validated during development). It is
# used here purely as an "alarm bell": compute it before and after
# relaxing, and if it changes, some strand tunneled through another
# during the simulation (a bug), because the true continuous physical
# process being modeled — a wire moving through space without passing
# through itself — cannot change the knot type.

def _bareiss_det(M):
    """Exact integer determinant (fraction-free Gaussian elimination).

    Ordinary Gaussian elimination on an INTEGER matrix, tracking the
    exact rational pivots, would need fractions; the Bareiss algorithm
    instead updates every entry with (M[i][j]*M[k][k] - M[i][k]*M[k][j])
    and then does an EXACT integer division by the previous pivot
    (`prev`) — a classical fact about this update formula guarantees that
    division is always exact (no remainder), so the whole computation
    stays in exact integer arithmetic with no floating-point rounding
    error, which matters here because the input matrix entries are small
    integers (Fox coloring relations, see `_det_one_projection`) but
    determinants of moderately-sized crossing-number knots can already
    exceed float64 precision safely if we were to round -- exact integers
    avoid any doubt.
    """
    M = [row[:] for row in M]
    n = len(M)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for k in range(n - 1):
        if M[k][k] == 0:
            for r in range(k + 1, n):
                if M[r][k] != 0:
                    M[k], M[r] = M[r], M[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
        prev = M[k][k]
    return sign * M[n - 1][n - 1]

def _det_one_projection(P, rng):
    """Knot determinant from one generic projection; None if degenerate.

    Steps:
      1. Project the 3D curve onto a random plane (basis e1, e2 chosen
         perpendicular to a random viewing direction v) to get a planar
         diagram (X, Y), keeping the ORIGINAL 3D coordinate along v (Z) at
         each point so over/under can be resolved later.
      2. Find every pair of (non-adjacent) segments in this planar
         diagram that cross, using the standard 2D segment-intersection
         formula (`den`, `s`, `u`): for two segments starting at
         (X[i],Y[i]) and (X[j],Y[j]) with direction vectors (dx[i],dy[i])
         and (dx[j],dy[j]), solving for the two segment parameters at
         their intersection point reduces to this closed-form 2x2 solve
         vectorized over ALL pairs at once (`np.triu_indices`).
      3. A projection is REJECTED as "degenerate" (return None, so
         knot_det() will simply retry with a different random projection)
         if any crossing lands too close to a segment endpoint (would
         make the combinatorics ambiguous) or if two crossings happen to
         have nearly the same depth along v (the projection is
         accidentally near a triple point or a tangency) — the caller
         retries with a fresh random projection until it gets a clean
         one, and knot_det() requires it to agree on the SAME determinant
         from two independent generic projections in a row before trusting
         the value (protecting against any classification bug slipping
         through this heuristic degeneracy filter).
      4. Given C clean crossings, sort them by the arclength position of
         their UNDER-strand passage (`unders`, `u_sorted`) — this is the
         standard way to linearly order the crossings for building a knot
         diagram's Fox coloring / Wirtinger presentation matrix: walking
         along the strand, it passes UNDER at position u_sorted[q] and
         continues to the next arc q+1 (labelled by the arc it's on AFTER
         that undercrossing, cyclically).
      5. Build the C x C Fox coloring relation matrix M: at the crossing
         where arc q (incoming) becomes arc q+1 (outgoing) by passing
         under the OVER-strand belonging to arc `o` (found via
         `searchsorted` on the sorted under-positions — the arc whose
         under-crossing most recently preceded this crossing's over-
         passage), the standard relation is 2*x_o - x_{q-1} - x_q = 0 in
         the Z/n Fox coloring group presentation (colors must satisfy: at
         each crossing, twice the over-arc's color equals the sum of the
         two under-arc colors it separates). Each row of M encodes exactly
         one such relation.
      6. This presentation matrix always has rank C-1 (colorings have a
         1-parameter redundancy: shifting every color by a constant is
         always a solution), so one row and one column are dropped before
         taking the determinant (`M[:-1]` rows, `row[:-1]` columns) — the
         absolute value of the determinant of any such (C-1)x(C-1) minor
         is the knot determinant, independent of which row/column was
         dropped, which is why `abs()` is applied at the end (the sign is
         an artifact of the arbitrary minor choice / row ordering).

    Returns None (ask the caller to retry) if the projection is
    degenerate for any of the above reasons, or if the diagram somehow
    ends up with zero crossings on a supposedly nontrivial curve.
    """
    N = len(P)
    v = rng.standard_normal(3); v /= np.linalg.norm(v)
    a = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(v, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    X, Y, Z = P @ e1, P @ e2, P @ v
    x2, y2 = np.roll(X, -1), np.roll(Y, -1)
    dx, dy = x2 - X, y2 - Y
    # all segment pairs (vectorized 2D intersection)
    i, j = np.triu_indices(N, k=2)
    keep = ~((i == 0) & (j == N - 1))
    i, j = i[keep], j[keep]
    den = dx[i] * dy[j] - dy[i] * dx[j]
    rx, ry = X[j] - X[i], Y[j] - Y[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (rx * dy[j] - ry * dx[j]) / den
        u = (rx * dy[i] - ry * dx[i]) / den
    hit = (np.abs(den) > 1e-14) & (s > 0) & (s < 1) & (u > 0) & (u < 1)
    i, j, s, u = i[hit], j[hit], s[hit], u[hit]
    # degenerate if any crossing sits too close to a vertex or depths tie
    if len(s) and (s.min() < 1e-3 or s.max() > 1 - 1e-3 or
                   u.min() < 1e-3 or u.max() > 1 - 1e-3):
        return None
    z1 = Z[i] + s * (np.roll(Z, -1)[i] - Z[i])
    z2 = Z[j] + u * (np.roll(Z, -1)[j] - Z[j])
    if len(z1) and np.abs(z1 - z2).min() < 1e-7:
        return None
    C = len(i)
    if C == 0:
        return 1
    pos1, pos2 = i + s, j + u          # positions along the strand
    over1 = z1 > z2                    # passage 1 is the over-strand?
    unders = np.where(over1, pos2, pos1)
    overs = np.where(over1, pos1, pos2)
    order = np.argsort(unders)
    u_sorted = unders[order]
    # arc q runs from under q to under q+1 (cyclic); under q: in=q-1, out=q
    row_of = np.empty(C, int); row_of[order] = np.arange(C)
    M = [[0] * C for _ in range(C)]
    for c in range(C):
        q = row_of[c]
        o = (np.searchsorted(u_sorted, overs[c]) - 1) % C
        M[c][o] += 2
        M[c][(q - 1) % C] -= 1
        M[c][q % C] -= 1
    M = [row[:-1] for row in M[:-1]]
    return abs(_bareiss_det(M))

def knot_det(P, tries=8, seed=0):
    """Determinant with consensus over random projections (None if
    unstable).

    Calls _det_one_projection() with successive random projections
    (skipping degenerate ones, which return None) until two IN A ROW
    agree on the same value, and returns that value — a cheap but
    effective safeguard against trusting a fluke degenerate-but-not-
    detected projection. Returns None (meaning "could not confirm a
    determinant") if `tries` projections are exhausted without two
    consecutive agreements; callers (relax()) treat None as "skip the
    check" rather than as an error, since a knot determinant can
    legitimately be hard to pin down from a marginal curve (e.g. mid-
    relaxation with many nearly-coincident strands) without that being a
    sign anything is actually wrong.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(tries):
        d = _det_one_projection(P, rng)
        if d is not None:
            vals.append(d)
        if len(vals) >= 2 and vals[-1] == vals[-2]:
            return vals[-1]
    return None

# ------------------------------------------------------------ twist energy
#
# See the module docstring's "TWIST" paragraph for the physical picture:
# a welded wire loop's linking number Lk = Wr + Tw is fixed at the weld,
# so as the centerline's writhe Wr changes during relaxation, twist Tw
# must absorb the difference, at an elastic cost. This section computes
# writhe and its gradient with respect to vertex positions so that cost
# can be included as a force in relax() (see the `beta > 0` branch there).

def _wr_pairs(N):
    """Non-adjacent segment pairs (same quadrature as writhe()).

    Precomputes, ONCE per relax() call (not every step), the full list of
    index pairs (i, j) with j >= i+2 (and excluding the pair (0, N-1),
    which are actually ADJACENT through the closure) used by both
    writhe_grad() and the plain writhe() double sum. Because this is the
    complete O(N^2) pair list (unlike build_pairs()'s spatial cutoff for
    contact), it is only affordable because it's computed once and reused;
    the twist force evaluation itself (writhe_grad) is the expensive O(N^2)
    part done repeatedly (see `twist_update` in relax() for how its cost
    is amortized).
    """
    i, j = np.triu_indices(N, k=2)
    keep = ~((i == 0) & (j == N - 1))
    return i[keep], j[keep]

def _cross(u, v):
    """Vectorized 3D cross product, elementwise over the first axis.
    Equivalent to np.cross(u, v, axis=1) but noticeably faster on this
    project's (older, x86, non-Apple-Silicon) development machine — the
    generic np.cross has more axis-handling overhead than this hand-
    unrolled version, and this function is called inside the O(N^2)
    writhe_grad(), which matters for wall-clock time."""
    w = np.empty_like(u)
    w[:, 0] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    w[:, 1] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    w[:, 2] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return w

def writhe_grad(P, wi, wj, need_grad=True):
    """Discrete writhe (Gauss double sum over segment midpoints) and its
    analytic gradient wrt vertex positions.

    Writhe is defined by the Gauss linking-number-style double integral
    of a closed curve with itself:
        Wr = (1/4pi) * oint oint  (dr1 x dr2) . (r1-r2) / |r1-r2|^3
    Discretized here as a sum over all non-adjacent segment PAIRS (wi,
    wj from _wr_pairs), each segment represented by its direction vector
    (`a`, `b` = the edge vectors e[wi], e[wj]) and its MIDPOINT (`m`) as
    the representative point — a standard discretization for polygonal
    writhe (equivalent, up to the discretization scheme, to the plain
    writhe() function below, which instead uses one edge as `e[i]` fixed
    and sums over a range of `j`; both compute the same double sum, just
    organized differently — writhe_grad's fully-vectorized-over-all-pairs
    form is what makes an efficient analytic gradient practical, whereas
    writhe()'s Python-level loop over i is fine for a single evaluation
    but was not chosen as the basis for the gradient).

    When need_grad is True, the analytic gradient of Wr with respect to
    every vertex position is also returned, via the product/quotient rule
    applied to num/d3 = ((a x b).d) / |d|^3 for each pair, differentiated
    with respect to a, b, and d (which each depend linearly on up to 4 of
    the vertices: the two endpoints of segment wi and the two endpoints
    of segment wj, since a = P[wi+1]-P[wi], b = P[wj+1]-P[wj], and
    d = midpoint(wi) - midpoint(wj) = 0.5*(P[wi]+P[wi+1]-P[wj]-P[wj+1])).
    `Ga`, `Gb`, `Gd` are the partial derivatives of each pair's
    contribution with respect to a, b, and d respectively; the chain rule
    through a, b, d back to the 4 vertex positions gives the four
    `+-Ga/Gb +- 0.5*Gd` combinations scattered into the output via
    `np.bincount` (chosen over `np.add.at` purely for speed — see the
    inline comment; both compute the same scatter-add, but `add.at` was
    measured to be roughly 50x slower for this array size on this
    project's development hardware).

    Both the writhe value and the gradient were validated against
    numerical finite differences during development (see STATUS.md /
    project memory) to a relative error of order 1e-7, confirming the
    calculus above is implemented correctly.
    """
    N = len(P)
    e = np.roll(P, -1, 0) - P
    m = P + 0.5 * e
    a, b = e[wi], e[wj]
    d = m[wi] - m[wj]
    d2 = (d * d).sum(1)
    d3 = d2 * np.sqrt(d2)
    axb = _cross(a, b)
    num = (axb * d).sum(1)
    W = float((num / d3).sum()) / (2 * np.pi)
    if not need_grad:
        return W, None
    inv3 = (1.0 / d3)[:, None]
    Ga = _cross(b, d) * inv3
    Gb = _cross(d, a) * inv3
    Gd = axb * inv3 - (3 * num / (d3 * d2))[:, None] * d
    # bincount scatter (np.add.at is ~50x slower here)
    idx = np.concatenate([wi, (wi + 1) % N, wj, (wj + 1) % N])
    vals = np.concatenate([-Ga + 0.5 * Gd, Ga + 0.5 * Gd,
                           -Gb - 0.5 * Gd, Gb - 0.5 * Gd])
    G = np.stack([np.bincount(idx, weights=vals[:, c], minlength=N)
                  for c in range(3)], axis=1)
    return W, G / (2 * np.pi)

# ----------------------------------------------------------------- metrics
#
# Diagnostic quantities computed once at the end of each run (and folded
# into the JSON output) to characterize and classify the relaxed shape,
# used by make_summary.py's classify() to sort results into "flat coil" /
# "open 3D shape" / "rope-like skein" categories for the viewer.

def curvatures(P, h):
    """Per-vertex discrete curvature kappa_i = theta_i / (average of the
    two adjacent edge lengths) — the same turning angle used in
    bending(), just converted from a raw angle to a curvature (units of
    1/length) by dividing by the local arclength scale.  Used to report
    the [kappa_min, kappa_max] range in each run's summary: a very high
    kappa_max signals a sharply bent (nearly kinked) region, useful for
    spotting numerically marginal or physically extreme configurations
    (see STATUS.md's note about kappa_max ~ 450-500 runs being marginal
    at N=600 and wanting a finer discretization).
    """
    e = np.roll(P, -1, 0) - P
    le = np.linalg.norm(e, axis=1)
    u, lu = np.roll(e, 1, 0), np.roll(le, 1)
    c = np.clip((u * e).sum(1) / (lu * le), -1, 1)
    return np.arccos(c) / (0.5 * (lu + le))

def writhe(P):
    """Plain O(N^2) writhe evaluation (Gauss double sum), independent of
    writhe_grad()'s pair-precomputation path — kept as the simple,
    obviously-correct reference implementation used for final reporting
    (metrics()) and as the finite-difference cross-check target during
    development of writhe_grad(); the two are numerically equivalent
    (same discretization) but organized differently (a straightforward
    Python-level double loop here vs. the fully-vectorized pair-list form
    needed for an efficient gradient), so keeping both makes it easy to
    catch any regression in the fast path against the simple one.
    """
    N = len(P)
    e = np.roll(P, -1, 0) - P
    mid = P + 0.5 * e
    W = 0.0
    for i in range(N):
        j = np.arange(i + 2, N if i > 0 else N - 1)
        rij = mid[i] - mid[j]
        d3 = np.linalg.norm(rij, axis=1)**3
        cr = np.cross(e[i], e[j])
        W += np.sum((cr * rij).sum(1) / d3)
    return float(W / (2 * np.pi))

def gyration(P):
    """Principal radii of gyration: the square roots of the eigenvalues
    of the (mass-normalized) covariance matrix of the point cloud,
    sorted DESCENDING (largest spatial extent first). Used to describe
    the overall shape: e.g. planarity = gyr[2]/gyr[0] (smallest / largest)
    is near 0 for a flat coil (negligible extent along its normal
    direction) and the ratio gyr[1]/gyr[0] (second- to first-largest) is
    small for an elongated prolate shape like a rope-like skein — see
    make_summary.py's classify() for how these ratios are actually used
    to categorize each result.
    """
    Q = P - P.mean(0)
    ev = np.linalg.eigvalsh(Q.T @ Q / len(P))
    return np.sqrt(np.maximum(ev[::-1], 0))

def metrics(P, h, d0, w_excl):
    """Bundle of summary statistics computed once on the final relaxed
    curve: curvature range/mean, principal radii of gyration + derived
    planarity ratio, the fraction of vertices currently in (near-)contact
    with some other part of the wire (`contact_frac` — re-derives a fresh,
    slightly looser (1.15*d0) contact pair list here rather than reusing
    relax()'s working `pairs`, since that list's exact contents depend on
    the internal neighbor-list "skin" bookkeeping and may be stale by an
    amount that doesn't matter for the physics but would make this
    end-of-run diagnostic less precise), and the final writhe. All of
    this is what ends up in each result JSON's "metrics" field and is
    what make_summary.py's classify() reads to sort results into shape
    categories for the viewer.
    """
    kap = curvatures(P, h)
    g = gyration(P)
    pairs = build_pairs(P, 1.3 * d0 + h, w_excl)
    touched = np.zeros(len(P), bool)
    if len(pairs):
        dist = seg_closest(P, pairs)[0]
        near = pairs[dist < 1.15 * d0]
        touched[near[:, 0]] = True
        touched[near[:, 1]] = True
    return {
        'kappa_min': float(kap.min()), 'kappa_max': float(kap.max()),
        'kappa_mean': float(kap.mean()),
        'gyr': [float(x) for x in g],
        'planarity': float(g[2] / g[0]),
        'contact_frac': float(touched.mean()),
        'writhe': writhe(P),
    }

# -------------------------------------------------------------------- flow

def relax(p, b, r=0.001, N=600, steps=60000, seed=1, log=None, qstar_mult=5,
          kick_every=0, kick_amp=30.0, ic='weave', beta=0.0, tw0=0.0,
          twist_update=4, snug=False):
    """Run the full relaxation for one (p, b) Turk's head knot and return
    a result dict ready to be JSON-serialized.

    kick_every > 0 adds a smooth random perturbation of amplitude
    kick_amp*r every kick_every steps (a bench-vibration surrogate that
    lets the wire escape shallow metastable states).

    beta > 0 enables twist energy for a welded isotropic rod:
    E_t = 4 pi^2 beta (Lk - Wr)^2 with Lk = Wr(tied) + tw0 frozen at the
    weld (tw0 = turns of twist applied before joining; beta = GJ/EI,
    ~0.77 for round metal wire).  ic: 'weave' (jig-tied TH) or 'torus'
    (in-hand torus presentation).

    PARAMETERS not already covered by the module docstring or by
    docstrings of the constructors:
      r          physical wire radius, in the same length units as the
                 curve's total contour length (fixed to 1 by normalize());
                 sets the hard-contact distance d0 = 2*r and, together
                 with N, determines whether the discretization spacing
                 h = 1/N is fine enough to resolve features at scale r
                 (a very thin wire needs a correspondingly large N).
      N          number of vertices / edges in the discretized closed
                 curve; edge rest length h = 1/N.
      steps      maximum number of relaxation steps to run before giving
                 up on convergence (see the convergence check below).
      seed       random seed for both the initial-condition symmetry-
                 breaking noise and (offset by 999) the kick perturbations;
                 also passed to knot_det() for reproducible projection
                 sampling.
      log        if truthy, print periodic progress lines to stdout.
      qstar_mult sets q* = 2*pi*qstar_mult in precondition(); larger values
                 damp a wider range of high frequencies per step (more
                 conservative / slower net progress per step but more
                 numerically robust); rarely needs tuning from the default.
      twist_update  when beta > 0, the O(N^2) writhe GRADIENT (needed for
                 the twist force) is only recomputed every `twist_update`
                 steps rather than every step, since it is the single most
                 expensive operation in the loop and the twist force
                 changes slowly compared to the bending force; the writhe
                 VALUE itself (needed just for the energy, not a force) is
                 still recomputed every step from the post-move geometry
                 for a numerically consistent energy trace (see the
                 `Eb_new` twist term below), only the more expensive
                 gradient is throttled.

    THE MAIN LOOP, in the order operations happen each iteration:

      1. (Optional) KICK.  Every `kick_every` steps (but not within the
         last 25000 steps of the budget, so the run always ends on an
         actually-converged, un-kicked state), add a smooth random
         perturbation D and apply it in small substeps (each capped at
         r/4) with a strong (`iters=8`) contact projection interleaved
         after every substep — this keeps even a fairly large kick from
         ever letting one segment jump clean through another before
         contact resolution gets a chance to see it. After the full kick,
         extra projection sweeps run (up to 10 more rounds) specifically
         until the minimum pair separation gmin recovers to at least
         0.95*d0 (i.e. contact is genuinely, not just approximately,
         resolved) — and if it can't get there, OR if the knot
         determinant of the kicked state no longer matches the
         pre-relaxation determinant det0 (checked with a cheap `tries=4`
         call, since this runs potentially many times per relaxation and
         a full 8-try check would be wasteful here), the ENTIRE kick is
         reverted (`P = Psave`) and a message is logged. This reject-on-
         violation design is what makes kicks safe to use aggressively:
         a kick that would have caused tunneling simply doesn't happen.

      2. FORCE ASSEMBLY.  Compute the bending force Fb. If twist is
         enabled (beta > 0), also compute (or reuse, per `twist_update`)
         the writhe gradient and add the twist force
         Ft = 8*pi^2*beta*(Lk - Wr_cur)*gradient(Wr) — this is exactly
         -d/dP [4*pi^2*beta*(Lk-Wr)^2], i.e. the negative gradient
         (a physical force) of the twist energy from the module
         docstring, using the chain rule through Wr(P).

      3. PRECONDITION + PROJECT.  Run the combined force through the
         Sobolev preconditioner and then tangent_project() to get a
         inextensibility-respecting descent direction `d`.

      4. STEP-SIZE CAP (anti-tunneling).  `s_max` is the tightest of: 2h
         (never move more than 2 edge-lengths in one step, a basic CFL-
         like sanity bound), or the LARGER of r/2 (always allow at least
         a modest fraction of the wire radius per step so relaxation
         doesn't stall completely near contact) and a distance-based
         budget `(gmin - d0 - 2*acc_disp)/4` derived from how much "room"
         remains before the nearest candidate pair would actually reach
         contact distance d0, accounting for how much accumulated
         movement (`acc_disp`, see step 6) has already eaten into the
         neighbor-list's safety skin since it was last rebuilt. This is
         the crucial anti-tunneling safeguard described in the module
         docstring: it guarantees no single step can move any vertex far
         enough to jump past a segment that current information says is
         nearby, even though the true adaptive step size `s` (see step 5)
         is otherwise free to grow much larger when nothing is nearby.

      5. TAKE THE STEP, then PROJECT CONSTRAINTS (restore exact edge
         lengths and resolve any contact violation) with the standard,
         lighter-weight `iters=4` used in normal (non-kick) stepping.

      6. ADAPT STEP SIZE.  Recompute bending (+ twist, if enabled) energy
         at the new position; if energy decreased, grow `s` by 10% (up to
         a cap of 2h); if it increased, shrink `s` by half (down to a
         floor `s_min`) — a simple, robust "growing/shrinking step size"
         line-search surrogate, cheaper than an actual line search and
         adequate since the preconditioned direction is already a good
         descent direction. `acc_disp` (the amount of movement since the
         neighbor-list `pairs` was last rebuilt) accumulates by `min(s,
         s_max)`; once it exceeds half the neighbor-list "skin" margin,
         `pairs` is rebuilt from the current geometry (a fresh KD-tree
         query) and `acc_disp` resets to 0 — this is the standard
         Verlet-list-style bookkeeping that lets contact pairs be found
         with a spatial cutoff query only occasionally rather than every
         step, while still being conservative enough that a real new
         contact can never appear between rebuilds without being caught
         (because the skin margin was sized, via `cutoff` and `w_excl`
         above, to cover exactly the maximum possible movement before the
         next scheduled rebuild). A WARNING is printed if the true
         minimum pair distance at rebuild time is ever found to have
         dropped below 0.9*r — this should never happen given the
         anti-tunneling step cap and is a signal something is
         numerically wrong if it does.

      7. CONVERGENCE CHECK (every 100 steps, only after step 8000, only
         once at least 6000 steps have passed since the last kick, and
         only once the adaptive step size `s` has shrunk to near its
         floor `s_min` — all three gates exist because a run can
         temporarily look "flat" in energy while still slowly working
         through a shallow decline, a fresh kick's aftermath, or a large
         step still oscillating around a not-yet-found minimum, none of
         which should be mistaken for true convergence). Convergence is
         declared, and the loop breaks, once the energy from ~3000 steps
         ago differs from the current energy by less than 0.2% of the
         current energy (accounting for typical residual "contact
         chatter" fluctuation rather than requiring the energy trace to
         be perfectly flat) — `conv_at` records the step this happened
         at (None if the `steps` budget was exhausted first).

    FINALIZATION.  After the loop (whether by convergence or by
    exhausting `steps`), recompute the final bending energy and knot
    determinant `det1`; print a WARNING if `det1` disagrees with the
    pre-relaxation `det0` (see the module docstring's "VERIFICATION"
    section — this is the primary correctness safeguard for the whole
    project). Assemble and return the result dict: the (p, b, r, N, seed,
    ic, beta, tw0) run parameters, the frozen linking number Lk and
    initial/final writhe, the final twist energy, step/convergence
    bookkeeping, the final bending energy (both raw and normalized by
    4*pi^2 — a natural comparison unit since a bare circle of length 1
    has exactly this bending energy, so E_bend_over_4pi2 is roughly "how
    many circles' worth of bending energy" the shape costs, e.g. ~m^2 for
    an m-times-covered circle), the determinant check results, wall-clock
    time, a downsampled energy history (last 50 recorded points, each 100
    steps apart) for inspecting the convergence trace, the metrics()
    bundle, and finally the full list of relaxed vertex positions
    (rounded to 6 decimals to keep the JSON file size reasonable) for
    rendering by snapshot.py / viewer.html.
    """
    h = 1.0 / N
    rng_kick = np.random.default_rng(seed + 999)
    d0 = 2 * r
    w_excl = max(2, int(1.5 * d0 / h) + 1)
    qstar = 2 * np.pi * qstar_mult
    if ic == 'weave':
        P0 = th_weave(p, b, N, seed=seed, snug=snug, r_wire=r)
    elif ic == 'coil':
        P0 = th_coil(p, b, N, r_wire=r, seed=seed)
    else:
        P0 = torus_knot(p, b, N, seed=seed)
    P = normalize(resample_closed(P0, N))
    P = project_edges(P, h, iters=40)
    det0 = knot_det(P, seed=seed)
    wi, wj = _wr_pairs(N)
    Wr0, _ = writhe_grad(P, wi, wj, need_grad=False)
    Lk = Wr0 + tw0
    Ft = np.zeros_like(P)
    Wr_cur = Wr0

    skin = max(2 * d0, 2 * h)
    cutoff = d0 + skin + h
    pairs = build_pairs(P, cutoff, w_excl)
    gmin = pair_dmin(P, pairs)
    acc_disp = 0.0

    s = h / 4
    s_min = h / 400
    Eb, _ = bending(P, h)
    Eb += 4 * np.pi**2 * beta * (Lk - Wr0)**2
    E_hist = [[0, Eb]]
    conv_at = None
    last_kick = 0
    t0 = time.time()
    for step in range(steps):
        if kick_every and step and step % kick_every == 0 \
                and step < steps - 25000:
            Psave = P.copy()
            t = np.arange(N) / N
            D = np.zeros_like(P)
            for k in range(1, 5):
                D += (kick_amp * r / k) * np.cos(
                    2 * np.pi * k * t[:, None] + rng_kick.uniform(0, 2 * np.pi, 3)) \
                    * rng_kick.standard_normal(3)
            # substeps capped at r/4 with strong projection between
            nsub = max(1, int(np.ceil(np.linalg.norm(D, axis=1).max() / (r / 4))))
            for _ in range(nsub):
                P = P + D / nsub
                pairs = build_pairs(P, cutoff, w_excl)
                P = project_constraints(P, h, pairs, d0, iters=8)
            # extra projection sweeps until overlap is fully resolved
            for _ in range(10):
                gmin = pair_dmin(P, pairs)
                if gmin >= 0.95 * d0:
                    break
                P = project_constraints(P, h, pairs, d0, iters=8)
                pairs = build_pairs(P, cutoff, w_excl)
            # reject the kick unless separation is clean and the knot unchanged
            if gmin < 0.95 * d0 or (det0 is not None and
                                    knot_det(P, tries=4, seed=step) != det0):
                P = Psave
                pairs = build_pairs(P, cutoff, w_excl)
                gmin = pair_dmin(P, pairs)
                if log:
                    print(f'  kick at {step} rejected', flush=True)
            acc_disp = 0.0
            last_kick = step
        _, Fb = bending(P, h)
        if beta > 0:
            if step % twist_update == 0:
                Wr_cur, gW = writhe_grad(P, wi, wj)
            # else: Wr_cur carried over from the acceptance evaluation below
            Ft = 8 * np.pi**2 * beta * (Lk - Wr_cur) * gW
            Fb = Fb + Ft
        d = precondition(Fb, qstar)
        d = tangent_project(d, P)
        m = float(np.linalg.norm(d, axis=1).max())
        # anti-tunneling cap: r/2 near contact, up to 2h when strands are far
        s_max = min(2 * h, max(r / 2, (gmin - d0 - 2 * acc_disp) / 4))
        if m > 0:
            P = P + d * (min(s, s_max) / m)
        P = project_constraints(P, h, pairs, d0)

        Eb_new, _ = bending(P, h)
        if beta > 0:
            Wr_cur, _ = writhe_grad(P, wi, wj, need_grad=False)
            Eb_new += 4 * np.pi**2 * beta * (Lk - Wr_cur)**2
        s = min(s * 1.1, 2 * h) if Eb_new < Eb else max(s * 0.5, s_min)
        Eb = Eb_new

        acc_disp += min(s, s_max)
        if acc_disp > skin / 2:
            pairs = build_pairs(P, cutoff, w_excl)
            acc_disp = 0.0
            gmin = pair_dmin(P, pairs)
            if gmin < 0.9 * d0 / 2:
                print(f'  WARNING step {step}: min separation {gmin:.4g} < 0.9r', flush=True)

        if step % 100 == 0:
            E_hist.append([step, Eb])
            if log and step % 4000 == 0:
                print(f'  {step:6d}  Eb={Eb:10.3f}  s={s:.2e}  gmin={gmin:.4f}', flush=True)
            if step >= 8000 and step - last_kick > 6000 and s <= 2 * s_min:
                Eref = next(e for st, e in E_hist if st >= step - 3000)
                if abs(Eref - Eb) < 2e-3 * abs(Eb):
                    conv_at = step
                    break

    Eb, _ = bending(P, h)
    det1 = knot_det(P, seed=seed + 100)
    if det0 is not None and det1 is not None and det0 != det1:
        print(f'  WARNING: determinant changed {det0} -> {det1} (knot type NOT preserved!)',
              flush=True)
    Wr_fin, _ = writhe_grad(P, wi, wj, need_grad=False)
    res = {
        'p': p, 'b': b, 'r': r, 'N': N, 'seed': seed,
        'ic': ic, 'beta': beta, 'tw0': tw0, 'Lk': Lk,
        'Wr_initial': Wr0, 'Wr_final': Wr_fin,
        'E_twist': 4 * math.pi**2 * beta * (Lk - Wr_fin)**2,
        'steps_run': step + 1, 'converged_at': conv_at,
        'E_bend': Eb, 'E_bend_over_4pi2': Eb / (4 * math.pi**2),
        'det_initial': det0, 'det_final': det1,
        'wall_s': round(time.time() - t0, 1),
        'E_hist': E_hist[-50:],
        'metrics': metrics(P, h, d0, w_excl),
        'points': [[round(float(x), 6) for x in row] for row in P],
    }
    return res

# ------------------------------------------------ general multi-loop relax
#
# relax() above only ever builds its OWN initial curve from (p, b) via
# th_weave/th_coil/torus_knot — it has no notion of an arbitrary knot's
# crossing diagram. This section adds a second entry point, relax_general(),
# that instead accepts ANY closed curve (or several, for links) as raw
# input — used by server.py to relax whatever the browser's Knot view is
# currently displaying, Turk's-head or general C=2/C=3 catalog knot alike,
# rather than only the (p, b)-parametrized family.
#
# Scope, deliberately: bending + inextensibility + hard contact only, same
# as relax()'s beta=0 default — NO twist support here. Generalizing twist
# to several independently-closed loops needs mutual linking-number cross
# terms between components (not just each loop's own writhe), which is out
# of scope for this general-purpose entry point; relax()'s (p, b) path
# remains the place for the validated, twist-calibrated TH experiments.
# Similarly, knot-determinant verification (knot_det) only makes sense for
# a single closed curve, so it's applied when there's exactly one loop and
# skipped (not an error, just unavailable) for multi-component links.
#
# Unlike relax(), coordinates are NOT rescaled to unit total length: the
# caller's units are used as-is (the browser sends its three.js world
# coordinates, with r already in those same units — see index.html's
# getKnotPointsForRelax()), only recentered at the combined centroid for
# numerical hygiene.
#
# Reuses relax()'s already-validated per-loop building blocks (bending,
# precondition, tangent_project, project_edges, resample_closed, knot_det)
# completely UNCHANGED — each operates on whatever single array it's given,
# so calling them once per loop in a Python loop is already correct; only
# CONTACT needs new code, since it must consider segment pairs from
# DIFFERENT loops, which the original build_pairs/seg_closest/
# project_constraints (single %N-cyclic array) cannot express.

def _pack_loops(loops):
    """Concatenate a list of (Ni,3) loop arrays into one (sum Ni,3) array,
    plus a `nxt` index array giving, for each global point, the global
    index of the NEXT point along its OWN loop (wrapping within that loop,
    not into the next one) — the multi-loop generalization of the `%N`
    cyclic indexing every single-loop function here relies on."""
    starts = np.concatenate([[0], np.cumsum([len(l) for l in loops])]).astype(int)
    Pall = np.concatenate(loops, axis=0)
    nxt = np.arange(len(Pall)) + 1
    for s0, s1 in zip(starts[:-1], starts[1:]):
        nxt[s1 - 1] = s0
    return Pall, nxt, starts

def _unpack_loops(Pall, starts, loops):
    """Inverse of _pack_loops: scatter a (possibly modified) packed array
    back into the original per-loop arrays, in place."""
    for k in range(len(loops)):
        loops[k][:] = Pall[starts[k]:starts[k + 1]]

def seg_closest_multi(P, nxt, pairs):
    """Exactly seg_closest()'s math (see that function's docstring for the
    derivation), generalized to a packed multi-loop array: segment i is
    (P[i], P[nxt[i]]) instead of (P[i], P[(i+1)%N])."""
    i, j = pairs[:, 0], pairs[:, 1]
    a0, a1 = P[i], P[nxt[i]]
    b0, b1 = P[j], P[nxt[j]]
    d1, d2 = a1 - a0, b1 - b0
    r12 = a0 - b0
    a = (d1 * d1).sum(1); e = (d2 * d2).sum(1)
    f = (d2 * r12).sum(1); c = (d1 * r12).sum(1)
    bb = (d1 * d2).sum(1)
    denom = a * e - bb * bb
    s = np.where(denom > 1e-18 * a * e, (bb * f - c * e) / np.maximum(denom, 1e-30), 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = np.clip((bb * s + f) / e, 0.0, 1.0)
    s = np.clip((bb * t - c) / a, 0.0, 1.0)
    pa = a0 + s[:, None] * d1
    pb = b0 + t[:, None] * d2
    n = pa - pb
    dist = np.linalg.norm(n, axis=1)
    n = n / np.maximum(dist, 1e-30)[:, None]
    return dist, s, t, n

def build_pairs_multi(loops, cutoff, w_excl):
    """Multi-loop generalization of build_pairs(): a KD-tree over ALL
    segment midpoints from every loop combined, excluding pairs that are
    (a) in the same loop AND within that loop's own w_excl[k] of each
    other along the strand (cyclic within that loop) — cross-loop pairs
    are never excluded, since two different components can legitimately
    touch anywhere. Returns (pairs, Pall, nxt, starts) — the packed arrays
    are returned too since the caller (the step loop below) needs them
    again immediately for the gmin/anti-tunneling calculation."""
    Pall, nxt, starts = _pack_loops(loops)
    N = len(Pall)
    mid = 0.5 * (Pall + Pall[nxt])
    tree = cKDTree(mid)
    pr = tree.query_pairs(r=cutoff, output_type='ndarray')
    if len(pr) == 0:
        return pr.reshape(0, 2), Pall, nxt, starts
    sizes = np.diff(starts)
    loop_id = np.zeros(N, dtype=int)
    local_idx = np.zeros(N, dtype=int)
    for k in range(len(loops)):
        loop_id[starts[k]:starts[k + 1]] = k
        local_idx[starts[k]:starts[k + 1]] = np.arange(sizes[k])
    li, lj = loop_id[pr[:, 0]], loop_id[pr[:, 1]]
    same = li == lj
    dd = np.abs(local_idx[pr[:, 0]] - local_idx[pr[:, 1]])
    dd = np.minimum(dd, sizes[li] - dd)
    w = np.asarray(w_excl)[li]
    keep = (~same) | (dd > w)
    return pr[keep], Pall, nxt, starts

def project_constraints_multi(loops, hs, pairs, d0, iters=4):
    """Multi-loop generalization of project_constraints(): per-loop
    edge-length restoration (project_edges, unchanged) interleaved with
    contact push-apart resolved across the packed combined array, scattered
    back into the individual loop arrays after each sub-iteration (see
    _pack_loops/_unpack_loops). Narrows to near-active pairs after the
    first pass, exactly like the single-loop version — but, matching
    project_constraints() exactly, that narrowing is LOCAL to this call
    only (the parameter `pairs` is rebound, not mutated): the caller's own
    wide, skin-margin-inclusive candidate list must persist across many
    steps until relax_general() explicitly rebuilds it, so callers must
    NOT do `pairs = project_constraints_multi(...)` — this returns nothing
    meaningful; `loops` is updated in place."""
    for it in range(iters):
        for k in range(len(loops)):
            project_edges(loops[k], hs[k])
        if len(pairs) == 0:
            continue
        Pall, nxt, starts = _pack_loops(loops)
        dist, s, t, n = seg_closest_multi(Pall, nxt, pairs)
        if it == 0:
            near = dist < 1.15 * d0
            if not near.any():
                pairs = pairs[:0]
                continue
            pairs = pairs[near]
            dist, s, t, n = dist[near], s[near], t[near], n[near]
        act = dist < d0
        if act.any():
            i, j = pairs[act, 0], pairs[act, 1]
            push = (0.25 * (d0 - dist[act]))[:, None] * n[act]
            np.add.at(Pall, i, (1 - s[act, None]) * push)
            np.add.at(Pall, nxt[i], s[act, None] * push)
            np.add.at(Pall, j, -(1 - t[act, None]) * push)
            np.add.at(Pall, nxt[j], -t[act, None] * push)
            _unpack_loops(Pall, starts, loops)

def relax_general(loops0, r=0.001, steps=60000, log=None, qstar_mult=5,
                   point_budget=800, min_per_loop=60):
    """General elastic relaxation of an arbitrary curve (or several, for
    links) — see this section's header comment for scope and how it
    differs from relax(). loops0: list of (Ni,3) arrays, raw (not
    necessarily uniformly resampled) initial curves in the caller's own
    units; r: wire radius in those same units.

    Returns a result dict analogous to relax()'s, generalized to a list of
    point arrays: 'loops' (final points per component), 'E_bend',
    'E_bend_over_4pi2', 'steps_run', 'converged_at', 'det_initial'/
    'det_final' (None when there's more than one loop), 'wall_s'.
    """
    nloops = len(loops0)
    d0 = 2 * r
    all0 = np.concatenate(loops0, axis=0)
    centroid = all0.mean(axis=0)
    per_loop = max(min_per_loop, point_budget // nloops)

    loops, hs, w_excl = [], [], []
    for P0 in loops0:
        P = resample_closed(P0 - centroid, per_loop)
        h = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1).sum() / len(P)
        P = project_edges(P, h, iters=40)
        loops.append(P); hs.append(h)
        w_excl.append(max(2, int(1.5 * d0 / h) + 1))

    det0 = knot_det(loops[0]) if nloops == 1 else None

    qstar = 2 * np.pi * qstar_mult
    hMin0 = min(hs)
    skin = max(2 * d0, 2 * hMin0)
    cutoff = d0 + skin + hMin0
    pairs, Pall, nxt, _ = build_pairs_multi(loops, cutoff, w_excl)
    gmin = float(seg_closest_multi(Pall, nxt, pairs)[0].min()) if len(pairs) else np.inf
    acc_disp = 0.0

    s = hMin0 / 4
    s_min = hMin0 / 400
    Eb = sum(bending(P, h)[0] for P, h in zip(loops, hs))
    E_hist = [[0, Eb]]
    conv_at = None
    t0 = time.time()
    for step in range(steps):
        dirs = []
        for P, h in zip(loops, hs):
            _, F = bending(P, h)
            d = precondition(F, qstar)
            d = tangent_project(d, P)
            dirs.append(d)
        m = max(float(np.linalg.norm(d, axis=1).max()) for d in dirs)
        hMin = min(hs)
        s_max = min(2 * hMin, max(r / 2, (gmin - d0 - 2 * acc_disp) / 4))
        step_size = min(s, s_max)
        if m > 0:
            for P, d in zip(loops, dirs):
                P += d * (step_size / m)
        project_constraints_multi(loops, hs, pairs, d0)   # mutates loops in place

        Eb_new = sum(bending(P, h)[0] for P, h in zip(loops, hs))
        s = min(s * 1.1, 2 * hMin) if Eb_new < Eb else max(s * 0.5, s_min)
        Eb = Eb_new

        acc_disp += min(s, s_max)
        if acc_disp > skin / 2:
            pairs, Pall, nxt, _ = build_pairs_multi(loops, cutoff, w_excl)
            acc_disp = 0.0
            gmin = float(seg_closest_multi(Pall, nxt, pairs)[0].min()) if len(pairs) else np.inf
            if gmin < 0.9 * d0 / 2:
                print(f'  WARNING step {step}: min separation {gmin:.4g} < 0.9r', flush=True)

        if step % 100 == 0:
            E_hist.append([step, Eb])
            if log and step % 4000 == 0:
                print(f'  {step:6d}  Eb={Eb:10.3f}  s={s:.2e}  gmin={gmin:.4f}', flush=True)
            if step >= 8000 and s <= 2 * s_min:
                Eref = next(e for st, e in E_hist if st >= step - 3000)
                if abs(Eref - Eb) < 2e-3 * abs(Eb):
                    conv_at = step
                    break

    Eb = sum(bending(P, h)[0] for P, h in zip(loops, hs))
    det1 = knot_det(loops[0]) if nloops == 1 else None
    if det0 is not None and det1 is not None and det0 != det1:
        print(f'  WARNING: determinant changed {det0} -> {det1} (knot type NOT preserved!)',
              flush=True)
    return {
        'loops': [[[round(float(x), 6) for x in row] for row in P] for P in loops],
        'r': r, 'steps_run': step + 1, 'converged_at': conv_at,
        'E_bend': Eb, 'E_bend_over_4pi2': Eb / (4 * math.pi**2),
        'det_initial': det0, 'det_final': det1,
        'wall_s': round(time.time() - t0, 1),
    }

# -------------------------------------------------------------------- main

# 2-lead knots validated against experiment and retired from the suite
# (see STATUS.md: TH(2,b) knots reliably flatten to a doubly-covered
# circle and matching the bench was confirmed early; --sweep now only
# iterates the more interesting/uncertain 3+-lead cases below).
SWEEP = [(3, 2), (3, 4), (3, 5), (3, 7), (3, 8),
         (4, 3), (4, 5), (4, 7),
         (5, 2), (5, 3), (5, 4), (5, 6),
         (7, 2), (7, 3), (8, 3), (9, 2), (9, 4)]

def run_one(p, b, args, outdir):
    """Run relax() for a single (p, b) pair using the parsed CLI `args`,
    print a one-line human-readable summary, and write the full result
    dict as JSON to `outdir`. The output filename encodes every
    parameter that affects the physics (radius, seed, kick schedule,
    initial-condition variant, twist turns, snug flag) so that different
    variants of the "same" (p, b) knot never collide on disk — this
    naming convention is also what make_summary.py and snapshot.py rely
    on to label results in the viewer and in filenames.
    """
    kick = f'_k{args.kick_every}' if args.kick_every else ''
    tag = ('' if args.ic == 'weave' else '_' + args.ic) + \
          (f'_tw{args.tw0:g}' if args.beta > 0 else '') + \
          ('_snug' if args.snug else '')
    name = f'th_{p}L{b}B_r{args.r}_s{args.seed}{kick}{tag}'
    print(f'== {p} leads x {b} bights  (r={args.r}, N={args.N}, seed={args.seed}{kick}{tag})',
          flush=True)
    res = relax(p, b, r=args.r, N=args.N, steps=args.steps, seed=args.seed, log=True,
                kick_every=args.kick_every, kick_amp=args.kick_amp,
                ic=args.ic, beta=args.beta, tw0=args.tw0, snug=args.snug)
    with open(os.path.join(outdir, name + '.json'), 'w') as f:
        json.dump(res, f)
    m = res['metrics']
    print(f'   done in {res["wall_s"]}s steps={res["steps_run"]} conv={res["converged_at"]}\n'
          f'   det {res["det_initial"]} -> {res["det_final"]}  '
          f'Eb={res["E_bend"]:.2f} (={res["E_bend_over_4pi2"]:.3f} x 4pi^2)  '
          f'planarity={m["planarity"]:.4f}  contact={m["contact_frac"]:.2f}  '
          f'Wr={m["writhe"]:.2f}  kappa=[{m["kappa_min"]:.1f},{m["kappa_max"]:.1f}]', flush=True)

if __name__ == '__main__':
    # NOTE: this CLI runs each requested knot SEQUENTIALLY in a single
    # process. For running many knots at once across CPU cores, use
    # sweep_parallel.py instead — it wraps the same relax() function in a
    # ProcessPoolExecutor and accepts the same physics flags.
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=int)
    ap.add_argument('--b', type=int)
    ap.add_argument('--r', type=float, default=0.001)
    ap.add_argument('--N', type=int, default=600)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--kick-every', type=int, default=0)
    ap.add_argument('--kick-amp', type=float, default=30.0)
    ap.add_argument('--ic', choices=['weave', 'torus', 'coil'], default='weave')
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--tw0', type=float, default=0.0)
    ap.add_argument('--snug', action='store_true',
                    help='as-tied initial state: narrow snug band on the jig')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--from-idx', type=int, default=0)
    ap.add_argument('--to-idx', type=int, default=len(SWEEP))
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(outdir, exist_ok=True)
    if args.sweep:
        for (p, b) in SWEEP[args.from_idx:args.to_idx]:
            run_one(p, b, args, outdir)
    else:
        if args.p is None or args.b is None:
            ap.error('need --p and --b (or --sweep)')
        run_one(args.p, args.b, args, outdir)
