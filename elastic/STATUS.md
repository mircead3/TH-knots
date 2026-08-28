# Elastic-wire TH knots — status (2026-08-18)

Simulating springy-wire Turk's head knots (welded loop) and comparing against
physical experiments. Model: bending energy + inextensibility + hard self-contact
at wire radius r (default 0.001·L); optional twist term; no friction/gravity.

## Files
- `relax.py` — simulator. Key flags: `--p --b` (leads/bights), `--r`, `--N`,
  `--ic weave|torus|coil`, `--snug` (as-tied band on jig), `--kick-every/--kick-amp`
  (vibration surrogate, tunnel-safe), `--beta --tw0` (twist: E=4pi^2*beta*(Lk-Wr)^2,
  Lk frozen at weld; beta≈0.77 round wire). Knot type verified every run via Fox
  determinant of initial+final curve (`knot_det`); mismatch = strand tunneling = bug.
- `sweep_parallel.py` — one knot per process. Wrap in `caffeinate -i` (Mac sleeps!).
- `make_summary.py` — rebuilds `results.js` for `viewer.html` (open in Safari).
- `snapshot.py` — 3-view PNGs into `snaps/`.
- `results/` — ONE canonical run per knot (20 knots). `results_archive/` — variants
  (torus calibration, failed jig-twist, superseded ICs). `results_torusIC_r0.005/` —
  earliest runs (wrong ICs for 3+ leads; torus knots, not THs).

## Bench protocol (user)
Tied snug on a cylindrical jig, ends crimped. Then opened by hand into an equalized
coil and released: B>L stays flat coil; B<L<2B springs to symmetric 3D shape;
L>2B springs to tight rope-like skein (tight core, B-fold + 2-fold symmetric).

## Matches (canonical results)
- B>L → flat coils ✓ (3L4B needs the coil/hand-equalized start — `--ic coil` — or kicks).
- B<L<2B → open 3D ✓ (3L2B=fig-8 Wr=0, 4L3B basket, 5L3B, 5L4B).
- L>2B → snug-release (`--snug`) gives symmetric elongated skeins; 9L4B "very close
  to the physical knot" (user). In-hand torus knots (`--ic torus` + twist): coil ↔
  lobed-3D switch reproduced, validating the twist term.

## Open items
1. 5L6B: sim insists on a wavy wreath from EVERY IC/protocol (loose, snug, coil,
   kicks, twist). Bench flat coil may be friction-held (user: "badly in need" of the
   hand-equalization step). Bench test: vibrate a bench 5x6 coil — does it creep to
   a wreath?
2. Rope cores not "tight everywhere"; end loops splay; B-fold symmetry only partial.
   Missing squeeze: friction arrest (not modeled) or small tying twist.
   DECISIVE bench test pending: marker line on wire before tying — spiral turns = Tw0
   (then rerun ropes with `--beta 0.77 --tw0 <measured>`).
3. 7L3B central part doesn't bundle (L=2B+1 boundary case).
4. Jig-twist tw0=+1/wrap ruled out (disordered balls). Lobe-count in twist calibration:
   sim gives `bights` lobes, user remembers n lobes for n×(n+1) — verify on bench.
5. kappa_max~450 runs are marginal at N=600; use N=900 for p≥7 (slow Intel i5 Mac,
   ~30-60s/knot typical, ropes minutes).

## Continuing on a smaller model
Everything needed is in this file + the code. Typical commands:
  caffeinate -i python3 sweep_parallel.py 9,2 --snug --N 900 --jobs 1
  python3 make_summary.py && python3 snapshot.py
Don't trust any run whose det changed; don't add raw np.add.at over big index
arrays (very slow here); keep per-step displacement caps (tunneling safety).
