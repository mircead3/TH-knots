#!/usr/bin/env python3
"""Render result JSONs to small PNGs (3 orthographic views along the
gyration principal axes) for quick shape inspection. No deps beyond numpy.

viewer.html gives an interactive, rotatable 3D view of any single result
and is the right tool for actually inspecting a shape closely, but it
needs a browser (see that file's own comments on the Safari-vs-Chrome
WebGL issue on this project's development machine) and only shows one
result at a time. This script instead produces a flat, dependency-free
static PNG per result — three fixed orthographic projections side by
side — cheap enough to generate for every result in results/ in one
shell command and then look at many of them in a row (e.g. via this
project's Read tool, or any image viewer) without launching a browser or
picking knots one at a time from a dropdown. It is the primary way shape
regressions and new-result sanity checks were done throughout this
project's development (see STATUS.md).

Deliberately implemented with a hand-rolled minimal PNG writer
(write_png()) instead of depending on matplotlib or Pillow: this
project's Python environment does not have matplotlib installed (see
STATUS.md), and adding a new dependency just for quick-look PNGs was not
worth it when an uncompressed-but-zlib-DEFLATEd truecolor PNG is only a
few dozen lines of code to emit directly.

Usage:
  python3 snapshot.py                          # every file in results/
  python3 snapshot.py results/th_9L4B_*.json   # (shell-glob-expanded) subset
  python3 snapshot.py 'results/*coil*.json' 'results/*snug*.json'  # (quoted,
                                                # script-side glob) subsets
Output: one PNG per input JSON, same basename, into elastic/snaps/.
"""
import glob, json, os, struct, sys, zlib
import numpy as np

def write_png(path, img):          # img: (H,W,3) uint8
    """Minimal, dependency-free PNG encoder for a single RGB truecolor
    image (no palette, no interlacing, no filtering beyond the mandatory
    per-scanline filter-type byte, which is always 0/"None" here — see
    the `b'\\x00' + ...` prefix on every row). Just enough of the PNG
    spec to produce a valid file: signature, IHDR (image header: width,
    height, 8-bit depth, color type 2 = truecolor RGB), IDAT (the zlib-
    DEFLATE-compressed raw scanline data), IEND. Each chunk is written as
    length + type + data + CRC32-of-(type+data), exactly per the PNG
    chunk format, computed inline by the local `chunk()` helper.
    """
    H, W, _ = img.shape
    raw = b''.join(b'\x00' + img[y].tobytes() for y in range(H))
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    open(path, 'wb').write(png)

def render(P, r_wire, size=300):
    """Render one relaxed curve P (an (N,3) array of points) as a
    horizontal strip of 3 fixed orthographic projections, each `size`
    pixels square, and return the combined (size, 3*size, 3) uint8 image.

    The three views are chosen not as arbitrary world-axis projections
    but along the curve's OWN principal axes of gyration (`evec`, found
    by eigendecomposing the point cloud's covariance matrix P.T@P,
    ascending eigenvalues, then reversed with `[:, ::-1]` so the first
    axis is the direction of GREATEST spatial extent) — this means the
    three panels are always "top" (looking down the smallest-extent
    axis, i.e. the natural view for a flat coil), "side," and "side2"
    regardless of how the raw simulation coordinates happen to be
    oriented, since relax.py's normalize() only centers the curve, it
    does not orient it consistently between different (p,b,seed,...)
    runs.

    Each of the 3 views projects onto 2 of the 3 principal axes
    (`views = [(0,1),(0,2),(1,2)]`) and uses the THIRD (unprojected) axis
    as a depth value (`depth`) purely to decide DRAW ORDER (`order =
    argsort(depth)`, painter's algorithm: draw far points first so near
    points correctly overpaint them where the wire's projection
    self-overlaps) — this is a cheap depth cue, not a true 3D renderer
    (no actual occlusion testing, no lighting), but is enough to make
    self-crossings and contact regions visually legible at a glance.

    Each sample point is drawn as a small filled disk of radius
    proportional to the actual wire radius `r_wire` relative to the
    image's spatial span (`rad`, `disk` mask) rather than as a single
    pixel, so the rendered tube thickness is roughly physically
    accurate relative to the coil's overall size — e.g. a very thin
    wire (small r) on a large coil renders as a thin line, while a
    thick-wire, tightly-packed rope-like result renders with visibly
    touching/overlapping disks, giving an honest visual impression of
    how much of the wire's own radius is "used up" by the geometry.

    Color is a simple 2-color gradient by fractional arclength position
    `t = n/N` (blue-ish at the start of the parametrization, orange-ish
    at the end, smoothly interpolated) — purely to make it possible to
    visually trace a single strand through crossings and follow how the
    winding order relates to the final shape (e.g. to check that a
    "3-fold symmetric" looking shape really is traced by 3 colored
    passes in order, or to spot where two different-looking arcs are
    actually adjacent along the strand). It carries no other meaning.
    """
    P = P - P.mean(0)
    ev, evec = np.linalg.eigh(P.T @ P)      # ascending; evec cols = axes
    A = P @ evec[:, ::-1]                   # x = largest axis, z = smallest
    views = [(0, 1), (0, 2), (1, 2)]        # top, side, side2
    span = np.abs(A).max() * 1.12
    tiles = []
    N = len(A)
    for (i, j) in views:
        img = np.zeros((size, size, 3), np.uint8)
        img[:, :] = (16, 18, 34)
        px = ((A[:, i] / span) * (size / 2 - 4) + size / 2).astype(int)
        py = ((-A[:, j] / span) * (size / 2 - 4) + size / 2).astype(int)
        depth = A[:, 3 - i - j]
        order = np.argsort(depth)
        rad = max(1, int(r_wire / (2 * span) * size))
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        disk = (xx**2 + yy**2) <= rad * rad
        for n in order:
            t = n / N
            c = np.array([80 + 175 * t, 120 + 100 * (1 - t), 255 - 175 * t], np.uint8)
            x0, y0 = px[n], py[n]
            xs, ys = xx[disk] + x0, yy[disk] + y0
            ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
            img[ys[ok], xs[ok]] = c
        tiles.append(img)
    return np.concatenate(tiles, axis=1)

if __name__ == '__main__':
    # With no arguments, snapshot every result currently in elastic/
    # results/; otherwise treat each argument as a glob PATTERN (matched
    # by this script via `glob.glob`, not necessarily pre-expanded by the
    # shell) so callers can pass a quoted pattern like 'results/*coil*.json'
    # to select a subset without the shell needing to expand it first.
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, 'snaps')
    os.makedirs(outdir, exist_ok=True)
    pats = sys.argv[1:] or [os.path.join(here, 'results', '*.json')]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            res = json.load(open(f))
            img = render(np.array(res['points']), res['r'])
            out = os.path.join(outdir, os.path.basename(f).replace('.json', '.png'))
            write_png(out, img)
            print(out)
