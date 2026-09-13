# C=3 Conditions on (a, b, c, d)

For complexity C=3 with leads `m`, the four parameters `(a, b, c, d)` must satisfy
the following. Throughout, `L3 = 3·m`.

Implemented in `validC3` (live check) and re-derived in `computeValidC3` (list
generation) in `index.html`.

## Range / minimum
1. `a, b, c, d ≥ 2`
2. `m ≥ 3` (`minLeads()` for C=3)

## Divisibility (mod 3)
3. `a % 3 ≠ 0`, `c % 3 ≠ 0`, `d % 3 ≠ 0`
   - `d % 3 = 0` always bad: adjacent descending segments always overlap.
4. `(a + b) % 3 ≠ 0`
5. If `b % 3 = 0` **or** `(c + d) % 3 = 0`, then require `c + d ≤ b`
   - `b % 3 = 0` is allowed only when `c + d ≤ b` (ascending segments 2 & 4 then
     have disjoint y-ranges).

## Size bounds
6. `a + b + c ≤ 3m`
7. `a + c + d ≤ 3m`

## Canonical ordering (dedup of equivalent knots)
8. `a + b ≥ c + d`; if `a + b = c + d`, then `b ≥ d`
9. Not (`a + b + c = 3m` **and** `a + c + d = 3m` **and** `c < d`)
10. If `a + c + d = 3m`, then `a > d`   *(applied in `computeValidC3` only)*

## Geometric validity (`computeValidC3` only)
11. `hasAllCrossings`: every segment must carry at least one crossing.
12. Bight closure: an unexpected early closure (`comps[0].length !== mod`) no
    longer rejects the combination. With `generateBights` keyed on `(x, y, dir)`,
    a v-bight and ^-bight sharing a point is no longer mistaken for a loop
    closure. Any remaining early closure is surfaced in the on-screen red banner
    (`#c3ErrorDisplay`) as a possible condition bug, rather than silently dropped.

## Search-loop bounds (in `computeValidC3`)
- `a`: `2 … L3 − 4`, skipping `a % 3 = 0`
- `b`: `2 … L3 − a − 2`, skipping `b` where `(a + b) % 3 = 0`
- `c`: `2 … min(L3 − a − b, a + b − 2)`, skipping `c % 3 = 0`
- `d`: `2 … min(L3 − a − c, a + b − c)`, skipping `d % 3 = 0`

## Notes
- C=2 requires `m ≥ 4`; C=3 requires `m ≥ 3`.
- In `computeValidC3`, `B = m + 1` and `gcd(LEADS, B) = gcd(m, m+1) = 1`, so the
  C=3 valid list is always single-component (no links).
