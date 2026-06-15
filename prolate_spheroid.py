#!/usr/bin/env python3
"""
prolate_spheroid.py

Generate a pulsating prolate-spheroid point cloud for the patch-DFT Green
solver, matching the input format produced by the demo-sphere generator.

A prolate spheroid is the surface of revolution of an ellipse about its
major axis. With semi-minor axis b (the two short equatorial radii) and
semi-major axis c (the long polar radius), the aspect ratio is c : b.

Gary Koopmann's test specification
----------------------------------
    - aspect ratio 1 : 5 (minor : major), so c = 5 * b
    - pulsating like the sphere: all outward normal velocities = unity
    - for ka, the length scale 'a' is the radius of the prolate spheroid at
      its center slice, i.e. the equatorial semi-minor axis b
    - W = 1

This module emits the same columns the solver consumes:
    N, x, y, z, l, m, n, nx, ny, nz, vn_real, vn_imag, area

Coordinates are placed with the major axis along z. The center-slice radius
(equatorial radius) equals b. Pass --a-center to set b directly so that
ka = k * b lines up with the solver's --a argument.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def fibonacci_unit_sphere(n: int) -> np.ndarray:
    """Nearly equal-area points on the unit sphere (Fibonacci lattice)."""
    i = np.arange(n, dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    theta = 2.0 * np.pi * i / phi
    rxy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((rxy * np.cos(theta), rxy * np.sin(theta), z))


def prolate_points(n: int, b: float, c: float):
    """Map unit-sphere points onto a prolate spheroid with equatorial radius b
    and polar semi-axis c (major axis along z).

    Returns
    -------
    xyz      : (n,3) surface points
    normals  : (n,3) outward unit normals (exact, analytic)
    area     : (n,) per-point surface-area weights summing to total area
    """
    u = fibonacci_unit_sphere(n)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]

    # Surface point: equatorial axes scaled by b, polar axis scaled by c.
    x = b * ux
    y = b * uy
    z = c * uz
    xyz = np.column_stack((x, y, z))

    # Outward normal of ellipsoid x^2/b^2 + y^2/b^2 + z^2/c^2 = 1 is parallel
    # to the gradient (x/b^2, y/b^2, z/c^2).
    grad = np.column_stack((x / (b * b), y / (b * b), z / (c * c)))
    normals = grad / (np.linalg.norm(grad, axis=1)[:, None] + 1e-300)

    # Equal-area-ish weights: distribute the exact total surface area across
    # points in proportion to the local metric stretch of the Fibonacci map.
    # The unit-sphere Fibonacci lattice is equal-area on the sphere, so the
    # local area element scales by the surface Jacobian |r_theta x r_phi|.
    # For an ellipsoid parametrized through the unit-sphere direction u, the
    # local area scale is proportional to:
    #   J = sqrt( (b c)^2 (ux^2+uy^2) + (b^2)^2 uz^2 )  (b=a equatorial)
    # which is |grad-like| weighting; we then renormalize to the exact area.
    J = np.sqrt((b * c) ** 2 * (ux * ux + uy * uy) + (b * b) ** 2 * (uz * uz))
    w = J / J.sum()
    total_area = prolate_surface_area(b, c)
    area = w * total_area

    return xyz, normals, area


def prolate_surface_area(b: float, c: float) -> float:
    """Exact surface area of a prolate spheroid (c > b, major axis polar).

    A = 2*pi*b^2 * (1 + (c/b) * arcsin(e)/e),  e = sqrt(1 - b^2/c^2)
    Falls back to the sphere area when c == b.
    """
    if abs(c - b) < 1e-15:
        return 4.0 * np.pi * b * b
    if c > b:
        e = np.sqrt(1.0 - (b * b) / (c * c))
        return 2.0 * np.pi * b * b * (1.0 + (c / b) * np.arcsin(e) / e)
    # Oblate fallback (not used for 1:5 prolate, included for completeness).
    e = np.sqrt(1.0 - (c * c) / (b * b))
    return 2.0 * np.pi * b * b * (1.0 + ((1.0 - e * e) / e) * np.arctanh(e))


def write_prolate_csv(
    path: Path,
    n: int,
    a_center: float,
    ratio: float,
    w_multiplier: int = 1,
    verbose: bool = True,
) -> None:
    """Write a pulsating prolate-spheroid CSV.

    a_center : equatorial semi-minor axis b (the center-slice radius). This is
               the length scale you pass to the solver as --a.
    ratio    : major:minor aspect ratio (5 => 1:5 spheroid). c = ratio * b.
    """
    b = float(a_center)
    c = ratio * b
    xyz, normals, area = prolate_points(n, b, c)

    df = pd.DataFrame({
        "N": np.arange(1, n + 1),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "l": np.round(w_multiplier * 100 * normals[:, 0]).astype(int),
        "m": np.round(w_multiplier * 100 * normals[:, 1]).astype(int),
        "n": np.round(w_multiplier * 100 * normals[:, 2]).astype(int),
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "vn_real": np.ones(n), "vn_imag": np.zeros(n),
        "area": area,
    })
    df.to_csv(path, index=False)
    if verbose:
        print(
            f"Wrote pulsating prolate spheroid: {path}\n"
            f"  points N={n}, equatorial b(a_center)={b:g}, polar c={c:g}, "
            f"ratio 1:{ratio:g}\n"
            f"  total surface area={area.sum():.6g}, "
            f"W={w_multiplier}, all vn=1+0j (unit outward pulsation)"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a pulsating prolate-spheroid CSV for the patch-DFT solver")
    p.add_argument("out", help="Output CSV path")
    p.add_argument("--N", type=int, default=1440, help="Number of surface points")
    p.add_argument("--a-center", type=float, default=1.0,
                   help="Equatorial semi-minor axis b = center-slice radius (use as solver --a)")
    p.add_argument("--ratio", type=float, default=5.0, help="Major:minor aspect ratio (5 => 1:5)")
    p.add_argument("--W", type=int, default=1, help="Digital multiplier for l,m,n (Gary spec: W=1)")
    args = p.parse_args(argv)
    write_prolate_csv(Path(args.out), args.N, args.a_center, args.ratio, args.W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
