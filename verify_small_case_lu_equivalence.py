#!/usr/bin/env python3
"""Small verification: matrix-free GMRES versus dense LAPACK LU.

This is not a production route. It confirms on a small case that the
preconditioned matrix-free operator and a dense assembly of the same compressed
DFT operator solve the same algebraic equation without forming A^{-1}.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import patch_dft_green_solver_adaptive as legacy
from dam_dft_solver_streamlit_ready import (
    ResourceMonitor,
    assemble_compressed_dense_operator,
    build_model_lite,
    compute_outputs,
    relative_residual,
    solve_pressure,
    make_system_operator,
)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        csv_path = root / "sphere.csv"
        legacy.write_demo_sphere_csv(csv_path, n=96, radius=1.0, verbose=False)
        geom = legacy.load_geometry(csv_path, a=1.0, verbose=False)
        monitor = ResourceMonitor(path=None, memory_budget_mb=0, verbose=False)
        model = build_model_lite(
            geom=geom,
            ka=0.8,
            a=1.0,
            W=100,
            B_user=8,
            M_user=6,
            near_threshold=0.75,
            self_d_model="cap",
            lambda_velocity=None,
            patches_per_wavelength=3.0,
            max_patch_diameter=None,
            max_normal_angle_deg=35.0,
            min_points_per_patch=4,
            max_points_per_patch=96,
            far_error_tol=5e-2,
            m_schedule=[4, 6, 8, 10],
            disable_adaptive_m=False,
            max_rank_fraction=0.70,
            max_rank_absolute=None,
            rank_guard_action="warn",
            max_adapt_passes=4,
            max_adapt_far_blocks=0,
            max_patch_count=512,
            monitor=monitor,
            verbose=False,
        )
        p_iter, stats = solve_pressure(
            model=model,
            rtol=1e-7,
            maxiter=50,
            restart=20,
            preconditioner_name="block-jacobi",
            preconditioner_regularization=1e-10,
            fallback_krylov="lgmres",
            direct_fallback_max_n=0,
            monitor=monitor,
            verbose=False,
        )
        rhs = 1j * model.k * model.single_layer_on_vn
        dense = assemble_compressed_dense_operator(model)
        lu, piv = lu_factor(dense, overwrite_a=True, check_finite=False)
        p_lu = lu_solve((lu, piv), rhs, check_finite=False)
        operator = make_system_operator(model)
        comparison = {
            "N": len(rhs),
            "iterative_method": stats["method_used"],
            "iterative_residual": relative_residual(operator, p_iter, rhs),
            "lu_residual": relative_residual(operator, p_lu, rhs),
            "relative_pressure_difference": float(
                np.linalg.norm(p_iter - p_lu) / (np.linalg.norm(p_lu) + 1e-30)
            ),
            "iterative_power_W": compute_outputs(model, p_iter, 1.204, 343.0)[
                "radiated_power_W"
            ],
            "lu_power_W": compute_outputs(model, p_lu, 1.204, 343.0)[
                "radiated_power_W"
            ],
            "explicit_inverse_used": False,
        }
        print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
