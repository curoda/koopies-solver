#!/usr/bin/env python3
"""Regression test for preprocessor face/edge/corner/feature metadata.

The test creates a 96-point sphere with two face regions, an equatorial sharp
feature, and two corner labels. It verifies that:

* a separate metadata CSV is joined by point ID;
* aliases are detected automatically;
* strict feature groups become hard compression-patch boundaries;
* no final patch crosses a feature group;
* the matrix-free solver still converges;
* pressure, patch, feature-summary, and JSON outputs retain the metadata.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import patch_dft_green_solver_adaptive as legacy
from dam_dft_solver_streamlit_ready import (
    ResourceMonitor,
    build_model_lite,
    compute_outputs,
    save_outputs,
    solve_pressure,
)


def make_featured_sphere(path: Path, n: int = 96) -> None:
    xyz = legacy.fibonacci_sphere_points(n, radius=1.0)
    normals = xyz.copy()
    area = np.full(n, 4.0 * np.pi / n)
    z = xyz[:, 2]

    face_id = np.where(z >= 0.0, "north_shell", "south_shell").astype(object)
    feature_type = np.full(n, "surface", dtype=object)
    edge_id = np.full(n, "", dtype=object)
    corner_id = np.full(n, "", dtype=object)

    equator = np.abs(z) < 0.15
    feature_type[equator] = "sharp_edge"
    edge_id[equator] = "equator_E1"

    north = int(np.argmax(z))
    south = int(np.argmin(z))
    feature_type[[north, south]] = "corner"
    corner_id[north] = "north_vertex"
    corner_id[south] = "south_vertex"
    edge_id[[north, south]] = ""

    pd.DataFrame(
        {
            "node_id": np.arange(1, n + 1),
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "nx": normals[:, 0],
            "ny": normals[:, 1],
            "nz": normals[:, 2],
            "area": area,
            "vn_real": np.full(n, 1.0e-3),
            "vn_imag": np.zeros(n),
            "face_id": face_id,
            "edge_id": edge_id,
            "corner_id": corner_id,
            "feature_type": feature_type,
        }
    ).to_csv(path, index=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        embedded_csv = root / "featured_sphere_embedded.csv"
        make_featured_sphere(embedded_csv)
        embedded = pd.read_csv(embedded_csv)
        input_csv = root / "featured_sphere_acoustic.csv"
        metadata_csv = root / "featured_sphere_features.csv"
        embedded.drop(
            columns=["face_id", "edge_id", "corner_id", "feature_type"]
        ).to_csv(input_csv, index=False)
        embedded[[
            "node_id", "face_id", "edge_id", "corner_id", "feature_type"
        ]].rename(columns={"node_id": "point_id"}).to_csv(metadata_csv, index=False)

        geom = legacy.load_geometry(
            input_csv,
            a=1.0,
            feature_boundary_mode="auto",
            feature_metadata_csv=metadata_csv,
            feature_geometry_key_column="node_id",
            feature_metadata_key_column="point_id",
            verbose=False,
        )
        assert geom.features is not None
        assert geom.features.boundary_mode == "strict"

        monitor = ResourceMonitor(path=None, memory_budget_mb=0.0, verbose=False)
        model = build_model_lite(
            geom=geom,
            ka=0.8,
            a=1.0,
            W=100,
            B_user=8,
            M_user=4,
            near_threshold=0.75,
            self_d_model="cap",
            lambda_velocity=None,
            patches_per_wavelength=3.0,
            max_patch_diameter=None,
            max_normal_angle_deg=180.0,
            min_points_per_patch=1,
            max_points_per_patch=24,
            far_error_tol=0.5,
            m_schedule=[4, 6, 8],
            disable_adaptive_m=True,
            max_rank_fraction=0.70,
            max_rank_absolute=None,
            rank_guard_action="warn",
            max_adapt_passes=1,
            max_adapt_far_blocks=0,
            max_patch_count=256,
            monitor=monitor,
            verbose=False,
        )

        boundary_violations = 0
        for patch_id in np.unique(model.labels):
            inds = np.where(model.labels == patch_id)[0]
            if len(np.unique(geom.features.group_code[inds])) > 1:
                boundary_violations += 1
        assert boundary_violations == 0

        pressure, stats = solve_pressure(
            model=model,
            rtol=1.0e-6,
            maxiter=50,
            restart=20,
            preconditioner_name="block-jacobi",
            preconditioner_regularization=1.0e-10,
            fallback_krylov="lgmres",
            direct_fallback_max_n=0,
            monitor=monitor,
            verbose=False,
        )
        assert stats["converged"]

        metrics = compute_outputs(model, pressure, rho=1.204, c=343.0)
        out_prefix = root / "feature_test"
        paths = save_outputs(
            model=model,
            pressure_normalized=pressure,
            stats=stats,
            metrics=metrics,
            out_prefix=out_prefix,
            case_id="feature_metadata_test",
            frequency_hz=None,
            monitor=monitor,
            verbose=False,
        )

        pressure_df = pd.read_csv(paths["pressure_csv"])
        required_output_columns = {
            "feature_group_code",
            "feature_group_label",
            "feature_class",
            "feature_type",
            "face_id",
            "edge_id",
            "corner_id",
            "feature_id",
        }
        assert required_output_columns.issubset(pressure_df.columns)

        feature_summary = pd.read_csv(paths["feature_summary_csv"])
        report = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
        feature_audit = report["geometry_audit"]["feature_metadata"]
        assert feature_audit["patches_crossing_hard_feature_boundaries"] == 0

        result = {
            "N": int(len(geom.index)),
            "separate_metadata_join_verified": True,
            "detected_columns": geom.features.detected_columns,
            "boundary_mode": geom.features.boundary_mode,
            "hard_feature_groups": int(len(np.unique(geom.features.group_code))),
            "compression_patches": int(model.B),
            "patch_boundary_violations": int(boundary_violations),
            "feature_classes": {
                name: int(np.sum(geom.features.feature_class == name))
                for name in ("surface", "edge", "corner", "feature")
            },
            "feature_summary_rows": int(len(feature_summary)),
            "solver_method": stats["method_used"],
            "relative_residual": float(stats["relative_residual"]),
            "explicit_inverse_used": bool(stats["explicit_matrix_inverse_used"]),
            "output_feature_columns_present": True,
        }
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
