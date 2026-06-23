#!/usr/bin/env python3
"""
streamlit_app.py

Interactive front end for Gary Koopmann's patch-block DFT-compressed Green
acoustic radiation solver (patch_dft_green_solver.py).

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push this repo to GitHub and point Streamlit Community Cloud at
    streamlit_app.py as the entry point.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

import patch_dft_green_solver as solver
from prolate_spheroid import prolate_points, prolate_surface_area, fibonacci_unit_sphere


st.set_page_config(page_title="Koopmann Patch-DFT Acoustic Solver", layout="wide")

st.title("Patch-DFT Green Acoustic Radiation Solver")
st.caption(
    "Interactive front end for Gary Koopmann's patch-block DFT-compressed "
    "boundary solver. Configure the radiator and solver variables, then run."
)


# ---------------------------------------------------------------------------
# Geometry builders (return a solver.Geometry directly, no disk round-trip)
# ---------------------------------------------------------------------------

def build_sphere_geometry(n: int, radius: float) -> solver.Geometry:
    pts = solver.fibonacci_sphere_points(n, radius)
    normals = pts / (np.linalg.norm(pts, axis=1)[:, None] + 1e-300)
    area = np.full(n, 4.0 * np.pi * radius * radius / n)
    vn = np.ones(n, dtype=np.complex128)
    digital = np.round(100 * normals).astype(float)
    return solver.Geometry(
        index=np.arange(1, n + 1), xyz=pts, normals=normals,
        area=area, vn=vn, digital_lmn=digital,
    )


def build_prolate_geometry(n: int, a_center: float, ratio: float, w_mult: int) -> solver.Geometry:
    b = float(a_center)
    c = ratio * b
    xyz, normals, area = prolate_points(n, b, c)
    vn = np.ones(n, dtype=np.complex128)
    digital = np.round(w_mult * 100 * normals).astype(float)
    return solver.Geometry(
        index=np.arange(1, n + 1), xyz=xyz, normals=normals,
        area=area, vn=vn, digital_lmn=digital,
    )


def surface_triangulation(xyz: np.ndarray) -> np.ndarray | None:
    """Build watertight triangle connectivity for a star-shaped point cloud.

    All built-in radiators (sphere, prolate spheroid) are star-shaped about
    their centroid: every surface point is visible along a ray from the
    centroid. Taking the convex hull of the *unit direction* vectors from the
    centroid yields a consistent, closed triangulation of the original points,
    which renders as an opaque surface (correct hidden-line occlusion) even
    for elongated bodies where a plain convex hull of the points would skip
    the high-curvature poles.

    Returns an (n_tri, 3) int array of vertex indices into ``xyz``, or ``None``
    if a hull cannot be built (e.g. degenerate or non-star-shaped uploads).
    """
    if xyz is None or len(xyz) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        centroid = xyz.mean(axis=0)
        dirs = xyz - centroid
        norms = np.linalg.norm(dirs, axis=1)
        if np.any(norms <= 1e-12):
            return None
        dirs = dirs / norms[:, None]
        hull = ConvexHull(dirs)
        return hull.simplices.astype(int)
    except Exception:
        return None


def geometry_from_upload(file, a: float, max_points: int | None = None) -> solver.Geometry:
    df = pd.read_csv(file)
    if max_points is not None and len(df) > max_points:
        # Deterministic random subsample so big assemblies stay tractable.
        rng = np.random.default_rng(0)
        keep = rng.choice(len(df), size=int(max_points), replace=False)
        keep.sort()
        df = df.iloc[keep].reset_index(drop=True)
    columns = set(df.columns)
    idx_col = "N" if "N" in columns else ("index" if "index" in columns else None)
    index = df[idx_col].to_numpy() if idx_col else np.arange(1, len(df) + 1)
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise ValueError(f"Uploaded CSV must contain coordinate column '{col}'.")
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    vn = solver.complex_from_columns(df, "vn")
    if all(c in df.columns for c in ("nx", "ny", "nz")):
        normals = df[["nx", "ny", "nz"]].to_numpy(dtype=float)
        normals /= (np.linalg.norm(normals, axis=1)[:, None] + 1e-15)
    else:
        normals = solver.estimate_normals_pca(xyz)
    if "area" in df.columns:
        area = df["area"].to_numpy(dtype=float)
    else:
        area = np.full(len(df), 4.0 * np.pi * a * a / len(df))
    digital = df[["l", "m", "n"]].to_numpy(dtype=float) if all(c in df.columns for c in ("l", "m", "n")) else None
    return solver.Geometry(index=index, xyz=xyz, normals=normals, area=area, vn=vn, digital_lmn=digital)


# ---------------------------------------------------------------------------
# Sidebar: variable controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Geometry")
    geom_kind = st.radio(
        "Radiator",
        ["Prolate spheroid (Koopmann test)", "Pulsating sphere", "Upload CSV"],
    )

    uploaded = None
    if geom_kind == "Prolate spheroid (Koopmann test)":
        n_points = st.slider("Surface points N", 120, 4000, 1440, step=60)
        a_center = st.number_input("Center-slice radius b  (use as a)", value=1.0, min_value=1e-6, format="%.6g")
        ratio = st.number_input("Aspect ratio (major:minor)", value=5.0, min_value=1.0, format="%.4g")
        st.caption("Gary's spec: ratio 1:5, unit outward velocity, a = center-slice radius, W = 1.")
    elif geom_kind == "Pulsating sphere":
        n_points = st.slider("Surface points N", 120, 4000, 720, step=60)
        sphere_radius = st.number_input("Sphere radius a", value=1.0, min_value=1e-6, format="%.6g")
    else:
        uploaded = st.file_uploader("CSV with x,y,z,vn[,nx,ny,nz,area]", type=["csv"])
        upload_max_points = None
        if uploaded is not None:
            # Peek at the row count without committing to a solve. Large or
            # multi-component clouds make the adaptive refinement scale badly,
            # so warn up front and offer a downsample cap.
            try:
                _peek = pd.read_csv(uploaded)
                uploaded.seek(0)  # rewind so the real read still works
                _n_rows = len(_peek)
                _components = (
                    _peek["component"].nunique()
                    if "component" in _peek.columns else 1
                )
            except Exception:
                _n_rows, _components = None, 1

            if _n_rows is not None:
                st.caption(f"Loaded {_n_rows:,} points across {_components} component(s).")
                if _components > 1:
                    st.warning(
                        "This file has multiple components. The solver assumes a "
                        "single closed star-shaped body, so multi-part assemblies "
                        "may be slow or inaccurate."
                    )
                if _n_rows > 1500:
                    st.warning(
                        f"{_n_rows:,} points is large. Solve time grows super-linearly "
                        "on complex geometry (a few thousand scattered points can take "
                        "minutes). Consider downsampling below."
                    )
                if st.checkbox("Downsample before solving", value=(_n_rows > 1500)):
                    upload_max_points = st.slider(
                        "Max points to solve", 120,
                        int(min(max(_n_rows, 240), 4000)),
                        int(min(_n_rows, 1500)), step=60,
                        help="Randomly subsamples the uploaded points to this many before solving.",
                    )

    st.header("Acoustic / solver variables")
    ka = st.number_input("ka  (dimensionless wave number)", value=1.0, format="%.6g")
    a_scale = st.number_input("a  (length scale, same units as x,y,z)", value=1.0, min_value=1e-6, format="%.6g")
    W = st.number_input("W  (digital DFT rounding multiplier)", value=1, min_value=1, step=1)

    st.subheader("Patch / compression")
    auto_B = st.checkbox("Auto patch count B", value=True)
    B_user = None if auto_B else st.number_input("Patches B", value=64, min_value=1, step=1)
    auto_M = st.checkbox("Auto plane-wave modes M", value=True)
    M_user = None if auto_M else st.number_input("Modes per far block M", value=16, min_value=1, step=1)
    near_threshold = st.slider("Near-block threshold (rad about centroid)", 0.0, 3.14159, 0.75, step=0.05)
    self_d_model = st.selectbox("Self-cell D correction", ["cap", "zero"], index=0)

    st.subheader("Adaptive refinement")
    st.caption(
        "Patches are recursively split until small enough, sharp edges are "
        "protected, and each patch can grow its own basis size M until the "
        "far-block projection error is below tolerance."
    )
    patches_per_wavelength = st.number_input(
        "Patches per shortest wavelength", value=6.0, min_value=1.0, step=0.5, format="%.1f",
        help="Patch diameter target = shortest wavelength / this value. Higher = finer patches.",
    )
    max_diam_on = st.checkbox("Override max patch diameter", value=False)
    max_patch_diameter = (
        st.number_input("Max patch diameter (length units)", value=1.0, min_value=1e-6, format="%.6g")
        if max_diam_on else None
    )
    max_normal_angle_deg = st.slider(
        "Max normal spread per patch (deg, edge protection)", 5.0, 90.0, 35.0, step=5.0,
        help="Patches whose normals spread beyond this angle are split. Guards sharp edges/corners.",
    )
    min_points_per_patch = st.number_input(
        "Min points per patch", value=1, min_value=1, step=1,
    )
    vel_lambda_on = st.checkbox("Specify velocity wavelength", value=False,
        help="If the prescribed normal velocity varies on a length scale shorter than "
             "the acoustic wavelength, set it here so patches resolve it too.")
    lambda_velocity = (
        st.number_input("Velocity wavelength (length units)", value=1.0, min_value=1e-6, format="%.6g")
        if vel_lambda_on else None
    )

    disable_adaptive_M = not st.checkbox(
        "Adaptive M (grow basis until far-error tol)", value=True,
        help="When on, each patch grows its plane-wave basis M through the schedule "
             "until the far-block projection error is below tolerance.",
    )
    far_error_tol = st.select_slider(
        "Far-block projection error tolerance",
        options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        value=1e-4,
        format_func=lambda v: f"{v:.0e}",
    )
    m_schedule_text = st.text_input(
        "M schedule (comma-separated)", value="8, 12, 16, 24, 32, 48, 64",
        help="Ascending basis sizes the adaptive-M search steps through.",
    )

    st.subheader("GMRES")
    rtol = st.select_slider(
        "Relative tolerance",
        options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        value=1e-5,
        format_func=lambda v: f"{v:.0e}",
    )
    maxiter = st.number_input("Max iterations", value=200, min_value=1, step=10)

    run = st.button("Run solver", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Build geometry for preview / run
# ---------------------------------------------------------------------------

def make_geometry():
    if geom_kind == "Prolate spheroid (Koopmann test)":
        return build_prolate_geometry(int(n_points), float(a_center), float(ratio), int(W))
    if geom_kind == "Pulsating sphere":
        return build_sphere_geometry(int(n_points), float(sphere_radius))
    if uploaded is None:
        return None
    cap = upload_max_points if "upload_max_points" in globals() else None
    return geometry_from_upload(uploaded, float(a_scale), max_points=cap)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run:
    try:
        geom = make_geometry()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build geometry: {exc}")
        st.stop()

    if geom is None:
        st.warning("Upload a CSV first, or pick a built-in radiator.")
        st.stop()

    logs: list[str] = []
    orig_log = solver.log

    def capture_log(msg, verbose=True):
        if verbose:
            logs.append(str(msg))

    solver.log = capture_log  # capture solver's internal logging
    import time as _time
    n_pts = len(geom.xyz)
    if n_pts > 1500:
        st.info(
            f"Solving {n_pts:,} points. Large/complex geometry can take a while "
            "(a few thousand scattered points may run for minutes). Progress below."
        )
    status = st.status("Starting...", expanded=True)
    try:
        try:
            m_schedule = solver.parse_m_schedule(m_schedule_text)
        except Exception:
            m_schedule = (8, 12, 16, 24, 32, 48, 64)

        _t0 = _time.time()
        status.update(label="Running out-of-process memory-lean solver...", state="running")
        
        import tempfile
        import subprocess
        import sys
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            in_csv = os.path.join(tmpdir, "in.csv")
            out_prefix = os.path.join(tmpdir, "out")
            
            # Write geom to CSV
            df = pd.DataFrame({
                "N": geom.index,
                "x": geom.xyz[:, 0], "y": geom.xyz[:, 1], "z": geom.xyz[:, 2],
                "nx": geom.normals[:, 0], "ny": geom.normals[:, 1], "nz": geom.normals[:, 2],
                "area": geom.area,
                "vn_real": np.real(geom.vn), "vn_imag": np.imag(geom.vn),
            })
            if geom.digital_lmn is not None:
                df["l"] = geom.digital_lmn[:, 0]
                df["m"] = geom.digital_lmn[:, 1]
                df["n"] = geom.digital_lmn[:, 2]
            df.to_csv(in_csv, index=False)
            
            cmd = [
                sys.executable, "dam_dft_solver_streamlit_ready.py", in_csv,
                "--out", out_prefix,
                "--ka", str(float(ka)),
                "--a", str(float(a_scale)),
                "--W", str(int(W)),
                "--near-threshold", str(float(near_threshold)),
                "--self-d-model", self_d_model,
                "--patches-per-wavelength", str(float(patches_per_wavelength)),
                "--max-normal-angle-deg", str(float(max_normal_angle_deg)),
                "--min-points-per-patch", str(int(min_points_per_patch)),
                "--far-error-tol", str(float(far_error_tol)),
                "--M-schedule", ",".join(map(str, m_schedule)),
                "--rtol", str(float(rtol)),
                "--maxiter", str(int(maxiter)),
            ]
            if B_user: cmd.extend(["--B", str(int(B_user))])
            if M_user: cmd.extend(["--M", str(int(M_user))])
            if lambda_velocity: cmd.extend(["--lambda-v", str(float(lambda_velocity))])
            if max_patch_diameter: cmd.extend(["--max-patch-diameter", str(float(max_patch_diameter))])
            if disable_adaptive_M: cmd.append("--disable-adaptive-M")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            logs.extend(result.stdout.splitlines())
            if result.stderr:
                logs.extend(result.stderr.splitlines())
                
            if result.returncode != 0:
                st.error("Solver process failed. Check the logs.")
                st.stop()
                
            # Read outputs
            import json
            with open(out_prefix + "_report.json", "r") as f:
                report = json.load(f)
            out_df = pd.read_csv(out_prefix + "_pressure.csv")
            
            p = out_df["p_normalized_real"].to_numpy() + 1j * out_df["p_normalized_imag"].to_numpy()
            labels = out_df["patch"].to_numpy()
            
            # Map report to what session_state expects
            stats = {
                "gmres_info": report["solver"]["gmres_info"],
                "relative_residual": report["solver"]["relative_residual"]
            }
            metrics = {
                "surface_impedance": report["surface_impedance_normalized"]["real"] + 1j * report["surface_impedance_normalized"]["imag"],
                "radiated_power_rhoc1": report["radiated_power_rhoc1"]
            }
            
            # Fake the model object fields since we don't have it
            class DummyModel: pass
            model = DummyModel()
            model.B = report["B_patches"]
            model.M = report["M_plane_wave_terms_max_per_far_block"]
            model.W = report["W_multiplier"]
            model.ka = report["ka"]
            model.a = report["a"]
            model.k = report["k"]
            model.near_blocks = report["near_blocks_dense"]
            model.far_blocks = report["far_blocks_dft_compressed"]
            model.lambda_acoustic = report["lambda_acoustic"]
            model.lambda_velocity = report["lambda_velocity"]
            model.lambda_star = report["lambda_star"]
            model.max_patch_diameter_target = report["max_patch_diameter_target"]
            model.far_error_summary = report["far_block_error_control"]
            model.labels = labels

        status.update(
            label=f"Done in {_time.time() - _t0:.1f}s total.", state="complete", expanded=False,
        )

    finally:
        solver.log = orig_log

    # Stash everything needed to re-render so widget interactions (e.g. the
    # display-mode checkbox) don't re-trigger the solve or blank the output.
    # Streamlit reruns the whole script on every widget change; the "Run
    # solver" button is only True on its own click, so the results must live
    # in session_state to survive subsequent reruns.
    st.session_state["results"] = {
        "xyz": geom.xyz,
        "normals": geom.normals,
        "area": geom.area,
        "index": geom.index,
        "vn": geom.vn,
        "p": p,
        "labels": model.labels,
        "stats": stats,
        "metrics": metrics,
        "logs": logs,
        "model_info": {
            "B": int(model.B), "M": int(model.M), "W": int(model.W),
            "ka": float(model.ka), "a": float(model.a), "k": float(model.k),
            "near_blocks": int(model.near_blocks),
            "far_blocks": int(model.far_blocks),
            "lambda_acoustic": float(model.lambda_acoustic),
            "lambda_velocity": (float(model.lambda_velocity)
                                if model.lambda_velocity is not None else None),
            "lambda_star": float(model.lambda_star),
            "max_patch_diameter_target": (float(model.max_patch_diameter_target)
                                          if model.max_patch_diameter_target is not None else None),
            "far_error_summary": {k: (float(v) if isinstance(v, (int, float)) else v)
                                  for k, v in model.far_error_summary.items()},
        },
    }


if "results" in st.session_state:
    R = st.session_state["results"]
    xyz = R["xyz"]
    p = R["p"]
    stats = R["stats"]
    metrics = R["metrics"]
    logs = R["logs"]
    labels = R["labels"]
    mi = R["model_info"]
    Z = metrics["surface_impedance"]

    # ---- Top-line metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Re(Z)", f"{np.real(Z):.4e}")
    c2.metric("Im(Z)", f"{np.imag(Z):.4e}")
    c3.metric("Radiated power (rho*c=1)", f"{metrics['radiated_power_rhoc1']:.4e}")
    conv = "converged" if stats["gmres_info"] == 0 else f"info={int(stats['gmres_info'])}"
    c4.metric("GMRES", conv, delta=f"res {stats['relative_residual']:.1e}")

    fes = mi.get("far_error_summary", {})
    if fes:
        st.caption(
            f"Adaptive refinement: {mi['B']} patches, "
            f"M per patch {int(fes.get('M_min', 0))}\u2013{int(fes.get('M_max', 0))} "
            f"(mean {fes.get('M_mean', 0):.1f}), "
            f"far-block error max S={fes.get('far_error_max_S', float('nan')):.1e} / "
            f"D={fes.get('far_error_max_D', float('nan')):.1e} "
            f"vs tol {fes.get('far_error_tol', float('nan')):.0e}, "
            f"{int(fes.get('adaptive_passes', 0))} pass(es)."
        )

    colL, colR = st.columns([3, 2])

    # ---- 3D surface-pressure render ----
    with colL:
        st.subheader("Surface pressure |p|")
        view_mode = st.radio(
            "Display mode",
            [
                "Points (hide back-facing)",
                "Points (see-through)",
                "Solid surface",
            ],
            index=0,
            help="'Hide back-facing' shows only the points in your direct "
                 "line of sight: an invisible solid hull occludes points on "
                 "the far side, and it updates as you rotate. 'See-through' "
                 "shows every point. 'Solid surface' shows the opaque shaded "
                 "mesh colored by |p|.",
        )
        pdf = pd.DataFrame({
            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "p_abs": np.abs(p), "patch": labels,
        })
        try:
            import plotly.graph_objects as go
            p_abs = np.abs(p)
            BG = "white"  # occluder + scene background share this color
            tris = surface_triangulation(xyz)
            traces = []

            if view_mode == "Solid surface" and tris is not None and len(tris):
                traces.append(go.Mesh3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                    intensity=p_abs, colorscale="Viridis",
                    colorbar=dict(title="|p|"), showscale=True,
                    opacity=1.0, flatshading=False,
                    lighting=dict(ambient=0.55, diffuse=0.6, specular=0.2,
                                  roughness=0.9, fresnel=0.1),
                    lightposition=dict(x=100, y=200, z=300),
                    name="|p|",
                ))
            else:
                hide_back = (view_mode == "Points (hide back-facing)")
                if hide_back and tris is not None and len(tris):
                    # Invisible (background-colored) solid hull, shrunk very
                    # slightly toward the centroid so it sits just inside the
                    # points. The WebGL depth buffer then hides points behind
                    # it (the far side), live as the user rotates, while the
                    # front-facing points stay visible.
                    centroid = xyz.mean(axis=0)
                    occ = centroid + (xyz - centroid) * 0.995
                    traces.append(go.Mesh3d(
                        x=occ[:, 0], y=occ[:, 1], z=occ[:, 2],
                        i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                        color=BG, opacity=1.0, flatshading=True,
                        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                        hoverinfo="skip", showscale=False, name="occluder",
                    ))
                elif hide_back:
                    st.caption(
                        "Could not triangulate this geometry into a closed "
                        "surface; showing all points instead."
                    )
                traces.append(go.Scatter3d(
                    x=pdf["x"], y=pdf["y"], z=pdf["z"],
                    mode="markers",
                    marker=dict(size=3, color=pdf["p_abs"], colorscale="Viridis",
                                colorbar=dict(title="|p|"), showscale=True),
                    name="|p|",
                ))

            fig = go.Figure(data=traces)
            axis_bg = dict(backgroundcolor=BG, showbackground=True,
                           gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.15)")
            fig.update_layout(
                scene=dict(aspectmode="data", xaxis=axis_bg, yaxis=axis_bg, zaxis=axis_bg),
                paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0), height=560,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.scatter_chart(pdf, x="x", y="z", color="p_abs")

    # ---- Run report + downloads ----
    with colR:
        st.subheader("Run report")
        report = {
            "N_points": int(len(R["index"])),
            "B_patches": mi["B"],
            "M_modes_per_far_block": mi["M"],
            "W": mi["W"],
            "ka": mi["ka"], "a": mi["a"], "k": mi["k"],
            "near_blocks_dense": mi["near_blocks"],
            "far_blocks_dft_compressed": mi["far_blocks"],
            "lambda_acoustic": mi["lambda_acoustic"],
            "lambda_velocity": mi["lambda_velocity"],
            "lambda_star": mi["lambda_star"],
            "max_patch_diameter_target": mi["max_patch_diameter_target"],
            "adaptive": mi["far_error_summary"],
            "gmres": stats,
            "surface_impedance_real": float(np.real(Z)),
            "surface_impedance_imag": float(np.imag(Z)),
            "radiated_power_rhoc1": metrics["radiated_power_rhoc1"],
        }
        st.json(report, expanded=False)

        out_df = pd.DataFrame({
            "N": R["index"],
            "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
            "nx": R["normals"][:, 0], "ny": R["normals"][:, 1], "nz": R["normals"][:, 2],
            "area": R["area"],
            "vn_real": np.real(R["vn"]), "vn_imag": np.imag(R["vn"]),
            "p_real": np.real(p), "p_imag": np.imag(p),
            "p_abs": np.abs(p), "p_phase_rad": np.angle(p),
            "patch": labels,
        })
        st.download_button(
            "Download pressure CSV",
            out_df.to_csv(index=False).encode(),
            file_name="patch_dft_pressure.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download report JSON",
            json.dumps(report, indent=2).encode(),
            file_name="patch_dft_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Solver log"):
        st.code("\n".join(logs) or "(no log captured)")

    with st.expander("Pressure data table"):
        st.dataframe(out_df, use_container_width=True, height=300)

else:
    st.info(
        "Set the radiator and variables in the sidebar, then click **Run solver**. "
        "Defaults match Gary's prolate-spheroid test case (ratio 1:5, unit pulsation, W = 1)."
    )
