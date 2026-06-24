#!/usr/bin/env python3
"""Streamlit front end for the one-case-per-worker DAM DFT solver.

The Streamlit process never builds Green blocks or runs GMRES. It writes a job
specification and launches a separate worker process. Only status, logs, final
pressure CSV, patch summary, and JSON report are read back into the UI.

For a private server the default local worker process is adequate. For public
cloud deployment, point DFT_WORKER_SUBMIT_COMMAND at an external queue/container
launcher that accepts the job-spec path as its final argument and uses shared
storage for the job directory.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

HERE = Path(__file__).resolve().parent
SOLVER = HERE / "dam_dft_solver_streamlit_ready.py"
WRAPPER = HERE / "dft_worker_job.py"
JOB_ROOT = Path(os.environ.get("DFT_JOB_ROOT", str(HERE / "streamlit_jobs")))
JOB_ROOT.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "case"


def surface_triangulation(xyz):
    """Build a closed triangulation for a star-shaped point cloud.

    Takes the convex hull of the unit direction vectors from the centroid,
    which yields a consistent closed surface of the original points even for
    elongated bodies. Returns an (n_tri, 3) int array of indices into ``xyz``
    or ``None`` if a hull cannot be built (degenerate or non-star-shaped).
    """
    import numpy as np
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


def render_pressure_viewer(job_dir: Path) -> None:
    """Interactive 3D surface-pressure plot read from result_pressure.csv.

    The worker stays out of process; this only loads the final pressure CSV
    (no matrices) and renders it with Plotly inside the UI.
    """
    import numpy as np
    import pandas as pd

    pressure_csv = job_dir / "result_pressure.csv"
    if not pressure_csv.exists():
        return
    try:
        df = pd.read_csv(pressure_csv)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not read pressure CSV for plotting: {exc}")
        return
    if not all(c in df.columns for c in ("x", "y", "z")):
        return

    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)

    # Color by physical surface-pressure magnitude in Pascals. Prefer the
    # worker's peak-amplitude column; otherwise rescale the dimensionless
    # (rho*c) pressure by rho*c read from the run report.
    if "p_Pa_abs_peak" in df.columns:
        p_abs = df["p_Pa_abs_peak"].to_numpy(dtype=float)
    elif "p_Pa_real" in df.columns and "p_Pa_imag" in df.columns:
        p_abs = np.abs(
            df["p_Pa_real"].to_numpy() + 1j * df["p_Pa_imag"].to_numpy()
        )
    elif "p_normalized_real" in df.columns and "p_normalized_imag" in df.columns:
        report = load_json(job_dir / "result_report.json") or {}
        rho = float(report.get("rho_kg_m3", 1.204))
        c = float(report.get("c_m_s", 343.0))
        p_norm = np.abs(
            df["p_normalized_real"].to_numpy() + 1j * df["p_normalized_imag"].to_numpy()
        )
        p_abs = (rho * c) * p_norm
    else:
        return

    st.subheader("Surface pressure |p| (Pa)")
    color_options = ["|p| (Pa)"]
    if "patch" in df.columns:
        color_options.append("Patch id")
    for feat_col in ("feature_class", "feature_group_label"):
        if feat_col in df.columns:
            color_options.append(f"Feature: {feat_col}")

    ctrl_l, ctrl_r = st.columns(2)
    with ctrl_l:
        view_mode = st.radio(
            "Display mode",
            [
                "Points (hide back-facing)",
                "Points (see-through)",
                "Solid surface",
            ],
            index=0,
            key=f"viewmode_{job_dir.name}",
            help="'Hide back-facing' uses an invisible hull to occlude points "
                 "on the far side as you rotate. 'See-through' shows every "
                 "point. 'Solid surface' shows the opaque shaded mesh.",
        )
    with ctrl_r:
        color_by = st.selectbox(
            "Color by", color_options, index=0, key=f"colorby_{job_dir.name}",
        )

    if color_by == "|p| (Pa)":
        color_values = p_abs
        colorscale = "Viridis"
        colorbar_title = "|p| (Pa)"
        discrete = False
    elif color_by == "Patch id":
        color_values = df["patch"].to_numpy()
        colorscale = "Turbo"
        colorbar_title = "patch"
        discrete = False
    else:
        feat_col = color_by.split(": ", 1)[1]
        raw = df[feat_col].astype(str).fillna("none")
        categories = {name: i for i, name in enumerate(sorted(raw.unique()))}
        color_values = raw.map(categories).to_numpy()
        colorscale = "Turbo"
        colorbar_title = feat_col
        discrete = True

    try:
        import plotly.graph_objects as go
        BG = "white"
        tris = surface_triangulation(xyz)
        traces = []

        if view_mode == "Solid surface" and tris is not None and len(tris):
            traces.append(go.Mesh3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                intensity=p_abs, colorscale="Viridis",
                colorbar=dict(title="|p| (Pa)"), showscale=True,
                opacity=1.0, flatshading=False,
                lighting=dict(ambient=0.55, diffuse=0.6, specular=0.2,
                              roughness=0.9, fresnel=0.1),
                lightposition=dict(x=100, y=200, z=300),
                name="|p| (Pa)",
            ))
        else:
            hide_back = (view_mode == "Points (hide back-facing)")
            if hide_back and tris is not None and len(tris):
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
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=3, color=color_values, colorscale=colorscale,
                    colorbar=(None if discrete else dict(title=colorbar_title)),
                    showscale=not discrete,
                ),
                text=[f"|p|={v:.3g} Pa" for v in p_abs],
                name=colorbar_title,
            ))

        fig = go.Figure(data=traces)
        axis_bg = dict(backgroundcolor=BG, showbackground=True,
                       gridcolor="rgba(0,0,0,0.08)", zerolinecolor="rgba(0,0,0,0.15)")
        fig.update_layout(
            scene=dict(aspectmode="data", xaxis=axis_bg, yaxis=axis_bg, zaxis=axis_bg),
            paper_bgcolor=BG, margin=dict(l=0, r=0, t=0, b=0), height=560,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{job_dir.name}")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"3D viewer unavailable ({exc}); showing 2D scatter.")
        st.scatter_chart(
            pd.DataFrame({"x": xyz[:, 0], "z": xyz[:, 2], "p_abs": p_abs}),
            x="x", y="z", color="p_abs",
        )


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def tail_text(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


# Ordered solve pipeline stages mapped to a coarse completion fraction and a
# human-readable label. The worker appends one JSON record per stage to
# resources.jsonl; the UI reads the latest record to drive a live progress bar
# so the user does not have to refresh manually.
STAGE_PROGRESS = [
    ("worker_start", 0.02, "Starting worker"),
    ("geometry_loaded", 0.10, "Geometry loaded"),
    ("patching_complete", 0.20, "Patching complete"),
    ("adaptive_rank_complete", 0.28, "Adaptive rank selected"),
    ("memory_estimate_complete", 0.32, "Memory estimated"),
    ("block_build_progress", 0.45, "Building Green blocks"),
    ("model_complete", 0.55, "Model assembled"),
    ("preconditioner_complete", 0.60, "Preconditioner ready"),
    ("solve_progress", 0.75, "Solving (Krylov)"),
    ("gmres_complete", 0.80, "GMRES finished"),
    ("lgmres_complete", 0.90, "LGMRES fallback finished"),
    ("metrics_complete", 0.95, "Computing metrics"),
    ("outputs_saved", 0.99, "Saving outputs"),
    ("worker_cleanup_complete", 1.0, "Complete"),
]
_STAGE_RANK = {name: i for i, (name, _f, _l) in enumerate(STAGE_PROGRESS)}
_STAGE_INFO = {name: (frac, label) for name, frac, label in STAGE_PROGRESS}


def read_progress(job_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse resources.jsonl and return a coarse progress snapshot.

    Returns a dict with fraction (0..1), label, and the latest raw record, or
    None when no resource log exists yet. Cheap: only the tail is parsed.
    """
    log_path = job_dir / "resources.jsonl"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    latest: Optional[Dict[str, Any]] = None
    best_rank = -1
    block_fraction: Optional[float] = None
    solve_record: Optional[Dict[str, Any]] = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        stage = record.get("stage")
        if stage not in _STAGE_RANK:
            continue
        if stage == "block_build_progress":
            block_fraction = record.get("fraction", block_fraction)
        if stage == "solve_progress":
            solve_record = record
        rank = _STAGE_RANK[stage]
        if rank >= best_rank:
            best_rank = rank
            latest = record
    if latest is None:
        return None
    stage = latest.get("stage")
    frac, label = _STAGE_INFO.get(stage, (0.0, stage or "working"))

    # Refine within the long phases using sub-progress data.
    if stage == "block_build_progress" and block_fraction is not None:
        base, nxt = 0.32, 0.55
        frac = base + (nxt - base) * float(block_fraction)
        label = f"Building Green blocks ({block_fraction * 100:.0f}%)"
    elif stage == "solve_progress" and solve_record is not None:
        method = solve_record.get("method", "krylov")
        it = solve_record.get("iteration")
        res = solve_record.get("current_residual")
        target = solve_record.get("target_rtol")
        # Iterative-solver progress has no clean linear endpoint, so estimate it
        # from how far the residual has dropped toward the target on a log
        # scale (residual starts near 1.0 and must reach target_rtol). This is
        # only a visual hint; the bar may pause if convergence stalls.
        base, nxt = 0.60, 0.80
        if res is not None and target and res > 0:
            import math
            start_exp = 0.0           # log10(1.0)
            target_exp = math.log10(float(target))
            cur_exp = math.log10(float(res))
            span = (start_exp - target_exp) or 1.0
            done = (start_exp - cur_exp) / span
            frac = base + (nxt - base) * max(0.0, min(1.0, done))
        else:
            frac = base
        label = f"Solving ({method})"
        if it is not None:
            label += f"  iteration {it}"
        if res is not None:
            label += f"  residual {res:.2e}"
            if target:
                label += f" (target {float(target):.0e})"

    return {
        "fraction": max(0.0, min(1.0, float(frac))),
        "label": label,
        "stage": stage,
        "elapsed_s": latest.get("elapsed_s"),
        "rss_mb": latest.get("rss_mb"),
        "record": latest,
    }


def launch_job(spec_path: Path) -> Dict[str, Any]:
    backend = os.environ.get("DFT_WORKER_BACKEND", "local").strip().lower()
    if backend in {"queue", "shared_queue", "shared-folder"}:
        atomic_write_json(
            spec_path.parent / "status.json",
            {
                "state": "queued",
                "queued_unix": time.time(),
                "backend": "shared_folder_queue",
            },
        )
        return {
            "backend": "shared_folder_queue",
            "job_spec": str(spec_path),
        }

    external = os.environ.get("DFT_WORKER_SUBMIT_COMMAND", "").strip()
    if external:
        command = shlex.split(external) + [str(spec_path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "External worker submission failed: "
                + (completed.stderr or completed.stdout)
            )
        return {
            "backend": "external",
            "submission_command": command,
            "submission_stdout": completed.stdout,
        }

    command = [sys.executable, str(WRAPPER), str(spec_path)]
    process = subprocess.Popen(
        command,
        cwd=str(HERE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "backend": "local_process",
        "launcher_pid": process.pid,
        "process_group_id": process.pid,
        "submission_command": command,
    }


def cancel_local_job(job_dir: Path) -> None:
    launch = load_json(job_dir / "launch.json") or {}
    group_id = launch.get("process_group_id")
    if group_id:
        try:
            os.killpg(int(group_id), signal.SIGTERM)
        except ProcessLookupError:
            pass
    status = load_json(job_dir / "status.json") or {}
    status.update({"state": "cancelled", "cancelled_unix": time.time()})
    atomic_write_json(job_dir / "status.json", status)


def create_job(
    uploaded_file: Any,
    feature_metadata_file: Any,
    case_id: str,
    frequency_hz: float,
    a: float,
    rho: float,
    c: float,
    W: int,
    B: int,
    M: int,
    patches_per_wavelength: float,
    near_threshold: float,
    far_error_tol: float,
    max_rank_fraction: float,
    max_adapt_far_blocks: int,
    max_points_per_patch: int,
    rtol: float,
    maxiter: int,
    gmres_restart: int,
    memory_budget_mb: float,
    self_d_model: str,
    feature_boundary_mode: str,
    face_column: str,
    edge_column: str,
    corner_column: str,
    feature_type_column: str,
    feature_type_map: str,
    feature_id_column: str,
    is_edge_column: str,
    is_corner_column: str,
    feature_connectivity_factor: float,
    feature_zero_id_is_missing: bool,
    feature_geometry_key_column: str,
    feature_metadata_key_column: str,
) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    job_id = f"{stamp}_{safe_name(case_id)}_{uuid.uuid4().hex[:8]}"
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / safe_name(uploaded_file.name)
    input_path.write_bytes(uploaded_file.getvalue())
    feature_metadata_path: Optional[Path] = None
    if feature_metadata_file is not None:
        feature_metadata_path = job_dir / safe_name(feature_metadata_file.name)
        feature_metadata_path.write_bytes(feature_metadata_file.getvalue())
    out_prefix = job_dir / "result"

    command = [
        sys.executable,
        str(SOLVER),
        str(input_path),
        "--case-id",
        case_id,
        "--frequency-hz",
        str(frequency_hz),
        "--a",
        str(a),
        "--rho",
        str(rho),
        "--c",
        str(c),
        "--W",
        str(W),
        "--patches-per-wavelength",
        str(patches_per_wavelength),
        "--near-threshold",
        str(near_threshold),
        "--far-error-tol",
        str(far_error_tol),
        "--max-rank-fraction",
        str(max_rank_fraction),
        "--rank-guard-action",
        "warn",
        "--max-adapt-far-blocks",
        str(max_adapt_far_blocks),
        "--max-points-per-patch",
        str(max_points_per_patch),
        "--preconditioner",
        "block-jacobi",
        "--rtol",
        str(rtol),
        "--maxiter",
        str(maxiter),
        "--gmres-restart",
        str(gmres_restart),
        "--fallback-krylov",
        "lgmres",
        "--direct-fallback-max-n",
        "0",
        "--memory-budget-mb",
        str(memory_budget_mb),
        "--self-d-model",
        self_d_model,
        "--feature-boundary-mode",
        feature_boundary_mode,
        "--feature-connectivity-factor",
        str(feature_connectivity_factor),
        "--resource-log",
        str(job_dir / "resources.jsonl"),
        "--out",
        str(out_prefix),
    ]
    if not feature_zero_id_is_missing:
        command.append("--feature-zero-id-is-valid")

    if feature_metadata_path is not None:
        command.extend(["--feature-metadata-csv", str(feature_metadata_path)])
        if str(feature_geometry_key_column).strip():
            command.extend([
                "--feature-geometry-key-column",
                str(feature_geometry_key_column).strip(),
            ])
        if str(feature_metadata_key_column).strip():
            command.extend([
                "--feature-metadata-key-column",
                str(feature_metadata_key_column).strip(),
            ])

    if str(feature_type_map).strip():
        command.extend(["--feature-type-map", str(feature_type_map).strip()])

    optional_feature_columns = {
        "--face-column": face_column,
        "--edge-column": edge_column,
        "--corner-column": corner_column,
        "--feature-type-column": feature_type_column,
        "--feature-id-column": feature_id_column,
        "--is-edge-column": is_edge_column,
        "--is-corner-column": is_corner_column,
    }
    for flag, value in optional_feature_columns.items():
        if str(value).strip():
            command.extend([flag, str(value).strip()])
    if B > 0:
        command.extend(["--B", str(B)])
    if M > 0:
        command.extend(["--M", str(M)])

    job_spec = {
        "job_id": job_id,
        "case_id": case_id,
        "created_unix": time.time(),
        "working_directory": str(HERE),
        "command": command,
        "input_csv": str(input_path),
        "feature_metadata_csv": (
            str(feature_metadata_path) if feature_metadata_path is not None else None
        ),
        "out_prefix": str(out_prefix),
    }
    spec_path = job_dir / "job_spec.json"
    atomic_write_json(spec_path, job_spec)
    launch = launch_job(spec_path)
    atomic_write_json(job_dir / "launch.json", launch)
    return job_dir


def render_job(job_dir: Path) -> None:
    st.subheader(f"Job: {job_dir.name}")
    status = load_json(job_dir / "status.json")
    launch = load_json(job_dir / "launch.json") or {}
    if status is None:
        st.info("Submitted; waiting for the worker to create a status file.")
        state = "submitted"
    else:
        state = str(status.get("state", "unknown"))
        if state == "queued":
            st.info("Job is queued for the separate worker container.")
        elif state == "running":
            st.info("Worker is running one acoustic case in a separate process/container.")
        elif state == "completed":
            st.success("Worker completed and converged.")
        elif state == "completed_unconverged":
            st.warning("Worker finished, but the requested residual was not reached.")
        elif state == "cancelled":
            st.warning("Job was cancelled.")
        else:
            st.error(f"Worker state: {state}")
        st.json(status, expanded=False)

    is_active = state in {"submitted", "queued", "running"}

    # Live progress driven by the worker resource log, so the user sees the
    # solve advancing without manually refreshing.
    progress = read_progress(job_dir)
    if progress is not None:
        if is_active:
            st.progress(progress["fraction"], text=progress["label"])
            elapsed = progress.get("elapsed_s")
            rss = progress.get("rss_mb")
            bits = []
            if elapsed is not None:
                bits.append(f"elapsed {float(elapsed):.0f}s")
            if rss is not None:
                bits.append(f"RSS {float(rss):.0f} MiB")
            if bits:
                st.caption("Worker active: " + ", ".join(bits))
        elif progress["stage"] == "worker_cleanup_complete":
            st.progress(1.0, text="Complete")
    elif is_active:
        st.progress(0.0, text="Waiting for the worker to report progress...")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh status", use_container_width=True):
            st.rerun()
    with col2:
        if (
            is_active
            and launch.get("backend") == "local_process"
            and st.button("Cancel local worker", use_container_width=True)
        ):
            cancel_local_job(job_dir)
            st.rerun()

    # Auto-refresh loop while the job is still working. This polls the status
    # and resource log on a fixed cadence so the progress bar advances on its
    # own; it stops as soon as the job leaves an active state.
    if is_active:
        st.caption("Auto-refreshing every 2s while the worker runs.")
        time.sleep(2.0)
        st.rerun()

    log_text = tail_text(job_dir / "worker.log")
    if log_text:
        with st.expander("Worker log", expanded=state in {"failed", "running"}):
            st.code(log_text, language="text")

    report_path = job_dir / "result_report.json"
    report = load_json(report_path)
    if report:
        solver = report.get("solver", {})
        metric_cols = st.columns(4)
        metric_cols[0].metric("Radiated power", f"{report.get('radiated_power_W', float('nan')):.6g} W")
        metric_cols[1].metric("Max peak pressure", f"{report.get('max_pressure_peak_Pa', float('nan')):.6g} Pa")
        metric_cols[2].metric("Relative residual", f"{solver.get('relative_residual', float('nan')):.3e}")
        metric_cols[3].metric("Method", str(solver.get("method_used", "")))
        feature_info = (
            report.get("geometry_audit", {}).get("feature_metadata", {})
        )
        if feature_info.get("metadata_present"):
            st.caption(
                "Feature-aware patching: "
                f"mode={feature_info.get('boundary_mode')}, "
                f"hard groups={feature_info.get('hard_group_count')}, "
                "patch-boundary violations="
                f"{feature_info.get('patches_crossing_hard_feature_boundaries')}"
            )
        with st.expander("Run report", expanded=False):
            st.json(report, expanded=False)

    render_pressure_viewer(job_dir)

    downloadable = [
        ("Pressure CSV", job_dir / "result_pressure.csv", "text/csv"),
        ("Patch summary CSV", job_dir / "result_patch_summary.csv", "text/csv"),
        ("Feature summary CSV", job_dir / "result_feature_summary.csv", "text/csv"),
        ("Run report JSON", report_path, "application/json"),
        ("Resource log JSONL", job_dir / "resources.jsonl", "application/json"),
        ("Worker log", job_dir / "worker.log", "text/plain"),
    ]
    existing = [entry for entry in downloadable if entry[1].exists()]
    if existing:
        st.caption("Only final vectors, summaries, and logs are returned to Streamlit; no matrices are loaded into the UI.")
        for label, path, mime in existing:
            st.download_button(
                label=label,
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
                key=f"download_{job_dir.name}_{path.name}",
            )


st.set_page_config(page_title="DAM DFT Acoustic Solver", layout="wide")
st.title("DAM feature-aware patch-DFT acoustic solver")
st.caption("One mode per isolated worker process; preprocessor-aware patching; matrix-free preconditioned Krylov solution; no explicit matrix inversion.")

with st.form("new_job", clear_on_submit=False):
    uploaded = st.file_uploader("One mode CSV", type=["csv"])
    left, right = st.columns(2)
    with left:
        case_id = st.text_input("Case ID", value="pitch")
        frequency_hz = st.number_input("Frequency (Hz)", min_value=0.001, value=276.17, format="%.6f")
        a = st.number_input("Reference length a (m)", min_value=1e-9, value=0.225, format="%.9f")
        rho = st.number_input("Air density rho (kg/m³)", min_value=0.001, value=1.204, format="%.6f")
        c = st.number_input("Sound speed c (m/s)", min_value=1.0, value=343.0, format="%.6f")
        memory_budget_mb = st.number_input("Worker memory budget (MiB)", min_value=128.0, value=1400.0, step=64.0)
    with right:
        B = st.number_input("Initial patches B (0 = automatic)", min_value=0, value=16, step=1)
        M = st.number_input("Initial local rank M (0 = automatic)", min_value=0, value=0, step=1)
        W = st.number_input("Digital direction multiplier W", min_value=1, value=100, step=1)
        patches_per_wavelength = st.number_input("Patches per wavelength", min_value=1.0, value=3.0, step=0.5)
        near_threshold = st.number_input("Near threshold", min_value=0.0, value=0.75, step=0.05)
        self_d_model = st.selectbox("D self-cell model", ["cap", "zero"], index=0)

    with st.expander("Preprocessor feature metadata", expanded=True):
        st.caption(
            "The worker auto-detects face, edge, corner, feature-type, and feature-ID columns. "
            "Metadata may be embedded in the acoustic CSV or uploaded as a separate table."
        )
        feature_metadata_upload = st.file_uploader(
            "Optional separate feature metadata CSV",
            type=["csv"],
            key="feature_metadata_upload",
            help="The table is joined one-to-one to the acoustic CSV by node_id/N/index or explicit key overrides.",
        )
        f1, f2, f3 = st.columns(3)
        feature_boundary_mode = f1.selectbox(
            "Feature boundary policy",
            ["auto", "strict", "face-only", "off"],
            index=0,
            help=(
                "auto: strict edge/corner/feature groups when available, otherwise face-only; "
                "strict: always use all detected feature metadata as hard patch boundaries."
            ),
        )
        feature_connectivity_factor = f2.number_input(
            "Unnamed feature connectivity factor",
            min_value=0.5,
            value=2.5,
            step=0.25,
            help=(
                "If edge/corner points are flagged but not assigned IDs, nearby points within this "
                "multiple of median spacing are grouped as one feature component."
            ),
        )
        feature_zero_id_is_missing = f2.checkbox(
            "Treat zero edge/corner IDs as missing",
            value=True,
            help="Disable this only when your preprocessor uses feature ID 0 as a real entity.",
        )
        f3.markdown(
            "Recognized aliases include `face_id`, `surface_id`, `edge_id`, `corner_id`, "
            "`feature_type`, `feature_id`, `is_edge`, and `is_corner`."
        )
        c1, c2, c3, c4 = st.columns(4)
        face_column = c1.text_input("Face/region column override", value="")
        edge_column = c2.text_input("Edge column override", value="")
        corner_column = c3.text_input("Corner column override", value="")
        feature_type_column = c4.text_input("Feature-type column override", value="")
        c5, c6, c7, c8 = st.columns(4)
        feature_id_column = c5.text_input("Feature-ID column override", value="")
        is_edge_column = c6.text_input("Edge-flag column override", value="")
        is_corner_column = c7.text_input("Corner-flag column override", value="")
        c8.caption("Blank override fields use automatic alias detection.")
        feature_type_map = st.text_input(
            "Optional feature-type code map",
            value="",
            placeholder="0:surface,1:edge,2:corner",
            help="Use this when feature_type contains numeric or proprietary codes.",
        )
        k1, k2 = st.columns(2)
        feature_geometry_key_column = k1.text_input(
            "Acoustic CSV join-key override",
            value="",
            help="Used only when a separate metadata CSV is uploaded.",
        )
        feature_metadata_key_column = k2.text_input(
            "Metadata CSV join-key override",
            value="",
            help="Used only when a separate metadata CSV is uploaded.",
        )

    with st.expander("Convergence and compression controls", expanded=False):
        c1, c2, c3 = st.columns(3)
        far_error_tol = c1.number_input("Far-block error tolerance", min_value=1e-8, value=1e-4, format="%.1e")
        max_rank_fraction = c2.number_input("Maximum rank / patch points", min_value=0.05, max_value=0.95, value=0.70, step=0.05)
        max_adapt_far_blocks = c3.number_input("Far blocks checked in adaptation (0 = all)", min_value=0, value=256, step=16)
        max_points_per_patch = c1.number_input("Maximum points per patch", min_value=16, value=96, step=16)
        rtol = c2.number_input("Krylov relative tolerance", min_value=1e-10, value=1e-5, format="%.1e")
        maxiter = c2.number_input("Maximum Krylov iterations/cycles", min_value=1, value=100, step=10)
        gmres_restart = c3.number_input("GMRES restart", min_value=5, value=50, step=5)

    submitted = st.form_submit_button("Submit one mode to worker", type="primary")

if submitted:
    if uploaded is None:
        st.error("Upload one mode CSV before submitting.")
    else:
        job_dir = create_job(
            uploaded_file=uploaded,
            feature_metadata_file=feature_metadata_upload,
            case_id=case_id,
            frequency_hz=float(frequency_hz),
            a=float(a),
            rho=float(rho),
            c=float(c),
            W=int(W),
            B=int(B),
            M=int(M),
            patches_per_wavelength=float(patches_per_wavelength),
            near_threshold=float(near_threshold),
            far_error_tol=float(far_error_tol),
            max_rank_fraction=float(max_rank_fraction),
            max_adapt_far_blocks=int(max_adapt_far_blocks),
            max_points_per_patch=int(max_points_per_patch),
            rtol=float(rtol),
            maxiter=int(maxiter),
            gmres_restart=int(gmres_restart),
            memory_budget_mb=float(memory_budget_mb),
            self_d_model=str(self_d_model),
            feature_boundary_mode=str(feature_boundary_mode),
            face_column=str(face_column),
            edge_column=str(edge_column),
            corner_column=str(corner_column),
            feature_type_column=str(feature_type_column),
            feature_type_map=str(feature_type_map),
            feature_id_column=str(feature_id_column),
            is_edge_column=str(is_edge_column),
            is_corner_column=str(is_corner_column),
            feature_connectivity_factor=float(feature_connectivity_factor),
            feature_zero_id_is_missing=bool(feature_zero_id_is_missing),
            feature_geometry_key_column=str(feature_geometry_key_column),
            feature_metadata_key_column=str(feature_metadata_key_column),
        )
        st.session_state["active_job_dir"] = str(job_dir)
        st.success(f"Submitted {job_dir.name}")

existing_jobs = sorted(
    [path for path in JOB_ROOT.iterdir() if path.is_dir()],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if existing_jobs:
    default_job = st.session_state.get("active_job_dir")
    default_index = 0
    labels = [path.name for path in existing_jobs]
    if default_job:
        try:
            default_index = labels.index(Path(default_job).name)
        except ValueError:
            default_index = 0
    selected_name = st.selectbox("View job", labels, index=default_index)
    selected_dir = JOB_ROOT / selected_name
    st.session_state["active_job_dir"] = str(selected_dir)
    render_job(selected_dir)
else:
    st.info("No jobs have been submitted from this deployment yet.")
