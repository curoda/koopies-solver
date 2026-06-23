#!/usr/bin/env python3
"""
DAM patch-DFT Green solver -- Streamlit/large-case execution upgrade.

This module keeps the original boundary equation and patch-block DFT basis:

    (1/2 I - D_DFT) p = i k S_DFT v_n

It changes the execution architecture rather than the acoustic formulation:

* one input case per process;
* matrix-free Krylov solution with a local block-Jacobi preconditioner;
* LAPACK LU factorization only on small self-patch blocks (no explicit inverse);
* the single-layer operator is applied once to v_n and then discarded;
* only D blocks needed by the iterative solve remain in memory;
* adaptive-rank guard prevents silent full-rank "compression";
* memory estimates and live RSS snapshots are written to a JSONL log;
* optional LGMRES fallback remains matrix-free;
* optional dense LAPACK LU fallback is disabled by default and limited to small N;
* model objects are never serialized and are explicitly released after each case;
* preprocessor face/edge/corner/feature metadata are read and enforced as hard patch boundaries.

The validated geometry, patching, Green kernels, self terms, and local DFT
construction are imported from patch_dft_green_solver_adaptive.py, which must
be in the same directory.
"""

from __future__ import annotations

# Avoid BLAS thread oversubscription on small Streamlit hosts. A deployment can
# override any of these variables before launching the worker.
import os
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import gc
import json
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres, lgmres

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import patch_dft_green_solver_adaptive as legacy

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    psutil = None


# ---------------------------------------------------------------------------
# Logging, resource monitoring, and exceptions
# ---------------------------------------------------------------------------


def log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


class MemoryBudgetExceeded(RuntimeError):
    """Raised before allocation when the estimated/current memory is unsafe."""


class CompressionGuardError(RuntimeError):
    """Raised when requested DFT accuracy cannot be achieved below rank cap."""


class ResourceMonitor:
    """Record elapsed time and process memory without retaining model objects."""

    def __init__(
        self,
        path: Optional[Path] = None,
        memory_budget_mb: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.path = path
        self.memory_budget_mb = float(memory_budget_mb)
        self.verbose = verbose
        self.started = time.perf_counter()
        self.records: List[Dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def current_rss_mb(self) -> float:
        if psutil is not None:
            return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0**2))
        try:
            import resource

            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # Linux reports KiB; macOS reports bytes.
            if sys.platform == "darwin":
                return value / (1024.0**2)
            return value / 1024.0
        except Exception:
            return float("nan")

    def snapshot(self, stage: str, **details: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "stage": stage,
            "elapsed_s": time.perf_counter() - self.started,
            "rss_mb": self.current_rss_mb(),
            "pid": os.getpid(),
        }
        record.update(details)
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, default=_json_default) + "\n")
        log(
            f"[resource] {stage}: RSS={record['rss_mb']:.1f} MiB, "
            f"elapsed={record['elapsed_s']:.1f} s",
            self.verbose,
        )
        if (
            self.memory_budget_mb > 0
            and np.isfinite(record["rss_mb"])
            and record["rss_mb"] > self.memory_budget_mb
        ):
            raise MemoryBudgetExceeded(
                f"Current RSS {record['rss_mb']:.1f} MiB exceeds configured "
                f"budget {self.memory_budget_mb:.1f} MiB at stage '{stage}'."
            )
        return record


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


# ---------------------------------------------------------------------------
# DFT patch basis with a rank/compression guard
# ---------------------------------------------------------------------------

PatchBasis = legacy.PatchBasis
Geometry = legacy.Geometry


def rank_cap_for_patch(
    n_points: int,
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
) -> int:
    """Maximum local rank, intentionally below full point rank when possible."""
    if n_points <= 1:
        return 1
    fraction = min(max(float(max_rank_fraction), 0.05), 0.99)
    cap = int(math.floor(fraction * n_points))
    cap = max(1, min(cap, n_points - 1))
    if max_rank_absolute is not None and max_rank_absolute > 0:
        cap = min(cap, int(max_rank_absolute))
    return max(1, cap)


def build_one_patch_basis_guarded(
    geom: Geometry,
    inds: np.ndarray,
    requested_m: int,
    k: float,
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
) -> PatchBasis:
    inds = np.asarray(inds, dtype=int)
    center = geom.xyz[inds].mean(axis=0)
    normal = geom.normals[inds].mean(axis=0)
    normal /= np.linalg.norm(normal) + 1e-15
    cap = rank_cap_for_patch(len(inds), max_rank_fraction, max_rank_absolute)
    m_eff = int(max(1, min(requested_m, cap)))
    base_dirs = legacy.fibonacci_directions(m_eff)
    directions = legacy.rotated_directions(base_dirs, normal)
    local = geom.xyz[inds] - center
    plane_waves = np.exp(-1j * k * (local @ directions.T))
    basis, _ = np.linalg.qr(plane_waves, mode="reduced")
    return PatchBasis(
        indices=inds,
        center=center,
        normal=normal,
        basis=basis,
        directions=directions,
        M_requested=m_eff,
        diameter=legacy.max_pairwise_diameter(geom.xyz, inds),
        normal_spread_deg=math.degrees(legacy.normal_spread_angle(geom.normals, inds)),
        region=(
            int(geom.patch_region[inds[0]])
            if geom.patch_region is not None and len(inds)
            else -1
        ),
    )


def next_guarded_schedule_value(
    current_m: int,
    schedule: Sequence[int],
    n_points: int,
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
) -> int:
    cap = rank_cap_for_patch(n_points, max_rank_fraction, max_rank_absolute)
    for value in schedule:
        candidate = min(int(value), cap)
        if candidate > current_m:
            return candidate
    return current_m


def initial_guarded_m(
    n_points: int,
    k: float,
    diameter: float,
    schedule: Sequence[int],
    user_m: Optional[int],
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
) -> int:
    cap = rank_cap_for_patch(n_points, max_rank_fraction, max_rank_absolute)
    if user_m is not None:
        return max(1, min(int(user_m), cap))
    kd = abs(k) * diameter
    estimate = max(8, int(math.ceil(4.0 * (1.0 + kd) ** 2)))
    for value in schedule:
        if value >= estimate:
            return max(1, min(int(value), cap))
    return max(1, min(int(max(schedule)), cap))


def build_guarded_patch_bases(
    geom: Geometry,
    labels: np.ndarray,
    ranks: Sequence[int],
    k: float,
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
) -> List[PatchBasis]:
    patches: List[PatchBasis] = []
    for patch_id, requested_m in enumerate(ranks):
        inds = np.where(labels == patch_id)[0]
        if len(inds) == 0:
            continue
        patches.append(
            build_one_patch_basis_guarded(
                geom,
                inds,
                int(requested_m),
                k,
                max_rank_fraction,
                max_rank_absolute,
            )
        )
    return patches


def projection_error_from_core(
    matrix: np.ndarray,
    core: np.ndarray,
) -> float:
    """Frobenius projection error without reconstructing the full block.

    With orthonormal receiver/source bases, the two-sided DFT approximation is
    an orthogonal projection in the Frobenius inner product. Therefore
    ||A-P(A)||_F^2 = ||A||_F^2 - ||Q_a^H A Q_b||_F^2.
    """
    norm2 = float(np.vdot(matrix, matrix).real)
    if norm2 <= 1e-60:
        return 0.0
    core2 = float(np.vdot(core, core).real)
    return float(math.sqrt(max(norm2 - core2, 0.0) / norm2))


def projection_error(
    matrix: np.ndarray,
    receiver_basis: np.ndarray,
    source_basis: np.ndarray,
) -> float:
    core = receiver_basis.conj().T @ matrix @ source_basis
    return projection_error_from_core(matrix, core)


def enumerate_far_pairs(
    patches: Sequence[PatchBasis],
    centroid: np.ndarray,
    near_threshold: float,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for a_idx, pa in enumerate(patches):
        for b_idx, pb in enumerate(patches):
            separation = legacy.patch_center_angle_or_distance(pa, pb, centroid)
            if a_idx != b_idx and separation >= near_threshold:
                pairs.append((a_idx, b_idx))
    return pairs


def choose_adaptation_pairs(
    all_pairs: Sequence[Tuple[int, int]],
    max_pairs: int,
) -> List[Tuple[int, int]]:
    if max_pairs <= 0 or len(all_pairs) <= max_pairs:
        return list(all_pairs)
    positions = np.linspace(0, len(all_pairs) - 1, num=max_pairs, dtype=int)
    return [all_pairs[int(i)] for i in np.unique(positions)]


def adapt_patch_bases_guarded(
    geom: Geometry,
    patches: List[PatchBasis],
    k: float,
    near_threshold: float,
    self_d_model: str,
    m_schedule: Sequence[int],
    far_error_tol: float,
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
    rank_guard_action: str,
    max_adapt_passes: int,
    max_adapt_far_blocks: int,
    verbose: bool,
) -> Tuple[List[PatchBasis], Dict[str, Any]]:
    """Adapt local ranks without permitting silent full-rank degeneration."""
    centroid = geom.xyz.mean(axis=0)
    all_far_pairs = enumerate_far_pairs(patches, centroid, near_threshold)
    sampled_pairs = choose_adaptation_pairs(all_far_pairs, max_adapt_far_blocks)
    history: List[Dict[str, Any]] = []
    saturated_patch_ids: set[int] = set()
    unresolved_pairs = 0

    for pass_id in range(max_adapt_passes):
        desired = [patch.M_requested for patch in patches]
        worst_s = 0.0
        worst_d = 0.0
        sum_s = 0.0
        sum_d = 0.0
        unresolved_pairs = 0
        saturated_patch_ids.clear()

        for a_idx, b_idx in sampled_pairs:
            pa = patches[a_idx]
            pb = patches[b_idx]
            s_exact, d_exact = legacy.green_blocks_exact(
                geom.xyz,
                geom.normals,
                geom.area,
                pa.indices,
                pb.indices,
                k,
                self_d_model=self_d_model,
            )
            error_s = projection_error(s_exact, pa.basis, pb.basis)
            error_d = projection_error(d_exact, pa.basis, pb.basis)
            del s_exact, d_exact
            error = max(error_s, error_d)
            worst_s = max(worst_s, error_s)
            worst_d = max(worst_d, error_d)
            sum_s += error_s
            sum_d += error_d

            if error > far_error_tol:
                unresolved_pairs += 1
                new_a = next_guarded_schedule_value(
                    desired[a_idx],
                    m_schedule,
                    len(pa.indices),
                    max_rank_fraction,
                    max_rank_absolute,
                )
                new_b = next_guarded_schedule_value(
                    desired[b_idx],
                    m_schedule,
                    len(pb.indices),
                    max_rank_fraction,
                    max_rank_absolute,
                )
                if new_a == desired[a_idx]:
                    saturated_patch_ids.add(a_idx)
                else:
                    desired[a_idx] = new_a
                if new_b == desired[b_idx]:
                    saturated_patch_ids.add(b_idx)
                else:
                    desired[b_idx] = new_b

        changed_ids = [
            idx for idx, patch in enumerate(patches)
            if desired[idx] > patch.M_requested
        ]
        for idx in changed_ids:
            patches[idx] = build_one_patch_basis_guarded(
                geom,
                patches[idx].indices,
                desired[idx],
                k,
                max_rank_fraction,
                max_rank_absolute,
            )

        count = max(len(sampled_pairs), 1)
        record = {
            "pass": pass_id + 1,
            "far_pairs_total": len(all_far_pairs),
            "far_pairs_checked": len(sampled_pairs),
            "max_error_S": worst_s,
            "max_error_D": worst_d,
            "mean_error_S": sum_s / count,
            "mean_error_D": sum_d / count,
            "unresolved_pairs": unresolved_pairs,
            "changed_patches": len(changed_ids),
            "saturated_patches": len(saturated_patch_ids),
            "M_min": min(p.M_requested for p in patches),
            "M_mean": float(np.mean([p.M_requested for p in patches])),
            "M_max": max(p.M_requested for p in patches),
        }
        history.append(record)
        log(
            "Adaptive M pass "
            f"{record['pass']}: checked {record['far_pairs_checked']}/"
            f"{record['far_pairs_total']} far blocks; max errors "
            f"S={worst_s:.3e}, D={worst_d:.3e}; unresolved="
            f"{unresolved_pairs}; rank range={record['M_min']}-"
            f"{record['M_max']}",
            verbose,
        )

        if unresolved_pairs == 0 or not changed_ids:
            break

    final_worst_s = history[-1]["max_error_S"] if history else 0.0
    final_worst_d = history[-1]["max_error_D"] if history else 0.0
    rank_fractions = [
        p.M_requested / max(len(p.indices), 1) for p in patches
    ]
    summary: Dict[str, Any] = {
        "far_error_tol": float(far_error_tol),
        "far_error_max_S": float(final_worst_s),
        "far_error_max_D": float(final_worst_d),
        "unresolved_pairs": int(unresolved_pairs),
        "rank_guard_hit_count": int(len(saturated_patch_ids)),
        "rank_guard_patch_ids": sorted(int(i) for i in saturated_patch_ids),
        "rank_fraction_max": float(max(rank_fractions) if rank_fractions else 0.0),
        "M_min": int(min(p.M_requested for p in patches) if patches else 0),
        "M_mean": float(np.mean([p.M_requested for p in patches]) if patches else 0.0),
        "M_max": int(max(p.M_requested for p in patches) if patches else 0),
        "adaptive_passes": len(history),
        "far_pairs_total": len(all_far_pairs),
        "far_pairs_checked": len(sampled_pairs),
        "sampled_error_control": len(sampled_pairs) < len(all_far_pairs),
        "history": history,
    }

    if unresolved_pairs > 0:
        message = (
            f"Adaptive DFT error target {far_error_tol:.3e} was not reached for "
            f"{unresolved_pairs} checked far blocks before one or more patches "
            f"hit the rank guard. Increase patch count, reduce patch diameter, "
            f"or explicitly relax the tolerance; do not silently use full rank."
        )
        if rank_guard_action == "error":
            raise CompressionGuardError(message)
        log("WARNING: " + message, verbose)

    return patches, summary


# ---------------------------------------------------------------------------
# Memory-lean D-only model and block construction
# ---------------------------------------------------------------------------


@dataclass
class DBlock:
    receiver_patch: int
    source_patch: int
    kind: str  # "near" dense block or "far" DFT core
    matrix: np.ndarray


@dataclass
class SolverModelLite:
    geom: Geometry
    ka: float
    a: float
    k: float
    W: int
    B: int
    M: int
    labels: np.ndarray
    patches: List[PatchBasis]
    d_blocks: List[DBlock]
    single_layer_on_vn: np.ndarray
    near_blocks: int
    far_blocks: int
    pqr: np.ndarray
    lambda_acoustic: float
    lambda_velocity: Optional[float]
    lambda_star: float
    max_patch_diameter_target: Optional[float]
    patch_summary: List[Dict[str, Any]]
    far_error_summary: Dict[str, Any]
    memory_estimate: Dict[str, float]


def classify_pair(
    a_idx: int,
    b_idx: int,
    patches: Sequence[PatchBasis],
    centroid: np.ndarray,
    near_threshold: float,
) -> bool:
    if a_idx == b_idx:
        return True
    separation = legacy.patch_center_angle_or_distance(
        patches[a_idx], patches[b_idx], centroid
    )
    return separation < near_threshold


def estimate_model_memory(
    patches: Sequence[PatchBasis],
    near_threshold: float,
    n_points: int,
) -> Dict[str, float]:
    """Conservative estimate of solver-model and peak temporary memory."""
    centroid = np.mean([p.center for p in patches], axis=0)
    complex_bytes = np.dtype(np.complex128).itemsize
    float_bytes = np.dtype(np.float64).itemsize
    d_entries = 0
    near_entries = 0
    far_entries = 0
    max_exact_entries = 0
    block_count = 0
    for a_idx, pa in enumerate(patches):
        for b_idx, pb in enumerate(patches):
            block_count += 1
            exact_entries = len(pa.indices) * len(pb.indices)
            max_exact_entries = max(max_exact_entries, exact_entries)
            if classify_pair(a_idx, b_idx, patches, centroid, near_threshold):
                near_entries += exact_entries
                d_entries += exact_entries
            else:
                core_entries = pa.basis.shape[1] * pb.basis.shape[1]
                far_entries += core_entries
                d_entries += core_entries

    basis_entries = sum(p.basis.size for p in patches)
    self_entries = sum(len(p.indices) ** 2 for p in patches)
    stored_d_mb = d_entries * complex_bytes / (1024.0**2)
    basis_mb = basis_entries * complex_bytes / (1024.0**2)
    preconditioner_mb = self_entries * complex_bytes / (1024.0**2)
    vectors_mb = n_points * (8 * complex_bytes + 12 * float_bytes) / (1024.0**2)
    python_overhead_mb = block_count * 0.0015
    # green_blocks_exact simultaneously creates vector differences, ranges,
    # masks, phases, S, D, and work arrays. Use a conservative 128 B/pair.
    temporary_mb = max_exact_entries * 128.0 / (1024.0**2)
    model_mb = (
        stored_d_mb
        + basis_mb
        + preconditioner_mb
        + vectors_mb
        + python_overhead_mb
    )
    peak_incremental_mb = model_mb + temporary_mb
    return {
        "stored_D_blocks_mb": stored_d_mb,
        "near_D_entries": float(near_entries),
        "far_D_core_entries": float(far_entries),
        "basis_mb": basis_mb,
        "block_jacobi_LU_mb": preconditioner_mb,
        "vectors_geometry_mb": vectors_mb,
        "python_block_overhead_mb": python_overhead_mb,
        "largest_exact_block_entries": float(max_exact_entries),
        "largest_exact_block_temporary_mb": temporary_mb,
        "estimated_model_incremental_mb": model_mb,
        "estimated_peak_incremental_mb": peak_incremental_mb,
    }


def enforce_estimated_memory_budget(
    estimate: Dict[str, float],
    monitor: ResourceMonitor,
    safety_fraction: float = 0.85,
) -> None:
    if monitor.memory_budget_mb <= 0:
        return
    current = monitor.current_rss_mb()
    predicted = current + estimate["estimated_peak_incremental_mb"]
    safe_limit = monitor.memory_budget_mb * safety_fraction
    if predicted > safe_limit:
        raise MemoryBudgetExceeded(
            "Predicted peak memory is unsafe for the configured host: "
            f"current RSS {current:.1f} MiB + estimated incremental peak "
            f"{estimate['estimated_peak_incremental_mb']:.1f} MiB = "
            f"{predicted:.1f} MiB, above safety limit {safe_limit:.1f} MiB. "
            "Increase patch compression, reduce N, use more worker memory, or "
            "process the geometry with a hierarchical/FMM backend."
        )


def enforce_max_points_per_patch(
    geom: Geometry,
    labels: np.ndarray,
    max_points_per_patch: int,
) -> np.ndarray:
    """Split oversized existing patches without crossing their current boundaries."""
    if max_points_per_patch <= 0:
        return labels
    patch_lists = [np.where(labels == value)[0] for value in np.unique(labels)]
    while any(len(inds) > max_points_per_patch for inds in patch_lists):
        refined: List[np.ndarray] = []
        for inds in patch_lists:
            if len(inds) <= max_points_per_patch:
                refined.append(inds)
                continue
            left, right = legacy.split_indices_by_farthest_pair(geom.xyz, inds)
            if len(left) == 0 or len(right) == 0:
                refined.append(inds)
            else:
                refined.extend([left, right])
        if len(refined) == len(patch_lists) and all(
            len(inds) <= max_points_per_patch for inds in refined
        ):
            patch_lists = refined
            break
        patch_lists = refined
    return legacy.relabel_from_patch_lists(patch_lists, geom.xyz.shape[0])


def build_model_lite(
    geom: Geometry,
    ka: float,
    a: float,
    W: int,
    B_user: Optional[int],
    M_user: Optional[int],
    near_threshold: float,
    self_d_model: str,
    lambda_velocity: Optional[float],
    patches_per_wavelength: float,
    max_patch_diameter: Optional[float],
    max_normal_angle_deg: float,
    min_points_per_patch: int,
    max_points_per_patch: int,
    far_error_tol: float,
    m_schedule: Sequence[int],
    disable_adaptive_m: bool,
    max_rank_fraction: float,
    max_rank_absolute: Optional[int],
    rank_guard_action: str,
    max_adapt_passes: int,
    max_adapt_far_blocks: int,
    max_patch_count: int,
    monitor: ResourceMonitor,
    verbose: bool,
) -> SolverModelLite:
    n_points = geom.xyz.shape[0]
    k = ka / a
    lambda_acoustic = 2.0 * np.pi / abs(k) if abs(k) > 0 else np.inf
    lambda_star = (
        min(lambda_acoustic, lambda_velocity)
        if lambda_velocity is not None
        else lambda_acoustic
    )
    if max_patch_diameter is None and np.isfinite(lambda_star):
        max_patch_diameter = lambda_star / patches_per_wavelength

    b0 = B_user if B_user is not None else legacy.auto_patch_count(n_points)
    log("\n--- Memory-lean patch-DFT setup ---", verbose)
    log(f"Input points N={n_points}", verbose)
    log(f"ka={ka:g}, a={a:g}, k={k:g}", verbose)
    log(
        f"lambda_acoustic={lambda_acoustic:.6g}; "
        f"lambda_velocity={lambda_velocity}; lambda_star={lambda_star:.6g}",
        verbose,
    )
    log(f"Initial patch count B0={b0}", verbose)

    labels, _initial_summary = legacy.refine_patches_by_diameter_and_edges(
        geom=geom,
        initial_B=b0,
        max_patch_diameter=max_patch_diameter,
        min_points_per_patch=min_points_per_patch,
        max_normal_angle_rad=math.radians(max_normal_angle_deg),
        verbose=verbose,
    )
    labels = enforce_max_points_per_patch(
        geom, labels, max_points_per_patch
    )
    unique = np.unique(labels)
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([remap[int(value)] for value in labels], dtype=int)
    patch_count = len(unique)
    if patch_count > max_patch_count:
        raise MemoryBudgetExceeded(
            f"Patch refinement produced B={patch_count}, above the configured "
            f"maximum {max_patch_count}. Review preprocessor feature-group cardinality "
            "(especially per-element IDs), increase the normal-angle limit, or "
            "deliberately raise --max-patch-count after reviewing the memory estimate."
        )
    patch_sizes = np.bincount(labels, minlength=patch_count)
    monitor.snapshot(
        "patching_complete",
        B=patch_count,
        patch_size_min=int(patch_sizes.min()),
        patch_size_mean=float(patch_sizes.mean()),
        patch_size_max=int(patch_sizes.max()),
        max_points_per_patch=int(max_points_per_patch),
    )

    schedule = sorted(set(int(value) for value in m_schedule if int(value) > 0))
    if not schedule:
        raise ValueError("M schedule cannot be empty.")
    initial_ranks: List[int] = []
    for patch_id in range(patch_count):
        inds = np.where(labels == patch_id)[0]
        diameter = legacy.max_pairwise_diameter(geom.xyz, inds)
        initial_ranks.append(
            initial_guarded_m(
                len(inds),
                k,
                diameter,
                schedule,
                M_user,
                max_rank_fraction,
                max_rank_absolute,
            )
        )

    patches = build_guarded_patch_bases(
        geom,
        labels,
        initial_ranks,
        k,
        max_rank_fraction,
        max_rank_absolute,
    )
    if disable_adaptive_m:
        far_error_summary: Dict[str, Any] = {
            "adaptive_passes": 0,
            "far_error_tol": far_error_tol,
            "far_error_max_S": None,
            "far_error_max_D": None,
            "unresolved_pairs": None,
            "rank_guard_hit_count": 0,
            "M_min": min(initial_ranks),
            "M_mean": float(np.mean(initial_ranks)),
            "M_max": max(initial_ranks),
        }
        log("Adaptive M disabled by user.", verbose)
    else:
        patches, far_error_summary = adapt_patch_bases_guarded(
            geom=geom,
            patches=patches,
            k=k,
            near_threshold=near_threshold,
            self_d_model=self_d_model,
            m_schedule=schedule,
            far_error_tol=far_error_tol,
            max_rank_fraction=max_rank_fraction,
            max_rank_absolute=max_rank_absolute,
            rank_guard_action=rank_guard_action,
            max_adapt_passes=max_adapt_passes,
            max_adapt_far_blocks=max_adapt_far_blocks,
            verbose=verbose,
        )
    monitor.snapshot(
        "adaptive_rank_complete",
        M_min=min(p.M_requested for p in patches),
        M_max=max(p.M_requested for p in patches),
    )

    memory_estimate = estimate_model_memory(
        patches, near_threshold=near_threshold, n_points=n_points
    )
    log(
        "Estimated incremental model memory "
        f"{memory_estimate['estimated_model_incremental_mb']:.1f} MiB; "
        "estimated incremental peak "
        f"{memory_estimate['estimated_peak_incremental_mb']:.1f} MiB.",
        verbose,
    )
    enforce_estimated_memory_budget(memory_estimate, monitor)
    monitor.snapshot("memory_estimate_complete", **memory_estimate)

    centroid = geom.xyz.mean(axis=0)
    d_blocks: List[DBlock] = []
    single_layer_on_vn = np.zeros(n_points, dtype=np.complex128)
    near_count = 0
    far_count = 0
    final_far_errors_s: List[float] = []
    final_far_errors_d: List[float] = []
    final_far_blocks_above_tolerance = 0
    total_pairs = patch_count * patch_count
    built_pairs = 0
    next_progress = 0.1
    log(
        "Building D blocks and applying S to v_n once; S blocks are not retained...",
        verbose,
    )

    for a_idx, pa in enumerate(patches):
        receiver = pa.indices
        for b_idx, pb in enumerate(patches):
            source = pb.indices
            is_near = classify_pair(
                a_idx, b_idx, patches, centroid, near_threshold
            )
            s_exact, d_exact = legacy.green_blocks_exact(
                geom.xyz,
                geom.normals,
                geom.area,
                receiver,
                source,
                k,
                self_d_model=self_d_model,
            )
            if is_near:
                single_layer_on_vn[receiver] += s_exact @ geom.vn[source]
                d_blocks.append(DBlock(a_idx, b_idx, "near", d_exact))
                near_count += 1
            else:
                ga = pa.basis
                gb = pb.basis
                # Build temporary spectral cores. S is applied once and discarded;
                # only the D core is retained for Krylov matvecs.
                s_core = ga.conj().T @ s_exact @ gb
                d_core = ga.conj().T @ d_exact @ gb
                source_coeff = gb.conj().T @ geom.vn[source]
                single_layer_on_vn[receiver] += ga @ (s_core @ source_coeff)
                error_s = projection_error_from_core(s_exact, s_core)
                error_d = projection_error_from_core(d_exact, d_core)
                final_far_errors_s.append(error_s)
                final_far_errors_d.append(error_d)
                if max(error_s, error_d) > far_error_tol:
                    final_far_blocks_above_tolerance += 1
                d_blocks.append(DBlock(a_idx, b_idx, "far", d_core))
                far_count += 1
                del s_core, d_core, source_coeff
            del s_exact, d_exact

            built_pairs += 1
            fraction = built_pairs / total_pairs
            if fraction >= next_progress or built_pairs == total_pairs:
                monitor.snapshot(
                    "block_build_progress",
                    completed_pairs=built_pairs,
                    total_pairs=total_pairs,
                    fraction=fraction,
                )
                next_progress += 0.1

    far_error_summary.update(
        {
            "far_error_max_S_all_blocks": float(max(final_far_errors_s, default=0.0)),
            "far_error_max_D_all_blocks": float(max(final_far_errors_d, default=0.0)),
            "far_error_mean_S_all_blocks": float(np.mean(final_far_errors_s) if final_far_errors_s else 0.0),
            "far_error_mean_D_all_blocks": float(np.mean(final_far_errors_d) if final_far_errors_d else 0.0),
            "far_blocks_above_tolerance_all_blocks": int(final_far_blocks_above_tolerance),
            "final_validation_checked_all_far_blocks": True,
        }
    )
    if final_far_blocks_above_tolerance > 0:
        message = (
            f"Final validation found {final_far_blocks_above_tolerance} far blocks "
            f"above the projection tolerance {far_error_tol:.3e}. Refine patches "
            "or relax the tolerance only after an accuracy study."
        )
        if rank_guard_action == "error":
            raise CompressionGuardError(message)
        log("WARNING: " + message, verbose)

    m_report = max(p.M_requested for p in patches)
    pqr = legacy.digital_pqr_from_dirs(
        legacy.fibonacci_directions(max(m_report, 1)), W
    )[:m_report]

    patch_summary: List[Dict[str, Any]] = []
    for patch_id, patch in enumerate(patches):
        n_local = len(patch.indices)
        cap = rank_cap_for_patch(
            n_local, max_rank_fraction, max_rank_absolute
        )
        patch_record: Dict[str, Any] = {
            "patch": patch_id,
            "n_points": n_local,
            "diameter": float(patch.diameter),
            "normal_spread_deg": float(patch.normal_spread_deg),
            "region": int(patch.region),
            "M_requested": int(patch.M_requested),
            "rank_stored": int(patch.basis.shape[1]),
            "rank_cap": int(cap),
            "rank_fraction": float(patch.basis.shape[1] / max(n_local, 1)),
            "rank_guard_reached": bool(patch.M_requested >= cap),
        }
        patch_record.update(legacy.feature_subset_summary(geom, patch.indices))
        patch_summary.append(patch_record)

    monitor.snapshot(
        "model_complete",
        near_blocks=near_count,
        far_blocks=far_count,
        retained_single_layer_blocks=0,
        far_blocks_above_tolerance=final_far_blocks_above_tolerance,
    )
    return SolverModelLite(
        geom=geom,
        ka=ka,
        a=a,
        k=k,
        W=W,
        B=patch_count,
        M=m_report,
        labels=labels,
        patches=patches,
        d_blocks=d_blocks,
        single_layer_on_vn=single_layer_on_vn,
        near_blocks=near_count,
        far_blocks=far_count,
        pqr=pqr,
        lambda_acoustic=float(lambda_acoustic),
        lambda_velocity=lambda_velocity,
        lambda_star=float(lambda_star),
        max_patch_diameter_target=max_patch_diameter,
        patch_summary=patch_summary,
        far_error_summary=far_error_summary,
        memory_estimate=memory_estimate,
    )


def apply_d_operator(model: SolverModelLite, vector: np.ndarray) -> np.ndarray:
    result = np.zeros_like(vector, dtype=np.complex128)
    for block in model.d_blocks:
        pa = model.patches[block.receiver_patch]
        pb = model.patches[block.source_patch]
        receiver = pa.indices
        source = pb.indices
        if block.kind == "near":
            result[receiver] += block.matrix @ vector[source]
        else:
            source_coeff = pb.basis.conj().T @ vector[source]
            result[receiver] += pa.basis @ (block.matrix @ source_coeff)
    return result


# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner -- local LU solves, never matrix inversion
# ---------------------------------------------------------------------------


@dataclass
class BlockJacobiPreconditioner:
    factors: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    regularized_patches: List[int]

    def apply(self, vector: np.ndarray) -> np.ndarray:
        result = np.zeros_like(vector, dtype=np.complex128)
        for indices, lu, piv in self.factors:
            result[indices] = lu_solve((lu, piv), vector[indices], check_finite=False)
        return result


def build_block_jacobi_preconditioner(
    model: SolverModelLite,
    regularization: float,
    verbose: bool,
) -> BlockJacobiPreconditioner:
    self_blocks = {
        block.receiver_patch: block
        for block in model.d_blocks
        if block.receiver_patch == block.source_patch
    }
    factors: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    regularized: List[int] = []

    for patch_id, patch in enumerate(model.patches):
        block = self_blocks.get(patch_id)
        if block is None or block.kind != "near":
            raise RuntimeError(f"Missing dense self block for patch {patch_id}.")
        n_local = len(patch.indices)
        local_a = 0.5 * np.eye(n_local, dtype=np.complex128) - block.matrix
        needs_regularization = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LinAlgWarning)
            lu, piv = lu_factor(
                local_a.copy(), overwrite_a=True, check_finite=False
            )
            if any(issubclass(item.category, LinAlgWarning) for item in caught):
                needs_regularization = True
        diag_abs = np.abs(np.diag(lu))
        if (
            not np.all(np.isfinite(diag_abs))
            or diag_abs.size == 0
            or np.min(diag_abs) <= 1e-13 * max(np.max(diag_abs), 1.0)
        ):
            needs_regularization = True

        if needs_regularization:
            scale = max(float(np.linalg.norm(local_a, ord=np.inf)), 1.0)
            local_a = local_a + (regularization * scale) * np.eye(
                n_local, dtype=np.complex128
            )
            lu, piv = lu_factor(
                local_a, overwrite_a=True, check_finite=False
            )
            regularized.append(patch_id)
        factors.append((patch.indices, lu, piv))

    log(
        f"Block-Jacobi preconditioner: {len(factors)} local LU factors; "
        f"regularized patches={len(regularized)}.",
        verbose,
    )
    return BlockJacobiPreconditioner(factors, regularized)


# ---------------------------------------------------------------------------
# Matrix-free Krylov solve, optional small-N direct verification/fallback
# ---------------------------------------------------------------------------


def make_system_operator(model: SolverModelLite) -> LinearOperator:
    n_points = len(model.geom.vn)

    def matvec(pressure: np.ndarray) -> np.ndarray:
        return 0.5 * pressure - apply_d_operator(model, pressure)

    return LinearOperator(
        (n_points, n_points), matvec=matvec, dtype=np.complex128
    )


def relative_residual(
    operator: LinearOperator,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(operator.matvec(solution) - rhs)
        / (np.linalg.norm(rhs) + 1e-30)
    )


def assemble_compressed_dense_operator(model: SolverModelLite) -> np.ndarray:
    """Assemble the DFT-approximated matrix for small-case verification only."""
    n_points = len(model.geom.vn)
    matrix = 0.5 * np.eye(n_points, dtype=np.complex128)
    for block in model.d_blocks:
        pa = model.patches[block.receiver_patch]
        pb = model.patches[block.source_patch]
        receiver = pa.indices
        source = pb.indices
        if block.kind == "near":
            d_approx = block.matrix
        else:
            d_approx = pa.basis @ block.matrix @ pb.basis.conj().T
        matrix[np.ix_(receiver, source)] -= d_approx
    return matrix


def solve_pressure(
    model: SolverModelLite,
    rtol: float,
    maxiter: int,
    restart: int,
    preconditioner_name: str,
    preconditioner_regularization: float,
    fallback_krylov: str,
    direct_fallback_max_n: int,
    monitor: ResourceMonitor,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rhs = 1j * model.k * model.single_layer_on_vn
    operator = make_system_operator(model)
    n_points = len(rhs)

    preconditioner = None
    preconditioner_operator = None
    if preconditioner_name == "block-jacobi":
        preconditioner = build_block_jacobi_preconditioner(
            model,
            regularization=preconditioner_regularization,
            verbose=verbose,
        )
        preconditioner_operator = LinearOperator(
            (n_points, n_points),
            matvec=preconditioner.apply,
            dtype=np.complex128,
        )
        monitor.snapshot(
            "preconditioner_complete",
            local_lu_factors=len(preconditioner.factors),
            regularized_patches=len(preconditioner.regularized_patches),
        )

    gmres_residuals: List[float] = []

    def gmres_callback(value: Any) -> None:
        try:
            gmres_residuals.append(float(value))
        except Exception:
            pass

    log(
        f"Solving by matrix-free GMRES, preconditioner={preconditioner_name}, "
        f"restart={restart}, maxiter={maxiter}...",
        verbose,
    )
    started = time.perf_counter()
    try:
        pressure, info = gmres(
            operator,
            rhs,
            M=preconditioner_operator,
            rtol=rtol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            callback=gmres_callback,
            callback_type="pr_norm",
        )
    except TypeError:  # older SciPy compatibility
        pressure, info = gmres(
            operator,
            rhs,
            M=preconditioner_operator,
            tol=rtol,
            restart=restart,
            maxiter=maxiter,
            callback=gmres_callback,
        )
    gmres_seconds = time.perf_counter() - started
    residual = relative_residual(operator, pressure, rhs)
    method_used = "gmres"
    fallback_info: Optional[int] = None
    fallback_iterations = 0

    log(
        f"GMRES info={info}, true relative residual={residual:.3e}, "
        f"callback iterations={len(gmres_residuals)}.",
        verbose,
    )
    monitor.snapshot(
        "gmres_complete",
        gmres_info=int(info),
        relative_residual=residual,
        callback_iterations=len(gmres_residuals),
    )

    if (info != 0 or residual > max(10.0 * rtol, 1e-10)) and fallback_krylov == "lgmres":
        log("GMRES did not meet the target; trying matrix-free LGMRES...", verbose)
        lgmres_count = 0

        def lgmres_callback(_x: np.ndarray) -> None:
            nonlocal lgmres_count
            lgmres_count += 1

        started_fallback = time.perf_counter()
        candidate, fallback_info = lgmres(
            operator,
            rhs,
            x0=pressure,
            M=preconditioner_operator,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            inner_m=max(10, restart),
            outer_k=3,
            callback=lgmres_callback,
        )
        fallback_seconds = time.perf_counter() - started_fallback
        candidate_residual = relative_residual(operator, candidate, rhs)
        fallback_iterations = lgmres_count
        if candidate_residual < residual:
            pressure = candidate
            residual = candidate_residual
            method_used = "lgmres"
        log(
            f"LGMRES info={fallback_info}, residual={candidate_residual:.3e}, "
            f"outer iterations={lgmres_count}.",
            verbose,
        )
        monitor.snapshot(
            "lgmres_complete",
            lgmres_info=int(fallback_info),
            relative_residual=candidate_residual,
            outer_iterations=lgmres_count,
            elapsed_solver_s=fallback_seconds,
        )

    direct_used = False
    direct_seconds = 0.0
    if (
        (residual > max(10.0 * rtol, 1e-10))
        and direct_fallback_max_n > 0
        and n_points <= direct_fallback_max_n
    ):
        dense_mb = n_points * n_points * 16.0 / (1024.0**2)
        if monitor.memory_budget_mb > 0:
            predicted = monitor.current_rss_mb() + 2.5 * dense_mb
            if predicted > 0.9 * monitor.memory_budget_mb:
                raise MemoryBudgetExceeded(
                    "Small-N direct fallback was requested, but its dense matrix/LU "
                    f"would raise estimated RSS to {predicted:.1f} MiB."
                )
        log(
            "Krylov methods did not meet the target; using optional small-N "
            "LAPACK LU fallback (factorization/triangular solves, no inverse).",
            verbose,
        )
        started_direct = time.perf_counter()
        dense_operator = assemble_compressed_dense_operator(model)
        lu, piv = lu_factor(
            dense_operator, overwrite_a=True, check_finite=False
        )
        candidate = lu_solve((lu, piv), rhs, check_finite=False)
        candidate_residual = relative_residual(operator, candidate, rhs)
        direct_seconds = time.perf_counter() - started_direct
        del dense_operator, lu, piv
        if candidate_residual < residual:
            pressure = candidate
            residual = candidate_residual
            method_used = "dense_lapack_lu"
            direct_used = True
        monitor.snapshot(
            "direct_fallback_complete",
            relative_residual=candidate_residual,
            elapsed_solver_s=direct_seconds,
        )

    converged = residual <= max(10.0 * rtol, 1e-10)
    stats: Dict[str, Any] = {
        "method_used": method_used,
        "converged": bool(converged),
        "requested_rtol": float(rtol),
        "relative_residual": float(residual),
        "gmres_info": int(info),
        "gmres_callback_iterations": len(gmres_residuals),
        "gmres_seconds": gmres_seconds,
        "gmres_preconditioned_residual_history": gmres_residuals,
        "preconditioner": preconditioner_name,
        "preconditioner_regularized_patches": (
            preconditioner.regularized_patches if preconditioner else []
        ),
        "fallback_krylov": fallback_krylov,
        "fallback_info": None if fallback_info is None else int(fallback_info),
        "fallback_iterations": fallback_iterations,
        "direct_fallback_enabled_below_N": int(direct_fallback_max_n),
        "direct_fallback_used": direct_used,
        "direct_fallback_seconds": direct_seconds,
        "explicit_matrix_inverse_used": False,
    }
    return pressure, stats


# ---------------------------------------------------------------------------
# Results and output -- no matrices or models are serialized
# ---------------------------------------------------------------------------


def compute_outputs(
    model: SolverModelLite,
    pressure_normalized: np.ndarray,
    rho: float,
    c: float,
) -> Dict[str, Any]:
    area = model.geom.area
    velocity = model.geom.vn
    denominator = np.sum(area * np.abs(velocity) ** 2) + 1e-30
    impedance_normalized = (
        np.sum(area * pressure_normalized * np.conj(velocity)) / denominator
    )
    power_normalized = 0.5 * np.real(
        np.sum(area * pressure_normalized * np.conj(velocity))
    )
    acoustic_scale = rho * c
    pressure_pa = acoustic_scale * pressure_normalized
    impedance_physical = acoustic_scale * impedance_normalized
    power_w = acoustic_scale * power_normalized
    point_power_w = 0.5 * np.real(area * pressure_pa * np.conj(velocity))
    return {
        "surface_impedance_normalized": impedance_normalized,
        "surface_impedance_physical_Pa_s_per_m": impedance_physical,
        "radiated_power_rhoc1": float(power_normalized),
        "radiated_power_W": float(power_w),
        "pressure_pa": pressure_pa,
        "point_power_w": point_power_w,
        "max_pressure_peak_Pa": float(np.max(np.abs(pressure_pa))),
        "rho_kg_m3": float(rho),
        "c_m_s": float(c),
    }


def feature_audit(
    geom: Geometry,
    patch_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    features = geom.features
    if features is None:
        return {
            "metadata_present": False,
            "boundary_mode": "off",
            "hard_group_count": 0,
            "patches_crossing_hard_feature_boundaries": 0,
        }

    classes = features.feature_class
    metadata_group_count = int(len(np.unique(features.group_code)))
    audit: Dict[str, Any] = {
        "metadata_present": True,
        "boundary_mode": features.boundary_mode,
        "detected_columns": features.detected_columns,
        "metadata_group_count": metadata_group_count,
        "hard_group_count": (
            0 if features.boundary_mode == "off" else metadata_group_count
        ),
        "connectivity_radius": float(features.connectivity_radius),
        "feature_type_map": features.feature_type_map,
        "zero_feature_id_is_missing": bool(features.zero_feature_id_is_missing),
        "class_counts": {
            name: int(np.sum(classes == name))
            for name in ("surface", "edge", "corner", "feature")
        },
        "face_id_count": int(len({value for value in features.face_id if value})),
        "edge_id_count": int(len({value for value in features.edge_id if value})),
        "corner_id_count": int(len({value for value in features.corner_id if value})),
        "feature_id_count": int(len({value for value in features.feature_id if value})),
    }
    crossing = 0
    if patch_labels is not None and features.boundary_mode != "off":
        for patch_id in np.unique(patch_labels):
            inds = np.where(patch_labels == patch_id)[0]
            if len(np.unique(features.group_code[inds])) > 1:
                crossing += 1
    audit["patches_crossing_hard_feature_boundaries"] = int(crossing)
    return audit


def feature_group_dataframe(
    geom: Geometry,
    patch_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    features = geom.features
    if features is None:
        return pd.DataFrame(
            columns=[
                "feature_group_code", "feature_group_label", "n_points",
                "area", "patch_count", "feature_class", "feature_type",
                "face_id", "edge_id", "corner_id", "feature_id",
            ]
        )

    records: List[Dict[str, Any]] = []
    for group_code in np.unique(features.group_code):
        inds = np.where(features.group_code == group_code)[0]
        summary = legacy.feature_subset_summary(geom, inds)
        record: Dict[str, Any] = {
            "feature_group_code": int(group_code),
            "feature_group_label": summary["feature_group_label"],
            "n_points": int(len(inds)),
            "area": float(np.sum(geom.area[inds])),
            "patch_count": (
                int(len(np.unique(patch_labels[inds])))
                if patch_labels is not None
                else 0
            ),
            "feature_class": summary["feature_class"],
            "feature_type": summary["feature_type"],
            "face_id": summary["face_id"],
            "edge_id": summary["edge_id"],
            "corner_id": summary["corner_id"],
            "feature_id": summary["feature_id"],
            "x_min": float(np.min(geom.xyz[inds, 0])),
            "x_max": float(np.max(geom.xyz[inds, 0])),
            "y_min": float(np.min(geom.xyz[inds, 1])),
            "y_max": float(np.max(geom.xyz[inds, 1])),
            "z_min": float(np.min(geom.xyz[inds, 2])),
            "z_max": float(np.max(geom.xyz[inds, 2])),
        }
        records.append(record)
    return pd.DataFrame(records).sort_values("feature_group_code")


def geometry_audit(
    geom: Geometry,
    patch_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    total_area = float(np.sum(geom.area))
    closure_vector = np.sum(geom.area[:, None] * geom.normals, axis=0)
    closure_ratio = float(
        np.linalg.norm(closure_vector) / max(total_area, 1e-30)
    )
    normal_lengths = np.linalg.norm(geom.normals, axis=1)
    return {
        "total_area": total_area,
        "normal_closure_vector": closure_vector.tolist(),
        "normal_closure_ratio": closure_ratio,
        "normal_length_max_error": float(np.max(np.abs(normal_lengths - 1.0))),
        "area_min": float(np.min(geom.area)),
        "area_max": float(np.max(geom.area)),
        "area_nonpositive_count": int(np.sum(geom.area <= 0.0)),
        "feature_metadata": feature_audit(geom, patch_labels),
    }


def save_outputs(
    model: SolverModelLite,
    pressure_normalized: np.ndarray,
    stats: Dict[str, Any],
    metrics: Dict[str, Any],
    out_prefix: Path,
    case_id: Optional[str],
    frequency_hz: Optional[float],
    monitor: ResourceMonitor,
    verbose: bool,
) -> Dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    geom = model.geom
    pressure_pa = np.asarray(metrics["pressure_pa"])
    point_power_w = np.asarray(metrics["point_power_w"])

    result_df = pd.DataFrame(
        {
            "N": geom.index,
            "x": geom.xyz[:, 0],
            "y": geom.xyz[:, 1],
            "z": geom.xyz[:, 2],
            "nx": geom.normals[:, 0],
            "ny": geom.normals[:, 1],
            "nz": geom.normals[:, 2],
            "area": geom.area,
            "vn_real": np.real(geom.vn),
            "vn_imag": np.imag(geom.vn),
            "p_normalized_real": np.real(pressure_normalized),
            "p_normalized_imag": np.imag(pressure_normalized),
            "p_Pa_real": np.real(pressure_pa),
            "p_Pa_imag": np.imag(pressure_pa),
            "p_Pa_abs_peak": np.abs(pressure_pa),
            "p_phase_rad": np.angle(pressure_pa),
            "point_active_power_W": point_power_w,
            "patch": model.labels,
        }
    )
    if geom.digital_lmn is not None:
        result_df.insert(4, "l", geom.digital_lmn[:, 0])
        result_df.insert(5, "m", geom.digital_lmn[:, 1])
        result_df.insert(6, "n", geom.digital_lmn[:, 2])
    if geom.features is not None:
        features = geom.features
        result_df["feature_group_code"] = features.group_code
        result_df["feature_group_label"] = features.group_label
        result_df["feature_class"] = features.feature_class
        result_df["feature_type"] = features.feature_type_raw
        result_df["face_id"] = features.face_id
        result_df["edge_id"] = features.edge_id
        result_df["corner_id"] = features.corner_id
        result_df["feature_id"] = features.feature_id

    pressure_path = Path(str(out_prefix) + "_pressure.csv")
    patch_path = Path(str(out_prefix) + "_patch_summary.csv")
    feature_path = Path(str(out_prefix) + "_feature_summary.csv")
    pqr_path = Path(str(out_prefix) + "_fibonacci_pqr.csv")
    report_path = Path(str(out_prefix) + "_report.json")

    result_df.to_csv(pressure_path, index=False)
    pd.DataFrame(model.patch_summary).to_csv(patch_path, index=False)
    feature_group_dataframe(geom, model.labels).to_csv(feature_path, index=False)
    pd.DataFrame(model.pqr, columns=["p", "q", "r"]).to_csv(
        pqr_path, index=False
    )

    z_norm = metrics["surface_impedance_normalized"]
    z_phys = metrics["surface_impedance_physical_Pa_s_per_m"]
    report: Dict[str, Any] = {
        "case_id": case_id,
        "frequency_hz": frequency_hz,
        "N_points": int(len(geom.index)),
        "B_patches": int(model.B),
        "M_plane_wave_terms_max_per_far_block": int(model.M),
        "W_multiplier": int(model.W),
        "ka": float(model.ka),
        "a": float(model.a),
        "k": float(model.k),
        "lambda_acoustic": float(model.lambda_acoustic),
        "lambda_velocity": model.lambda_velocity,
        "lambda_star": float(model.lambda_star),
        "max_patch_diameter_target": model.max_patch_diameter_target,
        "near_blocks_dense": int(model.near_blocks),
        "far_blocks_dft_compressed": int(model.far_blocks),
        "retained_single_layer_blocks": 0,
        "far_block_error_control": model.far_error_summary,
        "memory_estimate": model.memory_estimate,
        "solver": stats,
        "geometry_audit": geometry_audit(geom, model.labels),
        "surface_impedance_normalized": {
            "real": float(np.real(z_norm)),
            "imag": float(np.imag(z_norm)),
        },
        "surface_impedance_physical_Pa_s_per_m": {
            "real": float(np.real(z_phys)),
            "imag": float(np.imag(z_phys)),
        },
        "radiated_power_rhoc1": metrics["radiated_power_rhoc1"],
        "radiated_power_W": metrics["radiated_power_W"],
        "max_pressure_peak_Pa": metrics["max_pressure_peak_Pa"],
        "rho_kg_m3": metrics["rho_kg_m3"],
        "c_m_s": metrics["c_m_s"],
        "resource_log": monitor.records,
        "implementation_notes": [
            "The boundary equation and patch-DFT Green representation are unchanged.",
            "No explicit matrix inverse is formed.",
            "GMRES/LGMRES use a matrix-free D operator.",
            "The block-Jacobi preconditioner uses independent LAPACK LU factors of self-patch blocks.",
            "S_DFT is applied once to the prescribed normal velocity and S blocks are discarded.",
            "Only final pressure vectors, patch summaries, directions, and reports are saved.",
            "Dense LAPACK LU is disabled by default and is available only as an explicit small-N fallback.",
            "Preprocessor face, edge, corner, and feature metadata are carried to outputs and used as hard patch boundaries according to feature_boundary_mode.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    log(f"Saved pressure vector: {pressure_path}", verbose)
    log(f"Saved feature summary: {feature_path}", verbose)
    log(f"Saved report: {report_path}", verbose)
    return {
        "pressure_csv": str(pressure_path),
        "patch_summary_csv": str(patch_path),
        "feature_summary_csv": str(feature_path),
        "pqr_csv": str(pqr_path),
        "report_json": str(report_path),
    }


# ---------------------------------------------------------------------------
# CLI -- exactly one acoustic case per worker process
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Memory-lean patch-DFT acoustic solver for one case per process"
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        help="CSV with x,y,z,vn_real/vn_imag and recommended normals/area",
    )
    parser.add_argument("--case-id", default=None)
    parser.add_argument(
        "--feature-metadata-csv",
        default=None,
        help="Optional separate preprocessor metadata CSV joined to the acoustic input by point ID.",
    )
    parser.add_argument(
        "--feature-geometry-key-column",
        default=None,
        help="Point-ID column in the acoustic geometry CSV for a separate metadata join.",
    )
    parser.add_argument(
        "--feature-metadata-key-column",
        default=None,
        help="Point-ID column in the separate feature metadata CSV.",
    )
    parser.add_argument(
        "--feature-boundary-mode",
        choices=["auto", "strict", "face-only", "off"],
        default="auto",
        help=(
            "How preprocessor topology metadata constrains compression patches. "
            "auto uses strict edge/corner/feature groups when present, otherwise face-only."
        ),
    )
    parser.add_argument("--face-column", default=None, help="Explicit face/surface/region ID column name.")
    parser.add_argument("--edge-column", default=None, help="Explicit edge ID column name.")
    parser.add_argument("--corner-column", default=None, help="Explicit corner/vertex ID column name.")
    parser.add_argument("--feature-type-column", default=None, help="Explicit feature type/class column name.")
    parser.add_argument(
        "--feature-type-map",
        default="",
        help="Optional code map such as '0:surface,1:edge,2:corner' or a JSON object.",
    )
    parser.add_argument("--feature-id-column", default=None, help="Explicit general feature ID column name.")
    parser.add_argument("--is-edge-column", default=None, help="Optional Boolean edge-flag column name.")
    parser.add_argument("--is-corner-column", default=None, help="Optional Boolean corner-flag column name.")
    parser.add_argument(
        "--feature-zero-id-is-valid",
        action="store_true",
        help=(
            "By default zero edge/corner/general-feature IDs are treated as missing "
            "sentinels. Set this flag when zero is a real feature identifier."
        ),
    )
    parser.add_argument(
        "--feature-connectivity-factor",
        type=float,
        default=2.5,
        help=(
            "When edge/corner points have no IDs, connect points within this multiple "
            "of the median point spacing to create automatic feature components."
        ),
    )
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=None,
        help="When supplied, compute k=2*pi*f/c and ka=k*a.",
    )
    parser.add_argument("--ka", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.204)
    parser.add_argument("--c", type=float, default=343.0)
    parser.add_argument("--W", type=int, default=100)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--lambda-v", type=float, default=None)
    parser.add_argument("--patches-per-wavelength", type=float, default=6.0)
    parser.add_argument("--max-patch-diameter", type=float, default=None)
    parser.add_argument("--max-normal-angle-deg", type=float, default=180.0)
    parser.add_argument("--min-points-per-patch", type=int, default=4)
    parser.add_argument(
        "--max-points-per-patch",
        type=int,
        default=96,
        help="0 disables point-count splitting; 64-128 is a practical range for large cases.",
    )
    parser.add_argument("--max-patch-count", type=int, default=512)
    parser.add_argument("--far-error-tol", type=float, default=1e-4)
    parser.add_argument(
        "--M-schedule", default="8,12,16,24,32,48,64,96,128"
    )
    parser.add_argument("--disable-adaptive-M", action="store_true")
    parser.add_argument("--max-adapt-passes", type=int, default=4)
    parser.add_argument(
        "--max-adapt-far-blocks",
        type=int,
        default=0,
        help="0 checks all far blocks; positive values use a deterministic sample.",
    )
    parser.add_argument("--near-threshold", type=float, default=0.75)
    parser.add_argument("--self-d-model", choices=["zero", "cap"], default="cap")
    parser.add_argument("--max-rank-fraction", type=float, default=0.70)
    parser.add_argument("--max-rank-absolute", type=int, default=0)
    parser.add_argument(
        "--rank-guard-action", choices=["warn", "error"], default="warn"
    )
    parser.add_argument(
        "--preconditioner",
        choices=["block-jacobi", "none"],
        default="block-jacobi",
    )
    parser.add_argument(
        "--preconditioner-regularization", type=float, default=1e-10
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--gmres-restart", type=int, default=50)
    parser.add_argument(
        "--fallback-krylov", choices=["none", "lgmres"], default="lgmres"
    )
    parser.add_argument(
        "--direct-fallback-max-n",
        type=int,
        default=0,
        help="Disabled at 0. Never use this as the large-N production path.",
    )
    parser.add_argument(
        "--memory-budget-mb",
        type=float,
        default=0.0,
        help="0 disables the guard. Streamlit workers should set a real budget.",
    )
    parser.add_argument(
        "--resource-log",
        default=None,
        help="JSONL resource log path. Defaults to <out>_resources.jsonl.",
    )
    parser.add_argument("--out", default="dam_dft_output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--make-demo-sphere", default=None)
    parser.add_argument("--demo-N", type=int, default=720)
    parser.add_argument("--demo-radius", type=float, default=1.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet

    if args.make_demo_sphere:
        legacy.write_demo_sphere_csv(
            Path(args.make_demo_sphere),
            args.demo_N,
            args.demo_radius,
            verbose=verbose,
        )
        return 0
    if not args.input_csv:
        parser.error("input_csv is required unless --make-demo-sphere is used")

    out_prefix = Path(args.out)
    resource_log = (
        Path(args.resource_log)
        if args.resource_log
        else Path(str(out_prefix) + "_resources.jsonl")
    )
    monitor = ResourceMonitor(
        resource_log,
        memory_budget_mb=args.memory_budget_mb,
        verbose=verbose,
    )

    geom: Optional[Geometry] = None
    model: Optional[SolverModelLite] = None
    pressure: Optional[np.ndarray] = None
    exit_code = 1
    try:
        monitor.snapshot("worker_start")
        geom = legacy.load_geometry(
            Path(args.input_csv),
            a=args.a,
            verbose=verbose,
            face_column=args.face_column,
            edge_column=args.edge_column,
            corner_column=args.corner_column,
            feature_type_column=args.feature_type_column,
            feature_id_column=args.feature_id_column,
            is_edge_column=args.is_edge_column,
            is_corner_column=args.is_corner_column,
            feature_boundary_mode=args.feature_boundary_mode,
            feature_connectivity_factor=args.feature_connectivity_factor,
            feature_metadata_csv=(
                Path(args.feature_metadata_csv) if args.feature_metadata_csv else None
            ),
            feature_geometry_key_column=args.feature_geometry_key_column,
            feature_metadata_key_column=args.feature_metadata_key_column,
            zero_feature_id_is_missing=not args.feature_zero_id_is_valid,
            feature_type_map=args.feature_type_map,
        )
        loaded_feature_audit = feature_audit(geom)
        monitor.snapshot(
            "geometry_loaded",
            N=len(geom.index),
            feature_metadata_present=loaded_feature_audit.get("metadata_present", False),
            feature_boundary_mode=loaded_feature_audit.get("boundary_mode", "off"),
            feature_group_count=loaded_feature_audit.get("hard_group_count", 0),
        )

        if args.frequency_hz is not None:
            k = 2.0 * np.pi * args.frequency_hz / args.c
            ka = k * args.a
        else:
            ka = args.ka

        model = build_model_lite(
            geom=geom,
            ka=ka,
            a=args.a,
            W=args.W,
            B_user=args.B,
            M_user=args.M,
            near_threshold=args.near_threshold,
            self_d_model=args.self_d_model,
            lambda_velocity=args.lambda_v,
            patches_per_wavelength=args.patches_per_wavelength,
            max_patch_diameter=args.max_patch_diameter,
            max_normal_angle_deg=args.max_normal_angle_deg,
            min_points_per_patch=args.min_points_per_patch,
            max_points_per_patch=args.max_points_per_patch,
            far_error_tol=args.far_error_tol,
            m_schedule=legacy.parse_m_schedule(args.M_schedule),
            disable_adaptive_m=args.disable_adaptive_M,
            max_rank_fraction=args.max_rank_fraction,
            max_rank_absolute=(
                args.max_rank_absolute if args.max_rank_absolute > 0 else None
            ),
            rank_guard_action=args.rank_guard_action,
            max_adapt_passes=args.max_adapt_passes,
            max_adapt_far_blocks=args.max_adapt_far_blocks,
            max_patch_count=args.max_patch_count,
            monitor=monitor,
            verbose=verbose,
        )
        pressure, stats = solve_pressure(
            model=model,
            rtol=args.rtol,
            maxiter=args.maxiter,
            restart=args.gmres_restart,
            preconditioner_name=args.preconditioner,
            preconditioner_regularization=args.preconditioner_regularization,
            fallback_krylov=args.fallback_krylov,
            direct_fallback_max_n=args.direct_fallback_max_n,
            monitor=monitor,
            verbose=verbose,
        )
        metrics = compute_outputs(
            model, pressure, rho=args.rho, c=args.c
        )
        monitor.snapshot(
            "metrics_complete",
            radiated_power_W=metrics["radiated_power_W"],
            max_pressure_peak_Pa=metrics["max_pressure_peak_Pa"],
        )
        paths = save_outputs(
            model=model,
            pressure_normalized=pressure,
            stats=stats,
            metrics=metrics,
            out_prefix=out_prefix,
            case_id=args.case_id,
            frequency_hz=args.frequency_hz,
            monitor=monitor,
            verbose=verbose,
        )
        monitor.snapshot("outputs_saved", **paths)
        exit_code = 0 if stats["converged"] else 2
    except Exception as exc:
        error_record = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            monitor.snapshot("worker_failed", **error_record)
        except Exception:
            pass
        error_path = Path(str(out_prefix) + "_error.json")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(error_record, indent=2), encoding="utf-8"
        )
        log(f"ERROR: {type(exc).__name__}: {exc}", True)
        exit_code = 1
    finally:
        # A worker handles exactly one case. Explicit cleanup is still useful on
        # long-lived local servers and ensures no model is retained by callers.
        pressure = None
        model = None
        geom = None
        gc.collect()
        try:
            monitor.snapshot("worker_cleanup_complete")
        except Exception:
            pass

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
