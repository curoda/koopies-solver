#!/usr/bin/env python3
"""Run a CSV manifest sequentially, one DAM DFT case per fresh worker process.

This runner is intentionally simple. A child process builds and solves one mode,
writes only final results, exits, and releases all model memory before the next
mode starts. It is suitable for a private Streamlit server. For public/cloud
production, the same command can be submitted to an external queue/container.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
SOLVER = HERE / "dam_dft_solver_streamlit_ready.py"


def add_optional(command: List[str], flag: str, value: object) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return
    command.extend([flag, text])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sequential one-case-per-process runner for DAM DFT cases"
    )
    parser.add_argument("manifest_csv")
    parser.add_argument("--output-dir", default="dft_batch_results")
    parser.add_argument("--memory-budget-mb", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError("Manifest contains no cases.")

    records: List[Dict[str, object]] = []
    overall_code = 0
    for case_number, row in enumerate(rows, start=1):
        case_id = (row.get("case_id") or f"case_{case_number:03d}").strip()
        input_value = (row.get("input_csv") or "").strip()
        if not input_value:
            raise ValueError(f"Manifest row {case_number} has no input_csv.")
        input_path = Path(input_value)
        if not input_path.is_absolute():
            input_path = (manifest_path.parent / input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = case_dir / case_id
        log_path = case_dir / "worker.log"
        command = [
            args.python,
            str(SOLVER),
            str(input_path),
            "--case-id",
            case_id,
            "--out",
            str(out_prefix),
            "--resource-log",
            str(case_dir / "resources.jsonl"),
        ]
        if args.memory_budget_mb > 0:
            command.extend(["--memory-budget-mb", str(args.memory_budget_mb)])

        feature_metadata_value = (row.get("feature_metadata_csv") or "").strip()
        if feature_metadata_value:
            feature_metadata_path = Path(feature_metadata_value)
            if not feature_metadata_path.is_absolute():
                feature_metadata_path = (manifest_path.parent / feature_metadata_path).resolve()
            if not feature_metadata_path.exists():
                raise FileNotFoundError(feature_metadata_path)
            command.extend(["--feature-metadata-csv", str(feature_metadata_path)])

        field_map = {
            "frequency_hz": "--frequency-hz",
            "feature_boundary_mode": "--feature-boundary-mode",
            "feature_geometry_key_column": "--feature-geometry-key-column",
            "feature_metadata_key_column": "--feature-metadata-key-column",
            "face_column": "--face-column",
            "edge_column": "--edge-column",
            "corner_column": "--corner-column",
            "feature_type_column": "--feature-type-column",
            "feature_type_map": "--feature-type-map",
            "feature_id_column": "--feature-id-column",
            "is_edge_column": "--is-edge-column",
            "is_corner_column": "--is-corner-column",
            "feature_connectivity_factor": "--feature-connectivity-factor",
            "ka": "--ka",
            "a": "--a",
            "rho": "--rho",
            "c": "--c",
            "W": "--W",
            "B": "--B",
            "M": "--M",
            "lambda_v": "--lambda-v",
            "patches_per_wavelength": "--patches-per-wavelength",
            "max_patch_diameter": "--max-patch-diameter",
            "max_normal_angle_deg": "--max-normal-angle-deg",
            "min_points_per_patch": "--min-points-per-patch",
            "max_points_per_patch": "--max-points-per-patch",
            "max_patch_count": "--max-patch-count",
            "far_error_tol": "--far-error-tol",
            "M_schedule": "--M-schedule",
            "max_adapt_passes": "--max-adapt-passes",
            "max_adapt_far_blocks": "--max-adapt-far-blocks",
            "near_threshold": "--near-threshold",
            "self_d_model": "--self-d-model",
            "max_rank_fraction": "--max-rank-fraction",
            "max_rank_absolute": "--max-rank-absolute",
            "rank_guard_action": "--rank-guard-action",
            "preconditioner": "--preconditioner",
            "preconditioner_regularization": "--preconditioner-regularization",
            "rtol": "--rtol",
            "maxiter": "--maxiter",
            "gmres_restart": "--gmres-restart",
            "fallback_krylov": "--fallback-krylov",
            "direct_fallback_max_n": "--direct-fallback-max-n",
        }
        for field, flag in field_map.items():
            add_optional(command, flag, row.get(field))
        if str(row.get("disable_adaptive_M", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }:
            command.append("--disable-adaptive-M")
        if str(row.get("feature_zero_id_is_valid", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }:
            command.append("--feature-zero-id-is-valid")
        extra_args = (row.get("extra_args") or "").strip()
        if extra_args:
            command.extend(shlex.split(extra_args))

        started = time.time()
        print(f"[{case_number}/{len(rows)}] Starting {case_id}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.run(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                cwd=str(HERE),
                check=False,
            )
        elapsed = time.time() - started
        report_path = Path(str(out_prefix) + "_report.json")
        record: Dict[str, object] = {
            "case_id": case_id,
            "input_csv": str(input_path),
            "return_code": process.returncode,
            "elapsed_s": elapsed,
            "worker_log": str(log_path),
            "report_json": str(report_path) if report_path.exists() else "",
        }
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "solver_method": report.get("solver", {}).get("method_used"),
                    "converged": report.get("solver", {}).get("converged"),
                    "relative_residual": report.get("solver", {}).get(
                        "relative_residual"
                    ),
                    "radiated_power_W": report.get("radiated_power_W"),
                    "max_pressure_peak_Pa": report.get("max_pressure_peak_Pa"),
                    "N_points": report.get("N_points"),
                    "B_patches": report.get("B_patches"),
                    "M_max": report.get("M_plane_wave_terms_max_per_far_block"),
                    "feature_boundary_mode": report.get("geometry_audit", {}).get(
                        "feature_metadata", {}
                    ).get("boundary_mode"),
                    "hard_feature_groups": report.get("geometry_audit", {}).get(
                        "feature_metadata", {}
                    ).get("hard_group_count"),
                    "feature_patch_boundary_violations": report.get(
                        "geometry_audit", {}
                    ).get("feature_metadata", {}).get(
                        "patches_crossing_hard_feature_boundaries"
                    ),
                }
            )
        records.append(record)
        pd.DataFrame(records).to_csv(output_dir / "batch_summary.csv", index=False)
        print(
            f"[{case_number}/{len(rows)}] Finished {case_id}: "
            f"return_code={process.returncode}, elapsed={elapsed:.1f}s",
            flush=True,
        )
        if process.returncode not in (0, 2):
            overall_code = 1
            if args.stop_on_error:
                break
        elif process.returncode == 2:
            overall_code = max(overall_code, 2)

    print(f"Batch summary: {output_dir / 'batch_summary.csv'}", flush=True)
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
