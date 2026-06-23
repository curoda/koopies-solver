#!/usr/bin/env python3
"""Single-concurrency shared-folder worker for DAM DFT Streamlit jobs.

The Streamlit container writes one job directory containing job_spec.json. This
worker claims one unprocessed job at a time, runs dft_worker_job.py, and only
then looks for the next job. A shared volume is sufficient; no model matrices
cross the process/container boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "dft_worker_job.py"


def try_claim(job_dir: Path) -> Optional[int]:
    claim_path = job_dir / "worker.claim"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"pid={os.getpid()}\nclaimed={time.time()}\n".encode("utf-8"))
    return fd


def job_is_terminal(job_dir: Path) -> bool:
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
    except Exception:
        return False
    return state in {"completed", "completed_unconverged", "failed", "cancelled"}


def find_next_job(root: Path) -> Optional[Path]:
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for job_dir in candidates:
        if not (job_dir / "job_spec.json").exists():
            continue
        if job_is_terminal(job_dir):
            continue
        fd = try_claim(job_dir)
        if fd is None:
            continue
        os.close(fd)
        return job_dir
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="DAM DFT shared-folder worker")
    parser.add_argument(
        "--job-root",
        default=os.environ.get("DFT_JOB_ROOT", str(HERE / "streamlit_jobs")),
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = Path(args.job_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Worker watching {root}", flush=True)
    while True:
        job_dir = find_next_job(root)
        if job_dir is None:
            if args.once:
                return 0
            time.sleep(max(args.poll_seconds, 0.2))
            continue
        print(f"Claimed {job_dir.name}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(WRAPPER), str(job_dir / "job_spec.json")],
            cwd=str(HERE),
            check=False,
        )
        print(
            f"Finished {job_dir.name} with return code {completed.returncode}",
            flush=True,
        )
        if args.once:
            return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
