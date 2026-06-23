#!/usr/bin/env python3
"""Execute one serialized DAM DFT job and write durable status files."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: dft_worker_job.py JOB_SPEC.json", file=sys.stderr)
        return 2
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    job_dir = spec_path.parent
    status_path = job_dir / "status.json"
    log_path = job_dir / "worker.log"
    command = [str(value) for value in spec["command"]]
    started = time.time()
    atomic_write_json(
        status_path,
        {
            "state": "running",
            "started_unix": started,
            "worker_pid": __import__("os").getpid(),
            "command": command,
        },
    )
    try:
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=spec.get("working_directory") or str(job_dir),
                check=False,
            )
        finished = time.time()
        state = "completed" if completed.returncode == 0 else (
            "completed_unconverged" if completed.returncode == 2 else "failed"
        )
        atomic_write_json(
            status_path,
            {
                "state": state,
                "started_unix": started,
                "finished_unix": finished,
                "elapsed_s": finished - started,
                "return_code": completed.returncode,
                "worker_pid": __import__("os").getpid(),
                "command": command,
            },
        )
        return completed.returncode
    except Exception as exc:
        finished = time.time()
        atomic_write_json(
            status_path,
            {
                "state": "failed",
                "started_unix": started,
                "finished_unix": finished,
                "elapsed_s": finished - started,
                "return_code": 1,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
