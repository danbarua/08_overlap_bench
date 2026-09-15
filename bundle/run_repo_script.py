#!/usr/bin/env python3
"""Mighty-colab bundle adapter: extract 07_posn, set up imports, run target script.

This script is the entry point for mighty-colab jobs that use the 08_overlap_bench
codebase. It:

1. Extracts the 07_posn git bundle into the job workspace
2. Checks out the pinned commit
3. Sets up sys.path to import from 08_overlap_bench/src
4. Changes to the 08_overlap_bench directory
5. Tees the target script's stdout/stderr to outputs/<script>.log
6. Runs the target script with all remaining arguments

Step 5 exists because mighty-colab does not stream or capture a job's own
stdout/stderr anywhere durable -- only its own [job] phase transitions reach
the local apply.log. Without this, a script's entire progress trace (loss
curves, per-image timing, any traceback printed before a crash) is gone the
moment the VM is, leaving nothing to diagnose a failure with beyond mighty
-colab's own generic reason string. Declare outputs/<script>.log as a
(required: false) artifact in the job spec so it offloads on completion --
including on_run_fail: offload_anyway -- alongside the real outputs.

Typical usage in a mighty-colab job spec:

    code:
      kind: bundle
      root: ./bundle
      entry: run_repo_script.py
      args:
        - scripts/run_pursuit_b.py
        - --device
        - cuda
        - --steps
        - "20000"
        - --out
        - outputs/pursuit-b-20000.json
"""

from pathlib import Path
import os
import runpy
import subprocess
import sys


class _Tee:
    """Write to both an underlying stream and a log file, flushing every write.

    Line-buffered on both ends: a killed process leaves as much of the
    transcript on disk as was actually printed, not just whatever fit in an
    OS-level buffer.
    """

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log_file.write(data)
        self._log_file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()


HERE = Path(__file__).resolve().parent
JOB_ROOT = HERE.parent
OVERLAP = HERE / "08_overlap_bench"
POSN_FOR_BUNDLE = HERE / "07_posn"  # Where we extract it in the bundle
POSN_COMMIT = "18a60649d927efb1504472cb33bd52b22db0f277"


# Extract 07_posn from the bundle if not already present
if not POSN_FOR_BUNDLE.exists():
    print(f"[bundle] Extracting 07_posn from bundle...")
    subprocess.run(
        ["git", "clone", str(JOB_ROOT / "inputs" / "07_posn.bundle"), str(POSN_FOR_BUNDLE)],
        check=True,
    )
    subprocess.run(["git", "checkout", POSN_COMMIT], cwd=POSN_FOR_BUNDLE, check=True)
    print(f"[bundle] 07_posn ready at {POSN_FOR_BUNDLE}")

# Set up imports and working directory
# Patch REFERENCE_REPO to point to where we extracted it
os.environ["REFERENCE_REPO_OVERRIDE"] = str(POSN_FOR_BUNDLE)
target = OVERLAP / sys.argv[1]
sys.argv = [str(target), *sys.argv[2:]]
sys.path.insert(0, str(OVERLAP / "src"))
os.chdir(OVERLAP)

# Tee stdout/stderr to a durable log file before running the target, so its
# full progress trace survives even if the process is killed or the job
# fails partway through -- see module docstring.
log_path = OVERLAP / "outputs" / f"{target.stem}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
log_file = open(log_path, "a", buffering=1)
sys.stdout = _Tee(sys.stdout, log_file)
sys.stderr = _Tee(sys.stderr, log_file)

# Run the target script
print(f"[bundle] Running {target.name} from {OVERLAP}, log at {log_path}")
runpy.run_path(str(target), run_name="__main__")
