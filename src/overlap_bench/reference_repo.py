"""Access to 07_posn's own code, at the commit DESIGN.labkit.md pins.

Only ``07_posn``'s root goes on ``sys.path``, and only here. Never 08's own
root: 07's ``src`` is a regular package (has ``src/__init__.py``), so once
07's root is on the path, any competing top-level ``src`` or ``defns`` from
this project would either shadow it or be shadowed by it. This project's own
package is imported as ``overlap_bench`` via ``src/`` on the path (see
``pyproject.toml``'s ``pythonpath``), never via 08's repo root -- so there is
nothing here for 07's modules to collide with.

Note: In mighty-colab bundle environments, REFERENCE_REPO may be overridden
via REFERENCE_REPO_OVERRIDE environment variable for path isolation.
"""

import os
import subprocess
import sys

from overlap_bench.paths import REFERENCE_COMMIT, REFERENCE_REPO as _DEFAULT_REFERENCE_REPO

# Allow override in bundle environments
REFERENCE_REPO = os.environ.get("REFERENCE_REPO_OVERRIDE", _DEFAULT_REFERENCE_REPO)

_inserted = False


def verify_reference_commit() -> str:
    """Return 07_posn's HEAD sha; raise if it does not match the pinned commit,
    or if the imported paths (src/, defns.py) have uncommitted changes.

    07's working tree is known to be dirty outside src/ (DESIGN.labkit.md's
    isolation note: modified .gitignore, untracked docs/_archive/ and
    fixtures) -- that's expected and out of scope. What H0a's "the code is
    the reference's" actually needs is that src/ and defns.py, specifically,
    match the pinned commit exactly.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REFERENCE_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != REFERENCE_COMMIT:
        raise ValueError(
            f"07_posn HEAD is {head}, DESIGN.labkit.md pins {REFERENCE_COMMIT}. "
            "The reference implementation moved since the design was locked; "
            "this is a new pursuit, not a harness bug -- do not silently re-pin."
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "defns.py"],
        cwd=REFERENCE_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty.strip():
        raise ValueError(
            f"07_posn's src/ or defns.py has uncommitted changes at HEAD "
            f"{head}:\n{dirty}\nThe code is not shown to be the reference's "
            "while these paths differ from the pinned commit."
        )
    return head


def ensure_on_path() -> None:
    """Idempotently put 07_posn's root on sys.path, after verifying its commit."""
    global _inserted
    verify_reference_commit()
    if not _inserted:
        sys.path.insert(0, REFERENCE_REPO)
        _inserted = True
