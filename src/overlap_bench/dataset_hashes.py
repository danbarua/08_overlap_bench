"""H0c: sha256 pins for the CAE benchmark files, locked in DESIGN.labkit.md.

``data/CAE/datasets/`` in the exploration log is gitignored there, so its
commit pins nothing. This table is the lock; a mismatch means the copied
file is not the one the design document's numbers were computed against.
"""

import hashlib
from pathlib import Path

from overlap_bench.paths import CAE_DIR

# Locked table, DESIGN.labkit.md "On the question: the benchmark".
LOCKED_SHA256: dict[str, str] = {
    "2shapes_train.npz": "304b3224e453429e7fef1ee96fcff1c3c3cf138ce0804ca31f6dc2d135006730",
    "2shapes_val.npz": "9772425914944b31a07b40fdea967aab08d36d2461f202387eea6690c66fbb59",
    "2shapes_test.npz": "e9dbb9faa8acdf068ba9711347d31a44b96789962315031e7e2cc1af448c0693",
    "3shapes_train.npz": "f79e21ca86e151f010e51ed4c9ededcbc8f515d03bb2261ef209c9c3b6ea4064",
    "3shapes_val.npz": "9baec24f4e4dded964794e52f71a1dc934505533845d275f2d29aa35a6e24984",
    "3shapes_test.npz": "4abd3f666c58ec5d18cc313b92bfd492793569065dde20d7a14a51fb69013807",
    "MNIST_shapes_train.npz": "b0777f2803df09cf1dadfda9a84e093d8be139e2fcb6be6a4142ce2d180b718e",
    "MNIST_shapes_val.npz": "09900916adbd1d5011b285c5a1f3613c5969bb5cb87e76f43e33e545b2474d23",
    "MNIST_shapes_test.npz": "1ae2507abaf01254aa65a62e78bc3283881ca310e2b1bb35cc21430a9d4189b7",
}

CAE_DIR = Path(CAE_DIR)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_dataset_hashes(directory: Path | None = None) -> dict[str, str]:
    """Return {filename: actual_sha256} for every locked file; raise on any mismatch."""
    directory = directory if directory is not None else CAE_DIR
    actual = {}
    mismatches = []
    for filename, expected in LOCKED_SHA256.items():
        path = directory / filename
        if not path.exists():
            mismatches.append(f"{filename}: missing at {path}")
            continue
        digest = sha256_of(path)
        actual[filename] = digest
        if digest != expected:
            mismatches.append(f"{filename}: expected {expected}, got {digest}")
    if mismatches:
        raise ValueError("H0c failed:\n" + "\n".join(mismatches))
    return actual
