#!/usr/bin/env bash
# Create a git bundle from the 07_posn repository.
# This bundle is used as input data for mighty-colab jobs.
#
# Usage:
#   ./scripts/create_posn_bundle.sh [OUTPUT_PATH] [POSN_REPO_PATH]
#
# Default:
#   ./scripts/create_posn_bundle.sh /tmp/07_posn.bundle ~/Code/AI/07_posn

set -euo pipefail

OUTPUT="${1:-/tmp/07_posn.bundle}"
POSN_REPO="${2:-$HOME/Code/AI/07_posn}"

if [ ! -d "$POSN_REPO/.git" ]; then
    echo "ERROR: 07_posn repository not found at $POSN_REPO" >&2
    exit 1
fi

echo "Creating git bundle from $POSN_REPO..." >&2
(cd "$POSN_REPO" && git bundle create "$OUTPUT" --all)

if [ ! -f "$OUTPUT" ]; then
    echo "ERROR: Bundle creation failed" >&2
    exit 1
fi

SIZE=$(du -h "$OUTPUT" | cut -f1)
SHA256=$(sha256sum "$OUTPUT" | cut -d' ' -f1)

echo "Bundle created: $OUTPUT" >&2
echo "Size: $SIZE" >&2
echo "SHA256: $SHA256" >&2
echo ""
echo "Add to job spec data section:"
echo "  - url: gs://labkit-build-colab-jobs/inputs/07_posn.bundle?X-Goog-Signature=..."
echo "    dest: inputs/07_posn.bundle"
echo "    sha256: $SHA256"
