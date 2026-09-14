#!/usr/bin/env bash
# Sync src/ and scripts/ into bundle/08_overlap_bench/ so mighty-colab job
# bundles ship the code that's actually on disk.
#
# run_repo_script.py (the job entry point) expects bundle/08_overlap_bench/
# to already contain the repo's src/ and scripts/ trees -- it only extracts
# 07_posn from a git bundle at runtime, it does not clone this repo. Run this
# before every `mighty-colab job plan`/`apply`, or the job will fail at
# `run` with a FileNotFoundError for whatever script you pointed `code.args`
# at, well after paying for provision/install/verify/stage.
#
# Usage:
#   ./scripts/sync_bundle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/bundle/08_overlap_bench"

mkdir -p "$DEST"
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    "$ROOT/src" "$DEST/"
rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
    "$ROOT/scripts" "$DEST/"

echo "Synced $ROOT/{src,scripts} -> $DEST" >&2
du -sh "$DEST" >&2
