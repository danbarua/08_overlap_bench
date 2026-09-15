#!/usr/bin/env python3
"""Download and hash-verify every artifact a completed mighty-colab job produced.

Replaces the repeated by-hand pattern of: sign a GET URL for one artifact,
urllib.request it, compute sha256, compare to the envelope, repeat per file.
Reads the job's own local envelope.json for the artifact list (path, sha256,
bytes) -- no need to remember or retype GCS paths.

Usage:
    python scripts/fetch_job_artifacts.py <job_id> [--bucket BUCKET] [--signer SA] [--region REGION]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

JOBS_DIR = Path.home() / ".config" / "colab-cli" / "jobs"


def sign_get(gcs_path: str, bucket: str, signer: str, region: str, duration_hours: int = 1) -> str:
    cmd = [
        "gcloud", "storage", "sign-url", f"gs://{bucket}/{gcs_path}",
        f"--impersonate-service-account={signer}",
        f"--region={region}",
        "--http-verb=GET",
        f"--duration={duration_hours}h",
        "--format=value(signed_url)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"sign-url failed for {gcs_path}: {result.stderr}")
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--bucket", default="labkit-build-colab-jobs")
    parser.add_argument("--signer", default="colab-job-signer@labkit-build.iam.gserviceaccount.com")
    parser.add_argument("--region", default="US")
    parser.add_argument("--dest-dir", default="outputs", help="local directory to write into")
    args = parser.parse_args()

    envelope_path = JOBS_DIR / args.job_id / "envelope.json"
    if not envelope_path.exists():
        raise SystemExit(f"no local envelope at {envelope_path} -- run `mighty-colab job status {args.job_id}` first")
    envelope = json.loads(envelope_path.read_text())

    if envelope.get("workload") != "succeeded":
        print(f"WARNING: workload={envelope.get('workload')!r}, not 'succeeded' -- fetching whatever artifacts exist anyway", file=sys.stderr)

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for art in envelope.get("artifacts", []):
        if art.get("status") != "ok":
            print(f"SKIP {art['path']}: status={art.get('status')!r}", file=sys.stderr)
            ok = False
            continue
        # url_id is "https://.../<gcs_path>#<redacted-fragment>" -- the GCS
        # path is everything between the bucket host and the fragment.
        url_id = art["url_id"]
        gcs_path = url_id.split(".com/", 1)[1].split("#", 1)[0]
        local_name = Path(art["path"]).name
        dest = dest_dir / local_name

        signed = sign_get(gcs_path, args.bucket, args.signer, args.region)
        h = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(signed, timeout=120) as resp, open(dest, "wb") as out:
            while chunk := resp.read(1024 * 1024):
                h.update(chunk)
                total += len(chunk)
                out.write(chunk)
        digest = h.hexdigest()
        expect = art.get("sha256")
        match = expect is None or digest == expect
        status = "OK" if match else "MISMATCH"
        print(f"{status} {dest} ({total} bytes, sha256 {digest[:16]}...)", file=sys.stderr)
        if not match:
            print(f"  expected {expect}", file=sys.stderr)
            ok = False

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
