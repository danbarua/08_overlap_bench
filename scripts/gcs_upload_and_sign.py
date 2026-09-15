#!/usr/bin/env python3
"""Upload files to GCS and generate signed URLs for mighty-colab jobs.

Describe each job once, by name and (optionally) dataset and resume parent.
Data inputs, artifacts, and control placeholders are all derived from that
one declaration -- no hand-maintained, copy-pasted per-run path lists. Every
job that ran this session before this rewrite is still in JOBS below so its
control placeholder and signed URLs stay reproducible; extending the roster
for a new run means adding one JobSpec, not editing three parallel lists.

Usage:
    python scripts/gcs_upload_and_sign.py [--bucket BUCKET] [--signer SA] [--region REGION]
    python scripts/gcs_upload_and_sign.py --skip-upload   # datasets already uploaded

Requires:
    - gcloud CLI with per-directory config (CLOUDSDK_CONFIG in environment)
    - Authentication: gcloud auth application-default login
    - Impersonation permission on the signer service account
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

BUCKET_DEFAULT = "labkit-build-colab-jobs"
SIGNER_DEFAULT = "colab-job-signer@labkit-build.iam.gserviceaccount.com"
REGION_DEFAULT = "US"
DURATION_HOURS = 8  # matches every URL class: data GET, artifact/control PUT+GET.
# A shorter GET window bought nothing -- the job's own wall_clock budget is
# ~90min and credentials of this class already sit in the mode-0600 secrets
# sidecar for 8h regardless. A short window only forces re-signing mid
# debugging session, racing the clock against unrelated iteration.


@dataclass
class JobSpec:
    """One mighty-colab job's derived data/artifact/control paths.

    `name` is both the job's control-result directory and (by default) its
    output/checkpoint/log basename. Override `out_name`/`ckpt_name` only to
    match an already-established, differently-named artifact from before
    this registry existed.
    """

    name: str
    dataset: str | None = None  # locked CAE benchmark this job trains/evaluates on
    resume_from: str | None = None  # name of an earlier JobSpec whose checkpoint this resumes
    out_name: str | None = None
    ckpt_name: str | None = None
    has_checkpoint: bool = True
    has_log: bool = True
    extra_inputs: list[tuple[str, str]] = field(default_factory=list)  # (local_path, gcs_path)

    def out(self) -> str:
        return f"outputs/{self.out_name or self.name}.json"

    def ckpt(self) -> str:
        return f"outputs/{self.ckpt_name or self.name}.pt"

    def log(self) -> str:
        return f"outputs/{self.out_name or self.name}.log"

    def control(self) -> str:
        return f"control/{self.name}/result.json"


DATASETS = ["2shapes", "MNIST_shapes", "3shapes"]

JOBS = [
    JobSpec("pursuit-b-1000-verify", dataset="2shapes", ckpt_name="ckpt-1000-verify"),
    JobSpec("pursuit-b-20000", dataset="2shapes", ckpt_name="ckpt-20000"),
    JobSpec("pursuit-b-mechanism", has_checkpoint=False, has_log=False),
    JobSpec("pursuit-b-attractor", has_checkpoint=False, has_log=False, out_name="attractor-10000"),
    JobSpec("pursuit-b-mnist-10000", dataset="MNIST_shapes", ckpt_name="ckpt-mnist-10000"),
    JobSpec("pursuit-b-3shapes-10000", dataset="3shapes", ckpt_name="ckpt-3shapes-10000"),
    JobSpec("pursuit-b-mnist-20000", dataset="MNIST_shapes", ckpt_name="ckpt-mnist-20000"),
    JobSpec("pursuit-b-3shapes-20000", dataset="3shapes", ckpt_name="ckpt-3shapes-20000"),
    JobSpec(
        "pursuit-b-mnist-40000",
        dataset="MNIST_shapes",
        ckpt_name="ckpt-mnist-40000",
        resume_from="pursuit-b-mnist-20000",
    ),
    JobSpec(
        "pursuit-b-3shapes-40000",
        dataset="3shapes",
        ckpt_name="ckpt-3shapes-40000",
        resume_from="pursuit-b-3shapes-20000",
    ),
    JobSpec(
        "pursuit-b-mnist-attractor-40000",
        dataset="MNIST_shapes",
        has_checkpoint=False,
        resume_from="pursuit-b-mnist-40000",
    ),
    JobSpec(
        "pursuit-b-3shapes-attractor-40000",
        dataset="3shapes",
        has_checkpoint=False,
        resume_from="pursuit-b-3shapes-40000",
    ),
]

DATASET_FILES = {
    "2shapes": [
        ("data/cae/2shapes_train.npz", "inputs/2shapes_train.npz"),
        ("data/cae/2shapes_val.npz", "inputs/2shapes_val.npz"),
    ],
    "MNIST_shapes": [
        ("data/cae/MNIST_shapes_train.npz", "inputs/MNIST_shapes_train.npz"),
        ("data/cae/MNIST_shapes_val.npz", "inputs/MNIST_shapes_val.npz"),
    ],
    "3shapes": [
        ("data/cae/3shapes_train.npz", "inputs/3shapes_train.npz"),
        ("data/cae/3shapes_val.npz", "inputs/3shapes_val.npz"),
    ],
}
COMMON_FILES = [
    ("/tmp/07_posn.bundle", "inputs/07_posn.bundle"),
]


def upload_file(local_path: str, gcs_path: str, bucket: str) -> bool:
    full_gcs = f"gs://{bucket}/{gcs_path}"
    print(f"Uploading {local_path} to {full_gcs}...", file=sys.stderr)
    result = subprocess.run(
        ["gcloud", "storage", "cp", local_path, full_gcs],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        return False
    print("  OK", file=sys.stderr)
    return True


def create_placeholder(gcs_path: str, bucket: str) -> bool:
    full_gcs = f"gs://{bucket}/{gcs_path}"
    print(f"Creating placeholder {gcs_path}...", file=sys.stderr)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-", full_gcs],
        input=b"{}", capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.decode()}", file=sys.stderr)
        return False
    print("  OK", file=sys.stderr)
    return True


def sign_url(gcs_path: str, verb: str, duration_hours: int, bucket: str, signer: str, region: str) -> str | None:
    cmd = [
        "gcloud", "storage", "sign-url", f"gs://{bucket}/{gcs_path}",
        f"--impersonate-service-account={signer}",
        f"--region={region}",
        f"--http-verb={verb}",
        f"--duration={duration_hours}h",
        "--format=value(signed_url)",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"ERROR signing {gcs_path} {verb}: {result.stderr}", file=sys.stderr)
        return None
    return result.stdout.strip()


def main() -> dict[str, str]:
    parser = argparse.ArgumentParser(description="Upload files to GCS and generate signed URLs")
    parser.add_argument("--bucket", default=BUCKET_DEFAULT)
    parser.add_argument("--signer", default=SIGNER_DEFAULT)
    parser.add_argument("--region", default=REGION_DEFAULT)
    parser.add_argument("--skip-upload", action="store_true", help="Skip dataset/bundle upload; only sign URLs")
    args = parser.parse_args()

    bucket, signer, region = args.bucket, args.signer, args.region

    datasets_used = sorted({job.dataset for job in JOBS if job.dataset})
    files_to_upload = list(COMMON_FILES)
    for ds in datasets_used:
        files_to_upload += DATASET_FILES[ds]
    for job in JOBS:
        files_to_upload += job.extra_inputs

    if not args.skip_upload:
        print("=== Uploading input files ===", file=sys.stderr)
        for local, gcs in files_to_upload:
            if Path(local).exists():
                upload_file(local, gcs, bucket)
            else:
                print(f"WARNING: {local} not found, skipping", file=sys.stderr)

    print("\n=== Creating control-result placeholders ===", file=sys.stderr)
    for job in JOBS:
        create_placeholder(job.control(), bucket)

    print("\n=== Signing URLs ===", file=sys.stderr)
    urls: dict[str, str] = {}

    for local, gcs in files_to_upload:
        print(f"Signing GET {gcs}...", file=sys.stderr)
        url = sign_url(gcs, "GET", DURATION_HOURS, bucket, signer, region)
        if url:
            urls[f"{gcs}:GET"] = url

    for job in JOBS:
        for gcs in [job.out(), *([job.ckpt()] if job.has_checkpoint else []), *([job.log()] if job.has_log else [])]:
            print(f"Signing PUT {gcs}...", file=sys.stderr)
            put_url = sign_url(gcs, "PUT", DURATION_HOURS, bucket, signer, region)
            if put_url:
                urls[f"{gcs}:PUT"] = put_url

        print(f"Signing PUT {job.control()}...", file=sys.stderr)
        put_url = sign_url(job.control(), "PUT", DURATION_HOURS, bucket, signer, region)
        if put_url:
            urls[f"{job.control()}:PUT"] = put_url
        print(f"Signing GET {job.control()}...", file=sys.stderr)
        get_url = sign_url(job.control(), "GET", DURATION_HOURS, bucket, signer, region)
        if get_url:
            urls[f"{job.control()}:GET"] = get_url

    # Resume-input GET URLs: a job with `resume_from` downloads its parent's
    # checkpoint from the exact GCS path the parent's own artifact PUT wrote
    # to. No separate upload -- the object already exists there once the
    # parent job's own offload has completed.
    by_name = {job.name: job for job in JOBS}
    for job in JOBS:
        if not job.resume_from:
            continue
        parent = by_name[job.resume_from]
        gcs = parent.ckpt()
        print(f"Signing GET {gcs} (resume input for {job.name})...", file=sys.stderr)
        url = sign_url(gcs, "GET", DURATION_HOURS, bucket, signer, region)
        if url:
            urls[f"{gcs}:RESUME_GET:{job.name}"] = url

    print("\n=== Signed URLs ===", file=sys.stderr)
    print(json.dumps(urls, indent=2))
    return urls


if __name__ == "__main__":
    main()
