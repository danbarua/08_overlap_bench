#!/usr/bin/env python3
"""Upload files to GCS and generate signed URLs for mighty-colab jobs.

Usage:
    python scripts/gcs_upload_and_sign.py [--bucket BUCKET] [--signer SA] [--region REGION]

Requires:
    - gcloud CLI with per-directory config (CLOUDSDK_CONFIG in environment)
    - Authentication: gcloud auth application-default login
    - Impersonation permission on the signer service account
"""

import subprocess
import json
import argparse
from pathlib import Path
import os
import sys


def upload_file(local_path, gcs_path, bucket):
    """Upload file to GCS."""
    full_gcs = f"gs://{bucket}/{gcs_path}"
    print(f"Uploading {local_path} to {full_gcs}...", file=sys.stderr)
    result = subprocess.run(
        ["gcloud", "storage", "cp", local_path, full_gcs],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}", file=sys.stderr)
        return False
    print(f"  OK", file=sys.stderr)
    return True


def create_placeholder(gcs_path, bucket):
    """Create a placeholder object in GCS."""
    full_gcs = f"gs://{bucket}/{gcs_path}"
    print(f"Creating placeholder {gcs_path}...", file=sys.stderr)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-", full_gcs],
        input=b"{}",
        capture_output=True,
        timeout=30
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.decode()}", file=sys.stderr)
        return False
    print(f"  OK", file=sys.stderr)
    return True


def sign_url(gcs_path, verb, duration_hours, bucket, signer, region):
    """Sign a GCS URL."""
    cmd = [
        "gcloud", "storage", "sign-url", f"gs://{bucket}/{gcs_path}",
        f"--impersonate-service-account={signer}",
        f"--region={region}",
        f"--http-verb={verb}",
        f"--duration={duration_hours}h",
        "--format=value(signed_url)"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"ERROR signing {gcs_path} {verb}: {result.stderr}", file=sys.stderr)
        return None
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Upload files to GCS and generate signed URLs"
    )
    parser.add_argument(
        "--bucket", default="labkit-build-colab-jobs",
        help="GCS bucket name (default: labkit-build-colab-jobs)"
    )
    parser.add_argument(
        "--signer", default="colab-job-signer@labkit-build.iam.gserviceaccount.com",
        help="Service account for signing (default: colab-job-signer@labkit-build.iam.gserviceaccount.com)"
    )
    parser.add_argument(
        "--region", default="US",
        help="GCS region for signing (default: US)"
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip upload; only sign URLs"
    )
    args = parser.parse_args()

    bucket = args.bucket
    signer = args.signer
    region = args.region

    # Files to upload: (local_path, gcs_path)
    files_to_upload = [
        ("data/cae/2shapes_train.npz", "inputs/2shapes_train.npz"),
        ("data/cae/2shapes_val.npz", "inputs/2shapes_val.npz"),
        ("/tmp/07_posn.bundle", "inputs/07_posn.bundle"),
    ]

    if not args.skip_upload:
        print("=== Uploading input files ===", file=sys.stderr)
        for local, gcs in files_to_upload:
            if Path(local).exists():
                upload_file(local, gcs, bucket)
            else:
                print(f"WARNING: {local} not found, skipping", file=sys.stderr)

    # Create control-result placeholders
    print("\n=== Creating control-result placeholders ===", file=sys.stderr)
    for job_name in ["pursuit-b-20000", "pursuit-b-mechanism"]:
        obj_path = f"control/{job_name}/result.json"
        create_placeholder(obj_path, bucket)

    # Sign all URLs
    print("\n=== Signing URLs ===", file=sys.stderr)
    urls = {}

    # Input GET URLs (2-hour duration)
    for local, gcs in files_to_upload:
        print(f"Signing GET {gcs}...", file=sys.stderr)
        url = sign_url(gcs, "GET", 2, bucket, signer, region)
        if url:
            urls[f"{gcs}:GET"] = url

    # Output PUT URLs (8-hour duration)
    artifacts = [
        "outputs/pursuit-b-20000.json",
        "outputs/pursuit-b-20000.timing.json",
        "outputs/ckpt-20000.pt",
        "outputs/pursuit-b-mechanism.json",
    ]
    for gcs in artifacts:
        print(f"Signing PUT {gcs}...", file=sys.stderr)
        url = sign_url(gcs, "PUT", 8, bucket, signer, region)
        if url:
            urls[f"{gcs}:PUT"] = url

    # Control result URLs (PUT and GET, 8-hour)
    controls = [
        "control/pursuit-b-20000/result.json",
        "control/pursuit-b-mechanism/result.json",
    ]
    for gcs in controls:
        print(f"Signing PUT {gcs}...", file=sys.stderr)
        put_url = sign_url(gcs, "PUT", 8, bucket, signer, region)
        if put_url:
            urls[f"{gcs}:PUT"] = put_url
        
        print(f"Signing GET {gcs}...", file=sys.stderr)
        get_url = sign_url(gcs, "GET", 8, bucket, signer, region)
        if get_url:
            urls[f"{gcs}:GET"] = get_url

    # Output as JSON
    print("\n=== Signed URLs ===", file=sys.stderr)
    print(json.dumps(urls, indent=2))
    return urls


if __name__ == "__main__":
    main()
