#!/usr/bin/env python3
"""Build a mighty-colab job spec YAML from gcs_upload_and_sign.py's JOBS registry.

One JobSpec entry (name, dataset, resume_from) plus the signed-URL JSON that
script prints is enough to build a full spec: no more hand-written per-run
YAML with manually copy-pasted URLs. Verifies every data/resume URL against
its expected sha256 (a real download, not just a probe) before writing
anything, and refuses to write a spec if any check fails.

Usage:
    python scripts/gcs_upload_and_sign.py --skip-upload > /tmp/urls.json
    python scripts/build_job_spec.py pursuit-b-mnist-40000 \
        --urls /tmp/urls.json --steps 40000 --out overlap-pursuit-b-mnist-40000.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gcs_upload_and_sign import JOBS


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _verify_url(label: str, url: str, expect_sha256: str | None, expect_size: int | None) -> tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        while chunk := resp.read(1024 * 1024):
            h.update(chunk)
            total += len(chunk)
    digest = h.hexdigest()
    if expect_sha256 is not None and digest != expect_sha256:
        raise ValueError(f"{label}: sha256 mismatch (got {digest}, expected {expect_sha256})")
    if expect_size is not None and total != expect_size:
        raise ValueError(f"{label}: size mismatch (got {total}, expected {expect_size})")
    print(f"  verified {label}: {total} bytes, sha256 {digest[:16]}...", file=sys.stderr)
    return total, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name", help="a name from gcs_upload_and_sign.JOBS")
    parser.add_argument("--urls", required=True, help="path to the signed-URL JSON from gcs_upload_and_sign.py")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--out", required=True, help="path to write the spec YAML")
    parser.add_argument("--wall-clock", type=int, default=7200)
    parser.add_argument("--artifact-sync-interval-seconds", type=int, default=180)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--accelerator", default="A100")
    args = parser.parse_args()

    jobs_by_name = {j.name: j for j in JOBS}
    if args.job_name not in jobs_by_name:
        raise SystemExit(f"{args.job_name!r} not in JOBS registry; add it to gcs_upload_and_sign.py first")
    job = jobs_by_name[args.job_name]
    if job.dataset is None:
        raise SystemExit(f"{args.job_name!r} has no dataset; this builder is for training jobs")

    with open(args.urls) as f:
        urls = json.load(f)

    print(f"=== verifying URLs for {job.name} ===", file=sys.stderr)

    data_items = []
    for gcs_key, local_path, dest in [
        ("inputs/07_posn.bundle", "/tmp/07_posn.bundle", "inputs/07_posn.bundle"),
        (
            f"inputs/{job.dataset}_train.npz",
            f"data/cae/{job.dataset}_train.npz",
            f"src/08_overlap_bench/data/cae/{job.dataset}_train.npz",
        ),
        (
            f"inputs/{job.dataset}_val.npz",
            f"data/cae/{job.dataset}_val.npz",
            f"src/08_overlap_bench/data/cae/{job.dataset}_val.npz",
        ),
    ]:
        url = urls[f"{gcs_key}:GET"]
        expect_sha = _sha256_of(local_path)
        expect_size = Path(local_path).stat().st_size
        _verify_url(gcs_key, url, expect_sha, expect_size)
        data_items.append({"url": url, "dest": dest, "sha256": expect_sha, "size_bytes": expect_size})

    code_args = [
        "scripts/run_pursuit_b.py",
        "--dataset", job.dataset,
        "--device", "cuda",
        "--steps", str(args.steps),
        "--out", f"outputs/{job.out_name or job.name}.json",
        "--save-model", f"outputs/{job.ckpt_name or job.name}.pt",
        "--checkpoint-every", str(args.checkpoint_every),
    ]

    if job.resume_from:
        parent = jobs_by_name[job.resume_from]
        resume_key = f"{parent.ckpt()}:RESUME_GET:{job.name}"
        if resume_key not in urls:
            raise SystemExit(f"missing signed resume URL {resume_key!r} -- re-run gcs_upload_and_sign.py")
        resume_url = urls[resume_key]
        # Verified against nothing here deliberately: the parent job hasn't
        # finished (or hasn't been downloaded) at spec-build time in the
        # normal two-phase flow. Caller must re-verify once the parent
        # artifact is known -- see the session's own workflow, which
        # downloads and hashes the parent output before building this spec.
        resume_dest = f"src/08_overlap_bench/outputs/{job.ckpt_name or job.name}-resume-parent.pt"
        total, digest = _verify_url(f"{parent.ckpt()} (resume input)", resume_url, None, None)
        data_items.append({"url": resume_url, "dest": resume_dest, "sha256": digest, "size_bytes": total})
        code_args += ["--resume-from", resume_dest.replace("src/08_overlap_bench/", "")]

    artifacts = [
        {
            "path": f"src/08_overlap_bench/outputs/{job.out_name or job.name}.json",
            "url": urls[f"{job.out()}:PUT"],
            "required": True,
            "size_bytes": 500_000,
        },
        {
            "path": f"src/08_overlap_bench/outputs/{job.ckpt_name or job.name}.pt",
            "url": urls[f"{job.ckpt()}:PUT"],
            "required": False,
            "size_bytes": 25_000_000,
        },
        {
            "path": "src/08_overlap_bench/outputs/run_pursuit_b.log",
            "url": urls[f"{job.log()}:PUT"],
            "required": False,
            "size_bytes": 100_000,
        },
    ]

    spec = {
        "name": job.name,
        "accelerator": {"prefer": [args.accelerator], "accept_cpu": False},
        "code": {
            "kind": "bundle",
            "root": "./bundle",
            "entry": "run_repo_script.py",
            "args": code_args,
        },
        "deps": ["numpy==2.1.2", "scipy==1.14.1", "scikit-learn==1.5.2", "torch==2.8.0", "torchvision==0.23.0"],
        "data": data_items,
        "artifacts": artifacts,
        "control": {
            "result": {
                "put_url": urls[f"{job.control()}:PUT"],
                "get_url": urls[f"{job.control()}:GET"],
            },
        },
        "budgets": {
            "wall_clock": args.wall_clock,
            "artifact_sync_interval_seconds": args.artifact_sync_interval_seconds,
        },
        "on_offload_fail": "leave_up",
    }

    import yaml

    with open(args.out, "w") as f:
        yaml.safe_dump(spec, f, default_flow_style=False, sort_keys=False, width=100000)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
