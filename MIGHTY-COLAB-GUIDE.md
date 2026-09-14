# Running Pursuit B on mighty-colab

This document explains how to run the 20,000-step Pursuit B training and mechanism audit on Google Colab via mighty-colab.

## Prerequisites

- `mighty-colab` CLI installed
- GCS bucket and signing service account set up (see `gcloud-dogfood.sh`)
- Per-directory gcloud config with valid ADC (Application Default Credentials)
- All input files uploaded to GCS

## Quick Start: Run the 20,000-step job

```bash
cd ~/Code/AI/08_overlap_bench

# Plan the job (no VM allocated yet)
mighty-colab job plan overlap-pursuit-b-20000.yaml

# Apply the plan to run on Colab
mighty-colab job apply overlap-pursuit-b-20000.yaml --timeout 3600

# Poll for completion
JID=$(mighty-colab --json job plan overlap-pursuit-b-20000.yaml | jq -r .job_id)
mighty-colab job status "$JID" --poll
```

## Detailed Steps

### 1. Ensure GCS files are ready

All input files must be uploaded and signed URLs valid:

```bash
# Create 07_posn bundle (one-time)
bash scripts/create_posn_bundle.sh /tmp/07_posn.bundle

# Upload files and generate signed URLs
python scripts/gcs_upload_and_sign.py \
  --bucket labkit-build-colab-jobs \
  --signer colab-job-signer@labkit-build.iam.gserviceaccount.com \
  --region US
```

This outputs signed URLs. Copy them into the job spec:

```yaml
data:
  - url: [signed GET URL from script output]
    dest: inputs/07_posn.bundle
    sha256: ee863262b2dfc86f6d32a6225625557e7abc24b0aecc1ac89a5908e78c9b74c4
  # ... repeat for other inputs
```

### 2. Plan the job

```bash
mighty-colab job plan overlap-pursuit-b-20000.yaml
```

Output:
```
Job ID: [UUID]
...
ready to apply: mighty-colab job apply overlap-pursuit-b-20000.yaml
```

### 3. Apply (allocate VM and run)

```bash
mighty-colab job apply overlap-pursuit-b-20000.yaml --timeout 3600 --leave-up
```

The `--leave-up` flag preserves the VM for inspection if needed; omit it for automatic cleanup.

### 4. Monitor progress

```bash
mighty-colab job status <JOB_ID> --poll
```

This polls every 10 seconds and prints:
- Current workload state (running, succeeded, failed, etc.)
- Remote result availability
- Cleanup status

### 5. Retrieve results

Once `status` reports `done: true` and `ok: true`:

```bash
# Results were uploaded to:
# gs://labkit-build-colab-jobs/outputs/pursuit-b-20000.json
# gs://labkit-build-colab-jobs/outputs/pursuit-b-20000.timing.json
# gs://labkit-build-colab-jobs/outputs/ckpt-20000.pt

gcloud storage cp gs://labkit-build-colab-jobs/outputs/pursuit-b-20000.json .
gcloud storage cp gs://labkit-build-colab-jobs/outputs/ckpt-20000.pt outputs/
```

### 6. Clean up

```bash
mighty-colab job destroy <JOB_ID>
```

Verify the VM is released:

```bash
mighty-colab sessions  # should not list the job's endpoint
```

## Job Specifications

### overlap-pursuit-b-20000.yaml

Trains M1 (layer 2 only) for 20,000 Adam steps from random initialization.

**Inputs:**
- `07_posn.bundle`: Git bundle of the reference 07_posn repo, commit 18a60649
- `2shapes_train.npz`: Training set (16,000 images)
- `2shapes_val.npz`: Validation set (50 images)

**Outputs:**
- `pursuit-b-20000.json`: Final metrics and loss curve
- `pursuit-b-20000.timing.json`: Wall-clock timing data
- `ckpt-20000.pt`: Trained K_2 and delta_omega checkpoint

**Accelerator:** A100 GPU (48GB VRAM), ~4 hours wall time

**Budget:** 14,400 seconds (4 hours)

### overlap-pursuit-b-mechanism.yaml

Deterministic CPU audit of the 20,000-step mechanism (once checkpoints are available).

**Inputs:**
- `2shapes_val.npz`
- Checkpoints: ckpt-250.pt, ckpt-500.pt, ckpt-1000.pt, ckpt-2000.pt, ckpt-4000.pt, ckpt-6000.pt, ckpt-8000.pt, ckpt-10000.pt

**Output:**
- `pursuit-b-mechanism.json`: Full mechanism audit with ablations

**Accelerator:** CPU (no GPU required)

**Budget:** 14,400 seconds (4 hours)

## Troubleshooting

### "connection lost" / "401" after ~60 minutes

This is the Google OAuth token refresh boundary. It's not fatal:

```bash
mighty-colab job status <JOB_ID> --poll
```

This refreshes the token and continues. The job on the VM continues running.

### "PUT rejected" / "403" on artifact upload

The signed URL may have expired. Re-sign and re-plan:

```bash
python scripts/gcs_upload_and_sign.py  # generates fresh URLs
# Update the YAML spec with new URLs
mighty-colab job plan updated.yaml
```

### Job shows "running" but no output appears

This is normal. The job has no stdout/stderr contract—stdout goes nowhere. Check the control-result backstop:

```bash
# control-result.get_url is in the spec
curl [get_url from spec]
```

Or wait for `status --poll` to absorb the remote result.

## File Locations

```
~/Code/AI/08_overlap_bench/
  bundle/
    run_repo_script.py           # mighty-colab entry point adapter
  scripts/
    gcs_upload_and_sign.py       # Upload files, sign URLs
    create_posn_bundle.sh        # Create 07_posn.bundle
    run_pursuit_b.py             # Training script
    verification/
      analyze_pursuit_b_mechanism.py  # Mechanism audit
  overlap-pursuit-b-20000.yaml   # Job spec for 20k-step run
```

## References

- `gcloud-dogfood.sh`: Set up per-directory gcloud and verify signing
- `docs/09_job_usage.md`: mighty-colab user guide
- `docs/20_job_spec.md`: Detailed job spec schema
- `run_pursuit_b.py`: See --help for training options
