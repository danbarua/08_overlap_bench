# Pursuit B: mighty-colab job specifications

Generated 2026-09-14. All signed URLs valid for their durations (GET 2–8h, PUT 8h).  
Checkpoints for mechanism audit must be uploaded after 500/1000/2000/4000/6000/8000/10000-step runs complete.

---

## 1. Primary Pursuit B: 20,000 steps from scratch

```yaml
name: overlap-pursuit-b-20000

ignore_warnings: true

accelerator:
  prefer: [A100]
  accept_cpu: false

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
    - --save-model
    - outputs/ckpt-20000.pt

deps:
  - numpy==1.26.4
  - scipy==1.15.2
  - scikit-learn==1.6.1
  - torch==2.2.2
  - torchvision==0.17.2

data:
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/inputs/07_posn.bundle?X-Goog-Signature=...
    dest: inputs/07_posn.bundle
    sha256: ee863262b2dfc86f6d32a6225625557e7abc24b0aecc1ac89a5908e78c9b74c4

  - url: https://storage.googleapis.com/labkit-build-colab-jobs/inputs/2shapes_train.npz?X-Goog-Signature=...
    dest: src/08_overlap_bench/data/cae/2shapes_train.npz
    sha256: 304b3224e453429e7fef1ee96fcff1c3c3cf138ce0804ca31f6dc2d135006730

  - url: https://storage.googleapis.com/labkit-build-colab-jobs/inputs/2shapes_val.npz?X-Goog-Signature=...
    dest: src/08_overlap_bench/data/cae/2shapes_val.npz
    sha256: 9772425914944b31a07b40fdea967aab08d36d2461f202387eea6690c66fbb59

artifacts:
  - path: src/08_overlap_bench/outputs/pursuit-b-20000.json
    url: https://storage.googleapis.com/labkit-build-colab-jobs/outputs/pursuit-b-20000.json?X-Goog-Signature=...
    required: true

  - path: src/08_overlap_bench/outputs/pursuit-b-20000.timing.json
    url: https://storage.googleapis.com/labkit-build-colab-jobs/outputs/pursuit-b-20000.timing.json?X-Goog-Signature=...
    required: true

  - path: src/08_overlap_bench/outputs/ckpt-20000.pt
    url: https://storage.googleapis.com/labkit-build-colab-jobs/outputs/ckpt-20000.pt?X-Goog-Signature=...
    required: true

control:
  result:
    put_url: https://storage.googleapis.com/labkit-build-colab-jobs/control/pursuit-b-20000/result.json?X-Goog-Signature=...
    get_url: https://storage.googleapis.com/labkit-build-colab-jobs/control/pursuit-b-20000/result.json?X-Goog-Signature=...

budgets:
  wall_clock: 14400

on_offload_fail: leave_up
```

**To populate signed URLs:** Replace each `?X-Goog-Signature=...` with the actual signed URL from the signing output above.

---

## 2. Mechanism audit (after checkpoints available)

Requires checkpoints from 500/1000/2000/4000/6000/8000/10000-step runs.

Once those complete, upload the checkpoints and sign URLs:

```bash
# Upload checkpoints after runs complete
gcloud storage cp outputs/ckpt-*.pt gs://labkit-build-colab-jobs/runs/

# Sign GET URLs for each (2-hour)
gcloud storage sign-url gs://labkit-build-colab-jobs/runs/ckpt-250.pt \
  --impersonate-service-account=colab-job-signer@labkit-build.iam.gserviceaccount.com \
  --region=US --http-verb=GET --duration=2h --format='value(signed_url)'
# ... repeat for ckpt-500.pt, ckpt-1000.pt, etc.
```

Then use this spec:

```yaml
name: overlap-pursuit-b-mechanism

ignore_warnings: true

accelerator:
  prefer: []
  accept_cpu: true

code:
  kind: bundle
  root: ./bundle
  entry: run_repo_script.py
  args:
    - scripts/verification/analyze_pursuit_b_mechanism.py
    - --out
    - outputs/pursuit-b-mechanism.json

deps:
  - numpy==1.26.4
  - scipy==1.15.2
  - scikit-learn==1.6.1
  - torch==2.2.2
  - torchvision==0.17.2

data:
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/inputs/07_posn.bundle?X-Goog-Signature=...
    dest: inputs/07_posn.bundle
    sha256: ee863262b2dfc86f6d32a6225625557e7abc24b0aecc1ac89a5908e78c9b74c4

  - url: https://storage.googleapis.com/labkit-build-colab-jobs/inputs/2shapes_val.npz?X-Goog-Signature=...
    dest: src/08_overlap_bench/data/cae/2shapes_val.npz
    sha256: 9772425914944b31a07b40fdea967aab08d36d2461f202387eea6690c66fbb59

  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-250.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-250.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-500.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-500.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-1000.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-1000.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-2000.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-2000.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-4000.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-4000.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-6000.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-6000.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-8000.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-8000.pt
  - url: https://storage.googleapis.com/labkit-build-colab-jobs/runs/ckpt-10000.pt?X-Goog-Signature=...
    dest: src/08_overlap_bench/outputs/ckpt-10000.pt

artifacts:
  - path: src/08_overlap_bench/outputs/pursuit-b-mechanism.json
    url: https://storage.googleapis.com/labkit-build-colab-jobs/outputs/pursuit-b-mechanism.json?X-Goog-Signature=...
    required: true

control:
  result:
    put_url: https://storage.googleapis.com/labkit-build-colab-jobs/control/pursuit-b-mechanism/result.json?X-Goog-Signature=...
    get_url: https://storage.googleapis.com/labkit-build-colab-jobs/control/pursuit-b-mechanism/result.json?X-Goog-Signature=...

budgets:
  wall_clock: 14400
```

---

## Signed URLs (copy into specs above)

Paste these as replacements for `?X-Goog-Signature=...` placeholders in the specs.

[Copy the JSON output from the signing script above into this section]
