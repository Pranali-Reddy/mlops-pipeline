# Architecture

## Goals

- **Reproducibility**: every model in production is traceable to a specific
  commit, dataset version, and set of hyperparameters.
- **Safety by default**: no model reaches production without passing an
  automated comparison against the current incumbent.
- **Portability**: no cloud-specific services in the core pipeline — it runs
  the same on a laptop (Docker Compose) as on any Kubernetes cluster (EKS,
  AKS, GKE, or bare-metal).
- **Observability**: every serving replica exposes health and metrics
  endpoints out of the box.

## Pipeline stages

```
 ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
 │   Ingest    │───▶│  Preprocess   │───▶│   Augment +    │───▶│  Model Registry │
 │ (raw data)  │    │ clean/split/  │    │  Train (N      │    │  (Staging ->    │
 │             │    │   scale       │    │  candidates)   │    │   Production)   │
 └─────────────┘    └──────────────┘    └───────────────┘    └────────┬───────┘
                                                                        │
                                                                        ▼
 ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌────────────────┐
 │  Monitoring  │◀───│   Serving     │◀───│   Container    │◀───│  CI/CD deploys │
 │ (Prometheus) │    │  (FastAPI)    │    │   (Docker)     │    │   the image    │
 └─────────────┘    └──────────────┘    └───────────────┘    └────────────────┘
```

### 1. Ingest & preprocess (`src/preprocess.py`)

Loads raw data, deduplicates, drops nulls, splits train/test, and fits a
`StandardScaler`. Both the split dataset and the fitted scaler are persisted
to disk (`data/processed/`) so that training and serving use *exactly* the
same scaling — a common source of training/serving skew if handled ad hoc.

### 2. Augment & train (`src/train.py`)

- **Augmentation**: Gaussian jitter creates `AUGMENT_FACTOR` noisy copies of
  the training set. This is the "augmented" part of the pipeline — it's
  intentionally a small, swappable function so a team can drop in SMOTE,
  mixup, or a generative augmentation model without touching anything else.
- **Multi-candidate training**: Random Forest, Gradient Boosting, and
  Logistic Regression are trained in the same run, each as its own MLflow
  run nested in one experiment. All hyperparameters, metrics (accuracy, F1,
  precision, recall), and the fitted model artifact are logged.
- **Registration & promotion gate**: the best candidate (by weighted F1) is
  registered to the MLflow Model Registry as a new version, moved to
  *Staging*, and compared against whatever is currently in *Production*. It
  is only promoted if it beats the incumbent — otherwise the existing
  Production model is left untouched and the new version stays parked in
  Staging for manual review.

This gate is the core safety mechanism of the pipeline: **retraining never
silently degrades production.**

### 3. Serve (`src/serve.py`)

A FastAPI app that resolves `models:/wine-classifier-prod/Production` from
the registry at startup — it always serves whatever the registry currently
marks as Production, without needing to know a specific run ID or file path.
Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Scale input, run inference, return class + probabilities |
| `GET /health` | Liveness/readiness, used by k8s probes and CI smoke tests |
| `GET /metrics` | Prometheus-format counters/histograms |
| `POST /reload` | Re-resolve the Production model without restarting the pod |

### 4. Containerization (`Dockerfile`)

A single multi-stage Dockerfile with a shared `base` layer (dependencies +
source) and two leaf targets:

- `train` — runs `preprocess.py && train.py` then exits (used as a K8s `Job`)
- `serve` — long-running `uvicorn` process (used as a K8s `Deployment`)

This keeps both images in lockstep on dependency versions and avoids
duplicated `pip install` layers.

### 5. Orchestration (`k8s/`)

| Manifest | Purpose |
|---|---|
| `00-namespace.yaml` | Isolates all resources under `mlops` |
| `01-mlflow.yaml` | Tracking server + registry backend, backed by a PVC |
| `02-training-job.yaml` | One-shot `Job` for a training run, parameterized by image tag |
| `03-serving-deployment.yaml` | `Deployment` (2-8 replicas via HPA), `Service`, pod anti-affinity |
| `04-ingress.yaml` | External routing to `/api` (serving) and `/mlflow` (tracking UI) |
| `05-retrain-cronjob.yaml` | Weekly scheduled retraining; swap the schedule for drift-triggered logic |

### 6. CI/CD (`.github/workflows/ci-cd.yml`)

`test → build-and-push → train-and-register → deploy`, each gated on the
previous stage succeeding. Training runs *on the cluster*, not in the CI
runner, so the promotion decision always happens against the same
Production registry the serving layer reads from — no drift between a
CI-local evaluation and what actually gets served.

### 7. Observability (`monitoring/prometheus.yml`)

Scrapes the serving API's `/metrics` endpoint. Recommended dashboard panels:
request rate by status, p50/p95/p99 latency, and pod-level CPU/memory (from
the HPA metrics already being collected). Model-quality metrics (e.g. label
drift, prediction distribution shift) are a natural next addition — logged
as custom Prometheus metrics or as a scheduled batch job that compares
recent prediction logs against training-time distributions.

## Key design decisions & trade-offs

- **MLflow over a fully custom registry**: gives experiment tracking, model
  registry, and a UI for free, at the cost of running one more service. The
  same ideas (staging/production gating) would apply with any registry.
- **Gaussian-jitter augmentation over SMOTE**: chosen for domain-generality
  in this template — SMOTE assumes class imbalance is the issue, which isn't
  always true. Swap freely; the interface (`augment(X, y) -> X, y`) is the
  same either way.
- **Weighted F1 as the sole promotion metric**: simple and interpretable for
  a template, but production systems usually gate on a *basket* of metrics
  (fairness slices, latency budget, calibration) — extend
  `run_training()`'s comparison logic accordingly.
- **Scheduled retraining over drift-triggered**: simpler to reason about and
  demo. A drift-triggered `CronJob`-equivalent would need a monitoring
  component computing a drift statistic and calling the K8s API to create a
  `Job` on demand — noted as a documented extension point rather than built
  in, to keep the template focused.
