<<<<<<< HEAD
# MLOps Pipeline for Augmented Model Deployment

An end-to-end, cloud-agnostic MLOps pipeline: data preprocessing, augmentation,
experiment tracking, model registry with automated promotion gating,
containerized serving, CI/CD, and Kubernetes deployment with autoscaling
and monitoring.

Stack: **MLflow · scikit-learn · FastAPI · Docker · Kubernetes · GitHub Actions · Prometheus**

## Why "augmented"

Two things are "augmented" here, deliberately:

1. **Data augmentation** — the training stage generates jittered synthetic
   copies of the training set (`src/train.py::augment`) so the model sees a
   wider distribution before it ever reaches production. Swap this function
   out for SMOTE, mixup, or synthetic record generation as your domain needs.
2. **Augmented decision-making in deployment** — a new model is never blindly
   shipped. It's registered to *Staging*, evaluated against whatever is
   currently in *Production*, and only promoted if it wins. A human is never
   required to approve routine promotions, but the gate is auditable in MLflow.

## Project layout

```
mlops-pipeline/
├── src/
│   ├── preprocess.py     # Load, clean, split, scale data
│   ├── train.py          # Augment, train candidates, track, register, promote
│   └── serve.py           # FastAPI serving layer (loads Production model)
├── tests/
│   └── test_pipeline.py   # Unit tests run in CI before any image is built
├── k8s/                    # Kubernetes manifests (namespace, MLflow, jobs, serving, ingress, cron)
├── monitoring/
│   └── prometheus.yml      # Scrape config for the serving API
├── .github/workflows/
│   └── ci-cd.yml           # test -> build -> train -> deploy pipeline
├── Dockerfile               # Multi-stage: shared base -> train image -> serve image
├── docker-compose.yml        # One-command local stack (MLflow + train + serve + Prometheus)
├── Makefile                   # make install / test / train / serve / up / down
└── requirements.txt
```

## Quickstart (local)

```bash
make install                # pip install -r requirements.txt

# Option A: run stages individually against a local MLflow file store
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
make preprocess
make train
make serve                  # API on http://localhost:8000

# Option B: full containerized stack in one command
make up                     # MLflow UI :5000, API :8000, Prometheus :9090
```

Try a prediction once the API is up:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}'
```

## Deploying to Kubernetes

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-mlflow.yaml

# Substitute your registry/tag, then run training as a Job
export IMAGE_REGISTRY=ghcr.io/your-org
export IMAGE_TAG=v1
envsubst < k8s/02-training-job.yaml | kubectl apply -f -
kubectl wait --for=condition=complete job/model-training -n mlops --timeout=600s

# Roll out the serving deployment (2 replicas, HPA 2-8, anti-affinity)
envsubst < k8s/03-serving-deployment.yaml | kubectl apply -f -

kubectl apply -f k8s/04-ingress.yaml
kubectl apply -f k8s/05-retrain-cronjob.yaml   # weekly scheduled retraining
```

## CI/CD (`.github/workflows/ci-cd.yml`)

Every push to `main` runs, in order:

1. **test** — lint (flake8) + unit tests (pytest), always runs, blocks everything else
2. **build-and-push** — builds the `train` and `serve` Docker targets, pushes to GHCR
3. **train-and-register** — runs the training Job on the cluster; training itself
   handles registry promotion gating (see above)
4. **deploy** — rolls out the new serving image, waits for rollout to complete,
   then runs a smoke test against `/health`

Required repo secrets: `KUBECONFIG_DATA` (base64-encoded kubeconfig).
`GITHUB_TOKEN` is provided automatically for GHCR auth.

## Observability

- `/health` — liveness/readiness (used by k8s probes and CI smoke test)
- `/metrics` — Prometheus format: request counts by status, latency histogram
- `/reload` — hot-swap to the latest Production model version without a pod restart
- `monitoring/prometheus.yml` scrapes the serving API; wire it into Grafana or
  your existing observability stack for dashboards/alerting

## Extending this project

| Want to... | Change this |
|---|---|
| Use real data instead of the demo wine dataset | `src/preprocess.py::load_raw_data` |
| Change the augmentation strategy | `src/train.py::augment` |
| Add/remove candidate models | `src/train.py::get_candidate_models` |
| Change promotion criteria (e.g. require min. accuracy, not just "beats incumbent") | `src/train.py::run_training` |
| Swap MLflow's backend store for Postgres | `docker-compose.yml` / `k8s/01-mlflow.yaml` |
| Add drift-triggered (not just scheduled) retraining | pair a monitoring rule with the K8s API to create a Job from `k8s/02-training-job.yaml` on demand |

See `ARCHITECTURE.md` for a deeper dive into pipeline design and decisions,
and `slides/` for a presentation-ready overview.
=======
# mlops-pipeline
MLOps Pipeline for Augmented Machine Learning Model Deployment
>>>>>>> 00b113314d3f0e25c42a6445e61aef694432dd18
