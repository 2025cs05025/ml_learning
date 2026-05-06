# Heart Disease Prediction — MLOps End-to-End Pipeline

**Author:** SANDIP BHATTACHARYYA — `2025cs05025`  
**Course:** MLOps (S2-25_AMLCSZG523)

An end-to-end ML pipeline for predicting heart disease risk using the UCI Heart Disease dataset: preprocessing, EDA, training with MLflow, batch inference, FastAPI, Docker, Prometheus/Grafana monitoring, GitHub Actions CI/CD, and optional Kubernetes (`k8s/`).

**Jump to:** [Prerequisites](#prerequisites) · [Architecture](#architecture) · [Project structure](#project-structure) · [Data (`data/`)](#data) · [Quick start](#quick-start-end-to-end) · [Kubernetes](#kubernetes-deployment-minikube) · [Model details](#model-details) · [API](#api-endpoints) · [CI/CD](#cicd-pipeline-github-actions) · [Monitoring](#monitoring-prometheus-and-grafana) · [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python** 3.9+ (CI uses 3.11; tested locally on 3.9–3.12)
- **Docker** + **Docker Compose** — use **`docker compose`** (V2 plugin) or **`docker-compose`** (standalone CLI); this repo’s examples often use `docker-compose`
- Free ports for the monitoring demo: **8000** (API), **9090** (Prometheus), **3000** (Grafana)
- **Git**
- **kubectl** + a cluster (optional — for `k8s/`)
- **minikube** (optional — for local Kubernetes, same idea as course reference repos)
- macOS / Linux (Windows WSL2 should work)

---

## Architecture

High-level data and serving flow:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Data (CSV) │────▶│  EDA + clean │────▶│  Train + MLflow │────▶│  Model (.pkl)│
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                        │
                    ┌──────────────┐      ┌─────────────────┐           │
                    │  Prometheus  │◀──── │ FastAPI + Docker│ ◀─────────┘
                    │  /metrics    │      │  /predict       │
                    └──────────────┘      └────────┬────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │ Kubernetes (opt) │
                                         │ k8s/ manifests   │
                                         └──────────────────┘
```

**Docker Compose** adds **Grafana** (port **3000**); Prometheus scrapes the API using `monitoring/prometheus.yml`.

---

## Project structure

```
ml_learning/                       # repository root (clone folder name may differ)
├── .github/workflows/ci.yml       # CI/CD Pipeline (4 jobs: lint → test → train → docker-build-smoke)
├── data/
│   ├── heart_disease_UCI_dataset.csv      # Bundled raw UCI-style input
│   ├── heart_disease_processed_dataset.csv  # Produced by EDA / preprocess (see Data section)
│   └── batch_predictions.csv      # Optional output from batch inference
├── models/                        # best_model.pkl, feature_names.pkl, training_metadata.json
├── mlruns/                        # MLflow tracking store
├── screenshots/                   # EDA and training plots
├── src/
│   ├── api/api.py                 # FastAPI: /health, /predict, /metrics
│   ├── config/paths.py
│   ├── data_preprocessing/pre_processing_data.py  # load_data, clean_data, save_cleaned_csv
│   ├── eda/eda.py                 # CLI EDA → clean CSV + plots
│   └── model_training/            # train.py, inference.py (batch scoring)
├── monitoring/                    # prometheus.yml, grafana/dashboards/
├── k8s/                           # namespace, deployment, service, HPA, ingress
├── scripts/test_full_flow.sh      # Optional end-to-end: lint, tests, pipeline, Compose
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── pytest.ini
└── README.md
```

**Note:** There is **no** `notebooks/` directory in this repo; EDA is **script-driven** via `src/eda/eda.py`. You can still open a Jupyter kernel and import the same modules if your coursework allows notebooks alongside scripts.

For what each file under `data/` means and how it maps to `RAW_DATA_CSV` / `CLEAN_DATA_CSV`, see **[Data (`data/`)](#data)** below.

---

## Quick start (end-to-end)

All commands assume the **repository root** (directory containing this `README.md`).

### 1. Clone and setup

```bash
git clone <YOUR_REPO_URL>
cd ml_learning

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2. Explore the data (EDA)

**Script / CLI (primary in this repo):**

```bash
python3 src/eda/eda.py all
```

This loads `data/heart_disease_UCI_dataset.csv`, writes `data/heart_disease_processed_dataset.csv`, and saves plots under `screenshots/`.

Individual steps: `load`, `inspect`, `preprocess`, `eda`, `all` — see `python3 src/eda/eda.py --help`.

### 3. Train the model

```bash
python3 src/model_training/train.py
```

This trains **Logistic Regression** and **Random Forest** with **GridSearchCV** (ROC-AUC), logs runs to **MLflow** under `mlruns/`, and saves the best pipeline to `models/best_model.pkl` plus `models/feature_names.pkl` and `models/training_metadata.json`.

### 4. MLflow UI

```bash
python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5050
```

Open **http://127.0.0.1:5050** (use another port if busy).

### 5. Batch inference (local)

```bash
python3 src/model_training/inference.py --output data/batch_predictions.csv
```

Uses the same **raw → preprocess** path as training (`load_data` + `clean_data`), then the saved sklearn **Pipeline**.

### 6. Unit tests

```bash
python3 -m pytest tests/ -v
```

Covers preprocessing, model pipelines, and API **`PatientData`** validation (see `tests/test_pipeline.py`).

**JUnit XML (matches CI artifact):**

```bash
python3 -m pytest tests/ -v --tb=short --junitxml=pytest-results.xml
```

### 7. Start the API locally

```bash
python3 src/api/api.py
# or:
python3 -m uvicorn api.api:app --app-dir src --host 0.0.0.0 --port 8000
```

- **Swagger:** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/health  

### 8. Docker (API image only)

Train first so `models/best_model.pkl` and `models/feature_names.pkl` exist.

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Smoke checks:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'
```

### 9. Monitoring stack (API + Prometheus + Grafana)

From repo root (after `models/` exists — Compose mounts `./models` read-only):

```bash
docker-compose build
docker-compose up -d --build
```

- **API docs:** http://127.0.0.1:8000/docs  
- **Prometheus:** http://127.0.0.1:9090  
- **Grafana:** http://127.0.0.1:3000 — login **admin** / **admin123**  

Send a few `POST /predict` requests, then check targets in Prometheus and the provisioned Grafana dashboard under `monitoring/grafana/dashboards/`.

Stop:

```bash
docker-compose down
```

**Compose services (fixed container names):** `heart-disease-api`, `prometheus`, `grafana`.

### 10. One-shot script (optional)

```bash
bash scripts/test_full_flow.sh
```

Runs lint, tests (writes `pytest-results.xml`), EDA, training, batch inference, then brings up Compose (stack may stay running — stop with `docker-compose down`). **Docker must be running.**

---

## Kubernetes deployment (Minikube)

**Prerequisites:** Docker Desktop (or compatible runtime), **minikube**, **kubectl**.

**1. Start cluster**

```bash
minikube start --driver=docker
```

**2. Build image (repo root)**

```bash
docker build -t heart-disease-api:latest .
```

**3. Load image into Minikube**

```bash
minikube image load heart-disease-api:latest
```

**4. Apply manifests**

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
# Optional: hpa.yaml, ingress.yaml as needed
```

**5. Wait for rollout**

```bash
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=120s
```

**6. Check resources**

```bash
kubectl get pods,svc,deployment -n heart-disease
```

**7. Get service URL**

```bash
minikube service heart-disease-api-service -n heart-disease --url
```

**8. Test**

```bash
curl -fsS http://127.0.0.1:<PORT>/health
curl -fsS -X POST http://127.0.0.1:<PORT>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'
```

**Cleanup (example)**

```bash
kubectl delete -n heart-disease deployment,service --all
kubectl delete namespace heart-disease
minikube stop
```

### Kubernetes manifest summary

- **`k8s/namespace.yaml`** — namespace `heart-disease`
- **`k8s/deployment.yaml`** — Deployment `heart-disease-api`, image `heart-disease-api:latest`, probes on `/health`, 2 replicas by default
- **`k8s/service.yaml`** — LoadBalancer Service `heart-disease-api-service`, port **80** → container **8000**
- **`k8s/hpa.yaml`**, **`k8s/ingress.yaml`** — optional autoscaling / ingress (adjust hostnames and image pull policy for your environment)

---

## Model details

- **Dataset:** UCI Heart Disease (bundled as `data/heart_disease_UCI_dataset.csv`; ~303 rows, 13 clinical features, binary `target`)
- **Preprocessing:** shared in `src/data_preprocessing/pre_processing_data.py`; EDA writes `data/heart_disease_processed_dataset.csv`
- **Models:** **Logistic Regression** (with scaling) and **Random Forest** (with imputation), tuned via **GridSearchCV** with stratified CV, scoring **ROC-AUC**
- **Selection:** best model by **hold-out test ROC-AUC**; saved to `models/best_model.pkl` with `models/feature_names.pkl` for inference
- **Experiment tracking:** **MLflow** under `mlruns/` (experiment name default `heart_disease_prediction`)

Exact metrics for your run are in `models/training_metadata.json` and in the MLflow UI.

---

## API endpoints

**Base URL** (local): `http://127.0.0.1:8000`

- **`GET /docs`** — Swagger UI  
- **`GET /health`** — Liveness; JSON includes model load status  
- **`POST /predict`** — JSON body `PatientData` → prediction, label, confidence, risk level  
- **`GET /metrics`** — Prometheus text exposition  

### `/predict` request body (example)

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 2,
  "ca": 0,
  "thal": 1
}
```

### Sample response shape

```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.7081,
  "risk_level": "HIGH"
}
```

(Field names match `PredictionResponse` in `src/api/api.py`.)

---

## CI/CD pipeline (GitHub Actions)

Workflow: **`.github/workflows/ci.yml`** — in the GitHub UI the workflow name is **CI/CD Pipeline**.

**Triggers:** push and pull request to **`main`** / **`master`**, plus **workflow_dispatch** (manual).

**Jobs (sequential):**

1. **`lint`** — `flake8 src tests`
2. **`test`** — `pytest tests/` (writes `pytest-results.xml`), uploads artifact **`test-results`** (`if: always()` on the upload step)
3. **`train`** — `python src/eda/eda.py preprocess`, `python src/model_training/train.py`, uploads **`ml-training`** (`models/` + `mlruns/`) on success
4. **`docker-build-smoke`** — downloads **`ml-training`**, `docker build -t heart-disease-api:ci .`, smoke **`/health`** and **`POST /predict`**

CI **does not** start Grafana or Prometheus; use **Docker Compose** locally for that.

**Artifacts:** **`test-results`**, **`ml-training`** (14-day retention) — download from **Actions → workflow run → Artifacts**.

**Local parity (core ML steps):**

```bash
python3 -m flake8 src tests
python3 -m pytest tests/ -v
python3 src/eda/eda.py preprocess
python3 src/model_training/train.py
```

---

## Monitoring (Prometheus and Grafana)

### Compose quick start

```bash
docker-compose up -d --build
```

**Services**

- **heart-disease-api** — http://localhost:8000  
- **prometheus** — http://localhost:9090  
- **grafana** — http://localhost:3000 — **admin** / **admin123**  

### Dashboard

Dashboard JSON is provisioned from `monitoring/grafana/dashboards/` (e.g. heart disease API panels tied to scraped metrics).

### Metrics (API)

Prometheus counters/histograms/gauges include (names from `src/api/api.py`):

- **`predict_requests_total`** — `/predict` requests by `status`
- **`predict_request_latency_seconds`** — latency histogram
- **`prediction_label_total`** — counts by predicted label
- **`prediction_risk_level_total`** — counts by risk band
- **`last_prediction`**, **`last_prediction_confidence`** — gauges for latest request
- **`prediction_confidence`** — confidence histogram

### Logging

Structured logging to **stdout** and, when configured, **`API_LOG_PATH`** (under Compose often `./logs/api_requests.log` via `docker-compose.yml`).

### Cleanup

```bash
docker-compose down
```

---

## Troubleshooting

- **Port 8000 in use** — `lsof -nP -iTCP:8000 -sTCP:LISTEN`; stop the conflicting process or use only Compose for the API.
- **`Clean data not found` when training** — Run `python3 src/eda/eda.py all` or `preprocess` first.
- **Grafana “No data”** — Send at least one `POST /predict`; check http://127.0.0.1:9090/targets (job for the API should be **UP**).
- **`docker-compose` vs `docker compose`** — Install the [Compose CLI](https://docs.docker.com/compose/install/) or use the Docker Desktop plugin; subcommands are the same.
- **Docker permission denied (e.g. Colima)** — `colima stop && colima start`; verify `docker info`.
- **`/predict` 500 in container** — Rebuild image; this repo pins **`scikit-learn==1.6.1`** in `requirements.txt` for pickle compatibility.
- **Training parallel errors** — `export MLOPS_N_JOBS=1` before `train.py` in restricted environments.

---

## Data

### Files in `data/`

- **`heart_disease_UCI_dataset.csv`** — Bundled **raw** UCI-style heart disease data. Default input for EDA and for `load_data()` when scoring raw rows in batch inference.
- **`heart_disease_processed_dataset.csv`** — **Generated** by `python src/eda/eda.py preprocess` or `all`. Numeric features + binary `target`; **`train.py` loads this** for modelling. Not required for the HTTP API (the API scores **raw** rows via the same cleaning code in memory).
- **`batch_predictions.csv`** — **Optional.** Created only if you run batch inference **with** `--output`, e.g. `python src/model_training/inference.py --output data/batch_predictions.csv`. It is **not** needed for training, CI, or serving. **Purpose:** save offline predictions (class, probability, and `actual_target` when the input still has labels) for reports, coursework, or archiving—without calling `POST /predict`. Omit `--output` if you only want a terminal preview.

### Path constants (`src/config/paths.py`)

- **`RAW_DATA_CSV`** → `data/heart_disease_UCI_dataset.csv`
- **`CLEAN_DATA_CSV`** → `data/heart_disease_processed_dataset.csv` (same file EDA writes and training reads by default)

### Citation

- [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---


