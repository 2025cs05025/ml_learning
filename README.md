# Heart Disease Prediction — MLOps Pipeline

**Author:** SANDIP BHATTACHARYYA — `2025cs05025` · **Course:** MLOps (S2-25_AMLCSZG523)

End-to-end ML on the UCI Heart Disease dataset: preprocessing, EDA, training with MLflow, FastAPI, Docker, Prometheus/Grafana (`docker-compose.yml`), optional Kubernetes (`k8s/`).

---

## Prerequisites

- **Python** 3.9+ · **Git** · **Docker** + **Compose** (`docker-compose` or `docker compose`) for the monitoring stack  
- Free ports for the demo: **8000** (API), **9090** (Prometheus), **3000** (Grafana)  
- **kubectl** only if you use `k8s/`

---

## Flow (run in order)

```mermaid
flowchart LR
  RAW[data/ raw CSV] --> EDA[eda.py → heart_clean.csv]
  EDA --> TRAIN[train.py → models/ + mlruns/]
  TRAIN --> API[FastAPI /predict]
  API --> METRICS[/metrics → Prometheus → Grafana]
```

1. **EDA** — `src/eda/eda.py` writes `data/heart_clean.csv` and plots under `screenshots/`.  
2. **Train** — `src/model_training/train.py` saves `models/best_model.pkl`, `feature_names.pkl`, MLflow under `mlruns/`.  
3. **Serve** — `src/api/api.py` or the **Dockerfile** image; Compose adds Prometheus + Grafana.

---

## Project structure

```
ml_learning/
├── .github/workflows/ci.yml   # lint → pytest → preprocess → train → Docker smoke
├── data/                      # raw CSV, heart_clean.csv, batch_predictions.csv
├── models/                    # pickles + training_metadata.json
├── mlruns/                    # MLflow store
├── src/
│   ├── config/paths.py
│   ├── data_preprocessing/
│   ├── eda/eda.py
│   ├── model_training/        # train.py, predict.py
│   └── api/api.py             # /health, /predict, /metrics
├── monitoring/                # prometheus.yml, grafana/dashboards/
├── scripts/test_full_flow.sh  # optional: full local pipeline + Compose
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

**Where things live:** Application code under `src/`. Trained weights and pickles under `models/` (not `src/model_training/`). Paths are centralized in `src/config/paths.py`.

---

## Setup (once per machine)

From the **repo root** (folder containing this `README.md`):

```bash
cd /path/to/ml_learning

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

**Windows:** if `python3` is missing, use `py -m venv .venv` and `py -m pip install -r requirements.txt`.

---

## Main pipeline

```bash
source .venv/bin/activate

python3 src/eda/eda.py all
python3 src/model_training/train.py
python3 src/model_training/predict.py --output data/batch_predictions.csv
```

---

## Docker

Docker packages the **FastAPI** app (`src/api/api.py` via `uvicorn`). Config files: **`Dockerfile`** (API image only) and **`docker-compose.yml`** (API + Prometheus + Grafana).

### Before building any image

Train once so **`models/best_model.pkl`** and **`models/feature_names.pkl`** exist (see [Main pipeline](#main-pipeline)). The Compose file mounts `./models` read-only into the API container.

### Sanity check

```bash
docker --version
docker compose version || docker-compose --version
docker info
```

These docs use **`docker-compose`** (hyphen, standalone CLI). If you use Compose V2 as a plugin, replace with **`docker compose`** (space) — same subcommands.

### Run the API container only

From repo root:

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

- **Swagger:** http://127.0.0.1:8000/docs  
- **Health:** `curl -fsS http://127.0.0.1:8000/health`  
- Image runs `uvicorn` with `PYTHONPATH` set so `api.api:app` resolves (`Dockerfile`).

### Run API + Prometheus + Grafana (Compose)

`docker-compose.yml` defines three services:

- **heart-disease-api** (port **8000**) — Built from `Dockerfile`; exposes `/predict`, `/health`, and **`/metrics`** for Prometheus.
- **prometheus** (port **9090**) — Scrapes the API using `monitoring/prometheus.yml`.
- **grafana** (port **3000**) — Dashboards under `monitoring/grafana/`; login **admin** / **admin123**.

Start (foreground or detached):

```bash
docker-compose up --build

# detached:
docker-compose up -d --build
```

Check containers: `docker-compose ps`. Stop and remove containers:

```bash
docker-compose down
```

Compose sets fixed **`container_name`** values (`heart-disease-api`, `prometheus`, `grafana`). If you see *“container name already in use”*, remove the old container or run `docker-compose down` in this repo first:

```bash
docker rm -f heart-disease-api   # only if a stray container blocks the name
```

Optional API logs on disk: `./logs/` is mounted for request logging when using Compose.

### CI (GitHub Actions)

The workflow builds the image (`docker build …`) and runs a short smoke test (`GET /health`, `POST /predict`) — see `.github/workflows/ci.yml` job **`docker-build-smoke`**.

---

## What to run next (pick one)

- **MLflow UI** — `python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5050` → open http://127.0.0.1:5050  
- **API without Docker** — `python3 src/api/api.py` → http://127.0.0.1:8000/docs  
- **API Docker image only** — After `models/` exists: `docker build -t heart-disease-api:latest .` then `docker run --rm -p 8000:8000 heart-disease-api:latest`  
- **API + Prometheus + Grafana** — `docker-compose up --build` — Grafana http://127.0.0.1:3000 (`admin` / `admin123`)  
- **Everything in one script** (bash; Docker required for last step) — `bash scripts/test_full_flow.sh` — leaves Compose running on success; stop with `docker-compose down`  
- **Kubernetes** — Build/load image, then `kubectl apply -f k8s/`

**Compose tip:** If `docker-compose` is missing, install the [Compose CLI](https://docs.docker.com/compose/install/) or use the Docker Desktop plugin form (`docker compose`). Same subcommands.

---

## Commands reference

Run from repo root with venv activated unless noted.

- **Lint** — `python3 -m flake8 src tests`  
- **Tests** — `python3 -m pytest tests/ -v`  
- **JUnit XML** (matches CI artifact) — `python3 -m pytest tests/ -v --tb=short --junitxml=pytest-results.xml`  
- **CI order locally** — `flake8` → `pytest` → `python3 src/eda/eda.py preprocess` → `python3 src/model_training/train.py`  
- **EDA subcommands** — `python3 src/eda/eda.py --help`  
- **Coverage** (example) — `python3 -m pytest tests/test_pipeline.py -v --cov=data_preprocessing --cov=model_training --cov=api --cov-report=term-missing`

**GitHub Actions:** Workflow `.github/workflows/ci.yml` runs on **push/PR to `main` or `master`** and **manual dispatch**. Job 1: lint, tests, preprocess, train, upload artifacts. Job 2 (after job 1): Docker build + `/health` and `/predict` smoke test. Expect **~15–35 minutes**; job 2 stays “Queued” until job 1 finishes.

---

## Data

Bundled: `data/heart_disease_uci.csv`. Citation: [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).

---

## Troubleshooting (short)

- **Port 8000 in use** — `lsof -nP -iTCP:8000 -sTCP:LISTEN` — stop the other process or use Compose only.  
- **`Clean data not found`** — Run `python3 src/eda/eda.py all` (or `preprocess`) before training.  
- **Grafana “No data”** — Send at least one `POST /predict`; check Prometheus targets http://127.0.0.1:9090/targets.  
- **Docker permission denied** (e.g. Colima) — `colima stop && colima start` then `docker info`.  
- **`/predict` 500 in Docker** — Rebuild; models expect `scikit-learn==1.6.1` per `requirements.txt`.  
- **`docker-compose` not found** — Install Compose or use `docker compose` (V2 plugin) per Docker docs.  
- **Parallel training errors in sandbox** — `export MLOPS_N_JOBS=1` before `train.py`.

More detail: see comments in `docker-compose.yml`, `monitoring/prometheus.yml`, and `.github/workflows/ci.yml`.

---

## Licence / academic use

Coursework submission; dataset subject to UCI terms of use.
