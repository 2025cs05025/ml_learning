# Heart Disease Prediction — MLOps End-to-End Pipeline

**Author:** SANDIP BHATTACHARYYA — `2025cs05025`  
**Course:** MLOps (S2-25_AMLCSZG523)

An end-to-end ML pipeline for predicting heart disease risk using the UCI Heart Disease dataset: preprocessing, EDA, training with MLflow, FastAPI serving, Docker, Prometheus/Grafana monitoring, and optional Kubernetes manifests (`k8s/`).

**Jump to:** [Architecture](#architecture) · [Project structure](#project-structure) · [Quick Start](#quick-start-end-to-end) · [Troubleshooting](#troubleshooting) · [All commands](#all-commands-copy-paste-reference) · [Rubric mapping](#rubric-mapping-course-mlops-s2-25_amlcszg523)

---

## Prerequisites

- **Python** 3.9+ (tested with 3.9–3.12)
- **Docker** + **Docker Compose** (for multi-container monitoring). You need at least one of: **`docker compose`** (Compose V2 plugin, included with Docker Desktop) **or** the standalone **`docker-compose`** CLI (e.g. `brew install docker-compose` on macOS if `docker compose` is missing). The README and status checks use `docker-compose …` in places so both paths work.
- Free ports when using Compose: **8000** (API), **9090** (Prometheus), **3000** (Grafana)
- **Git**
- Internet access the first time you `python3 -m pip install -r requirements.txt`
- **kubectl** + a cluster (optional — only if you apply `k8s/` manifests)
- macOS / Linux (Windows WSL2 should work; adjust paths such as `~/repos/…` to match where you cloned this repo)

---

## Architecture

High-level flow (**Architecture** → **Project structure** → **Quick Start**, plus ASCII overview):

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
                                         ┌──────---────────┐
                                         │ Kubernetes (opt)│
                                         │ k8s/ manifests  │
                                         └─────────────────┘
```

Docker Compose adds **Grafana** (port **3000**) with dashboards under `monitoring/grafana/`; it reads metrics Prometheus scraped from the API.

### Detailed flow (Mermaid)

File-level path through this repository:

```mermaid
flowchart TD
  A[data/heart_disease_uci.csv (raw)] --> B[EDA: src/eda/eda.py all]
  B --> C[data/heart_clean.csv + screenshots/ plots]
  C --> D[Training: src/model_training/train.py]
  D --> E[models/ best_model.pkl + feature_names.pkl + training_metadata.json]
  D --> F[mlruns/ (MLflow params, metrics, artifacts)]
  E --> G[API: src/api/api.py (/predict, /health, /metrics)]
  G --> H[Prometheus scrape /metrics\nmonitoring/prometheus.yml]
  H --> I[Grafana dashboard\nmonitoring/grafana/dashboards/heart_disease_api.json]
  G --> J[logs/api_requests.log (optional file logging)]
```

Tip: `bash scripts/test_full_flow.sh` runs this path end-to-end (including Docker Compose monitoring).

---

## Project structure

```
Mlops-assignment/
├── .github/workflows/ci.yml       # CI: lint, pytest, train, artifact upload, Docker smoke
├── data/
│   ├── heart_disease_uci.csv      # Bundled raw / UCI-style input
│   ├── heart_clean.csv            # Produced by EDA
│   └── batch_predictions.csv      # Optional output from batch predict
├── models/                        # best_model.pkl, feature_names.pkl, training_metadata.json
├── mlruns/                        # MLflow tracking store
├── screenshots/                   # EDA and training plots
├── src/
│   ├── api/api.py                 # FastAPI + Prometheus /metrics
│   ├── config/paths.py
│   ├── data_preprocessing/        # Cleaning aligned with training
│   ├── eda/eda.py                 # CLI EDA → clean CSV + plots
│   └── model_training/            # train.py, predict.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/        # e.g. heart_disease_api.json
├── k8s/                           # Namespace, Deployment, Service, optional HPA/Ingress
├── scripts/test_full_flow.sh      # One-shot lint/tests/pipeline + Compose checks
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start (End-to-End)

All commands below assume you are in the **repository root** (the folder that contains `README.md`).

### 1. Install Docker + Docker Compose (only for monitoring demo)

- **macOS**: install Docker Desktop (includes Compose plugin). After install, open Docker Desktop once.
- **Linux**: install Docker Engine + Compose plugin.

Sanity check:

```bash
docker --version
docker compose version || docker-compose --version
docker info
```

**Two Compose styles:** **`docker compose`** (space) needs the **Compose V2 plugin** bundled with Docker Desktop. If `docker compose up --build` fails with **`unknown flag: --build`** or **`compose` is not a docker command**, use the standalone binary **`docker-compose`** (hyphen), e.g. `docker-compose up --build`. Install it if needed: `brew install docker-compose` (macOS) or see [Compose install](https://docs.docker.com/compose/install/).

### 2. Create venv + install deps

```bash
cd ~/repos/Mlops-assignment

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. Run the ML pipeline (EDA → train → batch predict)

```bash
source .venv/bin/activate

python3 src/eda/eda.py all
python3 src/model_training/train.py
python3 src/model_training/predict.py --output data/batch_predictions.csv
```

### 4. (Optional) Build and run the API Docker image only

Train first so `models/best_model.pkl` and `models/feature_names.pkl` exist.

Notes:
- `docker build ...` is a **build step** — run it once, and re-run only when you change code/dependencies/Dockerfile.
- `docker run ...` is a **run step** — run it each time you want to start the API container.
- If you use the full stack in **step 5** (`docker-compose up ...`), you can **skip** this section (Compose builds + runs the API).

```bash
cd ~/repos/Mlops-assignment

docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

### 5. Run the monitoring demo (API + Prometheus + Grafana)

1) Make sure nothing else is using the demo ports (especially **8000**):

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN || true
```

2) Build the Docker images (if you’ve never built them, or after code changes):

```bash
docker-compose build
```

3) Start the stack:

```bash
docker-compose down -v --remove-orphans
docker-compose up -d --build --force-recreate
docker-compose ps
sample:
  
NAME                IMAGE                                COMMAND                  SERVICE             CREATED         STATUS                   PORTS
grafana             grafana/grafana:10.1.5               "/run.sh"                grafana             9 seconds ago   Up 8 seconds             0.0.0.0:3000->3000/tcp, [::]:3000->3000/tcp
heart-disease-api   mlops-assignment-heart-disease-api   "uvicorn api.api:app…"   heart-disease-api   9 seconds ago   Up 8 seconds (healthy)   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp
prometheus          prom/prometheus:v2.47.2              "/bin/prometheus --c…"   prometheus          9 seconds ago   Up 8 seconds             0.0.0.0:9090->9090/tcp, [::]:9090->9090/tcp

```

4) Generate traffic (creates metrics) and verify Prometheus:

```bash
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}' >/dev/null

sleep 6
curl -sS "http://127.0.0.1:9090/api/v1/query" --data-urlencode 'query=predict_requests_total{status="success"}'
```

5) Open UIs:

- API docs: `http://127.0.0.1:8000/docs`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (login `admin` / `admin123`)

#### Status checks (copy/paste)

- **Containers running** — Run `docker-compose ps`. Expect `heart-disease-api`, `prometheus`, and `grafana` to be `Up`.
- **API health** — `curl -fsS http://127.0.0.1:8000/health`. Expect JSON with `"status":"ok"`.
- **Prometheus ready** — `curl -fsS http://127.0.0.1:9090/-/ready`. Expect `Prometheus is Ready.`
- **Grafana health** — `curl -fsS http://127.0.0.1:3000/api/health`. Expect JSON with `"database":"ok"`.
- **Prometheus scrape target** — Open `http://127.0.0.1:9090/targets`. Expect job `heart-disease-api` to be **UP**.
- **Key metric** — `curl -sS "http://127.0.0.1:9090/api/v1/query" --data-urlencode 'query=predict_requests_total{status="success"}'`. Expect a non-empty `result`.
- **Grafana dashboard** — Open `http://127.0.0.1:3000`. After a few `POST /predict` calls, panels should show data.

6) Stop the stack when done:

```bash
docker-compose down
```

---

**Problems after Quick Start?** Use the standalone **[Troubleshooting](#troubleshooting)** section at the end of this file (monitoring, Docker, Python, `test_full_flow.sh`).

## All commands (copy-paste reference)

Examples use **`python3`** (macOS / Linux / Windows when available). On **Windows**, if `python3` is not on your PATH, use the **`py`** launcher (e.g. `py -m pip install -r requirements.txt`) and activate the venv with `.venv\Scripts\activate.bat` or `.\.venv\Scripts\Activate.ps1`.

Every command assumes you opened a terminal **in the project folder** (the directory that contains `README.md`) — see [Part A](#part-a--open-the-project-folder-in-the-terminal).

### 1. Environment setup

```bash
cd ~/repos/Mlops-assignment

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2. Main pipeline (run in this order)

```bash
python3 src/eda/eda.py all
python3 src/model_training/train.py
python3 src/model_training/predict.py --output data/batch_predictions.csv
```

### 3. MLflow UI (use a second terminal; keep project root + venv)

```bash
cd ~/repos/Mlops-assignment
source .venv/bin/activate

python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5050
```

Open in browser: `http://127.0.0.1:5050` — Stop with **Ctrl+C** in that terminal.

If port **5050** is busy:

```bash
python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5051
```

Then open `http://127.0.0.1:5051`.

### 4. FastAPI inference server

```bash
source .venv/bin/activate
python3 src/api/api.py
```

Defaults: `PORT` env (default **8000**). Docs: `http://127.0.0.1:8000/docs`.

Alternative (same app):

```bash
python3 -m uvicorn api.api:app --app-dir src --host 0.0.0.0 --port 8000
```

### 5. Lint & unit tests

```bash
source .venv/bin/activate

python3 -m flake8 src tests

python3 -m pytest tests/ -v
python3 -m pytest tests/test_pipeline.py -v

python3 -m pytest tests/test_pipeline.py -v \
  --cov=data_preprocessing --cov=model_training --cov=api \
  --cov-report=term-missing
```

### 6. EDA script — individual steps (optional)

```bash
python3 src/eda/eda.py --help

python3 src/eda/eda.py load
python3 src/eda/eda.py inspect
python3 src/eda/eda.py preprocess
python3 src/eda/eda.py eda
python3 src/eda/eda.py all
```

### 7. Training & prediction — extra flags

```bash
python3 src/model_training/train.py --help
python3 src/model_training/train.py --experiment my_experiment

python3 src/model_training/predict.py --help
python3 src/model_training/predict.py --output data/batch_predictions.csv
python3 src/model_training/predict.py --raw data/heart_disease_uci.csv --model models/best_model.pkl --feature-names models/feature_names.pkl
```

### 9. Same order as GitHub Actions CI (lint → tests → preprocess → train)

```bash
source .venv/bin/activate

python3 -m flake8 src tests
python3 -m pytest tests/ -v
python3 src/eda/eda.py preprocess
python3 src/model_training/train.py
```

### 10. Docker image (API only)

Train first so `models/best_model.pkl` and `models/feature_names.pkl` exist. From repo root:

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Open `http://127.0.0.1:8000/docs`. The image sets `PYTHONPATH=/app/src` and runs `uvicorn api.api:app`.

### 11. Full stack: API + Prometheus + Grafana

Requires the same trained `models/` directory (mounted read-only).

```bash
# Compose V2 plugin (Docker Desktop):
docker compose up --build

# If you get "unknown flag: --build", use the standalone CLI instead:
docker-compose up --build
```

- **API (Swagger):** http://localhost:8000/docs
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 — login `admin` / `admin123`

API request logs (when using Compose) go to `./logs/` via `API_LOG_PATH`.

### 12. Kubernetes (optional)

Manifests live in `k8s/`. Build and load the image into your cluster (or push to a registry and set `image:`), then:

```bash
kubectl apply -f k8s/
```

Replace the hostname in `k8s/ingress.yaml` if you use Ingress. On Minikube / Docker Desktop Kubernetes, use `kubectl get svc -n heart-disease` for the LoadBalancer URL.

---

## Docker Compose, Prometheus, Grafana, and Kubernetes

This repo ships **container** and **cluster** configs for assignment criteria on deployment and monitoring:

- **`Dockerfile`** — FastAPI service only; copies `src/` and `models/`.
- **`docker-compose.yml`** — wires the API to **Prometheus** (`monitoring/prometheus.yml` scrapes `/metrics`) and **Grafana** (dashboard JSON under `monitoring/grafana/dashboards/`).
- **`k8s/`** — Namespace, Deployment, Service (LoadBalancer), HPA, Ingress templates.

Train locally before building or composing images so `models/` is non-empty.

---

## Understanding this repository (for anyone reading the code)

**Story in one sentence.** Data files live under `data/`; **application code** lives under **`src/`** (config, preprocessing, EDA, training, API); **trained weights** (pickles) live in `models/` — do not confuse `models/` with `src/model_training/`.

**Paths (plain language):**

- **`src/config/`** — Defines important paths once (`PROJECT_ROOT`, raw/clean CSV paths, `screenshots/`, `models/`, `mlruns/`). Change paths here instead of hunting through scripts.
- **`src/data_preprocessing/`** — Turns the raw UCI-style CSV into a numeric table: maps categories, fills missing values, binarises the disease label. Used by EDA and prediction so scoring matches training.
- **`src/eda/`** — CLI to inspect data quality, draw charts into `screenshots/`, and write `data/heart_clean.csv`. Run this before training.
- **`src/model_training/`** — **Scripts only:** `train.py` fits models and writes MLflow logs; `predict.py` loads `models/best_model.pkl` and scores rows.
- **`models/`** — **Outputs only:** `best_model.pkl` (fitted sklearn pipeline), `feature_names.pkl` (column order), `training_metadata.json` (metrics and versions). Created when you train.
- **`mlruns/`** — MLflow experiment store (params, metrics, artifacts). Created when you train. View with `python3 -m mlflow ui`.
- **`screenshots/`** — PNG plots from EDA and training (confusion matrix, ROC per model).
- **`scripts/`** — Helper shell scripts (e.g. full pipeline in one command).

**Suggested reading order if you are new to the codebase:**  
`src/config/paths.py` → `src/data_preprocessing/pre_processing_data.py` → `src/eda/eda.py` → `src/model_training/train.py` → `src/model_training/predict.py`.

Tests use `pytest.ini` (`pythonpath = src`) so imports like `from config.paths import …` work without manual `PYTHONPATH`.

**Glossary.** *Hold-out test set* = data hidden during tuning, used only for final metrics. *Stratified split* = train/test keep roughly the same proportion of sick vs healthy rows. *Pipeline* = sklearn object that chains preprocessing steps with the classifier.

---

## Data acquisition (assignment criterion 1)

- **Bundled copy:** The project includes `data/heart_disease_uci.csv` (UCI-style heart disease data) so you can run the pipeline without downloading anything.
- **Provenance / citation:** [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — use this citation in your report.
- **Optional fresh download:** From the [UCI page](https://archive.ics.uci.edu/ml/datasets/Heart+Disease), download the Cleveland or merged files and replace `data/heart_disease_uci.csv` **only if** the layout matches what `src/data_preprocessing/pre_processing_data.py` expects (comma-separated header row; space-separated data rows). When in doubt, keep the bundled file.

---

## Automated tests — `test_pipeline` (assignment criterion 5)

Run **from the repository root** (the folder that contains `README.md` and `pytest.ini`). The file `pytest.ini` adds `src/` to Python’s path so imports resolve.

```bash
# Activate your venv first (Part B), then:
python3 -m pip install -r requirements.txt

# All tests under tests/
python3 -m pytest tests/ -v

# Only the pipeline test module
python3 -m pytest tests/test_pipeline.py -v

# Same tests with coverage (packages live under src/)
python3 -m pytest tests/test_pipeline.py -v \
  --cov=data_preprocessing --cov=model_training --cov=api \
  --cov-report=term-missing
```

**What is tested:** loading raw data, cleaning (no missing/`?`, binary target, numeric features), train/test split shapes, **Logistic Regression** and **Random Forest** pipelines (`fit` / `predict` / `predict_proba`), API **`PatientData`** validation.

---

## CI/CD — GitHub Actions (assignment criterion 5)

Workflow file: `.github/workflows/ci.yml`.

Runs automatically on each **push** or **pull request** to `main`/`master`.

You can also run it **manually** in GitHub:

1. Open **Actions**
2. Select workflow **CI**
3. Click **Run workflow** (this uses `workflow_dispatch` in `.github/workflows/ci.yml`)

**Job `lint-test-train`**

1. **flake8** on `src/` and `tests/`
2. **pytest** (`tests/`)
3. **`python3 src/eda/eda.py preprocess`** — builds `data/heart_clean.csv`
4. **`python3 src/model_training/train.py`** — trains both models and writes **MLflow** under `mlruns/`
5. **Artifact upload** — `models/` and `mlruns/` as `ml-training` (14-day retention) when the job succeeds.

**Job `docker-build-smoke`** (runs after `lint-test-train` succeeds)

1. Downloads the `ml-training` artifact.
2. **`docker build -t heart-disease-api:ci .`**
3. Starts the container, hits **`GET /health`** and **`POST /predict`** with sample JSON, then removes the container.

Push this repo to GitHub to enable the workflow. Local one-shot parity (without Actions): `bash scripts/test_full_flow.sh`.

---

## What you need on your computer

1. **Python 3.9 or newer** (3.10 / 3.11 / 3.12 is fine).
   - Check: open a terminal and run `python3 --version` (on Windows you can try `py --version`).
   - If Python is missing, install it from [python.org](https://www.python.org/downloads/) and try again.

2. **Internet** (only the first time, to download Python packages).

3. **This project folder** on your machine (the folder that contains this `README.md`, `src/` with the Python packages, `models/`, `data/`, etc.).  
   In the examples below we call it **`Mlops-assignment`**.  
   Replace the path with **your** real location.

---

## Part A — Open the project folder in the terminal

### macOS or Linux

1. Open **Terminal**.
2. Go to the project folder (change the path to match where you saved the project):

```bash
cd ~/repos/Mlops-assignment
```

If you are not sure of the path, drag the **`Mlops-assignment`** folder from Finder into the Terminal window after typing `cd ` (with a space). Press Enter.

### Windows

1. Open **Command Prompt** or **PowerShell**.
2. Go to the project folder (change the path to match your PC):

```cmd
cd C:\Users\YourName\repos\Mlops-assignment
```

**Important:** All commands in the next parts must be run **from this project folder** (the same place as `README.md`).

---

## Part B — Create a virtual environment (recommended)

A virtual environment keeps this project’s libraries separate from the rest of your computer.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After this, your prompt may start with `(.venv)`.

### Windows (Command Prompt)

```cmd
python3 -m venv .venv
.venv\Scripts\activate.bat
```

### Windows (PowerShell)

```powershell
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `python3` is not on your PATH (common on some Windows setups), use `py -m venv .venv` instead.

---

## Part C — Install all required packages

Still inside the project folder, with the virtual environment **activated** (see Part B):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Wait until it finishes without errors.

**If `pip` fails:** use `python3 -m pip install -r requirements.txt`.

---

## Part D — Run the full workflow (copy–paste in order)

Run **each block** below **one after the other**. Stay in the project folder; keep the virtual environment activated.

### Step 1 — Data cleaning and EDA (charts)

```bash
python3 src/eda/eda.py all
```

**What you should see:** Text in the terminal about loading data, cleaning, and saving plots.  
**Files created:** `data/heart_clean.csv`, images under `screenshots/`.

---

### Step 2 — Train models and write MLflow logs

```bash
python3 src/model_training/train.py
```

**What you should see:** Training progress, metrics, and messages about saving the best model.  
**Files created or updated:** `models/best_model.pkl`, `models/feature_names.pkl`, `models/training_metadata.json`, and a folder `mlruns/` with experiment data.

---

### Step 3 — (Optional) Batch predictions

```bash
python3 src/model_training/predict.py --output data/batch_predictions.csv
```

**What you should see:** A short table of predictions in the terminal, and a file `data/batch_predictions.csv`.

---

## Part E — Open the MLflow web UI

The MLflow UI is a **web page** on your computer. The server runs in the terminal; **you must open the browser yourself**.

1. Open a **new** terminal window (keep it simple: use another tab or window).
2. Go to the **same** project folder and activate the **same** virtual environment (repeat Part A + Part B `activate` only if needed).
3. Start the server:

**macOS / Linux / Windows (same command if you are inside the project folder):**

```bash
python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5050
```

4. Open a web browser (Chrome, Firefox, Safari, Edge) and type this in the address bar:

```text
http://127.0.0.1:5050
```

Press Enter.

**What you should see:** The MLflow page with an experiment named **`heart_disease_prediction`** and separate runs for each model.

**To stop the server:** click inside the terminal where `python3 -m mlflow ui` is running and press **Ctrl+C**.

### If the browser shows an error or nothing loads

- **Port 5050 busy:** try `5051` instead in the command and open `http://127.0.0.1:5051`.
- **Empty UI:** make sure you ran **Part D Step 2** (`python3 src/model_training/train.py`) at least once.
- **Wrong folder:** you must run `python3 -m mlflow ui` from the folder that contains the `mlruns` directory (same folder as this README).
- **Models or artifacts show “Upload failed” / missing model folder:** Training must log a full MLflow model under each run’s `artifacts/model/` (not only `model_pickled/`). If you trained before fixing this, run **Part D Step 2** again (`python3 src/model_training/train.py`). If Python was built without LZMA (`import _lzma` fails), install a Python that includes it (official python.org builds do; with pyenv on macOS run `brew install xz` then reinstall that Python version), or use the current `src/model_training/train.py`, which works around missing LZMA when logging models.

---

## One-shot script (optional, macOS / Linux only)

If you use **bash** and already completed Part B and Part C:

```bash
bash scripts/test_full_flow.sh
```

This runs EDA, training, and predictions in one go. You still start **MLflow** separately (Part E).

---

## Project layout (for reference)

- **`data/heart_disease_uci.csv`** — Raw dataset (bundled).
- **`data/heart_clean.csv`** — Cleaned data (created by Step 1).
- **`src/eda/eda.py`** — Cleaning + EDA plots.
- **`src/data_preprocessing/`** — Preprocessing shared by EDA and prediction.
- **`src/model_training/train.py`** — Training, tuning, MLflow, save best model.
- **`src/model_training/predict.py`** — Load raw data → preprocess → predict.
- **`src/api/api.py`** — FastAPI app (`/predict`, `/health`, `/metrics`).
- **`models/`** — Saved pickles (`best_model.pkl`) and metadata; see `models/README.txt`.
- **`Dockerfile`** — Container image for the API (`uvicorn api.api:app`).
- **`docker-compose.yml`** — API + Prometheus + Grafana for local monitoring.
- **`monitoring/`** — Prometheus config and Grafana provisioning/dashboards.
- **`k8s/`** — Kubernetes Namespace, Deployment, Service, HPA, Ingress.
- **`pytest.ini`** — Puts `src` on the path for `pytest`.
- **`.flake8`** — Lint rules for CI / local `python3 -m flake8 src tests`.
- **`.dockerignore`** — Smaller `docker build` context.
- **`.github/workflows/ci.yml`** — Lint → tests → preprocess → train → artifact → Docker smoke test.
- **`mlruns/`** — MLflow database (created when you train).
- **`screenshots/`** — Charts.
- **`requirements.txt`** — Python dependencies.

---

## Rubric mapping (course MLOps S2-25_AMLCSZG523)

Use this list in your report to map each criterion to evidence in this repo.

- **1 — Data acquisition & EDA [5]** — Dataset, clean/preprocess (missing values, encoding), EDA visuals (histograms, correlation heatmap, class balance). **Evidence:** `data/heart_disease_uci.csv` (bundled) and the **Dataset citation** line at the end of this section. **Preprocess:** `src/data_preprocessing/pre_processing_data.py` (`load_data`, `clean_data`; imputation, category maps, binary target). **EDA:** `src/eda/eda.py` → PNGs in `screenshots/` (`class_balance.png`, `feature_histograms.png`, `correlation_heatmap.png`, `age_by_target.png`). Run Part D Step 1: `python3 src/eda/eda.py all`.

- **2 — Feature engineering & models [8]** — Scaling/encoding, ≥2 classifiers, documented tuning, CV + metrics (accuracy, precision, recall, ROC-AUC). **Evidence:** `build_logistic_pipeline()` (imputer + `StandardScaler` + LR), `build_random_forest_pipeline()` (imputer + RF) in `src/model_training/train.py`. **Tuning:** `GridSearchCV` + stratified CV. **Metrics:** `_binary_metrics`, `classification_report`, confusion matrix & ROC plots in `screenshots/` (`cm_*.png`, `roc_*.png`).

- **3 — Experiment tracking [5]** — MLflow (or similar): params, metrics, artifacts, plots. **Evidence:** `mlflow` in `train.py` (experiment `heart_disease_prediction`), logged params/metrics/plots + sklearn model per run. **UI:** Part E — `python3 -m mlflow ui --backend-store-uri ./mlruns`.

- **4 — Packaging & reproducibility [7]** — Saved model, `requirements.txt`, reproducible preprocessing. **Evidence:** `models/best_model.pkl` (joblib sklearn `Pipeline`), `models/feature_names.pkl`, `models/training_metadata.json`; MLflow artifacts under `mlruns/`; same `clean_data`/`load_data` in `src/model_training/predict.py` as training; `requirements.txt`. See `models/README.txt`.

- **5 — CI/CD & testing [8]** — Unit tests; pipeline with lint, tests, training; artifacts/logs. **Evidence:** `tests/test_pipeline.py` and **Automated tests** section. **Lint:** `flake8` + `.flake8`. **CI:** `.github/workflows/ci.yml` (flake8 → pytest → preprocess → train → upload `models/` + `mlruns/` → Docker build + `/predict` smoke test).

- **6 — Model containerisation [5]** — Dockerised API; **`/predict`** JSON in/out; prediction + confidence. **Evidence:** `Dockerfile` (`uvicorn api.api:app`, `PYTHONPATH=/app/src`), `docker build` / `docker run` ([All commands §10](#10-docker-image-api-only)). **Schema:** `PatientData` in `src/api/api.py`; response includes `prediction`, `confidence`, `label`, `risk_level`.

- **7 — Production deployment [7]** — Cloud or Kubernetes; manifests; LB/Ingress; verify endpoints. **Evidence:** `k8s/` — `namespace.yaml`, `deployment.yaml`, `service.yaml` (LoadBalancer), `hpa.yaml`, `ingress.yaml`. Build/load `heart-disease-api:latest`, then `kubectl apply -f k8s/`. For coursework: screenshots of `kubectl get pods,svc`, browser `/health` and `/predict`, Ingress/LB URL (edit host in `ingress.yaml`).

- **8 — Monitoring & logging [3]** — API request logging; Prometheus/Grafana or dashboard. **Evidence:** `src/api/api.py` → stdout + file (`API_LOG_PATH`, default `api_requests.log`). **Metrics:** `GET /metrics`. **Stack:** `docker-compose.yml`, `monitoring/prometheus.yml`, Grafana provisioning, `monitoring/grafana/dashboards/heart_disease_api.json`.

- **9 — Documentation & reporting [2]** — Setup, modelling choices, MLflow summary, architecture, CI/deploy screenshots, repo link. **Evidence:** This README. **Submission:** PDF/DOCX (~10 pages), architecture diagram, screenshots (CI, Docker/K8s, Grafana), short video, deployed URL or local access — per instructor.

**Dataset citation:** UCI Heart Disease — https://archive.ics.uci.edu/ml/datasets/Heart+Disease

---

## Test the complete flow (end-to-end)

Use this once **Part B + Part C** (venv + `python3 -m pip install -r requirements.txt`) are done. All paths assume the **repository root**.

### Option A — one script (recommended)

Runs the **full MLOps path** on your machine:

1. **flake8** → **pytest** → **`python3 src/eda/eda.py all`** → **train** → **batch predict**
2. **Container / monitoring (default):** runs **Docker Compose** — API **:8000** + Prometheus **:9090** + Grafana **:3000**, sends **`POST /predict`** traffic, checks Prometheus metrics and Grafana dashboards, then runs **`docker compose down`** or **`docker-compose down`** (stack stopped when the script finishes).
   - **`--demo`** (macOS): tries to start Docker Desktop or install/start Colima first, then runs the same default flow.
   - **`--skip-docker`:** steps 1–5 only (no Compose).

```bash
# End-to-end (default): lint → tests → EDA → train → predict → Docker Compose monitoring
bash scripts/test_full_flow.sh
```

**Note:** The default run needs ports **8000**, **9090**, and **3000** free. Do not run another API on **8000** at the same time.

On success you will have: `data/heart_clean.csv`, plots under `screenshots/`, `models/*.pkl`, `mlruns/`, and `data/batch_predictions.csv`. Step **[6/6]** validates the **Compose** monitoring stack (not a separate “smoke” mode).

### Option B — same steps, manual copy-paste

1. **`python3 -m flake8 src tests`** — Expect exit code 0.
2. **`python3 -m pytest tests/ -v --tb=short`** — Expect all tests passed.
3. **`python3 src/eda/eda.py all`** — No traceback; `screenshots/*.png` and `data/heart_clean.csv` created.
4. **`python3 src/model_training/train.py`** — Saved-model message; `models/best_model.pkl`, `models/feature_names.pkl`, `mlruns/` updated.
5. **`python3 src/model_training/predict.py --output data/batch_predictions.csv`** — Preview printed; CSV written.
6. **Compose monitoring** — Same as **`scripts/test_full_flow.sh`** step **[6/6]** — `docker compose` or `docker-compose` brings up API + Prometheus + Grafana, health checks, `/predict`, Prometheus queries, then tears the stack down. Optional: single-container API ([Quick Start step 4](#4-optional-build-and-run-the-api-docker-image-only)).
7. *(optional)* **`python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 -p 5050`** — Browser `http://127.0.0.1:5050` shows experiment **`heart_disease_prediction`**.

### Option C — REST API (after step 5)

Terminal 1:

```bash
source .venv/bin/activate
python3 src/api/api.py
```

Terminal 2:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}'
```

You should see JSON with **`prediction`**, **`confidence`**, **`label`**, **`risk_level`**. Interactive docs: `http://127.0.0.1:8000/docs`.

### Option D — Docker image (after step 5)

CI (**docker-build-smoke**) builds the image and smoke-tests **`/health`** and **`POST /predict`**. Locally, build and run the API container on port **8000**:

```bash
docker build -t heart-disease-api:latest .
docker run --rm -p 8000:8000 heart-disease-api:latest
```

Then repeat the **`curl`** commands against `http://127.0.0.1:8000` (or use `/docs`).

### Option E — API + Prometheus + Grafana

Automated check (same as **default** `test_full_flow.sh` — builds stack, hits metrics APIs, tears down):

```bash
bash scripts/test_full_flow.sh
```

To **keep** the stack running and explore the UI manually:

```bash
docker compose up --build
# or:  docker-compose up --build
```

Then open API (**8000**), Prometheus (**9090**), Grafana (**3000**, admin / admin123). Send a few **`POST /predict`** requests, then open the **Heart Disease API (Prometheus)** dashboard in Grafana.

### Option F — Kubernetes (advanced)

Build/push or `docker save` / `docker load` the image into your cluster, then:

```bash
kubectl apply -f k8s/
kubectl get pods,svc -n heart-disease
```

Port-forward if you do not have a real LoadBalancer:

```bash
kubectl port-forward -n heart-disease svc/heart-disease-api-service 8080:80
curl http://127.0.0.1:8080/health
```

---

## Troubleshooting

Standalone reference if **[Quick Start](#quick-start-end-to-end)** or **`scripts/test_full_flow.sh`** fails: monitoring stack, Docker/Compose, Python/venv, MLflow, and the one-shot script.

### A. Grafana, Prometheus, and Compose

#### 1) Grafana shows “No data”

- Confirm Prometheus is scraping:

```bash
curl -sS "http://127.0.0.1:9090/api/v1/query" --data-urlencode 'query=up{job="heart-disease-api"}'
```

- Create at least one `/predict` request (metrics appear only after first increment for labeled counters):

```bash
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":1}' >/dev/null
```

- In Grafana, set time range to **Last 5 minutes** and refresh.

#### 2) Prometheus not ready / not starting

Check logs:

```bash
docker-compose logs --tail=200 prometheus
```

If you see a config error about scrape timeout, ensure in `monitoring/prometheus.yml` that:
`scrape_timeout <= scrape_interval`.

#### 3) Port 8000 already in use (very common)

Symptoms: Prometheus scrapes Docker API, but your `curl http://127.0.0.1:8000/...` hits a different local server, so metrics never match.

Check who is listening:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Stop that process (example PID):

```bash
kill <PID>
```

Then restart the compose stack:

```bash
docker-compose down -v --remove-orphans
docker-compose up -d --build --force-recreate
```

#### 4) `/predict` returns 500 in Docker

This project pins `scikit-learn==1.6.1` because the saved model was trained with that version.
If you changed dependencies, rebuild the image:

```bash
docker-compose up -d --build --force-recreate
docker-compose logs --tail=120 heart-disease-api
```

#### 5) Docker/Colima permission denied (macOS)

If you see: `permission denied ... ~/.colima/default/docker.sock`, restart Colima and verify:

```bash
colima stop && colima start
docker info
```

### B. Python, venv, MLflow, and training

- **`python3: command not found`** — Install Python 3 and put `python3` on your PATH (on Windows, try the **`py`** launcher).
- **`No module named ...`** — Activate the venv ([Part B](#part-b--create-a-virtual-environment-recommended)) and run `python3 -m pip install -r requirements.txt` again.
- **`Clean data not found` when training** — Run `python3 src/eda/eda.py all` first ([Part D](#part-d--run-the-full-workflow-copypaste-in-order), Step 1).
- **MLflow page blank** — Train first (`python3 src/model_training/train.py`). Use `./mlruns` and the project root as in [Part E](#part-e--open-the-mlflow-web-ui).
- **`test_full_flow.sh` fails at train** (`PermissionError` / `sysconf` / **joblib**) — Some sandboxes block parallel workers. Run the script in a normal terminal, or set `n_jobs=1` in `GridSearchCV` inside `train.py`.

### C. `test_full_flow.sh` and Docker bootstrap

- **Docker step skipped or fails** — Start **Docker Desktop** (macOS/Windows). Free the ports the script needs. For Python-only steps: `bash scripts/test_full_flow.sh --skip-docker`.
- **Compose / monitoring fails (ports / bind)** — Keep **8000**, **9090**, **3000** free. Stop any local `python3 src/api/api.py` or another Compose stack on those ports; run `docker compose down` or `docker-compose down` in this repo if something is still bound.
- **`unknown flag: --build`** when running **`docker compose up --build`** — Your CLI may not have the Compose **V2 plugin**; use **`docker-compose up --build`** (install: `brew install docker-compose` on macOS, or [Docker Compose install](https://docs.docker.com/compose/install/)).
- **`unknown shorthand flag: 'f'` or Compose not found** — Install Compose (e.g. `brew install docker-compose`). Run the script from the **repo root** (no fragile `-f` path).
- **`docker` not on PATH** — On macOS with [Homebrew](https://brew.sh), try `bash scripts/test_full_flow.sh --demo` (Docker Desktop or **Colima**). Or install [Docker Desktop](https://docs.docker.com/desktop/) manually. Python-only: `bash scripts/test_full_flow.sh --skip-docker`.

---

## Licence / academic use

For coursework submission; dataset subject to UCI terms of use.
