# CybersecurityML

A machine learning service for network intrusion detection. It classifies network sessions as **normal** or a **possible attack** using a Random Forest model trained on ~9,500 labelled sessions.

---

## How it works

Raw session features (packet size, protocol, login attempts, encryption, browser type, etc.) are fed through a preprocessing pipeline and into a trained Random Forest classifier. The model outputs a binary prediction and an attack-probability score.

**Best model metrics** (RF on 17 features, 80/20 stratified split):

| Metric | Value |
|---|---|
| Accuracy | 0.883 |
| Sensitivity (recall) | 0.746 |
| Specificity | 0.994 |
| Precision | 0.991 |
| F1 | 0.851 |
| ROC-AUC | 0.875 |

---

## Project structure

```
src/          Application package (config, preprocessing, model loading, inference)
api/          FastAPI REST API
training/     Standalone model training script
scripts/      Utility scripts (local batch inference, API sample calls)
tests/        Unit and smoke tests
data/         Raw, interim, and processed datasets (gitignored)
models/       Saved model artifacts (.joblib)
config.yaml   Single source of truth for app and training configuration
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — used as the package manager throughout

### Install dependencies

```bash
uv sync
```

### Run the API locally

```bash
uv run uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Run inference on a single session |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "network_packet_size": 599,
    "protocol_type": "TCP",
    "login_attempts": 4,
    "session_duration": 492.98,
    "encryption_used": "DES",
    "ip_reputation_score": 0.61,
    "failed_logins": 1,
    "browser_type": "Edge",
    "unusual_time_access": 0
  }'
```

**Example response:**

```json
{
  "prediction": 1,
  "label": "Possible attack session",
  "probability": 0.87
}
```

### Run the tests

```bash
uv run pytest
```

---

## Scripts

### Batch local inference

Runs the full pipeline locally over the entire raw dataset and writes a timestamped CSV to `data/inference_output/`:

```bash
uv run python -m scripts.run_local_inference
```

### Sample inference against the deployed API

Picks 10 random sessions (seed=42) from the raw dataset, calls the REST API, and saves timestamped `inputs.json` / `outputs.json` to `data/inference_output/`:

```bash
uv run  python -m scripts.call_rest_api_sample_inference
```

Requires `API_URL` to be set (see [Deployment](#deployment) below).

---

## Deployment

The service is containerised and can be deployed to **Google Cloud Run**.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) — authenticated and configured
- A GCP project with the following APIs enabled:
  - Cloud Run
  - Artifact Registry

### Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
GCP_PROJECT_ID=your-project-id
GCP_REGION=europe-west1
GCP_SERVICE_NAME=cybersecurity-intrusion-api
GCP_ARTIFACT_REGISTRY_REPO=your-artifact-registry-repo
API_URL=https://your-cloud-run-url   # set this after first deploy
```

### Deploy

```bash
bash deploy/deploy.sh
```

This will:
1. Build the Docker image
2. Push it to Artifact Registry
3. Deploy (or re-deploy) the Cloud Run service
4. Print the service URL

---

## Re-training the model

Edit the `training` section of `config.yaml` then run:

```bash
uv run python -m training.model_training
```

A new timestamped artifact is saved to `models/`. Update `app.model_path` in `config.yaml` to point to it.

---

## Security note

The Cloud Run service is currently deployed with `--allow-unauthenticated`, making it publicly accessible. If you expose this to the internet you should add authentication. A simple option is to protect it with an **API key**:

- Use [Cloud Run with an API Gateway](https://cloud.google.com/api-gateway/docs) in front of it, or
- Add a FastAPI dependency that checks an `X-API-Key` header against a secret stored in [Google Secret Manager](https://cloud.google.com/secret-manager).

Without some form of auth, anyone with the URL can query the model.

---

## Credits

**Dataset:** [Cybersecurity Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset?resource=download) by dnkumars, published on Kaggle.