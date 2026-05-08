# Vel.ai Training Orchestrator

A Cloud Run service that receives a training request and spawns parallel Cloud Run Jobs — one per walk-forward training period — enabling fully parallelized model training runs.

---

## Architecture Overview

```
HTTP POST /create-job
        │
        ▼
┌─────────────────────────┐
│   Orchestrator Service  │  ← Cloud Run Service (this repo)
│   (FastAPI on port 8080)│
└────────────┬────────────┘
             │
             ├── 1. Generate walk-forward training periods
             ├── 2. Build per-period task payloads
             ├── 3. Upload manifest.json to GCS
             └── 4. Trigger Cloud Run Job
                        │
             ┌──────────▼──────────┐
             │   Cloud Run Job     │
             │  task_count = N     │  (N = number of periods)
             │  parallelism = N    │  (all run simultaneously)
             └─────────────────────┘
                   │        │
              Task 0    Task 1  ...  Task N
                   │        │
             Each task reads manifest[CLOUD_RUN_TASK_INDEX]
             from GCS and executes its assigned training period
```

The orchestrator is **non-blocking** — it returns immediately after triggering the job without waiting for worker completion.

---

## Repository Structure

```
.
├── main.py               # FastAPI application — orchestrator service entrypoint
├── config.py             # Dataclasses for experiment configuration
├── generate_periods.py   # Walk-forward period generation using NYSE trading calendar
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build config (python:3.10-slim, 8 uvicorn workers)
└── test.ipynb            # Notebook for testing period generation locally
```

---

## API Reference

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "status_code": 200,
  "timestamp": "2026-05-08T12:00:00.000000",
  "service": "orchestrator",
  "version": "2.0.0"
}
```

---

### `POST /create-job`
Generates training periods and triggers a Cloud Run Job with one parallel task per period.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `experiment_name` | string | Yes | Name of the experiment (used in job naming) |
| `prediction_horizon` | int | Yes | Forecast horizon in days |
| `train_start` | string (YYYY-MM-DD) | Yes | Start of overall training window |
| `train_end` | string (YYYY-MM-DD) | Yes | End of overall training window |
| `test_period_months` | int | No | Length of each test period in months |
| `validation_period_months` | int | No | Length of each validation period in months |
| `validation_offset_months` | int | No | Offset for validation window |
| `test_period_start_offset_months` | int | No | Offset for test period start |
| `trading_calendar` | string | No | Trading calendar to use (default: `NYSE`) |

**Example Request:**
```json
{
  "experiment_name": "Live_System_2026",
  "prediction_horizon": 7,
  "train_start": "2020-01-01",
  "train_end": "2026-01-01"
}
```

**Success Response (HTTP 200):**
```json
{
  "status": "triggered",
  "job_name": "live-system-2026-20260508120000",
  "execution_name": "projects/{PROJECT}/locations/{REGION}/jobs/{JOB_NAME}/executions/{EXEC_ID}",
  "total_periods": 13,
  "parallelism": 13,
  "manifest_uri": "gs://{GCS_BUCKET}/manifests/{job_name}/manifest.json",
  "dispatched_at": "2026-05-08T12:00:00.123456"
}
```

**Error Response (HTTP 400/500):**
```json
{
  "status": "error",
  "error": "Missing required field: experiment_name"
}
```

---

## Environment Variables

These must be configured on the Cloud Run Service:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_PROJECT` | Yes | — | GCP project ID |
| `GCS_BUCKET` | Yes | — | GCS bucket for manifests and results |
| `WORKER_IMAGE` | Yes | — | Container image URI of the worker service |
| `WORKER_SA_EMAIL` | Yes | — | Service account email for worker tasks |
| `GCP_REGION` | No | `europe-west1` | Cloud Run region |
| `WORKER_CPU` | No | `8` | vCPUs allocated per worker task |
| `WORKER_MEMORY` | No | `32Gi` | Memory allocated per worker task |
| `JOB_MAX_RETRIES` | No | `1` | Per-task retry attempts on failure |
| `JOB_TIMEOUT` | No | `14400` | Per-task timeout in seconds (4 hours) |
| `MLFLOW_TRACKING_URI` | No | — | MLflow tracking server URI (forwarded to workers) |

> `CLOUD_RUN_EXECUTION` is auto-set by Cloud Run and used internally for run ID correlation.

---

## How It Works

### 1. Period Generation
`generate_periods.py` uses the NYSE trading calendar to produce walk-forward training/validation/test windows. Each period is an 8-element tuple:

```
[train_start, train_end,
 internal_train_start, internal_train_end,
 validation_start, validation_end,
 test_start, test_end]
```

Only periods that overlap with the requested `[train_start, train_end]` window are included.

### 2. Manifest Upload
All per-period task payloads are serialized to a single `manifest.json` and uploaded to GCS:

```
gs://{GCS_BUCKET}/manifests/{job_name}/manifest.json
```

Each payload contains:
- `experiment_name`, `prediction_horizon`, `train_test_period`
- Pass-through request fields (e.g. `test_period_months`)
- Metadata: `period_id`, `dispatched_at`, `run_id`

### 3. Cloud Run Job Execution
A Cloud Run Job is created with:
- `task_count` = number of periods
- `parallelism` = number of periods (all tasks run simultaneously)

Each task reads its assigned payload from the manifest using the `CLOUD_RUN_TASK_INDEX` environment variable (0-based index auto-set by Cloud Run).

---

## Deployment

### Build & Push with Cloud Build

The service is containerized using `python:3.10-slim` and runs 8 Uvicorn workers:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "8"]
```

Build and deploy to Cloud Run:

```bash
gcloud builds submit --tag gcr.io/{PROJECT_ID}/training-orchestrator

gcloud run deploy training-orchestrator \
  --image gcr.io/{PROJECT_ID}/training-orchestrator \
  --region europe-west1 \
  --set-env-vars GCP_PROJECT={PROJECT_ID},GCS_BUCKET={BUCKET},WORKER_IMAGE={WORKER_IMAGE_URI},WORKER_SA_EMAIL={SA_EMAIL}
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn[standard]` | ASGI server |
| `pandas` | Data manipulation |
| `pandas-market-calendars` | NYSE/trading calendar calculations |
| `google-cloud-storage` | GCS manifest upload/read |
| `google-cloud-run` | Cloud Run Jobs API client |
| `google-cloud-pubsub` | Retained from legacy Pub/Sub architecture |

---

## Notes

- Job names are derived from `experiment_name` + timestamp and are capped at **49 characters** (Cloud Run constraint).
- If a job with the same name already exists, it is updated in place before execution.
- The orchestrator does not track worker completion — use Cloud Run Job logs or MLflow for monitoring results.
