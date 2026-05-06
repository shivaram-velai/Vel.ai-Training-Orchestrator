"""
orchestrator/main.py
====================
Cloud Run Service – Orchestrator (Cloud Run Jobs architecture)

Receives an HTTP POST, generates walk-forward periods, then triggers a
Cloud Run Job where each period runs as a parallel task with its own
dedicated CPU + memory allocation.

Flow:
    POST /create-job
        │
        ▼
    generate N periods  (using Config overrides, same as Pub/Sub version)
        │
        ▼
    upload periods manifest → GCS  (workers read by CLOUD_RUN_TASK_INDEX)
        │
        ▼
    create + trigger Cloud Run Job  (task_count=N, parallelism=N)
        │
        ├── Task 0  → reads period[0] from GCS → trains → writes results
        ├── Task 1  → reads period[1] from GCS → trains → writes results
        ├── ...
        └── Task N  → reads period[N] from GCS → trains → writes results

ENV VARS (set on this Cloud Run Service):
    GCP_PROJECT        – GCP project ID
    GCP_REGION         – Cloud Run region  (e.g. europe-west1)
    GCS_BUCKET         – bucket for manifests + results
    WORKER_IMAGE       – worker container image URI
    WORKER_CPU         – CPU per task      (default: "8")
    WORKER_MEMORY      – memory per task   (default: "32Gi")
    WORKER_SA_EMAIL    – service account email for worker tasks
    JOB_MAX_RETRIES    – per-task retry attempts (default: 1)
    JOB_TIMEOUT        – per-task timeout seconds (default: 14400 = 4hrs)
"""

import json
import logging
import os
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from google.cloud import run_v2, storage

from generate_periods import get_train_periods
from config import Config

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI()

# ── Env ────────────────────────────────────────────────────────────────────────
GCP_PROJECT     = os.environ["GCP_PROJECT"]
GCP_REGION      = os.environ.get("GCP_REGION", "europe-west1")
GCS_BUCKET      = os.environ["GCS_BUCKET"]
WORKER_IMAGE    = os.environ["WORKER_IMAGE"]
WORKER_CPU      = os.environ.get("WORKER_CPU", "8")
WORKER_MEMORY   = os.environ.get("WORKER_MEMORY", "32Gi")
WORKER_SA_EMAIL = os.environ["WORKER_SA_EMAIL"]
JOB_MAX_RETRIES = int(os.environ.get("JOB_MAX_RETRIES", 1))
JOB_TIMEOUT     = int(os.environ.get("JOB_TIMEOUT", 14400))  # 4 hours


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    try:
        return {
            "status"     : "healthy",
            "status_code": 200,
            "timestamp"  : datetime.utcnow().isoformat(),
            "service"    : "orchestrator",
            "version"    : "2.0.0",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)},
        )


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_config(request: dict) -> Config:
    """
    Build a Config object from the request, applying all optional overrides.
    Mirrors the Pub/Sub version's config building block exactly.
    """
    config = Config()
    config.experiment.prediction_horizon = request["prediction_horizon"]

    optional_fields = [
        "test_period_months",
        "validation_period_months",
        "validation_offset_months",
        "test_period_start_offset_months",
        "trading_calendar",
    ]
    for field in optional_fields:
        if request.get(field) is not None:
            setattr(config.experiment, field, request[field])

    return config


def build_task_payloads(request: dict, train_periods: list) -> list[dict]:
    """
    Build one payload dict per period for the GCS manifest.

    Mirrors the Pub/Sub version:
      - experiment_name      : base name only (no per-period suffix, consistent
                               with Pub/Sub version's f"{base_experiment_name}")
      - prediction_horizon   : from request
      - train_test_period    : the period tuple/dict
      - pass-through fields  : everything else in request that isn't a reserved key
                               (same **{k: v ...} spread as Pub/Sub version)
      - _meta                : orchestration metadata

    Period ID extraction uses period[-2] (index into the period tuple/list)
    to match the Pub/Sub version's `period[-2].replace("-", "")`.
    If your periods are dicts, falls back to period["test_start"].
    """
    reserved_keys = {
        "experiment_name", "prediction_horizon", "train_test_period", "_meta",
        # orchestrator-only fields — don't forward to workers
        "train_start", "train_end",
    }

    run_id    = os.environ.get("CLOUD_RUN_EXECUTION", str(uuid.uuid4())[:8])
    base_name = request["experiment_name"]
    payloads  = []

    for period in train_periods:
        # Match Pub/Sub version: period[-2] for list/tuple, fallback for dict
        if isinstance(period, (list, tuple)):
            period_id = str(period[-2]).replace("-", "")
        else:
            period_id = period.get("test_start", "unknown").replace("-", "")

        payloads.append({
            # ── Required worker fields ──────────────────────────────────
            "experiment_name"    : base_name,           # no suffix, same as Pub/Sub version
            "prediction_horizon" : request["prediction_horizon"],
            "train_test_period"  : period,

            # ── Pass-through: all extra request fields ──────────────────
            # Mirrors: **{k: v for k, v in request.items() if k not in reserved_keys}
            **{k: v for k, v in request.items() if k not in reserved_keys},

            # ── Orchestration metadata ──────────────────────────────────
            "_meta": {
                "period_id"    : period_id,
                "dispatched_at": datetime.utcnow().isoformat(),
                "run_id"       : run_id,
            },
        })

    return payloads


def upload_manifest(payloads: list[dict], job_name: str) -> str:
    """
    Upload the full periods manifest to GCS as a JSON array.
    Each worker task reads its own entry using CLOUD_RUN_TASK_INDEX.

    GCS layout: gs://{GCS_BUCKET}/manifests/{job_name}/manifest.json
    """
    gcs_path = f"manifests/{job_name}/manifest.json"
    client   = storage.Client()
    bucket   = client.bucket(GCS_BUCKET)
    blob     = bucket.blob(gcs_path)

    blob.upload_from_string(
        json.dumps(payloads, indent=2),
        content_type="application/json",
    )

    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"
    logger.info(f"Manifest uploaded → {gcs_uri}")
    return gcs_uri


def trigger_cloud_run_job(
    job_name     : str,
    payloads     : list[dict],
    manifest_uri : str,
) -> str:
    """
    Create (or update) a Cloud Run Job and immediately trigger an execution.

    All task config is in the manifest — no env var overrides needed here
    because pass-through fields are already baked into each manifest entry
    (handled in build_task_payloads via the **{...} spread).

    Each task receives from GCP automatically:
        CLOUD_RUN_TASK_INDEX  – 0-based index into the manifest array
        CLOUD_RUN_TASK_COUNT  – total number of tasks
        CLOUD_RUN_EXECUTION   – execution ID for tracing

    Returns the execution name.
    """
    client     = run_v2.JobsClient()
    parent     = f"projects/{GCP_PROJECT}/locations/{GCP_REGION}"
    job_path   = f"{parent}/jobs/{job_name}"
    task_count = len(payloads)

    # Only infrastructure env vars go here — all business config is in manifest
    env_vars = [
        run_v2.EnvVar(name="GCP_PROJECT",  value=GCP_PROJECT),
        run_v2.EnvVar(name="GCS_BUCKET",   value=GCS_BUCKET),
        run_v2.EnvVar(name="MANIFEST_URI", value=manifest_uri),
        run_v2.EnvVar(name="MLFLOW_TRACKING_URI", value=os.environ.get("MLFLOW_TRACKING_URI", "")),
    ]

    task_template = run_v2.TaskTemplate(
        containers=[
            run_v2.Container(
                image=WORKER_IMAGE,
                env=env_vars,
                resources=run_v2.ResourceRequirements(
                    limits={
                        "cpu"   : WORKER_CPU,
                        "memory": WORKER_MEMORY,
                    }
                ),
            )
        ],
        service_account=WORKER_SA_EMAIL,
        max_retries=JOB_MAX_RETRIES,
        timeout=f"{JOB_TIMEOUT}s",
    )

    job = run_v2.Job(
        template=run_v2.ExecutionTemplate(
            task_count =task_count,
            parallelism=task_count,   # all tasks fire simultaneously
            template   =task_template,
        )
    )

    # Create or update
    try:
        client.get_job(name=job_path)
        logger.info(f"Job {job_name} exists — updating ...")
        operation = client.update_job(job=run_v2.Job(name=job_path, **job))
        operation.result()
    except Exception:
        logger.info(f"Creating new job: {job_name}")
        operation = client.create_job(parent=parent, job=job, job_id=job_name)
        operation.result()

    # Trigger — don't wait for completion, return immediately
    logger.info(f"Triggering {job_name} with {task_count} parallel tasks ...")
    exec_operation = client.run_job(name=job_path)
    execution      = exec_operation.metadata

    execution_name = getattr(execution, "name", job_name)
    logger.info(f"Execution triggered → {execution_name}")
    return execution_name


# ── Orchestrator endpoint ──────────────────────────────────────────────────────
@app.post("/create-job")
async def trigger_workers(request: Request):
    """
    Main orchestration endpoint.

    Required body fields:
        experiment_name       : str
        prediction_horizon    : int
        train_start           : str  "YYYY-MM-DD"
        train_end             : str  "YYYY-MM-DD"

    Optional body fields (forwarded to every worker via manifest pass-through):
        test_period_months                : int
        validation_period_months          : int
        validation_offset_months          : int
        test_period_start_offset_months   : int
        trading_calendar                  : str
    """
    try:
        body    = await request.body()
        payload = json.loads(body)

        # ── 1. Validate ────────────────────────────────────────────────────
        required_fields = ["experiment_name", "prediction_horizon", "train_start", "train_end"]
        missing = [f for f in required_fields if f not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing field(s): {missing}")

        logger.info("=" * 60)
        logger.info(f"Orchestrator triggered: {payload['experiment_name']}")
        logger.info(f"  Range   : {payload['train_start']} → {payload['train_end']}")
        logger.info(f"  Horizon : {payload['prediction_horizon']}")
        logger.info("=" * 60)

        # ── 2. Build config (mirrors Pub/Sub version exactly) ──────────────
        config = build_config(payload)

        # ── 3. Generate periods (pass config, same as Pub/Sub version) ─────
        train_periods = get_train_periods(payload["train_start"], payload["train_end"], config)

        if not train_periods:
            raise HTTPException(
                status_code=400,
                detail=f"No periods for range {payload['train_start']} → {payload['train_end']}",
            )

        logger.info(f"Generated {len(train_periods)} period(s)")
        logger.info(f"Train periods: {train_periods}")

        # ── 4. Build task payloads ─────────────────────────────────────────
        task_payloads = build_task_payloads(payload, train_periods)

        # ── 5. Upload manifest to GCS ──────────────────────────────────────
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name  = f"{payload['experiment_name'].lower().replace('_', '-')}-{timestamp}"
        job_name  = job_name[:49]  # Cloud Run Job name max 49 chars

        manifest_uri = upload_manifest(task_payloads, job_name)

        # ── 6. Trigger Cloud Run Job ───────────────────────────────────────
        execution_name = trigger_cloud_run_job(
            job_name     = job_name,
            payloads     = task_payloads,
            manifest_uri = manifest_uri,
        )

        # ── 7. Response ────────────────────────────────────────────────────
        logger.info("── Dispatch complete ─────────────────────────────────")
        logger.info(f"  Job       : {job_name}")
        logger.info(f"  Execution : {execution_name}")
        logger.info(f"  Tasks     : {len(task_payloads)} (all parallel)")
        logger.info(f"  Manifest  : {manifest_uri}")
        logger.info("=" * 60)

        return JSONResponse(
            status_code=200,
            content={
                "status"        : "triggered",
                "job_name"      : job_name,
                "execution_name": execution_name,
                "total_periods" : len(task_payloads),
                "parallelism"   : len(task_payloads),
                "manifest_uri"  : manifest_uri,
                "dispatched_at" : datetime.utcnow().isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[orchestrator] Error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)},
        )