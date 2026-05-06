"""
orchestrator/main.py
====================
Cloud Run Service – Orchestrator

Receives an HTTP POST, generates walk-forward periods, then triggers a
Cloud Run Job where each period runs as a parallel task with its own
dedicated CPU + memory allocation.

Flow:
    POST /create-job
        │
        ▼
    generate N periods
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

import requests
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
WORKER_IMAGE    = os.environ["WORKER_IMAGE"]   # e.g. europe-west1-docker.pkg.dev/proj/repo/xgb-worker:latest
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
def build_task_payloads(request: dict, train_periods: list) -> list[dict]:
    """
    Build one payload dict per period.
    Stored in GCS as a manifest; each worker task reads its own entry
    using CLOUD_RUN_TASK_INDEX.
    """
    run_id = os.environ.get("CLOUD_RUN_EXECUTION", str(uuid.uuid4())[:8])
    base_name = request["experiment_name"]
    payloads = []

    for period in train_periods:
        period_id = period.get("test_start", "unknown").replace("-", "")
        payloads.append({
            "experiment_name"    : f"{base_name}_{period_id}",
            "prediction_horizon" : request["prediction_horizon"],
            "train_test_period"  : period,
            "_meta": {
                "period_id"    : period_id,
                "dispatched_at": datetime.utcnow().isoformat(),
                "run_id"       : run_id,
            },
        })

    return payloads


def upload_manifest(payloads: list[dict], job_name: str) -> str:
    """
    Upload the periods manifest to GCS.
    Returns the GCS URI so workers can read it.

    Layout: gs://{bucket}/manifests/{job_name}/manifest.json
    The file is a JSON array; worker reads index CLOUD_RUN_TASK_INDEX.
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


def build_env_overrides(request: dict) -> dict:
    """
    Collect optional config fields to pass to workers as env vars.
    Workers read these alongside MANIFEST_URI and CLOUD_RUN_TASK_INDEX.
    """
    overrides = {}
    optional_fields = [
        "test_period_months",
        "validation_period_months",
        "validation_offset_months",
        "test_period_start_offset_months",
        "trading_calendar",
    ]
    for field in optional_fields:
        if request.get(field) is not None:
            # Env vars are strings; worker casts back to correct type
            overrides[field.upper()] = str(request[field])
    return overrides


def trigger_cloud_run_job(
    job_name    : str,
    payloads    : list[dict],
    manifest_uri: str,
    env_overrides: dict,
) -> str:
    """
    Create (or update) a Cloud Run Job and immediately trigger an execution.

    Each task gets:
        CLOUD_RUN_TASK_INDEX  – injected by GCP (0-based index into manifest)
        MANIFEST_URI          – GCS path to the full periods manifest
        GCS_BUCKET            – for writing results
        + any optional config overrides

    Returns the execution name.
    """
    client     = run_v2.JobsClient()
    parent     = f"projects/{GCP_PROJECT}/locations/{GCP_REGION}"
    job_path   = f"{parent}/jobs/{job_name}"
    task_count = len(payloads)

    # ── Base env vars every task needs ────────────────────────────────────
    base_env = {
        "GCP_PROJECT" : GCP_PROJECT,
        "GCS_BUCKET"  : GCS_BUCKET,
        "MANIFEST_URI": manifest_uri,
    }
    all_env = {**base_env, **env_overrides}

    env_vars = [
        run_v2.EnvVar(name=k, value=v)
        for k, v in all_env.items()
    ]

    # ── Task template ─────────────────────────────────────────────────────
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

    # ── Job definition ────────────────────────────────────────────────────
    job = run_v2.Job(
        template=run_v2.ExecutionTemplate(
            task_count  =task_count,
            parallelism =task_count,   # all tasks fire simultaneously
            template    =task_template,
        )
    )

    # ── Create or update the job ──────────────────────────────────────────
    try:
        existing = client.get_job(name=job_path)
        logger.info(f"Job {job_name} exists — updating ...")
        operation = client.update_job(job=run_v2.Job(name=job_path, **job))
        operation.result()  # wait for update to complete
    except Exception:
        logger.info(f"Creating new job: {job_name}")
        operation = client.create_job(parent=parent, job=job, job_id=job_name)
        operation.result()  # wait for creation

    # ── Trigger execution ─────────────────────────────────────────────────
    logger.info(f"Triggering execution of job {job_name} with {task_count} parallel tasks ...")
    exec_operation = client.run_job(name=job_path)
    execution      = exec_operation.metadata   # don't wait — fire and return

    execution_name = getattr(execution, "name", job_name)
    logger.info(f"Execution triggered → {execution_name}")
    return execution_name


# ── Orchestrator endpoint ──────────────────────────────────────────────────────
@app.post("/create-job")
async def trigger_workers(request: Request):
    """
    Main orchestration endpoint.

    1. Validates the request.
    2. Generates walk-forward periods.
    3. Uploads a manifest JSON to GCS.
    4. Creates + triggers a Cloud Run Job where:
           task_count  = number of periods
           parallelism = number of periods   (all run simultaneously)
       Each task reads its period from the manifest using CLOUD_RUN_TASK_INDEX.

    Required body fields:
        experiment_name       : str
        prediction_horizon    : int
        train_start           : str  "YYYY-MM-DD"
        train_end             : str  "YYYY-MM-DD"

    Optional body fields:
        test_period_months                : int
        validation_period_months          : int
        validation_offset_months          : int
        test_period_start_offset_months   : int
        trading_calendar                  : str
        worker_cpu                        : str  (overrides WORKER_CPU env var)
        worker_memory                     : str  (overrides WORKER_MEMORY env var)
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

        # ── 2. Generate periods ────────────────────────────────────────────
        train_periods = get_train_periods(payload["train_start"], payload["train_end"])
        if not train_periods:
            raise HTTPException(
                status_code=400,
                detail=f"No periods for range {payload['train_start']} → {payload['train_end']}",
            )

        logger.info(f"Generated {len(train_periods)} period(s)")

        # ── 3. Build task payloads ─────────────────────────────────────────
        task_payloads = build_task_payloads(payload, train_periods)

        # ── 4. Upload manifest to GCS ──────────────────────────────────────
        # Job name: sanitised experiment name + short timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        job_name  = f"{payload['experiment_name'].lower().replace('_', '-')}-{timestamp}"
        job_name  = job_name[:49]  # Cloud Run Job name max length

        manifest_uri  = upload_manifest(task_payloads, job_name)
        env_overrides = build_env_overrides(payload)

        # ── 5. Trigger Cloud Run Job ───────────────────────────────────────
        execution_name = trigger_cloud_run_job(
            job_name     = job_name,
            payloads     = task_payloads,
            manifest_uri = manifest_uri,
            env_overrides= env_overrides,
        )

        # ── 6. Respond immediately ─────────────────────────────────────────
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