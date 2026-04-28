import os
import json
import logging
from datetime import datetime
import traceback
from concurrent.futures import as_completed

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from google.cloud import pubsub_v1

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
GCP_PROJECT  = os.environ["GCP_PROJECT"]
PUBSUB_TOPIC = os.environ["PUBSUB_TOPIC"]  # e.g. "xgb-training-periods"
WORKER_URL   = os.environ.get(
    "WORKER_URL",
    "https://vel-ai-training-pipeline-622310346154.europe-west1.run.app/pubsub",
)

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    try:
        return {
            "status"      : "healthy",
            "status_code" : 200,
            "timestamp"   : datetime.utcnow().isoformat(),
            "service"     : "orchestrator-pipeline",
            "version"     : "1.0.0",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status"   : "unhealthy",
                "error"    : str(e),
                "traceback": traceback.format_exc(),
            },
        )


# ── Helpers ────────────────────────────────────────────────────────────────────
def build_worker_payloads(request: dict, train_periods: list) -> list[dict]:
    """
    Build one Pub/Sub message payload per period.
    Each payload matches exactly what the worker's /pubsub endpoint expects:
        - experiment_name
        - prediction_horizon
        - train_test_period
        - _meta  (orchestration metadata, ignored by worker training logic)
    """
    base_experiment_name = request["experiment_name"]
    payloads = []

    for period in train_periods:
        # Period id from test_start for unique experiment naming
        period_id = period.get("test_start", "unknown").replace("-", "")

        payloads.append({
            # ── Required worker fields ──────────────────────────────────
            "experiment_name"    : f"{base_experiment_name}",
            "prediction_horizon" : request["prediction_horizon"],
            "train_test_period"  : period,   # {train_start, train_end, test_start, test_end}

            # ── Orchestration metadata (for logging/debugging) ──────────
            "_meta": {
                "period_id"    : period_id,
                "dispatched_at": datetime.utcnow().isoformat(),
                "run_id"       : os.environ.get("CLOUD_RUN_EXECUTION", "local"),
            },
        })

    return payloads


def publish_to_pubsub(payloads: list[dict]) -> tuple[list[str], list[str]]:
    """
    Publish all payloads to the Pub/Sub topic that pushes to the worker.
    Returns (succeeded_period_ids, failed_period_ids).
    """
    publisher  = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT, PUBSUB_TOPIC)

    # Map future → period_id so we can report which ones failed
    future_to_period: dict = {}

    for payload in payloads:
        period_id = payload["_meta"]["period_id"]
        data      = json.dumps(payload).encode("utf-8")

        future = publisher.publish(
            topic_path,
            data,
            # Message attributes — visible in Pub/Sub console without decoding body
            period_id        = period_id,
            experiment_name  = payload["experiment_name"],
            dispatched_at    = payload["_meta"]["dispatched_at"],
        )
        future_to_period[future] = period_id

    succeeded, failed = [], []

    for future in as_completed(future_to_period):
        period_id = future_to_period[future]
        try:
            msg_id = future.result(timeout=30)
            logger.info(f"  ✓  [{period_id}]  published  message_id={msg_id}")
            succeeded.append(period_id)
        except Exception as exc:
            logger.error(f"  ✗  [{period_id}]  publish failed: {exc}")
            failed.append(period_id)

    return succeeded, failed


# ── Orchestrator endpoint ──────────────────────────────────────────────────────
@app.post("/create-pubsub")
async def trigger_workers(request: dict):
    """
    Orchestrator entry point.

    1. Validates the request.
    2. Builds the config / period list.
    3. Publishes one Pub/Sub message per period.
       Pub/Sub push subscription forwards each message to:
           POST {WORKER_URL}  →  worker /pubsub endpoint
    4. Returns a dispatch summary.

    Required fields:
        experiment_name       : str   – base name; period suffix appended per message
        prediction_horizon    : int   – trading days ahead to predict
        train_start           : str   – "YYYY-MM-DD"  walk-forward window start
        train_end             : str   – "YYYY-MM-DD"  walk-forward window end

    Optional fields:
        test_period_months                : int
        validation_period_months          : int
        validation_offset_months          : int
        test_period_start_offset_months   : int
        trading_calendar                  : str
    """
    try:
        # ── 1. Validate required fields ────────────────────────────────────
        required_fields = ["experiment_name", "prediction_horizon", "train_start", "train_end"]
        missing = [f for f in required_fields if f not in request]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field(s): {missing}",
            )

        logger.info("=" * 60)
        logger.info(f"Orchestrator triggered — experiment: {request['experiment_name']}")
        logger.info(f"  Range     : {request['train_start']} → {request['train_end']}")
        logger.info(f"  Horizon   : {request['prediction_horizon']}")
        logger.info(f"  Worker URL: {WORKER_URL}")
        logger.info("=" * 60)

        # ── 2. Build config ────────────────────────────────────────────────
        config = Config()
        config.experiment.prediction_horizon = request["prediction_horizon"]

        # Optional overrides
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

        # ── 3. Generate periods ────────────────────────────────────────────
        train_periods = get_train_periods(request["train_start"], request["train_end"])

        if not train_periods:
            raise HTTPException(
                status_code=400,
                detail=f"No periods generated for range {request['train_start']} → {request['train_end']}",
            )

        logger.info(f"Generated {len(train_periods)} period(s) — dispatching to Pub/Sub ...\n")

        # ── 4. Build payloads ──────────────────────────────────────────────
        payloads = build_worker_payloads(request, train_periods)

        # ── 5. Publish ─────────────────────────────────────────────────────
        succeeded, failed = publish_to_pubsub(payloads)

        # ── 6. Response ────────────────────────────────────────────────────
        logger.info("")
        logger.info("── Dispatch summary ──────────────────────────────────")
        logger.info(f"  Total     : {len(payloads)}")
        logger.info(f"  Succeeded : {len(succeeded)}")
        logger.info(f"  Failed    : {len(failed)}")
        if failed:
            logger.warning(f"  Failed periods: {failed}")
        logger.info("=" * 60)

        status_code = 200 if not failed else 207  # 207 = partial success

        return JSONResponse(
            status_code=status_code,
            content={
                "status"           : "dispatched" if not failed else "partial",
                "worker_url"       : WORKER_URL,
                "total_periods"    : len(payloads),
                "succeeded"        : len(succeeded),
                "failed"           : len(failed),
                "failed_periods"   : failed,
                "succeeded_periods": succeeded,
                "dispatched_at"    : datetime.utcnow().isoformat(),
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