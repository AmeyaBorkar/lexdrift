"""Celery application configuration for LexDrift background workers.

Uses Redis as both broker and result backend. All settings are pulled from
the central config module so a single .env controls everything.
"""

from celery import Celery
from celery.schedules import crontab

from lexdrift.config import settings

app = Celery("lexdrift")

app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task routing
    task_default_queue="default",
    task_routes={
        "lexdrift.workers.ingest.*": {"queue": "ingest"},
        "lexdrift.workers.analyze.*": {"queue": "analyze"},
        "lexdrift.workers.monitor.*": {"queue": "monitor"},
    },
    # Beat schedule (periodic tasks)
    beat_schedule={
        "poll-edgar-for-new-filings": {
            "task": "lexdrift.workers.monitor.poll_edgar",
            "schedule": 1800.0,  # every 30 minutes
        },
        "run-daily-pipeline": {
            "task": "lexdrift.workers.pipeline.run_daily_pipeline",
            "schedule": crontab(hour=6, minute=0),  # every day at 6 AM UTC
        },
    },
    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Auto-discover tasks in worker modules
app.autodiscover_tasks(["lexdrift.workers.ingest", "lexdrift.workers.analyze", "lexdrift.workers.monitor", "lexdrift.workers.pipeline"])
