"""
Async work queue — decouples HTTP ingestion from LLM normalisation.

Why asyncio.Queue instead of FastAPI BackgroundTasks?
──────────────────────────────────────────────────────
BackgroundTasks is a list of callbacks that runs at the end of a single
request's lifecycle. It has no backpressure, no visibility, and no
separation from the HTTP layer — the endpoint *still* owns the work.

asyncio.Queue lives outside every request. The worker task runs forever as
an independent coroutine. When the next webhook arrives the endpoint writes
one UUID into the queue and returns 202 in microseconds, whether or not the
worker is currently busy with a slow LLM call.

This is the correct decoupling boundary for an ingestion pipeline and matches
how production systems (Celery + Redis, RabbitMQ, SQS) work — the HTTP tier
and the processing tier are completely separate.

Production note
───────────────
An in-process asyncio.Queue does NOT survive a process restart. Events sitting
in the queue when the process crashes are lost. However, every event row is
committed to the DB with processing_status="pending" *before* it enters the
queue, so a startup sweep (SELECT * FROM webhook_events WHERE
processing_status='pending') can re-enqueue stale events on boot. In
production, replace this with Redis + Celery or a managed queue (SQS, Pub/Sub)
for durability and horizontal scale.
"""

import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)

# Single in-process queue shared across the entire application lifetime.
_event_queue: asyncio.Queue[uuid.UUID] = asyncio.Queue()


async def enqueue(event_id: uuid.UUID) -> None:
    """Push an event ID onto the processing queue. Returns in O(1)."""
    await _event_queue.put(event_id)


async def drain() -> None:
    """
    Block until every item in the queue has been fully processed.

    Used for graceful shutdown (let in-flight events complete before the
    process exits) and in the test suite (ensure processing finishes before
    assertions run).
    """
    await _event_queue.join()


async def _worker_loop() -> None:
    """
    Long-running coroutine that processes events one at a time.

    Single-threaded by design:
    • SQLite (used in tests) does not support concurrent writes.
    • The state-ordering SQL WHERE clause already handles concurrent events
      for the same entity correctly; running parallel workers for different
      entities is safe and left as a production scale-out step.
    """
    from .processor import process_event  # local import avoids circular deps

    logger.info("Normalisation worker started")
    while True:
        event_id = await _event_queue.get()
        try:
            await process_event(event_id)
        except Exception:
            # process_event handles its own errors; this catches anything that
            # escapes — e.g. a session-factory failure before the try block.
            logger.exception("Worker: unhandled error for event %s", event_id)
        finally:
            _event_queue.task_done()  # unblocks drain() / Queue.join() waiters


def start_worker() -> asyncio.Task:
    """
    Spawn the worker coroutine as a background asyncio.Task.

    Call once from the FastAPI lifespan context manager.
    Returns the Task so the caller can cancel it on shutdown.
    """
    return asyncio.create_task(
        _worker_loop(),
        name="webhook-normalisation-worker",
    )
