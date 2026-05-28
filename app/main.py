"""AI Webhook Ingestion Service — FastAPI application.

Layer responsibilities (enforced here)
───────────────────────────────────────
HTTP layer  → parse JSON, return 202/200/400, nothing else.
             Does NOT call the LLM. Does NOT normalise. Does NOT upsert entities.

Queue layer → enqueue(event_id) hands off work to the background worker.
             The vendor never waits for processing.

Storage     → two DB-level idempotency guards (see WebhookEvent docstring)
             before any INSERT is attempted.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, init_db
from .models import Invoice, Shipment, WebhookEvent
from .processor import compute_payload_hash, extract_vendor_event_id
from .schemas import (
    BatchIngestionResponse, BatchResult,
    EventDetail, InvoiceDetail, ShipmentDetail,
    UnclassifiedDetail, WebhookIngestionResponse,
)
from .worker import enqueue, start_worker, drain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start order: DB → worker task.
    Stop order : drain in-flight events → cancel worker.
    """
    await init_db()
    worker_task = start_worker()
    logger.info("Database initialised, normalisation worker started")
    yield
    # Graceful shutdown: let events already in the queue finish processing.
    await drain()
    worker_task.cancel()
    logger.info("Worker drained and stopped")


app = FastAPI(
    title="AI Webhook Ingestion Service",
    description="Ingests vendor webhooks, classifies and normalises them with an LLM, and stores the results.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Ingestion endpoint ────────────────────────────────────────────────────────

@app.post("/webhooks", response_model=WebhookIngestionResponse, status_code=202)
async def ingest_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Accept any JSON payload from a vendor webhook.

    Returns 202 Accepted immediately; normalisation runs in the background.
    Returns 200 (status=duplicate) if an identical or semantically equivalent
    payload was already received.

    Deduplication — two DB-level layers, both checked before any INSERT:

    Layer 1 — Vendor idempotency key (semantic dedup)
        extract_vendor_event_id() reads well-known fields (event_msg_id,
        event_id, advisory_id, or doc_ref+transaction.kind for GFP) from the
        raw payload synchronously. If a row with that vendor_event_id already
        exists, return 200 immediately — no LLM call, no second INSERT.
        This is protected by a partial unique index on vendor_event_id where
        vendor_event_id IS NOT NULL (DB constraint, not application logic).

    Layer 2 — SHA-256 payload hash (exact-byte dedup)
        Catches retries from vendors that don't supply a stable event ID.
        Protected by a UNIQUE constraint on payload_hash.
        The application does NOT pre-check; it attempts INSERT and catches
        IntegrityError. This is equivalent to PostgreSQL's
            INSERT INTO webhook_events (...) ON CONFLICT (payload_hash) DO NOTHING
        and is race-condition-safe because the uniqueness is enforced inside
        the DB transaction, not in application code.
    """
    try:
        raw_payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    # ── Layer 1: Vendor idempotency key ──────────────────────────────────────
    # Synchronous, <1ms, no LLM. Rejects semantic duplicates even when the
    # vendor's retry has slightly different bytes (e.g. added a timestamp field).
    vendor_event_id = extract_vendor_event_id(raw_payload)
    if vendor_event_id:
        existing_by_vid = (await db.execute(
            select(WebhookEvent).where(WebhookEvent.vendor_event_id == vendor_event_id)
        )).scalar_one_or_none()
        if existing_by_vid:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "duplicate",
                    "event_id": str(existing_by_vid.id),
                    "message": "Vendor event ID already processed; idempotent.",
                },
            )

    # ── Layer 2: Payload hash — attempt INSERT, catch DB constraint ──────────
    # No pre-check query. The UNIQUE constraint on payload_hash is the guard.
    # IntegrityError == ON CONFLICT DO NOTHING in the DB.
    payload_hash = compute_payload_hash(raw_payload)

    event = WebhookEvent(
        id=uuid.uuid4(),
        payload_hash=payload_hash,
        raw_payload=raw_payload,
        vendor_event_id=vendor_event_id,   # stored NOW — not waiting for the LLM
        event_type="pending",
        processing_status="pending",
    )
    db.add(event)

    try:
        await db.commit()
    except IntegrityError:
        # payload_hash collision — concurrent duplicate or exact retry.
        # This is equivalent to ON CONFLICT DO NOTHING: we simply look up what's
        # already there and tell the caller.
        await db.rollback()
        existing = (await db.execute(
            select(WebhookEvent).where(WebhookEvent.payload_hash == payload_hash)
        )).scalar_one_or_none()
        return JSONResponse(
            status_code=200,
            content={
                "status": "duplicate",
                "event_id": str(existing.id) if existing else "unknown",
                "message": "Payload already received; processing is idempotent.",
            },
        )

    # Hand off to the queue — endpoint returns 202 before LLM is called.
    await enqueue(event.id)

    return WebhookIngestionResponse(
        status="accepted",
        event_id=event.id,
        message="Webhook accepted; normalisation in progress.",
    )


# ── Batch ingestion endpoint ──────────────────────────────────────────────────

@app.post("/webhooks/batch", response_model=BatchIngestionResponse, status_code=202)
async def ingest_batch(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a JSON array of payloads and ingest each one independently.
    Returns a per-payload result (accepted / duplicate / error).
    Max 100 payloads per request.

    Each payload is committed individually so a duplicate in the middle of the
    array doesn't roll back all the preceding accepted events.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    if not isinstance(body, list):
        raise HTTPException(status_code=400, detail="Batch endpoint expects a JSON array [ {...}, {...} ]")

    if len(body) > 100:
        raise HTTPException(status_code=400, detail="Batch limited to 100 payloads per request")

    results: list[BatchResult] = []
    accepted_ids: list[uuid.UUID] = []

    for i, payload in enumerate(body):
        if not isinstance(payload, dict):
            results.append(BatchResult(index=i, status="error", error="Item is not a JSON object"))
            continue

        # Layer 1: Vendor idempotency key
        vendor_event_id = extract_vendor_event_id(payload)
        if vendor_event_id:
            existing_by_vid = (await db.execute(
                select(WebhookEvent).where(WebhookEvent.vendor_event_id == vendor_event_id)
            )).scalar_one_or_none()
            if existing_by_vid:
                results.append(BatchResult(index=i, status="duplicate", event_id=str(existing_by_vid.id)))
                continue

        # Layer 2: Payload hash — attempt INSERT, catch DB constraint
        payload_hash = compute_payload_hash(payload)
        event = WebhookEvent(
            id=uuid.uuid4(),
            payload_hash=payload_hash,
            raw_payload=payload,
            vendor_event_id=vendor_event_id,
            event_type="pending",
            processing_status="pending",
        )
        db.add(event)
        try:
            await db.commit()
            accepted_ids.append(event.id)
            results.append(BatchResult(index=i, status="accepted", event_id=str(event.id)))
        except IntegrityError:
            await db.rollback()
            existing = (await db.execute(
                select(WebhookEvent).where(WebhookEvent.payload_hash == payload_hash)
            )).scalar_one_or_none()
            results.append(BatchResult(
                index=i, status="duplicate",
                event_id=str(existing.id) if existing else None,
            ))

    # Enqueue all accepted events after the loop — avoids interleaving DB
    # writes with queue operations.
    for event_id in accepted_ids:
        await enqueue(event_id)

    return BatchIngestionResponse(
        total=len(body),
        accepted=len(accepted_ids),
        duplicates=sum(1 for r in results if r.status == "duplicate"),
        errors=sum(1 for r in results if r.status == "error"),
        results=results,
    )


# ── Read endpoints ────────────────────────────────────────────────────────────

@app.get("/events/{event_id}", response_model=EventDetail)
async def get_event(event_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    event = await db.get(WebhookEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@app.get("/shipments/{entity_id:path}", response_model=ShipmentDetail)
async def get_shipment(entity_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Shipment).where(Shipment.entity_id == entity_id))
    shipment = result.scalar_one_or_none()
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    return shipment


@app.get("/invoices/{entity_id:path}", response_model=InvoiceDetail)
async def get_invoice(entity_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Invoice).where(Invoice.entity_id == entity_id))
    invoice = result.scalar_one_or_none()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice


@app.get("/shipments", response_model=list[ShipmentDetail])
async def list_shipments(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Shipment).order_by(Shipment.updated_at.desc()).limit(100))
    return result.scalars().all()


@app.get("/invoices", response_model=list[InvoiceDetail])
async def list_invoices(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Invoice).order_by(Invoice.updated_at.desc()).limit(100))
    return result.scalars().all()


@app.get("/events", response_model=list[EventDetail])
async def list_events(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WebhookEvent).order_by(desc(WebhookEvent.created_at)).limit(50)
    )
    return result.scalars().all()


@app.get("/unclassified", response_model=list[UnclassifiedDetail])
async def list_unclassified(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WebhookEvent)
        .where(WebhookEvent.event_type == "unclassified")
        .order_by(desc(WebhookEvent.created_at))
        .limit(100)
    )
    rows = result.scalars().all()
    out = []
    for row in rows:
        nd = row.normalized_data or {}
        out.append(UnclassifiedDetail(
            id=row.id,
            received_at=row.received_at,
            occurred_at=row.occurred_at,
            vendor_event_id=row.vendor_event_id,
            possible_category=nd.get("possible_category"),
            title=nd.get("title"),
            notes=nd.get("notes"),
            processing_status=row.processing_status,
            raw_payload=row.raw_payload,
        ))
    return out


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())
