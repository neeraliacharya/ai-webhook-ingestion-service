"""Background processing: normalization, deduplication, and entity upsert.

Three production realities handled here:

Reality 1 — Sub-second acknowledgment
    process_event() is always called via BackgroundTasks, which FastAPI runs
    AFTER the HTTP response is sent. The vendor receives 202 in <100ms; the
    slow LLM call happens invisibly in the background.

Reality 2 — Duplicate / retry payloads
    Two layers of deduplication:
    a) extract_vendor_event_id() pulls well-known idempotency keys
       (event_msg_id, event_id, doc_ref+kind, etc.) from the raw payload
       synchronously at ingestion time — before any LLM call.
    b) compute_payload_hash() is a SHA-256 of the full payload as a fallback
       for vendors that don't include a stable event ID.
    Both are checked at ingestion; either match returns 200 "duplicate"
    immediately without storing or processing the event again.

Reality 3 — Out-of-order event delivery
    The canonical entity state is driven by the vendor's own occurred_at
    timestamp, NOT by the order we received or inserted events.
    _upsert_shipment / _upsert_invoice issue a SQL UPDATE with a WHERE clause:
        WHERE status_occurred_at IS NULL
           OR status_occurred_at < :new_occurred_at
    This check happens inside the DB transaction, making it atomic.
    A late-arriving PICKED_UP (Apr 19) will never overwrite an already-stored
    IN_TRANSIT (Apr 21), even under concurrent load.
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import update, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .database import async_session_factory
from .llm import normalize_payload
from .models import Invoice, Shipment, WebhookEvent
from .schemas import LLMNormalizedEvent

logger = logging.getLogger(__name__)

# ── State ordering ─────────────────────────────────────────────────────────────
# Used only as a tiebreaker when neither event has an occurred_at timestamp.
_SHIPMENT_ORDER: dict[str, int] = {
    "PICKED_UP": 0,
    "IN_TRANSIT": 1,
    "OUT_FOR_DELIVERY": 2,
    "DELIVERED": 3,
}

_INVOICE_ORDER: dict[str, int] = {
    "ISSUED": 0,
    "PAID": 1,
    "VOIDED": 1,
    "REFUNDED": 2,
}

# ── Reality 2: Vendor idempotency key extraction ──────────────────────────────

# Known field names that vendors use as stable, unique event identifiers.
# Checked in order; first match wins.
_VENDOR_ID_FIELDS = (
    "event_msg_id",    # Maersk: "MAEU-EVT-2026-04-22-0001"
    "event_id",        # Ocean Network Express: "ONE-2026-04-28-114"
    "advisory_id",     # Marine Traffic Advisory: "MTA-2026-04-26-EU-007"
    "notification_id", # Hapag-Lloyd rate notices
    "hold_id",         # CBP hold notices
    "dispute_id",      # Carrier dispute events
    "maintenance_id",  # Platform maintenance notices
    "alert_id",        # Generic alert systems
    "message_id",      # Generic messaging platforms
)


def extract_idempotency_key(payload: dict) -> str | None:
    """
    Synchronously extract a stable vendor idempotency key from the raw payload.

    This runs at ingestion time (< 1ms, no LLM) so we can reject duplicate
    retries immediately, before storing anything or calling the LLM.

    Returns a string key, or None if no recognisable event ID is present.
    The key is intentionally narrow: it must uniquely identify a single
    logical event, not just a document. For example, GlobalFreightPay's
    "doc_ref" alone is NOT unique (the same invoice has an ISSUED event and
    a PAID event). We use doc_ref + transaction.kind as the composite key.

    Vendor-to-key mapping
    ─────────────────────
    Maersk                → event_msg_id
    Ocean Network Express → event_id
    Marine Traffic        → advisory_id
    Hapag-Lloyd notices   → notification_id
    CBP holds             → hold_id
    Carrier disputes      → dispute_id
    GlobalFreightPay      → doc_ref + "::" + transaction.kind  (composite)
    """
    # Direct field lookup — covers Maersk, ONE, most advisory/notification vendors
    for field in _VENDOR_ID_FIELDS:
        if val := payload.get(field):
            return str(val)

    # GlobalFreightPay composite key: doc_ref + transaction.kind
    # "GFP-INV-2026-Q2-08821::freight invoice raised"  (ISSUED event)
    # "GFP-INV-2026-Q2-08821::settled in full"          (PAID event — different key)
    if doc_ref := payload.get("doc_ref"):
        txn = payload.get("transaction", {})
        if isinstance(txn, dict) and (kind := txn.get("kind")):
            return f"{doc_ref}::{kind}"

    return None


# Backwards-compatible alias used by main.py import
extract_vendor_event_id = extract_idempotency_key

# ── Reality 1 & 2: Payload hash (exact-duplicate fallback) ───────────────────

def compute_payload_hash(payload: dict) -> str:
    """
    SHA-256 of the canonical (sorted-keys, UTF-8) JSON representation.

    Catches cases where the vendor sends the exact same bytes multiple times
    but does not include a stable event ID field.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_dt(val: object) -> datetime | None:
    """
    Coerce a value from the LLM's normalized dict to a datetime, or None.

    The LLM returns the `normalized` payload as a plain dict[str, Any], so
    datetime-typed fields (e.g. `due_at`) arrive as ISO 8601 strings rather
    than datetime objects. This function handles that transparently so the DB
    layer always receives proper datetime instances.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        logger.warning("Could not parse datetime value: %r", val)
        return None


def _as_utc(dt: datetime) -> datetime:
    """Normalise to UTC; treat naive datetimes (SQLite) as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_more_recent(
    new_occurred_at: datetime | None,
    current_occurred_at: datetime | None,
    new_status: str | None,
    current_status: str | None,
    state_order: dict[str, int],
) -> bool:
    """
    Python-level check used only for the INSERT path (no existing row).
    The UPDATE path uses an atomic SQL WHERE clause instead.
    """
    if new_occurred_at and current_occurred_at:
        return _as_utc(new_occurred_at) > _as_utc(current_occurred_at)
    if new_occurred_at and not current_occurred_at:
        return True
    if not new_occurred_at and current_occurred_at:
        return False
    new_rank = state_order.get(new_status or "", -1)
    cur_rank = state_order.get(current_status or "", -1)
    return new_rank > cur_rank


# ── Reality 3: Atomic, event-time-ordered entity upserts ─────────────────────

async def _upsert_shipment(event: WebhookEvent, normalized: LLMNormalizedEvent, db: AsyncSession) -> None:
    n = normalized.normalized or {}

    result = await db.execute(select(Shipment).where(Shipment.entity_id == normalized.entity_id))
    shipment = result.scalar_one_or_none()

    if shipment is None:
        # First event for this entity — insert with whatever state we have.
        db.add(Shipment(
            entity_id=normalized.entity_id,
            current_status=normalized.status,
            status_occurred_at=normalized.occurred_at,
            carrier=n.get("carrier"),
            container_id=n.get("container_id"),
            bl_number=n.get("bl_number"),
            origin_port=n.get("origin_port"),
            destination_port=n.get("destination_port"),
            vessel=n.get("vessel"),
            consignee=n.get("consignee"),
            last_event_id=event.id,
        ))
    else:
        # Entity already exists.
        #
        # Reality 3 fix: advance current_status ONLY if this event is
        # chronologically later than what is already stored.
        # The WHERE clause makes this check atomic inside the DB transaction —
        # no race condition even under concurrent writes.
        #
        # Example: IN_TRANSIT (Apr 21) is already stored.
        # PICKED_UP (Apr 19) arrives late.
        # WHERE status_occurred_at < '2026-04-19...' → False → 0 rows updated.
        # State stays IN_TRANSIT. ✓
        if normalized.occurred_at:
            await db.execute(
                update(Shipment)
                .where(
                    Shipment.entity_id == normalized.entity_id,
                    or_(
                        Shipment.status_occurred_at.is_(None),
                        Shipment.status_occurred_at < normalized.occurred_at,
                    ),
                )
                .values(
                    current_status=normalized.status,
                    status_occurred_at=normalized.occurred_at,
                    last_event_id=event.id,
                )
                .execution_options(synchronize_session="fetch")
            )
        else:
            # No timestamp: fall back to lifecycle-order tiebreaker.
            if _is_more_recent(None, shipment.status_occurred_at, normalized.status, shipment.current_status, _SHIPMENT_ORDER):
                shipment.current_status = normalized.status
                shipment.last_event_id = event.id

        # Always enrich with any newly available fields (never overwrite with null).
        await db.execute(
            update(Shipment)
            .where(Shipment.entity_id == normalized.entity_id)
            .values(**{
                field: n[field]
                for field in ("carrier", "container_id", "bl_number", "origin_port", "destination_port", "vessel", "consignee")
                if n.get(field) is not None
            })
            .execution_options(synchronize_session="fetch")
        )


async def _upsert_invoice(event: WebhookEvent, normalized: LLMNormalizedEvent, db: AsyncSession) -> None:
    n = normalized.normalized or {}

    result = await db.execute(select(Invoice).where(Invoice.entity_id == normalized.entity_id))
    invoice = result.scalar_one_or_none()

    if invoice is None:
        db.add(Invoice(
            entity_id=normalized.entity_id,
            current_status=normalized.status,
            status_occurred_at=normalized.occurred_at,
            invoice_ref=n.get("invoice_ref"),
            carrier=n.get("carrier"),
            amount=n.get("amount"),
            currency=n.get("currency"),
            related_bl=n.get("related_bl"),
            due_at=_coerce_dt(n.get("due_at")),   # LLM returns ISO string; coerce to datetime
            remitter=n.get("remitter"),
            last_event_id=event.id,
        ))
    else:
        if normalized.occurred_at:
            await db.execute(
                update(Invoice)
                .where(
                    Invoice.entity_id == normalized.entity_id,
                    or_(
                        Invoice.status_occurred_at.is_(None),
                        Invoice.status_occurred_at < normalized.occurred_at,
                    ),
                )
                .values(
                    current_status=normalized.status,
                    status_occurred_at=normalized.occurred_at,
                    last_event_id=event.id,
                )
                .execution_options(synchronize_session="fetch")
            )
        else:
            if _is_more_recent(None, invoice.status_occurred_at, normalized.status, invoice.current_status, _INVOICE_ORDER):
                invoice.current_status = normalized.status
                invoice.last_event_id = event.id

        # Build the enrichment dict; due_at needs datetime coercion since the
        # LLM returns it as an ISO string inside the normalized dict.
        enrich = {
            field: n[field]
            for field in ("invoice_ref", "carrier", "amount", "currency", "related_bl", "remitter")
            if n.get(field) is not None
        }
        if n.get("due_at") is not None:
            enrich["due_at"] = _coerce_dt(n["due_at"])
        if enrich:
            await db.execute(
                update(Invoice)
                .where(Invoice.entity_id == normalized.entity_id)
                .values(**enrich)
                .execution_options(synchronize_session="fetch")
            )


# ── Background worker ─────────────────────────────────────────────────────────

async def process_event(event_id: uuid.UUID) -> None:
    """
    Normalize a stored webhook event and upsert the corresponding entity.

    Called via FastAPI BackgroundTasks — runs AFTER the 202 response is sent,
    so the vendor never waits for this (Reality 1).
    """
    async with async_session_factory() as db:
        event = await db.get(WebhookEvent, event_id)
        if not event:
            logger.error("process_event: event %s not found", event_id)
            return

        try:
            normalized = await normalize_payload(event.raw_payload)

            event.event_type = normalized.event_type
            event.entity_id = normalized.entity_id
            # Prefer the LLM's extracted idempotency_key if we didn't get one at ingestion.
            # (The LLM can often parse an idempotency key from deeply nested fields that
            # extract_idempotency_key() doesn't know about.)
            if normalized.idempotency_key and not event.vendor_event_id:
                event.vendor_event_id = normalized.idempotency_key
            event.status = normalized.status
            event.occurred_at = normalized.occurred_at
            event.normalized_data = normalized.model_dump(mode="json")
            event.processing_status = "done"

            if normalized.event_type == "shipment" and normalized.entity_id:
                await _upsert_shipment(event, normalized, db)
            elif normalized.event_type == "invoice" and normalized.entity_id:
                await _upsert_invoice(event, normalized, db)

            await db.commit()
            logger.info(
                "Processed event %s → %s %s [%s]",
                event_id, normalized.event_type, normalized.entity_id, normalized.status,
            )

        except Exception as exc:
            await db.rollback()
            try:
                event.processing_status = "failed"
                event.error_message = str(exc)[:2000]
                await db.commit()
            except Exception:
                pass
            logger.exception("Failed to process event %s", event_id)
