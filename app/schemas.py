import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

# ── Canonical state types ─────────────────────────────────────────────────────
# All vendor-specific milestone strings collapse into one of these values.
# Defined as a type alias so they can be referenced by the state machine,
# the DB layer, and the API response models from a single source of truth.

CanonicalShipmentState = Literal[
    "PICKED_UP",
    "IN_TRANSIT",
    "OUT_FOR_DELIVERY",
    "DELIVERED",
]

CanonicalInvoiceState = Literal[
    "ISSUED",
    "PAID",
    "VOIDED",
    "REFUNDED",
]


class NormalizedShipmentData(BaseModel):
    carrier: str | None = None
    container_id: str | None = None
    bl_number: str | None = None
    origin_port: str | None = None
    destination_port: str | None = None
    vessel: str | None = None
    consignee: str | None = None


class NormalizedInvoiceData(BaseModel):
    invoice_ref: str | None = None
    carrier: str | None = None
    amount: float | None = None
    currency: str | None = None
    related_bl: str | None = None
    due_at: datetime | None = None
    remitter: str | None = None


class LLMNormalizedEvent(BaseModel):
    """
    The canonical internal schema that every vendor payload collapses into.

    This is the heart of the system. No matter how different Maersk's JSON
    looks from GlobalFreightPay's, the LLM always returns this exact shape.
    Downstream code (processor, DB layer, API) only ever sees this type.

    Field design rationale
    ──────────────────────
    idempotency_key  — the vendor's *own* stable event identifier (not our UUID).
                       Used for semantic deduplication before any LLM call.
                       Distinct from our internal event_id, which is always a
                       system-generated UUID.

    vendor_name      — extracted by the LLM from each payload's carrier/issuer
                       fields so downstream consumers don't need vendor-specific
                       parsing logic.

    status           — always one of the CanonicalShipmentState or
                       CanonicalInvoiceState values above, or null for
                       unclassified events.

    occurred_at      — the vendor's event timestamp, NOT our server time.
                       Using the vendor's time (not arrival order) is what
                       makes out-of-order delivery safe.
    """

    event_type: Literal["shipment", "invoice", "unclassified"]
    idempotency_key: str | None = None    # vendor's own stable event identifier
    vendor_name: str | None = None        # e.g. "Maersk", "Ocean Network Express"
    entity_id: str | None = None          # master BL, doc_ref, etc.
    status: str | None = None             # one of the canonical states above
    occurred_at: datetime | None = None   # vendor's event time, not server time
    normalized: dict[str, Any] | None = None
    # Unclassified-specific fields
    possible_category: str | None = None  # e.g. "port advisory", "weather alert"
    title: str | None = None              # human-readable summary of the event
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: str | None = None


# ── API response models ───────────────────────────────────────────────────────

class WebhookIngestionResponse(BaseModel):
    status: Literal["accepted", "duplicate"]
    event_id: uuid.UUID
    message: str


class EventDetail(BaseModel):
    id: uuid.UUID
    received_at: datetime
    event_type: str
    entity_id: str | None
    status: str | None
    occurred_at: datetime | None
    normalized_data: dict | None
    processing_status: str
    error_message: str | None

    model_config = {"from_attributes": True}


class ShipmentDetail(BaseModel):
    id: uuid.UUID
    entity_id: str
    current_status: str
    status_occurred_at: datetime | None
    carrier: str | None
    container_id: str | None
    bl_number: str | None
    origin_port: str | None
    destination_port: str | None
    vessel: str | None
    consignee: str | None
    updated_at: datetime

    model_config = {"from_attributes": True}


class InvoiceDetail(BaseModel):
    id: uuid.UUID
    entity_id: str
    current_status: str
    status_occurred_at: datetime | None
    invoice_ref: str | None
    carrier: str | None
    amount: float | None
    currency: str | None
    related_bl: str | None
    due_at: datetime | None
    remitter: str | None
    updated_at: datetime

    model_config = {"from_attributes": True}


class BatchResult(BaseModel):
    index: int
    status: str          # "accepted" | "duplicate" | "error"
    event_id: str | None = None
    error: str | None = None


class BatchIngestionResponse(BaseModel):
    total: int
    accepted: int
    duplicates: int
    errors: int
    results: list[BatchResult]


class UnclassifiedDetail(BaseModel):
    """An event that could not be mapped to a shipment or invoice."""

    id: uuid.UUID
    received_at: datetime
    occurred_at: datetime | None
    vendor_event_id: str | None
    possible_category: str | None   # LLM's best guess at a category label
    title: str | None               # human-readable one-liner
    notes: str | None
    processing_status: str
    raw_payload: dict | None

    model_config = {"from_attributes": True}
