import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, Numeric, String, Text, Uuid, func, text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class WebhookEvent(Base):
    """
    Append-only log of every received webhook payload.

    Two idempotency layers (both DB-level):
    ────────────────────────────────────────
    1. payload_hash UNIQUE — catches byte-exact duplicate payloads.
       The application tries INSERT; if it violates this constraint the DB
       raises IntegrityError (equivalent to PostgreSQL's ON CONFLICT DO NOTHING).
       No pre-check query; the constraint is the only guard.

    2. ix_webhook_events_vendor_event_id_partial — partial unique index on
       vendor_event_id WHERE vendor_event_id IS NOT NULL.
       Catches semantic duplicates (same vendor event, slightly different bytes).
       vendor_event_id is nullable (not all vendors supply one), so a simple
       UNIQUE constraint would wrongly block multiple NULL rows — a partial
       index is the correct DB-level primitive here.
    """

    __tablename__ = "webhook_events"
    __table_args__ = (
        # Partial unique index on vendor_event_id — only enforced when non-null.
        # Equivalent SQL:
        #   CREATE UNIQUE INDEX ix_webhook_events_vendor_event_id_partial
        #   ON webhook_events (vendor_event_id)
        #   WHERE vendor_event_id IS NOT NULL;
        Index(
            "ix_webhook_events_vendor_event_id_partial",
            "vendor_event_id",
            unique=True,
            postgresql_where=text("vendor_event_id IS NOT NULL"),
            sqlite_where=text("vendor_event_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True, default=uuid.uuid4)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Layer 1 dedup: SHA-256 of canonical (sorted-keys) JSON.
    # unique=True creates the DB constraint; IntegrityError is the application's signal.
    payload_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Filled in by the background normaliser
    event_type: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    entity_id: Mapped[str | None] = mapped_column(String(200), index=True)
    # Layer 2 dedup: vendor's own stable event identifier (idempotency_key in the
    # canonical schema). Stored at ingestion time before any LLM call.
    vendor_event_id: Mapped[str | None] = mapped_column(String(200))  # indexed via __table_args__
    status: Mapped[str | None] = mapped_column(String(50))
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    normalized_data: Mapped[dict | None] = mapped_column(JSON)

    processing_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Shipment(Base):
    """Current canonical state of a shipment entity."""

    __tablename__ = "shipments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    current_status: Mapped[str] = mapped_column(String(50), nullable=False)
    status_occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    carrier: Mapped[str | None] = mapped_column(String(200))
    container_id: Mapped[str | None] = mapped_column(String(50))
    bl_number: Mapped[str | None] = mapped_column(String(100))
    origin_port: Mapped[str | None] = mapped_column(String(100))
    destination_port: Mapped[str | None] = mapped_column(String(100))
    vessel: Mapped[str | None] = mapped_column(String(200))
    consignee: Mapped[str | None] = mapped_column(String(200))

    last_event_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False), ForeignKey("webhook_events.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Invoice(Base):
    """Current canonical state of an invoice entity."""

    __tablename__ = "invoices"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True, native_uuid=False), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    current_status: Mapped[str] = mapped_column(String(50), nullable=False)
    status_occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    invoice_ref: Mapped[str | None] = mapped_column(String(200))
    carrier: Mapped[str | None] = mapped_column(String(200))
    amount: Mapped[float | None] = mapped_column(Numeric(15, 2))
    currency: Mapped[str | None] = mapped_column(String(3))
    related_bl: Mapped[str | None] = mapped_column(String(100))
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    remitter: Mapped[str | None] = mapped_column(String(200))

    last_event_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True, native_uuid=False), ForeignKey("webhook_events.id")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
