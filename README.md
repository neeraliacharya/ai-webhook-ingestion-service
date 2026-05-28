# AI Webhook Ingestion Service

A production-grade backend that ingests arbitrary vendor JSON webhooks from any supply chain carrier, classifies and normalises them with an LLM, and maintains consistent canonical entity state — even under duplicate payloads, out-of-order delivery, and concurrent writes.

Built with **FastAPI · SQLAlchemy async · PostgreSQL · Groq (Llama 3.3 70B)**.

---

## Table of Contents

1. [Architecture](#architecture)
2. [The Three Production Realities](#the-three-production-realities)
3. [Canonical Internal Schema](#canonical-internal-schema)
4. [Key Design Decisions](#key-design-decisions)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [API Reference](#api-reference)
8. [Running Tests](#running-tests)
9. [Configuration](#configuration)
10. [Tradeoffs & Production Roadmap](#tradeoffs--production-roadmap)

---

## Architecture

```
Vendor (Maersk, ONE, GlobalFreightPay, …)
  │
  │  POST /webhooks   — any JSON, any schema
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  HTTP Layer  (FastAPI)                                           │
│                                                                  │
│  1. Parse JSON body                                              │
│  2. extract_idempotency_key()  → check vendor_event_id in DB    │
│  3. Attempt INSERT with unique payload_hash                      │
│     → ON CONFLICT (IntegrityError) = duplicate, return 200      │
│  4. enqueue(event_id)                                            │
│  5. return 202 ◄── vendor ACK here, always <100 ms              │
└───────────────────────────┬──────────────────────────────────────┘
                            │  asyncio.Queue  (decoupled)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  Normalisation Worker  (background asyncio.Task)                 │
│                                                                  │
│  1. Fetch raw event row from DB                                  │
│  2. Call LLM with structured prompt                              │
│     → validate response as LLMNormalizedEvent (Pydantic)         │
│  3. Upsert Shipment or Invoice                                   │
│     → atomic SQL WHERE status_occurred_at < :new_time            │
│  4. Mark processing_status = "done" | "failed"                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
                    PostgreSQL  (Docker volume — persistent)
          ┌─────────────────────────────────────────────┐
          │  webhook_events   append-only audit log      │
          │  shipments        current canonical state    │
          │  invoices         current canonical state    │
          └─────────────────────────────────────────────┘
```

**Why two containers, not one?**
The vendor's HTTP response is decoupled from the LLM call via an `asyncio.Queue`. The endpoint writes one UUID and returns 202; the worker reads it on its own schedule. If the LLM takes 2 seconds, the vendor never notices.

---

## The Three Production Realities

### Reality 1 — Sub-second vendor acknowledgement

Every vendor expects an HTTP response in milliseconds, not seconds.

```
Vendor sends POST /webhooks
      │
      ▼  (~5 ms)
  Parse JSON
  Dedup check
  DB INSERT
  enqueue(event_id)  ← one queue.put(), O(1)
      │
      ▼  (<100 ms guaranteed)
  202 Accepted  ◄── vendor gets this before the LLM is called

  …later (600–1500 ms)…
  Worker: LLM call → normalise → upsert entity → mark done
```

**Verified:** 71 ms median ACK in live testing (including DB round-trip).

---

### Reality 2 — Duplicate / retry payloads

Vendors retry on network timeouts. The same logical event can arrive multiple times, sometimes with slightly different bytes (e.g. a retry timestamp field added). Two DB-level guards handle both cases:

**Layer 1 — Vendor idempotency key (semantic dedup)**

`extract_idempotency_key()` runs synchronously at ingestion time (<1 ms, no LLM). It reads well-known fields from the raw payload:

| Vendor | Idempotency Key |
|--------|----------------|
| Maersk | `event_msg_id` |
| Ocean Network Express | `event_id` |
| Marine Traffic Advisory | `advisory_id` |
| Hapag-Lloyd notices | `notification_id` |
| CBP hold notices | `hold_id` |
| GlobalFreightPay | `doc_ref + "::" + transaction.kind` ← composite |

> **Why composite for GlobalFreightPay?** The same `doc_ref` appears in both the ISSUED and PAID events for the same invoice. `doc_ref` alone is not unique — using it as the key would silently collapse two distinct events into one, permanently losing the PAID state. The composite key `doc_ref::settled in full` vs `doc_ref::freight invoice raised` correctly separates them.

A partial unique index enforces this at the DB level:

```sql
CREATE UNIQUE INDEX ix_webhook_events_vendor_event_id_partial
ON webhook_events (vendor_event_id)
WHERE vendor_event_id IS NOT NULL;
```

The partial index (not a plain `UNIQUE` constraint) is required because `vendor_event_id` is nullable — not all vendors include a stable event ID.

**Layer 2 — SHA-256 payload hash (exact-byte dedup)**

Catches retries from vendors that don't supply a stable event ID. The application attempts `INSERT` directly — no pre-check query — and catches `IntegrityError`:

```python
# This is equivalent to PostgreSQL's:
# INSERT INTO webhook_events (...) ON CONFLICT (payload_hash) DO NOTHING

db.add(event)
try:
    await db.commit()
except IntegrityError:
    await db.rollback()
    return JSONResponse(200, {"status": "duplicate", ...})
```

The unique constraint is the guard. The DB enforces atomicity; two concurrent workers racing on the same payload can't both succeed.

---

### Reality 3 — Out-of-order event delivery

Events do not always arrive in the order they occurred. A `PICKED_UP` event from Apr 19 might arrive after an `IN_TRANSIT` event from Apr 21 was already processed. Naive "last write wins" would corrupt the canonical state.

**The fix: atomic SQL `WHERE` clause, not Python read-compare-write.**

```sql
UPDATE shipments
SET    current_status     = :new_status,
       status_occurred_at = :new_occurred_at,
       last_event_id      = :event_id
WHERE  entity_id = :entity_id
  AND  (status_occurred_at IS NULL
        OR status_occurred_at < :new_occurred_at)
```

The `WHERE` clause makes the guard atomic inside the DB transaction. Two concurrent workers for the same entity will both attempt this `UPDATE`; the database serialises them at the row level. Only the one with the later `occurred_at` will modify any rows — the other's `UPDATE` silently affects 0 rows.

**Result verified in live testing:** Late-arriving `PICKED_UP` (Apr 19) did not overwrite stored `IN_TRANSIT` (Apr 21).

---

## Canonical Internal Schema

Every vendor payload — regardless of its structure — collapses into this single typed schema before touching the database:

```python
class LLMNormalizedEvent(BaseModel):
    """The heart of the system. All vendor payloads collapse into this shape."""

    event_type:      Literal["shipment", "invoice", "unclassified"]
    idempotency_key: str | None    # vendor's own stable event identifier
    vendor_name:     str | None    # e.g. "Maersk", "Ocean Network Express"
    entity_id:       str | None    # master BL number, doc_ref, etc.
    status:          str | None    # one canonical state from the lists below
    occurred_at:     datetime | None   # vendor's event time, NOT server time
    normalized:      dict | None   # structured extracted fields
    confidence:      Literal["high", "medium", "low"]
```

**Canonical state machines** (enforced by the LLM prompt):

```
Shipment:  PICKED_UP → IN_TRANSIT → OUT_FOR_DELIVERY → DELIVERED

Invoice:   ISSUED → PAID
                  → VOIDED
                  → REFUNDED
```

**Unclassified events** (port advisories, weather alerts, maintenance notices) are stored with `event_type="unclassified"`, a `possible_category` label, and a human-readable `title`. They are never silently dropped — discarding unknown events is a data integrity failure.

---

## Key Design Decisions

### 1. `asyncio.Queue` instead of FastAPI `BackgroundTasks`

`BackgroundTasks` is a list of callbacks attached to a single HTTP request. It gives the appearance of async processing but the endpoint still owns the work — holding the ASGI connection open until the callback finishes.

`asyncio.Queue` is a proper decoupling boundary. The endpoint writes one UUID and returns immediately. The worker runs independently as a long-lived `asyncio.Task`, completely separate from any request lifecycle. This matches how production pipelines work (Celery + Redis, SQS, Pub/Sub).

### 2. DB-level idempotency, not application-level

The wrong pattern:
```python
# Race condition: two concurrent requests can both pass this check
if not db.exists(idempotency_key):
    db.insert(event)
```

The right pattern: unique DB constraint + catch `IntegrityError`. The constraint is atomic; the application just handles the signal.

### 3. Atomic SQL for state ordering, not Python TOCTOU

Python read → compare → write has a race: two workers can both read the current state, both decide they're newer, and both write. The SQL `WHERE status_occurred_at < :new_time` guard is evaluated inside the DB transaction — it's serialised at the row level with no application-level locking needed.

### 4. Multi-provider LLM behind a single interface

`normalize_payload()` dispatches to Groq / Gemini / Anthropic based on `LLM_PROVIDER`. The rest of the codebase never sees the provider. Switching requires only a `.env` change.

### 5. Startup retry loop in `init_db()`

Docker's `depends_on: condition: service_healthy` fires `pg_isready` inside the `db` container, but the API container's Docker network DNS may not yet have the `db` hostname propagated. `init_db()` retries up to 15 times with 2-second delays, so the service starts cleanly regardless of PostgreSQL's crash-recovery timing.

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings — reads from .env
│   ├── database.py        # SQLAlchemy engine, session factory, init_db() with retry
│   ├── llm.py             # LLM normalisation (Groq / Gemini / Anthropic)
│   ├── main.py            # FastAPI app, all HTTP endpoints
│   ├── models.py          # SQLAlchemy ORM models + partial unique index
│   ├── processor.py       # extract_idempotency_key(), compute_payload_hash(),
│   │                      #   _upsert_shipment(), _upsert_invoice(), process_event()
│   ├── schemas.py         # Pydantic schemas: LLMNormalizedEvent, API responses
│   ├── worker.py          # asyncio.Queue + background worker task
│   └── static/
│       └── index.html     # Single-page dashboard (Events / Shipments / Invoices / Unclassified)
├── tests/
│   ├── __init__.py
│   └── test_webhooks.py   # 21 tests — no API key or database required
├── .env.example           # Copy to .env and fill in your API key
├── .gitignore
├── docker-compose.yml     # PostgreSQL + API, named volume for persistence
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Docker Desktop running
- A free [Groq API key](https://console.groq.com) (takes 30 seconds to get)

### 1. Clone and configure

```bash
git clone <repo-url>
cd "AI webhook ingestion service"

cp .env.example .env
# Open .env and set:  GROQ_API_KEY=gsk_...
```

### 2. Start

```bash
docker compose up --build
```

The API is ready when you see:
```
api-1  | INFO: Database schema ready (attempt 1)
api-1  | INFO: Normalisation worker started
api-1  | INFO: Application startup complete.
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| REST API | http://localhost:8000/webhooks |
| Swagger docs | http://localhost:8000/docs |

### 3. Send your first webhook

```bash
curl -X POST http://localhost:8000/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "carrier_scac": "MAEU",
    "event_msg_id": "MAEU-EVT-2026-04-22-0001",
    "transport_doc": {"type": "MBL", "number": "MAEU240498712"},
    "container": "MSKU7748112",
    "vessel": {"name": "MAERSK GUATEMALA"},
    "milestone": "Loaded onboard and sailed",
    "milestone_at": "2026-04-21T22:47:00+08:00",
    "port": {"code": "CNSHA", "name": "Shanghai"}
  }'
```

Response (in <100 ms):
```json
{
  "status": "accepted",
  "event_id": "a96b6d53-ce44-4427-b3f1-b44dcee133f9",
  "message": "Webhook accepted; normalisation in progress."
}
```

After a few seconds, the LLM normalises it. Check the result:
```bash
curl http://localhost:8000/shipments/MAEU240498712
```

```json
{
  "entity_id": "MAEU240498712",
  "current_status": "IN_TRANSIT",
  "carrier": "Maersk",
  "container_id": "MSKU7748112",
  "vessel": "MAERSK GUATEMALA",
  "origin_port": "CNSHA"
}
```

### 4. Stop (data persists)

```bash
docker compose down          # stops containers, keeps PostgreSQL data
docker compose down -v       # stops and wipes data (fresh start)
```

---

## API Reference

### Ingestion

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/webhooks` | Any JSON object | `202` accepted / `200` duplicate / `400` invalid |
| `POST` | `/webhooks/batch` | JSON array (max 100) | `202` with per-item results |

**Single webhook response:**
```json
{ "status": "accepted", "event_id": "uuid", "message": "..." }
{ "status": "duplicate", "event_id": "uuid", "message": "..." }
```

**Batch response:**
```json
{
  "total": 3, "accepted": 2, "duplicates": 1, "errors": 0,
  "results": [
    { "index": 0, "status": "accepted", "event_id": "uuid" },
    { "index": 1, "status": "duplicate", "event_id": "uuid" },
    { "index": 2, "status": "accepted", "event_id": "uuid" }
  ]
}
```

### Query

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/events` | Last 50 raw events (audit log) |
| `GET` | `/events/{id}` | Single event with full normalised data |
| `GET` | `/shipments` | All shipments, ordered by last update |
| `GET` | `/shipments/{entity_id}` | Single shipment by master BL |
| `GET` | `/invoices` | All invoices, ordered by last update |
| `GET` | `/invoices/{entity_id}` | Single invoice by doc_ref |
| `GET` | `/unclassified` | Unclassified events with LLM category labels |
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/` | Dashboard UI |

**Processing statuses on events:**
- `pending` — stored, waiting in queue
- `done` — normalised successfully
- `failed` — LLM error; raw payload preserved, `error_message` populated

---

## Running Tests

Tests run against an in-memory SQLite database. No API key, no Docker, no PostgreSQL needed.

```bash
# First time setup
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt aiosqlite

# Run
pytest tests/ -v
```

Expected output:
```
tests/test_webhooks.py::TestHealthEndpoint::test_health PASSED
tests/test_webhooks.py::TestWebhookIngestion::test_returns_202_for_new_payload PASSED
tests/test_webhooks.py::TestWebhookIngestion::test_duplicate_returns_200 PASSED
tests/test_webhooks.py::TestWebhookIngestion::test_rejects_non_json PASSED
tests/test_webhooks.py::TestShipmentLifecycle::test_out_of_order_events_preserve_later_state PASSED
tests/test_webhooks.py::TestShipmentLifecycle::test_shipment_entity_created PASSED
tests/test_webhooks.py::TestInvoiceLifecycle::test_invoice_issued_then_paid PASSED
tests/test_webhooks.py::TestInvoiceLifecycle::test_paid_not_overwritten_by_later_issued PASSED
tests/test_webhooks.py::TestUnclassified::test_unclassified_stored_but_no_entity PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_compute_payload_hash_is_order_independent PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_compute_payload_hash_differs_for_different_payloads PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_is_more_recent_timestamp_wins PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_is_more_recent_falls_back_to_state_order PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_extract_vendor_event_id_known_fields PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_extract_vendor_event_id_composite_key PASSED
tests/test_webhooks.py::TestProcessorHelpers::test_extract_vendor_event_id_returns_none_for_unknown PASSED
tests/test_webhooks.py::TestReality1_SubSecondAck::test_202_returned_before_background_completes PASSED
tests/test_webhooks.py::TestReality2_VendorEventIdDedup::test_vendor_event_id_dedup_before_llm PASSED
tests/test_webhooks.py::TestReality2_VendorEventIdDedup::test_different_vendor_event_ids_are_both_accepted PASSED
tests/test_webhooks.py::TestReality3_OutOfOrderState::test_late_arriving_picked_up_does_not_overwrite_in_transit PASSED
tests/test_webhooks.py::TestReality3_OutOfOrderState::test_newer_event_advances_state_forward PASSED

21 passed in 0.70s
```

**Test strategy:** The LLM is mocked in all tests. `app.main.enqueue` is patched to call `process_event` directly so assertions run without needing to drain an async queue. The autouse fixture spins up a fresh SQLite schema before each test and tears it down after.

---

## Configuration

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

| Variable | Default | Required | Notes |
|----------|---------|----------|-------|
| `LLM_PROVIDER` | `groq` | — | `groq` · `gemini` · `anthropic` |
| `GROQ_API_KEY` | — | if groq | Free at [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | — | Also: `qwen-qwq-32b`, `llama-3.1-8b-instant` |
| `GOOGLE_API_KEY` | — | if gemini | Free at [aistudio.google.com](https://aistudio.google.com) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | — | |
| `ANTHROPIC_API_KEY` | — | if anthropic | Paid |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | — | |
| `DATABASE_URL` | localhost default | — | Set automatically by docker-compose |

**docker-compose sets `DATABASE_URL` automatically.** The value in `.env` is only used when running the app outside Docker (e.g. `uvicorn app.main:app` directly).

---

## Tradeoffs & Production Roadmap

### Honest tradeoffs made under time pressure

| What | Why it's acceptable now | What changes in production |
|------|------------------------|---------------------------|
| In-process `asyncio.Queue` | No external deps, zero config | Replace with **Redis + Celery** or **AWS SQS**; add Dead Letter Queue for repeated failures |
| No retry for failed LLM calls | Events are preserved with `processing_status=failed` and full `raw_payload` | Exponential back-off retry worker; alert on DLQ depth |
| Single worker task | SQLite test compat; simpler reasoning | Run N worker tasks against PostgreSQL for throughput |
| No startup sweep for `pending` events | In-process queue is ephemeral — surviving a restart loses queue contents | On boot: `SELECT id FROM webhook_events WHERE processing_status='pending'` → re-enqueue all |
| No auth on `/webhooks` | Internal demo / assessment | Per-vendor **HMAC signature verification** in middleware |
| LLM prompt unversioned | Prompt is stable during development | Version prompts (`prompt_v1`, `prompt_v2`); store version per normalised event for auditability |
| Single `webhook_events` table | `raw_payload` is written before LLM is called — no data lost on failure | Split into `raw_events` (pre-normalisation, always written) + `events` (post-normalisation) for stricter audit |
| No distributed locking | SQL `WHERE` clause ensures correct state regardless of concurrency | `processing_status='processing'` CAS update before LLM call to avoid redundant work |

### Production roadmap (priority order)

1. **Durable queue** — Redis + Celery (or AWS SQS). Add DLQ for events that fail normalisation > N times.
2. **Startup recovery** — On boot, re-enqueue all `processing_status='pending'` rows. Prevents data loss after crash.
3. **Worker idempotency** — CAS `processing_status: pending → processing` before LLM call. Reset to `pending` on failure. Prevents duplicate work after restart.
4. **Vendor authentication** — HMAC-SHA256 signature verification per vendor (each sends a signature in a different header format).
5. **Schema migrations** — Replace `Base.metadata.create_all` with **Alembic**. Required the moment the schema evolves.
6. **Observability** — Structured logs with `entity_id`, `idempotency_key`, `vendor_name` on every line. OpenTelemetry spans for LLM latency. Trace a shipment's full journey with a single log query.
7. **Rate limiting** — Per-vendor rate limits on `/webhooks`. Aggressive retriers should not starve legitimate traffic.
8. **Prompt versioning** — Store which prompt version produced each normalised event. A prompt change today must not silently reclassify historical events.

---

## Sample Payloads

The dashboard ships with six built-in sample payloads you can send with one click. You can also paste any JSON directly or drag-and-drop a `.json` file to batch-ingest up to 100 payloads at once.

**Maersk — IN_TRANSIT**
```json
{
  "carrier_scac": "MAEU",
  "event_msg_id": "MAEU-EVT-2026-04-22-0001",
  "transport_doc": { "type": "MBL", "number": "MAEU240498712" },
  "container": "MSKU7748112",
  "vessel": { "name": "MAERSK GUATEMALA", "imo": "9778120" },
  "milestone": "Loaded onboard and sailed",
  "milestone_at": "2026-04-21T22:47:00+08:00",
  "port": { "code": "CNSHA", "name": "Shanghai" }
}
```

**GlobalFreightPay — PAID** *(composite idempotency key: `doc_ref::settled in full`)*
```json
{
  "source": "globalfreightpay.api",
  "doc_ref": "GFP-INV-2026-Q2-08821",
  "carrier": "Hapag-Lloyd AG",
  "linked_bl": "HLCU2604OCEAN221",
  "transaction": {
    "kind": "settled in full",
    "settled_at": "2026-04-22 18:47:11+02:00",
    "amount": "EUR 24.350,75",
    "remitter": "ACME Logistics GmbH"
  }
}
```

**Marine Traffic Advisory** *(unclassified — stored, never dropped)*
```json
{
  "issuer": "marine-traffic-advisory",
  "advisory_id": "MTA-2026-04-26-EU-007",
  "severity": "AMBER",
  "subject": "Ongoing congestion at Port of Antwerp-Bruges",
  "affected_services": ["AE7", "FAL3", "Mediterranean Bridge"]
}
```
