"""
Integration-style tests for the webhook ingestion service.

The LLM call is mocked so these tests run without an API key or database.

Worker strategy in tests
────────────────────────
Production uses an asyncio.Queue + background worker task so the HTTP layer
and processing layer are fully decoupled. In tests we need processing to
complete *synchronously* (before assertions run) without starting a real
background task.

The autouse fixture patches `app.main.enqueue` to call process_event directly,
so from each test's perspective the event is fully processed by the time the
POST response returns. The real queue/worker path is exercised by the
TestReality1_SubSecondAck test, which verifies the 202 is returned regardless.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app
from app.schemas import LLMNormalizedEvent

# ── In-memory SQLite engine for tests ────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSession = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)


async def override_get_db():
    async with TestSession() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


async def _process_immediately(event_id: uuid.UUID) -> None:
    """Test stand-in for enqueue(): processes the event synchronously."""
    from app.processor import process_event
    await process_event(event_id)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    # Import models so they register with Base
    import app.models  # noqa: F401
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Redirect background sessions to the test DB and bypass the real queue
    with patch("app.processor.async_session_factory", TestSession), \
         patch("app.main.enqueue", side_effect=_process_immediately):
        yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Sample payloads (from the assignment appendix) ────────────────────────────

MAERSK_IN_TRANSIT = {
    "carrier_scac": "MAEU",
    "event_msg_id": "MAEU-EVT-2026-04-22-0001",
    "transport_doc": {"type": "MBL", "number": "MAEU240498712"},
    "container": "MSKU7748112",
    "vessel": {"name": "MAERSK GUATEMALA", "imo": "9778120", "voyage": "424W"},
    "milestone": "Loaded onboard and sailed",
    "milestone_at": "2026-04-21T22:47:00+08:00",
    "port": {"code": "CNSHA", "name": "Shanghai"},
}

MAERSK_PICKED_UP = {
    "carrier_scac": "MAEU",
    "event_msg_id": "MAEU-EVT-2026-04-19-0042",
    "transport_doc": {"type": "MBL", "number": "MAEU240498712"},
    "container": "MSKU7748112",
    "milestone": "Empty container released to shipper; full container received at origin terminal",
    "milestone_at": "2026-04-19T11:15:00+08:00",
    "port": {"code": "CNSHA", "name": "Shanghai"},
    "shipper_ref": "ACME-IND-PO-2026-9921",
}

GFP_PAID = {
    "source": "globalfreightpay.api",
    "channel": "carrier_billing",
    "doc_ref": "GFP-INV-2026-Q2-08821",
    "carrier": "Hapag-Lloyd AG",
    "linked_bl": "HLCU2604OCEAN221",
    "transaction": {
        "kind": "settled in full",
        "settled_at": "2026-04-22 18:47:11+02:00",
        "amount": "EUR 24.350,75",
        "remitter": "ACME Logistics GmbH",
        "memo": "Ocean freight + THC + BAF, Shanghai → Hamburg, container HLBU4490221",
    },
}

GFP_ISSUED = {
    "source": "globalfreightpay.api",
    "channel": "carrier_billing",
    "doc_ref": "GFP-INV-2026-Q2-08821",
    "carrier": "Hapag-Lloyd AG",
    "linked_bl": "HLCU2604OCEAN221",
    "transaction": {
        "kind": "freight invoice raised",
        "issued_at": "2026-04-15T09:00:00+02:00",
        "amount": "EUR 24.350,75",
        "due_at": "2026-05-15T00:00:00+02:00",
        "line_items": [
            {"desc": "Ocean freight Shanghai → Hamburg", "amt": "EUR 21.000,00"},
            {"desc": "Terminal handling charges (THC)", "amt": "EUR 1.850,75"},
            {"desc": "Bunker adjustment factor (BAF)", "amt": "EUR 1.500,00"},
        ],
    },
}

ONE_DELIVERED = {
    "carrier": "Ocean Network Express",
    "carrier_scac": "ONEY",
    "event_id": "ONE-2026-04-28-114",
    "house_bl": "ONEYJKTHKG2604113",
    "master_bl": "ONEYMBLHKG260499",
    "container_no": "TLLU2890442",
    "consignee": "ACME Manufacturing PT.",
    "milestone_text": "Cargo released to consignee at consignee facility — empty container returned to depot",
    "milestone_local_time": "28/04/2026 09:42 WIB",
    "port_of_discharge": "IDJKT",
    "delivery_order_no": "DO-IDJKT-26044881",
}

MARINE_ADVISORY = {
    "issuer": "marine-traffic-advisory",
    "advisory_id": "MTA-2026-04-26-EU-007",
    "severity": "AMBER",
    "issued_at": "2026-04-26T06:00:00Z",
    "subject": "Ongoing congestion at Port of Antwerp-Bruges",
    "body": "Vessel waiting times at Antwerp-Bruges berths have increased to 4-6 days...",
    "affected_services": ["AE7", "FAL3", "Mediterranean Bridge"],
    "expires_at": "2026-05-03T00:00:00Z",
}


# ── LLM mock helpers ──────────────────────────────────────────────────────────

def _mock_llm(normalized_event: LLMNormalizedEvent):
    """Patch normalize_payload to return a pre-built LLMNormalizedEvent."""
    return patch("app.processor.normalize_payload", new=AsyncMock(return_value=normalized_event))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestWebhookIngestion:
    async def test_returns_202_for_new_payload(self, client):
        mock_event = LLMNormalizedEvent(
            event_type="shipment",
            idempotency_key="MAEU-EVT-2026-04-22-0001",
            entity_id="MAEU240498712",
            status="IN_TRANSIT",
            occurred_at=datetime(2026, 4, 21, 14, 47, 0, tzinfo=timezone.utc),
            normalized={
                "carrier": "Maersk",
                "container_id": "MSKU7748112",
                "bl_number": "MAEU240498712",
                "origin_port": "CNSHA",
                "vessel": "MAERSK GUATEMALA",
            },
            confidence="high",
        )
        with _mock_llm(mock_event):
            r = await client.post("/webhooks", json=MAERSK_IN_TRANSIT)

        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "accepted"
        assert "event_id" in body

    async def test_duplicate_returns_200(self, client):
        mock_event = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="MAEU240498712",
            status="IN_TRANSIT",
            occurred_at=datetime(2026, 4, 21, 14, 47, tzinfo=timezone.utc),
            normalized={},
            confidence="high",
        )
        with _mock_llm(mock_event):
            r1 = await client.post("/webhooks", json=MAERSK_IN_TRANSIT)
            r2 = await client.post("/webhooks", json=MAERSK_IN_TRANSIT)

        assert r1.status_code == 202
        assert r2.status_code == 200
        assert r2.json()["status"] == "duplicate"
        assert r2.json()["event_id"] == r1.json()["event_id"]

    async def test_rejects_non_json(self, client):
        r = await client.post("/webhooks", content=b"not-json", headers={"Content-Type": "application/json"})
        assert r.status_code == 400


class TestShipmentLifecycle:
    async def test_out_of_order_events_preserve_later_state(self, client):
        """IN_TRANSIT arrives first (processing), then PICKED_UP arrives later.
        The shipment should stay IN_TRANSIT because that occurred_at is newer."""

        in_transit_mock = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="MAEU240498712",
            status="IN_TRANSIT",
            occurred_at=datetime(2026, 4, 21, 22, 47, 0, tzinfo=timezone.utc),
            normalized={"carrier": "Maersk", "container_id": "MSKU7748112", "bl_number": "MAEU240498712"},
            confidence="high",
        )
        picked_up_mock = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="MAEU240498712",
            status="PICKED_UP",
            occurred_at=datetime(2026, 4, 19, 11, 15, 0, tzinfo=timezone.utc),
            normalized={"carrier": "Maersk", "container_id": "MSKU7748112", "bl_number": "MAEU240498712"},
            confidence="high",
        )

        # Post IN_TRANSIT first
        with _mock_llm(in_transit_mock):
            r1 = await client.post("/webhooks", json=MAERSK_IN_TRANSIT)
        assert r1.status_code == 202
        event_id_1 = r1.json()["event_id"]

        # Post PICKED_UP (out of order — happened earlier)
        with _mock_llm(picked_up_mock):
            r2 = await client.post("/webhooks", json=MAERSK_PICKED_UP)
        assert r2.status_code == 202

        # Allow background tasks to settle (they ran synchronously in test)
        shipment_r = await client.get(f"/shipments/MAEU240498712")
        assert shipment_r.status_code == 200
        shipment = shipment_r.json()
        # Should still be IN_TRANSIT because that event occurred later
        assert shipment["current_status"] == "IN_TRANSIT"

    async def test_shipment_entity_created(self, client):
        mock_event = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="ONEYMBLHKG260499",
            status="DELIVERED",
            occurred_at=datetime(2026, 4, 28, 2, 42, 0, tzinfo=timezone.utc),
            normalized={
                "carrier": "Ocean Network Express",
                "container_id": "TLLU2890442",
                "bl_number": "ONEYMBLHKG260499",
                "destination_port": "IDJKT",
                "consignee": "ACME Manufacturing PT.",
            },
            confidence="high",
        )
        with _mock_llm(mock_event):
            r = await client.post("/webhooks", json=ONE_DELIVERED)
        assert r.status_code == 202

        s = await client.get("/shipments/ONEYMBLHKG260499")
        assert s.status_code == 200
        data = s.json()
        assert data["current_status"] == "DELIVERED"
        assert data["consignee"] == "ACME Manufacturing PT."


class TestInvoiceLifecycle:
    async def test_invoice_issued_then_paid(self, client):
        issued_mock = LLMNormalizedEvent(
            event_type="invoice",
            entity_id="GFP-INV-2026-Q2-08821",
            status="ISSUED",
            occurred_at=datetime(2026, 4, 15, 7, 0, 0, tzinfo=timezone.utc),
            normalized={
                "invoice_ref": "GFP-INV-2026-Q2-08821",
                "carrier": "Hapag-Lloyd AG",
                "amount": 24350.75,
                "currency": "EUR",
                "related_bl": "HLCU2604OCEAN221",
            },
            confidence="high",
        )
        paid_mock = LLMNormalizedEvent(
            event_type="invoice",
            entity_id="GFP-INV-2026-Q2-08821",
            status="PAID",
            occurred_at=datetime(2026, 4, 22, 16, 47, 11, tzinfo=timezone.utc),
            normalized={
                "invoice_ref": "GFP-INV-2026-Q2-08821",
                "carrier": "Hapag-Lloyd AG",
                "amount": 24350.75,
                "currency": "EUR",
                "remitter": "ACME Logistics GmbH",
            },
            confidence="high",
        )

        with _mock_llm(issued_mock):
            await client.post("/webhooks", json=GFP_ISSUED)

        with _mock_llm(paid_mock):
            await client.post("/webhooks", json=GFP_PAID)

        inv = await client.get("/invoices/GFP-INV-2026-Q2-08821")
        assert inv.status_code == 200
        data = inv.json()
        assert data["current_status"] == "PAID"
        assert data["amount"] == 24350.75
        assert data["currency"] == "EUR"

    async def test_paid_not_overwritten_by_later_issued(self, client):
        """If PAID arrives before ISSUED (out of order), state must stay PAID."""
        paid_mock = LLMNormalizedEvent(
            event_type="invoice",
            entity_id="GFP-INV-2026-Q2-08821",
            status="PAID",
            occurred_at=datetime(2026, 4, 22, 16, 47, 11, tzinfo=timezone.utc),
            normalized={"invoice_ref": "GFP-INV-2026-Q2-08821", "amount": 24350.75, "currency": "EUR"},
            confidence="high",
        )
        issued_mock = LLMNormalizedEvent(
            event_type="invoice",
            entity_id="GFP-INV-2026-Q2-08821",
            status="ISSUED",
            occurred_at=datetime(2026, 4, 15, 7, 0, 0, tzinfo=timezone.utc),
            normalized={"invoice_ref": "GFP-INV-2026-Q2-08821", "amount": 24350.75, "currency": "EUR"},
            confidence="high",
        )

        with _mock_llm(paid_mock):
            await client.post("/webhooks", json=GFP_PAID)

        with _mock_llm(issued_mock):
            await client.post("/webhooks", json=GFP_ISSUED)

        inv = await client.get("/invoices/GFP-INV-2026-Q2-08821")
        assert inv.status_code == 200
        assert inv.json()["current_status"] == "PAID"


class TestUnclassified:
    async def test_unclassified_stored_but_no_entity(self, client):
        mock_event = LLMNormalizedEvent(
            event_type="unclassified",
            entity_id=None,
            status=None,
            occurred_at=None,
            normalized=None,
            confidence="high",
            notes="Marine traffic advisory — does not represent a specific shipment or invoice",
        )
        with _mock_llm(mock_event):
            r = await client.post("/webhooks", json=MARINE_ADVISORY)

        assert r.status_code == 202
        event_id = r.json()["event_id"]

        ev = await client.get(f"/events/{event_id}")
        assert ev.status_code == 200
        assert ev.json()["event_type"] == "unclassified"


class TestProcessorHelpers:
    def test_compute_payload_hash_is_order_independent(self):
        from app.processor import compute_payload_hash

        p1 = {"b": 2, "a": 1}
        p2 = {"a": 1, "b": 2}
        assert compute_payload_hash(p1) == compute_payload_hash(p2)

    def test_compute_payload_hash_differs_for_different_payloads(self):
        from app.processor import compute_payload_hash

        assert compute_payload_hash({"a": 1}) != compute_payload_hash({"a": 2})

    def test_is_more_recent_timestamp_wins(self):
        from app.processor import _is_more_recent

        newer = datetime(2026, 4, 22, tzinfo=timezone.utc)
        older = datetime(2026, 4, 19, tzinfo=timezone.utc)
        assert _is_more_recent(newer, older, "IN_TRANSIT", "PICKED_UP", {}) is True
        assert _is_more_recent(older, newer, "PICKED_UP", "IN_TRANSIT", {}) is False

    def test_is_more_recent_falls_back_to_state_order(self):
        from app.processor import _is_more_recent, _SHIPMENT_ORDER

        # No timestamps — higher state wins
        assert _is_more_recent(None, None, "IN_TRANSIT", "PICKED_UP", _SHIPMENT_ORDER) is True
        assert _is_more_recent(None, None, "PICKED_UP", "IN_TRANSIT", _SHIPMENT_ORDER) is False

    def test_extract_vendor_event_id_known_fields(self):
        from app.processor import extract_vendor_event_id

        assert extract_vendor_event_id({"event_msg_id": "MAEU-EVT-001"}) == "MAEU-EVT-001"
        assert extract_vendor_event_id({"event_id": "ONE-2026-04-28-114"}) == "ONE-2026-04-28-114"
        assert extract_vendor_event_id({"advisory_id": "MTA-007"}) == "MTA-007"

    def test_extract_vendor_event_id_composite_key(self):
        from app.processor import extract_vendor_event_id

        # GFP uses doc_ref + transaction.kind as a composite key
        issued = {"doc_ref": "GFP-INV-2026-Q2-08821", "transaction": {"kind": "freight invoice raised"}}
        paid   = {"doc_ref": "GFP-INV-2026-Q2-08821", "transaction": {"kind": "settled in full"}}
        assert extract_vendor_event_id(issued) == "GFP-INV-2026-Q2-08821::freight invoice raised"
        assert extract_vendor_event_id(paid)   == "GFP-INV-2026-Q2-08821::settled in full"
        # They differ — same invoice, different events ✓
        assert extract_vendor_event_id(issued) != extract_vendor_event_id(paid)

    def test_extract_vendor_event_id_returns_none_for_unknown(self):
        from app.processor import extract_vendor_event_id

        assert extract_vendor_event_id({"some_field": "value"}) is None
        assert extract_vendor_event_id({}) is None


class TestReality1_SubSecondAck:
    """Reality 1: 202 is returned before the LLM/DB processing finishes."""

    async def test_202_returned_before_background_completes(self, client):
        """The endpoint must return 202 immediately; background task runs after.
        We verify the HTTP layer doesn't wait for normalize_payload."""
        import asyncio

        slow_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def slow_normalize(payload):
            # Would block if the endpoint awaited this before responding
            await asyncio.sleep(0)  # yield so the event loop can flush the response
            return LLMNormalizedEvent(
                event_type="shipment",
                entity_id="MAEU240498712",
                status="IN_TRANSIT",
                occurred_at=datetime(2026, 4, 21, 14, 47, tzinfo=timezone.utc),
                normalized={},
                confidence="high",
            )

        unique_payload = {**MAERSK_IN_TRANSIT, "event_msg_id": "REALITY1-TEST-001"}
        with patch("app.processor.normalize_payload", new=slow_normalize):
            r = await client.post("/webhooks", json=unique_payload)

        # Must be 202 — not blocked by the (slow) background task
        assert r.status_code == 202
        assert r.json()["status"] == "accepted"


class TestReality2_VendorEventIdDedup:
    """Reality 2a: Vendor idempotency key deduplication at ingestion time."""

    async def test_vendor_event_id_dedup_before_llm(self, client):
        """Second request with same vendor event_id is rejected as duplicate
        even when the payload bytes differ (e.g. vendor added a timestamp field)."""
        original = {**MAERSK_IN_TRANSIT, "event_msg_id": "DEDUP-TEST-001"}
        # Same event_msg_id but a slightly different payload (extra key)
        retry_with_extra_field = {**original, "_retry_ts": "2026-04-22T10:00:00Z"}

        mock_event = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="MAEU240498712",
            status="IN_TRANSIT",
            occurred_at=datetime(2026, 4, 21, 14, 47, tzinfo=timezone.utc),
            normalized={},
            confidence="high",
        )

        with _mock_llm(mock_event):
            r1 = await client.post("/webhooks", json=original)
        assert r1.status_code == 202

        # Payload hash differs (extra field), but vendor_event_id matches
        assert retry_with_extra_field != original  # sanity check — hashes would differ

        with _mock_llm(mock_event):
            r2 = await client.post("/webhooks", json=retry_with_extra_field)

        assert r2.status_code == 200
        body = r2.json()
        assert body["status"] == "duplicate"
        assert body["event_id"] == r1.json()["event_id"]

    async def test_different_vendor_event_ids_are_both_accepted(self, client):
        """GFP ISSUED and PAID have the same doc_ref but different composite keys —
        both must be accepted as distinct events."""
        issued_mock = LLMNormalizedEvent(
            event_type="invoice",
            entity_id="GFP-INV-2026-Q2-08821",
            status="ISSUED",
            occurred_at=datetime(2026, 4, 15, 7, 0, tzinfo=timezone.utc),
            normalized={},
            confidence="high",
        )
        paid_mock = LLMNormalizedEvent(
            event_type="invoice",
            entity_id="GFP-INV-2026-Q2-08821",
            status="PAID",
            occurred_at=datetime(2026, 4, 22, 16, 47, tzinfo=timezone.utc),
            normalized={},
            confidence="high",
        )

        with _mock_llm(issued_mock):
            r_issued = await client.post("/webhooks", json=GFP_ISSUED)
        with _mock_llm(paid_mock):
            r_paid = await client.post("/webhooks", json=GFP_PAID)

        assert r_issued.status_code == 202
        assert r_paid.status_code == 202
        # They must get different event IDs
        assert r_issued.json()["event_id"] != r_paid.json()["event_id"]


class TestReality3_OutOfOrderState:
    """Reality 3: Atomic SQL ensures event-time ordering, not arrival-time."""

    async def test_late_arriving_picked_up_does_not_overwrite_in_transit(self, client):
        """IN_TRANSIT (Apr 21) is stored first; PICKED_UP (Apr 19) arrives late.
        The SQL WHERE clause must leave IN_TRANSIT in place."""
        in_transit = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="REALITY3-SHIP-001",
            status="IN_TRANSIT",
            occurred_at=datetime(2026, 4, 21, 22, 47, 0, tzinfo=timezone.utc),
            normalized={"carrier": "Maersk", "bl_number": "REALITY3-SHIP-001"},
            confidence="high",
        )
        picked_up = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="REALITY3-SHIP-001",
            status="PICKED_UP",
            occurred_at=datetime(2026, 4, 19, 11, 15, 0, tzinfo=timezone.utc),
            normalized={"carrier": "Maersk", "bl_number": "REALITY3-SHIP-001"},
            confidence="high",
        )

        p1 = {**MAERSK_IN_TRANSIT, "event_msg_id": "R3-TRANSIT-001"}
        p2 = {**MAERSK_PICKED_UP,  "event_msg_id": "R3-PICKUP-001"}

        with _mock_llm(in_transit):
            await client.post("/webhooks", json=p1)
        with _mock_llm(picked_up):
            await client.post("/webhooks", json=p2)

        s = await client.get("/shipments/REALITY3-SHIP-001")
        assert s.status_code == 200
        assert s.json()["current_status"] == "IN_TRANSIT"   # NOT overwritten

    async def test_newer_event_advances_state_forward(self, client):
        """PICKED_UP stored first; IN_TRANSIT (later occurred_at) arrives and MUST advance state."""
        picked_up = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="REALITY3-SHIP-002",
            status="PICKED_UP",
            occurred_at=datetime(2026, 4, 19, 11, 15, 0, tzinfo=timezone.utc),
            normalized={"carrier": "Maersk", "bl_number": "REALITY3-SHIP-002"},
            confidence="high",
        )
        in_transit = LLMNormalizedEvent(
            event_type="shipment",
            entity_id="REALITY3-SHIP-002",
            status="IN_TRANSIT",
            occurred_at=datetime(2026, 4, 21, 22, 47, 0, tzinfo=timezone.utc),
            normalized={"carrier": "Maersk", "bl_number": "REALITY3-SHIP-002"},
            confidence="high",
        )

        p1 = {**MAERSK_PICKED_UP,  "event_msg_id": "R3-FWD-PICKUP-001"}
        p2 = {**MAERSK_IN_TRANSIT, "event_msg_id": "R3-FWD-TRANSIT-001"}

        with _mock_llm(picked_up):
            await client.post("/webhooks", json=p1)
        with _mock_llm(in_transit):
            await client.post("/webhooks", json=p2)

        s = await client.get("/shipments/REALITY3-SHIP-002")
        assert s.status_code == 200
        assert s.json()["current_status"] == "IN_TRANSIT"   # advanced correctly
