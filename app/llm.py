"""LLM-based webhook classification and normalization.

Supports multiple backends via LLM_PROVIDER env var:
  - "groq"      (default) — free tier, very fast, OpenAI-compatible
  - "gemini"    — Google Gemini, also free
  - "anthropic" — original Anthropic/Claude (requires paid credits)

Set the matching API key in .env:
  GROQ_API_KEY=gsk_...
  GOOGLE_API_KEY=...
  ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import re

from .config import settings
from .schemas import LLMNormalizedEvent

# ── System prompt (shared across all providers) ──────────────────────────────
_SYSTEM_PROMPT = """You are a webhook normalization engine for a supply chain platform. \
Vendors send JSON payloads in completely different, undocumented structures. \
Your job is to classify each payload and normalize it into a strict internal schema.

## Classification rules
- "shipment"     — any update about a physical parcel or container moving through a logistics network
- "invoice"      — any financial document: invoice raised, payment, settlement, void, or refund
- "unclassified" — anything that does not clearly belong to either of the above

## Canonical shipment states (in lifecycle order)
- PICKED_UP        — container/parcel received at origin; empty container released to shipper; gate-in at terminal
- IN_TRANSIT       — vessel sailed; loaded on board; departed origin; in motion between ports
- OUT_FOR_DELIVERY — last-mile delivery initiated; out for delivery to consignee address
- DELIVERED        — delivered to final recipient; cargo released to consignee; handed to recipient

## Canonical invoice states
- ISSUED    — invoice created, raised, or sent
- PAID      — invoice settled, paid in full, payment received
- VOIDED    — invoice cancelled before payment
- REFUNDED  — payment reversed after settlement

## Output format
Return ONLY a valid JSON object — no markdown, no explanation, no code fences — with exactly this structure:

{
  "event_type": "shipment | invoice | unclassified",
  "idempotency_key": "<vendor's own stable event/message ID if present, else null — e.g. event_msg_id, event_id, advisory_id, notification_id>",
  "vendor_name": "<the name of the vendor or carrier that sent this payload, e.g. 'Maersk', 'Ocean Network Express', 'Hapag-Lloyd AG', 'Marine Traffic Advisory' — derive from carrier fields, issuer, or source>",
  "entity_id": "<the single primary tracking ID that ties events together: use master BL number for shipments, doc_ref for invoices, or the best available identifier>",
  "status": "<one canonical status from the lists above, or null for unclassified>",
  "occurred_at": "<ISO 8601 datetime with timezone, or null>",
  "normalized": {
    "carrier": "<carrier name or null>",
    "container_id": "<container number or null>",
    "bl_number": "<bill of lading number — master BL preferred over house BL — or null>",
    "origin_port": "<UN/LOCODE or human-readable port name, or null>",
    "destination_port": "<UN/LOCODE or human-readable port name, or null>",
    "vessel": "<vessel name or null>",
    "consignee": "<consignee name or null>",
    "invoice_ref": "<invoice reference number or null>",
    "amount": "<numeric float — normalize European formats like '24.350,75' to 24350.75 — or null>",
    "currency": "<ISO 4217 code, e.g. EUR, USD — or null>",
    "related_bl": "<linked bill of lading number or null>",
    "due_at": "<ISO 8601 datetime or null>",
    "remitter": "<paying party name or null>"
  },
  "possible_category": "<for unclassified only — your best single label for this event type, e.g. 'port advisory', 'weather alert', 'regulatory notice', 'carrier announcement', 'customs hold', 'system notification' — or null for shipment/invoice>",
  "title": "<for unclassified only — a short human-readable one-liner summarising what this event is about, e.g. 'Port congestion at Antwerp-Bruges (AMBER)' — or null for shipment/invoice>",
  "confidence": "high | medium | low",
  "notes": "<brief notes on ambiguities, assumptions, or missing data — or null>"
}

## Important rules
- Convert all datetime formats to ISO 8601 with timezone offset (e.g. WIB = +07:00)
- European amounts like '24.350,75' → 24350.75 (period = thousands separator, comma = decimal)
- When both house BL and master BL exist, use master BL as entity_id
- The entity_id MUST be consistent across all events for the same shipment or invoice
- For unclassified events: set status=null, normalized=null, and ALWAYS populate possible_category and title
- For shipment/invoice events: set possible_category=null and title=null"""


def _parse_response(raw_text: str) -> LLMNormalizedEvent:
    """Strip optional markdown fences and parse the JSON response."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return LLMNormalizedEvent.model_validate(json.loads(text))


# ── Groq (default) ────────────────────────────────────────────────────────────
async def _normalize_groq(raw_payload: dict) -> LLMNormalizedEvent:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Normalize this webhook payload:\n\n{json.dumps(raw_payload, indent=2)}"},
        ],
        response_format={"type": "json_object"},  # guaranteed JSON — no fence stripping needed
        temperature=0,
        max_tokens=1024,
    )
    return _parse_response(response.choices[0].message.content)


# ── Google Gemini ─────────────────────────────────────────────────────────────
async def _normalize_gemini(raw_payload: dict) -> LLMNormalizedEvent:
    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=_SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json", "temperature": 0},
    )
    response = await model.generate_content_async(
        f"Normalize this webhook payload:\n\n{json.dumps(raw_payload, indent=2)}"
    )
    return _parse_response(response.text)


# ── Anthropic / Claude ────────────────────────────────────────────────────────
async def _normalize_anthropic(raw_payload: dict) -> LLMNormalizedEvent:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"Normalize this webhook payload:\n\n{json.dumps(raw_payload, indent=2)}"}],
    )
    return _parse_response(response.content[0].text)


# ── Public entry point ────────────────────────────────────────────────────────
async def normalize_payload(raw_payload: dict) -> LLMNormalizedEvent:
    provider = settings.llm_provider.lower()
    if provider == "groq":
        return await _normalize_groq(raw_payload)
    elif provider == "gemini":
        return await _normalize_gemini(raw_payload)
    elif provider == "anthropic":
        return await _normalize_anthropic(raw_payload)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: groq, gemini, anthropic")
