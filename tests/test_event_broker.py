"""
Unit and Integration Test Suite for Event Ingestion Gateway & Broker Subsystem.

Validates:
1. Mathematical event volume measurement and broker sizing logic.
2. InMemoryEventBroker publication, concurrency, and consumer execution.
3. Idempotent deduplication via idempotency keys.
4. Exponential retry backoff on transient consumer failure.
5. Dead-letter queue (DLQ) routing when retries are exhausted.
6. EventGateway integration with Member 1's TransactionIngestionPipeline & FleetManager.
7. RabbitMQEventBroker client fallback behavior when daemon is offline.
8. API endpoints: GET /v1/system/broker-status and POST /v1/system/broker-publish-test.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
import pytest

from src.ingestion.event_broker import (
    BrokerMessage,
    BrokerSizingProfile,
    EventBroker,
    EventGateway,
    InMemoryEventBroker,
    RabbitMQEventBroker,
    calculate_broker_sizing,
    event_gateway,
    QUEUE_TRANSACTIONS,
    QUEUE_TELEMETRY,
    QUEUE_DEAD_LETTER,
)
from src.ingestion.schema import (
    TransactionRecord,
    TransactionLocation,
    PaymentMethod,
    TransactionStatus,
    TransactionItem,
)
from src.fleet.fleet_manager import FleetManager, Vendor, VendorStatus
from src.api.webhooks.dead_letter import dead_letter_queue
from src.api.app import app


# ---------------------------------------------------------------------------
# 1. Volume Measurement & Sizing Logic Tests
# ---------------------------------------------------------------------------

def test_broker_sizing_vijayawada_deployment():
    """Verify standard Vijayawada deployment workload measurement and RabbitMQ selection."""
    profile = calculate_broker_sizing(
        active_vendors=3000,
        tx_per_vendor_day=20.0,
        shift_hours=10.0,
        telemetry_interval_sec=30.0,
    )

    # 3,000 * 20 / (10 * 3600) = 60,000 / 36,000 = 1.666... tx/sec
    assert pytest.approx(profile.tx_events_per_sec, 0.05) == 1.67
    # 3,000 / 30 = 100.0 telemetry/sec
    assert pytest.approx(profile.telemetry_events_per_sec, 0.1) == 100.0
    # Total rate ~ 101.67 events/sec
    assert pytest.approx(profile.total_events_per_sec, 0.1) == 101.67

    # Broker verdict must be RabbitMQ
    assert profile.recommended_broker == "RabbitMQ (AMQP 0-9-1)"
    assert "RabbitMQ comfortably sustains" in profile.rationale
    assert "Kafka would introduce unnecessary operational complexity" in profile.rationale

    as_dict = profile.to_dict()
    assert as_dict["active_vendors"] == 3000
    assert as_dict["total_events_per_sec"] == 101.67


def test_broker_sizing_statewide_expansion():
    """Verify 10-city statewide expansion workload measurement."""
    profile = calculate_broker_sizing(
        active_vendors=25000,
        tx_per_vendor_day=25.0,
        shift_hours=12.0,
        telemetry_interval_sec=30.0,
    )
    # 25,000 * 25 / 43,200 = 14.47 tx/sec
    # 25,000 / 30 = 833.33 telemetry/sec
    # Total ~ 847.8 events/sec
    assert 800.0 < profile.total_events_per_sec < 900.0
    # Still comfortably within single-node RabbitMQ capacity (< 25,000 msg/sec)
    assert profile.recommended_broker == "RabbitMQ (AMQP 0-9-1)"


def test_broker_sizing_kafka_threshold():
    """Verify that astronomical volumes (> 25k events/sec) correctly recommend Apache Kafka."""
    profile = calculate_broker_sizing(
        active_vendors=1000000,
        tx_per_vendor_day=50.0,
        shift_hours=8.0,
        telemetry_interval_sec=10.0,
    )
    assert profile.total_events_per_sec > 25000
    assert profile.recommended_broker == "Apache Kafka"
    assert "Partitioned distributed commit log recommended" in profile.rationale


# ---------------------------------------------------------------------------
# 2. InMemoryEventBroker Publication & Consumer Roundtrip
# ---------------------------------------------------------------------------

def test_in_memory_broker_publish_and_consume():
    """Verify message publication, asynchronous worker consumption, and ACKs."""
    broker = InMemoryEventBroker(num_workers=2)
    broker.start()

    received_messages = []

    def sample_handler(payload: dict) -> bool:
        received_messages.append(payload)
        return True

    broker.subscribe("test.queue", sample_handler, max_retries=2, retry_backoff_sec=0.05)

    msg_payload = {"event": "test_ping", "value": 42}
    published = broker.publish("test.queue", msg_payload)
    assert published is True

    # Allow worker thread to process
    time.sleep(0.15)
    broker.stop()

    assert len(received_messages) == 1
    assert received_messages[0]["value"] == 42
    metrics = broker.get_metrics()
    assert metrics["published_count"] == 1
    assert metrics["consumed_count"] == 1
    assert metrics["dead_letter_count"] == 0


# ---------------------------------------------------------------------------
# 3. Idempotent Processing
# ---------------------------------------------------------------------------

def test_idempotent_event_deduplication():
    """Verify that duplicate messages with identical idempotency_key are rejected."""
    broker = InMemoryEventBroker(num_workers=1)
    broker.start()

    received = []
    broker.subscribe("test.idempotent", lambda p: received.append(p) or True)

    key = "unique-order-uuid-999"
    pub1 = broker.publish("test.idempotent", {"order_id": 999, "seq": 1}, headers={"idempotency_key": key})
    pub2 = broker.publish("test.idempotent", {"order_id": 999, "seq": 2}, headers={"idempotency_key": key})

    assert pub1 is True
    assert pub2 is True  # Duplicate acknowledged safely

    time.sleep(0.15)
    broker.stop()

    # Only one message should have been delivered to the consumer
    assert len(received) == 1
    assert received[0]["seq"] == 1
    assert broker.published_count == 1  # 2nd was deduplicated at broker level


# ---------------------------------------------------------------------------
# 4. Exponential Retry Backoff & DLQ Routing
# ---------------------------------------------------------------------------

def test_exponential_retry_and_dead_letter_routing():
    """Verify that failing consumers retry with backoff and route to DLX/DLQ upon exhaustion."""
    broker = InMemoryEventBroker(num_workers=2)
    broker.start()

    attempts = []
    dlq_received = []

    def failing_handler(payload: dict) -> bool:
        attempts.append(time.time())
        return False  # Always fail to trigger retry

    def dlq_handler(payload: dict) -> bool:
        dlq_received.append(payload)
        return True

    # Subscribe failing handler with max 2 retries (3 total attempts) and 0.05s initial backoff
    broker.subscribe("test.failing", failing_handler, max_retries=2, retry_backoff_sec=0.05)
    broker.subscribe(QUEUE_DEAD_LETTER, dlq_handler, max_retries=1, retry_backoff_sec=0.0)

    broker.publish("test.failing", {"faulty_tx": "TX-FAIL-101"})

    # Wait for initial attempt + 2 retries (0.05s, 0.1s) + DLQ routing (~0.4s)
    time.sleep(0.4)
    broker.stop()

    assert len(attempts) == 3  # Initial + 2 retries
    # Verify exponential backoff intervals
    interval_1 = attempts[1] - attempts[0]
    interval_2 = attempts[2] - attempts[1]
    assert interval_1 >= 0.04  # ~0.05s
    assert interval_2 >= 0.08  # ~0.10s (exponential scale 0.05 * 2^1)

    # Must have arrived at dead letter queue
    assert len(dlq_received) == 1
    assert dlq_received[0]["payload"]["faulty_tx"] == "TX-FAIL-101"
    assert dlq_received[0]["reason"] == "MAX_RETRIES_EXCEEDED"

    metrics = broker.get_metrics()
    assert metrics["retry_count"] == 2
    assert metrics["dead_letter_count"] == 1


# ---------------------------------------------------------------------------
# 5. Event Gateway Integration Tests
# ---------------------------------------------------------------------------

def test_event_gateway_transaction_ingestion():
    """Verify EventGateway consumes transaction events, validates via pipeline, and updates fleet."""
    gateway = EventGateway(broker=InMemoryEventBroker(num_workers=2))
    gateway.broker.start()

    # Pre-register a vendor in fleet manager
    v = Vendor(
        vendor_id="VND-STREAM-01",
        category="food",
        name="Rao Tiffins",
        current_lat=16.5062,
        current_lng=80.6480,
    )
    gateway.fleet_manager.register_vendor(v)

    tx_id = f"TX-GW-{uuid.uuid4().hex[:8].upper()}"
    rec = TransactionRecord(
        transaction_id=tx_id,
        vendor_id="VND-STREAM-01",
        customer_id="CUST-444",
        event_timestamp=datetime.now(timezone.utc),
        amount_inr=320.0,
        payment_method=PaymentMethod.UPI,
        status=TransactionStatus.COMPLETED,
        location=TransactionLocation(latitude=16.5150, longitude=80.6350),
        items=[TransactionItem(item_name="Dosa", unit_price_inr=80.0, quantity=4, total_price_inr=320.0)],
        source="test:event_gateway",
    )

    published = gateway.publish_transaction(rec)
    assert published is True

    # Allow consumer worker to process
    time.sleep(0.2)
    gateway.broker.stop()

    # Check vendor location updated in fleet manager
    updated_v = gateway.fleet_manager.get_vendor("VND-STREAM-01")
    assert updated_v is not None
    assert pytest.approx(updated_v.current_lat, 0.001) == 16.5150
    assert pytest.approx(updated_v.current_lng, 0.001) == 80.6350
    assert updated_v.last_heartbeat is not None


def test_event_gateway_telemetry_update():
    """Verify EventGateway processes GPS telemetry pings and registers coordinates."""
    gateway = EventGateway(broker=InMemoryEventBroker(num_workers=2))
    gateway.broker.start()

    v = Vendor(
        vendor_id="VND-GPS-99",
        category="beverages",
        name="Bhavani Juices",
        current_lat=16.5000,
        current_lng=80.6400,
    )
    gateway.fleet_manager.register_vendor(v)

    pub = gateway.publish_telemetry("VND-GPS-99", 16.5200, 80.6500)
    assert pub is True

    time.sleep(0.2)
    gateway.broker.stop()

    updated = gateway.fleet_manager.get_vendor("VND-GPS-99")
    assert pytest.approx(updated.current_lat, 0.0001) == 16.5200
    assert pytest.approx(updated.current_lng, 0.0001) == 80.6500


# ---------------------------------------------------------------------------
# 6. RabbitMQ Fallback Mechanics
# ---------------------------------------------------------------------------

def test_rabbitmq_broker_graceful_fallback():
    """Verify RabbitMQ client falls back to InMemoryEventBroker when daemon is unavailable."""
    # Point to an unreachable port
    broker = RabbitMQEventBroker(amqp_url="amqp://guest:guest@127.0.0.1:59999/")
    # If connection fails, it routes through InMemory fallback
    metrics = broker.get_metrics()
    assert "broker_type" in metrics

    # Verify publish and subscribe work seamlessly via fallback
    received = []
    broker.subscribe("fallback.test", lambda p: received.append(p) or True)
    broker.start()
    broker.publish("fallback.test", {"status": "ok"})
    time.sleep(0.1)
    broker.stop()

    assert len(received) == 1
    assert received[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoint Verification
# ---------------------------------------------------------------------------

def test_api_broker_status_endpoint():
    """Verify GET /v1/system/broker-status returns live telemetry and sizing metrics."""
    route_handler = None
    for r in app.routes:
        if getattr(r, "path", None) == "/v1/system/broker-status" and "GET" in r.methods:
            route_handler = r.endpoint
            break
    assert route_handler is not None

    coro = route_handler(active_vendors=5000, tx_per_vendor=22.0, shift_hours=10.0, telemetry_interval=30.0)
    res = None
    try:
        coro.send(None)
    except StopIteration as e:
        res = e.value

    assert res is not None
    assert res["gateway_status"] == "ONLINE"
    assert "active_broker" in res
    assert "sizing_profile" in res
    assert res["sizing_profile"]["active_vendors"] == 5000
    assert res["sizing_profile"]["recommended_broker"] == "RabbitMQ (AMQP 0-9-1)"
    assert "telemetry" in res


def test_api_broker_publish_test_endpoint():
    """Verify POST /v1/system/broker-publish-test publishes test events."""
    route_handler = None
    for r in app.routes:
        if getattr(r, "path", None) == "/v1/system/broker-publish-test" and "POST" in r.methods:
            route_handler = r.endpoint
            break
    assert route_handler is not None

    from src.api.schemas import BrokerPublishTestRequest
    # Test telemetry event publication
    req = BrokerPublishTestRequest(
        event_type="telemetry",
        vendor_id="VND-TEST-01",
        latitude=16.5080,
        longitude=80.6450,
    )
    coro = route_handler(req=req, api_key=None)
    res = None
    try:
        coro.send(None)
    except StopIteration as e:
        res = e.value

    assert res["status"] == "PUBLISHED"
    assert res["event_type"] == "telemetry"
    assert res["vendor_id"] == "VND-TEST-01"

    # Test transaction event publication
    req_tx = BrokerPublishTestRequest(
        event_type="transaction",
        vendor_id="VND-TEST-02",
        amount_inr=180.0,
        latitude=16.5100,
        longitude=80.6420,
    )
    coro_tx = route_handler(req=req_tx, api_key=None)
    res_tx = None
    try:
        coro_tx.send(None)
    except StopIteration as e:
        res_tx = e.value

    assert res_tx["status"] == "PUBLISHED"
    assert res_tx["event_type"] == "transaction"
    assert res_tx["vendor_id"] == "VND-TEST-02"
    assert "transaction_id" in res_tx
