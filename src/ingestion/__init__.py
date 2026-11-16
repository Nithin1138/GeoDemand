"""
Ingestion module for real vendor transactions and telemetry (Phase 11.1).
"""

from src.ingestion.schema import (
    PaymentMethod,
    TransactionItem,
    TransactionLocation,
    TransactionRecord,
    TransactionStatus,
)
from src.ingestion.validator import (
    IngestionResult,
    TransactionIngestionPipeline,
)
from src.ingestion.event_broker import (
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
    EXCHANGE_DIRECT,
    EXCHANGE_DLX,
)

__all__ = [
    "PaymentMethod",
    "TransactionItem",
    "TransactionLocation",
    "TransactionRecord",
    "TransactionStatus",
    "IngestionResult",
    "TransactionIngestionPipeline",
    "BrokerSizingProfile",
    "EventBroker",
    "EventGateway",
    "InMemoryEventBroker",
    "RabbitMQEventBroker",
    "calculate_broker_sizing",
    "event_gateway",
    "QUEUE_TRANSACTIONS",
    "QUEUE_TELEMETRY",
    "QUEUE_DEAD_LETTER",
    "EXCHANGE_DIRECT",
    "EXCHANGE_DLX",
]
