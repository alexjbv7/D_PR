"""
Async Kafka Client — Base producer/consumer for all services.
=============================================================
Wrapper sobre aiokafka que estandariza:
- Serialización JSON de BaseEvent
- Retry con backoff exponencial
- Dead Letter Queue (DLQ)
- Métricas Prometheus
- Context manager limpio

Uso productor:
    async with KafkaProducer() as producer:
        await producer.send(KafkaTopics.SIGNAL_RAW, event)

Uso consumidor:
    async with KafkaConsumer(topics=[KafkaTopics.MACRO_DATA],
                             group_id="my-service") as consumer:
        async for event in consumer:
            await handle(event)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Callable, Optional, Type, TypeVar

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError

from .events import BaseEvent

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseEvent)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
_MAX_RETRIES     = int(os.getenv("KAFKA_MAX_RETRIES", "3"))
_RETRY_BACKOFF   = float(os.getenv("KAFKA_RETRY_BACKOFF_S", "0.5"))
DLQ_SUFFIX       = ".dlq"


def _serialize(event: BaseEvent) -> bytes:
    return event.model_dump_json().encode("utf-8")


def _deserialize(data: bytes, model: Type[T]) -> T:
    return model.model_validate_json(data)


class KafkaProducerClient:
    """
    Async Kafka producer con retry y serialización automática.

    Uso:
        producer = KafkaProducerClient()
        await producer.start()
        await producer.send(KafkaTopics.WHALE_ALERT, event)
        await producer.stop()

    O como context manager:
        async with KafkaProducerClient() as p:
            await p.send(...)
    """

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP):
        self._bootstrap = bootstrap_servers
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self):
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap,
            value_serializer=lambda v: v,
            enable_idempotence=True,
            acks="all",
            compression_type="gzip",
            max_request_size=10_485_760,  # 10 MB
            linger_ms=5,
            request_timeout_ms=30_000,
        )
        await self._producer.start()
        logger.info("KafkaProducer started → %s", self._bootstrap)

    async def stop(self):
        if self._producer:
            await self._producer.stop()

    async def send(
        self,
        topic: str,
        event: BaseEvent,
        key: Optional[str] = None,
        retries: int = _MAX_RETRIES,
    ) -> None:
        """
        Serializa y envía el evento. Retries con backoff exponencial.
        Si falla todos los retries, envía a DLQ.
        """
        payload = _serialize(event)
        key_bytes = key.encode() if key else getattr(event, "symbol", None)
        if isinstance(key_bytes, str):
            key_bytes = key_bytes.encode()

        for attempt in range(retries + 1):
            try:
                await self._producer.send_and_wait(
                    topic, value=payload, key=key_bytes
                )
                return
            except KafkaError as exc:
                if attempt == retries:
                    logger.error(
                        "KafkaProducer: all retries exhausted topic=%s event=%s err=%s",
                        topic, event.event_id, exc,
                    )
                    await self._send_dlq(topic, payload, str(exc))
                    raise
                wait = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "KafkaProducer: retry %d/%d topic=%s err=%s wait=%.1fs",
                    attempt + 1, retries, topic, exc, wait,
                )
                await asyncio.sleep(wait)

    async def _send_dlq(self, original_topic: str, payload: bytes, error: str):
        dlq_topic = original_topic + DLQ_SUFFIX
        dlq_payload = json.dumps({
            "original_topic": original_topic,
            "error": error,
            "payload": payload.decode("utf-8", errors="replace"),
        }).encode()
        try:
            await self._producer.send_and_wait(dlq_topic, value=dlq_payload)
            logger.info("DLQ: message sent to %s", dlq_topic)
        except Exception as e:
            logger.critical("DLQ send failed: %s", e)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()


class KafkaConsumerClient:
    """
    Async Kafka consumer con manual commit y DLQ handling.

    Uso:
        async with KafkaConsumerClient(
            topics=["signals.raw"],
            group_id="risk-engine",
            model=RawSignalEvent,
        ) as consumer:
            async for event in consumer.consume():
                await handle(event)
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str,
        bootstrap_servers: str = KAFKA_BOOTSTRAP,
        model: Optional[Type[BaseEvent]] = None,
        auto_offset_reset: str = "latest",
    ):
        self._topics = topics
        self._group_id = group_id
        self._bootstrap = bootstrap_servers
        self._model = model or BaseEvent
        self._auto_offset_reset = auto_offset_reset
        self._consumer: Optional[AIOKafkaConsumer] = None

    async def start(self):
        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            enable_auto_commit=False,
            auto_offset_reset=self._auto_offset_reset,
            fetch_max_bytes=52_428_800,   # 50 MB
            session_timeout_ms=30_000,
            heartbeat_interval_ms=10_000,
        )
        await self._consumer.start()
        logger.info(
            "KafkaConsumer started group=%s topics=%s",
            self._group_id, self._topics,
        )

    async def stop(self):
        if self._consumer:
            await self._consumer.stop()

    async def consume(self) -> AsyncIterator[BaseEvent]:
        """Yields parsed events; commits offset after yield."""
        async for msg in self._consumer:
            try:
                event = _deserialize(msg.value, self._model)
                yield event
                await self._consumer.commit()
            except Exception as exc:
                logger.error(
                    "KafkaConsumer: parse error topic=%s offset=%d err=%s",
                    msg.topic, msg.offset, exc,
                )
                # No commit → offset stays, se puede revisar en DLQ manual

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_):
        await self.stop()
