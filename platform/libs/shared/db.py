"""
Database connections — asyncpg (Postgres/TimescaleDB) + motor (MongoDB).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import asyncpg
import motor.motor_asyncio

logger = logging.getLogger(__name__)

PG_DSN    = os.getenv("POSTGRES_DSN", "postgresql://trading:trading@localhost:5432/trading_db")
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGODB_DB", "los_ojos")


class PostgresPool:
    def __init__(self, dsn: str = PG_DSN, min_size: int = 5, max_size: int = 20):
        self._dsn = dsn
        self._min = min_size
        self._max = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min,
            max_size=self._max,
            command_timeout=30,
        )
        logger.info("Postgres pool connected (min=%d max=%d)", self._min, self._max)

    async def close(self):
        if self._pool:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("PostgresPool not connected")
        return self._pool

    async def fetch(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args_list: list):
        async with self._pool.acquire() as conn:
            return await conn.executemany(query, args_list)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()


class MongoClient:
    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self._uri = uri
        self._db_name = db_name
        self._client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self._db = None

    def connect(self):
        self._client = motor.motor_asyncio.AsyncIOMotorClient(
            self._uri,
            serverSelectionTimeoutMS=5000,
        )
        self._db = self._client[self._db_name]
        logger.info("MongoDB client ready → %s/%s", self._uri, self._db_name)

    def collection(self, name: str):
        return self._db[name]

    def close(self):
        if self._client:
            self._client.close()
