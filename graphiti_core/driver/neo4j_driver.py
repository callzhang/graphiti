"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase, EagerResult
from neo4j.exceptions import ClientError
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphDriverSession, GraphProvider
from graphiti_core.driver.neo4j.operations.community_edge_ops import Neo4jCommunityEdgeOperations
from graphiti_core.driver.neo4j.operations.community_node_ops import Neo4jCommunityNodeOperations
from graphiti_core.driver.neo4j.operations.entity_edge_ops import Neo4jEntityEdgeOperations
from graphiti_core.driver.neo4j.operations.entity_node_ops import Neo4jEntityNodeOperations
from graphiti_core.driver.neo4j.operations.episode_node_ops import Neo4jEpisodeNodeOperations
from graphiti_core.driver.neo4j.operations.episodic_edge_ops import Neo4jEpisodicEdgeOperations
from graphiti_core.driver.neo4j.operations.graph_ops import Neo4jGraphMaintenanceOperations
from graphiti_core.driver.neo4j.operations.has_episode_edge_ops import (
    Neo4jHasEpisodeEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.next_episode_edge_ops import (
    Neo4jNextEpisodeEdgeOperations,
)
from graphiti_core.driver.neo4j.operations.saga_node_ops import Neo4jSagaNodeOperations
from graphiti_core.driver.neo4j.operations.search_ops import Neo4jSearchOperations
from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations
from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations
from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations
from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations
from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations
from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations
from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations
from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import Transaction
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import semaphore_gather

logger = logging.getLogger(__name__)


def _compact_query_summary(query: str, *, limit: int = 240) -> str:
    compact = " ".join(query.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def classify_neo4j_error(exc: BaseException) -> dict[str, str | None]:
    message = str(exc)
    lowered = message.lower()
    error_class = "neo4j_error"
    if "defunct connection" in lowered:
        error_class = "neo4j_connection_defunct"
    elif "transaction closed" in lowered:
        error_class = "neo4j_transaction_closed"
    elif "incompletecommit" in lowered or "incomplete commit" in lowered:
        error_class = "neo4j_incomplete_commit"
    elif "entitynotfound" in lowered or "entity not found" in lowered:
        error_class = "neo4j_entity_not_found"
    elif "sessionexpired" in lowered or "session expired" in lowered:
        error_class = "neo4j_session_expired"
    elif "serviceunavailable" in lowered or "service unavailable" in lowered:
        error_class = "neo4j_service_unavailable"
    elif "parametermissing" in lowered or "expected parameter" in lowered:
        error_class = "neo4j_parameter_missing"
    return {
        "error_class": error_class,
        "neo4j_code": getattr(exc, "code", None) or getattr(exc, "neo4j_code", None),
        "gql_status": getattr(exc, "gql_status", None),
        "message": message,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Neo4jDriver(GraphDriver):
    provider = GraphProvider.NEO4J
    default_group_id: str = ''

    def __init__(
        self,
        uri: str,
        user: str | None,
        password: str | None,
        database: str = 'neo4j',
        *,
        keep_alive: bool = True,
        liveness_check_timeout: float | None = None,
        max_connection_lifetime: float | None = None,
        connection_timeout: float | None = None,
        connection_acquisition_timeout: float | None = None,
        max_transaction_retry_time: float | None = None,
        max_connection_pool_size: int | None = None,
        event_observer: Any | None = None,
    ):
        super().__init__()
        driver_kwargs: dict[str, Any] = {
            "uri": uri,
            "auth": (user or '', password or ''),
            "keep_alive": keep_alive,
        }
        if liveness_check_timeout is not None:
            driver_kwargs["liveness_check_timeout"] = liveness_check_timeout
        if max_connection_lifetime is not None:
            driver_kwargs["max_connection_lifetime"] = max_connection_lifetime
        if connection_timeout is not None:
            driver_kwargs["connection_timeout"] = connection_timeout
        if connection_acquisition_timeout is not None:
            driver_kwargs["connection_acquisition_timeout"] = connection_acquisition_timeout
        if max_transaction_retry_time is not None:
            driver_kwargs["max_transaction_retry_time"] = max_transaction_retry_time
        if max_connection_pool_size is not None:
            driver_kwargs["max_connection_pool_size"] = max_connection_pool_size
        self.client = AsyncGraphDatabase.driver(
            **driver_kwargs,
        )
        self._database = database
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._index_init_task: asyncio.Task[Any] | None = None
        self._event_observer = event_observer

        # Instantiate Neo4j operations
        self._entity_node_ops = Neo4jEntityNodeOperations()
        self._episode_node_ops = Neo4jEpisodeNodeOperations()
        self._community_node_ops = Neo4jCommunityNodeOperations()
        self._saga_node_ops = Neo4jSagaNodeOperations()
        self._entity_edge_ops = Neo4jEntityEdgeOperations()
        self._episodic_edge_ops = Neo4jEpisodicEdgeOperations()
        self._community_edge_ops = Neo4jCommunityEdgeOperations()
        self._has_episode_edge_ops = Neo4jHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = Neo4jNextEpisodeEdgeOperations()
        self._search_ops = Neo4jSearchOperations()
        self._graph_ops = Neo4jGraphMaintenanceOperations()

        # Schedule the indices and constraints to be built
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Schedule the build_indices_and_constraints to run
            self._index_init_task = loop.create_task(self.build_indices_and_constraints())
        except RuntimeError:
            # No event loop running, this will be handled later
            pass

        self.aoss_client = None

    def _report_event(self, kind: str, payload: dict[str, Any] | None = None, **extra: Any) -> None:
        if self._event_observer is None:
            return
        event = dict(payload or {})
        event["kind"] = kind
        event["observed_at"] = str(event.get("observed_at") or _utc_now_iso())
        event.update(extra)
        self._event_observer(event)

    # --- Operations properties ---

    @property
    def entity_node_ops(self) -> EntityNodeOperations:
        return self._entity_node_ops

    @property
    def episode_node_ops(self) -> EpisodeNodeOperations:
        return self._episode_node_ops

    @property
    def community_node_ops(self) -> CommunityNodeOperations:
        return self._community_node_ops

    @property
    def saga_node_ops(self) -> SagaNodeOperations:
        return self._saga_node_ops

    @property
    def entity_edge_ops(self) -> EntityEdgeOperations:
        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self) -> EpisodicEdgeOperations:
        return self._episodic_edge_ops

    @property
    def community_edge_ops(self) -> CommunityEdgeOperations:
        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self) -> HasEpisodeEdgeOperations:
        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self) -> NextEpisodeEdgeOperations:
        return self._next_episode_edge_ops

    @property
    def search_ops(self) -> SearchOperations:
        return self._search_ops

    @property
    def graph_ops(self) -> GraphMaintenanceOperations:
        return self._graph_ops

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        """Neo4j transaction with real commit/rollback semantics."""
        async with self.client.session(database=self._database) as session:
            tx = await session.begin_transaction()
            try:
                yield _Neo4jTransaction(tx)
                await tx.commit()
            except BaseException:
                await tx.rollback()
                raise

    async def execute_query(self, cypher_query_: LiteralString, **kwargs: Any) -> EagerResult:
        # Check if database_ is provided in kwargs.
        # If not populated, set the value to retain backwards compatibility
        query_name = kwargs.pop('query_name', None)
        params = kwargs.pop('params', None)
        if params is None:
            params = {}
        params.setdefault('database_', self._database)
        query_summary = _compact_query_summary(cypher_query_)

        try:
            result = await self.client.execute_query(cypher_query_, parameters_=params, **kwargs)
        except ClientError as exc:
            if 'EquivalentSchemaRuleAlreadyExists' in str(exc):
                raise
            details = classify_neo4j_error(exc)
            self._report_event(
                'query_error',
                details,
                query_name=query_name,
                query_summary=query_summary,
            )
            logger.error(
                'Error executing Neo4j query: class=%s neo4j_code=%s gql_status=%s query=%s params=%s error=%s',
                details["error_class"],
                details["neo4j_code"],
                details["gql_status"],
                query_summary,
                params,
                details["message"],
            )
            raise
        except Exception as exc:
            details = classify_neo4j_error(exc)
            self._report_event(
                'query_error',
                details,
                query_name=query_name,
                query_summary=query_summary,
            )
            logger.error(
                'Error executing Neo4j query: class=%s neo4j_code=%s gql_status=%s query=%s params=%s error=%s',
                details["error_class"],
                details["neo4j_code"],
                details["gql_status"],
                query_summary,
                params,
                details["message"],
            )
            raise

        self._report_event('query_ok', query_name=query_name, query_summary=query_summary)
        return result

    def session(self, database: str | None = None) -> GraphDriverSession:
        _database = database or self._database
        return self.client.session(database=_database)  # type: ignore

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return None

            init_task = self._index_init_task
            if init_task is not None:
                self._index_init_task = None
                if not init_task.done():
                    init_task.cancel()
                if init_task is not asyncio.current_task():
                    await asyncio.gather(init_task, return_exceptions=True)

            try:
                await self.client.close()
            except BufferError as exc:
                if 'Existing exports of data: object cannot be re-sized' not in str(exc):
                    raise
                logger.warning(
                    'Ignoring known neo4j async driver BufferError during close; shutdown will continue'
                )

            self._closed = True
            return None

    def delete_all_indexes(self) -> Coroutine:
        return self.client.execute_query(
            'CALL db.indexes() YIELD name DROP INDEX name',
        )

    async def _execute_index_query(self, query: LiteralString) -> EagerResult | None:
        """Execute an index creation query, ignoring 'index already exists' errors.

        Neo4j can raise EquivalentSchemaRuleAlreadyExists when concurrent CREATE INDEX
        IF NOT EXISTS queries race, even though the index exists. This is safe to ignore.
        """
        try:
            return await self.execute_query(query)
        except ClientError as e:
            # Ignore "equivalent index already exists" error (race condition with IF NOT EXISTS)
            if 'EquivalentSchemaRuleAlreadyExists' in str(e):
                logger.debug(f'Index already exists (concurrent creation): {query[:50]}...')
                return None
            raise

    def _extract_index_name(self, query: str) -> str | None:
        match = re.search(
            r'CREATE\s+(?:FULLTEXT\s+)?INDEX\s+([A-Za-z_][A-Za-z0-9_]*)\s+IF\s+NOT\s+EXISTS',
            query,
            re.IGNORECASE,
        )
        if match is None:
            return None
        return match.group(1)

    async def _list_existing_index_names(self) -> set[str]:
        try:
            result = await self.execute_query('SHOW INDEXES YIELD name RETURN collect(name) AS names')
        except Exception:
            result = await self.execute_query('CALL db.indexes() YIELD name RETURN collect(name) AS names')

        records = getattr(result, 'records', None) or []
        if not records:
            return set()

        record = records[0]
        names = record.get('names') if hasattr(record, 'get') else record['names']
        return {str(name) for name in (names or []) if name}

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        if delete_existing:
            await self.delete_all_indexes()

        range_indices: list[LiteralString] = get_range_indices(self.provider)

        fulltext_indices: list[LiteralString] = get_fulltext_indices(self.provider)

        index_queries: list[LiteralString] = range_indices + fulltext_indices
        existing_names = await self._list_existing_index_names()
        missing_queries = [
            query
            for query in index_queries
            if (index_name := self._extract_index_name(query)) is None or index_name not in existing_names
        ]
        if missing_queries:
            await semaphore_gather(*[self._execute_index_query(query) for query in missing_queries])

    async def health_check(self) -> None:
        """Check Neo4j connectivity by running the driver's verify_connectivity method."""
        try:
            await self.client.verify_connectivity()
            self._report_event('health_check_ok', query_name='verify_connectivity')
            return None
        except Exception as e:
            self._report_event(
                'health_check_error',
                classify_neo4j_error(e),
                query_name='verify_connectivity',
            )
            print(f'Neo4j health check failed: {e}')
            raise


class _Neo4jTransaction(Transaction):
    """Wraps a Neo4j AsyncTransaction for the Transaction ABC."""

    def __init__(self, tx: Any):
        self._tx = tx

    async def run(self, query: str, **kwargs: Any) -> Any:
        return await self._tx.run(query, **kwargs)
