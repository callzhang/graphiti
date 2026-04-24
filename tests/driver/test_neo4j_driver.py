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

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from neo4j.exceptions import ClientError

from graphiti_core.driver import neo4j_driver as module


@pytest.mark.asyncio
async def test_close_ignores_known_buffer_error_and_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    message = "Existing exports of data: object cannot be re-sized"
    mock_client = type("MockClient", (), {"close": AsyncMock(side_effect=BufferError(message))})()
    warnings: list[tuple[object, ...]] = []

    monkeypatch.setattr(module.AsyncGraphDatabase, "driver", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(module.logger, "warning", lambda *args, **kwargs: warnings.append(args))

    driver = module.Neo4jDriver("bolt://localhost:7687", "neo4j", "secret")

    await driver.close()
    await driver.close()

    assert mock_client.close.await_count == 1
    assert warnings


@pytest.mark.asyncio
async def test_close_propagates_unexpected_buffer_error(monkeypatch: pytest.MonkeyPatch):
    mock_client = type("MockClient", (), {"close": AsyncMock(side_effect=BufferError("unexpected"))})()

    monkeypatch.setattr(module.AsyncGraphDatabase, "driver", lambda *args, **kwargs: mock_client)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))

    driver = module.Neo4jDriver("bolt://localhost:7687", "neo4j", "secret")

    with pytest.raises(BufferError, match="unexpected"):
        await driver.close()


@pytest.mark.asyncio
async def test_execute_query_suppresses_equivalent_schema_error_logging(monkeypatch: pytest.MonkeyPatch):
    client = type("MockClient", (), {})()
    client.execute_query = AsyncMock(side_effect=ClientError("EquivalentSchemaRuleAlreadyExists"))
    logged_errors: list[tuple[object, ...]] = []

    monkeypatch.setattr(module.AsyncGraphDatabase, "driver", lambda *args, **kwargs: client)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(module.logger, "error", lambda *args, **kwargs: logged_errors.append(args))

    driver = module.Neo4jDriver("bolt://localhost:7687", "neo4j", "secret")

    with pytest.raises(ClientError, match="EquivalentSchemaRuleAlreadyExists"):
        await driver.execute_query("CREATE INDEX x IF NOT EXISTS")

    assert logged_errors == []


@pytest.mark.asyncio
async def test_build_indices_only_creates_missing_indexes(monkeypatch: pytest.MonkeyPatch):
    client = type("MockClient", (), {})()

    monkeypatch.setattr(module.AsyncGraphDatabase, "driver", lambda *args, **kwargs: client)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))

    driver = module.Neo4jDriver("bolt://localhost:7687", "neo4j", "secret")
    driver.execute_query = AsyncMock(
        return_value=SimpleNamespace(records=[{"names": ["entity_uuid", "episode_content"]}])
    )
    driver._execute_index_query = AsyncMock()

    monkeypatch.setattr(
        module,
        "get_range_indices",
        lambda provider: [
            "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)",
            "CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)",
        ],
    )
    monkeypatch.setattr(
        module,
        "get_fulltext_indices",
        lambda provider: [
            """CREATE FULLTEXT INDEX episode_content IF NOT EXISTS
            FOR (e:Episodic) ON EACH [e.content]""",
            """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name]""",
        ],
    )

    await driver.build_indices_and_constraints()

    driver.execute_query.assert_awaited_once_with(
        "SHOW INDEXES YIELD name RETURN collect(name) AS names"
    )
    assert [call.args[0] for call in driver._execute_index_query.await_args_list] == [
        "CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)",
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
            FOR (n:Entity) ON EACH [n.name]""",
    ]


def test_report_event_sends_structured_payload_to_observer(monkeypatch: pytest.MonkeyPatch):
    observed: list[dict[str, object]] = []
    client = type("MockClient", (), {})()

    monkeypatch.setattr(module.AsyncGraphDatabase, "driver", lambda *args, **kwargs: client)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))

    driver = module.Neo4jDriver(
        "bolt://localhost:7687",
        "neo4j",
        "secret",
        event_observer=observed.append,
    )

    driver._report_event(
        "query_error",
        {"error_class": "neo4j_connection_defunct", "message": "defunct connection"},
        query_name="demo",
    )

    assert observed[0]["kind"] == "query_error"
    assert observed[0]["error_class"] == "neo4j_connection_defunct"
    assert observed[0]["query_name"] == "demo"
    assert observed[0]["observed_at"]


@pytest.mark.asyncio
async def test_execute_query_reports_success_and_failure_events(monkeypatch: pytest.MonkeyPatch):
    observed: list[dict[str, object]] = []
    client = type("MockClient", (), {})()
    client.execute_query = AsyncMock(
        side_effect=[SimpleNamespace(records=[]), RuntimeError("defunct connection")]
    )

    monkeypatch.setattr(module.AsyncGraphDatabase, "driver", lambda *args, **kwargs: client)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))

    driver = module.Neo4jDriver(
        "bolt://localhost:7687",
        "neo4j",
        "secret",
        event_observer=observed.append,
    )

    await driver.execute_query("MATCH (n) RETURN n", query_name="read_nodes")
    with pytest.raises(RuntimeError, match="defunct connection"):
        await driver.execute_query("MATCH (n) RETURN n", query_name="read_nodes")

    assert observed[0]["kind"] == "query_ok"
    assert observed[0]["query_name"] == "read_nodes"
    assert observed[1]["kind"] == "query_error"
    assert observed[1]["error_class"] == "neo4j_connection_defunct"
