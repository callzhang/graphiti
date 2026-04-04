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
from unittest.mock import AsyncMock

import pytest

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
