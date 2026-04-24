from datetime import datetime, timezone

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.record_parsers import entity_edge_from_record
from graphiti_core.edges import get_entity_edge_from_record, normalize_entity_edge_episodes


def _record(*, episodes):
    now = datetime(2026, 4, 24, tzinfo=timezone.utc)
    return {
        "uuid": "edge-1",
        "source_node_uuid": "source-1",
        "target_node_uuid": "target-1",
        "fact": "Alice owns Atlas",
        "fact_embedding": None,
        "fact_embedding_model": None,
        "name": "OWNS",
        "group_id": "g1",
        "episodes": episodes,
        "created_at": now,
        "valid_at": now,
        "invalid_at": None,
        "attributes": {},
    }


def test_normalize_entity_edge_episodes_handles_empty_and_csv_values() -> None:
    assert normalize_entity_edge_episodes(None) == []
    assert normalize_entity_edge_episodes("") == []
    assert normalize_entity_edge_episodes("   ") == []
    assert normalize_entity_edge_episodes("ep-1,ep-2") == ["ep-1", "ep-2"]
    assert normalize_entity_edge_episodes(["ep-1", " ", "ep-2"]) == ["ep-1", "ep-2"]


def test_get_entity_edge_from_record_normalizes_empty_string_episodes() -> None:
    edge = get_entity_edge_from_record(_record(episodes=""), GraphProvider.NEO4J)
    assert edge.episodes == []


def test_entity_edge_from_record_normalizes_csv_episodes() -> None:
    edge = entity_edge_from_record(_record(episodes="ep-1, ep-2"))
    assert edge.episodes == ["ep-1", "ep-2"]
