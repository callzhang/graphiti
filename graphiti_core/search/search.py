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

import logging
import math
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from time import time
from typing import Any

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.errors import SearchRerankerError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather, validate_group_ids
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    DEFAULT_MMR_LAMBDA,
    community_aware_edge_search,
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    entity_anchored_edge_search,
    episode_fulltext_search,
    episode_similarity_search,
    multi_hop_edge_search,
    episode_mentions_reranker,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    maximal_marginal_relevance,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)
from graphiti_core.tracer import NoOpTracer, Tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recency-weighted reranking
# ---------------------------------------------------------------------------
RECENCY_HALF_LIFE_DAYS = 90.0


def _resolve_query_reference_time(search_filter: SearchFilters | None) -> datetime | None:
    if search_filter is None:
        return None

    def _normalize(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    for groups in (getattr(search_filter, 'valid_at', None), getattr(search_filter, 'invalid_at', None)):
        for or_list in groups or []:
            for date_filter in or_list:
                candidate = _normalize(getattr(date_filter, 'date', None))
                if candidate is not None:
                    return candidate
    return None


def _recency_weight(
    ts: datetime | None,
    *,
    reference_time: datetime | None = None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Return a multiplicative weight in (0, 1] that decays with temporal distance."""
    if ts is None:
        return 0.5  # neutral when timestamp unknown
    now = reference_time or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_days = abs((now - ts).total_seconds()) / 86_400
    return 1.0 / (1.0 + age_days / half_life_days)


def _edge_is_single_state(edge: EntityEdge) -> bool:
    return bool((getattr(edge, 'attributes', None) or {}).get('single_state'))


def _edge_slot_key(edge: EntityEdge) -> tuple[str, str, str]:
    return (
        str(getattr(edge, 'group_id', '') or ''),
        str(getattr(edge, 'source_node_uuid', '') or ''),
        str(getattr(edge, 'name', '') or ''),
    )


def _edge_is_active_at(edge: EntityEdge, reference_time: datetime) -> bool:
    valid_at = getattr(edge, 'valid_at', None)
    invalid_at = getattr(edge, 'invalid_at', None)
    if valid_at is not None:
        if valid_at.tzinfo is None:
            valid_at = valid_at.replace(tzinfo=timezone.utc)
        if valid_at > reference_time:
            return False
    if invalid_at is not None:
        if invalid_at.tzinfo is None:
            invalid_at = invalid_at.replace(tzinfo=timezone.utc)
        if invalid_at <= reference_time:
            return False
    return True


def _resolve_single_state_edge_slots(
    edges: list[EntityEdge],
    scores: list[float],
    *,
    reference_time: datetime | None,
    edge_score_details: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[EntityEdge], list[float]]:
    if reference_time is None or not edges:
        return edges, scores

    chosen_by_slot: dict[tuple[str, str, str], int] = {}
    grouped_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        if not _edge_is_single_state(edge):
            continue
        grouped_indices[_edge_slot_key(edge)].append(index)

    for slot_key, indices in grouped_indices.items():
        active_indices = [index for index in indices if _edge_is_active_at(edges[index], reference_time)]
        chosen_indices = active_indices or indices
        chosen_index = max(
            chosen_indices,
            key=lambda index: (
                float(scores[index]),
                getattr(edges[index], 'valid_at', None) or getattr(edges[index], 'created_at', None) or datetime.min.replace(tzinfo=timezone.utc),
            ),
        )
        chosen_by_slot[slot_key] = chosen_index
        if edge_score_details is not None:
            chosen_uuid = edges[chosen_index].uuid
            edge_score_details.setdefault(chosen_uuid, {}).setdefault(
                "single_state_resolution",
                {
                    "slot": list(slot_key),
                    "reference_time": reference_time.isoformat(),
                    "bucket": "active" if active_indices else "fallback",
                    "candidate_count": len(indices),
                    "chosen_uuid": chosen_uuid,
                },
            )

    resolved_edges: list[EntityEdge] = []
    resolved_scores: list[float] = []
    for index, (edge, score) in enumerate(zip(edges, scores, strict=False)):
        if _edge_is_single_state(edge):
            slot_key = _edge_slot_key(edge)
            chosen_index = chosen_by_slot.get(slot_key)
            if chosen_index != index:
                if edge_score_details is not None:
                    edge_score_details.setdefault(edge.uuid, {})["single_state_resolution"] = {
                        "slot": list(slot_key),
                        "reference_time": reference_time.isoformat(),
                        "bucket": "suppressed",
                        "candidate_count": len(grouped_indices.get(slot_key, [])),
                        "chosen_uuid": edges[chosen_index].uuid if chosen_index is not None else None,
                    }
                continue
        resolved_edges.append(edge)
        resolved_scores.append(score)

    return resolved_edges, resolved_scores


def _rerank_edges_with_recency(
    edges: list[EntityEdge],
    scores: list[float],
    *,
    reference_time: datetime | None = None,
) -> tuple[list[EntityEdge], list[float]]:
    """Re-sort edges by query-time distance decay so closer facts rank higher."""
    if not edges:
        return edges, scores
    pairs = []
    for edge, score in zip(edges, scores):
        ts = getattr(edge, 'created_at', None) or getattr(edge, 'valid_at', None)
        weighted = score * _recency_weight(ts, reference_time=reference_time)
        pairs.append((weighted, edge))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return [p[1] for p in pairs], [p[0] for p in pairs]


def _edge_method_name(method: EdgeSearchMethod) -> str:
    mapping = {
        EdgeSearchMethod.bm25: "bm25",
        EdgeSearchMethod.cosine_similarity: "cosine_similarity",
        EdgeSearchMethod.bfs: "breadth_first_search",
    }
    return mapping.get(method, str(getattr(method, "value", method)))


def _rrf_contribution(rank: int) -> float:
    return 1 / rank


def _edge_method_raw_score(edge: EntityEdge, method_name: str) -> float | None:
    attributes = getattr(edge, 'attributes', None) or {}
    raw_scores = attributes.get("__search_method_scores")
    if not isinstance(raw_scores, dict):
        return None
    raw_score = raw_scores.get(method_name)
    if raw_score is None:
        return None
    return float(raw_score)


def _formula_method_alias(method_name: str) -> str:
    aliases = {
        "cosine_similarity": "cosine",
        "entity_anchored": "entity_anchored",
        "bm25": "bm25",
    }
    return aliases.get(method_name, method_name)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, 'value') else value


def _resolve_tracer(search_tracer: Tracer | None) -> Tracer:
    return search_tracer if search_tracer is not None else NoOpTracer()


def _supports_graph_native_edge_search(driver: GraphDriver) -> bool:
    return getattr(driver, 'provider', None) == GraphProvider.NEO4J


def _build_rrf_score_tree(detail: dict[str, Any]) -> dict[str, Any]:
    methods = detail.get("methods") or {}
    method_terms: list[dict[str, Any]] = []
    base_expression_terms: list[str] = []
    base_children: list[dict[str, Any]] = []

    for method_name, method_detail in methods.items():
        rank = int(method_detail.get("rank") or 0)
        if rank <= 0:
            continue
        weight = float(method_detail.get("weight") or 1.0)
        signal = float(method_detail.get("rrf_contribution") or _rrf_contribution(rank))
        raw_score = method_detail.get("raw_score")
        method_alias = _formula_method_alias(method_name)
        term_value = weight * signal
        term_symbol = f"term_{method_name}"
        signal_symbol = f"signal_{method_name}"
        weight_symbol = f"weight_{method_name}"
        rank_symbol = f"rank_{method_alias}"
        base_expression_terms.append(f"1/{rank_symbol}")
        base_children.append(
            {
                "symbol": term_symbol,
                "value": round(term_value, 6),
                "equation": f"{term_symbol} = {weight_symbol} * {signal_symbol}",
                "children": [
                    {"symbol": weight_symbol, "value": round(weight, 6)},
                    {
                        "symbol": signal_symbol,
                        "value": round(signal, 6),
                        "equation": f"{signal_symbol} = 1 / {rank_symbol}",
                        "children": [
                            {"symbol": rank_symbol, "value": rank},
                        ],
                    },
                ],
            }
        )
        method_terms.append(
            {
                "method": method_name,
                "equations": {
                    "term": f"{term_symbol} = {weight_symbol} * {signal_symbol}",
                    "signal": f"{signal_symbol} = 1 / {rank_symbol}",
                },
                "variables": {
                    rank_symbol: rank,
                    weight_symbol: round(weight, 6),
                    signal_symbol: round(signal, 6),
                    term_symbol: round(term_value, 6),
                    **(
                        {f"raw_score_{method_name}": round(float(raw_score), 6)}
                        if raw_score is not None
                        else {}
                    ),
                },
            }
        )

    base_score = float(detail.get("pre_recency_score") or 0.0)
    recency_weight = float(detail.get("recency_weight") or 0.0)
    final_score = float(detail.get("final_score") or 0.0)
    sum_expression = " + ".join(base_expression_terms) if base_expression_terms else "0"
    final_expression = f"final_score = recency_weight * ({sum_expression})"

    return {
        "model": "rrf_with_recency_decay",
        "equations": {
            "final_score": final_expression,
            "base_score": f"base_score = {sum_expression}",
            "method_term": "term_method = weight_method * signal_method",
            "method_signal": "signal_method = 1 / rank_method",
        },
        "variables": {
            "final_score": round(final_score, 6),
            "recency_weight": round(recency_weight, 6),
            "base_score": round(base_score, 6),
            "final_rank": detail.get("final_rank"),
        },
        "method_terms": method_terms,
        "tree": {
            "symbol": "final_score",
            "value": round(final_score, 6),
            "equation": final_expression,
            "children": [
                {"symbol": "recency_weight", "value": round(recency_weight, 6)},
                {
                    "symbol": "base_score",
                    "value": round(base_score, 6),
                    "equation": f"base_score = {sum_expression}",
                    "children": base_children,
                },
            ],
        },
    }


def _build_generic_score_tree(detail: dict[str, Any]) -> dict[str, Any]:
    base_score = float(detail.get("pre_recency_score") or 0.0)
    recency_weight = float(detail.get("recency_weight") or 0.0)
    final_score = float(detail.get("final_score") or 0.0)
    methods = detail.get("methods") or {}
    method_terms = []
    for method_name, method_detail in methods.items():
        rank = method_detail.get("rank")
        method_alias = _formula_method_alias(method_name)
        method_terms.append(
            {
                "method": method_name,
                "equations": {
                    "rank_observation": f"rank_{method_alias} = observed backend rank",
                },
                "variables": {
                    f"rank_{method_alias}": rank,
                },
            }
        )
    return {
        "model": "generic_reranker_with_recency_decay",
        "equations": {
            "final_score": "final_score = recency_weight * base_score",
            "base_score": "base_score = backend_reranker_output",
        },
        "variables": {
            "final_score": round(final_score, 6),
            "recency_weight": round(recency_weight, 6),
            "base_score": round(base_score, 6),
            "final_rank": detail.get("final_rank"),
        },
        "method_terms": method_terms,
        "tree": {
            "symbol": "final_score",
            "value": round(final_score, 6),
            "equation": "final_score = recency_weight * base_score",
            "children": [
                {"symbol": "recency_weight", "value": round(recency_weight, 6)},
                {
                    "symbol": "base_score",
                    "value": round(base_score, 6),
                    "equation": "base_score = backend_reranker_output",
                },
            ],
        },
    }


def _attach_edge_score_formula_trees(
    edge_score_details: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    for detail in edge_score_details.values():
        reranker = str(detail.get("reranker") or "")
        if reranker == EdgeReranker.rrf.value:
            detail["score_formula"] = _build_rrf_score_tree(detail)
            continue
        detail["score_formula"] = _build_generic_score_tree(detail)
    return edge_score_details


@contextmanager
def _trace_phase(
    search_tracer: Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    with search_tracer.start_span(name) as span:
        if attributes:
            span.add_attributes(attributes)
        try:
            yield span
            span.set_status('ok')
        except Exception as e:
            span.set_status('error', str(e))
            span.record_exception(e)
            raise


async def search(
    clients: GraphitiClients,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    query_vector: list[float] | None = None,
    driver: GraphDriver | None = None,
) -> SearchResults:
    start = time()
    validate_group_ids(group_ids)

    driver = driver or clients.driver
    embedder = clients.embedder
    cross_encoder = clients.cross_encoder
    search_tracer = _resolve_tracer(getattr(clients, 'tracer', None))

    if query.strip() == '':
        return SearchResults()

    if (
        (
            config.edge_config
            and EdgeSearchMethod.cosine_similarity in config.edge_config.search_methods
        )
        or (config.edge_config and EdgeReranker.mmr == config.edge_config.reranker)
        or (
            config.node_config
            and NodeSearchMethod.cosine_similarity in config.node_config.search_methods
        )
        or (config.node_config and NodeReranker.mmr == config.node_config.reranker)
        or (
            config.episode_config
            and EpisodeSearchMethod.cosine_similarity in config.episode_config.search_methods
        )
        or (
            config.community_config
            and CommunitySearchMethod.cosine_similarity in config.community_config.search_methods
        )
        or (config.community_config and CommunityReranker.mmr == config.community_config.reranker)
    ):
        with _trace_phase(
            search_tracer,
            'search.embed_query_vector',
            {
                'query.length': len(query),
                'query_vector.provided': query_vector is not None,
            },
        ) as span:
            search_vector = (
                query_vector
                if query_vector is not None
                else await embedder.create(input_data=[query.replace('\n', ' ')])
            )
            span.add_attributes({'query_vector.dimension': len(search_vector)})
    else:
        search_vector = [0.0] * EMBEDDING_DIM

    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids and group_ids != [''] else None
    with _trace_phase(
        search_tracer,
        'search.execute_scopes',
        {
            'group_id.count': len(group_ids or []),
            'scope.edges': config.edge_config is not None,
            'scope.nodes': config.node_config is not None,
            'scope.episodes': config.episode_config is not None,
            'scope.communities': config.community_config is not None,
            'limit': config.limit,
        },
    ) as span:
        (
            (edges, edge_reranker_scores, edge_score_details),
            (nodes, node_reranker_scores),
            (episodes, episode_reranker_scores),
            (communities, community_reranker_scores),
        ) = await semaphore_gather(
            edge_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.edge_config,
                search_filter,
                center_node_uuid,
                bfs_origin_node_uuids,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            node_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.node_config,
                search_filter,
                center_node_uuid,
                bfs_origin_node_uuids,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            episode_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.episode_config,
                search_filter,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            community_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.community_config,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
        )
        span.add_attributes(
            {
                'result.edges': len(edges),
                'result.nodes': len(nodes),
                'result.episodes': len(episodes),
                'result.communities': len(communities),
            }
        )

    results = SearchResults(
        edges=edges,
        edge_reranker_scores=edge_reranker_scores,
        edge_score_details=edge_score_details,
        nodes=nodes,
        node_reranker_scores=node_reranker_scores,
        episodes=episodes,
        episode_reranker_scores=episode_reranker_scores,
        communities=communities,
        community_reranker_scores=community_reranker_scores,
    )

    latency = (time() - start) * 1000

    logger.debug(f'search returned context in {latency} ms')

    return results


async def edge_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: EdgeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EntityEdge], list[float]]:
    if config is None:
        return [], [], {}
    search_tracer = _resolve_tracer(search_tracer)
    query_reference_time = _resolve_query_reference_time(search_filter)

    with _trace_phase(
        search_tracer,
        'search.edge_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
            'bfs_origin_count': len(bfs_origin_node_uuids or []),
            'center_node_uuid.provided': center_node_uuid is not None,
        },
    ) as span:
        # Build search tasks based on configured search methods
        search_tasks: list[tuple[str, Any]] = []
        if EdgeSearchMethod.bm25 in config.search_methods:
            search_tasks.append(
                (_edge_method_name(EdgeSearchMethod.bm25), edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit))
            )
        if EdgeSearchMethod.cosine_similarity in config.search_methods:
            search_tasks.append(
                (
                    _edge_method_name(EdgeSearchMethod.cosine_similarity),
                    edge_similarity_search(
                        driver,
                        query_vector,
                        None,
                        None,
                        search_filter,
                        group_ids,
                        2 * limit,
                        config.sim_min_score,
                    ),
                )
            )
        if EdgeSearchMethod.bfs in config.search_methods:
            search_tasks.append(
                (
                    _edge_method_name(EdgeSearchMethod.bfs),
                    edge_bfs_search(
                        driver,
                        bfs_origin_node_uuids,
                        config.bfs_max_depth,
                        search_filter,
                        group_ids,
                        2 * limit,
                    ),
                )
            )
        if (
            EdgeSearchMethod.cosine_similarity in config.search_methods
            and _supports_graph_native_edge_search(driver)
        ):
            search_tasks.append(
                (
                    "entity_anchored",
                    entity_anchored_edge_search(
                        driver,
                        query_vector,
                        search_filter,
                        group_ids,
                        2 * limit,
                        config.sim_min_score,
                    ),
                )
            )
            search_tasks.append(
                (
                    "multi_hop",
                    multi_hop_edge_search(
                        driver,
                        query_vector,
                        search_filter,
                        group_ids,
                        limit,
                        config.sim_min_score,
                    ),
                )
            )
            search_tasks.append(
                (
                    "community_aware",
                    community_aware_edge_search(
                        driver,
                        query_vector,
                        search_filter,
                        group_ids,
                        limit,
                        config.sim_min_score,
                    ),
                )
            )

        # Execute only the configured search methods
        search_results: list[list[EntityEdge]] = []
        method_results: list[tuple[str, list[EntityEdge]]] = []
        if search_tasks:
            with _trace_phase(
                search_tracer,
                'search.edge_search.execute_methods',
                {
                    'method_count': len(search_tasks),
                    'candidate_limit': 2 * limit,
                },
            ) as method_span:
                labels = [label for label, _ in search_tasks]
                raw_results = list(await semaphore_gather(*[task for _, task in search_tasks]))
                method_results = list(zip(labels, raw_results, strict=False))
                search_results = [result for _, result in method_results]
                method_span.add_attributes(
                    {
                        'result_set_count': len(search_results),
                        'non_empty_result_sets': sum(1 for result in search_results if result),
                    }
                )

        if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            source_node_uuids = [
                edge.source_node_uuid for result in search_results for edge in result
            ]
            with _trace_phase(
                search_tracer,
                'search.edge_search.expand_bfs',
                {
                    'origin_node_count': len(source_node_uuids),
                    'candidate_limit': 2 * limit,
                },
            ):
                search_results.append(
                    await edge_bfs_search(
                        driver,
                        source_node_uuids,
                        config.bfs_max_depth,
                        search_filter,
                        group_ids,
                        2 * limit,
                    )
                )
                method_results.append(("breadth_first_search_expanded", search_results[-1]))

        edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}
        edge_score_details: dict[str, dict[str, Any]] = {}
        for method_name, result in method_results:
            for index, edge in enumerate(result, start=1):
                detail = edge_score_details.setdefault(
                    edge.uuid,
                    {
                        "reranker": _enum_value(config.reranker),
                        "methods": {},
                    },
                )
                method_detail: dict[str, Any] = {"rank": index, "weight": 1.0}
                raw_score = _edge_method_raw_score(edge, method_name)
                if raw_score is not None:
                    method_detail["raw_score"] = raw_score
                if config.reranker == EdgeReranker.rrf:
                    method_detail["rrf_contribution"] = _rrf_contribution(index)
                detail["methods"][method_name] = method_detail

        reranked_uuids: list[str] = []
        edge_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.edge_search.rerank',
            {
                'candidate_count': len(edge_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if (
                config.reranker == EdgeReranker.rrf
                or config.reranker == EdgeReranker.episode_mentions
            ):
                search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

                reranked_uuids, edge_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            elif config.reranker == EdgeReranker.mmr:
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.load_embeddings',
                    {'candidate_count': len(edge_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_edges(
                        driver, list(edge_uuid_map.values())
                    )
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, edge_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            elif config.reranker == EdgeReranker.cross_encoder:
                fact_to_uuid_map = {
                    edge.fact: edge.uuid for edge in list(edge_uuid_map.values())[:limit]
                }
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.cross_encoder_rank',
                    {'candidate_count': len(fact_to_uuid_map)},
                ):
                    reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
                reranked_uuids = [
                    fact_to_uuid_map[fact]
                    for fact, score in reranked_facts
                    if score >= reranker_min_score
                ]
                edge_scores = [score for _, score in reranked_facts if score >= reranker_min_score]
            elif config.reranker == EdgeReranker.node_distance:
                if center_node_uuid is None:
                    raise SearchRerankerError('No center node provided for Node Distance reranker')

                with _trace_phase(
                    search_tracer,
                    'search.edge_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    sorted_result_uuids, _ = rrf(
                        [[edge.uuid for edge in result] for result in search_results],
                        min_score=reranker_min_score,
                    )
                sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

                source_to_edge_uuid_map = defaultdict(list)
                for edge in sorted_results:
                    source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

                source_uuids = [source_node_uuid for source_node_uuid in source_to_edge_uuid_map]

                with _trace_phase(
                    search_tracer,
                    'search.edge_search.node_distance_rank',
                    {
                        'source_node_count': len(source_uuids),
                        'center_node_uuid.provided': center_node_uuid is not None,
                    },
                ):
                    reranked_node_uuids, edge_scores = await node_distance_reranker(
                        driver, source_uuids, center_node_uuid, min_score=reranker_min_score
                    )

                for node_uuid in reranked_node_uuids:
                    reranked_uuids.extend(source_to_edge_uuid_map[node_uuid])

        reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]
        pre_recency_scores = {
            edge.uuid: float(score)
            for edge, score in zip(reranked_edges, edge_scores, strict=False)
        }

        if config.reranker == EdgeReranker.episode_mentions:
            reranked_edges.sort(reverse=True, key=lambda edge: len(edge.episodes))

        span.add_attributes(
            {
                'candidate_count': len(edge_uuid_map),
                'reranked_count': len(reranked_edges),
                'returned_count': min(len(reranked_edges), limit),
            }
        )

        reranked_edges, edge_scores = _rerank_edges_with_recency(
            reranked_edges[: 2 * limit],
            edge_scores[: 2 * limit],
            reference_time=query_reference_time,
        )
        pre_resolution_count = len(reranked_edges)
        reranked_edges, edge_scores = _resolve_single_state_edge_slots(
            reranked_edges,
            edge_scores,
            reference_time=query_reference_time,
            edge_score_details=edge_score_details,
        )
        span.add_attributes(
            {
                'single_state.pre_resolution_count': pre_resolution_count,
                'single_state.post_resolution_count': len(reranked_edges),
                'single_state.reference_time': (
                    query_reference_time.isoformat() if query_reference_time is not None else ''
                ),
            }
        )
        final_edges = reranked_edges[:limit]
        final_scores = edge_scores[:limit]
        for final_rank, (edge, score) in enumerate(zip(final_edges, final_scores, strict=False), start=1):
            detail = edge_score_details.setdefault(
                edge.uuid,
                {"reranker": _enum_value(config.reranker), "methods": {}},
            )
            ts = getattr(edge, 'created_at', None) or getattr(edge, 'valid_at', None)
            detail["pre_recency_score"] = pre_recency_scores.get(edge.uuid)
            detail["recency_weight"] = _recency_weight(ts, reference_time=query_reference_time)
            detail["query_reference_time"] = (
                query_reference_time.isoformat() if query_reference_time is not None else None
            )
            detail["final_score"] = float(score)
            detail["final_rank"] = final_rank
        _attach_edge_score_formula_trees(edge_score_details)
        return final_edges, final_scores, edge_score_details


async def node_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: NodeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EntityNode], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.node_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
            'bfs_origin_count': len(bfs_origin_node_uuids or []),
            'center_node_uuid.provided': center_node_uuid is not None,
        },
    ) as span:
        # Build search tasks based on configured search methods
        search_tasks = []
        if NodeSearchMethod.bm25 in config.search_methods:
            search_tasks.append(
                node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
            )
        if NodeSearchMethod.cosine_similarity in config.search_methods:
            search_tasks.append(
                node_similarity_search(
                    driver,
                    query_vector,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.sim_min_score,
                )
            )
        if NodeSearchMethod.bfs in config.search_methods:
            search_tasks.append(
                node_bfs_search(
                    driver,
                    bfs_origin_node_uuids,
                    search_filter,
                    config.bfs_max_depth,
                    group_ids,
                    2 * limit,
                )
            )

        # Execute only the configured search methods
        search_results: list[list[EntityNode]] = []
        if search_tasks:
            with _trace_phase(
                search_tracer,
                'search.node_search.execute_methods',
                {
                    'method_count': len(search_tasks),
                    'candidate_limit': 2 * limit,
                },
            ) as method_span:
                search_results = list(await semaphore_gather(*search_tasks))
                method_span.add_attributes(
                    {
                        'result_set_count': len(search_results),
                        'non_empty_result_sets': sum(1 for result in search_results if result),
                    }
                )

        if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            origin_node_uuids = [node.uuid for result in search_results for node in result]
            with _trace_phase(
                search_tracer,
                'search.node_search.expand_bfs',
                {
                    'origin_node_count': len(origin_node_uuids),
                    'candidate_limit': 2 * limit,
                },
            ):
                search_results.append(
                    await node_bfs_search(
                        driver,
                        origin_node_uuids,
                        search_filter,
                        config.bfs_max_depth,
                        group_ids,
                        2 * limit,
                    )
                )

        search_result_uuids = [[node.uuid for node in result] for result in search_results]
        node_uuid_map = {node.uuid: node for result in search_results for node in result}

        reranked_uuids: list[str] = []
        node_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.node_search.rerank',
            {
                'candidate_count': len(node_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if config.reranker == NodeReranker.rrf:
                reranked_uuids, node_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            elif config.reranker == NodeReranker.mmr:
                with _trace_phase(
                    search_tracer,
                    'search.node_search.load_embeddings',
                    {'candidate_count': len(node_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_nodes(
                        driver, list(node_uuid_map.values())
                    )

                with _trace_phase(
                    search_tracer,
                    'search.node_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, node_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            elif config.reranker == NodeReranker.cross_encoder:
                name_to_uuid_map = {node.name: node.uuid for node in list(node_uuid_map.values())}

                with _trace_phase(
                    search_tracer,
                    'search.node_search.cross_encoder_rank',
                    {'candidate_count': len(name_to_uuid_map)},
                ):
                    reranked_node_names = await cross_encoder.rank(
                        query, list(name_to_uuid_map.keys())
                    )
                reranked_uuids = [
                    name_to_uuid_map[name]
                    for name, score in reranked_node_names
                    if score >= reranker_min_score
                ]
                node_scores = [
                    score for _, score in reranked_node_names if score >= reranker_min_score
                ]
            elif config.reranker == NodeReranker.episode_mentions:
                with _trace_phase(
                    search_tracer,
                    'search.node_search.episode_mentions_rank',
                    {'candidate_count': len(node_uuid_map)},
                ):
                    reranked_uuids, node_scores = await episode_mentions_reranker(
                        driver, search_result_uuids, min_score=reranker_min_score
                    )
            elif config.reranker == NodeReranker.node_distance:
                if center_node_uuid is None:
                    raise SearchRerankerError('No center node provided for Node Distance reranker')
                with _trace_phase(
                    search_tracer,
                    'search.node_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    seeded_uuids = rrf(search_result_uuids, min_score=reranker_min_score)[0]
                with _trace_phase(
                    search_tracer,
                    'search.node_search.node_distance_rank',
                    {
                        'source_node_count': len(seeded_uuids),
                        'center_node_uuid.provided': center_node_uuid is not None,
                    },
                ):
                    reranked_uuids, node_scores = await node_distance_reranker(
                        driver,
                        seeded_uuids,
                        center_node_uuid,
                        min_score=reranker_min_score,
                    )

        reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

        span.add_attributes(
            {
                'candidate_count': len(node_uuid_map),
                'reranked_count': len(reranked_nodes),
                'returned_count': min(len(reranked_nodes), limit),
            }
        )

        return reranked_nodes[:limit], node_scores[:limit]


async def episode_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    _query_vector: list[float],
    group_ids: list[str] | None,
    config: EpisodeSearchConfig | None,
    search_filter: SearchFilters,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EpisodicNode], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.episode_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
        },
    ) as span:
        with _trace_phase(
            search_tracer,
            'search.episode_search.execute_methods',
            {'candidate_limit': 2 * limit},
        ):
            search_tasks = []
            if EpisodeSearchMethod.bm25 in config.search_methods:
                search_tasks.append(
                    episode_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
                )
            if EpisodeSearchMethod.cosine_similarity in config.search_methods:
                search_tasks.append(
                    episode_similarity_search(
                        driver,
                        _query_vector,
                        search_filter,
                        group_ids,
                        2 * limit,
                        config.sim_min_score,
                    )
                )
            search_results: list[list[EpisodicNode]] = list(await semaphore_gather(*search_tasks))

        search_result_uuids = [[episode.uuid for episode in result] for result in search_results]
        episode_uuid_map = {
            episode.uuid: episode for result in search_results for episode in result
        }
        if EpisodeSearchMethod.cosine_similarity in config.search_methods:
            episode_embeddings = {
                episode.uuid: episode.content_embedding
                for episode in episode_uuid_map.values()
                if episode.content_embedding is not None
            }
            if episode_embeddings:
                with _trace_phase(
                    search_tracer,
                    'search.episode_search.compute_mmr_signal',
                    {'candidate_count': len(episode_embeddings)},
                ):
                    mmr_episode_uuids, _ = maximal_marginal_relevance(
                        _query_vector,
                        episode_embeddings,
                        DEFAULT_MMR_LAMBDA,
                        reranker_min_score,
                    )
                if mmr_episode_uuids:
                    search_result_uuids.append(mmr_episode_uuids)

        reranked_uuids: list[str] = []
        episode_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.episode_search.rerank',
            {
                'candidate_count': len(episode_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if config.reranker == EpisodeReranker.rrf:
                reranked_uuids, episode_scores = rrf(
                    search_result_uuids, min_score=reranker_min_score
                )
            elif config.reranker == EpisodeReranker.cross_encoder:
                with _trace_phase(
                    search_tracer,
                    'search.episode_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    rrf_result_uuids, episode_scores = rrf(
                        search_result_uuids, min_score=reranker_min_score
                    )
                rrf_results = [episode_uuid_map[uuid] for uuid in rrf_result_uuids][:limit]

                content_to_uuid_map = {episode.content: episode.uuid for episode in rrf_results}

                with _trace_phase(
                    search_tracer,
                    'search.episode_search.cross_encoder_rank',
                    {'candidate_count': len(content_to_uuid_map)},
                ):
                    reranked_contents = await cross_encoder.rank(
                        query, list(content_to_uuid_map.keys())
                    )
                reranked_uuids = [
                    content_to_uuid_map[content]
                    for content, score in reranked_contents
                    if score >= reranker_min_score
                ]
                episode_scores = [
                    score for _, score in reranked_contents if score >= reranker_min_score
                ]

        reranked_episodes = [episode_uuid_map[uuid] for uuid in reranked_uuids]
        span.add_attributes(
            {
                'candidate_count': len(episode_uuid_map),
                'reranked_count': len(reranked_episodes),
                'returned_count': min(len(reranked_episodes), limit),
            }
        )

        return reranked_episodes[:limit], episode_scores[:limit]


async def community_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[CommunityNode], list[float]]:
    if config is None:
        return [], []
    search_tracer = _resolve_tracer(search_tracer)

    with _trace_phase(
        search_tracer,
        'search.community_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
        },
    ) as span:
        with _trace_phase(
            search_tracer,
            'search.community_search.execute_methods',
            {'candidate_limit': 2 * limit},
        ):
            search_results: list[list[CommunityNode]] = list(
                await semaphore_gather(
                    *[
                        community_fulltext_search(driver, query, group_ids, 2 * limit),
                        community_similarity_search(
                            driver, query_vector, group_ids, 2 * limit, config.sim_min_score
                        ),
                    ]
                )
            )

        search_result_uuids = [
            [community.uuid for community in result] for result in search_results
        ]
        community_uuid_map = {
            community.uuid: community for result in search_results for community in result
        }

        reranked_uuids: list[str] = []
        community_scores: list[float] = []
        with _trace_phase(
            search_tracer,
            'search.community_search.rerank',
            {
                'candidate_count': len(community_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            if config.reranker == CommunityReranker.rrf:
                reranked_uuids, community_scores = rrf(
                    search_result_uuids, min_score=reranker_min_score
                )
            elif config.reranker == CommunityReranker.mmr:
                with _trace_phase(
                    search_tracer,
                    'search.community_search.load_embeddings',
                    {'candidate_count': len(community_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_communities(
                        driver, list(community_uuid_map.values())
                    )

                with _trace_phase(
                    search_tracer,
                    'search.community_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, community_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            elif config.reranker == CommunityReranker.cross_encoder:
                name_to_uuid_map = {
                    node.name: node.uuid for result in search_results for node in result
                }
                with _trace_phase(
                    search_tracer,
                    'search.community_search.cross_encoder_rank',
                    {'candidate_count': len(name_to_uuid_map)},
                ):
                    reranked_nodes = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
                reranked_uuids = [
                    name_to_uuid_map[name]
                    for name, score in reranked_nodes
                    if score >= reranker_min_score
                ]
                community_scores = [
                    score for _, score in reranked_nodes if score >= reranker_min_score
                ]

        reranked_communities = [community_uuid_map[uuid] for uuid in reranked_uuids]
        span.add_attributes(
            {
                'candidate_count': len(community_uuid_map),
                'reranked_count': len(reranked_communities),
                'returned_count': min(len(reranked_communities), limit),
            }
        )

        return reranked_communities[:limit], community_scores[:limit]
