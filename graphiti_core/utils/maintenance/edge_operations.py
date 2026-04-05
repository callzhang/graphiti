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
from datetime import datetime
from time import time
from typing import Any

from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import compute_edge_cap, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact

logger = logging.getLogger(__name__)


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str,
    created_at: datetime,
) -> list[EpisodicEdge]:
    episodic_edges: list[EpisodicEdge] = [
        EpisodicEdge(
            source_node_uuid=episode_uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=node.group_id,
        )
        for node in entity_nodes
    ]

    logger.debug(f'Built {len(episodic_edges)} episodic edges')

    return episodic_edges


def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode,
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
) -> list[EntityEdge]:
    start = time()

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    # Build mapping from edge type name to list of valid signatures
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signatures': edge_type_signatures_map.get(
                    type_name, [('Entity', 'Entity')]
                ),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Build name-to-node mapping for validation
    name_to_node: dict[str, EntityNode] = {node.name: node for node in nodes}

    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'nodes': [{'name': node.name, 'entity_types': node.labels} for node in nodes],
        'previous_episodes': [ep.content for ep in previous_episodes],
        'reference_time': episode.valid_at,
        'edge_types': edge_types_context,
        'custom_extraction_instructions': custom_extraction_instructions or '',
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_edges.edge(context),
        response_model=ExtractedEdges,
        max_tokens=extract_edges_max_tokens,
        group_id=group_id,
        prompt_name='extract_edges.edge',
    )
    all_edges_data = ExtractedEdges(**llm_response).edges

    # Validate entity names
    edges_data: list[ExtractedEdge] = []
    for edge_data in all_edges_data:
        source_name = edge_data.source_entity_name
        target_name = edge_data.target_entity_name

        # Validate LLM-returned names exist in the nodes list
        if source_name not in name_to_node:
            logger.warning(
                'Source entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        if target_name not in name_to_node:
            logger.warning(
                'Target entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        # Drop self-edges where source and target resolve to the same node
        source_node = name_to_node[source_name]
        target_node = name_to_node[target_name]
        if source_node.uuid == target_node.uuid:
            logger.info(
                'Dropping self-edge for node %s (source and target resolve to same node)',
                source_node.uuid,
            )
            continue

        edges_data.append(edge_data)

    # Cap extracted edges to prevent runaway resolution costs
    max_edges = compute_edge_cap(len(episode.content or ''))
    if len(edges_data) > max_edges:
        logger.info(
            'Capping extracted edges from %d to %d (content_chars=%d)',
            len(edges_data),
            max_edges,
            len(episode.content or ''),
        )
        edges_data = edges_data[:max_edges]

    end = time()
    logger.debug(f'Extracted {len(edges_data)} new edges in {(end - start) * 1000:.0f} ms')

    if len(edges_data) == 0:
        return []

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        # Filter out empty edges
        if not edge_data.fact.strip():
            continue

        # Names already validated above
        source_node = name_to_node.get(edge_data.source_entity_name)
        target_node = name_to_node.get(edge_data.target_entity_name)

        if source_node is None or target_node is None:
            logger.warning('Could not find source or target node for extracted edge')
            continue

        source_node_uuid = source_node.uuid
        target_node_uuid = target_node.uuid

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing valid_at date: {e}. Input: {valid_at}')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')
        edge = EntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            group_id=group_id,
            fact=edge_data.fact,
            episodes=[episode.uuid],
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
            reference_time=episode.valid_at,
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge {edge.uuid} from {edge.source_node_uuid} to {edge.target_node_uuid}'
        )

    logger.debug(f'Extracted edges: {[e.uuid for e in edges]}')

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    existing_edges_override: list[EntityEdge] | None = None,
) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
    """Resolve extracted edges against existing graph context.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
        A tuple of (resolved_edges, invalidated_edges, new_edges) where:
        - resolved_edges: All edges after resolution (may include existing edges if duplicates found)
        - invalidated_edges: Edges that were invalidated/contradicted by new information
        - new_edges: Only edges that are new to the graph (not duplicates of existing edges)
    """
    # Fast path: deduplicate exact matches within the extracted edges before parallel processing
    seen: dict[tuple[str, str, str], EntityEdge] = {}
    deduplicated_edges: list[EntityEdge] = []

    for edge in extracted_edges:
        key = (
            edge.source_node_uuid,
            edge.target_node_uuid,
            _normalize_string_exact(edge.fact),
        )
        if key not in seen:
            seen[key] = edge
            deduplicated_edges.append(edge)

    extracted_edges = deduplicated_edges

    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder
    embedding_started = time()
    await create_entity_edge_embeddings(embedder, extracted_edges)
    embedding_done = time()

    between_nodes_started = time()
    valid_edges_list: list[list[EntityEdge]] = await semaphore_gather(
        *[
            EntityEdge.get_between_nodes(driver, edge.source_node_uuid, edge.target_node_uuid)
            for edge in extracted_edges
        ]
    )
    between_nodes_done = time()

    # Merge override edges (e.g. from the recent Redis dedup cache) into
    # the per-extracted-edge candidate lists so that recently resolved edges
    # that are not yet visible in the graph-service indexes are still
    # considered during deduplication.
    if existing_edges_override:
        override_by_pair: dict[tuple[str, str], list[EntityEdge]] = {}
        for oe in existing_edges_override:
            key = (oe.source_node_uuid, oe.target_node_uuid)
            override_by_pair.setdefault(key, []).append(oe)

        for i, extracted_edge in enumerate(extracted_edges):
            pair_key = (extracted_edge.source_node_uuid, extracted_edge.target_node_uuid)
            overrides = override_by_pair.get(pair_key, [])
            if overrides:
                existing_uuids = {e.uuid for e in valid_edges_list[i]}
                for oe in overrides:
                    if oe.uuid not in existing_uuids:
                        valid_edges_list[i].append(oe)
                        existing_uuids.add(oe.uuid)

    # Merged search: one unrestricted search per edge, then partition results
    # into related_edges (between same node pair) vs invalidation_candidates.
    # This replaces two sequential search stages with one, cutting search overhead ~50%.
    search_started = time()
    between_node_uuid_sets: list[set[str]] = [
        {edge.uuid for edge in valid_edges} for valid_edges in valid_edges_list
    ]
    broad_search_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),
            )
            for extracted_edge in extracted_edges
        ]
    )
    search_done = time()

    # Partition search results: edges between same node pair → related_edges,
    # the rest → invalidation_candidates.
    # Supplementary fallback: ensure between-node edges from DB always appear
    # in related_edges even if they fell outside the search top-K.
    related_edges_lists: list[list[EntityEdge]] = []
    edge_invalidation_candidates: list[list[EntityEdge]] = []
    for broad_result, between_uuids, valid_edges in zip(
        broad_search_results, between_node_uuid_sets, valid_edges_list, strict=True
    ):
        related: list[EntityEdge] = []
        invalidation: list[EntityEdge] = []
        seen_related_uuids: set[str] = set()
        for edge in broad_result.edges:
            if edge.uuid in between_uuids:
                related.append(edge)
                seen_related_uuids.add(edge.uuid)
            else:
                invalidation.append(edge)
        # Supplementary: add between-node edges that didn't appear in search top-K
        for db_edge in valid_edges:
            if db_edge.uuid not in seen_related_uuids:
                related.append(db_edge)
        related_edges_lists.append(related)
        edge_invalidation_candidates.append(invalidation)

    logger.debug(
        f'Related edges: {[e.uuid for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Collect all node UUIDs referenced by edges that are not in the entities list
    referenced_node_uuids = set()
    for extracted_edge in extracted_edges:
        if extracted_edge.source_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.source_node_uuid)
        if extracted_edge.target_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.target_node_uuid)

    # Fetch missing nodes from the database
    if referenced_node_uuids:
        missing_nodes = await EntityNode.get_by_uuids(driver, list(referenced_node_uuids))
        for node in missing_nodes:
            uuid_entity_map[node.uuid] = node

    # Determine which edge types are relevant for each edge based on node signatures.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge.
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
        source_node_labels = (
            source_node.labels + ['Entity'] if source_node is not None else ['Entity']
        )
        target_node_labels = (
            target_node.labels + ['Entity'] if target_node is not None else ['Entity']
        )
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # --- Batched edge resolution: fast paths first, then one batched LLM call ---
    # Phase 1: Apply fast paths (no candidates / exact match) per edge
    fast_resolved: dict[int, tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = {}
    batch_indices: list[int] = []
    batch_contexts: list[dict[str, Any]] = []

    for i, (extracted_edge, related_edges, existing_edges) in enumerate(
        zip(extracted_edges, related_edges_lists, edge_invalidation_candidates, strict=True)
    ):
        # Fast path: no candidates at all
        if len(related_edges) == 0 and len(existing_edges) == 0:
            fast_resolved[i] = (extracted_edge, [], [])
            continue

        # Fast path: exact fact+endpoints match
        normalized_fact = _normalize_string_exact(extracted_edge.fact)
        exact_match = None
        for edge in related_edges:
            if (
                edge.source_node_uuid == extracted_edge.source_node_uuid
                and edge.target_node_uuid == extracted_edge.target_node_uuid
                and _normalize_string_exact(edge.fact) == normalized_fact
            ):
                exact_match = edge
                break
        if exact_match is not None:
            resolved = exact_match
            if episode is not None and episode.uuid not in resolved.episodes:
                resolved.episodes.append(episode.uuid)
            fast_resolved[i] = (resolved, [], [])
            continue

        # Needs LLM — prepare context
        related_ctx = [{'idx': j, 'fact': e.fact} for j, e in enumerate(related_edges)]
        offset = len(related_edges)
        invalidation_ctx = [
            {'idx': offset + j, 'fact': e.fact} for j, e in enumerate(existing_edges)
        ]
        batch_indices.append(i)
        batch_contexts.append({
            'edge_idx': i,
            'new_edge': extracted_edge.fact,
            'existing_edges': related_ctx,
            'edge_invalidation_candidates': invalidation_ctx,
        })

    # Phase 2: Batched LLM call for all edges that need resolution
    batch_responses: dict[int, EdgeDuplicate] = {}
    if batch_contexts:
        from graphiti_core.prompts.dedupe_edges import BatchEdgeResolutions, EdgeResolution

        try:
            raw_response = await llm_client.generate_response(
                prompt_library.dedupe_edges.resolve_edges_batch({'edges': batch_contexts}),
                response_model=BatchEdgeResolutions,
                model_size=ModelSize.small,
                prompt_name='dedupe_edges.resolve_edges_batch',
            )
            batch_result = BatchEdgeResolutions(**raw_response)
            for resolution in batch_result.resolutions:
                batch_responses[resolution.edge_idx] = EdgeDuplicate(
                    duplicate_facts=resolution.duplicate_facts,
                    contradicted_facts=resolution.contradicted_facts,
                )
        except Exception:
            # Fallback: per-edge LLM calls if batched prompt fails
            logger.warning(
                'Batched edge resolution failed, falling back to per-edge calls for %d edges',
                len(batch_contexts),
            )
            per_edge_results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
                await semaphore_gather(
                    *[
                        resolve_extracted_edge(
                            llm_client,
                            extracted_edges[idx],
                            related_edges_lists[idx],
                            edge_invalidation_candidates[idx],
                            episode,
                            edge_types_lst[idx],
                        )
                        for idx in batch_indices
                    ]
                )
            )
            for idx, result in zip(batch_indices, per_edge_results, strict=True):
                fast_resolved[idx] = result
            batch_indices = []
            batch_responses = {}

    # Phase 3: Post-LLM processing for batched edges
    for idx in batch_indices:
        response = batch_responses.get(idx)
        extracted_edge = extracted_edges[idx]
        related_edges = related_edges_lists[idx]
        existing_edges_for_edge = edge_invalidation_candidates[idx]

        if response is None:
            # No response from batch — treat as new edge
            fast_resolved[idx] = (extracted_edge, [], [])
            continue

        # Validate and apply duplicate detection
        duplicate_fact_ids = [
            j for j in response.duplicate_facts if 0 <= j < len(related_edges)
        ]
        resolved_edge = extracted_edge
        for dup_id in duplicate_fact_ids:
            resolved_edge = related_edges[dup_id]
            break
        if duplicate_fact_ids and episode is not None:
            resolved_edge.episodes.append(episode.uuid)

        # Process contradictions
        invalidation_candidates: list[EntityEdge] = []
        max_valid_idx = len(related_edges) + len(existing_edges_for_edge) - 1
        invalidation_offset = len(related_edges)
        for cidx in response.contradicted_facts:
            if 0 <= cidx < len(related_edges):
                invalidation_candidates.append(related_edges[cidx])
            elif invalidation_offset <= cidx <= max_valid_idx:
                invalidation_candidates.append(
                    existing_edges_for_edge[cidx - invalidation_offset]
                )

        # Temporal contradiction resolution
        now = utc_now()
        if resolved_edge.invalid_at and not resolved_edge.expired_at:
            resolved_edge.expired_at = now
        if resolved_edge.expired_at is None:
            invalidation_candidates.sort(
                key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at))
            )
            for candidate in invalidation_candidates:
                candidate_valid_at_utc = ensure_utc(candidate.valid_at)
                resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
                if (
                    candidate_valid_at_utc is not None
                    and resolved_edge_valid_at_utc is not None
                    and candidate_valid_at_utc > resolved_edge_valid_at_utc
                ):
                    resolved_edge.invalid_at = candidate.valid_at
                    resolved_edge.expired_at = now
                    break

        invalidated = resolve_edge_contradictions(resolved_edge, invalidation_candidates)
        resolved_edge.attributes = {}
        fast_resolved[idx] = (resolved_edge, invalidated, [])

    # Phase 4: Custom attribute extraction (still per-edge, parallel)
    async def _extract_attrs(edge_idx: int, resolved_edge: EntityEdge) -> None:
        edge_model = edge_types_lst[edge_idx].get(resolved_edge.name) if edge_types_lst[edge_idx] else None
        if edge_model is not None and len(edge_model.model_fields) != 0:
            ctx = {
                'fact': resolved_edge.fact,
                'reference_time': episode.valid_at if episode is not None else None,
                'existing_attributes': resolved_edge.attributes,
            }
            resp = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(ctx),
                response_model=edge_model,
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_attributes',
            )
            resolved_edge.attributes = resp

    attr_tasks = []
    for idx in range(len(extracted_edges)):
        result = fast_resolved[idx]
        attr_tasks.append(_extract_attrs(idx, result[0]))
    if attr_tasks:
        await semaphore_gather(*attr_tasks)

    resolve_done = time()

    # Collect results in order
    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    new_edges: list[EntityEdge] = []
    for i, extracted_edge in enumerate(extracted_edges):
        result = fast_resolved[i]
        resolved_edge = result[0]
        invalidated_chunk = result[1]

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_chunk)

        if resolved_edge.uuid == extracted_edge.uuid:
            new_edges.append(resolved_edge)

    logger.debug(f'Resolved edges: {[e.uuid for e in resolved_edges]}')
    logger.debug(f'New edges (non-duplicates): {[e.uuid for e in new_edges]}')

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )
    reembed_done = time()

    logger.info(
        'resolve_extracted_edges phases extracted=%d embedded_ms=%.1f between_nodes_ms=%.1f merged_search_ms=%.1f resolve_ms=%.1f reembed_ms=%.1f',
        len(extracted_edges),
        (embedding_done - embedding_started) * 1000,
        (between_nodes_done - between_nodes_started) * 1000,
        (search_done - search_started) * 1000,
        (resolve_done - search_done) * 1000,
        (reembed_done - resolve_done) * 1000,
    )

    return resolved_edges, invalidated_edges, new_edges


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        edge_valid_at_utc = ensure_utc(edge.valid_at)
        resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)

        if (
            edge_invalid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        ) or (
            edge_valid_at_utc is not None
            and resolved_edge_invalid_at_utc is not None
            and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        ):
            continue
        # New edge invalidates edge
        elif (
            edge_valid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_valid_at_utc < resolved_edge_valid_at_utc
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """Resolve an extracted edge against existing graph context.

    Parameters
    ----------
    llm_client : LLMClient
        Client used to invoke the LLM for deduplication and attribute extraction.
    extracted_edge : EntityEdge
        Newly extracted edge whose canonical representation is being resolved.
    related_edges : list[EntityEdge]
        Candidate edges with identical endpoints used for duplicate detection.
    existing_edges : list[EntityEdge]
        Broader set of edges evaluated for contradiction / invalidation.
    episode : EpisodicNode
        Episode providing content context when extracting edge attributes.
    edge_type_candidates : dict[str, type[BaseModel]] | None
        Custom edge types permitted for the current source/target signature.

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any duplicates, and edges to invalidate.
    """
    if len(related_edges) == 0 and len(existing_edges) == 0:
        # Still extract custom attributes even when no dedup/invalidation is needed
        edge_model = edge_type_candidates.get(extracted_edge.name) if edge_type_candidates else None
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_context = {
                'fact': extracted_edge.fact,
                'reference_time': episode.valid_at if episode is not None else None,
                'existing_attributes': extracted_edge.attributes,
            }
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_attributes',
            )
            extracted_edge.attributes = edge_attributes_response

        return extracted_edge, [], []

    # Fast path: if the fact text and endpoints already exist verbatim, reuse the matching edge.
    normalized_fact = _normalize_string_exact(extracted_edge.fact)
    for edge in related_edges:
        if (
            edge.source_node_uuid == extracted_edge.source_node_uuid
            and edge.target_node_uuid == extracted_edge.target_node_uuid
            and _normalize_string_exact(edge.fact) == normalized_fact
        ):
            resolved = edge
            if episode is not None and episode.uuid not in resolved.episodes:
                resolved.episodes.append(episode.uuid)
            return resolved, [], []

    start = time()

    # Prepare context for LLM with continuous indexing
    related_edges_context = [{'idx': i, 'fact': edge.fact} for i, edge in enumerate(related_edges)]

    # Invalidation candidates start where duplicate candidates end
    invalidation_idx_offset = len(related_edges)
    invalidation_edge_candidates_context = [
        {'idx': invalidation_idx_offset + i, 'fact': existing_edge.fact}
        for i, existing_edge in enumerate(existing_edges)
    ]

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
    }

    if related_edges or existing_edges:
        logger.debug(
            'Resolving edge: sent %d EXISTING FACTS%s and %d INVALIDATION CANDIDATES%s',
            len(related_edges),
            f' (idx 0-{len(related_edges) - 1})' if related_edges else '',
            len(existing_edges),
            f' (idx {invalidation_idx_offset}-{invalidation_idx_offset + len(existing_edges) - 1})'
            if existing_edges
            else '',
        )

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.resolve_edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
        prompt_name='dedupe_edges.resolve_edge',
    )
    response_object = EdgeDuplicate(**llm_response)
    duplicate_facts = response_object.duplicate_facts

    # Validate duplicate_facts are in valid range for EXISTING FACTS
    invalid_duplicates = [i for i in duplicate_facts if i < 0 or i >= len(related_edges)]
    if invalid_duplicates:
        logger.warning(
            'LLM returned invalid duplicate_facts idx values %s (valid range: 0-%d for EXISTING FACTS)',
            invalid_duplicates,
            len(related_edges) - 1,
        )

    duplicate_fact_ids: list[int] = [i for i in duplicate_facts if 0 <= i < len(related_edges)]

    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and episode is not None:
        resolved_edge.episodes.append(episode.uuid)

    # Process contradicted facts (continuous indexing across both lists)
    contradicted_facts: list[int] = response_object.contradicted_facts
    invalidation_candidates: list[EntityEdge] = []

    # Only process contradictions if there are edges to check against
    if related_edges or existing_edges:
        max_valid_idx = len(related_edges) + len(existing_edges) - 1
        invalid_contradictions = [i for i in contradicted_facts if i < 0 or i > max_valid_idx]
        if invalid_contradictions:
            logger.warning(
                'LLM returned invalid contradicted_facts idx values %s (valid range: 0-%d)',
                invalid_contradictions,
                max_valid_idx,
            )

        # Split contradicted facts into those from related_edges vs existing_edges based on offset
        for idx in contradicted_facts:
            if 0 <= idx < len(related_edges):
                # From EXISTING FACTS (duplicate candidates)
                invalidation_candidates.append(related_edges[idx])
            elif invalidation_idx_offset <= idx <= max_valid_idx:
                # From FACT INVALIDATION CANDIDATES (adjust index by offset)
                invalidation_candidates.append(existing_edges[idx - invalidation_idx_offset])

    # Only extract structured attributes if the edge's relation_type matches an allowed custom type
    # AND the edge model exists for this node pair signature
    edge_model = edge_type_candidates.get(resolved_edge.name) if edge_type_candidates else None
    if edge_model is not None and len(edge_model.model_fields) != 0:
        edge_attributes_context = {
            'fact': resolved_edge.fact,
            'reference_time': episode.valid_at if episode is not None else None,
            'existing_attributes': resolved_edge.attributes,
        }

        edge_attributes_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_attributes(edge_attributes_context),
            response_model=edge_model,  # type: ignore
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_attributes',
        )

        resolved_edge.attributes = edge_attributes_response
    else:
        resolved_edge.attributes = {}

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.uuid} -> {resolved_edge.uuid}, in {(end - start) * 1000} ms'
    )

    now = utc_now()

    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at)))
        for candidate in invalidation_candidates:
            candidate_valid_at_utc = ensure_utc(candidate.valid_at)
            resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
            if (
                candidate_valid_at_utc is not None
                and resolved_edge_valid_at_utc is not None
                and candidate_valid_at_utc > resolved_edge_valid_at_utc
            ):
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = resolve_edge_contradictions(
        resolved_edge, invalidation_candidates
    )
    duplicate_edges: list[EntityEdge] = [related_edges[idx] for idx in duplicate_fact_ids]

    return resolved_edge, invalidated_edges, duplicate_edges


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    if not duplicates_node_tuples:
        return []

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    if driver.provider == GraphProvider.NEPTUNE:
        query: LiteralString = """
            UNWIND $duplicate_node_uuids AS duplicate_tuple
            MATCH (n:Entity {uuid: duplicate_tuple.source})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple.target})
            RETURN DISTINCT
                n.uuid AS source_uuid,
                m.uuid AS target_uuid
        """

        duplicate_nodes = [
            {'source': source.uuid, 'target': target.uuid}
            for source, target in duplicates_node_tuples
        ]

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_nodes,
            routing_='r',
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            query = """
                UNWIND $duplicate_node_uuids AS duplicate
                MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = [{'src': src, 'dst': dst} for src, dst in duplicate_nodes_map]
        else:
            query: LiteralString = """
                UNWIND $duplicate_node_uuids AS duplicate_tuple
                MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = list(duplicate_nodes_map.keys())

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_node_uuids,
            routing_='r',
        )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
