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

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='List of idx values of duplicate facts (only from EXISTING FACTS range). Empty list if none.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of idx values of contradicted facts (from full idx range). Empty list if none.',
    )


class EdgeResolution(BaseModel):
    edge_idx: int = Field(..., description='Index of the new edge being resolved')
    duplicate_facts: list[int] = Field(
        ...,
        description='List of candidate idx values that are duplicates (only from EXISTING FACTS range). Empty list if none.',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='List of candidate idx values that are contradicted (from full idx range). Empty list if none.',
    )


class BatchEdgeResolutions(BaseModel):
    resolutions: list[EdgeResolution] = Field(
        ..., description='One resolution per new edge, in the order they appear in the input.'
    )


class Prompt(Protocol):
    resolve_edge: PromptVersion
    resolve_edges_batch: PromptVersion


class Versions(TypedDict):
    resolve_edge: PromptFunction
    resolve_edges_batch: PromptFunction


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates facts from fact lists and determines which existing '
            'facts are contradicted by the new fact.',
        ),
        Message(
            role='user',
            content=f"""
        Task:
        You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
        EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

        1. DUPLICATE DETECTION:
           - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
           - Facts with similar information that contain key differences should NOT be marked as duplicates.
           - If no duplicates, return an empty list for duplicate_facts.

        2. CONTRADICTION DETECTION:
           - Determine which facts the NEW FACT contradicts from either list.
           - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
           - Return all contradicted idx values in contradicted_facts.
           - If no contradictions, return an empty list for contradicted_facts.

        IMPORTANT:
        - duplicate_facts: ONLY idx values from EXISTING FACTS (cannot include FACT INVALIDATION CANDIDATES)
        - contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
        - The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

        Guidelines:
        1. Some facts may be very similar but will have key differences, particularly around numeric values.
           Do not mark these as duplicates.
        2. **State facts are unique per entity at any point in time.**
           Facts about targets, goals, ownership, leadership, status, decisions, or strategies
           represent mutable state — only the latest value is valid.
           If the NEW FACT updates the same entity's state (e.g., target 60万→100万,
           owner Kevin→Derek, status "in progress"→"completed"), mark ALL older
           versions in EXISTING FACTS as contradicted, even if their wording differs.
           This applies regardless of whether they use the same relation type name.

        <EXISTING FACTS>
        {context['existing_edges']}
        </EXISTING FACTS>

        <FACT INVALIDATION CANDIDATES>
        {context['edge_invalidation_candidates']}
        </FACT INVALIDATION CANDIDATES>

        <NEW FACT>
        {context['new_edge']}
        </NEW FACT>
        """,
        ),
    ]


def resolve_edges_batch(context: dict[str, Any]) -> list[Message]:
    edges_block = '\n---\n'.join(
        f"""EDGE {entry['edge_idx']}:
NEW FACT: {entry['new_edge']}

EXISTING FACTS:
{entry['existing_edges']}

FACT INVALIDATION CANDIDATES:
{entry['edge_invalidation_candidates']}"""
        for entry in context['edges']
    )
    num_edges = len(context['edges'])
    return [
        Message(
            role='system',
            content='You are a helpful assistant that de-duplicates facts from fact lists and determines which existing '
            'facts are contradicted by new facts.',
        ),
        Message(
            role='user',
            content=f"""
        Task:
        You will receive {num_edges} NEW FACTS. Each new fact has its own EXISTING FACTS and FACT INVALIDATION CANDIDATES
        with CONTINUOUS idx numbering within each edge's candidate lists.

        For EACH new edge, determine:

        1. DUPLICATE DETECTION:
           - If the NEW FACT represents identical factual information as any fact in its EXISTING FACTS, return those idx values in duplicate_facts.
           - Facts with similar information that contain key differences should NOT be marked as duplicates.
           - If no duplicates, return an empty list for duplicate_facts.

        2. CONTRADICTION DETECTION:
           - Determine which facts the NEW FACT contradicts from either of its lists.
           - Return all contradicted idx values in contradicted_facts.
           - If no contradictions, return an empty list for contradicted_facts.

        IMPORTANT:
        - duplicate_facts: ONLY idx values from that edge's EXISTING FACTS
        - contradicted_facts: idx values from EITHER list for that edge
        - The idx values are continuous within each edge (INVALIDATION CANDIDATES start where EXISTING FACTS end)
        - Return EXACTLY {num_edges} resolutions with edge_idx values matching the input

        Guidelines:
        1. Some facts may be very similar but will have key differences, particularly around numeric values.
           Do not mark these as duplicates.
        2. **State facts are unique per entity at any point in time.**
           Facts about targets, goals, ownership, leadership, status, decisions, or strategies
           represent mutable state — only the latest value is valid.
           If the NEW FACT updates the same entity's state (e.g., target 60万→100万,
           owner Kevin→Derek, status "in progress"→"completed"), mark ALL older
           versions as contradicted, even if their wording or relation type differs.

        {edges_block}
        """,
        ),
    ]


versions: Versions = {'resolve_edge': resolve_edge, 'resolve_edges_batch': resolve_edges_batch}
