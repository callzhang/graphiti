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
            content='You are a fact deduplication assistant. '
            'NEVER mark facts with key differences as duplicates.',
        ),
        Message(
            role='user',
            content=f"""
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>

Guidelines:
1. Some facts may be very similar but will have key differences, particularly around numeric values.
   Do not mark these as duplicates.
2. **Be conservative about contradictions for mutable state.**
   Facts about targets, goals, ownership, leadership, status, decisions, or strategies
   may represent mutable state, but ONLY mark an older fact as contradicted when
   the NEW FACT clearly updates the SAME state dimension or slot.
   Examples:
   - target 60万→100万 can contradict an older target fact
   - owner Kevin→Derek can contradict an older owner fact
   - status "in progress"→"completed" can contradict an older status fact
   Counterexamples:
   - a general status note like "节奏稳，继续推进" does NOT contradict a target value
   - an owner update does NOT contradict a target/budget/status fact
   - do NOT mark facts as contradicted solely because they mention the same entity
     if they describe different relationship dimensions.

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: duplicate_facts=[], contradicted_facts=[1] (same relationship but updated title — contradiction, NOT a duplicate)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: duplicate_facts=[], contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
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
        2. **Be conservative about contradictions for mutable state.**
           Only mark older facts as contradicted when the NEW FACT clearly updates the SAME
           state dimension or slot.
           - target updates contradict older target facts
           - owner updates contradict older owner facts
           - status updates contradict older status facts
           Do NOT contradict facts across different relationship dimensions just because they
           mention the same entity.

        {edges_block}
        """,
        ),
    ]


versions: Versions = {'resolve_edge': resolve_edge, 'resolve_edges_batch': resolve_edges_batch}
