"""Phase 4: reading entities and relationships out of chunks.

The sharpest risk here is attribution. Five chunks share one prompt (D14), so
the model has to say which chunk each triple came from; if that mapping is
wrong, `source_chunk_id` is a guess and R6's citations point at text that does
not support them. A wrong attribution is worse than a missing relationship,
which is why the out-of-range and negative-index cases are tested explicitly.
"""

from __future__ import annotations

import pytest

from app.config import EXTRACTION_BATCH_SIZE
from app.errors import LLMQuotaExceeded, LLMRateLimited
from app.graph.base import make_entity_id
from app.graph.extractor import (
    ENTITY_TYPES,
    RELATION_TYPES,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    build_extraction_prompt,
    extract_from_chunks,
    resolve_extraction,
)
from app.graph.memory_store import InMemoryGraphStore
from app.llm.fake import FakeLLM
from app.schemas import Chunk


def _chunks(texts: list[str], doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc_id}:{i}", doc_id=doc_id, index=i, text=text)
        for i, text in enumerate(texts)
    ]


def _result(*, entities=(), relations=()) -> ExtractionResult:
    return ExtractionResult(entities=list(entities), relations=list(relations))


def _entity(name: str, kind: str = "PERSON", index: int = 0) -> ExtractedEntity:
    return ExtractedEntity(name=name, type=kind, source_chunk_index=index)


def _relation(
    source: str, target: str, kind: str = "works_at", index: int = 0, **kwargs
) -> ExtractedRelation:
    return ExtractedRelation(
        source_name=source,
        source_type=kwargs.pop("source_type", "PERSON"),
        target_name=target,
        target_type=kwargs.pop("target_type", "ORG"),
        type=kind,
        source_chunk_index=index,
        **kwargs,
    )


HIRE = "Helios Labs announced that Sarah Chen joins as VP of Engineering."
DEAL = "Northwind Systems completed its acquisition of Helios Labs."


# --- the constrained vocabulary -------------------------------------------


def test_the_entity_vocabulary_is_closed() -> None:
    """An open vocabulary makes the graph unqueryable."""
    assert set(ENTITY_TYPES) == {
        "PERSON",
        "ORG",
        "PRODUCT",
        "SERVICE",
        "TECHNOLOGY",
        "LOCATION",
        "EVENT",
        "CONCEPT",
    }


def test_the_relation_vocabulary_has_an_escape_hatch() -> None:
    """Fixed types, plus OTHER so a real relationship is never discarded."""
    assert "OTHER" in RELATION_TYPES
    assert "works_at" in RELATION_TYPES
    assert "acquired_by" in RELATION_TYPES


def test_agency_client_relation_types_are_available() -> None:
    """Client work has somewhere specific to go, instead of being forced
    into `produces`/`collaborates_with` (the live bug this guards: a client
    receiving a service was extracted as "Agency09 produces BITS Pilani")."""
    assert "built" in RELATION_TYPES
    assert "delivered_service_to" in RELATION_TYPES
    assert "received_service" in RELATION_TYPES
    assert "used_technology" in RELATION_TYPES
    assert "SERVICE" in ENTITY_TYPES
    assert "TECHNOLOGY" in ENTITY_TYPES


def test_an_unknown_entity_type_is_rejected_by_the_schema() -> None:
    with pytest.raises(Exception):
        ExtractedEntity(name="x", type="SPACESHIP", source_chunk_index=0)


# --- the prompt -----------------------------------------------------------


def test_the_prompt_labels_each_chunk_with_its_index() -> None:
    """The label is how the model can attribute a triple to a chunk."""
    prompt = build_extraction_prompt(_chunks([HIRE, DEAL]))

    assert "[0]" in prompt
    assert "[1]" in prompt
    assert HIRE in prompt
    assert DEAL in prompt


def test_the_prompt_numbers_chunks_from_zero() -> None:
    prompt = build_extraction_prompt(_chunks([HIRE]))

    assert "[0]" in prompt
    assert "[1]" not in prompt


def test_the_model_is_told_the_allowed_types() -> None:
    """In the system prompt, not the user prompt.

    A fixed vocabulary belongs in the system instruction; repeating it in every
    per-batch prompt would spend tokens on every call for no gain.
    """
    from app.graph.extractor import SYSTEM_PROMPT

    for entity_type in ENTITY_TYPES:
        assert entity_type in SYSTEM_PROMPT
    for relation_type in RELATION_TYPES:
        assert relation_type in SYSTEM_PROMPT


def test_the_system_prompt_insists_on_correct_attribution() -> None:
    """The one instruction that R6's citations depend on."""
    from app.graph.extractor import SYSTEM_PROMPT

    assert "source_chunk_index" in SYSTEM_PROMPT
    assert "do not infer" in SYSTEM_PROMPT.lower()


# --- attribution: the part that must not be wrong ------------------------


def test_a_relation_is_attributed_to_the_chunk_it_names() -> None:
    chunks = _chunks([HIRE, DEAL])
    result = _result(
        entities=[
            _entity("Sarah Chen", "PERSON", 0),
            _entity("Helios Labs", "ORG", 0),
            _entity("Northwind Systems", "ORG", 1),
        ],
        relations=[
            _relation("Sarah Chen", "Helios Labs", "works_at", index=0),
            _relation(
                "Helios Labs",
                "Northwind Systems",
                "acquired_by",
                index=1,
                source_type="ORG",
            ),
        ],
    )

    entities, relations = resolve_extraction(result, chunks)

    by_type = {r.type: r for r in relations}
    assert by_type["works_at"].source_chunk_id == "doc1:0"
    assert by_type["acquired_by"].source_chunk_id == "doc1:1"
    assert len(entities) == 3


def test_a_shared_service_resolves_to_its_own_entity_not_a_client_edge() -> None:
    """The scenario `produces`/`collaborates_with` used to mangle: a service
    delivered to two different clients should resolve as ORG--built-->SERVICE,
    ORG--delivered_service_to-->ORG (vendor to client, direct) for each
    client, and ORG--received_service-->SERVICE for each client too - giving
    the two clients a real two-hop link through the service, via
    received_service, not through the agency."""
    delivery = (
        "Agency09 built CMS Implementation and delivered it to BITS Pilani "
        "and Ryan Group."
    )
    chunks = _chunks([delivery])
    result = _result(
        entities=[
            _entity("Agency09", "ORG", 0),
            _entity("CMS Implementation", "SERVICE", 0),
            _entity("BITS Pilani", "ORG", 0),
            _entity("Ryan Group", "ORG", 0),
        ],
        relations=[
            _relation(
                "Agency09", "CMS Implementation", "built", index=0,
                source_type="ORG", target_type="SERVICE",
            ),
            _relation(
                "Agency09", "BITS Pilani", "delivered_service_to", index=0,
                source_type="ORG", target_type="ORG",
            ),
            _relation(
                "Agency09", "Ryan Group", "delivered_service_to", index=0,
                source_type="ORG", target_type="ORG",
            ),
            _relation(
                "BITS Pilani", "CMS Implementation", "received_service", index=0,
                source_type="ORG", target_type="SERVICE",
            ),
            _relation(
                "Ryan Group", "CMS Implementation", "received_service", index=0,
                source_type="ORG", target_type="SERVICE",
            ),
        ],
    )

    entities, relations = resolve_extraction(result, chunks)

    types_by_name = {e.name: e.type for e in entities}
    assert types_by_name["CMS Implementation"] == "SERVICE"
    rel_types = {r.type for r in relations}
    assert rel_types == {"built", "delivered_service_to", "received_service"}
    assert "produces" not in rel_types
    assert "collaborates_with" not in rel_types

    received = [r for r in relations if r.type == "received_service"]
    assert {r.source_id for r in received} == {
        make_entity_id("ORG", "BITS Pilani"),
        make_entity_id("ORG", "Ryan Group"),
    }
    assert {r.target_id for r in received} == {make_entity_id("SERVICE", "CMS Implementation")}

    delivered = [r for r in relations if r.type == "delivered_service_to"]
    assert {r.source_id for r in delivered} == {make_entity_id("ORG", "Agency09")}
    assert {r.target_id for r in delivered} == {
        make_entity_id("ORG", "BITS Pilani"),
        make_entity_id("ORG", "Ryan Group"),
    }


def test_an_out_of_range_chunk_index_is_dropped() -> None:
    """The model claiming chunk 9 of a 2-chunk batch must not be believed."""
    chunks = _chunks([HIRE, DEAL])
    result = _result(
        entities=[_entity("Sarah Chen", "PERSON", 0)],
        relations=[_relation("Sarah Chen", "Helios Labs", index=9)],
    )

    _entities, relations = resolve_extraction(result, chunks)

    assert relations == []


def test_a_negative_chunk_index_is_dropped() -> None:
    chunks = _chunks([HIRE])
    result = _result(entities=[_entity("Sarah Chen", "PERSON", -1)])

    entities, _relations = resolve_extraction(result, chunks)

    assert entities == []


def test_entities_carry_the_chunk_that_mentioned_them() -> None:
    chunks = _chunks([HIRE, DEAL])
    result = _result(entities=[_entity("Northwind Systems", "ORG", 1)])

    entities, _relations = resolve_extraction(result, chunks)

    assert entities[0].mention_chunk_ids == ("doc1:1",)


def test_entity_ids_are_built_from_type_and_name() -> None:
    chunks = _chunks([HIRE])
    result = _result(entities=[_entity("Sarah Chen", "PERSON", 0)])

    entities, _relations = resolve_extraction(result, chunks)

    assert entities[0].entity_id == make_entity_id("PERSON", "Sarah Chen")


def test_a_relation_endpoint_gets_an_entity_even_if_not_listed_separately() -> None:
    """The model often names a relationship without listing both endpoints.

    Dropping those would lose the edge, since the store refuses a dangling
    relationship.
    """
    chunks = _chunks([HIRE])
    result = _result(
        relations=[_relation("Sarah Chen", "Helios Labs", "works_at", index=0)]
    )

    entities, relations = resolve_extraction(result, chunks)

    assert {e.entity_id for e in entities} == {
        make_entity_id("PERSON", "Sarah Chen"),
        make_entity_id("ORG", "Helios Labs"),
    }
    assert len(relations) == 1


def test_a_relation_pointing_at_itself_is_dropped() -> None:
    chunks = _chunks([HIRE])
    result = _result(
        relations=[
            _relation("Helios Labs", "Helios Labs", source_type="ORG", target_type="ORG")
        ]
    )

    _entities, relations = resolve_extraction(result, chunks)

    assert relations == []


def test_relation_ids_are_deterministic() -> None:
    """Re-extracting the same text must not create a duplicate edge."""
    chunks = _chunks([HIRE])
    result = _result(relations=[_relation("Sarah Chen", "Helios Labs", index=0)])

    first = resolve_extraction(result, chunks)[1][0]
    second = resolve_extraction(result, chunks)[1][0]

    assert first.rel_id == second.rel_id


def test_an_empty_name_is_dropped() -> None:
    chunks = _chunks([HIRE])
    result = _result(entities=[_entity("   ", "PERSON", 0)])

    entities, _relations = resolve_extraction(result, chunks)

    assert entities == []


# --- batching -------------------------------------------------------------


def test_chunks_are_batched_rather_than_sent_one_by_one() -> None:
    """D14's quota budget: ~40 calls for 200 chunks, not 200."""
    provider = FakeLLM()
    chunks = _chunks([f"text {i}" for i in range(EXTRACTION_BATCH_SIZE * 3)])
    for _ in range(3):
        provider.queue(_result())

    extract_from_chunks(chunks, provider=provider, store=InMemoryGraphStore())

    assert len(provider.calls) == 3


def test_a_partial_final_batch_is_still_sent() -> None:
    provider = FakeLLM()
    chunks = _chunks([f"text {i}" for i in range(EXTRACTION_BATCH_SIZE + 1)])
    for _ in range(2):
        provider.queue(_result())

    extract_from_chunks(chunks, provider=provider, store=InMemoryGraphStore())

    assert len(provider.calls) == 2


def test_no_chunks_makes_no_call() -> None:
    provider = FakeLLM()

    summary = extract_from_chunks([], provider=provider, store=InMemoryGraphStore())

    assert provider.calls == []
    assert summary.chunks_extracted == 0


def test_the_extraction_model_and_minimal_reasoning_are_used() -> None:
    provider = FakeLLM()
    provider.queue(_result())

    extract_from_chunks(_chunks([HIRE]), provider=provider, store=InMemoryGraphStore())

    call = provider.calls[0]
    assert call["model"] == "test-flash-lite-model"
    assert call["minimal_reasoning"] is True


# --- writing to the graph -------------------------------------------------


def test_entities_and_relations_reach_the_store() -> None:
    store = InMemoryGraphStore()
    provider = FakeLLM()
    provider.queue(
        _result(
            entities=[
                _entity("Sarah Chen", "PERSON", 0),
                _entity("Helios Labs", "ORG", 0),
            ],
            relations=[_relation("Sarah Chen", "Helios Labs", "works_at", index=0)],
        )
    )

    summary = extract_from_chunks(_chunks([HIRE]), provider=provider, store=store)

    assert store.stats() == {"entity_count": 2, "relation_count": 1}
    assert summary.entities_written == 2
    assert summary.relations_written == 1


def test_the_same_entity_across_two_chunks_becomes_one_node() -> None:
    """Helios in both chunks is one organisation, mentioned twice."""
    store = InMemoryGraphStore()
    provider = FakeLLM()
    provider.queue(
        _result(
            entities=[
                _entity("Helios Labs", "ORG", 0),
                _entity("Helios Labs", "ORG", 1),
            ]
        )
    )

    extract_from_chunks(_chunks([HIRE, DEAL]), provider=provider, store=store)

    assert store.stats()["entity_count"] == 1
    stored = store.get_entity(make_entity_id("ORG", "Helios Labs"))
    assert set(stored.mention_chunk_ids) == {"doc1:0", "doc1:1"}


def test_a_three_hop_chain_is_walkable_after_extraction() -> None:
    """The end-to-end point of the phase: the graph answers multi-hop."""
    store = InMemoryGraphStore()
    provider = FakeLLM()
    provider.queue(
        _result(
            relations=[
                _relation("Sarah Chen", "Helios Labs", "works_at", index=0),
                _relation(
                    "Helios Labs",
                    "Northwind Systems",
                    "acquired_by",
                    index=1,
                    source_type="ORG",
                ),
                _relation(
                    "Northwind Systems",
                    "Atlas",
                    "produces",
                    index=1,
                    source_type="ORG",
                    target_type="PRODUCT",
                ),
            ]
        )
    )

    extract_from_chunks(_chunks([HIRE, DEAL]), provider=provider, store=store)

    paths = store.shortest_paths(
        make_entity_id("PERSON", "Sarah Chen"),
        make_entity_id("PRODUCT", "Atlas"),
        max_hops=3,
    )

    assert paths, "Sarah Chen should reach Atlas in three hops"
    assert [r.type for r in paths[0].edges] == ["works_at", "acquired_by", "produces"]


# --- failures: one batch vs the whole document ---------------------------


def test_a_failed_batch_is_skipped_and_the_rest_continue() -> None:
    """One bad batch must not lose the whole document."""
    store = InMemoryGraphStore()
    provider = FakeLLM()
    provider.queue(ValueError("the model returned nonsense"))
    provider.queue(_result(entities=[_entity("Sarah Chen", "PERSON", 0)]))

    chunks = _chunks([f"text {i}" for i in range(EXTRACTION_BATCH_SIZE + 1)])
    summary = extract_from_chunks(chunks, provider=provider, store=store)

    assert summary.batches_failed == 1
    assert store.stats()["entity_count"] == 1


def test_a_daily_quota_error_stops_the_whole_run() -> None:
    """Retrying a daily cap is pointless; the job must stop and say why."""
    store = InMemoryGraphStore()
    provider = FakeLLM()
    provider.queue(LLMQuotaExceeded("daily limit reached"))
    provider.queue(_result(entities=[_entity("Never Reached", "PERSON", 0)]))

    chunks = _chunks([f"text {i}" for i in range(EXTRACTION_BATCH_SIZE + 1)])

    with pytest.raises(LLMQuotaExceeded):
        extract_from_chunks(chunks, provider=provider, store=store)

    assert len(provider.calls) == 1, "must not keep calling after a daily cap"


def test_a_rate_limit_is_not_treated_as_a_dead_end() -> None:
    """Per-minute limits are already retried inside the provider.

    By the time one reaches here the retries are exhausted, so the batch is
    skipped like any other failure rather than killing the run.
    """
    store = InMemoryGraphStore()
    provider = FakeLLM()
    provider.queue(LLMRateLimited("slow down"))
    provider.queue(_result(entities=[_entity("Sarah Chen", "PERSON", 0)]))

    chunks = _chunks([f"text {i}" for i in range(EXTRACTION_BATCH_SIZE + 1)])
    summary = extract_from_chunks(chunks, provider=provider, store=store)

    assert summary.batches_failed == 1
    assert store.stats()["entity_count"] == 1


# --- the cache: a re-upload must not spend quota -------------------------


def test_already_extracted_chunks_are_skipped(store_with_cache) -> None:
    """Journey 2: re-uploading the same document costs nothing."""
    store, cache = store_with_cache
    provider = FakeLLM()
    provider.queue(_result(entities=[_entity("Sarah Chen", "PERSON", 0)]))

    chunks = _chunks([HIRE])
    extract_from_chunks(chunks, provider=provider, store=store, cache=cache)
    first_calls = len(provider.calls)

    extract_from_chunks(chunks, provider=provider, store=store, cache=cache)

    assert first_calls == 1
    assert len(provider.calls) == 1, "second run must make no LLM call"


def test_only_the_new_chunks_of_a_grown_document_are_extracted(
    store_with_cache,
) -> None:
    store, cache = store_with_cache
    provider = FakeLLM()
    provider.queue(_result())
    provider.queue(_result())

    extract_from_chunks(_chunks([HIRE]), provider=provider, store=store, cache=cache)
    extract_from_chunks(
        _chunks([HIRE, DEAL]), provider=provider, store=store, cache=cache
    )

    assert len(provider.calls) == 2
    # The second prompt must contain only the chunk that was not done before.
    assert DEAL in provider.calls[1]["prompt"]
    assert HIRE not in provider.calls[1]["prompt"]


def test_a_failed_batch_is_not_marked_extracted(store_with_cache) -> None:
    """Otherwise a transient failure permanently loses those chunks."""
    store, cache = store_with_cache
    provider = FakeLLM()
    provider.queue(ValueError("boom"))
    provider.queue(_result(entities=[_entity("Sarah Chen", "PERSON", 0)]))

    chunks = _chunks([HIRE])
    extract_from_chunks(chunks, provider=provider, store=store, cache=cache)
    extract_from_chunks(chunks, provider=provider, store=store, cache=cache)

    assert len(provider.calls) == 2, "a failed batch must be retried next time"
    assert store.stats()["entity_count"] == 1
