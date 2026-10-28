"""Phase 11 bug fix: entity dedup was built and tested (Phase 5) but never
called from the upload pipeline - `resolve_duplicates` existed, nothing
invoked it, so aliased mentions of the same org/person across documents (or
even within one) stayed as separate graph nodes forever.

These tests exercise `run_job` directly, not through the API, since dedup's
effect - a merged entity, not a job-status field - is only visible on the
store it was wired to run against.
"""

from __future__ import annotations

from app.graph.extractor import ExtractedEntity, ExtractedRelation, ExtractionResult
from app.graph.extractor import InMemoryExtractionCache
from app.graph.memory_store import InMemoryGraphStore
from app.ingest.pipeline import run_job
from app.jobs.registry import JOB_STATE_SUCCEEDED, JobRegistry
from app.llm.fake import FakeLLM
from app.schemas import Chunk
from app.vector.fake import FakeEmbedder
from app.vector.memory_store import InMemoryVectorStore


def _chunks(texts: list[str], doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc_id}:{i}", doc_id=doc_id, index=i, text=text)
        for i, text in enumerate(texts)
    ]


def test_run_job_merges_duplicate_entities_the_extraction_produced() -> None:
    """Two spellings of the same client - "Ryan Group" and "Ryan Group Inc" -
    extracted from two different sentences must come out of the job as one
    entity, not two. `dedup_key` strips the "inc" suffix (Layer 1/2); the two
    names stay distinct `entity_id`s at extraction time (`normalize_name`
    does not strip suffixes), so this only merges if the pipeline actually
    calls `resolve_duplicates`."""
    chunks = _chunks(
        [
            "Agency09 delivered the UI/UX service to Ryan Group.",
            "Ryan Group Inc. also received the Website Design service.",
        ]
    )
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(name="Agency09", type="ORG", source_chunk_index=0),
            ExtractedEntity(name="UI/UX", type="SERVICE", source_chunk_index=0),
            ExtractedEntity(name="Ryan Group", type="ORG", source_chunk_index=0),
            ExtractedEntity(name="Website Design", type="SERVICE", source_chunk_index=1),
            ExtractedEntity(name="Ryan Group Inc", type="ORG", source_chunk_index=1),
        ],
        relations=[
            ExtractedRelation(
                source_name="Agency09", source_type="ORG",
                target_name="Ryan Group", target_type="ORG",
                type="delivered_service_to", source_chunk_index=0,
            ),
            ExtractedRelation(
                source_name="Ryan Group Inc", source_type="ORG",
                target_name="Website Design", target_type="SERVICE",
                type="received_service", source_chunk_index=1,
            ),
        ],
    )

    graph_store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    embedder = FakeEmbedder()
    registry = JobRegistry()
    job = registry.create(job_id="job1", doc_id="doc1", filename="agency.txt", chunks_total=2)

    run_job(
        job.job_id,
        chunks,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
        provider=FakeLLM(responses=[extraction]),
        graph_store=graph_store,
        extraction_cache=InMemoryExtractionCache(),
    )

    assert registry.get(job.job_id).state == JOB_STATE_SUCCEEDED

    org_entities = [e for e in graph_store.all_entities() if e.type == "ORG" and e.name.startswith("Ryan")]
    assert len(org_entities) == 1, f"expected one merged Ryan Group entity, got {org_entities!r}"

    merged = org_entities[0]
    assert "Ryan Group Inc" in merged.aliases or merged.name in ("Ryan Group", "Ryan Group Inc")
    assert merged.merged_from, "the surviving entity should record what was merged into it"

    # Both relations still point at the surviving entity - nothing orphaned.
    relations = merged.mention_chunk_ids
    assert "doc1:0" in relations and "doc1:1" in relations


def test_run_job_does_not_merge_genuinely_different_entities() -> None:
    """Two distinct clients must stay distinct after dedup runs."""
    chunks = _chunks(["Agency09 delivered UI/UX to Ryan Group and to Tata Motors."])
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(name="Agency09", type="ORG", source_chunk_index=0),
            ExtractedEntity(name="Ryan Group", type="ORG", source_chunk_index=0),
            ExtractedEntity(name="Tata Motors", type="ORG", source_chunk_index=0),
        ],
        relations=[
            ExtractedRelation(
                source_name="Agency09", source_type="ORG",
                target_name="Ryan Group", target_type="ORG",
                type="delivered_service_to", source_chunk_index=0,
            ),
            ExtractedRelation(
                source_name="Agency09", source_type="ORG",
                target_name="Tata Motors", target_type="ORG",
                type="delivered_service_to", source_chunk_index=0,
            ),
        ],
    )

    graph_store = InMemoryGraphStore()
    registry = JobRegistry()
    job = registry.create(job_id="job2", doc_id="doc1", filename="agency.txt", chunks_total=1)

    run_job(
        job.job_id,
        chunks,
        registry=registry,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
        provider=FakeLLM(responses=[extraction]),
        graph_store=graph_store,
        extraction_cache=InMemoryExtractionCache(),
    )

    org_names = {e.name for e in graph_store.all_entities() if e.type == "ORG"}
    assert org_names == {"Agency09", "Ryan Group", "Tata Motors"}
