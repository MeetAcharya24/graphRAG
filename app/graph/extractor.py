"""Read entities and relationships out of chunks (Phase 4).

Several chunks share one prompt to stay inside the free-tier quota (D14): at
~5 per call, a 200-chunk document costs ~40 calls instead of 200.

**The hazard batching introduces.** With five chunks in one prompt the model
must say which chunk each triple came from. If that is wrong, `source_chunk_id`
is a guess, and R6's promise - that every answer shows text supporting it -
quietly breaks: a citation would point at a chunk that does not contain the
fact. So chunks are labelled `[0] [1] ...`, every extracted item carries a
`source_chunk_index`, and `resolve_extraction` maps it back to the real
`chunk_id`, **dropping anything out of range**. A missing relationship is a
smaller loss than a misattributed one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from app.config import EXTRACTION_BATCH_SIZE, EXTRACTION_MAX_TOKENS, get_settings
from app.errors import AppError, LLMQuotaExceeded
from app.graph.base import make_entity_id
from app.logging_conf import get_logger
from app.schemas import Chunk, Entity, Relation

log = get_logger(__name__)

# Closed vocabularies. An open one makes the graph unqueryable: `works_at`,
# `employed_by` and `is_employed_at` would be three different edges.
#
# SERVICE and TECHNOLOGY exist so agency/client work has somewhere to land:
# without them, "Agency09 delivered Mobile App Development to BITS Pilani"
# has no entity for the service itself, and the model is left trying to wedge
# that fact into a direct Agency-to-client edge - observed live as the
# nonsensical "Agency09 produces BITS Pilani" (a client is not a product).
# With SERVICE/TECHNOLOGY entities, the same fact becomes two edges
# (Agency09 --built--> Mobile App Development, Mobile App Development
# --delivered_service_to--> BITS Pilani), and two clients sharing one service
# are now a real two-hop link through that service - not through Agency09.
ENTITY_TYPES = (
    "PERSON",
    "ORG",
    "PRODUCT",
    "SERVICE",
    "TECHNOLOGY",
    "LOCATION",
    "EVENT",
    "CONCEPT",
)

RELATION_TYPES = (
    "works_at",
    "founded",
    "acquired_by",
    "acquired",
    "produces",
    "part_of",
    "located_in",
    "reports_to",
    "collaborates_with",
    "competes_with",
    "mentions",
    # Agency/client/vendor work: see the ENTITY_TYPES note above. Kept
    # distinct from produces/collaborates_with, which are for an
    # organization's own output or a peer-to-peer partnership, not a
    # service rendered to a client.
    "built",
    "delivered_service_to",
    "used_technology",
    # The escape hatch: a real relationship that fits nothing above is kept
    # rather than discarded, with its own wording in `label`.
    "OTHER",
)

EntityType = Literal[
    "PERSON", "ORG", "PRODUCT", "SERVICE", "TECHNOLOGY", "LOCATION", "EVENT", "CONCEPT"
]


class ExtractedEntity(BaseModel):
    """One entity as the model reported it, before resolution."""

    name: str = Field(description="The entity's name exactly as written.")
    type: EntityType
    source_chunk_index: int = Field(
        description="Which numbered chunk this came from, as labelled in the prompt."
    )
    description: str = Field(default="", description="A short factual description.")


class ExtractedRelation(BaseModel):
    """One relationship as the model reported it, before resolution."""

    source_name: str
    source_type: EntityType
    target_name: str
    target_type: EntityType
    type: str = Field(description="One of the allowed relation types.")
    source_chunk_index: int = Field(
        description="Which numbered chunk states this relationship."
    )
    label: str = Field(default="", description="Free wording, used when type is OTHER.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """What one extraction call returns."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


@dataclass
class ExtractionSummary:
    """What one `extract_from_chunks` run did."""

    chunks_extracted: int = 0
    chunks_skipped: int = 0
    batches_done: int = 0
    batches_failed: int = 0
    entities_written: int = 0
    relations_written: int = 0
    failures: list[str] = field(default_factory=list)


class ExtractionCache(Protocol):
    """Which chunks have already been extracted.

    Backed by a property on the Neo4j :Chunk node in production (D10 moved it
    off disk, so it survives a restart), and by a set in tests.
    """

    def extracted_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        ...

    def mark_extracted(self, chunk_ids: Sequence[str]) -> None:
        ...


class InMemoryExtractionCache:
    """Implements `ExtractionCache` with a set."""

    def __init__(self) -> None:
        self._done: set[str] = set()

    def extracted_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        return {cid for cid in chunk_ids if cid in self._done}

    def mark_extracted(self, chunk_ids: Sequence[str]) -> None:
        self._done.update(chunk_ids)


class Neo4jExtractionCache:
    """Implements `ExtractionCache` using `:Chunk.extracted_at`.

    On the node rather than on disk (D10), so a restart - which the free host
    does every ~15 minutes idle - does not lose the protection and make a
    re-upload spend the quota again.
    """

    def __init__(self, driver, database: str) -> None:
        self._driver = driver
        self._database = database

    def _run(self, query: str, *, write: bool, **params):
        from app.db import transact

        return transact.run(
            self._driver,
            self._database,
            query,
            write=write,
            query_kind="extraction-cache",
            **params,
        )

    def extracted_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        from app.graph import cypher

        ids = list(chunk_ids)
        if not ids:
            return set()
        records = self._run(cypher.EXTRACTED_CHUNK_IDS, write=False, chunk_ids=ids)
        return set(records[0]["extracted"] or ()) if records else set()

    def mark_extracted(self, chunk_ids: Sequence[str]) -> None:
        from app.graph import cypher

        ids = list(chunk_ids)
        if not ids:
            return
        self._run(cypher.MARK_CHUNKS_EXTRACTED, write=True, chunk_ids=ids)


SYSTEM_PROMPT = f"""You read text and list the entities and relationships in it.

Rules:
- Use ONLY these entity types: {", ".join(ENTITY_TYPES)}
- Use ONLY these relationship types: {", ".join(RELATION_TYPES)}
  If a real relationship fits none of them, use OTHER and put your own wording
  in `label`. Never invent a new type.
- Agency/client/vendor work: a named offering (an implementation, a platform,
  an app, a campaign) is its own SERVICE entity, and a named tool, platform or
  language it was built with is its own TECHNOLOGY entity - never fold either
  into the organization that made or used it.
  - ORG `built` SERVICE - the organization that created or delivered it.
  - SERVICE `delivered_service_to` ORG - the client it was delivered to.
  - ORG `used_technology` TECHNOLOGY - a technology an org built with or adopted.
  A client receiving work is never `produces`, and a vendor relationship is
  never `collaborates_with` - those are for an organization's own output and
  for a peer-to-peer partnership between equals, not a service rendered to a
  client.
- Every entity and every relationship MUST set source_chunk_index to the number
  of the chunk it came from, exactly as labelled in the prompt. This is how the
  fact is traced back to its text, so a wrong number is worse than omitting the
  item entirely.
- Extract only what the text states. Do not infer, and do not add world
  knowledge.
- Use the name as written in the text, not an expanded or corrected form."""


def build_extraction_prompt(chunks: Sequence[Chunk]) -> str:
    """Number the chunks so each extracted item can point back at one."""
    blocks = [f"[{index}] {chunk.text}" for index, chunk in enumerate(chunks)]
    return (
        "Extract the entities and relationships from these numbered chunks.\n\n"
        + "\n\n".join(blocks)
    )


def _relation_id(source_id: str, target_id: str, rel_type: str, chunk_id: str) -> str:
    """A stable id, so re-extracting the same text updates rather than duplicates."""
    digest = hashlib.sha256(
        f"{source_id}|{target_id}|{rel_type}|{chunk_id}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _chunk_id_for(index: int, chunks: Sequence[Chunk]) -> str | None:
    """Map a model-supplied index to a real chunk id, or None if it is invalid."""
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    if not 0 <= index < len(chunks):
        return None
    return chunks[index].chunk_id


def resolve_extraction(
    result: ExtractionResult, chunks: Sequence[Chunk]
) -> tuple[list[Entity], list[Relation]]:
    """Turn a model response into graph objects, dropping anything unusable.

    Relationship endpoints become entities even when the model did not list
    them separately, because the store refuses a dangling edge and losing the
    whole relationship would be the worse outcome.
    """
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []

    def remember(name: str, entity_type: str, chunk_id: str, description: str = "") -> str | None:
        cleaned = name.strip()
        if not cleaned:
            return None

        entity_id = make_entity_id(entity_type, cleaned)
        existing = entities.get(entity_id)
        if existing is None:
            entities[entity_id] = Entity(
                entity_id=entity_id,
                name=cleaned,
                type=entity_type.upper(),
                description=description,
                mention_chunk_ids=(chunk_id,),
            )
            return entity_id

        mentions = list(existing.mention_chunk_ids)
        if chunk_id not in mentions:
            mentions.append(chunk_id)
        entities[entity_id] = existing.model_copy(
            update={
                "mention_chunk_ids": tuple(mentions),
                "description": existing.description or description,
            }
        )
        return entity_id

    for extracted in result.entities:
        chunk_id = _chunk_id_for(extracted.source_chunk_index, chunks)
        if chunk_id is None:
            log.warning(
                "Dropped an entity with an invalid chunk index: %r",
                extracted.source_chunk_index,
            )
            continue
        remember(extracted.name, extracted.type, chunk_id, extracted.description)

    for extracted in result.relations:
        chunk_id = _chunk_id_for(extracted.source_chunk_index, chunks)
        if chunk_id is None:
            log.warning(
                "Dropped a relation with an invalid chunk index: %r",
                extracted.source_chunk_index,
            )
            continue

        source_id = remember(extracted.source_name, extracted.source_type, chunk_id)
        target_id = remember(extracted.target_name, extracted.target_type, chunk_id)
        if source_id is None or target_id is None:
            continue
        if source_id == target_id:
            # A self-loop carries no retrievable information and would pad
            # every traversal.
            continue

        rel_type = extracted.type if extracted.type in RELATION_TYPES else "OTHER"

        relations.append(
            Relation(
                rel_id=_relation_id(source_id, target_id, rel_type, chunk_id),
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                source_chunk_id=chunk_id,
                confidence=extracted.confidence,
            )
        )

    return list(entities.values()), relations


def extract_from_chunks(
    chunks: Sequence[Chunk],
    *,
    provider,
    store,
    cache: ExtractionCache | None = None,
) -> ExtractionSummary:
    """Extract from `chunks` in batches and write the results to the graph.

    A batch that fails is logged and skipped - one bad batch must not lose the
    whole document. A daily quota error is different: retrying cannot help, so
    it stops the run and propagates, leaving the remaining chunks unmarked so a
    later run picks them up.
    """
    summary = ExtractionSummary()
    if not chunks:
        return summary

    pending = list(chunks)
    if cache is not None:
        already = cache.extracted_ids([chunk.chunk_id for chunk in pending])
        if already:
            summary.chunks_skipped = len(already)
            log.info("Skipping %d chunk(s) already extracted", len(already))
        pending = [chunk for chunk in pending if chunk.chunk_id not in already]

    if not pending:
        return summary

    model = get_settings().extraction_model

    for start in range(0, len(pending), EXTRACTION_BATCH_SIZE):
        batch = pending[start : start + EXTRACTION_BATCH_SIZE]

        try:
            result = provider.generate_structured(
                model=model,
                prompt=build_extraction_prompt(batch),
                schema=ExtractionResult,
                system=SYSTEM_PROMPT,
                max_tokens=EXTRACTION_MAX_TOKENS,
                minimal_reasoning=True,
            )
        except LLMQuotaExceeded:
            log.error(
                "Daily quota reached after %d batch(es); stopping. The remaining "
                "chunks stay unextracted and will be picked up next run.",
                summary.batches_done,
            )
            raise
        except (AppError, Exception) as exc:  # noqa: BLE001 - logged and skipped
            summary.batches_failed += 1
            message = getattr(exc, "message", None) or str(exc)
            summary.failures.append(message)
            log.warning("Extraction batch failed, skipping it: %s", message)
            continue

        entities, relations = resolve_extraction(result, batch)
        written_entities, written_relations = _write(store, entities, relations)

        summary.batches_done += 1
        summary.chunks_extracted += len(batch)
        summary.entities_written += written_entities
        summary.relations_written += written_relations

        if cache is not None:
            # Only after a successful write, so a transient failure does not
            # permanently lose these chunks.
            cache.mark_extracted([chunk.chunk_id for chunk in batch])

    log.info(
        "Extraction done: %d chunk(s), %d batch(es), %d failed, "
        "%d entities, %d relations",
        summary.chunks_extracted,
        summary.batches_done,
        summary.batches_failed,
        summary.entities_written,
        summary.relations_written,
    )
    return summary


def _write(store, entities: Sequence[Entity], relations: Sequence[Relation]) -> tuple[int, int]:
    """Upsert entities first, then relationships.

    Order matters: the store refuses a relationship whose endpoints do not
    exist yet. A single failing relationship is skipped rather than abandoning
    the rest of the batch's results.
    """
    for entity in entities:
        store.upsert_entity(entity)

    written = 0
    for relation in relations:
        try:
            store.upsert_relation(relation)
            written += 1
        except AppError as exc:
            log.warning("Skipped a relationship: %s", exc.message)
    return len(entities), written
