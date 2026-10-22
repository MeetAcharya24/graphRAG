"""Every Cypher statement in the project, in one file.

Kept together so the database contract is reviewable as a unit rather than
scattered across call sites.

Two things to know when reading these:

* **Relationships all share the type `RELATES`**, with the semantic label in a
  `type` property. Variable-length traversal (`-[:RELATES*1..3]-`) has to match
  a single relationship type, so a dynamic type per relation would make
  multi-hop queries impossible to write.
* **Hop limits are interpolated, not parameterised.** Cypher does not accept a
  parameter inside `*1..n`, so `max_hops` is formatted into the string. Every
  such function validates it as a bounded integer first, because formatting
  user input into a query is otherwise how injection happens.
"""

from __future__ import annotations

from app.config import MAX_HOPS

CHUNK_VECTOR_INDEX = "chunk_embeddings"

# --- schema ---------------------------------------------------------------

CONSTRAINTS = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
)

# Uniqueness on rel_id cannot be a constraint on a relationship in Community
# edition, so MERGE on rel_id does the deduplication instead.
INDEXES = (
    "CREATE INDEX relation_id IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.rel_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name_normalized)",
    "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
)


def vector_index(dimensions: int) -> str:
    """The chunk-embedding index. Cosine, to match the in-memory store."""
    return (
        f"CREATE VECTOR INDEX {CHUNK_VECTOR_INDEX} IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {int(dimensions)}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


# --- entities -------------------------------------------------------------

UPSERT_ENTITY = """
MERGE (e:Entity {entity_id: $entity_id})
ON CREATE SET
    e.name = $name,
    e.name_normalized = $name_normalized,
    e.type = $type,
    e.aliases = $aliases,
    e.aliases_normalized = $aliases_normalized,
    e.description = $description,
    e.mention_chunk_ids = $mention_chunk_ids
ON MATCH SET
    e.aliases = e.aliases + [a IN $aliases WHERE NOT a IN e.aliases],
    e.aliases_normalized =
        e.aliases_normalized + [a IN $aliases_normalized
                                WHERE NOT a IN e.aliases_normalized],
    e.mention_chunk_ids =
        e.mention_chunk_ids + [m IN $mention_chunk_ids
                               WHERE NOT m IN e.mention_chunk_ids],
    e.description = CASE
        WHEN e.description IS NULL OR e.description = ''
        THEN $description ELSE e.description END
RETURN e
"""

GET_ENTITY = "MATCH (e:Entity {entity_id: $entity_id}) RETURN e"

ALL_ENTITIES = "MATCH (e:Entity) RETURN e"

FIND_ENTITIES_BY_NAME = """
MATCH (e:Entity)
WHERE e.name_normalized = $name OR $name IN e.aliases_normalized
RETURN e
"""

DELETE_ORPHANED_ENTITIES = """
MATCH (e:Entity)
WHERE e.mention_chunk_ids IS NULL OR size(e.mention_chunk_ids) = 0
DETACH DELETE e
RETURN count(e) AS removed
"""

# --- dedup support (Phase 5) -----------------------------------------------

RELATIONS_OF_ENTITY = """
MATCH (e:Entity {entity_id: $entity_id})-[r:RELATES]-(:Entity)
RETURN DISTINCT r.rel_id AS rel_id, r.type AS type,
       r.source_chunk_id AS source_chunk_id, r.confidence AS confidence,
       startNode(r).entity_id AS source_id, endNode(r).entity_id AS target_id
"""

DELETE_ENTITY = """
MATCH (e:Entity {entity_id: $entity_id})
DETACH DELETE e
RETURN count(e) AS removed
"""

REMOVE_MENTIONS = """
MATCH (e:Entity)
WHERE any(m IN e.mention_chunk_ids WHERE m IN $chunk_ids)
SET e.mention_chunk_ids = [m IN e.mention_chunk_ids WHERE NOT m IN $chunk_ids]
RETURN count(e) AS touched
"""

# --- relationships --------------------------------------------------------

UPSERT_RELATION = """
MATCH (source:Entity {entity_id: $source_id})
MATCH (target:Entity {entity_id: $target_id})
MERGE (source)-[r:RELATES {rel_id: $rel_id}]->(target)
ON CREATE SET
    r.type = $type,
    r.source_chunk_id = $source_chunk_id,
    r.confidence = $confidence
ON MATCH SET
    r.type = $type,
    r.source_chunk_id = $source_chunk_id,
    r.confidence = $confidence
RETURN r
"""

GET_RELATION = """
MATCH (source:Entity)-[r:RELATES {rel_id: $rel_id}]->(target:Entity)
RETURN r, source.entity_id AS source_id, target.entity_id AS target_id
"""

ENDPOINTS_EXIST = """
MATCH (e:Entity) WHERE e.entity_id IN $entity_ids
RETURN collect(e.entity_id) AS found
"""

# Two plain queries rather than one with subqueries: `CALL () { ... }` is
# Neo4j 5.23+, and schema.py advertises 5.13+ as the requirement.
COUNT_ENTITIES = "MATCH (e:Entity) RETURN count(e) AS entity_count"
COUNT_RELATIONS = "MATCH ()-[r:RELATES]->() RETURN count(r) AS relation_count"

SHOW_VECTOR_INDEX = f"""
SHOW VECTOR INDEXES YIELD name, options
WHERE name = '{CHUNK_VECTOR_INDEX}'
RETURN options
"""

# --- traversal ------------------------------------------------------------


def _checked_hops(max_hops: int) -> int:
    """Validate a hop count before it is formatted into a query string."""
    hops = int(max_hops)
    if not 1 <= hops <= MAX_HOPS:
        raise ValueError(f"max_hops must be between 1 and {MAX_HOPS}, got {max_hops}")
    return hops


def neighbors(max_hops: int) -> str:
    """Entities within n hops of any seed, seeds excluded, direction ignored."""
    hops = _checked_hops(max_hops)
    return f"""
MATCH (seed:Entity) WHERE seed.entity_id IN $entity_ids
MATCH (seed)-[:RELATES*1..{hops}]-(other:Entity)
WHERE NOT other.entity_id IN $entity_ids
RETURN DISTINCT other
"""


def shortest_paths(max_hops: int) -> str:
    """All shortest walks between two entities, within the hop limit."""
    hops = _checked_hops(max_hops)
    return f"""
MATCH (source:Entity {{entity_id: $source_id}})
MATCH (target:Entity {{entity_id: $target_id}})
MATCH path = allShortestPaths((source)-[:RELATES*1..{hops}]-(target))
RETURN [n IN nodes(path) | n] AS path_nodes,
       [r IN relationships(path) | {{
           rel_id: r.rel_id,
           type: r.type,
           source_chunk_id: r.source_chunk_id,
           confidence: r.confidence,
           source_id: startNode(r).entity_id,
           target_id: endNode(r).entity_id
       }}] AS path_edges
"""


# --- chunks and vectors ---------------------------------------------------

UPSERT_CHUNKS = """
UNWIND $rows AS row
MERGE (d:Document {doc_id: row.doc_id})
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.doc_id = row.doc_id,
    c.index = row.index,
    c.text = row.text,
    c.embedding = row.embedding
MERGE (c)-[:PART_OF]->(d)
"""

QUERY_VECTORS = f"""
CALL db.index.vector.queryNodes('{CHUNK_VECTOR_INDEX}', $k, $vector)
YIELD node, score
RETURN node.chunk_id AS chunk_id,
       node.doc_id AS doc_id,
       node.index AS index,
       node.text AS text,
       score
"""

GET_CHUNKS_BY_IDS = """
MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids
RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, c.index AS index, c.text AS text
"""

DELETE_DOCUMENT = """
MATCH (c:Chunk {doc_id: $doc_id})
WITH collect(c) AS chunks, count(c) AS removed
FOREACH (c IN chunks | DETACH DELETE c)
WITH removed
OPTIONAL MATCH (d:Document {doc_id: $doc_id})
DETACH DELETE d
RETURN removed
"""

COUNT_CHUNKS = "MATCH (c:Chunk) RETURN count(c) AS total"

DOCUMENT_IDS = "MATCH (c:Chunk) RETURN collect(DISTINCT c.doc_id) AS doc_ids"

# The extraction cache lives on the chunk node, so it survives a restart -
# which the disk cache it replaced would not have (D10).
EXTRACTED_CHUNK_IDS = """
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids AND c.extracted_at IS NOT NULL
RETURN collect(c.chunk_id) AS extracted
"""

MARK_CHUNKS_EXTRACTED = """
MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids
SET c.extracted_at = datetime()
RETURN count(c) AS marked
"""

CHUNK_DIMENSION = """
MATCH (c:Chunk) WHERE c.embedding IS NOT NULL
RETURN size(c.embedding) AS dimension LIMIT 1
"""
