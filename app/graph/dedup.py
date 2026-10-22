"""Entity deduplication, layers 1-5 (Phase 5, D2).

*In simple words: realise "Sarah Chen", "S. Chen" and "Dr. Sarah Chen" are one
person - without merging two different people who share a name.*

Five layers, cheapest first, same type required before any of them run:

1. `dedup_key` - a stricter normal form than `app.graph.base.normalize_name`
   (which is deliberately shallow): strips personal titles and corporate
   suffixes, collapses punctuation.
2. Exact match on that key, across both entities' names *and* aliases.
3. Alias rules: initial expansion ("S Chen" <-> "Sarah Chen"), acronym <->
   expansion ("IBM" <-> "International Business Machines"), and a last-name
   match that only fires when the surname is unambiguous in the group.
4. Fuzzy: `rapidfuzz.token_set_ratio >= 92`.
5. Embedding cosine on name + description `>= 0.92`, only run when an
   embedder is supplied - it costs a Gemini call per entity.

Layer 6 (an LLM tiebreak for grey-band pairs) is deferred; `resolve_duplicates`
keeps a `tiebreaker` hook for it, and grey-band pairs are logged either way.

**One deliberate exception to "strip suffixes":** generational suffixes (Jr,
Sr, II, III, IV) are never stripped, and a mismatch on one is a hard
disqualifier even when every other layer would merge the pair - "John Smith"
and "John Smith Jr." are father and son, not a name variant, and
`token_set_ratio` scores that pair 100 (pure token subset) if left unguarded.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Sequence

from rapidfuzz import fuzz

from app.errors import EntityNotFound
from app.graph.base import GraphStore
from app.logging_conf import get_logger
from app.schemas import Entity
from app.vector.base import Embedder, Vector
from app.vector.memory_store import cosine_similarity

log = get_logger(__name__)

FUZZY_MERGE_THRESHOLD = 92.0
FUZZY_GREY_FLOOR = 80.0
EMBEDDING_MERGE_THRESHOLD = 0.92
EMBEDDING_GREY_FLOOR = 0.85

_PERSONAL_TITLES = {
    "dr", "mr", "mrs", "ms", "miss", "prof", "sir", "dame", "lord", "lady", "rev",
}
_ORG_SUFFIXES = {
    "inc", "llc", "ltd", "corp", "corporation", "co", "company", "gmbh", "plc",
}
# Deliberately NOT stripped - see the module docstring.
_GENERATIONAL_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

_PUNCTUATION_RE = re.compile(r"[.,]")


def dedup_key(name: str) -> str:
    """A stricter normal form than `normalize_name`, for grouping name variants.

    Free to compute (Layer 1): NFKC, casefold, drop periods/commas, strip one
    leading personal title and one trailing corporate suffix.
    """
    folded = unicodedata.normalize("NFKC", name).casefold().strip()
    folded = _PUNCTUATION_RE.sub("", folded)
    tokens = folded.split()

    if tokens and tokens[0] in _PERSONAL_TITLES:
        tokens = tokens[1:]
    if tokens and tokens[-1] in _ORG_SUFFIXES:
        tokens = tokens[:-1]

    return " ".join(tokens)


def _tokens(name: str) -> list[str]:
    return dedup_key(name).split()


def _generational_suffix(tokens: Sequence[str]) -> str | None:
    return tokens[-1] if tokens and tokens[-1] in _GENERATIONAL_SUFFIXES else None


def _initial_expansion_match(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> bool:
    """"S Chen" <-> "Sarah Chen": one differing token is an initial, rest exact."""
    if len(a_tokens) != len(b_tokens) or list(a_tokens) == list(b_tokens):
        return False

    for ta, tb in zip(a_tokens, b_tokens):
        if ta == tb:
            continue
        if len(ta) == 1:
            initial, full = ta, tb
        elif len(tb) == 1:
            initial, full = tb, ta
        else:
            return False
        if not full.startswith(initial):
            return False
    return True


def _acronym_match(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> bool:
    """"IBM" <-> "International Business Machines"."""

    def _try(acronym_tokens: Sequence[str], expansion_tokens: Sequence[str]) -> bool:
        if len(acronym_tokens) != 1 or len(expansion_tokens) < 2:
            return False
        acronym = acronym_tokens[0]
        if len(acronym) < 2 or not acronym.isalpha():
            return False
        initials = "".join(token[0] for token in expansion_tokens if token)
        return initials == acronym

    return _try(a_tokens, b_tokens) or _try(b_tokens, a_tokens)


def _unique_last_name_match(a: Entity, b: Entity, group: Sequence[Entity]) -> bool:
    """A bare surname matches a full name only if no other candidate shares it."""

    def _surname_shared_with(bare: Entity, full: Entity) -> str | None:
        bare_tokens, full_tokens = _tokens(bare.name), _tokens(full.name)
        if len(bare_tokens) == 1 and len(full_tokens) > 1 and bare_tokens[0] == full_tokens[-1]:
            return full_tokens[-1]
        return None

    surname = _surname_shared_with(a, b) or _surname_shared_with(b, a)
    if surname is None:
        return False

    others = (e for e in group if e.entity_id not in (a.entity_id, b.entity_id))
    return not any(_tokens(other.name)[-1:] == [surname] for other in others)


def _alias_rule_match(a: Entity, b: Entity, group: Sequence[Entity]) -> bool:
    a_tokens, b_tokens = _tokens(a.name), _tokens(b.name)
    return (
        _initial_expansion_match(a_tokens, b_tokens)
        or _acronym_match(a_tokens, b_tokens)
        or _unique_last_name_match(a, b, group)
    )


@dataclass(frozen=True)
class LayerMatch:
    layer: str
    score: float


@dataclass(frozen=True)
class PairResult:
    match: LayerMatch | None = None
    grey: LayerMatch | None = None


def _known_names(entity: Entity) -> set[str]:
    return {dedup_key(entity.name)} | {dedup_key(alias) for alias in entity.aliases}


def match_pair(a: Entity, b: Entity, group: Sequence[Entity]) -> PairResult:
    """Layers 1-4. Layer 5 (embeddings) is the resolver's job - it needs vectors."""
    if a.type != b.type:
        return PairResult()

    a_tokens, b_tokens = _tokens(a.name), _tokens(b.name)
    if _generational_suffix(a_tokens) != _generational_suffix(b_tokens):
        return PairResult()

    if _known_names(a) & _known_names(b):
        return PairResult(match=LayerMatch("exact", 1.0))

    if _alias_rule_match(a, b, group):
        return PairResult(match=LayerMatch("alias", 1.0))

    if min(len(a_tokens), len(b_tokens)) <= 1 and len(a_tokens) != len(b_tokens):
        # A bare one-word name (a surname on its own, say) is a trivial
        # token-subset of almost any longer name, so `token_set_ratio` would
        # score it ~100 regardless of whether they are really the same
        # entity. That judgement belongs to the unique-last-name rule above,
        # which checks the whole group for ambiguity; fuzzy matching here
        # would bypass that check.
        return PairResult()

    ratio = fuzz.token_set_ratio(dedup_key(a.name), dedup_key(b.name))
    if ratio >= FUZZY_MERGE_THRESHOLD:
        return PairResult(match=LayerMatch("fuzzy", ratio / 100))
    if ratio >= FUZZY_GREY_FLOOR:
        return PairResult(grey=LayerMatch("fuzzy", ratio / 100))
    return PairResult()


# --- merging -----------------------------------------------------------------


def _rewired_rel_id(source_id: str, target_id: str, rel_type: str, chunk_id: str) -> str:
    """The id a fresh extraction of this (source, target, type, chunk) would get.

    Mirrors `app.graph.extractor._relation_id`'s formula deliberately. A
    relation whose endpoint changed because of a merge must not keep its old
    id: both stores key a relationship's identity on `rel_id` alone when
    deciding whether a re-upsert updates it in place or creates a new edge,
    so reusing the old id while changing an endpoint leaves a stale,
    differently-addressed copy of the edge behind (and on the in-memory
    store, `delete_entity` would then delete the *rewired* relation when it
    cleans up the duplicate's own stale adjacency entry for that same id).
    """
    digest = hashlib.sha256(
        f"{source_id}|{target_id}|{rel_type}|{chunk_id}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


def merge_entity_into(store: GraphStore, canonical_id: str, duplicate_id: str) -> Entity:
    """Fold `duplicate_id` into `canonical_id` and delete the duplicate.

    Built entirely out of `GraphStore` primitives (`upsert_entity`,
    `relations_of`, `upsert_relation`, `delete_entity`), so it works the same
    way against the in-memory store and Neo4j without any merge-specific
    Cypher.
    """
    if canonical_id == duplicate_id:
        raise ValueError("an entity cannot be merged into itself")

    canonical = store.get_entity(canonical_id)
    duplicate = store.get_entity(duplicate_id)
    if canonical is None or duplicate is None:
        missing = canonical_id if canonical is None else duplicate_id
        raise EntityNotFound(f"Cannot merge: entity {missing!r} not found.")

    new_aliases = list(canonical.aliases)
    for candidate in (*duplicate.aliases, duplicate.name):
        if candidate != canonical.name and candidate not in new_aliases:
            new_aliases.append(candidate)

    new_mentions = list(canonical.mention_chunk_ids)
    new_mentions += [m for m in duplicate.mention_chunk_ids if m not in new_mentions]

    new_merged_from = list(canonical.merged_from)
    for candidate in (*duplicate.merged_from, duplicate.entity_id):
        if candidate not in new_merged_from:
            new_merged_from.append(candidate)

    merged = canonical.model_copy(
        update={
            "aliases": tuple(new_aliases),
            "mention_chunk_ids": tuple(new_mentions),
            "description": canonical.description or duplicate.description,
            "merged_from": tuple(new_merged_from),
        }
    )
    store.upsert_entity(merged)

    for relation in store.relations_of(duplicate_id):
        new_source = canonical_id if relation.source_id == duplicate_id else relation.source_id
        new_target = canonical_id if relation.target_id == duplicate_id else relation.target_id
        if new_source == new_target:
            # Would become a self-loop - the duplicate's own deletion below
            # removes it; recreating it on the canonical would be nonsense.
            continue
        store.upsert_relation(
            relation.model_copy(
                update={
                    "rel_id": _rewired_rel_id(
                        new_source, new_target, relation.type, relation.source_chunk_id
                    ),
                    "source_id": new_source,
                    "target_id": new_target,
                }
            )
        )

    store.delete_entity(duplicate_id)
    return store.get_entity(canonical_id)


# --- resolving the whole graph ------------------------------------------------


@dataclass(frozen=True)
class MergeDecision:
    canonical_id: str
    duplicate_id: str
    layer: str
    score: float


@dataclass(frozen=True)
class GreyBandPair:
    entity_a: str
    entity_b: str
    layer: str
    score: float


@dataclass(frozen=True)
class DedupReport:
    merges: tuple[MergeDecision, ...] = ()
    grey_band: tuple[GreyBandPair, ...] = ()


Tiebreaker = Callable[[Entity, Entity, str, float], "str | None"]


def _pick_canonical(members: Sequence[str], by_id: dict[str, Entity]) -> str:
    """Most mentions wins; ties break toward the lexicographically smallest id."""
    return min(members, key=lambda eid: (-len(by_id[eid].mention_chunk_ids), eid))


def resolve_duplicates(
    store: GraphStore,
    *,
    embedder: Embedder | None = None,
    tiebreaker: Tiebreaker | None = None,
) -> DedupReport:
    """Find and merge duplicate entities across the whole graph.

    Candidate pairs are processed in sorted order within each same-type group,
    so results are deterministic regardless of insertion order (D2's
    "merging is order-independent" requirement).
    """
    entities = sorted(store.all_entities(), key=lambda e: e.entity_id)
    by_id = {e.entity_id: e for e in entities}

    by_type: dict[str, list[Entity]] = {}
    for entity in entities:
        by_type.setdefault(entity.type, []).append(entity)

    parent: dict[str, str] = {e.entity_id: e.entity_id for e in entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    decisions: dict[frozenset[str], tuple[str, float]] = {}
    grey_band: list[GreyBandPair] = []

    for group in by_type.values():
        vectors: dict[str, Vector] = {}
        if embedder is not None and len(group) > 1:
            texts = [f"{e.name}: {e.description}" for e in group]
            computed = embedder.embed_documents(texts)
            vectors = dict(zip((e.entity_id for e in group), computed))

        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                result = match_pair(a, b, group)

                if (
                    result.match is None
                    and vectors.get(a.entity_id) is not None
                    and vectors.get(b.entity_id) is not None
                ):
                    score = cosine_similarity(vectors[a.entity_id], vectors[b.entity_id])
                    if score >= EMBEDDING_MERGE_THRESHOLD:
                        result = PairResult(match=LayerMatch("embedding", score))
                    elif score >= EMBEDDING_GREY_FLOOR:
                        result = PairResult(grey=LayerMatch("embedding", score))

                if result.grey is not None:
                    grey_band.append(
                        GreyBandPair(a.entity_id, b.entity_id, result.grey.layer, result.grey.score)
                    )

                if result.match is not None:
                    union(a.entity_id, b.entity_id)
                    decisions[frozenset((a.entity_id, b.entity_id))] = (
                        result.match.layer,
                        result.match.score,
                    )
                elif result.grey is not None and tiebreaker is not None:
                    resolved = tiebreaker(a, b, result.grey.layer, result.grey.score)
                    if resolved is not None:
                        union(a.entity_id, b.entity_id)
                        decisions[frozenset((a.entity_id, b.entity_id))] = (
                            "tiebreak",
                            result.grey.score,
                        )

    components: dict[str, list[str]] = {}
    for entity_id in parent:
        components.setdefault(find(entity_id), []).append(entity_id)

    merges: list[MergeDecision] = []
    for members in components.values():
        if len(members) < 2:
            continue
        canonical_id = _pick_canonical(members, by_id)
        for duplicate_id in sorted(m for m in members if m != canonical_id):
            layer, score = decisions.get(
                frozenset((canonical_id, duplicate_id)), ("transitive", 1.0)
            )
            merge_entity_into(store, canonical_id, duplicate_id)
            merges.append(MergeDecision(canonical_id, duplicate_id, layer, score))
            log.info(
                "Merged entity %r into %r (layer=%s, score=%.3f)",
                duplicate_id,
                canonical_id,
                layer,
                score,
            )

    for pair in grey_band:
        log.info(
            "Grey-band pair left unmerged: %r / %r (layer=%s, score=%.3f)",
            pair.entity_a,
            pair.entity_b,
            pair.layer,
            pair.score,
        )

    return DedupReport(merges=tuple(merges), grey_band=tuple(grey_band))
