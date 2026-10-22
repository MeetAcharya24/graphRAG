"""Phase 5: entity deduplication (layers 1-5, D2).

*In simple words: realise "Sarah Chen", "S. Chen" and "Dr. Sarah Chen" are one
person - without merging two different people who share a name.*

Layers 1-4 are pure and tested directly through `match_pair`, against a
`group` of candidates so the "unique last name" rule has something to be
unique against. Layer 5 (embedding cosine) and the end-to-end merge only make
sense with a store, so those run through `resolve_duplicates` against the
in-memory `GraphStore`.
"""

from __future__ import annotations

import logging

import pytest

from app.graph.base import make_entity_id
from app.graph.dedup import (
    DedupReport,
    dedup_key,
    match_pair,
    merge_entity_into,
    resolve_duplicates,
)
from app.graph.memory_store import InMemoryGraphStore
from app.schemas import Entity


def _entity(name: str, kind: str, **kwargs) -> Entity:
    entity_id = kwargs.pop("entity_id", None) or make_entity_id(kind, name)
    return Entity(entity_id=entity_id, name=name, type=kind, **kwargs)


# --- Layer 1: the dedup-specific normal form -------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Dr. Sarah Chen", "sarah chen"),
        ("Sarah Chen", "sarah chen"),
        ("Helios Labs, Inc.", "helios labs"),
        ("Helios Labs", "helios labs"),
        ("S. Chen", "s chen"),
        ("  Sarah   Chen  ", "sarah chen"),
        # Generational suffixes are kept - stripping "Jr." would erase the
        # one signal that says these are two different people.
        ("John Smith Jr.", "john smith jr"),
    ],
)
def test_dedup_key_normalizes_titles_suffixes_and_punctuation(raw, expected) -> None:
    assert dedup_key(raw) == expected


# --- Layers 2-4: pairwise matching ------------------------------------------


def test_different_types_never_match_even_with_the_same_name() -> None:
    apple_org = _entity("Apple", "ORG")
    apple_product = _entity("Apple", "PRODUCT")

    result = match_pair(apple_org, apple_product, group=[apple_org, apple_product])

    assert result.match is None


def test_layer2_exact_match_after_title_stripping() -> None:
    plain = _entity("Sarah Chen", "PERSON")
    titled = _entity("Dr. Sarah Chen", "PERSON", entity_id="PERSON:dr sarah chen")

    result = match_pair(plain, titled, group=[plain, titled])

    assert result.match is not None
    assert result.match.layer == "exact"


def test_layer2_exact_match_via_a_known_alias() -> None:
    helios = _entity("Helios Labs", "ORG", aliases=("Helios",))
    alias_only = _entity("Helios", "ORG", entity_id="ORG:helios-alt")

    result = match_pair(helios, alias_only, group=[helios, alias_only])

    assert result.match is not None
    assert result.match.layer == "exact"


def test_layer2_exact_match_after_stripping_an_org_suffix() -> None:
    plain = _entity("Helios Labs", "ORG")
    suffixed = _entity("Helios Labs, Inc.", "ORG", entity_id="ORG:helios labs, inc.")

    result = match_pair(plain, suffixed, group=[plain, suffixed])

    assert result.match is not None
    assert result.match.layer == "exact"


def test_layer3_initial_expansion_matches() -> None:
    full = _entity("Sarah Chen", "PERSON")
    initial = _entity("S. Chen", "PERSON", entity_id="PERSON:s. chen")

    result = match_pair(full, initial, group=[full, initial])

    assert result.match is not None
    assert result.match.layer == "alias"


def test_layer3_acronym_expansion_matches() -> None:
    expansion = _entity("International Business Machines", "ORG")
    acronym = _entity("IBM", "ORG", entity_id="ORG:ibm")

    result = match_pair(expansion, acronym, group=[expansion, acronym])

    assert result.match is not None
    assert result.match.layer == "alias"


def test_layer3_unique_last_name_matches_when_unambiguous() -> None:
    full = _entity("Sarah Chen", "PERSON")
    bare = _entity("Chen", "PERSON", entity_id="PERSON:chen")

    result = match_pair(full, bare, group=[full, bare])

    assert result.match is not None
    assert result.match.layer == "alias"


def test_layer3_unique_last_name_refuses_when_ambiguous() -> None:
    """Two Chens in the group means a bare "Chen" cannot be resolved safely."""
    sarah = _entity("Sarah Chen", "PERSON")
    victor = _entity("Victor Chen", "PERSON", entity_id="PERSON:victor chen")
    bare = _entity("Chen", "PERSON", entity_id="PERSON:chen")
    group = [sarah, victor, bare]

    assert match_pair(sarah, bare, group=group).match is None
    assert match_pair(victor, bare, group=group).match is None


def test_layer4_fuzzy_match_above_threshold() -> None:
    a = _entity("Northwind Systems", "ORG")
    b = _entity("Northwind System", "ORG", entity_id="ORG:northwind system")

    result = match_pair(a, b, group=[a, b])

    assert result.match is not None
    assert result.match.layer == "fuzzy"
    assert result.match.score >= 0.92


def test_layer4_grey_band_is_reported_but_not_matched() -> None:
    a = _entity("Northwind Systems", "ORG")
    b = _entity("Northwind Sys", "ORG", entity_id="ORG:northwind sys")

    result = match_pair(a, b, group=[a, b])

    assert result.match is None
    assert result.grey is not None
    assert result.grey.layer == "fuzzy"
    assert 0.80 <= result.grey.score < 0.92


def test_generational_suffix_blocks_a_match_fuzzy_would_otherwise_give() -> None:
    """John Smith and John Smith Jr. are different people - a father and son."""
    father = _entity("John Smith", "PERSON")
    son = _entity("John Smith Jr.", "PERSON", entity_id="PERSON:john smith jr.")

    result = match_pair(father, son, group=[father, son])

    assert result.match is None
    assert result.grey is None


def test_unrelated_names_neither_match_nor_land_in_the_grey_band() -> None:
    a = _entity("Sarah Chen", "PERSON")
    b = _entity("Victor Alvarez", "PERSON", entity_id="PERSON:victor alvarez")

    result = match_pair(a, b, group=[a, b])

    assert result.match is None
    assert result.grey is None


# --- end-to-end resolution against a store ----------------------------------


def test_the_sarah_chen_trio_collapses_to_one_node() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(
        _entity("Sarah Chen", "PERSON", mention_chunk_ids=("doc1:0", "doc1:1", "doc1:2"))
    )
    store.upsert_entity(
        _entity("S. Chen", "PERSON", entity_id="PERSON:s. chen", mention_chunk_ids=("doc2:0",))
    )
    store.upsert_entity(
        _entity(
            "Dr. Sarah Chen",
            "PERSON",
            entity_id="PERSON:dr. sarah chen",
            mention_chunk_ids=("doc3:0",),
        )
    )

    report = resolve_duplicates(store)

    entities = store.all_entities()
    assert len(entities) == 1
    canonical = entities[0]
    assert canonical.entity_id == make_entity_id("PERSON", "Sarah Chen")
    assert set(canonical.aliases) == {"S. Chen", "Dr. Sarah Chen"}
    assert set(canonical.mention_chunk_ids) == {"doc1:0", "doc1:1", "doc1:2", "doc2:0", "doc3:0"}
    assert set(canonical.merged_from) == {"PERSON:s. chen", "PERSON:dr. sarah chen"}
    assert len(report.merges) == 2


def test_helios_labs_and_its_inc_suffix_merge() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(_entity("Helios Labs", "ORG", mention_chunk_ids=("doc1:0", "doc1:1")))
    store.upsert_entity(
        _entity(
            "Helios Labs, Inc.",
            "ORG",
            entity_id="ORG:helios labs, inc.",
            mention_chunk_ids=("doc2:0",),
        )
    )

    resolve_duplicates(store)

    assert len(store.all_entities()) == 1


def test_person_apple_never_merges_with_org_apple() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(_entity("Apple", "PERSON", entity_id="PERSON:apple"))
    store.upsert_entity(_entity("Apple", "ORG", entity_id="ORG:apple"))

    report = resolve_duplicates(store)

    assert len(store.all_entities()) == 2
    assert report.merges == ()


def test_two_different_john_smiths_stay_separate() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(_entity("John Smith", "PERSON"))
    store.upsert_entity(_entity("John Smith Jr.", "PERSON", entity_id="PERSON:john smith jr."))

    report = resolve_duplicates(store)

    assert len(store.all_entities()) == 2
    assert report.merges == ()


def test_merging_is_order_independent() -> None:
    def build(order):
        store = InMemoryGraphStore()
        entities = {
            "full": _entity(
                "Sarah Chen", "PERSON", mention_chunk_ids=("doc1:0", "doc1:1")
            ),
            "initial": _entity(
                "S. Chen", "PERSON", entity_id="PERSON:s. chen", mention_chunk_ids=("doc2:0",)
            ),
            "titled": _entity(
                "Dr. Sarah Chen",
                "PERSON",
                entity_id="PERSON:dr. sarah chen",
                mention_chunk_ids=("doc3:0",),
            ),
        }
        for key in order:
            store.upsert_entity(entities[key])
        resolve_duplicates(store)
        canonical = store.all_entities()[0]
        return canonical.entity_id, set(canonical.aliases), set(canonical.mention_chunk_ids)

    forward = build(["full", "initial", "titled"])
    backward = build(["titled", "initial", "full"])

    assert forward == backward


def test_relationships_survive_rewiring_onto_the_canonical_entity() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(
        _entity("Sarah Chen", "PERSON", mention_chunk_ids=("doc1:0", "doc1:1"))
    )
    store.upsert_entity(
        _entity("S. Chen", "PERSON", entity_id="PERSON:s. chen", mention_chunk_ids=("doc2:0",))
    )
    helios = _entity("Helios Labs", "ORG")
    store.upsert_entity(helios)
    from app.schemas import Relation

    store.upsert_relation(
        Relation(
            rel_id="r1",
            source_id="PERSON:s. chen",
            target_id=helios.entity_id,
            type="works_at",
            source_chunk_id="doc2:0",
        )
    )

    resolve_duplicates(store)

    canonical_id = make_entity_id("PERSON", "Sarah Chen")
    # The merge mints a fresh rel_id for the rewired edge (see
    # `_rewired_rel_id`'s docstring), so look it up by its new endpoints
    # rather than assuming the old "r1" survives.
    rewired = [r for r in store.relations_of(canonical_id) if r.type == "works_at"]
    assert len(rewired) == 1
    assert rewired[0].source_id == canonical_id
    assert rewired[0].target_id == helios.entity_id
    assert store.get_relation("r1") is None, "the old id must not linger"


def test_grey_band_pairs_are_logged_and_left_unmerged(caplog) -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(_entity("Northwind Systems", "ORG"))
    store.upsert_entity(_entity("Northwind Sys", "ORG", entity_id="ORG:northwind sys"))

    with caplog.at_level(logging.INFO):
        report = resolve_duplicates(store)

    assert len(store.all_entities()) == 2
    assert len(report.grey_band) >= 1
    assert any("grey" in record.message.lower() for record in caplog.records)


def test_a_tiebreaker_hook_can_resolve_a_grey_band_pair() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(_entity("Northwind Systems", "ORG"))
    store.upsert_entity(_entity("Northwind Sys", "ORG", entity_id="ORG:northwind sys"))

    def always_merge(a, b, layer, score) -> str:
        return min(a.entity_id, b.entity_id)

    report = resolve_duplicates(store, tiebreaker=always_merge)

    assert len(store.all_entities()) == 1
    assert len(report.merges) == 1


# --- merge_entity_into, directly ---------------------------------------------


def test_merge_entity_into_unions_aliases_mentions_and_merged_from() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(
        _entity("Sarah Chen", "PERSON", aliases=("Dr. Chen",), mention_chunk_ids=("doc1:0",))
    )
    store.upsert_entity(
        _entity(
            "S. Chen",
            "PERSON",
            entity_id="PERSON:s. chen",
            mention_chunk_ids=("doc2:0",),
        )
    )

    canonical_id = make_entity_id("PERSON", "Sarah Chen")
    merged = merge_entity_into(store, canonical_id, "PERSON:s. chen")

    assert set(merged.aliases) == {"Dr. Chen", "S. Chen"}
    assert set(merged.mention_chunk_ids) == {"doc1:0", "doc2:0"}
    assert merged.merged_from == ("PERSON:s. chen",)
    assert store.get_entity("PERSON:s. chen") is None


def test_merge_entity_into_drops_a_relation_that_would_become_a_self_loop() -> None:
    store = InMemoryGraphStore()
    canonical = _entity("Sarah Chen", "PERSON")
    duplicate = _entity("S. Chen", "PERSON", entity_id="PERSON:s. chen")
    store.upsert_entity(canonical)
    store.upsert_entity(duplicate)
    from app.schemas import Relation

    store.upsert_relation(
        Relation(
            rel_id="r1",
            source_id=duplicate.entity_id,
            target_id=canonical.entity_id,
            type="OTHER",
            source_chunk_id="doc1:0",
        )
    )

    merge_entity_into(store, canonical.entity_id, duplicate.entity_id)

    assert store.get_relation("r1") is None


def test_merge_entity_into_rejects_merging_an_entity_into_itself() -> None:
    store = InMemoryGraphStore()
    store.upsert_entity(_entity("Sarah Chen", "PERSON"))

    with pytest.raises(ValueError):
        merge_entity_into(store, make_entity_id("PERSON", "Sarah Chen"), make_entity_id("PERSON", "Sarah Chen"))


def test_dedup_report_defaults_are_empty_tuples() -> None:
    report = DedupReport()

    assert report.merges == ()
    assert report.grey_band == ()
