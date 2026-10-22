"""Phase 3: the knowledge-graph store, as a conformance suite.

Every assertion runs against BOTH implementations: the in-memory fake by
default, and the real Neo4j store when `-m live` is given. That is the point
(D9) - a fake that drifts from the real database makes the offline suite
worthless, and these are the primitives Phase 6's multi-hop retrieval is built
on, so they have to be right in both.
"""

from __future__ import annotations

import pytest

from app.graph.base import make_entity_id
from app.schemas import Entity, Relation

# --- the fixture graph -----------------------------------------------------
#
#   chen --works_at--> helios --acquired_by--> northwind --produces--> atlas
#                        |
#                    located_in
#                        v
#                      berlin
#
# Chen to atlas is exactly 3 hops, which is the multi-hop case the whole
# project exists to answer.

CHEN = make_entity_id("PERSON", "Sarah Chen")
HELIOS = make_entity_id("ORG", "Helios Labs")
NORTHWIND = make_entity_id("ORG", "Northwind Systems")
ATLAS = make_entity_id("PRODUCT", "Atlas")
BERLIN = make_entity_id("LOCATION", "Berlin")
UNCONNECTED = make_entity_id("PERSON", "Nobody Related")


def _entity(entity_id: str, name: str, kind: str, **kwargs) -> Entity:
    return Entity(entity_id=entity_id, name=name, type=kind, **kwargs)


def _relation(rel_id: str, source: str, target: str, kind: str, **kwargs) -> Relation:
    return Relation(
        rel_id=rel_id,
        source_id=source,
        target_id=target,
        type=kind,
        source_chunk_id=kwargs.pop("source_chunk_id", "doc1:0"),
        **kwargs,
    )


@pytest.fixture
def populated(store):
    """The five-entity chain above, loaded into whichever store is under test."""
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(_entity(HELIOS, "Helios Labs", "ORG"))
    store.upsert_entity(_entity(NORTHWIND, "Northwind Systems", "ORG"))
    store.upsert_entity(_entity(ATLAS, "Atlas", "PRODUCT"))
    store.upsert_entity(_entity(BERLIN, "Berlin", "LOCATION"))
    store.upsert_entity(_entity(UNCONNECTED, "Nobody Related", "PERSON"))

    store.upsert_relation(_relation("r1", CHEN, HELIOS, "works_at"))
    store.upsert_relation(_relation("r2", HELIOS, NORTHWIND, "acquired_by"))
    store.upsert_relation(_relation("r3", NORTHWIND, ATLAS, "produces"))
    store.upsert_relation(_relation("r4", HELIOS, BERLIN, "located_in"))
    return store


# --- entity ids ------------------------------------------------------------


def test_entity_ids_combine_type_and_normalised_name() -> None:
    assert make_entity_id("PERSON", "Sarah Chen") == "PERSON:sarah chen"


def test_entity_ids_ignore_case_and_surrounding_space() -> None:
    assert make_entity_id("org", "  Helios Labs ") == make_entity_id("ORG", "helios labs")


def test_the_same_name_under_different_types_is_a_different_entity() -> None:
    """Apple the company is not Apple the product."""
    assert make_entity_id("ORG", "Apple") != make_entity_id("PRODUCT", "Apple")


# --- upserting entities ----------------------------------------------------


def test_an_entity_can_be_stored_and_read_back(store) -> None:
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))

    found = store.get_entity(CHEN)

    assert found is not None
    assert found.name == "Sarah Chen"
    assert found.type == "PERSON"


def test_an_unknown_entity_reads_back_as_none(store) -> None:
    assert store.get_entity("PERSON:nobody") is None


def test_upserting_the_same_entity_twice_does_not_duplicate_it(store) -> None:
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))

    assert len(store.all_entities()) == 1


def test_upserting_merges_aliases_rather_than_replacing_them(store) -> None:
    """Phase 5's dedup depends on this: aliases accumulate."""
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON", aliases=("S. Chen",)))
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON", aliases=("Dr. Sarah Chen",)))

    found = store.get_entity(CHEN)

    assert set(found.aliases) == {"S. Chen", "Dr. Sarah Chen"}


def test_upserting_merges_mention_chunks_without_duplicates(store) -> None:
    store.upsert_entity(
        _entity(CHEN, "Sarah Chen", "PERSON", mention_chunk_ids=("doc1:0",))
    )
    store.upsert_entity(
        _entity(CHEN, "Sarah Chen", "PERSON", mention_chunk_ids=("doc1:0", "doc2:3"))
    )

    found = store.get_entity(CHEN)

    assert set(found.mention_chunk_ids) == {"doc1:0", "doc2:3"}
    assert len(found.mention_chunk_ids) == 2


def test_a_description_is_filled_in_when_it_was_missing(store) -> None:
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(
        _entity(CHEN, "Sarah Chen", "PERSON", description="VP of Engineering")
    )

    assert store.get_entity(CHEN).description == "VP of Engineering"


# --- upserting relationships ----------------------------------------------


def test_a_relationship_can_be_stored(store) -> None:
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(_entity(HELIOS, "Helios Labs", "ORG"))

    store.upsert_relation(_relation("r1", CHEN, HELIOS, "works_at"))

    assert store.stats()["relation_count"] == 1


def test_upserting_the_same_relationship_twice_does_not_duplicate_it(store) -> None:
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(_entity(HELIOS, "Helios Labs", "ORG"))

    store.upsert_relation(_relation("r1", CHEN, HELIOS, "works_at"))
    store.upsert_relation(_relation("r1", CHEN, HELIOS, "works_at"))

    assert store.stats()["relation_count"] == 1


def test_two_different_relationships_between_the_same_pair_coexist(store) -> None:
    """A person can both work_at and found the same company."""
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(_entity(HELIOS, "Helios Labs", "ORG"))

    store.upsert_relation(_relation("r1", CHEN, HELIOS, "works_at"))
    store.upsert_relation(_relation("r2", CHEN, HELIOS, "founded"))

    assert store.stats()["relation_count"] == 2


def test_every_relationship_keeps_the_chunk_it_came_from(store) -> None:
    """This field is what makes a citation traceable back to its text (R6)."""
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))
    store.upsert_entity(_entity(HELIOS, "Helios Labs", "ORG"))
    store.upsert_relation(
        _relation("r1", CHEN, HELIOS, "works_at", source_chunk_id="doc9:7")
    )

    relation = store.get_relation("r1")

    assert relation is not None
    assert relation.source_chunk_id == "doc9:7"
    assert relation.type == "works_at"


def test_a_relationship_to_a_missing_entity_is_refused(store) -> None:
    """A dangling edge would break every traversal that crossed it."""
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON"))

    with pytest.raises(Exception):
        store.upsert_relation(_relation("r1", CHEN, "ORG:nonexistent", "works_at"))


# --- finding entities by name ---------------------------------------------


def test_an_entity_is_found_by_its_exact_name(populated) -> None:
    found = populated.find_entities_by_name("Sarah Chen")

    assert [e.entity_id for e in found] == [CHEN]


def test_finding_by_name_ignores_case(populated) -> None:
    assert [e.entity_id for e in populated.find_entities_by_name("sarah chen")] == [CHEN]


def test_an_entity_is_found_by_one_of_its_aliases(store) -> None:
    store.upsert_entity(_entity(CHEN, "Sarah Chen", "PERSON", aliases=("S. Chen",)))

    assert [e.entity_id for e in store.find_entities_by_name("S. Chen")] == [CHEN]


def test_an_unknown_name_finds_nothing(populated) -> None:
    assert populated.find_entities_by_name("Someone Else Entirely") == []


# --- neighbours ------------------------------------------------------------


def test_one_hop_returns_only_direct_neighbours(populated) -> None:
    found = {e.entity_id for e in populated.neighbors([CHEN], max_hops=1)}

    assert found == {HELIOS}


def test_two_hops_returns_exactly_the_two_hop_set(populated) -> None:
    found = {e.entity_id for e in populated.neighbors([CHEN], max_hops=2)}

    assert found == {HELIOS, NORTHWIND, BERLIN}


def test_three_hops_reaches_the_far_end_of_the_chain(populated) -> None:
    """The whole point of the project: chen to atlas is 3 hops."""
    found = {e.entity_id for e in populated.neighbors([CHEN], max_hops=3)}

    assert ATLAS in found


def test_neighbours_never_include_the_seeds_themselves(populated) -> None:
    found = {e.entity_id for e in populated.neighbors([CHEN], max_hops=3)}

    assert CHEN not in found


def test_an_unconnected_entity_has_no_neighbours(populated) -> None:
    assert populated.neighbors([UNCONNECTED], max_hops=3) == []


def test_neighbours_of_an_unknown_entity_is_empty_not_an_error(populated) -> None:
    assert populated.neighbors(["PERSON:ghost"], max_hops=2) == []


def test_no_seeds_returns_no_neighbours(populated) -> None:
    assert populated.neighbors([], max_hops=2) == []


def test_several_seeds_are_expanded_together(populated) -> None:
    found = {e.entity_id for e in populated.neighbors([CHEN, NORTHWIND], max_hops=1)}

    assert found == {HELIOS, ATLAS}


def test_traversal_follows_relationships_in_both_directions(populated) -> None:
    """Helios was acquired BY northwind, and must still be reachable from it."""
    found = {e.entity_id for e in populated.neighbors([ATLAS], max_hops=2)}

    assert NORTHWIND in found
    assert HELIOS in found


# --- shortest paths --------------------------------------------------------


def test_a_three_hop_path_is_found(populated) -> None:
    paths = populated.shortest_paths(CHEN, ATLAS, max_hops=3)

    assert paths, "no path found between chen and atlas"
    path = paths[0]
    assert [e.entity_id for e in path.nodes] == [CHEN, HELIOS, NORTHWIND, ATLAS]
    assert [r.type for r in path.edges] == ["works_at", "acquired_by", "produces"]


def test_a_path_longer_than_max_hops_is_not_returned(populated) -> None:
    """The hop limit is a real bound, not a suggestion."""
    assert populated.shortest_paths(CHEN, ATLAS, max_hops=2) == []


def test_a_one_hop_path_is_found(populated) -> None:
    paths = populated.shortest_paths(CHEN, HELIOS, max_hops=3)

    assert len(paths[0].edges) == 1


def test_no_path_exists_to_an_unconnected_entity(populated) -> None:
    assert populated.shortest_paths(CHEN, UNCONNECTED, max_hops=3) == []


def test_a_path_to_an_unknown_entity_is_empty_not_an_error(populated) -> None:
    assert populated.shortest_paths(CHEN, "PERSON:ghost", max_hops=3) == []


def test_a_path_from_an_entity_to_itself_is_empty(populated) -> None:
    assert populated.shortest_paths(CHEN, CHEN, max_hops=3) == []


def test_path_nodes_and_edges_line_up(populated) -> None:
    """N nodes means N-1 edges, or the rendered path would be nonsense."""
    for target in (HELIOS, NORTHWIND, ATLAS):
        for path in populated.shortest_paths(CHEN, target, max_hops=3):
            assert len(path.nodes) == len(path.edges) + 1


# --- stats and listing ----------------------------------------------------


def test_an_empty_store_reports_zero(store) -> None:
    assert store.stats() == {"entity_count": 0, "relation_count": 0}


def test_stats_count_what_was_added(populated) -> None:
    assert populated.stats() == {"entity_count": 6, "relation_count": 4}


def test_all_entities_returns_everything(populated) -> None:
    assert len(populated.all_entities()) == 6


# --- orphan cleanup -------------------------------------------------------


def test_an_entity_with_no_mentions_left_is_removable(store) -> None:
    """Deleting a document must not leave its entities behind forever."""
    store.upsert_entity(
        _entity(CHEN, "Sarah Chen", "PERSON", mention_chunk_ids=("doc1:0",))
    )
    store.upsert_entity(_entity(HELIOS, "Helios Labs", "ORG"))

    removed = store.delete_orphaned_entities()

    assert removed == 1
    assert store.get_entity(CHEN) is not None
    assert store.get_entity(HELIOS) is None


def test_orphan_cleanup_on_an_empty_store_removes_nothing(store) -> None:
    assert store.delete_orphaned_entities() == 0
