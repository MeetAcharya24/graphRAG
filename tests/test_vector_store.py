"""Phase 2: the vector store seam.

Written as a conformance suite from the start, parameterized over the store
implementations. The in-memory store is the only one today; Phase 3 adds the
Neo4j one and these same assertions run against it, so the fake cannot drift
from the real thing (D9).
"""

from __future__ import annotations

import math

import pytest

from app.schemas import Chunk
from app.vector.fake import FakeEmbedder
from app.vector.memory_store import InMemoryVectorStore, cosine_similarity

STORES = [InMemoryVectorStore]


@pytest.fixture(params=STORES, ids=lambda cls: cls.__name__)
def store(request):
    return request.param()


@pytest.fixture
def embedder():
    return FakeEmbedder()


def _chunks(doc_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"{doc_id}:{i}", doc_id=doc_id, index=i, text=text)
        for i, text in enumerate(texts)
    ]


def _add(store, embedder, doc_id: str, texts: list[str]) -> list[Chunk]:
    chunks = _chunks(doc_id, texts)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return chunks


# --- cosine similarity ----------------------------------------------------


def test_identical_vectors_score_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_magnitude_does_not_affect_the_score() -> None:
    """Cosine is about direction, so a longer vector must not rank higher."""
    assert cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)


def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


# --- adding and counting --------------------------------------------------


def test_a_new_store_is_empty(store) -> None:
    assert store.count() == 0


def test_adding_chunks_increases_the_count(store, embedder) -> None:
    _add(store, embedder, "doc1", ["alpha", "beta", "gamma"])

    assert store.count() == 3


def test_adding_the_same_chunk_twice_does_not_duplicate_it(store, embedder) -> None:
    """Re-ingesting a document must not double its chunks."""
    _add(store, embedder, "doc1", ["alpha", "beta"])
    _add(store, embedder, "doc1", ["alpha", "beta"])

    assert store.count() == 2


def test_adding_a_mismatched_number_of_vectors_is_rejected(store, embedder) -> None:
    chunks = _chunks("doc1", ["alpha", "beta"])

    with pytest.raises(ValueError):
        store.add(chunks, embedder.embed_documents(["alpha"]))


def test_adding_nothing_is_harmless(store) -> None:
    store.add([], [])

    assert store.count() == 0


# --- querying -------------------------------------------------------------


def test_query_returns_the_closest_chunk_first(store, embedder) -> None:
    _add(
        store,
        embedder,
        "doc1",
        [
            "Helios Labs hired Sarah Chen to lead the compiler team",
            "The cafeteria serves lunch between twelve and two",
            "Rainfall in the northern valley exceeded expectations",
        ],
    )

    results = store.query(embedder.embed_query("Helios Labs compiler Sarah Chen"), k=3)

    assert results[0].chunk.index == 0


def test_query_respects_k(store, embedder) -> None:
    _add(store, embedder, "doc1", [f"text {i}" for i in range(10)])

    assert len(store.query(embedder.embed_query("text 3"), k=4)) == 4


def test_query_returns_everything_when_k_exceeds_the_store(store, embedder) -> None:
    _add(store, embedder, "doc1", ["a", "b"])

    assert len(store.query(embedder.embed_query("a"), k=50)) == 2


def test_query_on_an_empty_store_returns_nothing(store, embedder) -> None:
    assert store.query(embedder.embed_query("anything"), k=5) == []


def test_results_are_ordered_by_descending_score(store, embedder) -> None:
    _add(store, embedder, "doc1", [f"subject number {i}" for i in range(8)])

    results = store.query(embedder.embed_query("subject number 3"), k=8)
    scores = [r.score for r in results]

    assert scores == sorted(scores, reverse=True)


def test_results_carry_the_chunk_and_its_identity(store, embedder) -> None:
    _add(store, embedder, "doc1", ["the only chunk here"])

    result = store.query(embedder.embed_query("the only chunk here"), k=1)[0]

    assert result.chunk.chunk_id == "doc1:0"
    assert result.chunk.doc_id == "doc1"
    assert result.chunk.text == "the only chunk here"
    assert -1.0 <= result.score <= 1.0


def test_a_query_of_the_wrong_dimension_is_rejected(store, embedder) -> None:
    """Guards against a dimension change silently returning nonsense."""
    _add(store, embedder, "doc1", ["alpha"])

    with pytest.raises(ValueError):
        store.query([0.1, 0.2], k=1)


# --- fetching by id ------------------------------------------------------


def test_chunks_can_be_fetched_by_id(store, embedder) -> None:
    _add(store, embedder, "doc1", ["alpha", "beta", "gamma"])

    fetched = store.get_by_ids(["doc1:2", "doc1:0"])

    assert {c.chunk_id for c in fetched} == {"doc1:0", "doc1:2"}


def test_unknown_ids_are_skipped_rather_than_raising(store, embedder) -> None:
    """Phase 7 validates LLM-supplied ids, so unknown ones are expected."""
    _add(store, embedder, "doc1", ["alpha"])

    assert [c.chunk_id for c in store.get_by_ids(["doc1:0", "nope:9"])] == ["doc1:0"]


def test_fetching_no_ids_returns_nothing(store, embedder) -> None:
    _add(store, embedder, "doc1", ["alpha"])

    assert store.get_by_ids([]) == []


# --- deleting -------------------------------------------------------------


def test_deleting_a_document_removes_exactly_its_chunks(store, embedder) -> None:
    _add(store, embedder, "doc1", ["alpha", "beta"])
    _add(store, embedder, "doc2", ["gamma"])

    removed = store.delete_doc("doc1")

    assert removed == 2
    assert store.count() == 1
    assert store.get_by_ids(["doc2:0"])[0].doc_id == "doc2"


def test_deleting_an_unknown_document_removes_nothing(store, embedder) -> None:
    _add(store, embedder, "doc1", ["alpha"])

    assert store.delete_doc("never-existed") == 0
    assert store.count() == 1


def test_deleted_chunks_stop_appearing_in_queries(store, embedder) -> None:
    _add(store, embedder, "doc1", ["a very distinctive phrase about compilers"])
    store.delete_doc("doc1")

    assert store.query(embedder.embed_query("compilers"), k=5) == []


# --- documents the store knows about -------------------------------------


def test_the_store_reports_which_documents_it_holds(store, embedder) -> None:
    _add(store, embedder, "doc1", ["alpha", "beta"])
    _add(store, embedder, "doc2", ["gamma"])

    assert store.document_ids() == {"doc1", "doc2"}


def test_vectors_are_stored_as_given(store, embedder) -> None:
    """A store that normalised or truncated vectors would skew every score."""
    chunks = _chunks("doc1", ["alpha"])
    vector = embedder.embed_documents(["alpha"])[0]
    store.add(chunks, [vector])

    results = store.query(vector, k=1)

    assert results[0].score == pytest.approx(1.0), "a vector must match itself exactly"
    assert not math.isnan(results[0].score)
