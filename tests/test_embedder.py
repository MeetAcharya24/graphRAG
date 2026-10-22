"""Phase 2: turning text into vectors through the Gemini API.

The most important test here is `test_each_text_gets_its_own_vector`. Passing a
list of plain strings to the API returns ONE vector for the whole list, with no
error raised (D17) - so a batching bug is silent, and only an explicit count
assertion catches it.
"""

from __future__ import annotations

import pytest

from app.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSIONS
from app.errors import LLMBadResponse, LLMRateLimited
from app.vector.base import TASK_DOCUMENT, TASK_QUERY
from app.vector.fake import FakeEmbedder


# --- the fake, which the rest of the suite relies on ---------------------


def test_the_fake_embedder_is_deterministic() -> None:
    """Retrieval assertions elsewhere depend on stable vectors."""
    embedder = FakeEmbedder()

    assert embedder.embed_query("hello") == embedder.embed_query("hello")


def test_the_fake_embedder_separates_different_text() -> None:
    embedder = FakeEmbedder()

    assert embedder.embed_query("hello") != embedder.embed_query("goodbye")


def test_the_fake_embedder_returns_the_configured_dimension() -> None:
    embedder = FakeEmbedder(dimension=16)

    assert embedder.dimension == 16
    assert len(embedder.embed_query("x")) == 16


def test_the_fake_embedder_returns_one_vector_per_text() -> None:
    embedder = FakeEmbedder()

    vectors = embedder.embed_documents(["one", "two", "three"])

    assert len(vectors) == 3


def test_the_fake_embedder_scores_shared_words_higher() -> None:
    """Enough signal for "the right chunk ranks first" to be testable offline."""
    from app.vector.memory_store import cosine_similarity

    embedder = FakeEmbedder()
    query = embedder.embed_query("helios labs compiler")
    close = embedder.embed_query("helios labs compiler team")
    far = embedder.embed_query("completely unrelated banana subject")

    assert cosine_similarity(query, close) > cosine_similarity(query, far)


# --- the real embedder, against a recording stub -------------------------


class _StubEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _StubResponse:
    def __init__(self, count: int, dimension: int) -> None:
        self.embeddings = [
            _StubEmbedding([float(i + 1)] * dimension) for i in range(count)
        ]


class _StubClient:
    """Stands in for genai.Client, recording exactly what it was handed."""

    def __init__(self, dimension: int = 8, fail_times: int = 0) -> None:
        self.dimension = dimension
        self.calls: list[dict] = []
        self._fail_times = fail_times
        self.models = self

    def embed_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})

        if self._fail_times > 0:
            self._fail_times -= 1
            raise LLMRateLimited("slow down")

        count = len(contents) if isinstance(contents, list) else 1
        return _StubResponse(count=count, dimension=self.dimension)


def _embedder(**kwargs):
    from app.vector.embedder import GeminiEmbedder

    stub = _StubClient(**kwargs)
    embedder = GeminiEmbedder(
        client=stub, model="test-embed-model", dimension=stub.dimension
    )
    return embedder, stub


def test_each_text_gets_its_own_vector() -> None:
    """D17: the API collapses list[str] into one vector and raises nothing."""
    embedder, _ = _embedder()

    vectors = embedder.embed_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3, "a batch must return one vector per input text"


def test_texts_are_wrapped_as_content_not_passed_as_strings() -> None:
    """The wrapping IS the fix for D17, so it is asserted directly."""
    embedder, stub = _embedder()

    embedder.embed_documents(["alpha", "beta"])

    sent = stub.calls[0]["contents"]
    assert isinstance(sent, list)
    assert len(sent) == 2
    assert not isinstance(sent[0], str), (
        "plain strings make the API return a single concatenated vector"
    )
    assert getattr(sent[0], "parts", None) is not None


def test_documents_and_queries_use_different_task_types() -> None:
    """Retrieval quality depends on telling the model which side it is on."""
    embedder, stub = _embedder()

    embedder.embed_documents(["a chunk"])
    embedder.embed_query("a question")

    assert stub.calls[0]["config"].task_type == TASK_DOCUMENT
    assert stub.calls[1]["config"].task_type == TASK_QUERY


def test_the_configured_dimension_is_requested() -> None:
    embedder, stub = _embedder()

    embedder.embed_query("x")

    assert stub.calls[0]["config"].output_dimensionality == embedder.dimension


def test_an_empty_list_makes_no_api_call() -> None:
    """Nothing to embed must cost nothing."""
    embedder, stub = _embedder()

    assert embedder.embed_documents([]) == []
    assert stub.calls == []


def test_a_large_batch_is_split_into_several_calls() -> None:
    embedder, stub = _embedder()
    texts = [f"text {i}" for i in range(EMBEDDING_BATCH_SIZE * 2 + 3)]

    vectors = embedder.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert len(stub.calls) == 3
    assert all(len(call["contents"]) <= EMBEDDING_BATCH_SIZE for call in stub.calls)


def test_a_rate_limit_is_retried_without_duplicating_work() -> None:
    """One transient 429 must not drop or double any vector."""
    embedder, stub = _embedder(fail_times=1)

    vectors = embedder.embed_documents(["a", "b", "c"])

    assert len(vectors) == 3
    assert len(stub.calls) == 2  # the failed attempt, then the successful one


def test_a_wrong_vector_count_from_the_api_is_an_error() -> None:
    """Defence in depth: if the API changes shape, fail loudly, not silently."""
    from app.vector.embedder import GeminiEmbedder

    class _Mismatched(_StubClient):
        def embed_content(self, *, model, contents, config=None):
            self.calls.append({"model": model, "contents": contents, "config": config})
            return _StubResponse(count=1, dimension=self.dimension)

    stub = _Mismatched()
    embedder = GeminiEmbedder(client=stub, model="m", dimension=stub.dimension)

    with pytest.raises(LLMBadResponse, match="expected 3"):
        embedder.embed_documents(["a", "b", "c"])


def test_a_wrong_dimension_from_the_api_is_an_error() -> None:
    """A silent dimension change would corrupt the Neo4j vector index."""
    from app.vector.embedder import GeminiEmbedder

    stub = _StubClient(dimension=8)
    embedder = GeminiEmbedder(client=stub, model="m", dimension=16)

    with pytest.raises(LLMBadResponse, match="dimension"):
        embedder.embed_query("x")


def test_the_configured_dimension_is_the_one_we_chose() -> None:
    """768 is deliberate (D16): ~3 KB per chunk instead of ~12 KB at 3072."""
    assert EMBEDDING_DIMENSIONS == 768
