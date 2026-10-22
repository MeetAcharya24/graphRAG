"""A scripted LLM provider for tests - no network, no cost, no quota.

Returns whatever the test told it to return, and records what it was asked, so
tests assert on wiring and prompt construction rather than on model quality.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.llm.base import ModelInfo, SchemaT


class FakeLLM:
    """Implements `LLMProvider` from a queue of canned responses."""

    name = "fake"

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        models: Sequence[ModelInfo] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._models = list(models or [])
        self.calls: list[dict] = []

    def queue(self, response: Any) -> None:
        """Add one response to be returned by the next call."""
        self._responses.append(response)

    def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[SchemaT],
        system: str | None = None,
        max_tokens: int,
        minimal_reasoning: bool = False,
    ) -> SchemaT:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "max_tokens": max_tokens,
                "minimal_reasoning": minimal_reasoning,
            }
        )

        if not self._responses:
            raise AssertionError(
                f"FakeLLM had no queued response for a {schema.__name__} call. "
                "Queue one with FakeLLM(responses=[...]) or .queue(...)."
            )

        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, schema):
            return response
        # A dict is validated, so a test cannot queue a shape the real model
        # could never produce.
        return schema.model_validate(response)
