"""Phase 0 / decision D4: model discovery picks the newest free-tier model.

These tests use canned payloads shaped like each provider's models-list
response, so they prove the selection logic without spending money, needing a
network, or guessing a model name.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import PROVIDER_ANTHROPIC, PROVIDER_GEMINI
from app.errors import ConfigError
from app.llm.base import (
    FAMILY_CHEAP,
    FAMILY_EMBED,
    FAMILY_STRONG,
    newest,
    parse_version_key,
)
from app.llm.gemini_provider import classify as gemini_classify
from app.llm.gemini_provider import parse_models as gemini_parse
from app.llm.gemini_provider import strip_prefix
from app.llm.models import write_choices

GENERATE = ["generateContent", "countTokens"]

# Shaped like google-genai's models.list() output. Names are synthetic: the
# real ids come from the live call, never from this file.
GEMINI_RESPONSE = [
    {
        "name": "models/gemini-3.0-flash",
        "display_name": "Gemini 3.0 Flash",
        "supported_actions": GENERATE,
    },
    {
        "name": "models/gemini-3.0-flash-lite",
        "display_name": "Gemini 3.0 Flash-Lite",
        "supported_actions": GENERATE,
    },
    {
        "name": "models/gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "supported_actions": GENERATE,
    },
    {
        "name": "models/gemini-2.5-flash-lite",
        "display_name": "Gemini 2.5 Flash-Lite",
        "supported_actions": GENERATE,
    },
    {
        "name": "models/gemini-3.0-flash-preview-09-2025",
        "display_name": "Gemini 3.0 Flash Preview",
        "supported_actions": GENERATE,
    },
    # Paid tier - must never be chosen.
    {
        "name": "models/gemini-3.0-pro",
        "display_name": "Gemini 3.0 Pro",
        "supported_actions": GENERATE,
    },
    # Image generation. These declare generateContent, so only a segment-aware
    # filter excludes them - the reason this fixture exists.
    {
        "name": "models/gemini-3.1-flash-image",
        "display_name": "Nano Banana 2",
        "supported_actions": GENERATE,
    },
    {
        "name": "models/gemini-3.1-flash-lite-image",
        "display_name": "Nano Banana 2 Lite",
        "supported_actions": GENERATE,
    },
    {
        "name": "models/gemini-2.5-flash-image",
        "display_name": "Nano Banana",
        "supported_actions": GENERATE,
    },
    # Text-capable, so it stays eligible; it just ranks low by version.
    {
        "name": "models/gemini-omni-1.1-flash",
        "display_name": "Gemini Omni 1.1 Flash",
        "supported_actions": GENERATE,
    },
    # Not text generation.
    {
        "name": "models/text-embedding-004",
        "display_name": "Text Embedding",
        "supported_actions": ["embedContent"],
    },
    {
        "name": "models/imagen-4.0-generate",
        "display_name": "Imagen 4",
        "supported_actions": ["predict"],
    },
    {
        "name": "models/veo-3.0-generate",
        "display_name": "Veo 3",
        "supported_actions": ["predictLongRunning"],
    },
]


# --- Gemini classification -------------------------------------------------


def test_strip_prefix_removes_the_models_namespace() -> None:
    assert strip_prefix("models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert strip_prefix("gemini-2.5-flash") == "gemini-2.5-flash"


def test_flash_lite_is_the_extraction_tier() -> None:
    assert gemini_classify("gemini-2.5-flash-lite", GENERATE) == FAMILY_CHEAP


def test_flash_is_the_answer_tier() -> None:
    assert gemini_classify("gemini-2.5-flash", GENERATE) == FAMILY_STRONG


def test_pro_models_are_excluded_because_they_are_paid() -> None:
    assert gemini_classify("gemini-3.0-pro", GENERATE) is None


def test_non_text_models_are_excluded() -> None:
    for model_id in (
        "text-embedding-004",
        "imagen-4.0-generate",
        "veo-3.0-generate",
        "gemini-2.5-flash-tts",
    ):
        assert gemini_classify(model_id, GENERATE) is None, model_id


def test_a_model_advertising_embed_content_is_the_embedding_tier() -> None:
    """D16: the embedding tier is found by capability, not by name."""
    assert gemini_classify("gemini-embedding-2", ["embedContent"]) == FAMILY_EMBED
    # Even a flash-named model, if it only embeds, belongs to the embed tier.
    assert gemini_classify("gemini-2.5-flash", ["embedContent"]) == FAMILY_EMBED


def test_models_that_can_neither_generate_nor_embed_are_excluded() -> None:
    assert gemini_classify("some-model", ["predict"]) is None
    assert gemini_classify("gemini-2.5-flash", ["countTokens"]) is None


def test_image_generation_models_are_excluded() -> None:
    """These declare generateContent but return images, not text."""
    for model_id in (
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image-preview",
    ):
        assert gemini_classify(model_id, GENERATE) is None, model_id


def test_omni_models_stay_eligible_because_they_generate_text() -> None:
    assert gemini_classify("gemini-omni-1.1-flash", GENERATE) == FAMILY_STRONG


def test_parse_models_keeps_only_free_text_families() -> None:
    parsed = gemini_parse(GEMINI_RESPONSE)

    assert {m.id for m in parsed} == {
        "gemini-3.0-flash",
        "gemini-3.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.0-flash-preview-09-2025",
        "gemini-omni-1.1-flash",
        # Kept since D16, as the embedding tier rather than a text tier.
        "text-embedding-004",
    }


def test_parse_models_skips_entries_without_a_name() -> None:
    assert gemini_parse([{"display_name": "nameless"}]) == []


# --- version ranking -------------------------------------------------------


def test_parse_version_key_reads_the_version_from_the_id() -> None:
    assert parse_version_key("gemini-2.5-flash") == (2, 5)
    assert parse_version_key("gemini-3.0-flash-lite") == (3, 0)
    assert parse_version_key("claude-haiku-4-5") == (4, 5)


def test_parse_version_key_handles_an_unversioned_id() -> None:
    assert parse_version_key("some-model") == ()


def test_parse_version_key_ignores_a_trailing_release_date() -> None:
    """A date suffix is not a version component.

    Left in, `-20250514` parses as minor version 20250514 and an older model
    outranks a newer one.
    """
    assert parse_version_key("claude-sonnet-4-20250514") == (4, 0)
    assert parse_version_key("claude-sonnet-4-5-20250929") == (4, 5)
    assert parse_version_key("claude-3-7-sonnet-20250219") == (3, 7)


def test_newest_picks_the_highest_version_per_tier() -> None:
    parsed = gemini_parse(GEMINI_RESPONSE)

    assert newest(parsed, FAMILY_CHEAP).id == "gemini-3.0-flash-lite"
    assert newest(parsed, FAMILY_STRONG).id == "gemini-3.0-flash"


def test_a_preview_never_beats_a_stable_model() -> None:
    """Gemini exposes previews alongside stable ids; stable must win."""
    parsed = gemini_parse(GEMINI_RESPONSE)
    chosen = newest(parsed, FAMILY_STRONG)

    assert chosen.is_stable is True
    assert "preview" not in chosen.id


def test_a_preview_is_chosen_only_when_nothing_stable_exists() -> None:
    parsed = gemini_parse(
        [{"name": "models/gemini-9.0-flash-preview", "supported_actions": GENERATE}]
    )

    assert newest(parsed, FAMILY_STRONG).id == "gemini-9.0-flash-preview"


def test_newest_returns_none_for_an_absent_tier() -> None:
    parsed = gemini_parse(
        [{"name": "models/gemini-2.5-flash", "supported_actions": GENERATE}]
    )

    assert newest(parsed, FAMILY_CHEAP) is None


def test_a_higher_versioned_image_model_never_wins_either_tier() -> None:
    """Regression guard for the leak found during live discovery.

    An image model with the highest version number must not be selected: it
    would be chosen silently and then fail on every question at runtime.
    """
    parsed = gemini_parse(
        [
            {"name": "models/gemini-9.0-flash-image", "supported_actions": GENERATE},
            {"name": "models/gemini-9.0-flash-lite-image", "supported_actions": GENERATE},
            {"name": "models/gemini-4.0-flash", "supported_actions": GENERATE},
            {"name": "models/gemini-4.0-flash-lite", "supported_actions": GENERATE},
        ]
    )

    assert newest(parsed, FAMILY_STRONG).id == "gemini-4.0-flash"
    assert newest(parsed, FAMILY_CHEAP).id == "gemini-4.0-flash-lite"


def test_the_live_listing_shape_picks_the_text_models() -> None:
    """End-to-end over the full fixture, including the image models."""
    parsed = gemini_parse(GEMINI_RESPONSE)

    assert newest(parsed, FAMILY_CHEAP).id == "gemini-3.0-flash-lite"
    assert newest(parsed, FAMILY_STRONG).id == "gemini-3.0-flash"
    assert not any("image" in m.id for m in parsed)


# --- Anthropic side still works -------------------------------------------


def test_anthropic_models_classify_into_the_same_two_tiers() -> None:
    from app.llm.anthropic_provider import parse_models as anthropic_parse

    parsed = anthropic_parse(
        [
            {"id": "claude-haiku-4-5", "created_at": "2025-10-01T00:00:00Z"},
            {"id": "claude-sonnet-5", "created_at": "2026-02-01T00:00:00Z"},
            {"id": "claude-opus-5", "created_at": "2026-04-01T00:00:00Z"},
        ]
    )

    assert newest(parsed, FAMILY_CHEAP).id == "claude-haiku-4-5"
    assert newest(parsed, FAMILY_STRONG).id == "claude-sonnet-5"
    assert "claude-opus-5" not in {m.id for m in parsed}


def test_date_suffixed_anthropic_ids_rank_by_date_not_by_suffix() -> None:
    """Regression: the real API returns date-suffixed ids.

    The earlier fixtures used date-less ids, so this ordering bug passed the
    suite while selecting an older model in production.
    """
    from app.llm.anthropic_provider import parse_models as anthropic_parse

    parsed = anthropic_parse(
        [
            {"id": "claude-sonnet-4-20250514", "created_at": "2025-05-14T00:00:00Z"},
            {"id": "claude-sonnet-4-5-20250929", "created_at": "2025-09-29T00:00:00Z"},
            {"id": "claude-haiku-3-5-20241022", "created_at": "2024-10-22T00:00:00Z"},
            {"id": "claude-haiku-4-5-20251001", "created_at": "2025-10-01T00:00:00Z"},
        ]
    )

    assert newest(parsed, FAMILY_STRONG).id == "claude-sonnet-4-5-20250929"
    assert newest(parsed, FAMILY_CHEAP).id == "claude-haiku-4-5-20251001"


def test_a_supplied_creation_date_outranks_the_parsed_version() -> None:
    """When the API gives a date it is authoritative."""
    from app.llm.anthropic_provider import parse_models as anthropic_parse

    parsed = anthropic_parse(
        [
            {"id": "claude-sonnet-9-0", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "claude-sonnet-4-5", "created_at": "2026-01-01T00:00:00Z"},
        ]
    )

    assert newest(parsed, FAMILY_STRONG).id == "claude-sonnet-4-5"


def test_anthropic_models_keep_their_creation_dates() -> None:
    from app.llm.anthropic_provider import parse_models as anthropic_parse

    parsed = anthropic_parse(
        [{"id": "claude-sonnet-5", "created_at": "2026-02-01T00:00:00Z"}]
    )

    assert parsed[0].created_at == datetime(2026, 2, 1, tzinfo=timezone.utc)
    assert parsed[0].created_label == "2026-02-01"


def test_anthropic_tolerates_a_bad_date() -> None:
    from app.llm.anthropic_provider import parse_models as anthropic_parse

    parsed = anthropic_parse([{"id": "claude-haiku-4-5", "created_at": "not-a-date"}])

    assert parsed[0].created_at is None
    assert parsed[0].created_label == "-"


# --- writing the choices into config.py ------------------------------------


CONFIG_STUB = (
    'DEFAULT_GEMINI_EXTRACTION_MODEL = ""\n'
    'DEFAULT_GEMINI_ANSWER_MODEL = ""\n'
    'DEFAULT_ANTHROPIC_EXTRACTION_MODEL = ""\n'
    'DEFAULT_ANTHROPIC_ANSWER_MODEL = ""\n'
    "OTHER = 1\n"
)


def test_write_choices_fills_in_the_gemini_slots(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text(CONFIG_STUB, encoding="utf-8")

    write_choices(
        PROVIDER_GEMINI, "gemini-3.0-flash-lite", "gemini-3.0-flash", config_path=config
    )

    written = config.read_text(encoding="utf-8")
    assert 'DEFAULT_GEMINI_EXTRACTION_MODEL = "gemini-3.0-flash-lite"' in written
    assert 'DEFAULT_GEMINI_ANSWER_MODEL = "gemini-3.0-flash"' in written
    assert "OTHER = 1" in written


def test_write_choices_leaves_the_other_providers_slots_alone(tmp_path: Path) -> None:
    """Writing Gemini ids must not touch the Anthropic ones."""
    config = tmp_path / "config.py"
    config.write_text(CONFIG_STUB, encoding="utf-8")

    write_choices(PROVIDER_GEMINI, "a", "b", config_path=config)

    written = config.read_text(encoding="utf-8")
    assert 'DEFAULT_ANTHROPIC_EXTRACTION_MODEL = ""' in written
    assert 'DEFAULT_ANTHROPIC_ANSWER_MODEL = ""' in written


def test_write_choices_targets_the_anthropic_slots_too(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text(CONFIG_STUB, encoding="utf-8")

    write_choices(
        PROVIDER_ANTHROPIC, "claude-haiku-4-5", "claude-sonnet-5", config_path=config
    )

    written = config.read_text(encoding="utf-8")
    assert 'DEFAULT_ANTHROPIC_EXTRACTION_MODEL = "claude-haiku-4-5"' in written
    assert 'DEFAULT_GEMINI_EXTRACTION_MODEL = ""' in written


def test_write_choices_is_rerunnable(tmp_path: Path) -> None:
    """Discovery must be safe to run again when a newer model ships."""
    config = tmp_path / "config.py"
    config.write_text(
        'DEFAULT_GEMINI_EXTRACTION_MODEL = "old-lite"\n'
        'DEFAULT_GEMINI_ANSWER_MODEL = "old-flash"\n',
        encoding="utf-8",
    )

    write_choices(PROVIDER_GEMINI, "new-lite", "new-flash", config_path=config)

    written = config.read_text(encoding="utf-8")
    assert "old-lite" not in written
    assert 'DEFAULT_GEMINI_EXTRACTION_MODEL = "new-lite"' in written


def test_write_choices_raises_when_the_markers_are_gone(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text("# someone deleted the markers\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        write_choices(PROVIDER_GEMINI, "a", "b", config_path=config)

    assert excinfo.value.hint is not None


def test_write_choices_rejects_an_unknown_provider(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    config.write_text(CONFIG_STUB, encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown provider"):
        write_choices("llama", "a", "b", config_path=config)


# --- the embedding tier (D16) ---------------------------------------------


EMBED_RESPONSE = [
    {"name": "models/gemini-embedding-2", "supported_actions": ["embedContent"]},
    {"name": "models/gemini-embedding-001", "supported_actions": ["embedContent"]},
    {
        "name": "models/gemini-embedding-2-preview",
        "supported_actions": ["embedContent"],
    },
]


def test_the_newest_stable_embedding_model_is_chosen() -> None:
    """The exact shape the live key returned on 2026-10-22."""
    parsed = gemini_parse(EMBED_RESPONSE)

    assert len(parsed) == 3
    assert newest(parsed, FAMILY_EMBED).id == "gemini-embedding-2"


def test_an_embedding_preview_never_beats_a_stable_one() -> None:
    chosen = newest(gemini_parse(EMBED_RESPONSE), FAMILY_EMBED)

    assert chosen.is_stable is True
    assert "preview" not in chosen.id


def test_embedding_models_do_not_leak_into_the_text_tiers() -> None:
    """An embedding model used for answers would fail every request."""
    parsed = gemini_parse(EMBED_RESPONSE)

    assert newest(parsed, FAMILY_CHEAP) is None
    assert newest(parsed, FAMILY_STRONG) is None


def test_text_models_do_not_leak_into_the_embedding_tier() -> None:
    parsed = gemini_parse(
        [{"name": "models/gemini-3.0-flash", "supported_actions": GENERATE}]
    )

    assert newest(parsed, FAMILY_EMBED) is None


def test_a_paid_embedding_model_is_never_classified_as_free() -> None:
    """Ordering bug: the capability check ran before the paid filter.

    A paid embedding model advertising embedContent was classified as the free
    embed tier, could win its tier, and would be written into config.py by
    --write - silently billing the user, against D5/D6.
    """
    assert gemini_classify("gemini-embedding-pro", ["embedContent"]) is None
    assert gemini_classify("gemini-pro-embedding-2", ["embedContent"]) is None


def test_a_non_text_model_is_never_classified_as_an_embedder() -> None:
    assert gemini_classify("imagen-4.0-embedding", ["embedContent"]) is None
    assert gemini_classify("gemini-2.5-flash-audio", ["embedContent"]) is None


def test_the_free_embedding_model_still_classifies() -> None:
    """The filter must not be so blunt it rejects what we actually use."""
    assert gemini_classify("gemini-embedding-2", ["embedContent"]) == FAMILY_EMBED
