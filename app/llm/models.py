"""Model discovery (decision D4): ask the API what this key can use.

Model ids are never hardcoded from memory. This module asks the *active*
provider which models the key can access, classifies them into the cheap
(extraction) and strong (answer) tiers, and picks the newest of each.

    python -m app.llm.models            # show what the key can use
    python -m app.llm.models --write    # also write the picks into config.py

For Gemini, only free-tier families are considered: Flash-Lite for extraction
and Flash for answers. Pro models are excluded because they are paid.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from app.config import PROJECT_ROOT, PROVIDER_ANTHROPIC, PROVIDER_GEMINI, get_settings
from app.errors import AppError, ConfigError
from app.llm.base import FAMILY_CHEAP, FAMILY_STRONG, ModelInfo, newest

_CONFIG_PATH = PROJECT_ROOT / "app" / "config.py"

# Which config.py constant holds each provider's chosen ids, and what the
# tiers are called in that provider's own vocabulary (for readable output).
_PROVIDER_CONFIG = {
    PROVIDER_GEMINI: {
        "extraction_const": "DEFAULT_GEMINI_EXTRACTION_MODEL",
        "answer_const": "DEFAULT_GEMINI_ANSWER_MODEL",
        "cheap_label": "Flash-Lite (free tier)",
        "strong_label": "Flash (free tier)",
        "env_extraction": "GEMINI_EXTRACTION_MODEL",
        "env_answer": "GEMINI_ANSWER_MODEL",
    },
    PROVIDER_ANTHROPIC: {
        "extraction_const": "DEFAULT_ANTHROPIC_EXTRACTION_MODEL",
        "answer_const": "DEFAULT_ANTHROPIC_ANSWER_MODEL",
        "cheap_label": "Haiku",
        "strong_label": "Sonnet",
        "env_extraction": "ANTHROPIC_EXTRACTION_MODEL",
        "env_answer": "ANTHROPIC_ANSWER_MODEL",
    },
}


def discover(provider_name: str | None = None) -> tuple[str, list[ModelInfo]]:
    """List the models a provider's key can use.

    Returns (provider_name, models). With no argument, uses the provider from
    .env; pass one to inspect the other provider without editing .env.
    """
    from app.llm.factory import build_provider, get_provider

    if provider_name is None:
        provider = get_provider()
    else:
        settings = get_settings()
        key = (
            settings.gemini_api_key
            if provider_name == PROVIDER_GEMINI
            else settings.anthropic_api_key
        )
        provider = build_provider(provider_name, key)

    return provider.name, provider.list_models()


def write_choices(
    provider_name: str,
    extraction_id: str,
    answer_id: str,
    config_path: Path | None = None,
) -> Path:
    """Rewrite that provider's two model-id constants in config.py."""
    meta = _PROVIDER_CONFIG.get(provider_name)
    if meta is None:
        raise ConfigError(f"Unknown provider for model discovery: {provider_name!r}")

    path = config_path or _CONFIG_PATH
    source = path.read_text(encoding="utf-8")

    for const, value in (
        (meta["extraction_const"], extraction_id),
        (meta["answer_const"], answer_id),
    ):
        pattern = re.compile(rf'^{const} = ".*"$', re.MULTILINE)
        source, count = pattern.subn(f'{const} = "{value}"', source, count=1)
        if not count:
            raise ConfigError(
                f"Could not find the {const} line in {path}.",
                hint="It may have been edited by hand; restore it to an empty string.",
            )

    # Atomic: this rewrites the app's own settings module, so an interrupt or a
    # read-only install must not leave config.py truncated.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temp_path.write_text(source, encoding="utf-8")
        os.replace(temp_path, path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise ConfigError(
            f"Could not write the chosen model ids to {path}: {exc}",
            hint=(
                "Check the file is writable, or set the ids in .env instead "
                "(the command prints the lines to paste)."
            ),
        ) from exc
    return path


def _render_table(models: Sequence[ModelInfo]) -> str:
    if not models:
        return "  (none available to this key)\n"
    width = max(len(m.id) for m in models)
    rows = sorted(models, key=lambda m: (m.is_stable, m.version_key, m.id), reverse=True)
    lines = [
        f"  {m.id:<{width}}  {m.stability_label:<7}  {m.created_label:<10}  {m.display_name}"
        for m in rows
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover usable LLM model ids.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the chosen ids into app/config.py",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_CONFIG),
        help="override the provider from .env",
    )
    args = parser.parse_args(argv)

    from app.logging_conf import configure_logging

    configure_logging(get_settings().log_level)

    try:
        provider_name, models = discover(args.provider)
    except AppError as exc:
        print(f"ERROR: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"  hint: {exc.hint}", file=sys.stderr)
        return 1

    meta = _PROVIDER_CONFIG[provider_name]

    print(f"\nProvider: {provider_name}")
    print("Models your API key can use, by tier:\n")
    print(f"EXTRACTION tier - {meta['cheap_label']}:")
    print(_render_table([m for m in models if m.family == FAMILY_CHEAP]))
    print(f"ANSWER tier - {meta['strong_label']}:")
    print(_render_table([m for m in models if m.family == FAMILY_STRONG]))

    extraction = newest(models, FAMILY_CHEAP)
    answer = newest(models, FAMILY_STRONG)

    # A key missing one tier can still work: reuse the other for both rather
    # than refusing to run. Symmetric, so either tier can be the missing one.
    if extraction is None and answer is not None:
        extraction = answer
        print(
            f"NOTE: no {meta['cheap_label']} model available; "
            "using the answer-tier model for extraction too.\n"
        )
    elif answer is None and extraction is not None:
        answer = extraction
        print(
            f"NOTE: no {meta['strong_label']} model available; "
            "using the extraction-tier model for answers too.\n"
        )

    if extraction is None or answer is None:
        print(
            "ERROR: this key exposes no usable text-generation models for either tier.",
            file=sys.stderr,
        )
        return 1

    print("Chosen (newest stable in each tier):")
    print(f"  extraction (per-chunk, cheap) : {extraction.id}")
    print(f"  answers    (final synthesis)  : {answer.id}\n")

    if args.write:
        path = write_choices(provider_name, extraction.id, answer.id)
        print(f"Written into {path}")
    else:
        print("Re-run with --write to store these in app/config.py, or set in .env:")
        print(f"  {meta['env_extraction']}={extraction.id}")
        print(f"  {meta['env_answer']}={answer.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
