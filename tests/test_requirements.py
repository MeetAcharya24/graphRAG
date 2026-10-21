"""Every third-party module `app/` imports must be declared in requirements.txt.

This test exists because of a real failure: `app/security/auth.py` imported
`itsdangerous`, which had been pip-installed locally but never added to
requirements.txt. All 177 tests passed, and the Render deploy died on import
with "Exited with status 1". Local tests cannot catch a missing declaration -
the package is already installed locally, so only the declaration is absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.MULTILINE
)

# Import name -> distribution name on PyPI, where the two differ.
_DISTRIBUTION_NAMES = {
    "google": "google-genai",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "multipart": "python-multipart",
}

# Installed as a dependency of something already declared.
_PROVIDED_BY_OTHERS = {
    "starlette",  # fastapi
    "httpx2",  # anthropic
}


def _declared_distributions() -> set[str]:
    names: set[str] = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # fastapi>=0.115 -> fastapi ; uvicorn[standard]>=0.32 -> uvicorn
        name = re.split(r"[<>=!\[;]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def _imported_top_level_modules() -> set[str]:
    modules: set[str] = set()
    for path in APP_DIR.rglob("*.py"):
        for match in _IMPORT_PATTERN.finditer(path.read_text(encoding="utf-8")):
            modules.add(match.group(1))
    return modules


def _is_third_party(module: str) -> bool:
    if module in ("app", "__future__"):
        return False
    if module in sys.stdlib_module_names:
        return False
    return module not in _PROVIDED_BY_OTHERS


def test_every_third_party_import_is_declared() -> None:
    declared = _declared_distributions()
    undeclared = []

    for module in sorted(_imported_top_level_modules()):
        if not _is_third_party(module):
            continue
        distribution = _DISTRIBUTION_NAMES.get(module, module).lower()
        if distribution not in declared:
            undeclared.append(f"{module} (expected '{distribution}' in requirements.txt)")

    assert not undeclared, (
        "app/ imports packages that requirements.txt does not declare, so a "
        "deploy will fail on import even though local tests pass: "
        + ", ".join(undeclared)
    )


def test_itsdangerous_specifically_is_declared() -> None:
    """The exact regression that broke the Render deploy of 431efb2."""
    assert "itsdangerous" in _declared_distributions()


def test_the_requirements_file_parses_to_something_sane() -> None:
    """A guard on the guard: if parsing broke, the test above passes vacuously."""
    declared = _declared_distributions()

    assert "fastapi" in declared
    assert "uvicorn" in declared  # proves the [standard] extra is stripped
    assert "anthropic" in declared  # proves a trailing comment is stripped
    assert len(declared) >= 10


def test_the_scanner_actually_finds_imports() -> None:
    """Another guard: an empty scan would make the main test pass vacuously."""
    modules = _imported_top_level_modules()

    assert "fastapi" in modules
    assert "itsdangerous" in modules
    assert len(modules) > 15
