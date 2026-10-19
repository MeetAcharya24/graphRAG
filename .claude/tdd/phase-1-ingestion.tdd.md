# TDD Evidence: Phase 1 — Document loading and chunking

**Source plan:** `D:\GraphRAG\PLAN.md`, Phase 1
**Date:** 2026-10-19 · **Runner:** `./.venv/Scripts/python.exe -m pytest` (pytest 9.1.1, Python 3.13.0)

## User journeys

1. As a user, I want to upload a PDF, TXT, or MD file, so its text enters the system.
2. As a user, I want re-uploading the same file to be recognised as the same document, so I don't get duplicates.
3. As a user, I want a clear message when my file can't be read, so I know whether to convert it.
4. As the system, I need stable chunk ids, because every future citation and graph edge points at one.

## Task report

### Task 1 — `app/ingest/loaders.py`

Dispatches on file extension; PDF via `pypdf`, TXT/MD as UTF-8 with replacement. `doc_id` is the
sha256 prefix of the file bytes, so ingestion is idempotent and filename-independent.

- **RED:** `pytest tests/test_loaders.py` → `ModuleNotFoundError: No module named 'app.ingest.loaders'`
- **GREEN:** `pytest tests/test_loaders.py` → 21 passed
- **Guarantees:** every page of a 10-page PDF is read; page breaks become paragraph breaks; a
  text-free PDF is rejected with a message naming scanning; unsupported extensions list what is
  supported; empty and whitespace-only files are rejected; undecodable bytes degrade rather than
  fail; CRLF is normalised; unicode and emoji survive.

### Task 2 — `app/ingest/chunker.py`

Paragraph-aware packing with word-boundary splits and a carried overlap.

- **RED:** `pytest tests/test_chunker.py` → `ModuleNotFoundError: No module named 'app.ingest.chunker'`
- **GREEN (after one defect, below):** `pytest tests/test_chunker.py` → 28 passed
- **Guarantees:** `chunk_id == f"{doc_id}:{index}"`, sequential from 0 and unique; chunking is
  deterministic; no chunk exceeds `CHUNK_TARGET_CHARS`; no chunk is whitespace-only; words are never
  split; every source word appears in some chunk; adjacent chunks overlap when a paragraph is split,
  within `CHUNK_OVERLAP_CHARS`; small paragraphs stay together with their blank lines.

**Defect found by the tests, not by inspection.** The first implementation split over-long
paragraphs to the full `CHUNK_TARGET_CHARS`. That left no room for the carried tail, so the packer
silently dropped the overlap to stay inside the size bound.
`test_consecutive_chunks_overlap_when_a_paragraph_is_split` reported `['', '', '', '']` for four
adjacent pairs. Fixed by capping pieces at `_BODY_LIMIT = CHUNK_TARGET_CHARS - CHUNK_OVERLAP_CHARS - 1`,
so overlap + separator + body always fits.

### Task 3 — `app/schemas.py`

Frozen `LoadedDocument`, `Chunk`, `DocumentSummary`. Frozen because they cross module boundaries and
the project rule is that callers never mutate what they are handed.

### Task 4 — `POST /api/documents`

- **RED:** `pytest tests/test_documents_api.py` → 11 failed (route absent)
- **GREEN:** `pytest tests/test_documents_api.py` → 13 passed
- **Guarantees:** returns `{doc_id, filename, chunk_count}` in the `{ok, data, error}` envelope;
  re-upload under a different name returns the same `doc_id`; failures map to 415 / 422 / 413 with a
  machine code and a hint; errors use the same envelope shape as successes.

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | Same bytes always give the same 16-hex doc_id | `test_loaders.py::test_doc_id_is_deterministic_for_the_same_bytes` | unit | PASS |
| 2 | doc_id comes from content, not filename | `test_loaders.py::test_doc_id_comes_from_content_not_filename` | unit | PASS |
| 3 | All 10 pages of a PDF are extracted | `test_loaders.py::test_loads_every_page_of_a_ten_page_pdf` | unit | PASS |
| 4 | PDF page breaks become paragraph breaks | `test_loaders.py::test_pdf_pages_are_separated_by_a_blank_line` | unit | PASS |
| 5 | A scanned PDF says it may be a scan | `test_loaders.py::test_a_pdf_with_no_extractable_text_says_it_may_be_a_scan` | unit | PASS |
| 6 | A corrupt PDF is a typed error, not a traceback | `test_loaders.py::test_a_corrupt_pdf_is_reported_as_a_typed_error` | unit | PASS |
| 7 | Unsupported extensions list the supported ones | `test_loaders.py::test_an_unsupported_extension_lists_what_is_supported` | unit | PASS |
| 8 | Undecodable bytes degrade instead of failing | `test_loaders.py::test_invalid_utf8_is_replaced_rather_than_crashing` | unit | PASS |
| 9 | Unicode and emoji survive load and chunk | `test_loaders.py::test_unicode_and_emoji_survive_loading`, `test_chunker.py::test_unicode_and_emoji_survive_chunking` | unit | PASS |
| 10 | chunk_id is `doc_id:index`, sequential and unique | `test_chunker.py::test_chunk_ids_are_doc_id_colon_index`, `::test_chunk_ids_are_unique` | unit | PASS |
| 11 | Chunking is deterministic | `test_chunker.py::test_chunking_is_deterministic` | unit | PASS |
| 12 | No chunk exceeds the target size | `test_chunker.py::test_no_chunk_exceeds_the_target_size` | unit | PASS |
| 13 | Words are never split across chunks | `test_chunker.py::test_words_are_never_split_across_chunks` | unit | PASS |
| 14 | No source word is dropped | `test_chunker.py::test_every_source_word_appears_in_some_chunk` | unit | PASS |
| 15 | Adjacent chunks overlap, within budget | `test_chunker.py::test_consecutive_chunks_overlap_when_a_paragraph_is_split`, `::test_overlap_stays_within_its_budget` | unit | PASS |
| 16 | Upload returns doc_id, filename, chunk_count | `test_documents_api.py::test_uploading_a_text_file_returns_the_document_summary` | integration | PASS |
| 17 | Re-upload under a new name is the same document | `test_documents_api.py::test_re_uploading_the_same_file_returns_the_same_doc_id` | integration | PASS |
| 18 | Failures map to 415 / 422 / 413 with a hint | `test_documents_api.py::test_an_unsupported_file_type_is_rejected_with_415` and siblings | integration | PASS |

## Coverage

```
pytest --cov=app --cov-report=term
app\ingest\chunker.py   78   0   100%
app\ingest\loaders.py   52   0   100%
app\schemas.py          17   0   100%
app\main.py             46   2    96%
TOTAL                  849 157    82%
```

140 tests, all passing (85 before Phase 1). Phase 1 modules are at 100%; the project total of 82%
clears the 80% threshold.

## Known gaps and notes

- **Two tests were wrong, not the code.** `test_tail_words_never_returns_a_partial_word` asserted
  that `_tail_words("a elephant", 4)` returns `"a"`; the helper correctly returns `""`, because a
  leading word is not a tail. And `test_a_tiny_trailing_remainder_is_merged_into_the_previous_chunk`
  passed **for the wrong reason** — every chunk already cleared the floor, so the merge path never
  ran. It is now an invariant-only test, with `_merge_tiny_tail`'s branches covered directly.
- **`app/main.py` lines 38–41 uncovered:** the lifespan warning when model ids are not yet
  discovered. Reaching it requires booting with blank model ids, which the test fixtures always set.
- **Scope pulled forward:** the 20 MB upload guard belongs to Phase 9 in PLAN.md. It was included
  because `MAX_UPLOAD_BYTES` and `UploadTooLarge` already existed unused, and an unbounded upload
  endpoint is a trivially avoidable footgun. Filename sanitization remains Phase 9.
- **Phase 1 persists nothing.** No Chroma, no graph, no file saved to `data/uploads/`. Those arrive
  in Phases 2 and 4.

## Merge evidence

Checkpoint commits on `master`, in order:

| Commit | Stage |
|---|---|
| `da6927d` | `test:` RED — reproducers failing on absent modules and route |
| `727bddc` | `feat:` GREEN — loaders, chunker, schemas, endpoint; 133 passing |
| (this commit) | `test:` — honest merge tests, defensive branches covered; 140 passing |
