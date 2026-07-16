# Changelog

## [0.4.1] - 2026-07-16

### Fixed

- **Scoped embed runs could poison unrelated collections.** When
  `embed -c <collection>` hit a zero-progress worker batch, the skip
  fallback selected the first pending document **globally** instead of
  within the scoped collection — and since both queries were unordered,
  the skip often didn't even target the document the worker was stuck on.
  In the worst case (a crashing worker) the loop silently marked thousands
  of out-of-scope documents as "embedded" with dummy vectors, one per
  iteration. `SkipNextUnembedded` now takes the collection scope, both it
  and `UnembeddedHashes` share a deterministic `ORDER BY hash`, every skip
  is logged with the document hash, and the orchestrator aborts after 10
  consecutive zero-progress batches instead of churning for hours.

  **If you ran v0.4.0 `embed -c` and saw inflated embedded counts**, repair
  the index with:
  `sqlite3 <index> "DELETE FROM content_vectors WHERE text='[skipped]' AND length(vec)=4; UPDATE content_vectors SET vec=NULL WHERE length(vec)=4;"`
  then re-run `picoqmd embed`.

## [0.4.0] - 2026-07-16

Port of the applicable qmd v2.1.0–v2.6.3 fixes, plus repairs to picoqmd's
own retrieval path found while porting. First release with a test suite
(`go test ./...`).

### Fixed

- **FTS5 queries with dots, hyphens, or operator words no longer error.**
  `toFTS5Query` previously emitted raw barewords (`v3.9.7*`), which are FTS5
  syntax errors for any token containing `.`/`-`/`:` and turned words like
  AND/OR/NOT into operators. Every term is now emitted as a quoted string
  (`"v3.9.7"*`) so the tokenizer treats punctuation as separators — version
  strings, hyphenated words, and operator words all match as phrases.
  Punctuation-only queries return no results instead of erroring.
  (qmd #463, #563)
- **`get`/`multi_get` now return document content.** Both previously
  returned only metadata — `Content` was never populated and `multi_get`'s
  `maxBytes` was parsed but unused. Content is read from disk (the index
  stores no full text; FTS is contentless by design).
- **BM25 results now carry real snippets with line citations.** The FTS
  table is contentless, so FTS5's `snippet()` always returned `""` — BM25
  results (and therefore reranker input, which is title+snippet) had no
  body text at all. Snippets are now extracted from the on-disk file for
  the returned page, highlighted `>>>term<<<`, with `Line` set to the first
  match (`path:L<n>` in output). `extractSnippet` now returns the match
  offset rather than the window start, improving vector-result citations
  too.
- **Partially-embedded documents now resume.** A document with a single
  embedded chunk counted as fully embedded and was never revisited —
  `status` lied about pending counts after any interrupted embed run. A
  document is now pending unless every chunk has a current vector.
  `SkipNextUnembedded` handles both pending shapes (no chunks / missing
  vectors) so the skip loop still terminates. (qmd #637)
- **SQLite busy timeout raised to 120s** (from zombiezen's 10s default),
  configurable via `PICOQMD_SQLITE_BUSY_TIMEOUT` (milliseconds, `0` =
  fail-fast). A scheduled sync racing the MCP daemon or an embed worker now
  queues instead of failing with `database is locked`. (qmd #686)

### Added

- **Line-range retrieval.** `get` accepts `:from` and `:from:count` ref
  suffixes (`release.md:120:40`, `#abc123:8:2`), plus `--from`/`--lines`
  flags (CLI) and `fromLine`/`maxLines` params (MCP). Out-of-range values
  clamp. Output is line-numbered by default (`--no-line-numbers` /
  `lineNumbers:false` to disable) with a `qmd://collection/path #docid
  (lines X-Y of Z)` header; `--full-path`/`fullPath` swaps the header to
  the on-disk path for piping into file tools. Refs also accept
  `qmd://collection/path` URIs. (qmd v2.5.3)
- **Embedding fingerprints.** Each vector is stamped with the embed model +
  chunker version (`fp` column, auto-migrated). Changing either marks
  affected documents pending for re-embed instead of silently searching
  stale vectors; existing vectors are adopted under the current fingerprint
  on first open (one-time, no forced re-embed). `status` shows the active
  fingerprint. (qmd v2.5.0)
- **`multi_get` skip reporting.** Files over `maxBytes` (default raised
  10 KB → 64 KB) or unreadable on disk are listed as `[skipped ...]` lines
  instead of being silently dropped. Duplicate matches across
  comma-separated patterns are deduped. (qmd #701, #702)
- **`--no-rerank` / `noRerank`.** Skips the LLM reranking stage and returns
  RRF-fused order — pairs with the existing `--no-expand`. (qmd #370)
- **`picoqmd embed -c <collection>`** embeds one collection's pending
  documents without re-indexing everything — makes huge collections
  (books-*) opt-in instead of all-or-nothing. (qmd v2.5.0 scoped embed)

### Fixed (post-review, found during rollout)

- **llama.cpp runtime extraction produced 0-byte dylibs.** The release
  tarball ships version chains (`libggml.dylib → libggml.0.9.7.dylib`) as
  symlinks; the extractor wrote symlink entries as regular files via
  `os.Create` + `io.Copy` (no body → empty file), so dlopen always failed.
  This is why **no vector had ever been successfully embedded** by picoqmd
  on this machine (all 9,390 chunk rows had NULL vectors). Symlinks are now
  recreated as symlinks; Linux versioned `.so.X.Y` names are matched too.
- **Chronic engine failure no longer poisons the index.** When the engine
  couldn't initialize at all, the orchestrator's skip loop marked every
  pending document as "skipped" with dummy vectors — silently destroying
  the pending queue. The embed worker now probes the engine at startup and
  exits with a distinct code; the orchestrator aborts with a clear error.

### Notes

- Deliberately not ported from qmd v2.1–v2.6.3: CJK trigram FTS (index
  doubling contradicts the small-footprint goal), tree-sitter AST chunking
  (needs cgo or a WASM runtime), `doctor`/`bench` (covered by `status` and
  `compare.sh`), and all Node/launcher/Metal/npm-packaging fixes (no Node
  runtime here).
- picoqmd already stored literal paths and already weighted the original
  query 2× in hybrid RRF, so qmd's #698 and #591 fixes don't apply.

## [0.3.0] - 2026-04-26

Three architectural improvements ported from tobi's qmd v2.0.0. All changes are
additive — empty `--intent` and absent `noExpand` argument reproduce 0.2.2
behavior exactly.

### Added

- **Strong-signal expansion bypass.** The hybrid pipeline now probes BM25
  before invoking the 1.7B query-expansion model. When the top BM25 result is
  at least 2× the score at rank 3 (scale-free, so it works against both raw
  FTS5 BM25 and RRF-normalized scores), the LLM expansion stage is skipped
  entirely. Saves 150–400ms per query on obvious literal matches without
  giving anything up — ambiguous/conceptual queries still flow through the
  full expansion path. The probe's BM25 results are reused as the lex-side
  rank list, so this is also one fewer BM25 call on every hybrid search.
- **`--no-expand` CLI flag** and corresponding `noExpand` argument on the
  `deep_search` MCP tool. Forces the bypass on regardless of probe outcome —
  useful for benchmarking and reproducing pre-bypass behavior.
- **BM25-aware snippets for vector results.** Vector and hybrid results no
  longer ship with empty `Snippet` strings. The winning chunk's text is run
  through a new `extractSnippet` helper that picks a window around the first
  query-term hit and highlights all term occurrences with `>>>...<<<` (the
  same format FTS5 already emits for BM25 paths). New `Line` field on
  `SearchResult` carries a 1-based line number computed from the winning
  chunk's byte position; the text/csv/files/md output formats now print
  `path:L<line>` so editors can jump directly to the citation.
- **`intent` parameter end-to-end.** Optional disambiguation hint threaded
  through query expansion, reranking, and snippet selection. CLI: `--intent`.
  MCP: optional `intent` property on `search`, `vector_search`, `deep_search`,
  and `research`. Empty intent reproduces prior behavior. The expansion
  prompt includes intent when present so generated alternatives stay in the
  user's frame; the rerank prompt includes intent so the cross-encoder scores
  against the intended meaning; snippets union query and intent terms when
  picking the highlight window.

### Changed

- **Public Store API:** `SearchVector` and `SearchVectorInCollection` now take
  the original text query as the first argument. Pass `""` if you only have
  the vector and don't need query-aware snippets. The text is used solely for
  snippet term-extraction; the embedding still drives ranking.
- **Embedder interface:** `ExpandQuery(query, intent)` and
  `Rerank(query, intent, candidates)` now accept an optional intent hint.
  Existing callers that pass `""` get unchanged behavior.
- **Searcher interface:** `Search(ctx, query, intent, collection, limit)`
  gains an intent argument.

### Why these are worth porting

`qmd` v2.0.0 introduced the `intent` parameter, a strong-signal bypass, and
BM25-aware snippets independently of each other; together they remove the
biggest perceived-latency cost on common queries (the LLM expansion call) and
make vector results actually citeable. The SDK split that qmd also did is
deferred here — picoqmd is currently consumed only via CLI and MCP, so
splitting `package main` into `pkg/store` and friends would be speculative
work without a real second consumer.

### Operator notes

- The MCP tool schema gained two new optional properties (`intent` everywhere
  hybrid/vector/research run, `noExpand` on `deep_search`). Existing clients
  that don't send these continue to work.
- The `Line` field on `SearchResult` is `omitempty` in JSON and `0` means
  "unknown" — older consumers that ignore unknown fields are unaffected.
- The chunks table schema is unchanged; no re-indexing required.

## [0.2.2] - 2026-04-26

Two related fixes for unattended operation under launchd, cron, or any
non-interactive harness.

### Fixes

- **Auto-quiet under non-TTY stdout.** When `os.Stdout` is not a terminal,
  picoqmd now suppresses per-document progress lines by default. Previously
  every `embed-worker` invocation printed `[worker] N/M docs` per ten chunks
  plus restart notices and skip notices, all of which were captured verbatim
  by launchd into `StandardOutPath`. A 30-minute cron schedule could grow the
  captured log to 60+ GB given enough document churn or a misbehaving
  document. New `--quiet` / `--verbose` flags override the auto-detect.

- **Worker subprocess no longer inherits parent stdout/stderr in quiet
  mode.** `embedAll` previously wired the worker's pipes to the parent's,
  which meant per-chunk progress kept flowing to the launchd log even after
  the parent went quiet. The worker now writes to `io.Discard` when running
  under `--quiet`, and the parent passes `--quiet` through to its child so
  the worker self-suppresses too.

- **Worker error notices route to `log.Printf` instead of direct stdout.**
  Chronic worker failures previously emitted one direct stdout line per
  retry. They now go through the standard logger, which respects `log`
  package settings and keeps the launchd-captured stdout file bounded even
  when something is genuinely broken.

- **Model-download progress bar is suppressed under non-TTY stdout.** The
  carriage-return based bar in `engine.go` accumulates as garbage when the
  output isn't a real terminal. It now only renders when progress is
  enabled.

### Why

`disabled.<label>.plist` does not actually unload an already-loaded launchd
service — it only prevents auto-load on next boot. Combined with picoqmd's
unbounded stdout output, a "disabled" picoqmd-refresh service ran every 30
minutes for weeks and grew its captured log to 68 GB on one user's system.
Fixing the unbounded output means picoqmd is safe to run under any
unattended scheduler regardless of what the operator did with the plist.

### Operator notes

- If you previously installed picoqmd under launchd / cron and your
  captured-stdout log file is large, truncate it: `: > /path/to/picoqmd.log`.
- If a picoqmd service has accidentally been left running under a renamed
  `disabled.*.plist`, evict it from current launchd memory with
  `launchctl remove com.example.picoqmd-refresh` (the file rename alone is
  not enough).
- Interactive use is unchanged: when run from a real terminal, picoqmd
  prints progress as before.

---

## [0.2.1] - 2026-03-02

- Collection-size normalization via per-collection RRF.
- BM25 column weights: title boosted 5× over content.
- Document length normalization via post-FTS5 b-correction (b=0.55).
- Fix infinite loop in `embedAll` when documents produce zero chunks.

## [0.2.0] - earlier

- Composite `research` tool: BM25 + vector + RRF in one call.
- `maxChars` server-side token budget on search and retrieval.
- Stale document flagging.

## [0.1.0]

- Initial release: BM25 + optional vector search, MCP server, Pure Go SQLite.
