# Changelog

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
