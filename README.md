# PicoQMD: a lightweight QMD alternative for low-resource computers

**A fully local search engine and MCP server in a single ~16MB Go binary (~11MB stripped).** PicoQMD is a from-scratch Go reimplementation of [tobi's QMD](https://github.com/tobi/qmd) built for machines where QMD's Node.js/Bun stack is too heavy: Raspberry Pi (including Pi Zero), old laptops, small VPSes, air-gapped boxes, and dev machines that just don't want another Node runtime.

It runs the same search pipeline as QMD (SQLite FTS5 BM25, semantic vector search, hybrid query expansion + Reciprocal Rank Fusion + cross-encoder reranking, the same GGUF models) without Node.js, Bun, Python, or native-module ABI headaches. One static binary and a SQLite file.

Point [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenClaw](https://github.com/openinterface/openclaw), [PicoClaw](https://github.com/sipeed/picoclaw), [MiniClaw](https://github.com/mattdef/miniclaw), or any other MCP agent at it for instant local search over code, docs, configs, and notes. No cloud, no telemetry, works offline.

## PicoQMD vs QMD

If you're looking for "QMD but for a lower-spec computer", this is the trade-off table:

|  | QMD | PicoQMD |
|--|-----|---------|
| Install | Node.js/Bun + npm package (native modules: better-sqlite3, sqlite-vec, node-llama-cpp) | one static Go binary (~16MB, ~11MB stripped) |
| Runtime | Node/Bun VM | none (measured **~29MB peak RSS** for a BM25 search over a 10,800-doc index) |
| Index size on disk | ~11.9GB measured on a 10,800-doc corpus | **0.78GB for the same corpus (~15× smaller)** with 256-dim Matryoshka vectors (default since v0.6.0) |
| BM25 keyword search | SQLite FTS5 | SQLite FTS5 (pure-Go driver, contentless index; documents are not duplicated into the DB) |
| Semantic vector search | sqlite-vec + node-llama-cpp | pure-Go brute-force cosine + llama.cpp via FFI (same EmbeddingGemma model) |
| Hybrid pipeline | query expansion → fan-out → RRF → rerank | same design, same GGUF models |
| MCP server | stdio + HTTP | stdio + HTTP |
| Minimum hardware for keyword search | needs Node-capable box | Raspberry Pi Zero / ARM32 / RISC-V |
| Vector/hybrid search | yes | yes, on arm64/amd64 (Linux, macOS) |
| Line-numbered `get` with `:from:count` ranges | yes | yes (v0.4.0) |
| Embedding fingerprints (stale-vector detection) | yes | yes; v0.5.0 hashes the model file's bytes, so a swapped model re-embeds instead of silently serving mismatched vectors |
| Search-quality benchmarking | `qmd bench` | `picoqmd bench` (v0.5.0) |
| Index diagnostics | `qmd doctor` | `picoqmd doctor` + `cleanup` (v0.5.0) |
| AST/tree-sitter code chunking | yes | not yet ([roadmap](ROADMAP.md)) |
| CJK trigram search | yes | no (left out on purpose; it doubles the index) |

PicoQMD is not a fork. It's an independent Go implementation that tracks QMD's retrieval design and ports the fixes that apply (v0.4.0 covered QMD v2.1 to v2.6.3; v0.5.0 covers the v2.8.x round). If you have a fast dev machine and live in the Node ecosystem, use QMD. If you want the same local search quality in a fraction of the footprint, or on hardware QMD can't run at all, use PicoQMD.

## Why PicoQMD?

Most search tools assume fast hardware. PicoQMD is built for everything else:

- **~16MB binary** (~11MB with `-ldflags="-s -w"`), smaller than most npm installs
- **~15× smaller index than QMD.** The 10,800-doc corpus that filled an 11.9GB QMD index fits in 0.78GB, measured on the same machine and models
- **3× smaller, 3× faster vectors.** Embeddings are Matryoshka-truncated to 256 dims by default (EmbeddingGemma is MRL-trained; ~97.6% of full quality). `PICOQMD_EMBED_DIM` overrides
- **Minimal RAM.** BM25 mode runs in tens of MB and fits alongside an agent on $10 hardware
- **Zero dependencies.** No runtime, no interpreters, no containers, no C toolchain (pure-Go SQLite)
- **MCP native.** stdio and HTTP transports, works with any MCP-compatible agent
- **Cross-compiles anywhere Go does:** ARM32, ARM64, RISC-V, x86 in one command
- **Scales up.** Add semantic vector search and hybrid re-ranking when your hardware allows
- **Degrades gracefully.** Without models, vector/hybrid tools are hidden from the agent; BM25, get, and observations still work
- **Safe under launchd/cron/systemd.** Progress output auto-quiets when stdout is not a TTY, so captured logs stay bounded

## What's New in v0.6.x

**v0.6.1: reranker fixed.** The v0.5.0 bench exposed the hybrid pipeline scoring worse than plain vector search; five compounding defects in the rerank stage (KV-cache contamination between candidates, an out-of-distribution prompt instead of Qwen3-Reranker's documented template, title-only candidate text, an RRF/rerank score-scale mismatch, and a too-small decode batch) are fixed. Measured: hybrid hit@10 60% → 100%, MRR 0.50 → 1.00 on the reference fixture, now the strongest pipeline.

**v0.6.0: Matryoshka-256 vectors** (full details in [CHANGELOG.md](CHANGELOG.md)):

- **Embeddings are truncated to 256 dims** (from EmbeddingGemma's 768) and L2-renormalized, at both document and query time. 3× smaller vector storage, 3× faster brute-force scans, ~97.6% of full-dimension quality. Measured on a 10,800-doc corpus: index 853MB → 743MB, vector search 0.5s → 0.3s, bench quality within noise of 768-dim (hit@10 unchanged at 100%, MRR 1.0 → 0.9 on a 5-query fixture).
- **`picoqmd migrate-vectors`** converts an existing index in place in seconds. MRL training means truncate+renormalize produces exactly what embedding at 256 dims would; no re-embed needed. Includes a VACUUM to reclaim the space.
- `PICOQMD_EMBED_DIM` overrides the target (0 = full model dimension). The dimension is part of the embedding fingerprint, so mixed-dimension search is impossible; `doctor` flags mismatches.

<details>
<summary>v0.5.0 changes (QMD v2.8.x ports)</summary>

Ports of the applicable QMD v2.8.x improvements plus hardening from a real-world failure:

- **Model-hash embedding fingerprints.** The fingerprint now includes a sha256 of the model file's bytes, not just its name. A re-downloaded model with the same filename used to invalidate every stored vector silently; now it just triggers a re-embed.
- **`picoqmd doctor`** reports model identity, per-fingerprint vector distribution, and stale or orphaned vectors, and exits non-zero on problems so cron jobs can gate on it.
- **`picoqmd cleanup [--dry-run]`** deletes stale and orphaned vectors so the next `sync` regenerates them.
- **`picoqmd bench <fixture.json>`** measures search quality (hit@k, precision, recall, MRR) per pipeline against a fixture of known-good queries. See `example-bench.json`.
- **Multi-collection scope.** `collection` accepts a comma-separated list everywhere; each collection is searched separately and the results merged, so a big collection can't crowd a small one out of the top-k.
- **Intent-aware expansion.** A dominant keyword match no longer skips LLM query expansion when the caller supplied an `intent` hint.
- New `-c/--collection` flag on `search`, `vsearch`, and `query`.

</details>

<details>
<summary>v0.4.0 changes (QMD v2.1 to v2.6.3 ports)</summary>

- **Robust FTS5 queries.** Version strings (`v3.9.7`), hyphenated terms (`real-time`), and operator words (AND/OR/NOT) can no longer produce FTS5 syntax errors; every term is emitted as a quoted phrase with prefix matching.
- **Real document retrieval.** `get`/`multi_get` return content from disk with line-numbered output, `qmd://` + `#docid` headers, and line-range refs: `get notes.md:120:40` reads 40 lines from line 120. `--full-path` swaps in the on-disk path for piping into editors and file tools.
- **BM25 snippets with line citations.** Snippets are extracted from the source file (`>>>term<<<` highlighting, `path:L<n>`), which also feeds the reranker real text instead of bare titles.
- **Embedding fingerprints.** Vectors are stamped with the model + chunker identity; changing either marks documents pending for re-embed instead of silently searching stale vectors.
- **Honest embed tracking.** A document only counts as embedded when *every* chunk has a current vector; interrupted embed runs resume instead of being forgotten.
- **Concurrency-safe SQLite.** 120s busy timeout (override: `PICOQMD_SQLITE_BUSY_TIMEOUT`, ms) so a scheduled sync racing the MCP daemon queues instead of throwing `database is locked`.
- **Scoped embedding.** `picoqmd embed -c <collection>` embeds one collection without re-indexing, so huge collections are opt-in.
- **`--no-rerank`** skips the cross-encoder for faster hybrid results on constrained hardware.
- **First test suite.** `go test ./...` covers query sanitization, retrieval, and embed tracking against a real store.

</details>

## Quick Start

```sh
# Install (or grab a prebuilt binary from Releases)
go install github.com/altSeneca/picoqmd@latest

# Index markdown docs (default)
picoqmd add ~/docs --no-embed

# Index a codebase: Go, Python, TypeScript, and markdown
picoqmd add ~/myproject --glob "**/*.{go,py,ts,md}" --no-embed

# Search, with prefix matching built in
picoqmd search "kubernetes deployment"
picoqmd search "deploy"          # matches "deployment", "deployed", "deploying"

# Retrieve with line ranges
picoqmd get notes.md:120:40      # 40 lines starting at line 120
```

## MCP Server

PicoQMD is an MCP server first. Point your agent at it and you get `search`, `get`, `multi_get`, and `status`, plus `vector_search`, `deep_search`, and `research` when models are available.

### Claude Code

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "picoqmd": {
    "command": "picoqmd",
    "args": ["mcp"]
  }
}
```

### OpenClaw / PicoClaw / MiniClaw / Any MCP Client

Stdio transport (default):

```sh
picoqmd mcp
```

HTTP transport for networked setups:

```sh
picoqmd mcp --http :8181
```

Any agent that speaks [Model Context Protocol](https://modelcontextprotocol.io/) can connect. The MCP server exposes the same search tools whether you're on a Mac Studio or a Pi Zero.

## MCP Tools Reference

| Tool | Description | Requires Models |
|------|-------------|----------------|
| `search` | BM25 keyword search via SQLite FTS5 with prefix matching, disk-extracted snippets, line citations | No |
| `vector_search` | Semantic similarity using embeddings | Yes |
| `deep_search` | Query expansion + fan-out + RRF + re-ranking (`noExpand`, `noRerank` to trim stages) | Yes |
| `research` | Composite: BM25 + vector in parallel, deduplicated via RRF, one call | Yes |
| `get` | Retrieve a document by path, `#docid`, or `qmd://` URI, with `:from:count` line ranges, line-numbered | No |
| `multi_get` | Batch retrieve by glob or comma-separated list; oversized files reported as skipped, never silently dropped | No |
| `status` | Index health, embedding fingerprint, pending counts, stale observation count | No |

Maintenance runs from the CLI: `doctor` diagnoses the index, `cleanup [--dry-run]` removes stale or orphaned vectors, and `bench <fixture.json>` scores search quality per pipeline.

**Common parameters** across search tools:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query (required) |
| `intent` | string | Optional disambiguation hint threaded through expansion, reranking, and snippets |
| `limit` | int | Max results, default 10 |
| `collection` | string | Collection name, or a comma-separated list; each is searched separately and the results merged |
| `minScore` | float | Minimum relevance score 0 to 1 |
| `maxChars` | int | Truncate response to this many characters (server-side token budget) |
| `note` | string | Save an observation linked to the top result |

**`get` / `multi_get` parameters:** `fromLine`, `maxLines`, `lineNumbers` (default true), `fullPath`, `maxBytes` (multi_get skip threshold, default 64KB).

## Two Modes

### BM25 only, for edge and constrained devices

```sh
picoqmd add ~/notes --no-embed
picoqmd add ~/src --glob "**/*.{go,py,rs,ts,js}" --no-embed
picoqmd search "meeting notes"
```

No models, no llama.cpp, no downloads. Just Go + SQLite FTS5 with prefix matching. This is the mode for Pi-Zero-class devices where every megabyte counts: keyword search over ~10,000 documents runs in under 30MB of RAM.

### Vector + hybrid, for capable hardware

```sh
picoqmd add ~/notes                    # downloads embedding model (~300MB)
picoqmd model download embedding       # or: reranker, expansion
picoqmd "semantic search query"        # auto-selects best pipeline
picoqmd embed -c big-collection        # embed one collection at a time
```

When you have the RAM, add semantic search with query expansion, RRF fusion, and cross-encoder re-ranking. Still local, still offline. Same models QMD uses:

| Model | Size | Purpose |
|-------|------|---------|
| [embeddinggemma-300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | ~300MB | Document & query embeddings |
| [qwen3-reranker-0.6b](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | ~600MB | Cross-encoder re-ranking |
| [qmd-query-expansion-1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf) | ~1GB | Query expansion |

## Search Modes

| Mode | Command | What it does |
|------|---------|--------------|
| BM25 | `picoqmd search "query"` | Instant keyword search via SQLite FTS5 with prefix matching |
| Vector | `picoqmd vsearch "query"` | Semantic similarity using embeddings |
| Hybrid | `picoqmd query "query"` | Expansion + fan-out + RRF + re-ranking (`--no-expand`, `--no-rerank` to trim) |
| Smart | `picoqmd "query"` | Auto-selects best pipeline for available models |

All three accept `-c collection` or `-c colA,colB` to scope the search.

## Platform Support

| Platform | BM25 | Vector/Hybrid | Binary |
|----------|------|---------------|--------|
| Linux arm32 (Pi Zero, Pi 1) | yes | no | ~11MB |
| Linux riscv64 | yes | no | ~10MB |
| Linux arm64 (Pi 3/4/5, SBCs) | yes | yes | ~11MB |
| Linux amd64 | yes | yes | ~11MB |
| macOS arm64 (Apple Silicon) | yes | yes | ~11MB |
| macOS amd64 (Intel) | yes | yes | ~12MB |

Cross-compile for your target in one line:

```sh
GOOS=linux GOARCH=arm GOARM=7 go build -ldflags="-s -w" -o picoqmd .
```

## Export / Import: index once, search anywhere

Build a full index (with embeddings) on a capable machine, then transfer it to a tiny device:

```sh
# On your workstation
picoqmd add ~/docs && picoqmd export -o docs.tar.gz

# On a Pi Zero / edge device
picoqmd import docs.tar.gz
picoqmd search "deployment guide"    # BM25 + precomputed embeddings, no models needed
```

The exported bundle contains the SQLite database with all embeddings baked in. The edge device gets semantic-quality ranking without downloading a single model.

## Remote Search

Don't want to run search on the edge device at all? Forward to a remote instance:

```sh
# Server
picoqmd mcp --http :8181

# Edge device
picoqmd search "query" --remote server:8181
```

## File Type Support

Index any text file, not just markdown. Use glob patterns with brace expansion:

```sh
picoqmd add . --glob "**/*.md"                        # markdown only (default)
picoqmd add . --glob "**/*.{go,py,ts,js,rs,md}"       # code + docs
picoqmd add . --glob "**/*.{yaml,yml,json,toml}"      # config files
```

PicoQMD automatically skips binary files, files over 1MB, and common noise directories (`.git`, `node_modules`, `vendor`, `__pycache__`, `build`, `dist`, `target`, etc.).

## Use Cases

- **QMD alternative on low-spec hardware:** the same local hybrid search without the Node.js runtime, on machines from a Pi Zero up
- **Claude Code MCP server:** fast, token-efficient search over large codebases without spinning up Elasticsearch
- **PicoClaw / MiniClaw search tool:** give your $10 AI agent fast local search over project docs, wikis, and codebases
- **OpenClaw on Raspberry Pi:** add document search to your self-hosted AI assistant without eating its RAM budget
- **Edge AI knowledge base:** deploy searchable documentation to field devices, kiosks, or air-gapped environments
- **Offline dev search:** index API docs, READMEs, and notes for airplane-mode development
- **Token-efficient MCP pipelines:** use `research` to cut context window usage by ~50% vs separate search calls

## Roadmap

See [ROADMAP.md](ROADMAP.md). Matryoshka 768→256 truncation shipped in v0.6.0. Next up: chunk-level incremental re-embedding, recency-aware ranking, binary quantization with two-phase rescoring for very large corpora, and tree-sitter AST chunking for code.

## Acknowledgments

PicoQMD is a Go reimplementation of [QMD](https://github.com/tobi/qmd) by [@tobi](https://github.com/tobi), which provides the architecture, hybrid search pipeline, models, and design. Built with [yzma](https://github.com/hybridgroup/yzma) (pure-Go llama.cpp bindings) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

See [GUIDE.md](GUIDE.md) for the full user guide, output formats, and configuration.

## License

[MIT](LICENSE)
