# PicoQMD — a lightweight QMD alternative for low-resource computers

**A fully local search engine and MCP server in a single ~15MB Go binary.** PicoQMD is a from-scratch Go reimplementation of [tobi's QMD](https://github.com/tobi/qmd) built for machines where QMD's Node.js/Bun stack is too heavy: Raspberry Pi (including Pi Zero), old laptops, small VPSes, air-gapped boxes, and dev machines that just don't want another Node runtime.

Same search pipeline as QMD — SQLite FTS5 BM25, semantic vector search, hybrid query expansion + Reciprocal Rank Fusion + cross-encoder reranking, the same GGUF models — with no Node.js, no Bun, no Python, no npm install, no native-module ABI headaches. One static binary and a SQLite file.

Give any AI agent — [Claude Code](https://docs.anthropic.com/en/docs/claude-code), [OpenClaw](https://github.com/openinterface/openclaw), [PicoClaw](https://github.com/sipeed/picoclaw), [MiniClaw](https://github.com/mattdef/miniclaw), or your own — instant local search over code, docs, configs, and notes. No cloud, no telemetry, works offline.

## PicoQMD vs QMD

If you're looking for **"QMD but for a lower-spec computer"**, this is the trade-off table:

|  | QMD | PicoQMD |
|--|-----|---------|
| Install | Node.js/Bun + npm package (native modules: better-sqlite3, sqlite-vec, node-llama-cpp) | one static Go binary (~15MB) |
| Runtime | Node/Bun VM | none — measured **~29MB peak RSS** for a BM25 search over a 10,800-doc index |
| BM25 keyword search | SQLite FTS5 | SQLite FTS5 (pure-Go driver, contentless index — documents are not duplicated into the DB) |
| Semantic vector search | sqlite-vec + node-llama-cpp | pure-Go brute-force cosine + llama.cpp via FFI (same EmbeddingGemma model) |
| Hybrid pipeline | query expansion → fan-out → RRF → rerank | same design, same GGUF models |
| MCP server | stdio + HTTP | stdio + HTTP |
| Minimum hardware for keyword search | needs Node-capable box | Raspberry Pi Zero / ARM32 / RISC-V |
| Vector/hybrid search | yes | yes, on arm64/amd64 (Linux, macOS) |
| Line-numbered `get` with `:from:count` ranges | yes | yes (v0.4.0) |
| Embedding fingerprints (stale-vector detection) | yes | yes (v0.4.0) |
| AST/tree-sitter code chunking | yes | not yet ([roadmap](ROADMAP.md)) |
| CJK trigram search | yes | no (kept out deliberately — doubles index size) |

PicoQMD is not a fork — it's an independent Go implementation that tracks QMD's retrieval design and ports its fixes (v0.4.0 covers the applicable QMD v2.1–v2.6.3 changes). If you have a beefy dev machine and live in the Node ecosystem, use QMD. If you want the same local search quality in a fraction of the footprint — or on hardware QMD can't run on at all — use PicoQMD.

## Why PicoQMD?

Most search tools assume beefy hardware. PicoQMD is built for the other end of the spectrum:

- **~15MB binary** (~11MB with `-ldflags="-s -w"`) — smaller than most npm installs
- **Minimal RAM** — BM25 mode runs in tens of MB; fits alongside an agent on $10 hardware
- **Zero dependencies** — no runtime, no interpreters, no containers, no C toolchain (pure-Go SQLite)
- **MCP native** — stdio and HTTP transports, works with any MCP-compatible agent
- **Cross-compiles anywhere Go does** — ARM32, ARM64, RISC-V, x86 in one command
- **Scales up** — add semantic vector search and hybrid re-ranking when your hardware allows
- **Graceful degradation** — without models, vector/hybrid tools are hidden from the agent; BM25, get, and observations still work
- **Safe under launchd/cron/systemd** — auto-quiets progress output when stdout is not a TTY, so captured logs stay bounded

## What's New in v0.4.0

Ports of the applicable QMD v2.1.0–v2.6.3 improvements, plus fixes to PicoQMD's own retrieval path (full details in [CHANGELOG.md](CHANGELOG.md)):

- **Robust FTS5 queries** — version strings (`v3.9.7`), hyphenated terms (`real-time`), and operator words (AND/OR/NOT) can no longer produce FTS5 syntax errors; every term is emitted as a quoted phrase with prefix matching.
- **Real document retrieval** — `get`/`multi_get` return content from disk with line-numbered output, `qmd://` + `#docid` headers, and line-range refs: `get notes.md:120:40` reads 40 lines from line 120. `--full-path` swaps in the on-disk path for piping into editors and file tools.
- **BM25 snippets with line citations** — snippets are extracted from the source file (`>>>term<<<` highlighting, `path:L<n>`), which also feeds the reranker real text instead of bare titles.
- **Embedding fingerprints** — vectors are stamped with the model + chunker identity; changing either marks documents pending for re-embed instead of silently searching stale vectors.
- **Honest embed tracking** — a document only counts as embedded when *every* chunk has a current vector; interrupted embed runs resume instead of being forgotten.
- **Concurrency-safe SQLite** — 120s busy timeout (override: `PICOQMD_SQLITE_BUSY_TIMEOUT`, ms) so a scheduled sync racing the MCP daemon queues instead of throwing `database is locked`.
- **Scoped embedding** — `picoqmd embed -c <collection>` embeds one collection without re-indexing, so huge collections are opt-in.
- **`--no-rerank`** — skip the cross-encoder for faster hybrid results on constrained hardware.
- **First test suite** — `go test ./...` covers query sanitization, retrieval, and embed tracking against a real store.

## Quick Start

```sh
# Install (or grab a prebuilt binary from Releases)
go install github.com/altSeneca/picoqmd@latest

# Index markdown docs (default)
picoqmd add ~/docs --no-embed

# Index a codebase — Go, Python, TypeScript, and markdown
picoqmd add ~/myproject --glob "**/*.{go,py,ts,md}" --no-embed

# Search — prefix matching built in
picoqmd search "kubernetes deployment"
picoqmd search "deploy"          # matches "deployment", "deployed", "deploying"

# Retrieve with line ranges
picoqmd get notes.md:120:40      # 40 lines starting at line 120
```

## MCP Server

PicoQMD is an MCP server first. Point your agent at it and get `search`, `get`, `multi_get`, `status` — plus `vector_search`, `deep_search`, and `research` when models are available.

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

**Common parameters** across search tools:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query (required) |
| `intent` | string | Optional disambiguation hint threaded through expansion, reranking, and snippets |
| `limit` | int | Max results, default 10 |
| `collection` | string | Filter to a specific collection |
| `minScore` | float | Minimum relevance score 0–1 |
| `maxChars` | int | Truncate response to this many characters (server-side token budget) |
| `note` | string | Save an observation linked to the top result |

**`get` / `multi_get` parameters:** `fromLine`, `maxLines`, `lineNumbers` (default true), `fullPath`, `maxBytes` (multi_get skip threshold, default 64KB).

## Two Modes

### BM25 Only — For Edge and Constrained Devices

```sh
picoqmd add ~/notes --no-embed
picoqmd add ~/src --glob "**/*.{go,py,rs,ts,js}" --no-embed
picoqmd search "meeting notes"
```

No models, no llama.cpp, no downloads. Just Go + SQLite FTS5 with prefix matching. This is the mode for Pi-Zero-class devices where every megabyte counts: keyword search over ~10,000 documents runs in under 30MB of RAM.

### Vector + Hybrid — For Capable Hardware

```sh
picoqmd add ~/notes                    # downloads embedding model (~300MB)
picoqmd model download embedding       # or: reranker, expansion
picoqmd "semantic search query"        # auto-selects best pipeline
picoqmd embed -c big-collection        # embed one collection at a time
```

When you have the RAM, unlock semantic search with query expansion, RRF fusion, and cross-encoder re-ranking — all still local, all still offline. Same models QMD uses:

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

## Platform Support

| Platform | BM25 | Vector/Hybrid | Binary |
|----------|------|---------------|--------|
| Linux arm32 (Pi Zero, Pi 1) | yes | — | ~9MB |
| Linux riscv64 | yes | — | ~9MB |
| Linux arm64 (Pi 3/4/5, SBCs) | yes | yes | ~11MB |
| Linux amd64 | yes | yes | ~11MB |
| macOS arm64 (Apple Silicon) | yes | yes | ~11MB |
| macOS amd64 (Intel) | yes | yes | ~11MB |

Cross-compile for your target in one line:

```sh
GOOS=linux GOARCH=arm GOARM=7 go build -ldflags="-s -w" -o picoqmd .
```

## Export / Import — Index Once, Search Anywhere

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

Index any text file — not just markdown. Use glob patterns with brace expansion:

```sh
picoqmd add . --glob "**/*.md"                        # markdown only (default)
picoqmd add . --glob "**/*.{go,py,ts,js,rs,md}"       # code + docs
picoqmd add . --glob "**/*.{yaml,yml,json,toml}"      # config files
```

PicoQMD automatically skips binary files, files over 1MB, and common noise directories (`.git`, `node_modules`, `vendor`, `__pycache__`, `build`, `dist`, `target`, etc.).

## Use Cases

- **QMD alternative on low-spec hardware** — same local hybrid search without the Node.js runtime, on machines from a Pi Zero up
- **Claude Code MCP server** — fast, token-efficient search over large codebases without spinning up Elasticsearch
- **PicoClaw / MiniClaw search tool** — give your $10 AI agent fast local search over project docs, wikis, and codebases
- **OpenClaw on Raspberry Pi** — add document search to your self-hosted AI assistant without eating its RAM budget
- **Edge AI knowledge base** — deploy searchable documentation to field devices, kiosks, or air-gapped environments
- **Offline dev search** — index API docs, READMEs, and notes for airplane-mode development
- **Token-efficient MCP pipelines** — use `research` to cut context window usage by ~50% vs separate search calls

## Roadmap

See [ROADMAP.md](ROADMAP.md) — next up: Matryoshka 768→256 embedding truncation (3× smaller/faster vectors), chunk-level incremental re-embedding, recency-aware ranking, binary quantization with two-phase rescoring for very large corpora, and tree-sitter AST chunking for code.

## Acknowledgments

PicoQMD is a Go reimplementation of [QMD](https://github.com/tobi/qmd) by [@tobi](https://github.com/tobi), which provides the architecture, hybrid search pipeline, models, and design. Built with [yzma](https://github.com/hybridgroup/yzma) (pure-Go llama.cpp bindings) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

See [GUIDE.md](GUIDE.md) for the full user guide, output formats, and configuration.

## License

[MIT](LICENSE)
