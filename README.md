# PicoQMD

**The search MCP server that fits where your agent does.** Single binary, ~15MB, runs on a Raspberry Pi Zero.

Give any AI agent — [OpenClaw](https://github.com/openinterface/openclaw), [PicoClaw](https://github.com/sipeed/picoclaw), [MiniClaw](https://github.com/mattdef/miniclaw), [Claude Code](https://docs.anthropic.com/en/docs/claude-code), or your own — instant local search over any text files: code, docs, configs, notes. No cloud. No Node.js. No Python. Just a Go binary and SQLite.

## Running under launchd / cron / systemd

picoqmd auto-detects when stdout is not a terminal (launchd, cron, pipes,
`tee >file`) and suppresses per-document progress output to keep
captured-stdout log files bounded. Force the behavior with `--quiet` or
override it with `--verbose`. See [CHANGELOG.md](CHANGELOG.md) for v0.2.2
details.

If you previously ran picoqmd under launchd before v0.2.2, your
StandardOutPath log file may be very large — safely truncate it with
`: > /path/to/your/picoqmd.log`.

## Why PicoQMD?

Most search tools assume beefy hardware. PicoQMD is built for the other end of the spectrum:

- **~11MB binary** — smaller than most npm installs
- **Minimal RAM** — BM25 mode needs almost nothing; runs alongside PicoClaw on $10 hardware
- **Zero dependencies** — no runtime, no interpreters, no containers
- **MCP native** — stdio and HTTP transports, works with any MCP-compatible agent
- **Pure Go** — cross-compiles to ARM32, ARM64, RISC-V, x86 in one command
- **Scales up** — add semantic vector search and hybrid re-ranking when your hardware allows
- **Graceful degradation** — without models, vector/hybrid/research tools are hidden from the agent; BM25, get, and observations still work

## What's New in v0.2.1

### Collection-Size Normalization

Unscoped searches now use per-collection Reciprocal Rank Fusion so small collections (4 docs) get fair representation against large ones (800+ docs). Each collection is searched independently and results are interleaved by rank, not raw count. When you pass the `collection` parameter, searches are scoped directly to that collection.

### BM25 Column Weights

Title matches are now boosted 5x over content matches via FTS5 `bm25()` column weights: `title=5.0, content=1.0, docid=0.0`. A document whose title contains your query term now reliably outranks one that merely mentions it in the body.

### Document Length Normalization (b-correction)

FTS5 hardcodes BM25's `b` parameter at 0.75, which over-penalizes long documents. PicoQMD now applies a post-FTS5 correction that shifts `b` to 0.55 — reducing the length penalty so comprehensive source files and detailed docs score fairly against short notes. Document lengths are tracked during indexing (`doc_len` column).

---

### v0.2.0

### Composite `research` Tool

One call replaces two. Runs BM25 + vector search in parallel, deduplicates results via Reciprocal Rank Fusion (K=60), and merges within a token budget — all server-side before results hit the agent's context window.

```
research({ query: "firebase auth flow", limit: 5, maxChars: 5000 })
```

Benchmark (3,145 docs indexed):

| Approach | Time | Response Size |
|----------|------|---------------|
| search + vector_search (2 calls) | 1,585ms | 2,766 chars |
| research (1 call) | 475ms | 1,351 chars |
| **Savings** | **70% faster** | **51% fewer chars** |

### `maxChars` — Server-Side Token Budget

All search and retrieval tools now accept `maxChars` to truncate responses before they enter the context window. No more dumping entire files into agent context.

```
get({ ref: "MEMORY.md", maxChars: 500 })
search({ query: "OSHA compliance", limit: 10, maxChars: 2000 })
```

### `note` — Observation-as-Sidecar

Save insights in the same call as search — no separate tool call needed. Notes are linked to the top result's document ID and content hash.

```
search({ query: "severity colors", note: "defined in shared module SeverityLevel.kt" })
```

Observations are persisted to `~/.config/picoqmd/observations.json` and survive across sessions.

### Stale Observation Flagging

When a document's content changes after an observation was saved, the `research` tool automatically flags it `[STALE]` in results. The `status` tool reports the count of stale observations. No manual cleanup needed — stale context is surfaced, not silently persisted.

### Graceful Degradation

Without embedding models, the MCP server hides vector-dependent tools entirely:

| Mode | Tools Exposed |
|------|--------------|
| **With models** | search, vector_search, deep_search, research, get, multi_get, status |
| **Without models** | search, get, multi_get, status |

The agent never sees tools it can't use. Instructions update to explain BM25-only mode. All new features (`maxChars`, `note`, stale flagging) work in both modes.

## Quick Start

```sh
# Install
go install github.com/altSeneca/picoqmd@latest

# Index markdown docs (default)
picoqmd add ~/docs --no-embed

# Index a codebase — Go, Python, TypeScript, and markdown
picoqmd add ~/myproject --glob "**/*.{go,py,ts,md}" --no-embed

# Search — prefix matching built in
picoqmd search "kubernetes deployment"
picoqmd search "deploy"          # matches "deployment", "deployed", "deploying"
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
| `search` | BM25 keyword search via SQLite FTS5 with prefix matching | No |
| `vector_search` | Semantic similarity using embeddings | Yes |
| `deep_search` | Query expansion + fan-out + RRF + re-ranking | Yes |
| `research` | Composite: BM25 + vector in parallel, deduplicated via RRF | Yes |
| `get` | Retrieve single document by path or docid | No |
| `multi_get` | Batch retrieve by glob pattern or comma-separated list | No |
| `status` | Index health, collection inventory, stale observation count | No |

**Common parameters** across search tools:

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query (required) |
| `limit` | int | Max results, default 10 |
| `collection` | string | Filter to a specific collection |
| `minScore` | float | Minimum relevance score 0-1 |
| `maxChars` | int | Truncate response to this many characters |
| `note` | string | Save an observation linked to the top result |

## Two Modes

### BM25 Only — For Edge and Constrained Devices

```sh
picoqmd add ~/notes --no-embed
picoqmd add ~/src --glob "**/*.{go,py,rs,ts,js}" --no-embed
picoqmd search "meeting notes"
```

No models, no llama.cpp, no downloads. Just Go + SQLite FTS5 with prefix matching. This is the mode for PicoClaw-class devices where every megabyte counts. The binary is ~11MB, runtime memory is minimal, and search is instant.

### Vector + Hybrid — For Capable Hardware

```sh
picoqmd add ~/notes                  # downloads embedding model (~300MB)
picoqmd model download               # all 3 models for full hybrid pipeline
picoqmd "semantic search query"      # auto-selects best pipeline
```

When you have the RAM, unlock semantic search with query expansion, RRF fusion, and cross-encoder re-ranking — all still local, all still offline.

| Model | Size | Purpose |
|-------|------|---------|
| [embeddinggemma-300M](https://huggingface.co/tobi/embeddinggemma-300M-GGUF) | ~300MB | Document & query embeddings |
| [qwen3-reranker-0.6b](https://huggingface.co/tobi/qwen3-reranker-0.6b-GGUF) | ~600MB | Cross-encoder re-ranking |
| [qmd-query-expansion-1.7B](https://huggingface.co/tobi/qmd-query-expansion-GGUF) | ~1GB | Query expansion |

## Search Modes

| Mode | Command | What it does |
|------|---------|--------------|
| BM25 | `picoqmd search "query"` | Instant keyword search via SQLite FTS5 with prefix matching |
| Vector | `picoqmd vsearch "query"` | Semantic similarity using embeddings |
| Hybrid | `picoqmd query "query"` | Expansion + fan-out + RRF + re-ranking |
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

- **PicoClaw / MiniClaw search tool** — give your $10 AI agent fast local search over project docs, wikis, and codebases
- **OpenClaw on Raspberry Pi** — add document search to your self-hosted AI assistant without eating its RAM budget
- **Claude Code MCP server** — fast, local search over large codebases without spinning up Elasticsearch
- **Edge AI knowledge base** — deploy searchable documentation to field devices, kiosks, or air-gapped environments
- **Offline dev search** — index API docs, READMEs, and notes for airplane-mode development
- **Token-efficient MCP pipelines** — use `research` to cut context window usage by ~50% vs separate search calls

## Acknowledgments

PicoQMD is a Go reimplementation of [QMD](https://github.com/tobi/qmd) by [@tobi](https://github.com/tobi), which provides the architecture, hybrid search pipeline, models, and design. Built with [yzma](https://github.com/hybridgroup/yzma) (pure-Go llama.cpp bindings) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

See [GUIDE.md](GUIDE.md) for the full user guide, output formats, and configuration.

## License

[MIT](LICENSE)
