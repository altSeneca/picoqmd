# PicoQMD User Guide

A fully local search engine for markdown documents. Single binary, zero cloud dependencies, runs anywhere.

PicoQMD is an optimized Go port of [QMD](https://www.npmjs.com/package/@tobilu/qmd) — supporting BM25 full-text search, semantic vector search, and a hybrid pipeline with query expansion, RRF fusion, and LLM re-ranking.

## Quick Start

```
# 1. Add a directory (auto-downloads models, indexes, embeds)
picoqmd add ~/notes

# 2. Search
picoqmd "meeting notes friday"
```

That's it. Two commands from zero to searching.

## Installation

```
# Build from source
go build -ldflags="-s -w" -o picoqmd .

# Move to PATH
sudo mv picoqmd /usr/local/bin/
```

## Core Concepts

**Collections** — A directory of files registered for indexing. Each collection has a name, path, glob pattern, and optional context description.

**Documents** — Individual files within a collection. Each gets a short content-hash ID (e.g., `#a3f2c1`) for quick reference.

**Chunks** — Documents are split into ~900-token pieces at structural boundaries (headings, paragraphs) for embedding.

**Embeddings** — Vector representations of chunks, enabling semantic search. Generated locally using a 300MB GGUF model.

## Search Modes

PicoQMD supports three search modes, each progressively more powerful:

### BM25 Keyword Search

Instant, zero-model search using SQLite FTS5. Best for exact terms, function names, identifiers.

```
picoqmd search "authentication middleware"
picoqmd search "class UserService" --format json
```

### Vector / Semantic Search

Finds related concepts even when exact words differ. Requires the embedding model.

```
picoqmd vsearch "how to handle user login"
picoqmd vsearch "error handling patterns" --limit 20
```

### Hybrid Deep Search

The full pipeline: query expansion, parallel BM25 + vector fan-out, RRF fusion, LLM re-ranking. Best quality, requires all three models.

```
picoqmd query "best practices for database migrations"
```

### Smart Search (Default)

When you run `picoqmd "query"` without a subcommand, it auto-detects which models are available and picks the best pipeline:

| Models Available | Pipeline Used |
|---|---|
| All 3 (embed + rerank + expand) | Full hybrid |
| Embedding only | BM25 + vector with RRF fusion |
| None | BM25 keyword only |

```
picoqmd "meeting notes friday"          # auto-selects best available
picoqmd "API endpoint design" --limit 5 --format md
```

## Commands

### `add` — Add and Index a Directory

```
picoqmd add <path> [flags]
```

This is the main onboarding command. On first run (when no models are present), it prompts you to choose a setup mode:

- **[1] BM25 only** — keyword search, no downloads, runs on any device
- **[2] BM25 + vector** — semantic search, downloads embedding model (~300MB)

Use `--no-embed` to skip the prompt and force BM25-only mode. In non-interactive environments (piped input), it defaults to vector mode.

Steps:
1. Registers the directory as a collection
2. Indexes all matching files
3. Downloads llama.cpp runtime + embedding model (unless BM25-only)
4. Generates embeddings for all documents (unless BM25-only)

Flags:
- `--name` — Collection name (defaults to directory basename)
- `--glob` — File pattern (default: `**/*.md`)
- `--context` — Human description for LLM context (improves search relevance)
- `--no-embed` — Skip embedding; BM25-only fast indexing

Examples:
```
picoqmd add ~/notes
picoqmd add ~/project/docs --name "project-docs" --context "Technical documentation for Project X"
picoqmd add ~/code --glob "**/*.{md,txt}" --name code-docs
picoqmd add ~/wiki --no-embed                    # BM25 only, instant
```

### `sync` — Re-index and Re-embed

```
picoqmd sync
picoqmd sync --no-embed    # re-index only, skip embedding
```

Detects changed files across all collections, re-indexes them, and generates embeddings for new/modified documents. Incremental — only processes what changed. Use `--no-embed` to re-index without triggering model downloads or embedding.

`update` and `embed` are aliases for `sync`.

### `search` — BM25 Full-text Search

```
picoqmd search <query> [--limit N] [--format FORMAT]
```

### `vsearch` — Semantic Vector Search

```
picoqmd vsearch <query> [--limit N] [--format FORMAT]
```

### `query` — Full Hybrid Search

```
picoqmd query <query> [--limit N] [--format FORMAT]
```

### `get` — Retrieve a Document

```
picoqmd get <ref>
```

Retrieve by docid (`#a3f2c1`) or file path.

### `status` — Index Statistics

```
picoqmd status
```

Shows collection count, document count, chunk count, and database path.

### `model` — Manage Models

```
picoqmd model list                    # show available models and status
picoqmd model download                # download all models
picoqmd model download embedding      # download just the embedding model
picoqmd model download reranker       # download just the reranker
picoqmd model download expansion      # download just the expansion model
```

Models (~2GB total):
| Model | Size | Purpose |
|---|---|---|
| embedding | ~300MB | Document and query embeddings (auto-downloaded by `add`) |
| reranker | ~600MB | Cross-encoder re-ranking (opt-in via `model download`) |
| expansion | ~1GB | Query expansion (opt-in via `model download`) |

### `mcp` — MCP Server

```
picoqmd mcp                           # stdio transport (for Claude Code/Desktop)
picoqmd mcp --http :8181              # HTTP transport
```

Exposes 6 tools: `search`, `vector_search`, `deep_search`, `get`, `multi_get`, `status`.

#### Claude Code Integration

Add to `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "picoqmd": {
      "command": "picoqmd",
      "args": ["mcp"]
    }
  }
}
```

### `context` — Context Descriptions

```
picoqmd context add "qmd://docs" "Project architecture and API documentation"
```

Attaches a human-readable description to a collection path, improving search relevance.

### `collection` — Collection Management

```
picoqmd collection add <path> [flags]   # alias for top-level 'add'
```

## Output Formats

All search commands support `--format`:

| Format | Description |
|---|---|
| `text` | Human-readable numbered list (default) |
| `json` | JSON array of result objects |
| `csv` | Comma-separated with header row |
| `md` | Markdown with headings and blockquotes |
| `files` | File paths only, one per line (for piping) |

Examples:
```
picoqmd "auth" --format json | jq '.[0].path'
picoqmd "config" --format files | xargs cat
picoqmd "API" --format md > results.md
```

## Named Indexes

Run multiple independent indexes with `--index`:

```
picoqmd add ~/work-notes --index work
picoqmd add ~/personal   --index personal
picoqmd "query" --index work
picoqmd status --index personal
```

Each named index gets its own database and config file.

## File Locations

| What | Path |
|---|---|
| Config | `~/.config/picoqmd/index.yml` |
| Database | `~/.cache/picoqmd/index.sqlite` |
| Models | `~/.cache/picoqmd/models/` |
| Shared library | `~/.cache/picoqmd/lib/libllama.dylib` |

Override with `XDG_CONFIG_HOME` and `XDG_CACHE_HOME`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `PICOQMD_LIB` | Path to llama.cpp shared library (skips auto-download) |
| `YZMA_LIB` | Fallback for llama.cpp library path |
| `XDG_CONFIG_HOME` | Override config directory |
| `XDG_CACHE_HOME` | Override cache directory |

## How the Hybrid Pipeline Works

```
Input query
    |
    v
[Query Expansion] — LLM generates 2 alternative queries (lex + vec)
    |
    v
[Fan-Out] — Each query searched via BM25 AND vector (parallel)
    |
    v
[RRF Fusion] — Reciprocal Rank Fusion merges all result lists
    |              with top-rank bonuses and original-query weighting
    v
[LLM Re-ranking] — Cross-encoder scores top 30 candidates
    |
    v
[Position-aware Blend] — Weighted merge of RRF + reranker scores
    |
    v
Final ranked results
```

## Export / Import

Transfer a fully indexed database between machines. Useful for deploying to small devices (arm32, riscv64) that can't run local models.

### Export (big machine)

```
picoqmd export -o picoqmd-export.tar.gz
```

Bundles the SQLite database (with all embeddings) and config into a portable archive. Collection paths are rewritten to be relative.

### Import (small device)

```
picoqmd import picoqmd-export.tar.gz
```

Extracts the database and config to local picoqmd directories. The device immediately gets BM25 search plus precomputed-embedding ranking without needing any models.

### Remote Search

Forward searches to a remote picoqmd MCP server instead of searching locally:

```
# On the big machine
picoqmd mcp --http :8181

# On any device
picoqmd search "query" --remote big-machine:8181
picoqmd "query" --remote big-machine:8181
```

The `--remote` flag works with all search commands (`search`, `vsearch`, `query`, and smart search).

## Device Sync

For IoT/sensor workflows where a small device generates data and a big machine indexes it:

```
# On big machine: add sensor directory
picoqmd add ~/sensor-data --name sensors --context "Temperature readings"

# On big machine: cron to pull + re-index
*/5 * * * * rsync -az pi@sensor:/var/log/temps/ ~/sensor-data/ && picoqmd sync
```

Then export or use `--remote` to give the sensor device search access.

## Tips

- **Use keyword search 80% of the time.** It's instant and handles exact lookups perfectly.
- **Reserve `query` / deep search** for complex conceptual questions.
- **Add context descriptions** to collections — they improve search quality.
- **Run `picoqmd sync` periodically** or via cron to keep the index fresh. PicoQMD does not auto-watch for file changes.
- **The `--no-embed` flag** is useful for fast initial indexing when you only need keyword search.
- **Use `--remote` on small devices** to leverage a bigger machine's full search pipeline.

---

# PicoQMD Cheat Sheet

```
# ── SETUP ──────────────────────────────────────────────
picoqmd add ~/notes                        # prompts: BM25-only or vector mode
picoqmd add ~/docs --name docs --no-embed  # BM25 only, skip model download
picoqmd add ~/code --glob "**/*.go"        # custom file pattern
picoqmd model download                     # download all 3 models for full hybrid
picoqmd model list                         # check model status

# ── SEARCH ─────────────────────────────────────────────
picoqmd "meeting notes"                    # smart search (auto-selects best pipeline)
picoqmd search "exact term"               # BM25 keyword search
picoqmd vsearch "conceptual meaning"       # semantic vector search
picoqmd query "complex question"           # full hybrid pipeline

# ── OPTIONS ────────────────────────────────────────────
--limit 20                                 # max results
--format json                              # output: text, json, csv, md, files
--index work                               # use named index

# ── MAINTENANCE ────────────────────────────────────────
picoqmd sync                               # re-index + re-embed changed files
picoqmd sync --no-embed                    # re-index only, no embedding
picoqmd update                             # alias for sync
picoqmd status                             # collection/document/chunk counts

# ── RETRIEVAL ──────────────────────────────────────────
picoqmd get "#a3f2c1"                      # fetch by docid
picoqmd get "docs/api.md"                  # fetch by path

# ── MCP SERVER ─────────────────────────────────────────
picoqmd mcp                                # stdio (Claude Code/Desktop)
picoqmd mcp --http :8181                   # HTTP endpoint

# ── CONTEXT ────────────────────────────────────────────
picoqmd context add "qmd://notes" "Personal notes and journal entries"

# ── EXPORT / IMPORT ────────────────────────────────────
picoqmd export -o bundle.tar.gz                              # export DB + config
picoqmd import bundle.tar.gz                                 # import on another machine
picoqmd import bundle.tar.gz --index sensors                 # import to named index

# ── REMOTE SEARCH ─────────────────────────────────────
picoqmd "query" --remote big-machine:8181                    # forward to remote MCP
picoqmd search "exact" --remote 192.168.1.10:8181            # remote BM25

# ── SHELL RECIPES ──────────────────────────────────────
picoqmd "auth" --format files | xargs cat                    # dump matching files
picoqmd "config" --format json | jq '.[].path'               # extract paths
picoqmd "TODO" --format csv > results.csv                    # export to CSV
watch -n 60 picoqmd sync                                     # auto-sync every minute
echo '0 2 * * * picoqmd sync' | crontab -                   # nightly sync at 2am

# ── MODELS ─────────────────────────────────────────────
# embedding  (~300MB) — auto-downloaded by 'add', enables vector search
# reranker   (~600MB) — opt-in, enables cross-encoder re-ranking
# expansion  (~1GB)   — opt-in, enables query expansion
# All 3 needed for full hybrid pipeline ('query' / smart search)
```
