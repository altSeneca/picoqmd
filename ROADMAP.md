# picoqmd Roadmap — resource-reducing retrieval (research synthesis, 2026-07-16)

Goal: full qmd replacement at a fraction of the resources. picoqmd's hardware
profile inverts fashionable-ANN assumptions — storage is cheap, CPU is scarce,
no GPU. The wins are cutting **bytes touched per query** and **redundant embed
compute**, not graph indexes (LEANN/HNSW) that trade disk for query-time
inference.

Current baseline (v0.4.0): pure-Go SQLite (zombiezen), contentless FTS5 BM25
(title 5×, post-hoc b=0.55), brute-force cosine over full float32 768-dim
vectors, weighted RRF fusion, EmbeddingGemma-300M-Q8 + Qwen3 reranker +
1.7B query expansion via llama.cpp FFI.

## Sequenced plan

### Phase 1 — cheap + fresh
1. **Matryoshka truncation 768→256** (~10 lines, no deps).
   EmbeddingGemma is MRL-trained: `vec[:256]` + L2-renormalize. −2.4% MTEB,
   exact 3× smaller disk/RAM and 3× faster brute-force scan (fewer BLOB bytes
   AND fewer float ops). Bump chunkerVersion / fingerprint; store dim.
   This alone makes embedding the 6,400-book corpus tractable.
2. **Flat file-hash incremental reindex.** Only re-chunk/re-embed changed
   files (path→content-hash already exists; extend to chunk-level so a
   one-paragraph edit re-embeds one chunk — ck reports 80–90% cache hits).
   Avoids the expensive llama.cpp FFI on unchanged content.
3. **Recency decay** — blend `exp(-ln2·age/halflife)` into fused score.
   Needs an `mtime` column. No surveyed local tool does this; pure quality
   gain for memory/daily-logs style corpora. (zombiezen `CreateFunction`
   supports pure-Go SQL scalar functions if needed.)
4. **Exact-phrase / NEAR bonus** — second `"quoted phrase"` MATCH, Go-side
   score multiplier. FTS5 bm25() is bag-of-words; SQLite maintainers
   declined native proximity ranking, blend-in-Go is the sanctioned path.

### Phase 2 — scale (books corpus, Pi-class hardware)
5. **Binary quantization + two-phase rescore.** Sign-bit pack each dim;
   phase 1 Hamming scan (`math/bits.OnesCount64(a^b)` — the one place pure
   Go matches SIMD) for a top-k×8 shortlist; phase 2 rescore shortlist with
   full-precision cosine. Published: 96–99% of float32 quality at ~25× scan
   speedup; stacked with Matryoshka-256 ≈ 64× storage reduction at 95–96%
   quality (Vespa). sqlite-vec's C extension can't load in pure-Go SQLite,
   but its binary-quant recipe ports cleanly.

### Phase 3 — code intelligence (differentiators)
6. **Tree-sitter AST chunking** (`smacker/go-tree-sitter`, cgo at compile
   time only). Chunk at function/class boundaries, fall back line-based;
   Roo Code's exact strategy. Fixes the 900-token splitter cutting
   functions mid-body in shared-src/android-src/ios-src.
7. **Aider-style personalized-PageRank repo map.** File graph from
   tree-sitter def/ref tags, PageRank seeded by query-mentioned
   identifiers, token-budgeted output. ~50-line power iteration; expose as
   `repo_map` MCP tool. Strongest edge vs "just grep."
8. **Polish:** fsnotify live watch; `.picoqmdignore`; path-segment +
   H1/H2 FTS columns; EmbeddingGemma QAT q4_0 (278MB) as default download.

## Explicitly rejected
- **LEANN selective recomputation** — saves disk (surplus) by spending
  query-time embed compute (scarce, GPU-less). Inverted economics.
- **sqlite-vec / vectorlite / usearch as engines** — C extensions can't
  dlopen in modernc/zombiezen; ANN unnecessary at single-user scale.
- **Custom FTS5 C aux functions** — no C ABI; multi-MATCH + Go blend instead.
- **Symbol-level call graph** — only if pivoting to code intelligence.

## Positioning
Claude Code and Cody both moved away from embeddings toward agentic grep.
picoqmd's edge is **cheap semantic recall returned as a small, ranked,
token-efficient payload** that beats a Glob→Grep→Read loop — ideas 1–3 keep
it cheap and fresh, 5 makes it scale, 7 makes it structural.

## Sources
LEANN arxiv.org/abs/2506.08276 · github.com/yichuan-w/LEANN ·
sqlite-vec github.com/asg017/sqlite-vec (binary-quant + matryoshka guides) ·
huggingface.co/blog/embedding-quantization · /embeddinggemma ·
blog.vespa.ai/combining-matryoshka-with-binary-quantization-using-embedder ·
github.com/BeaconBay/ck · run-llama/semtools · probelabs/probe ·
smacker/go-tree-sitter · fsnotify/fsnotify · aider.chat/docs/repomap.html ·
cursor.com/blog/secure-codebase-indexing · docs.roocode.com/features/codebase-indexing ·
cline.bot/blog/why-cline-doesnt-index-your-codebase · sqlite.org/fts5.html
