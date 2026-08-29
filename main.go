// picoqmd — A fully local markdown search engine. Single binary, zero cloud.
//
// Go reimplementation of QMD (github.com/tobilu/qmd) — BM25 full-text search,
// semantic vector search, and a hybrid pipeline with query expansion, RRF
// fusion, and LLM re-ranking. All offline.
//
// On amd64/arm64, uses yzma (github.com/hybridgroup/yzma) for pure-Go
// llama.cpp bindings via purego FFI. On other architectures (arm32, riscv64,
// mips), builds with BM25 + precomputed-embedding search only.
//
// Build: go build -ldflags="-s -w" -o picoqmd .
// Usage: picoqmd add ~/notes              (auto-downloads models, indexes, embeds)
//        picoqmd "meeting notes"           (smart search — auto-selects best pipeline)
//        picoqmd sync                      (re-index + re-embed changed files)
//        picoqmd search "meeting notes"    (explicit BM25 search)
//        picoqmd vsearch "semantic meaning"
//        picoqmd query "deep hybrid search"
//        picoqmd mcp --http :8181

package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
	"zombiezen.com/go/sqlite" // Pure Go SQLite — no CGO
	"zombiezen.com/go/sqlite/sqlitex"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	version       = "0.6.0"
	defaultDB     = "index.sqlite"
	chunkTarget   = 900 // target tokens per chunk
	chunkLookback = 200 // tokens to look back for break points
	chunkOverlap  = 135 // 15% of chunkTarget — overlap tokens between adjacent chunks

	// chunkerVersion is folded into the embedding fingerprint. Bump it when
	// ChunkDocument's output changes so existing vectors re-embed.
	chunkerVersion = "cv1"

	// embedEngineUnavailableExit is the embed-worker exit code for "the
	// embedding engine cannot initialize at all" — the orchestrator aborts
	// rather than skip-looping through every pending document.
	embedEngineUnavailableExit = 3

	rrfK              = 60      // RRF fusion constant
	maxRerank         = 30      // candidates sent to reranker
	maxIndexFileBytes = 1 << 20 // 1 MB file size limit for indexing

	// BM25 tuning: target parameters (FTS5 hardcodes k1=1.2, b=0.75)
	bm25TargetB  = 0.55 // lower b → less penalty on long docs (default FTS5: 0.75)
	bm25DefaultB = 0.75 // FTS5 built-in value

	// Strong-signal expansion bypass: skip the LLM ExpandQuery stage when BM25
	// alone gives a clear winner. Scale-free since SearchBM25 and
	// SearchBM25Normalized return scores on different scales.
	strongSignalRatio  = 2.0 // top-1 must be at least this multiple of top-N
	strongSignalRankN  = 3   // index used for the comparator (3 → top-1 vs top-3)
	strongSignalProbeN = 20  // candidates pulled by the probe BM25 call
)

// skipDirs are directories that should never be traversed during indexing.
var skipDirs = map[string]bool{
	".git": true, ".hg": true, ".svn": true,
	"node_modules": true, ".venv": true, "venv": true,
	"__pycache__": true, ".mypy_cache": true, ".cache": true,
	"vendor": true, "dist": true, "build": true,
	".next": true, ".nuxt": true, "target": true,
}

// expandGlob takes a glob pattern like "**/*.{go,py,ts}" and returns a list
// of simple patterns suitable for filepath.Match (e.g. ["*.go","*.py","*.ts"]).
// The "**/" prefix is stripped since we match against basenames during walk.
func expandGlob(pattern string) ([]string, error) {
	// Strip leading **/ — we walk recursively anyway
	pattern = strings.TrimPrefix(pattern, "**/")

	// Expand brace syntax: *.{go,py} → [*.go, *.py]
	if i := strings.Index(pattern, "{"); i >= 0 {
		j := strings.Index(pattern[i:], "}")
		if j < 0 {
			return []string{pattern}, nil
		}
		prefix := pattern[:i]
		suffix := pattern[i+j+1:]
		alts := strings.Split(pattern[i+1:i+j], ",")
		var out []string
		for _, alt := range alts {
			out = append(out, prefix+strings.TrimSpace(alt)+suffix)
		}
		return out, nil
	}
	return []string{pattern}, nil
}

// matchesAny checks if relPath matches any of the expanded glob patterns.
func matchesAny(patterns []string, relPath string) bool {
	base := filepath.Base(relPath)
	for _, p := range patterns {
		if strings.Contains(p, "/") {
			if ok, _ := filepath.Match(p, relPath); ok {
				return true
			}
		} else {
			if ok, _ := filepath.Match(p, base); ok {
				return true
			}
		}
	}
	return false
}

// isBinary checks whether data looks like binary content (contains null bytes
// in the first 512 bytes — same heuristic as git).
func isBinary(data []byte) bool {
	n := 512
	if len(data) < n {
		n = len(data)
	}
	return bytes.ContainsRune(data[:n], 0)
}

// ---------------------------------------------------------------------------
// Config (YAML — mirrors ~/.config/qmd/index.yml)
// ---------------------------------------------------------------------------

type Config struct {
	Collections []CollectionConfig `yaml:"collections"`
	Contexts    []ContextEntry     `yaml:"contexts,omitempty"`
}

type CollectionConfig struct {
	Name    string `yaml:"name"`
	Path    string `yaml:"path"`
	Glob    string `yaml:"glob,omitempty"`    // default: "**/*.md"
	Context string `yaml:"context,omitempty"` // human description
}

type ContextEntry struct {
	URI     string `yaml:"uri"`
	Context string `yaml:"context"`
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

type Document struct {
	ID         int64
	Path       string
	Title      string
	DocID      string // 6-char content hash
	Hash       string // full content hash for change detection
	Active     bool
	Content    string
	Context    string // inherited from collection/context tree
	Collection string // owning collection name
	AbsPath    string // on-disk filesystem path
}

// SkippedFile records a multi_get match that was not returned because its
// on-disk size exceeded the maxBytes threshold.
type SkippedFile struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

type SearchResult struct {
	DocID   string  `json:"docid"`
	Path    string  `json:"path"`
	Title   string  `json:"title"`
	Score   float64 `json:"score"`
	Snippet string  `json:"snippet,omitempty"`
	Line    int     `json:"line,omitempty"` // 1-based line where the snippet window begins (0 = unknown)
	Context string  `json:"context,omitempty"`
}

type Chunk struct {
	Hash string
	Seq  int
	Pos  int
	Text string
}

// ---------------------------------------------------------------------------
// Store — all SQLite operations
// ---------------------------------------------------------------------------

type Store struct {
	pool *sqlitex.Pool
	mu   sync.RWMutex
}

func NewStore(dbPath string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return nil, err
	}
	// WAL handles read/write concurrency but does not serialise concurrent
	// writers — a scheduled sync racing the MCP daemon or an embed worker
	// needs a generous busy timeout to queue instead of throwing
	// SQLITE_BUSY. 0 restores fail-fast.
	busyTimeout := 120 * time.Second
	if v := os.Getenv("PICOQMD_SQLITE_BUSY_TIMEOUT"); v != "" {
		if ms, err := strconv.Atoi(v); err == nil && ms >= 0 {
			busyTimeout = time.Duration(ms) * time.Millisecond
		}
	}
	pool, err := sqlitex.NewPool(dbPath, sqlitex.PoolOptions{
		PoolSize: 4,
		PrepareConn: func(conn *sqlite.Conn) error {
			conn.SetBusyTimeout(busyTimeout)
			return nil
		},
	})
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	s := &Store{pool: pool}
	if err := s.migrate(); err != nil {
		pool.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() { s.pool.Close() }

func (s *Store) migrate() error {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	err = sqlitex.ExecuteScript(conn, `
		CREATE TABLE IF NOT EXISTS collections (
			id      INTEGER PRIMARY KEY,
			name    TEXT UNIQUE NOT NULL,
			path    TEXT NOT NULL,
			glob    TEXT NOT NULL DEFAULT '**/*.md',
			context TEXT NOT NULL DEFAULT ''
		);

		CREATE TABLE IF NOT EXISTS documents (
			id     INTEGER PRIMARY KEY,
			col_id INTEGER NOT NULL REFERENCES collections(id),
			path   TEXT NOT NULL,
			title  TEXT NOT NULL DEFAULT '',
			docid  TEXT NOT NULL,
			hash   TEXT NOT NULL,
			active INTEGER NOT NULL DEFAULT 1,
			UNIQUE(col_id, path)
		);

		-- FTS5 full-text index for BM25 search
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
			title, content, docid,
			content='', contentless_delete=1,
			tokenize='porter unicode61'
		);

		CREATE TABLE IF NOT EXISTS content_vectors (
			hash TEXT NOT NULL,
			seq  INTEGER NOT NULL,
			pos  INTEGER NOT NULL,
			text TEXT NOT NULL,
			vec  BLOB,
			PRIMARY KEY (hash, seq)
		);

		-- Cached LLM responses for query expansion + reranking
		CREATE TABLE IF NOT EXISTS llm_cache (
			key   TEXT PRIMARY KEY,
			value TEXT NOT NULL,
			ts    INTEGER NOT NULL
		);

		CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(hash);
		CREATE INDEX IF NOT EXISTS idx_doc_docid ON documents(docid);
		CREATE INDEX IF NOT EXISTS idx_vec_hash ON content_vectors(hash);
	`, nil)
	if err != nil {
		return err
	}

	// Migration: add doc_len column for BM25 length normalization
	// ALTER TABLE fails silently if column already exists — that's fine
	sqlitex.Execute(conn, `ALTER TABLE documents ADD COLUMN doc_len INTEGER NOT NULL DEFAULT 0`, nil)

	// Migration: add embedding fingerprint (model + chunker identity) so a
	// model or chunker change marks existing vectors as pending instead of
	// silently searching stale embeddings.
	sqlitex.Execute(conn, `ALTER TABLE content_vectors ADD COLUMN fp TEXT NOT NULL DEFAULT ''`, nil)

	// Legacy adoption: vectors written before fingerprinting are assumed to
	// match the current model (picoqmd has only ever shipped one embed
	// model). One-time no-op afterwards.
	sqlitex.Execute(conn, `UPDATE content_vectors SET fp=? WHERE fp='' AND vec IS NOT NULL`,
		&sqlitex.ExecOptions{Args: []any{embedFingerprint()}})

	return nil
}

// UpsertCollection adds or updates a collection and returns its ID.
func (s *Store) UpsertCollection(name, path, glob, ctx string) (int64, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return 0, err
	}
	defer s.pool.Put(conn)

	if err := sqlitex.Execute(conn,
		`INSERT INTO collections (name, path, glob, context)
		 VALUES (?, ?, ?, ?)
		 ON CONFLICT(name) DO UPDATE SET path=excluded.path, glob=excluded.glob, context=excluded.context`,
		&sqlitex.ExecOptions{Args: []any{name, path, glob, ctx}}); err != nil {
		return 0, err
	}

	var id int64
	err = sqlitex.Execute(conn, `SELECT id FROM collections WHERE name = ?`,
		&sqlitex.ExecOptions{
			Args: []any{name},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				id = stmt.ColumnInt64(0)
				return nil
			},
		})
	return id, err
}

// UpsertDocument indexes a single document, updating FTS.
func (s *Store) UpsertDocument(colID int64, relPath, title, content string) error {
	hash := contentHash(content)
	docid := hash[:6]

	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	defer sqlitex.Save(conn)(&err)

	// Check if document exists and hash unchanged
	var existingHash string
	err = sqlitex.Execute(conn,
		`SELECT hash FROM documents WHERE col_id=? AND path=?`,
		&sqlitex.ExecOptions{
			Args: []any{colID, relPath},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				existingHash = stmt.ColumnText(0)
				return nil
			},
		})
	if err != nil {
		return err
	}
	if existingHash == hash {
		return nil // unchanged
	}

	// Upsert document row (doc_len = rune count / 4 ≈ token count)
	docLen := utf8.RuneCountInString(content) / 4
	err = sqlitex.Execute(conn,
		`INSERT INTO documents (col_id, path, title, docid, hash, active, doc_len)
		 VALUES (?, ?, ?, ?, ?, 1, ?)
		 ON CONFLICT(col_id, path) DO UPDATE SET title=excluded.title, docid=excluded.docid, hash=excluded.hash, active=1, doc_len=excluded.doc_len`,
		&sqlitex.ExecOptions{Args: []any{colID, relPath, title, docid, hash, docLen}})
	if err != nil {
		return err
	}

	// Get the rowid for FTS
	var rowid int64
	err = sqlitex.Execute(conn,
		`SELECT id FROM documents WHERE col_id=? AND path=?`,
		&sqlitex.ExecOptions{
			Args: []any{colID, relPath},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				rowid = stmt.ColumnInt64(0)
				return nil
			},
		})
	if err != nil {
		return err
	}

	// Update FTS — delete old entry (ignore "not found" since row may be new)
	if err = sqlitex.Execute(conn,
		`DELETE FROM documents_fts WHERE rowid=?`,
		&sqlitex.ExecOptions{Args: []any{rowid}}); err != nil {
		return err
	}

	return sqlitex.Execute(conn,
		`INSERT INTO documents_fts (rowid, title, content, docid) VALUES (?, ?, ?, ?)`,
		&sqlitex.ExecOptions{Args: []any{rowid, title, content, docid}})
}

// DeactivateStale marks documents as inactive if their paths are not in the
// given set of active paths. This handles deleted files and glob changes.
func (s *Store) DeactivateStale(colID int64, activePaths map[string]bool) error {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	var stalePaths []string
	err = sqlitex.Execute(conn,
		`SELECT path FROM documents WHERE col_id=? AND active=1`,
		&sqlitex.ExecOptions{
			Args: []any{colID},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				p := stmt.ColumnText(0)
				if !activePaths[p] {
					stalePaths = append(stalePaths, p)
				}
				return nil
			},
		})
	if err != nil {
		return err
	}

	for _, p := range stalePaths {
		if err := sqlitex.Execute(conn,
			`UPDATE documents SET active=0 WHERE col_id=? AND path=?`,
			&sqlitex.ExecOptions{Args: []any{colID, p}}); err != nil {
			return err
		}
	}
	if len(stalePaths) > 0 {
		fmt.Printf("  Deactivated %d stale documents\n", len(stalePaths))
	}
	return nil
}

// SearchBM25 performs FTS5 BM25 ranked search with column weights and b-correction.
func (s *Store) SearchBM25(query string, limit int) ([]SearchResult, error) {
	return s.searchBM25(query, "", limit)
}

// SearchBM25InCollection performs BM25 search scoped to a single collection.
func (s *Store) SearchBM25InCollection(query, collection string, limit int) ([]SearchResult, error) {
	return s.searchBM25(query, collection, limit)
}

func (s *Store) searchBM25(query, collection string, limit int) ([]SearchResult, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	// Compute average doc_len for b-correction
	var avgDocLen float64
	err = sqlitex.Execute(conn, `SELECT AVG(doc_len) FROM documents WHERE active=1 AND doc_len>0`,
		&sqlitex.ExecOptions{
			ResultFunc: func(stmt *sqlite.Stmt) error {
				avgDocLen = stmt.ColumnFloat(0)
				return nil
			},
		})
	if err != nil || avgDocLen == 0 {
		avgDocLen = 500 // reasonable fallback
	}

	ftsQuery := toFTS5Query(query)
	if ftsQuery == "" {
		return nil, nil // nothing searchable (e.g. punctuation-only query)
	}

	// Column weights: title=5.0, content=1.0, docid=0.0
	// No snippet() here: the FTS table is contentless (content=''), so FTS5
	// cannot reconstruct text — snippets are extracted from disk below.
	sql := `
		SELECT d.docid, d.path, d.title,
		       bm25(documents_fts, 5.0, 1.0, 0.0) AS score,
		       c.path || '/' || d.path AS abs_path,
		       c.context, d.doc_len, c.name AS col_name
		FROM documents_fts f
		JOIN documents d ON d.id = f.rowid
		JOIN collections c ON c.id = d.col_id
		WHERE documents_fts MATCH ?
		  AND d.active = 1`
	args := []any{ftsQuery}

	if collection != "" {
		sql += ` AND c.name = ?`
		args = append(args, collection)
	}

	sql += `
		ORDER BY score
		LIMIT ?`
	args = append(args, limit)

	type rawResult struct {
		result  SearchResult
		docLen  float64
		absPath string
	}
	var raw []rawResult

	err = sqlitex.Execute(conn, sql, &sqlitex.ExecOptions{
		Args: args,
		ResultFunc: func(stmt *sqlite.Stmt) error {
			raw = append(raw, rawResult{
				result: SearchResult{
					DocID:   stmt.ColumnText(0),
					Path:    stmt.ColumnText(1),
					Title:   stmt.ColumnText(2),
					Score:   -stmt.ColumnFloat(3), // bm25() returns negative scores
					Context: stmt.ColumnText(5),
				},
				absPath: stmt.ColumnText(4),
				docLen:  stmt.ColumnFloat(6),
			})
			return nil
		},
	})
	if err != nil {
		return nil, err
	}

	// Apply post-FTS5 b-correction: adjust from default b=0.75 to target b=0.55
	results := make([]SearchResult, len(raw))
	absPaths := make([]string, len(raw))
	for i, r := range raw {
		results[i] = r.result
		absPaths[i] = r.absPath
		if r.docLen > 0 {
			results[i].Score *= adjustBM25B(r.docLen, avgDocLen)
		}
	}

	// Re-sort after correction (order may shift slightly)
	order := make([]int, len(results))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return results[order[i]].Score > results[order[j]].Score })
	sorted := make([]SearchResult, len(results))
	sortedPaths := make([]string, len(results))
	for i, idx := range order {
		sorted[i] = results[idx]
		sortedPaths[i] = absPaths[idx]
	}
	results, absPaths = sorted, sortedPaths

	// Snippets from disk for the returned page. Bounded work: `limit` files,
	// each already capped at maxIndexFileBytes during indexing.
	terms := queryTerms(query)
	for i := range results {
		data, err := os.ReadFile(absPaths[i])
		if err != nil {
			continue // file moved since indexing; result is still valid
		}
		snip, windowStart := extractSnippet(string(data), terms, 200)
		if snip != "" {
			results[i].Snippet = snip
			results[i].Line = bytes.Count(data[:windowStart], []byte("\n")) + 1
		}
	}

	return results, nil
}

// adjustBM25B compensates for FTS5's hardcoded b=0.75, shifting toward target b.
// correction = (1 - targetB + targetB*docLen/avgDocLen) / (1 - defaultB + defaultB*docLen/avgDocLen)
func adjustBM25B(docLen, avgDocLen float64) float64 {
	ratio := docLen / avgDocLen
	return (1 - bm25TargetB + bm25TargetB*ratio) / (1 - bm25DefaultB + bm25DefaultB*ratio)
}

// ActiveCollectionNames returns distinct collection names that have active documents.
func (s *Store) ActiveCollectionNames() ([]string, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var names []string
	err = sqlitex.Execute(conn, `
		SELECT DISTINCT c.name FROM collections c
		JOIN documents d ON d.col_id = c.id
		WHERE d.active = 1`,
		&sqlitex.ExecOptions{
			ResultFunc: func(stmt *sqlite.Stmt) error {
				names = append(names, stmt.ColumnText(0))
				return nil
			},
		})
	return names, err
}

// parseScope splits a collection scope string ("a" or "a,b,c") into names.
// Empty string means "no scope" (all collections).
func parseScope(scope string) []string {
	var names []string
	for _, n := range strings.Split(scope, ",") {
		if n = strings.TrimSpace(n); n != "" {
			names = append(names, n)
		}
	}
	return names
}

// SearchBM25Scoped resolves a collection scope: "" searches all collections
// (per-collection RRF), a single name searches that collection, and a
// comma-separated list searches each listed collection separately and fuses
// via RRF — so unrelated result sets can't crowd each other out. (qmd 2.8.3)
func (s *Store) SearchBM25Scoped(query, scope string, limit int) ([]SearchResult, error) {
	names := parseScope(scope)
	switch len(names) {
	case 0:
		return s.SearchBM25Normalized(query, limit)
	case 1:
		return s.searchBM25(query, names[0], limit)
	default:
		return s.searchBM25RRF(query, names, limit)
	}
}

// SearchBM25Normalized runs BM25 per-collection and fuses via RRF so small
// collections get fair representation against large ones.
func (s *Store) SearchBM25Normalized(query string, limit int) ([]SearchResult, error) {
	names, err := s.ActiveCollectionNames()
	if err != nil || len(names) <= 1 {
		// Single collection or error — fall back to normal search
		return s.SearchBM25(query, limit)
	}
	return s.searchBM25RRF(query, names, limit)
}

// searchBM25RRF searches each named collection separately and fuses the
// ranked lists with reciprocal rank fusion.
func (s *Store) searchBM25RRF(query string, names []string, limit int) ([]SearchResult, error) {
	// Run per-collection searches
	type rankedList struct {
		results []SearchResult
	}
	var lists []rankedList
	for _, name := range names {
		results, err := s.SearchBM25InCollection(query, name, limit)
		if err != nil {
			continue
		}
		if len(results) > 0 {
			lists = append(lists, rankedList{results: results})
		}
	}

	if len(lists) == 0 {
		return nil, nil
	}

	// Per-collection RRF fusion: each collection is a separate ranked list
	scores := make(map[string]float64)
	docs := make(map[string]SearchResult)
	for _, list := range lists {
		for rank, r := range list.results {
			scores[r.DocID] += 1.0 / float64(rrfK+rank+1)
			if _, ok := docs[r.DocID]; !ok {
				docs[r.DocID] = r
			}
		}
	}

	type entry struct {
		docID string
		score float64
	}
	var entries []entry
	for id, sc := range scores {
		entries = append(entries, entry{id, sc})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].score > entries[j].score })

	var results []SearchResult
	for _, e := range entries {
		r := docs[e.docID]
		r.Score = e.score
		results = append(results, r)
		if len(results) >= limit {
			break
		}
	}
	return results, nil
}

// SearchVector performs optimized cosine similarity search over stored
// embeddings. `query` is the original text query and is used purely for
// snippet selection — pass "" if you only have the vector and don't need
// query-aware snippets.
func (s *Store) SearchVector(query string, queryVec []float32, limit int) ([]SearchResult, error) {
	return s.searchVector(query, queryVec, "", limit)
}

// SearchVectorInCollection performs vector search scoped to a single collection.
func (s *Store) SearchVectorInCollection(query string, queryVec []float32, collection string, limit int) ([]SearchResult, error) {
	return s.searchVector(query, queryVec, collection, limit)
}

// SearchVectorScoped resolves a collection scope: "" = all collections, a
// single name = that collection, a comma-separated list = those collections.
// Cosine scores are directly comparable across collections, so the multi
// case is a single filtered scan — no fusion needed.
func (s *Store) SearchVectorScoped(query string, queryVec []float32, scope string, limit int) ([]SearchResult, error) {
	return s.searchVector(query, queryVec, scope, limit)
}

func (s *Store) searchVector(query string, queryVec []float32, collection string, limit int) ([]SearchResult, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	// Build set of hashes belonging to the scoped collection(s), if filtered.
	// `collection` may be a single name or a comma-separated list.
	var collectionHashes map[string]bool
	if names := parseScope(collection); len(names) > 0 {
		collectionHashes = make(map[string]bool)
		placeholders := strings.Repeat("?,", len(names))
		placeholders = placeholders[:len(placeholders)-1]
		args := make([]any, len(names))
		for i, n := range names {
			args[i] = n
		}
		err = sqlitex.Execute(conn, `
			SELECT DISTINCT d.hash FROM documents d
			JOIN collections c ON c.id = d.col_id
			WHERE c.name IN (`+placeholders+`) AND d.active = 1`,
			&sqlitex.ExecOptions{
				Args: args,
				ResultFunc: func(stmt *sqlite.Stmt) error {
					collectionHashes[stmt.ColumnText(0)] = true
					return nil
				},
			})
		if err != nil {
			return nil, err
		}
	}

	type bestChunk struct {
		score float64
		seq   int
	}
	bestByHash := make(map[string]bestChunk)

	vecDim := len(queryVec)
	queryF64 := make([]float64, vecDim)
	var queryNormSq float64
	for i, v := range queryVec {
		f := float64(v)
		queryF64[i] = f
		queryNormSq += f * f
	}
	queryNorm := math.Sqrt(queryNormSq)
	if queryNorm == 0 {
		return nil, nil
	}

	decodeBuf := make([]float32, vecDim)

	err = sqlitex.Execute(conn, `SELECT hash, seq, vec FROM content_vectors WHERE vec IS NOT NULL`,
		&sqlitex.ExecOptions{
			ResultFunc: func(stmt *sqlite.Stmt) error {
				h := stmt.ColumnText(0)
				seq := stmt.ColumnInt(1)

				// Skip hashes not in the target collection
				if collectionHashes != nil && !collectionHashes[h] {
					return nil
				}

				vecLen := stmt.ColumnLen(2)
				if vecLen != vecDim*4 {
					return nil
				}
				raw := make([]byte, vecLen)
				stmt.ColumnBytes(2, raw)
				for i := 0; i < vecDim; i++ {
					bits := uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 | uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24
					decodeBuf[i] = math.Float32frombits(bits)
				}

				var dot, normSq float64
				n := vecDim &^ 7
				for i := 0; i < n; i += 8 {
					d0, d1 := float64(decodeBuf[i]), float64(decodeBuf[i+1])
					d2, d3 := float64(decodeBuf[i+2]), float64(decodeBuf[i+3])
					d4, d5 := float64(decodeBuf[i+4]), float64(decodeBuf[i+5])
					d6, d7 := float64(decodeBuf[i+6]), float64(decodeBuf[i+7])
					dot += queryF64[i]*d0 + queryF64[i+1]*d1 +
						queryF64[i+2]*d2 + queryF64[i+3]*d3 +
						queryF64[i+4]*d4 + queryF64[i+5]*d5 +
						queryF64[i+6]*d6 + queryF64[i+7]*d7
					normSq += d0*d0 + d1*d1 + d2*d2 + d3*d3 +
						d4*d4 + d5*d5 + d6*d6 + d7*d7
				}
				for i := n; i < vecDim; i++ {
					d := float64(decodeBuf[i])
					dot += queryF64[i] * d
					normSq += d * d
				}

				docNorm := math.Sqrt(normSq)
				if docNorm == 0 {
					return nil
				}
				sim := dot / (queryNorm * docNorm)

				if cur, ok := bestByHash[h]; !ok || sim > cur.score {
					bestByHash[h] = bestChunk{score: sim, seq: seq}
				}
				return nil
			},
		})
	if err != nil {
		return nil, err
	}

	type scored struct {
		hash  string
		score float64
		seq   int
	}
	sorted_ := make([]scored, 0, len(bestByHash))
	for h, b := range bestByHash {
		sorted_ = append(sorted_, scored{h, b.score, b.seq})
	}
	sort.Slice(sorted_, func(i, j int) bool { return sorted_[i].score > sorted_[j].score })
	if len(sorted_) > limit {
		sorted_ = sorted_[:limit]
	}

	terms := queryTerms(query)

	var results []SearchResult
	for _, s2 := range sorted_ {
		var r SearchResult
		var absPath string
		r.Score = s2.score
		err = sqlitex.Execute(conn, `
			SELECT d.docid, d.path, d.title, c.context, c.path
			FROM documents d
			JOIN collections c ON c.id = d.col_id
			WHERE d.hash = ? AND d.active = 1
			LIMIT 1`,
			&sqlitex.ExecOptions{
				Args: []any{s2.hash},
				ResultFunc: func(stmt *sqlite.Stmt) error {
					r.DocID = stmt.ColumnText(0)
					r.Path = stmt.ColumnText(1)
					r.Title = stmt.ColumnText(2)
					r.Context = stmt.ColumnText(3)
					absPath = filepath.Join(stmt.ColumnText(4), r.Path)
					return nil
				},
			})
		if err != nil {
			continue
		}
		if r.DocID == "" {
			continue
		}

		// Pull the winning chunk's text + position so we can build a
		// query-aware snippet and an absolute line number.
		var chunkText string
		var chunkPos int
		_ = sqlitex.Execute(conn, `SELECT text, pos FROM content_vectors WHERE hash = ? AND seq = ? LIMIT 1`,
			&sqlitex.ExecOptions{
				Args: []any{s2.hash, s2.seq},
				ResultFunc: func(stmt *sqlite.Stmt) error {
					chunkText = stmt.ColumnText(0)
					chunkPos = stmt.ColumnInt(1)
					return nil
				},
			})
		if chunkText != "" {
			snippet, windowStart := extractSnippet(chunkText, terms, 200)
			r.Snippet = snippet
			r.Line = computeLineNumber(absPath, chunkPos+windowStart)
		}

		results = append(results, r)
	}
	return results, nil
}

// GetDocument retrieves a single document by docid or path.
// parseRefRange splits a trailing :from or :from:count line-range suffix off
// a get ref. Only called when the full ref fails to resolve, so paths that
// happen to contain colons still work.
func parseRefRange(ref string) (base string, from, count int) {
	parts := strings.Split(ref, ":")
	if len(parts) >= 3 {
		if m, err1 := strconv.Atoi(parts[len(parts)-2]); err1 == nil {
			if n, err2 := strconv.Atoi(parts[len(parts)-1]); err2 == nil {
				return strings.Join(parts[:len(parts)-2], ":"), m, n
			}
		}
	}
	if len(parts) >= 2 {
		if n, err := strconv.Atoi(parts[len(parts)-1]); err == nil {
			return strings.Join(parts[:len(parts)-1], ":"), n, 0
		}
	}
	return ref, 0, 0
}

// lookupDocument resolves a bare ref — #docid, collection-relative path, or
// qmd://collection/path — to its metadata row. Content is not loaded.
func (s *Store) lookupDocument(ref string) (*Document, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	ref = strings.TrimPrefix(ref, "qmd://")
	ref = strings.TrimPrefix(ref, "#")
	var doc Document
	found := false

	query := `SELECT d.id, d.path, d.title, d.docid, d.hash, c.context, c.name, c.path || '/' || d.path
	           FROM documents d JOIN collections c ON c.id = d.col_id
	           WHERE (d.docid = ? OR d.path = ? OR c.name || '/' || d.path = ?) AND d.active = 1 LIMIT 1`

	err = sqlitex.Execute(conn, query, &sqlitex.ExecOptions{
		Args: []any{ref, ref, ref},
		ResultFunc: func(stmt *sqlite.Stmt) error {
			found = true
			doc.ID = stmt.ColumnInt64(0)
			doc.Path = stmt.ColumnText(1)
			doc.Title = stmt.ColumnText(2)
			doc.DocID = stmt.ColumnText(3)
			doc.Hash = stmt.ColumnText(4)
			doc.Context = stmt.ColumnText(5)
			doc.Collection = stmt.ColumnText(6)
			doc.AbsPath = stmt.ColumnText(7)
			return nil
		},
	})
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("document not found: %s", ref)
	}
	return &doc, nil
}

// GetDocument resolves ref — path, #docid, or qmd://collection/path, each
// optionally suffixed with :from or :from:count — and returns the document
// with Content read from disk plus the requested line range (0 = unset).
func (s *Store) GetDocument(ref string) (*Document, int, int, error) {
	doc, err := s.lookupDocument(ref)
	from, count := 0, 0
	if err != nil {
		base, f, c := parseRefRange(ref)
		if base == ref {
			return nil, 0, 0, err
		}
		doc, err = s.lookupDocument(base)
		if err != nil {
			return nil, 0, 0, err
		}
		from, count = f, c
	}

	data, err := os.ReadFile(doc.AbsPath)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("document %s is indexed but unreadable at %s (moved or deleted?) — run 'picoqmd sync': %w",
			doc.Path, doc.AbsPath, err)
	}
	doc.Content = string(data)
	return doc, from, count, nil
}

// MultiGet retrieves documents matching a glob pattern or comma-separated
// list of paths, with content read from disk. Files whose on-disk size
// exceeds maxBytes (when > 0) are reported in the skipped list instead of
// being silently dropped; files missing from disk are skipped with Size -1.
func (s *Store) MultiGet(pattern string, maxBytes int64) ([]Document, []SkippedFile, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, nil, err
	}
	defer s.pool.Put(conn)

	var docs []Document
	var skipped []SkippedFile
	seen := make(map[int64]bool)
	patterns := strings.Split(pattern, ",")
	for _, p := range patterns {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		likePattern := strings.ReplaceAll(p, "*", "%")
		var matches []Document
		err = sqlitex.Execute(conn, `
			SELECT d.id, d.path, d.title, d.docid, d.hash, c.context, c.name, c.path || '/' || d.path
			FROM documents d JOIN collections c ON c.id = d.col_id
			WHERE d.active = 1 AND d.path LIKE ?
			ORDER BY d.path`, &sqlitex.ExecOptions{
			Args: []any{likePattern},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				matches = append(matches, Document{
					ID:         stmt.ColumnInt64(0),
					Path:       stmt.ColumnText(1),
					Title:      stmt.ColumnText(2),
					DocID:      stmt.ColumnText(3),
					Hash:       stmt.ColumnText(4),
					Context:    stmt.ColumnText(5),
					Collection: stmt.ColumnText(6),
					AbsPath:    stmt.ColumnText(7),
					Active:     true,
				})
				return nil
			},
		})
		if err != nil {
			return nil, nil, err
		}
		for _, doc := range matches {
			if seen[doc.ID] {
				continue
			}
			seen[doc.ID] = true
			fi, statErr := os.Stat(doc.AbsPath)
			if statErr != nil {
				skipped = append(skipped, SkippedFile{Path: doc.Path, Size: -1})
				continue
			}
			if maxBytes > 0 && fi.Size() > maxBytes {
				skipped = append(skipped, SkippedFile{Path: doc.Path, Size: fi.Size()})
				continue
			}
			data, readErr := os.ReadFile(doc.AbsPath)
			if readErr != nil {
				skipped = append(skipped, SkippedFile{Path: doc.Path, Size: -1})
				continue
			}
			doc.Content = string(data)
			docs = append(docs, doc)
		}
	}
	return docs, skipped, nil
}

// renderDocument formats a document for display: a header with the qmd://
// URI (or on-disk path when fullPath), docid, and line range, followed by
// optionally line-numbered content. from/count of 0 mean start/whole file.
func renderDocument(doc *Document, from, count int, lineNumbers, fullPath bool) string {
	content := strings.TrimSuffix(doc.Content, "\n")
	lines := strings.Split(content, "\n")
	total := len(lines)
	if from < 1 {
		from = 1
	}
	if from > total {
		from = total
	}
	end := total
	if count > 0 && from-1+count < total {
		end = from - 1 + count
	}
	id := "qmd://" + doc.Collection + "/" + doc.Path
	if fullPath {
		id = doc.AbsPath
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s #%s (lines %d-%d of %d)\n", id, doc.DocID, from, end, total)
	for i := from; i <= end; i++ {
		if lineNumbers {
			fmt.Fprintf(&b, "%d→%s\n", i, lines[i-1])
		} else {
			b.WriteString(lines[i-1])
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// pendingHashCond selects active documents whose embeddings are incomplete:
// never chunked, any chunk still missing its vector, or any chunk embedded
// under a different model/chunker fingerprint. A document only counts as
// embedded when every chunk has a current vector — partial coverage from an
// interrupted run stays pending and resumes on the next embed.
// ?1 is a collection name (” = all collections), ?2 the current fingerprint.
const pendingHashCond = `
	d.active = 1
	AND (?1 = '' OR d.col_id IN (SELECT id FROM collections WHERE name = ?1))
	AND (
		NOT EXISTS (SELECT 1 FROM content_vectors v WHERE v.hash = d.hash)
		OR EXISTS (SELECT 1 FROM content_vectors v WHERE v.hash = d.hash AND (v.vec IS NULL OR v.fp <> ?2))
	)`

// UnembeddedHashes returns content hashes whose embeddings are missing,
// incomplete, or stale relative to the given fingerprint. collection scopes
// to one collection; "" means all. Ordered by hash so the embed worker and
// SkipNextUnembedded agree on which document is "next" — a skip must target
// the document the worker is actually stuck on.
func (s *Store) UnembeddedHashes(fp, collection string) ([]string, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var hashes []string
	err = sqlitex.Execute(conn,
		`SELECT DISTINCT d.hash FROM documents d WHERE `+pendingHashCond+` ORDER BY d.hash`,
		&sqlitex.ExecOptions{
			Args: []any{collection, fp},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				hashes = append(hashes, stmt.ColumnText(0))
				return nil
			},
		})
	return hashes, err
}

// CountUnembedded returns the number of active documents whose embeddings
// are missing, incomplete, or stale relative to the given fingerprint.
// collection scopes to one collection; "" means all.
func (s *Store) CountUnembedded(fp, collection string) (int, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return 0, err
	}
	defer s.pool.Put(conn)

	var count int
	err = sqlitex.Execute(conn,
		`SELECT COUNT(DISTINCT d.hash) FROM documents d WHERE `+pendingHashCond,
		&sqlitex.ExecOptions{
			Args: []any{collection, fp},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				count = stmt.ColumnInt(0)
				return nil
			},
		})
	return count, err
}

// SkipNextUnembedded marks the next unembedded document as embedded (with empty
// vector) so the orchestrator can make progress past problematic documents.
// It MUST use the same collection scope and ordering as UnembeddedHashes —
// a zero-progress worker is stuck on the first pending hash of ITS scope, and
// skipping any other document poisons unrelated collections while never
// unblocking the loop. Returns the skipped hash ("" if nothing was pending).
func (s *Store) SkipNextUnembedded(collection string) (string, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return "", err
	}
	defer s.pool.Put(conn)

	fp := embedFingerprint()
	var hash string
	err = sqlitex.Execute(conn,
		`SELECT DISTINCT d.hash FROM documents d WHERE `+pendingHashCond+` ORDER BY d.hash LIMIT 1`,
		&sqlitex.ExecOptions{
			Args: []any{collection, fp},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				hash = stmt.ColumnText(0)
				return nil
			},
		})
	if err != nil || hash == "" {
		return "", err
	}

	// Cover both pending shapes: a document with no chunks at all gets a
	// dummy row, and any chunks with missing/stale vectors are stamped so
	// the document stops matching pendingHashCond and the loop progresses.
	dummyVec := make([]byte, 4)
	if err := sqlitex.Execute(conn, `
		INSERT OR IGNORE INTO content_vectors (hash, seq, pos, text, vec, fp) VALUES (?, 0, 0, '[skipped]', ?, ?)
	`, &sqlitex.ExecOptions{
		Args: []any{hash, dummyVec, fp},
	}); err != nil {
		return "", err
	}
	err = sqlitex.Execute(conn, `
		UPDATE content_vectors SET vec = COALESCE(vec, ?), fp = ? WHERE hash = ? AND (vec IS NULL OR fp <> ?)
	`, &sqlitex.ExecOptions{
		Args: []any{dummyVec, fp, hash, fp},
	})
	return hash, err
}

// StoreChunks persists chunked text for a document hash.
func (s *Store) StoreChunks(hash string, chunks []Chunk) error {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	defer sqlitex.Save(conn)(&err)

	if err = sqlitex.Execute(conn, `DELETE FROM content_vectors WHERE hash=?`,
		&sqlitex.ExecOptions{Args: []any{hash}}); err != nil {
		return err
	}

	for _, c := range chunks {
		if err = sqlitex.Execute(conn,
			`INSERT INTO content_vectors (hash, seq, pos, text) VALUES (?, ?, ?, ?)`,
			&sqlitex.ExecOptions{Args: []any{c.Hash, c.Seq, c.Pos, c.Text}}); err != nil {
			return err
		}
	}
	return nil
}

// StoreVector writes the embedding vector for a chunk, stamped with the
// fingerprint of the model/chunker that produced it.
func (s *Store) StoreVector(hash string, seq int, vec []float32, fp string) error {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	return sqlitex.Execute(conn,
		`UPDATE content_vectors SET vec=?, fp=? WHERE hash=? AND seq=?`,
		&sqlitex.ExecOptions{Args: []any{float32ToBytes(vec), fp, hash, seq}})
}

// HasStaleFP reports whether any embedded chunk of a hash carries a
// fingerprint other than fp — meaning the model or chunker changed and the
// document must be re-chunked and re-embedded from scratch.
func (s *Store) HasStaleFP(hash, fp string) (bool, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return false, err
	}
	defer s.pool.Put(conn)

	var stale bool
	err = sqlitex.Execute(conn,
		`SELECT 1 FROM content_vectors WHERE hash = ? AND vec IS NOT NULL AND fp <> ? LIMIT 1`,
		&sqlitex.ExecOptions{
			Args: []any{hash, fp},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				stale = true
				return nil
			},
		})
	return stale, err
}

// Stats returns index statistics.
func (s *Store) Stats() (collections, documents, chunks int, err error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return 0, 0, 0, err
	}
	defer s.pool.Put(conn)

	count := func(table string) int {
		var n int
		_ = sqlitex.Execute(conn, "SELECT COUNT(*) FROM "+table,
			&sqlitex.ExecOptions{ResultFunc: func(stmt *sqlite.Stmt) error {
				n = stmt.ColumnInt(0)
				return nil
			}})
		return n
	}
	return count("collections"), count("documents"), count("content_vectors"), nil
}

// DocForHash returns the title and absolute file path for a document hash.
func (s *Store) DocForHash(hash string) (title, absPath string, err error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return "", "", err
	}
	defer s.pool.Put(conn)

	err = sqlitex.Execute(conn, `
		SELECT d.title, c.path || '/' || d.path
		FROM documents d JOIN collections c ON c.id = d.col_id
		WHERE d.hash = ? AND d.active = 1 LIMIT 1`,
		&sqlitex.ExecOptions{
			Args: []any{hash},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				title = stmt.ColumnText(0)
				absPath = stmt.ColumnText(1)
				return nil
			},
		})
	return
}

// HasChunks returns true if a document hash has been chunked.
func (s *Store) HasChunks(hash string) (bool, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return false, err
	}
	defer s.pool.Put(conn)

	var exists bool
	err = sqlitex.Execute(conn, `SELECT 1 FROM content_vectors WHERE hash = ? LIMIT 1`,
		&sqlitex.ExecOptions{
			Args: []any{hash},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				exists = true
				return nil
			},
		})
	return exists, err
}

// UnembeddedChunks returns chunk seq + text for a hash where vec IS NULL.
func (s *Store) UnembeddedChunks(hash string) ([]struct {
	Seq  int
	Text string
}, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var chunks []struct {
		Seq  int
		Text string
	}
	err = sqlitex.Execute(conn,
		`SELECT seq, text FROM content_vectors WHERE hash = ? AND (vec IS NULL OR fp <> ?) ORDER BY seq`,
		&sqlitex.ExecOptions{
			Args: []any{hash, embedFingerprint()},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				chunks = append(chunks, struct {
					Seq  int
					Text string
				}{
					Seq: stmt.ColumnInt(0), Text: stmt.ColumnText(1),
				})
				return nil
			},
		})
	return chunks, err
}

// EmbeddingsForDocIDs returns all stored embedding vectors for the given DocIDs.
// Used by stub platforms for precomputed-embedding search (centroid trick).
func (s *Store) EmbeddingsForDocIDs(docIDs []string) ([][]float32, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var vecs [][]float32
	for _, docID := range docIDs {
		var hash string
		err := sqlitex.Execute(conn, `SELECT hash FROM documents WHERE docid = ? AND active = 1 LIMIT 1`,
			&sqlitex.ExecOptions{
				Args: []any{docID},
				ResultFunc: func(stmt *sqlite.Stmt) error {
					hash = stmt.ColumnText(0)
					return nil
				},
			})
		if err != nil || hash == "" {
			continue
		}

		err = sqlitex.Execute(conn, `SELECT vec FROM content_vectors WHERE hash = ? AND vec IS NOT NULL`,
			&sqlitex.ExecOptions{
				Args: []any{hash},
				ResultFunc: func(stmt *sqlite.Stmt) error {
					vecLen := stmt.ColumnLen(0)
					if vecLen > 4 { // skip dummy placeholders (4 bytes)
						raw := make([]byte, vecLen)
						stmt.ColumnBytes(0, raw)
						vecs = append(vecs, bytesToFloat32(raw))
					}
					return nil
				},
			})
		if err != nil {
			continue
		}
	}
	return vecs, nil
}

// ---------------------------------------------------------------------------
// Markdown-aware chunker — preserves document structure
// ---------------------------------------------------------------------------

func breakScore(line string) int {
	trimmed := strings.TrimSpace(line)
	switch {
	case strings.HasPrefix(trimmed, "# "):
		return 100
	case strings.HasPrefix(trimmed, "## "):
		return 90
	case strings.HasPrefix(trimmed, "### ") || strings.HasPrefix(trimmed, "```"):
		return 80
	case strings.HasPrefix(trimmed, "#### "):
		return 70
	case strings.HasPrefix(trimmed, "##### "):
		return 60
	case strings.HasPrefix(trimmed, "---") || strings.HasPrefix(trimmed, "***"):
		return 60
	case strings.HasPrefix(trimmed, "###### "):
		return 50
	case trimmed == "":
		return 20
	case strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") || orderedListRe.MatchString(trimmed):
		return 5
	default:
		return 1
	}
}

// ChunkDocument splits markdown content into ~chunkTarget token pieces
// at structurally meaningful boundaries with 15% overlap between chunks.
func ChunkDocument(content string) []Chunk {
	lines := strings.Split(content, "\n")
	hash := contentHash(content)

	tokenEstimate := func(s string) int { return utf8.RuneCountInString(s) / 4 }

	var chunks []Chunk
	var buf strings.Builder
	var seq, pos int
	tokens := 0
	inCodeBlock := false
	var overlapPrefix string

	flush := func() {
		text := strings.TrimSpace(buf.String())
		if text != "" {
			chunks = append(chunks, Chunk{Hash: hash, Seq: seq, Pos: pos, Text: text})
			seq++
		}
		buf.Reset()
		tokens = 0
	}

	overlapFromEnd := func(allLines []string) string {
		tokCount := 0
		startIdx := len(allLines)
		for j := len(allLines) - 1; j >= 0; j-- {
			lt := tokenEstimate(allLines[j])
			tokCount += lt
			if tokCount >= chunkOverlap {
				startIdx = j
				break
			}
			startIdx = j
		}
		if startIdx >= len(allLines) {
			return ""
		}
		return strings.Join(allLines[startIdx:], "\n")
	}

	isInsideCodeFence := func(lookbackLines []string, idx int) bool {
		fence := false
		for j := 0; j < idx; j++ {
			if strings.HasPrefix(strings.TrimSpace(lookbackLines[j]), "```") {
				fence = !fence
			}
		}
		return fence
	}

	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			inCodeBlock = !inCodeBlock
		}

		lineTokens := tokenEstimate(line)
		tokens += lineTokens

		if tokens >= chunkTarget && !inCodeBlock {
			bestScore := 0
			bestOffset := 0

			lookbackLines := strings.Split(buf.String(), "\n")
			lookbackStart := len(lookbackLines) - (chunkLookback / 4)
			if lookbackStart < 1 {
				lookbackStart = 1
			}

			for j := lookbackStart; j < len(lookbackLines); j++ {
				if isInsideCodeFence(lookbackLines, j) {
					continue
				}
				sc := breakScore(lookbackLines[j])
				dist := float64(len(lookbackLines)-j) / float64(len(lookbackLines)-lookbackStart+1)
				adjusted := float64(sc) * (1.0 - dist*dist*0.7)
				if adjusted >= float64(bestScore) {
					bestScore = int(adjusted)
					bestOffset = j
				}
			}

			if bestScore > 1 && bestOffset > 0 {
				keep := strings.Join(lookbackLines[:bestOffset], "\n")
				remainder := strings.Join(lookbackLines[bestOffset:], "\n")

				buf.Reset()
				buf.WriteString(keep)
				keepLines := lookbackLines[:bestOffset]
				overlapPrefix = overlapFromEnd(keepLines)
				flush()
				pos = i - (len(lookbackLines) - bestOffset)
				if overlapPrefix != "" {
					buf.WriteString(overlapPrefix)
					buf.WriteString("\n")
				}
				buf.WriteString(remainder)
				buf.WriteString("\n")
				buf.WriteString(line)
				tokens = tokenEstimate(buf.String())
				continue
			}

			buf.WriteString(line)
			buf.WriteString("\n")
			allLines := strings.Split(buf.String(), "\n")
			overlapPrefix = overlapFromEnd(allLines)
			flush()
			pos = i + 1
			if overlapPrefix != "" {
				buf.WriteString(overlapPrefix)
				buf.WriteString("\n")
				tokens = tokenEstimate(buf.String())
			}
			continue
		}

		if buf.Len() == 0 {
			pos = i
			if overlapPrefix != "" {
				buf.WriteString(overlapPrefix)
				buf.WriteString("\n")
				tokens = tokenEstimate(buf.String())
				overlapPrefix = ""
			}
		}
		buf.WriteString(line)
		buf.WriteString("\n")
	}
	flush()
	return chunks
}

// ---------------------------------------------------------------------------
// MCP Server — stdio + HTTP transports
// ---------------------------------------------------------------------------

type MCPServer struct {
	store  *Store
	engine Embedder
	hybrid Searcher
	config *Config
}

type MCPRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type MCPResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id"`
	Result  any    `json:"result,omitempty"`
	Error   any    `json:"error,omitempty"`
}

func NewMCPServer(store *Store, engine Embedder, config *Config) *MCPServer {
	return &MCPServer{
		store:  store,
		engine: engine,
		hybrid: newHybridSearcher(store, engine),
		config: config,
	}
}

func (m *MCPServer) ServeHTTP(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /mcp", m.handleMCP)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		cols, docs, chunks, _ := m.store.Stats()
		json.NewEncoder(w).Encode(map[string]any{
			"status":      "ok",
			"collections": cols,
			"documents":   docs,
			"chunks":      chunks,
		})
	})

	log.Printf("picoqmd MCP server listening on %s", addr)
	return http.ListenAndServe(addr, mux)
}

func (m *MCPServer) handleMCP(w http.ResponseWriter, r *http.Request) {
	var req MCPRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", 400)
		return
	}

	resp := m.dispatch(req)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (m *MCPServer) ServeStdio() error {
	dec := json.NewDecoder(os.Stdin)
	enc := json.NewEncoder(os.Stdout)

	for {
		var req MCPRequest
		if err := dec.Decode(&req); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		resp := m.dispatch(req)
		if err := enc.Encode(resp); err != nil {
			return err
		}
	}
}

func (m *MCPServer) dispatch(req MCPRequest) MCPResponse {
	switch req.Method {
	case "initialize":
		return MCPResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{
			"protocolVersion": "2025-03-26",
			"capabilities": map[string]any{
				"tools": map[string]any{},
			},
			"serverInfo": map[string]any{
				"name":    "picoqmd",
				"version": version,
			},
			"instructions": m.buildInstructions(),
		}}

	case "tools/list":
		return MCPResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{
			"tools": m.toolDefinitions(),
		}}

	case "tools/call":
		result, err := m.callTool(req.Params)
		if err != nil {
			return MCPResponse{JSONRPC: "2.0", ID: req.ID, Error: map[string]any{
				"code": -1, "message": err.Error(),
			}}
		}
		return MCPResponse{JSONRPC: "2.0", ID: req.ID, Result: result}

	default:
		return MCPResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{}}
	}
}

func (m *MCPServer) buildInstructions() string {
	cols, docs, chunks, _ := m.store.Stats()
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("PicoQMD local search engine over %d documents in %d collections (%d chunks).\n\n", docs, cols, chunks))
	sb.WriteString("Collections:\n")
	for _, c := range m.config.Collections {
		sb.WriteString(fmt.Sprintf("  - %q: %s\n", c.Name, c.Context))
	}
	hasEmbed := modelExists(m.engine.ModelsDir(), defaultModels[0].Filename)
	if hasEmbed {
		sb.WriteString("\nTools: search (BM25), vector_search (semantic), deep_search (hybrid), research (composite BM25+vector), get, multi_get, status\n")
		sb.WriteString("\nPrefer `research` over calling search + vector_search separately — it deduplicates and merges server-side.\n")
		sb.WriteString("All search tools support `maxChars` to cap response size and `note` to save an observation linked to the top result.\n")
	} else {
		sb.WriteString("\nTools: search (BM25), get, multi_get, status\n")
		sb.WriteString("Note: Running in BM25-only mode. Install embedding models for vector/hybrid/research.\n")
	}
	return sb.String()
}

func (m *MCPServer) toolDefinitions() []map[string]any {
	intentDesc := "Optional purpose/disambiguation hint that guides expansion, reranking, and snippet selection. Empty = unchanged behavior."
	searchSchema := map[string]any{"type": "object", "properties": map[string]any{
		"query":      map[string]any{"type": "string", "description": "Search query"},
		"intent":     map[string]any{"type": "string", "description": intentDesc},
		"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
		"collection": map[string]any{"type": "string", "description": "Filter to a collection, or a comma-separated list (e.g. \"shared-src,ios-src\") to search several and merge without crowding"},
		"minScore":   map[string]any{"type": "number", "description": "Minimum score threshold (default 0)"},
		"maxChars":   map[string]any{"type": "integer", "description": "Truncate total response to this many characters (server-side token budget)"},
		"note":       map[string]any{"type": "string", "description": "Save an observation linked to the top result (persisted across sessions, auto-flagged stale when source changes)"},
	}, "required": []string{"query"}}

	tools := []map[string]any{
		{"name": "search", "description": "BM25 keyword search — finds documents containing exact words and phrases",
			"inputSchema": searchSchema},
	}

	hasEmbed := modelExists(m.engine.ModelsDir(), defaultModels[0].Filename)
	if hasEmbed {
		tools = append(tools, map[string]any{
			"name": "vector_search", "description": "Semantic vector search — finds related concepts even when exact words differ",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"query":      map[string]any{"type": "string", "description": "Search query"},
				"intent":     map[string]any{"type": "string", "description": intentDesc},
				"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
				"collection": map[string]any{"type": "string", "description": "Filter to a collection, or a comma-separated list (e.g. \"shared-src,ios-src\") to search several and merge without crowding"},
				"minScore":   map[string]any{"type": "number", "description": "Minimum score threshold (default 0.3)"},
				"maxChars":   map[string]any{"type": "integer", "description": "Truncate total response to this many characters"},
				"note":       map[string]any{"type": "string", "description": "Save an observation linked to the top result"},
			}, "required": []string{"query"}}})
		tools = append(tools, map[string]any{
			"name": "deep_search", "description": "Full hybrid pipeline: auto-expands query into variations, searches each by keyword and meaning, reranks for top hits",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"query":      map[string]any{"type": "string", "description": "Search query"},
				"intent":     map[string]any{"type": "string", "description": intentDesc},
				"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
				"collection": map[string]any{"type": "string", "description": "Filter to a collection, or a comma-separated list (e.g. \"shared-src,ios-src\") to search several and merge without crowding"},
				"minScore":   map[string]any{"type": "number", "description": "Minimum score threshold (default 0)"},
				"maxChars":   map[string]any{"type": "integer", "description": "Truncate total response to this many characters (server-side token budget)"},
				"note":       map[string]any{"type": "string", "description": "Save an observation linked to the top result (persisted across sessions, auto-flagged stale when source changes)"},
				"noExpand":   map[string]any{"type": "boolean", "description": "Skip the LLM query-expansion stage even if the strong-signal probe would otherwise run it. Useful for benchmarking."},
				"noRerank":   map[string]any{"type": "boolean", "description": "Skip the LLM reranking stage and return RRF-fused order directly. Faster on constrained hardware."},
			}, "required": []string{"query"}}})
		tools = append(tools, map[string]any{
			"name": "research", "description": "Composite search: runs BM25 + vector in parallel, deduplicates by docid via RRF, and merges within a token budget. One call instead of two.",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"query":      map[string]any{"type": "string", "description": "Search query"},
				"intent":     map[string]any{"type": "string", "description": intentDesc},
				"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
				"collection": map[string]any{"type": "string", "description": "Filter to a collection, or a comma-separated list (e.g. \"shared-src,ios-src\") to search several and merge without crowding"},
				"minScore":   map[string]any{"type": "number", "description": "Minimum score threshold (default 0)"},
				"maxChars":   map[string]any{"type": "integer", "description": "Truncate total response to this many characters (default: no limit)"},
				"note":       map[string]any{"type": "string", "description": "Save an observation linked to the top result"},
			}, "required": []string{"query"}}})
	}

	tools = append(tools,
		map[string]any{"name": "get", "description": "Retrieve a single document by path or docid (#abc123). Supports line-range suffixes: file.md:120 (from line 120) and file.md:120:40 (40 lines from line 120). Output is line-numbered with a qmd:// + #docid header.",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"ref":         map[string]any{"type": "string", "description": "File path, docid (#abc123), or qmd:// URI — optionally with :from or :from:count line-range suffix"},
				"fromLine":    map[string]any{"type": "integer", "description": "1-based first line to return (overrides ref suffix)"},
				"maxLines":    map[string]any{"type": "integer", "description": "Maximum lines to return"},
				"lineNumbers": map[string]any{"type": "boolean", "description": "Prefix each line with its line number (default true)"},
				"fullPath":    map[string]any{"type": "boolean", "description": "Use the on-disk filesystem path in the header instead of the qmd:// URI (handy for piping into file tools)"},
				"maxChars":    map[string]any{"type": "integer", "description": "Truncate response to this many characters"},
			}, "required": []string{"ref"}}},
		map[string]any{"name": "multi_get", "description": "Retrieve multiple documents by glob pattern or comma-separated list. Line-numbered output; oversized or unreadable files are reported as skipped rather than silently dropped.",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"pattern":     map[string]any{"type": "string", "description": "Glob pattern (e.g., docs/*.md) or comma-separated paths"},
				"maxBytes":    map[string]any{"type": "integer", "description": "Skip files over this size in bytes (default 65536)"},
				"lineNumbers": map[string]any{"type": "boolean", "description": "Prefix each line with its line number (default true)"},
				"fullPath":    map[string]any{"type": "boolean", "description": "Use on-disk filesystem paths in headers instead of qmd:// URIs"},
				"maxChars":    map[string]any{"type": "integer", "description": "Truncate total response to this many characters"},
			}, "required": []string{"pattern"}}},
		map[string]any{"name": "status", "description": "Index health: collection inventory, document counts, embedding status",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{}}},
	)

	return tools
}

func (m *MCPServer) callTool(params json.RawMessage) (any, error) {
	var call struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(params, &call); err != nil {
		return nil, err
	}

	var args struct {
		Query       string  `json:"query"`
		Intent      string  `json:"intent"`
		Ref         string  `json:"ref"`
		Pattern     string  `json:"pattern"`
		Limit       int     `json:"limit"`
		Collection  string  `json:"collection"`
		MinScore    float64 `json:"minScore"`
		MaxBytes    int64   `json:"maxBytes"`
		MaxLines    int     `json:"maxLines"`
		MaxChars    int     `json:"maxChars"`
		Note        string  `json:"note"`
		NoExpand    bool    `json:"noExpand"`
		NoRerank    bool    `json:"noRerank"`
		FromLine    int     `json:"fromLine"`
		LineNumbers *bool   `json:"lineNumbers"`
		FullPath    bool    `json:"fullPath"`
	}
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		return nil, fmt.Errorf("invalid tool arguments: %w", err)
	}
	if args.Limit == 0 {
		args.Limit = 10
	}
	if args.MaxBytes == 0 {
		args.MaxBytes = 65536
	}
	lineNumbers := args.LineNumbers == nil || *args.LineNumbers

	filterMinScore := func(results []SearchResult, minScore float64) []SearchResult {
		if minScore <= 0 {
			return results
		}
		var filtered []SearchResult
		for _, r := range results {
			if r.Score >= minScore {
				filtered = append(filtered, r)
			}
		}
		return filtered
	}

	// Helper: apply maxChars truncation to a tool result
	applyMaxChars := func(result map[string]any, maxChars int) map[string]any {
		if maxChars <= 0 {
			return result
		}
		content, ok := result["content"].([]map[string]any)
		if !ok || len(content) == 0 {
			return result
		}
		text, ok := content[0]["text"].(string)
		if !ok || len(text) <= maxChars {
			return result
		}
		content[0]["text"] = text[:maxChars] + "\n\n[... truncated to " + fmt.Sprintf("%d", maxChars) + " chars]"
		return result
	}

	// Helper: save observation note linked to a docid
	saveNote := func(results []SearchResult, note string) {
		if note == "" || len(results) == 0 {
			return
		}
		top := results[0]
		obs := Observation{
			DocID:     top.DocID,
			Path:      top.Path,
			Hash:      m.getDocHash(top.DocID),
			Note:      note,
			Timestamp: fmt.Sprintf("%d", time.Now().Unix()),
		}
		saveObservation(m.observationsPath(), obs)
	}

	switch call.Name {
	case "search":
		results, err := m.store.SearchBM25Scoped(args.Query, args.Collection, args.Limit)
		if err != nil {
			return nil, err
		}
		filtered := filterMinScore(results, args.MinScore)
		saveNote(filtered, args.Note)
		return applyMaxChars(toolResult(filtered), args.MaxChars), nil

	case "vector_search":
		qvec, err := m.engine.Embed(args.Query, true)
		if err != nil {
			return nil, err
		}
		snippetText := combineForSnippet(args.Query, args.Intent)
		results, err := m.store.SearchVectorScoped(snippetText, qvec, args.Collection, args.Limit)
		if err != nil {
			return nil, err
		}
		minScore := args.MinScore
		if minScore == 0 {
			minScore = 0.3
		}
		filtered := filterMinScore(results, minScore)
		saveNote(filtered, args.Note)
		return applyMaxChars(toolResult(filtered), args.MaxChars), nil

	case "deep_search":
		if t, ok := m.hybrid.(interface{ SetNoExpand(bool) }); ok {
			t.SetNoExpand(args.NoExpand)
		}
		if t, ok := m.hybrid.(interface{ SetNoRerank(bool) }); ok {
			t.SetNoRerank(args.NoRerank)
		}
		results, err := m.hybrid.Search(context.Background(), args.Query, args.Intent, args.Collection, args.Limit)
		if err != nil {
			return nil, err
		}
		filtered := filterMinScore(results, args.MinScore)
		saveNote(filtered, args.Note)
		return applyMaxChars(toolResult(filtered), args.MaxChars), nil

	case "research":
		// Composite: BM25 + vector search, deduplicated via RRF
		bm25Results, _ := m.store.SearchBM25Scoped(args.Query, args.Collection, args.Limit*2)
		var vecResults []SearchResult
		if qvec, err := m.engine.Embed(args.Query, true); err == nil {
			snippetText := combineForSnippet(args.Query, args.Intent)
			vecResults, _ = m.store.SearchVectorScoped(snippetText, qvec, args.Collection, args.Limit*2)
		}
		merged := simpleRRF(bm25Results, vecResults, args.Limit)
		filtered := filterMinScore(merged, args.MinScore)
		// Attach stale observations to results
		stale := getStaleObservations(m.observationsPath(), m.store)
		if len(stale) > 0 {
			staleMap := make(map[string]string)
			for _, s := range stale {
				staleMap[s.DocID] = s.Note
			}
			for i, r := range filtered {
				if note, ok := staleMap[r.DocID]; ok {
					filtered[i].Context += " [STALE observation: " + note + "]"
				}
			}
		}
		saveNote(filtered, args.Note)
		return applyMaxChars(toolResult(filtered), args.MaxChars), nil

	case "get":
		doc, from, count, err := m.store.GetDocument(args.Ref)
		if err != nil {
			return nil, err
		}
		if args.FromLine > 0 {
			from = args.FromLine // explicit param overrides ref suffix
		}
		if args.MaxLines > 0 && count == 0 {
			count = args.MaxLines
		}
		return applyMaxChars(textResult(renderDocument(doc, from, count, lineNumbers, args.FullPath)), args.MaxChars), nil

	case "multi_get":
		docs, skipped, err := m.store.MultiGet(args.Pattern, args.MaxBytes)
		if err != nil {
			return nil, err
		}
		var b strings.Builder
		for i := range docs {
			if i > 0 {
				b.WriteString("\n")
			}
			b.WriteString(renderDocument(&docs[i], 0, 0, lineNumbers, args.FullPath))
		}
		for _, sk := range skipped {
			if sk.Size < 0 {
				fmt.Fprintf(&b, "\n[skipped %s: unreadable on disk — run 'picoqmd sync']", sk.Path)
			} else {
				fmt.Fprintf(&b, "\n[skipped %s: %d bytes > maxBytes %d]", sk.Path, sk.Size, args.MaxBytes)
			}
		}
		if len(docs) == 0 && len(skipped) == 0 {
			b.WriteString("no documents match: " + args.Pattern)
		}
		return applyMaxChars(textResult(b.String()), args.MaxChars), nil

	case "status":
		cols, docs, chunks, _ := m.store.Stats()
		pending, _ := m.store.CountUnembedded(embedFingerprint(), "")
		stale := getStaleObservations(m.observationsPath(), m.store)
		return toolResult(map[string]any{
			"collections":       cols,
			"documents":         docs,
			"chunks":            chunks,
			"needsEmbedding":    pending,
			"hasVectorIndex":    pending == 0 && chunks > 0,
			"fingerprint":       embedFingerprint(),
			"staleObservations": len(stale),
		}), nil

	default:
		return nil, fmt.Errorf("unknown tool: %s", call.Name)
	}
}

// ---------------------------------------------------------------------------
// Observations — persistent notes linked to document docids
// ---------------------------------------------------------------------------

type Observation struct {
	DocID     string `json:"docid"`
	Path      string `json:"path"`
	Hash      string `json:"hash"` // content hash at time of observation
	Note      string `json:"note"`
	Timestamp string `json:"timestamp"`
	Stale     bool   `json:"stale,omitempty"`
}

func (m *MCPServer) observationsPath() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".config")
	}
	return filepath.Join(dir, "picoqmd", "observations.json")
}

func (m *MCPServer) getDocHash(docid string) string {
	doc, err := m.store.lookupDocument(docid)
	if err != nil {
		return ""
	}
	return doc.Hash
}

func saveObservation(path string, obs Observation) {
	var observations []Observation
	if data, err := os.ReadFile(path); err == nil {
		json.Unmarshal(data, &observations)
	}
	// Update existing observation for same docid, or append
	found := false
	for i, o := range observations {
		if o.DocID == obs.DocID {
			observations[i] = obs
			found = true
			break
		}
	}
	if !found {
		observations = append(observations, obs)
	}
	os.MkdirAll(filepath.Dir(path), 0o755)
	data, _ := json.MarshalIndent(observations, "", "  ")
	os.WriteFile(path, data, 0o644)
}

func getStaleObservations(path string, store *Store) []Observation {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var observations []Observation
	if err := json.Unmarshal(data, &observations); err != nil {
		return nil
	}
	var stale []Observation
	for _, obs := range observations {
		doc, err := store.lookupDocument(obs.DocID)
		if err != nil {
			// Document deleted — observation is stale
			obs.Stale = true
			stale = append(stale, obs)
			continue
		}
		if doc.Hash != obs.Hash {
			obs.Stale = true
			stale = append(stale, obs)
		}
	}
	return stale
}

func toolResult(data any) map[string]any {
	b, _ := json.Marshal(data)
	return textResult(string(b))
}

// textResult wraps already-formatted text as an MCP tool result without
// JSON-encoding it a second time.
func textResult(text string) map[string]any {
	return map[string]any{
		"content": []map[string]any{
			{"type": "text", "text": text},
		},
	}
}

// ---------------------------------------------------------------------------
// Collection indexing
// ---------------------------------------------------------------------------

func indexCollection(store *Store, col CollectionConfig) error {
	absPath, err := filepath.Abs(col.Path)
	if err != nil {
		return err
	}

	glob := col.Glob
	if glob == "" {
		glob = "**/*.md"
	}

	patterns, err := expandGlob(glob)
	if err != nil {
		return fmt.Errorf("bad glob %q: %w", glob, err)
	}

	colID, err := store.UpsertCollection(col.Name, absPath, glob, col.Context)
	if err != nil {
		return err
	}

	activePaths := make(map[string]bool)
	var count int
	err = filepath.WalkDir(absPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if d.IsDir() {
			name := d.Name()
			if name != "." && strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			if skipDirs[name] {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, _ := filepath.Rel(absPath, path)
		if !matchesAny(patterns, relPath) {
			return nil
		}

		info, err := d.Info()
		if err != nil || info.Size() > maxIndexFileBytes {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if isBinary(content) {
			return nil
		}

		activePaths[relPath] = true
		title := extractTitle(string(content), relPath)
		if err := store.UpsertDocument(colID, relPath, title, string(content)); err != nil {
			log.Printf("  skip %s: %v", relPath, err)
			return nil
		}
		count++
		return nil
	})
	if err != nil {
		return err
	}

	if err := store.DeactivateStale(colID, activePaths); err != nil {
		log.Printf("  warning: deactivate stale: %v", err)
	}

	fmt.Printf("  Indexed %d documents from %q\n", count, col.Name)
	return nil
}

// simpleRRF merges BM25 and vector results using reciprocal rank fusion.
func simpleRRF(bm25, vec []SearchResult, limit int) []SearchResult {
	scores := make(map[string]float64)
	docs := make(map[string]SearchResult)

	for rank, r := range bm25 {
		scores[r.DocID] += 1.0 / float64(rrfK+rank+1)
		docs[r.DocID] = r
	}
	for rank, r := range vec {
		scores[r.DocID] += 1.0 / float64(rrfK+rank+1)
		if _, ok := docs[r.DocID]; !ok {
			docs[r.DocID] = r
		}
	}

	type entry struct {
		docID string
		score float64
	}
	var entries []entry
	for id, sc := range scores {
		entries = append(entries, entry{id, sc})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].score > entries[j].score })

	var results []SearchResult
	for _, e := range entries {
		r := docs[e.docID]
		r.Score = e.score
		results = append(results, r)
		if len(results) >= limit {
			break
		}
	}
	return results
}

// ---------------------------------------------------------------------------
// Remote proxy — forward searches to a remote picoqmd MCP server
// ---------------------------------------------------------------------------

func remoteSearch(query, addr string, limit int, format string) error {
	mcpReq := MCPRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "tools/call",
		Params:  json.RawMessage(fmt.Sprintf(`{"name":"search","arguments":{"query":%q,"limit":%d}}`, query, limit)),
	}
	body, err := json.Marshal(mcpReq)
	if err != nil {
		return err
	}

	url := addr
	if !strings.Contains(url, "://") {
		url = "http://" + url
	}
	if !strings.HasSuffix(url, "/mcp") {
		url += "/mcp"
	}

	resp, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("remote search: %w", err)
	}
	defer resp.Body.Close()

	var mcpResp MCPResponse
	if err := json.NewDecoder(resp.Body).Decode(&mcpResp); err != nil {
		return fmt.Errorf("remote search: %w", err)
	}

	if mcpResp.Error != nil {
		return fmt.Errorf("remote error: %v", mcpResp.Error)
	}

	// Parse MCP tool result → content[0].text → []SearchResult
	resultMap, ok := mcpResp.Result.(map[string]any)
	if !ok {
		return fmt.Errorf("unexpected response format")
	}
	content, _ := resultMap["content"].([]any)
	if len(content) == 0 {
		fmt.Println("no results")
		return nil
	}
	textObj, _ := content[0].(map[string]any)
	text, _ := textObj["text"].(string)

	var results []SearchResult
	if err := json.Unmarshal([]byte(text), &results); err != nil {
		fmt.Println(text)
		return nil
	}

	return printResults(results, format)
}

// ---------------------------------------------------------------------------
// Config management
// ---------------------------------------------------------------------------

func configDir() string {
	if d := os.Getenv("XDG_CONFIG_HOME"); d != "" {
		return filepath.Join(d, "picoqmd")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "picoqmd")
}

func cacheDir() string {
	if d := os.Getenv("XDG_CACHE_HOME"); d != "" {
		return filepath.Join(d, "picoqmd")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "picoqmd")
}

func loadConfig(indexName string) (*Config, string, error) {
	dir := configDir()
	os.MkdirAll(dir, 0o755)

	name := "index"
	if indexName != "" {
		name = indexName
	}
	path := filepath.Join(dir, name+".yml")

	var cfg Config
	data, err := os.ReadFile(path)
	if err == nil {
		if yerr := yaml.Unmarshal(data, &cfg); yerr != nil {
			return nil, path, fmt.Errorf("parse config %s: %w", path, yerr)
		}
	}
	return &cfg, path, nil
}

func saveConfig(cfg *Config, path string) error {
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func dbPath(indexName string) string {
	name := "index"
	if indexName != "" {
		name = indexName
	}
	return filepath.Join(cacheDir(), name+".sqlite")
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// embedFingerprint identifies the embedding model + chunker that produced a
// vector. Vectors whose stored fingerprint differs are treated as pending so
// a model or chunker change triggers re-embedding instead of silently
// searching stale embeddings.
//
// The fingerprint includes a hash of the model FILE BYTES, not just its name:
// on 2026-08-29 a re-downloaded GGUF with the same filename silently
// invalidated every stored vector (searches returned cosine ~0.05 noise)
// while the name-only fingerprint saw no change.
var (
	embedFPOnce   sync.Once
	embedFPCached string
)

func embedFingerprint() string {
	embedFPOnce.Do(func() {
		name := defaultModels[0].Filename
		if h := modelFileHash(filepath.Join(cacheDir(), "models", name)); h != "" {
			embedFPCached = name + "@" + h + "|" + chunkerVersion
		} else {
			// Model not downloaded (BM25-only install): legacy name-only
			// form. Embedding requires the model, so no vector is ever
			// written under this fallback.
			embedFPCached = name + "|" + chunkerVersion
		}
		if dim := embedTargetDim(); dim > 0 {
			embedFPCached += fmt.Sprintf("|d%d", dim)
		}
	})
	return embedFPCached
}

// embedTargetDim returns the Matryoshka truncation dimension for embeddings.
// EmbeddingGemma is MRL-trained, so keeping the first N dims and
// L2-renormalizing is the sanctioned way to get an N-dim embedding: 256 dims
// keep ~97.6% of MTEB quality at 3x smaller storage and 3x faster brute-force
// scans. Override with PICOQMD_EMBED_DIM (0 = full model dimension).
func embedTargetDim() int {
	if v := os.Getenv("PICOQMD_EMBED_DIM"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return 256
}

// truncateMRL truncates a Matryoshka-trained embedding to dim and
// L2-renormalizes. No-op when dim is 0 or the vector is already <= dim.
func truncateMRL(vec []float32, dim int) []float32 {
	if dim <= 0 || len(vec) <= dim {
		return vec
	}
	out := make([]float32, dim)
	var norm float64
	for i := 0; i < dim; i++ {
		norm += float64(vec[i]) * float64(vec[i])
	}
	if norm == 0 {
		copy(out, vec[:dim])
		return out
	}
	inv := float32(1.0 / math.Sqrt(norm))
	for i := 0; i < dim; i++ {
		out[i] = vec[i] * inv
	}
	return out
}

// modelFileHash returns the first 16 hex chars of the model file's sha256.
// A sidecar cache (<model>.sha256: "size|mtime|hash") avoids re-hashing
// ~300MB on every invocation; it is refreshed whenever size or mtime change.
func modelFileHash(path string) string {
	st, err := os.Stat(path)
	if err != nil {
		return ""
	}
	sidecar := path + ".sha256"
	stamp := fmt.Sprintf("%d|%d|", st.Size(), st.ModTime().Unix())
	if b, err := os.ReadFile(sidecar); err == nil {
		if s := strings.TrimSpace(string(b)); strings.HasPrefix(s, stamp) {
			return strings.TrimPrefix(s, stamp)
		}
	}
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	sum := hex.EncodeToString(h.Sum(nil))[:16]
	os.WriteFile(sidecar, []byte(stamp+sum+"\n"), 0o644)
	return sum
}

func contentHash(content string) string {
	h := sha256.Sum256([]byte(content))
	return hex.EncodeToString(h[:])
}

var (
	titleRe       = regexp.MustCompile(`(?m)^#\s+(.+)$`)
	orderedListRe = regexp.MustCompile(`^\d+\.\s`)
)

func extractTitle(content, fallback string) string {
	m := titleRe.FindStringSubmatch(content)
	if len(m) > 1 {
		return strings.TrimSpace(m[1])
	}
	return strings.TrimSuffix(filepath.Base(fallback), filepath.Ext(fallback))
}

// snippetStopWords are common English fillers we don't anchor snippet windows
// or highlights on. Mirrors the FTS5 default stoplist loosely.
var snippetStopWords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "but": true,
	"of": true, "in": true, "on": true, "at": true, "to": true, "for": true,
	"is": true, "are": true, "was": true, "were": true, "be": true,
	"by": true, "with": true, "from": true, "as": true, "it": true,
	"this": true, "that": true, "these": true, "those": true,
	"i": true, "you": true, "he": true, "she": true, "we": true, "they": true,
}

// queryTerms tokenizes a free-text query into stoplist-filtered terms suitable
// for snippet anchoring and highlighting.
func queryTerms(query string) []string {
	if query == "" {
		return nil
	}
	fields := strings.Fields(query)
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		f = strings.Trim(f, ".,;:!?'\"()[]{}*")
		if len(f) < 2 {
			continue
		}
		if snippetStopWords[strings.ToLower(f)] {
			continue
		}
		out = append(out, f)
	}
	return out
}

// extractSnippet picks a window around the first occurrence of any `term` in
// `text` and highlights all term occurrences inside the window with
// >>>...<<< (matching the FTS5 BM25 snippet format already used elsewhere).
// If no terms are found, returns the head of the text. Returns the snippet and
// the byte offset where the window starts within `text`.
// extractSnippet returns a highlighted window around the first term hit plus
// citePos, the byte offset of that hit (window start when no term matches) —
// suitable for line-number citations.
func extractSnippet(text string, terms []string, windowChars int) (snippet string, citePos int) {
	if len(text) == 0 || windowChars <= 0 {
		return "", 0
	}
	lower := strings.ToLower(text)

	firstHit := -1
	for _, term := range terms {
		t := strings.ToLower(term)
		if len(t) < 2 {
			continue
		}
		if i := strings.Index(lower, t); i >= 0 {
			if firstHit < 0 || i < firstHit {
				firstHit = i
			}
		}
	}

	if firstHit < 0 {
		// No hits — return the head.
		if len(text) <= windowChars {
			return text, 0
		}
		end := windowChars
		for end < len(text) && !isSnippetBoundary(text[end]) {
			end++
		}
		return text[:end] + "...", 0
	}

	// Center the window around firstHit, biased a bit toward earlier text so
	// the term appears past the visual middle.
	start := firstHit - windowChars/3
	if start < 0 {
		start = 0
	}
	end := start + windowChars
	if end > len(text) {
		end = len(text)
		start = end - windowChars
		if start < 0 {
			start = 0
		}
	}

	// Align to word boundaries so we don't slice through tokens.
	for start > 0 && !isSnippetBoundary(text[start-1]) {
		start--
	}
	for end < len(text) && !isSnippetBoundary(text[end]) {
		end++
	}

	window := text[start:end]
	for _, term := range terms {
		if len(term) < 2 {
			continue
		}
		window = highlightCI(window, term, ">>>", "<<<")
	}

	if start > 0 {
		window = "..." + window
	}
	if end < len(text) {
		window = window + "..."
	}
	return window, firstHit
}

func isSnippetBoundary(b byte) bool {
	switch b {
	case ' ', '\n', '\t', '.', ',', ';', ':', '!', '?', ')', ']', '}', '"':
		return true
	}
	return false
}

// highlightCI wraps every case-insensitive occurrence of `term` in `text`
// with `prefix`/`suffix`, preserving the original casing of the matched span.
func highlightCI(text, term, prefix, suffix string) string {
	if len(term) == 0 {
		return text
	}
	lowerText := strings.ToLower(text)
	lowerTerm := strings.ToLower(term)
	tlen := len(term)

	var b strings.Builder
	b.Grow(len(text) + 8)
	i := 0
	for i < len(text) {
		idx := strings.Index(lowerText[i:], lowerTerm)
		if idx < 0 {
			b.WriteString(text[i:])
			break
		}
		b.WriteString(text[i : i+idx])
		b.WriteString(prefix)
		b.WriteString(text[i+idx : i+idx+tlen])
		b.WriteString(suffix)
		i += idx + tlen
	}
	return b.String()
}

// computeLineNumber returns the 1-based line number at byte offset `pos`
// within the file at `absPath`. Returns 0 if the file can't be read or pos is
// out of range — callers treat 0 as "unknown".
func computeLineNumber(absPath string, pos int) int {
	if absPath == "" || pos <= 0 {
		return 0
	}
	f, err := os.Open(absPath)
	if err != nil {
		return 0
	}
	defer f.Close()

	buf := make([]byte, pos)
	n, _ := io.ReadFull(f, buf)
	return bytes.Count(buf[:n], []byte("\n")) + 1
}

// ftsString renders a term as a double-quoted FTS5 string with embedded
// quotes doubled. Inside a string the tokenizer treats punctuation as plain
// separators, so "2026.4.10" becomes the phrase (2026 4 10), "real-time"
// becomes (real time), and operator words like AND lose their meaning.
// Returns "" for terms that contain no letters or digits (an empty phrase is
// an FTS5 syntax error).
func ftsString(term string) string {
	hasToken := false
	for _, r := range term {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			hasToken = true
			break
		}
	}
	if !hasToken {
		return ""
	}
	return `"` + strings.ReplaceAll(term, `"`, `""`) + `"`
}

// toFTS5Query converts a free-text query into FTS5 MATCH syntax that can
// never produce a syntax error: every term is emitted as a quoted string
// (see ftsString). Only the LAST unquoted term gets a trailing * — prefix
// expansion on every word made long natural-language queries scan the whole
// term index (`"the"*` over a book corpus enumerates a huge posting range
// per term), and porter stemming already matches morphological variants of
// interior terms. Stopword terms are dropped when at least one meaningful
// term remains: they carry ~zero BM25 weight (high document frequency) but
// cost a full doclist intersection each. User-quoted phrases are preserved
// verbatim. Returns "" when nothing searchable remains — callers must skip
// the MATCH entirely.
func toFTS5Query(query string) string {
	words := strings.Fields(query)

	type part struct {
		text      string // quoted FTS5 string, no star
		starrable bool   // unquoted term (user phrases never get a star)
		stopword  bool
	}
	var parts []part
	inPhrase := false
	var phrase []string
	hasContent := false // at least one non-stopword unquoted term seen

	appendTerm := func(w string) {
		w = strings.Trim(w, `"'`)
		if q := ftsString(w); q != "" {
			stop := snippetStopWords[strings.ToLower(w)]
			parts = append(parts, part{text: q, starrable: true, stopword: stop})
			if !stop {
				hasContent = true
			}
		}
	}
	appendPhrase := func(ws []string) {
		if q := ftsString(strings.Join(ws, " ")); q != "" {
			parts = append(parts, part{text: q})
		}
	}

	for _, w := range words {
		if !inPhrase && strings.HasPrefix(w, `"`) {
			inPhrase = true
			phrase = []string{strings.TrimPrefix(w, `"`)}
			if strings.HasSuffix(w, `"`) && len(w) > 1 {
				// Single-word quoted: "word"
				inPhrase = false
				phrase[0] = strings.TrimSuffix(phrase[0], `"`)
				appendPhrase(phrase)
			}
			continue
		}
		if inPhrase {
			if strings.HasSuffix(w, `"`) {
				phrase = append(phrase, strings.TrimSuffix(w, `"`))
				appendPhrase(phrase)
				inPhrase = false
			} else {
				phrase = append(phrase, w)
			}
			continue
		}
		appendTerm(w)
	}

	// Unclosed quote — treat remaining words as plain terms
	if inPhrase {
		for _, w := range phrase {
			appendTerm(w)
		}
	}

	// Drop stopword terms, but only when a meaningful term survives —
	// an all-stopword query ("to be or not to be") keeps everything.
	if hasContent {
		kept := parts[:0]
		for _, p := range parts {
			if !p.stopword {
				kept = append(kept, p)
			}
		}
		parts = kept
	}

	// Prefix-star the last unquoted term (search-as-you-type semantics).
	for i := len(parts) - 1; i >= 0; i-- {
		if parts[i].starrable {
			parts[i].text += "*"
			break
		}
	}

	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = p.text
	}
	return strings.Join(out, " AND ")
}

func cosineSim(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func float32ToBytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		bits := math.Float32bits(f)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToFloat32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		bits := uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
		v[i] = math.Float32frombits(bits)
	}
	return v
}

// isInteractive returns true if stdin is a terminal (not piped/redirected).
func isInteractive() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// addFileToTar adds a file to a tar writer with the given archive name.
func addFileToTar(tw *tar.Writer, filePath, name string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return err
	}

	hdr := &tar.Header{
		Name: name,
		Size: info.Size(),
		Mode: 0644,
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return err
	}

	_, err = io.Copy(tw, f)
	return err
}

// ---------------------------------------------------------------------------
// CLI — cobra commands
// ---------------------------------------------------------------------------

// quiet suppresses per-document progress output so picoqmd can run under
// launchd/cron without filling its captured-stdout log file. It is
// auto-enabled when stdout is not a terminal (pipes, files, launchd) and
// can be forced on/off via --quiet/--verbose. Read by helpers in llm.go
// and engine.go via the infof / progressEnabled accessors.
var quiet bool

// infof writes a progress line to stdout only when quiet is false.
// Use it for "this happened, here's the count" loop output.
// Reserve direct fmt.Print for one-shot final summaries that should
// always render (errors, totals).
func infof(format string, args ...any) {
	if !quiet {
		fmt.Printf(format, args...)
	}
}

// progressEnabled reports whether progress UI (download bars, redraws)
// should render. Always false when quiet is set.
func progressEnabled() bool { return !quiet }

// stdoutIsTerminal returns true when os.Stdout is attached to an
// interactive terminal. False under launchd, cron, pipes, or `tee >file`.
// We use the Stat mode rather than golang.org/x/term to avoid pulling
// in a new dependency for one boolean.
func stdoutIsTerminal() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

func main() {
	// Default --quiet to true when stdout is not a terminal so launchd /
	// cron / piped invocations don't grow log files unboundedly.
	// Users on a real terminal still see progress output.
	if !stdoutIsTerminal() {
		quiet = true
	}

	var indexName string
	var searchLimit int
	var searchFormat string
	var remoteAddr string
	var noExpand bool
	var noRerank bool
	var searchIntent string

	// --- smartSearch dispatches to the best available pipeline ---
	smartSearch := func(query, intent string, store *Store, engine Embedder, limit int, format string) error {
		hasEmbed := modelExists(engine.ModelsDir(), defaultModels[0].Filename)
		hasRerank := modelExists(engine.ModelsDir(), defaultModels[1].Filename)
		hasExpand := modelExists(engine.ModelsDir(), defaultModels[2].Filename)

		// Full hybrid: all 3 models available
		if hasEmbed && hasRerank && hasExpand {
			hybrid := newHybridSearcher(store, engine)
			if t, ok := hybrid.(interface{ SetNoExpand(bool) }); ok {
				t.SetNoExpand(noExpand)
			}
			if t, ok := hybrid.(interface{ SetNoRerank(bool) }); ok {
				t.SetNoRerank(noRerank)
			}
			results, err := hybrid.Search(context.Background(), query, intent, "", limit)
			if err != nil {
				return err
			}
			return printResults(results, format)
		}

		// BM25 + vector: embedding model only
		if hasEmbed {
			bm25Results, _ := store.SearchBM25(query, limit*2)
			qvec, err := engine.Embed(query, true)
			if err == nil {
				vecResults, _ := store.SearchVector(combineForSnippet(query, intent), qvec, limit*2)
				return printResults(simpleRRF(bm25Results, vecResults, limit), format)
			}
			if len(bm25Results) > limit {
				bm25Results = bm25Results[:limit]
			}
			return printResults(bm25Results, format)
		}

		// BM25 only: no models
		results, err := store.SearchBM25(query, limit)
		if err != nil {
			return err
		}
		return printResults(results, format)
	}

	root := &cobra.Command{
		Use:   "picoqmd [query]",
		Short: "Local markdown search engine — optimized Go port of QMD",
		Long: `picoqmd — a fully local search engine for markdown documents.

Quick start:
  picoqmd add ~/notes           Add, index, and embed a directory
  picoqmd "meeting notes"       Smart search (auto-selects best pipeline)
  picoqmd sync                  Re-index and re-embed changed files`,
		Version:            version,
		Args:               cobra.ArbitraryArgs,
		DisableFlagParsing: false,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 {
				return cmd.Help()
			}

			query := strings.Join(args, " ")

			// Remote proxy: forward to remote MCP server
			if remoteAddr != "" {
				return remoteSearch(query, remoteAddr, searchLimit, searchFormat)
			}

			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			engine := NewLLMEngine(cacheDir())
			return smartSearch(query, searchIntent, store, engine, searchLimit, searchFormat)
		},
	}
	root.PersistentFlags().StringVar(&indexName, "index", "", "named index (separate DB + config)")
	root.PersistentFlags().IntVar(&searchLimit, "limit", 10, "max search results")
	root.PersistentFlags().StringVar(&searchFormat, "format", "text", "output: text, json, csv, md, files")
	root.PersistentFlags().StringVar(&remoteAddr, "remote", "", "forward searches to remote picoqmd MCP server (host:port)")
	root.PersistentFlags().BoolVar(&quiet, "quiet", quiet, "suppress per-document progress output (auto-enabled when stdout is not a terminal)")
	root.PersistentFlags().BoolVar(new(bool), "verbose", false, "force progress output even when stdout is not a terminal")
	root.PersistentFlags().BoolVar(&noExpand, "no-expand", false, "skip the LLM query-expansion stage (forces strong-signal-only behavior)")
	root.PersistentFlags().BoolVar(&noRerank, "no-rerank", false, "skip the LLM reranking stage (RRF-fused order, faster on constrained hardware)")
	root.PersistentFlags().StringVar(&searchIntent, "intent", "", "optional disambiguation hint passed to expansion, reranking, and snippet selection")
	// --verbose overrides the auto-quiet behavior. We can't share a single
	// bool with --quiet (cobra rejects that), so we fix it up after parse:
	root.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
		if v, _ := cmd.Flags().GetBool("verbose"); v {
			quiet = false
		}
		return nil
	}

	// --- add (top-level) ---
	addRunE := func(cmd *cobra.Command, args []string) error {
		name, _ := cmd.Flags().GetString("name")
		glob, _ := cmd.Flags().GetString("glob")
		ctx, _ := cmd.Flags().GetString("context")
		noEmbed, _ := cmd.Flags().GetBool("no-embed")

		if name == "" {
			name = filepath.Base(args[0])
		}
		if glob == "" {
			glob = "**/*.md"
		}

		engine := NewLLMEngine(cacheDir())
		if !noEmbed && !modelExists(engine.ModelsDir(), defaultModels[0].Filename) {
			if isInteractive() {
				fmt.Println("Setup mode:")
				fmt.Println("  [1] BM25 only — instant keyword search, no downloads (~0MB)")
				fmt.Println("  [2] BM25 + vector — semantic search, downloads embedding model (~300MB)")
				fmt.Print("Choose [1/2] (default 2): ")
				var choice string
				fmt.Scanln(&choice)
				if strings.TrimSpace(choice) == "1" {
					noEmbed = true
				}
			}
		}

		cfg, cfgPath, err := loadConfig(indexName)
		if err != nil {
			return err
		}
		found := false
		for i, c := range cfg.Collections {
			if c.Name == name {
				cfg.Collections[i] = CollectionConfig{Name: name, Path: args[0], Glob: glob, Context: ctx}
				found = true
				break
			}
		}
		if !found {
			cfg.Collections = append(cfg.Collections, CollectionConfig{Name: name, Path: args[0], Glob: glob, Context: ctx})
		}
		if err := saveConfig(cfg, cfgPath); err != nil {
			return err
		}

		store, err := NewStore(dbPath(indexName))
		if err != nil {
			return err
		}
		defer store.Close()

		if !noEmbed {
			if err := engine.EnsureLib(); err != nil {
				return err
			}
			if err := ensureModel(engine.ModelsDir(), "embedding"); err != nil {
				return err
			}
		}

		if err := syncAll(store, engine, cfg, noEmbed); err != nil {
			return err
		}

		if noEmbed {
			fmt.Printf("Ready! Search with: picoqmd search \"your query\"\n")
		} else {
			fmt.Printf("Ready! Search with: picoqmd \"your query\"\n")
		}
		return nil
	}

	topAddCmd := &cobra.Command{
		Use:   "add <path>",
		Short: "Add a directory, index documents, and embed",
		Args:  cobra.ExactArgs(1),
		RunE:  addRunE,
	}
	topAddCmd.Flags().String("name", "", "collection name")
	topAddCmd.Flags().String("glob", "**/*.md", "file glob pattern, e.g. **/*.md or **/*.{go,py,ts,md}")
	topAddCmd.Flags().String("context", "", "collection description for LLM context")
	topAddCmd.Flags().Bool("no-embed", false, "skip embedding (BM25-only fast indexing)")

	// --- sync (replaces update + embed) ---
	syncRunE := func(cmd *cobra.Command, args []string) error {
		noEmbed, _ := cmd.Flags().GetBool("no-embed")

		cfg, _, err := loadConfig(indexName)
		if err != nil {
			return err
		}
		store, err := NewStore(dbPath(indexName))
		if err != nil {
			return err
		}
		defer store.Close()

		engine := NewLLMEngine(cacheDir())
		return syncAll(store, engine, cfg, noEmbed)
	}

	syncCmd := &cobra.Command{
		Use:   "sync",
		Short: "Re-index and re-embed changed files",
		RunE:  syncRunE,
	}
	syncCmd.Flags().Bool("no-embed", false, "skip embedding (BM25-only re-index)")

	updateCmd := &cobra.Command{
		Use:   "update",
		Short: "Re-index all collections (alias for sync)",
		RunE:  syncRunE,
	}
	updateCmd.Flags().Bool("no-embed", false, "skip embedding (BM25-only re-index)")

	embedCmd := &cobra.Command{
		Use:   "embed",
		Short: "Generate embeddings (alias for sync; -c embeds one collection without re-indexing)",
		RunE: func(cmd *cobra.Command, args []string) error {
			col, _ := cmd.Flags().GetString("collection")
			if col == "" {
				return syncRunE(cmd, args)
			}
			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()
			return embedAll(store, col)
		},
	}
	embedCmd.Flags().Bool("no-embed", false, "skip embedding (BM25-only re-index)")
	embedCmd.Flags().StringP("collection", "c", "", "embed only this collection's pending documents (skips re-indexing)")

	// --- collection add (backward compat) ---
	collectionCmd := &cobra.Command{Use: "collection", Short: "Manage document collections"}
	collAddCmd := &cobra.Command{
		Use:   "add <path>",
		Short: "Add a directory as a collection",
		Args:  cobra.ExactArgs(1),
		RunE:  addRunE,
	}
	collAddCmd.Flags().String("name", "", "collection name")
	collAddCmd.Flags().String("glob", "**/*.md", "file glob pattern, e.g. **/*.md or **/*.{go,py,ts,md}")
	collAddCmd.Flags().String("context", "", "collection description for LLM context")
	collAddCmd.Flags().Bool("no-embed", false, "skip embedding (BM25-only fast indexing)")
	collectionCmd.AddCommand(collAddCmd)

	// --- search (BM25) ---
	searchCmd := &cobra.Command{
		Use:   "search <query>",
		Short: "BM25 full-text search",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.Join(args, " ")
			limit, _ := cmd.Flags().GetInt("limit")
			format, _ := cmd.Flags().GetString("format")

			if remoteAddr != "" {
				return remoteSearch(query, remoteAddr, limit, format)
			}

			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			var results []SearchResult
			if col, _ := cmd.Flags().GetString("collection"); col != "" {
				results, err = store.SearchBM25Scoped(query, col, limit)
			} else {
				results, err = store.SearchBM25(query, limit)
			}
			if err != nil {
				return err
			}
			return printResults(results, format)
		},
	}
	searchCmd.Flags().Int("limit", 10, "max results")
	searchCmd.Flags().String("format", "text", "output: text, json, csv, md, files")
	searchCmd.Flags().StringP("collection", "c", "", "collection name, or comma-separated list")

	// --- vsearch (vector) ---
	vsearchCmd := &cobra.Command{
		Use:   "vsearch <query>",
		Short: "Semantic vector search",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.Join(args, " ")
			limit, _ := cmd.Flags().GetInt("limit")
			format, _ := cmd.Flags().GetString("format")

			if remoteAddr != "" {
				return remoteSearch(query, remoteAddr, limit, format)
			}

			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			engine := NewLLMEngine(cacheDir())
			qvec, err := engine.Embed(query, true)
			if err != nil {
				return err
			}

			col, _ := cmd.Flags().GetString("collection")
			results, err := store.SearchVectorScoped(combineForSnippet(query, searchIntent), qvec, col, limit)
			if err != nil {
				return err
			}
			return printResults(results, format)
		},
	}
	vsearchCmd.Flags().Int("limit", 10, "max results")
	vsearchCmd.Flags().String("format", "text", "output: text, json, csv, md, files")
	vsearchCmd.Flags().StringP("collection", "c", "", "collection name, or comma-separated list")

	// --- query (hybrid) ---
	queryCmd := &cobra.Command{
		Use:   "query <query>",
		Short: "Full hybrid search: expansion + BM25 + vector + RRF + reranking",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			query := strings.Join(args, " ")
			limit, _ := cmd.Flags().GetInt("limit")
			format, _ := cmd.Flags().GetString("format")

			if remoteAddr != "" {
				return remoteSearch(query, remoteAddr, limit, format)
			}

			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			engine := NewLLMEngine(cacheDir())
			hybrid := newHybridSearcher(store, engine)
			if t, ok := hybrid.(interface{ SetNoExpand(bool) }); ok {
				t.SetNoExpand(noExpand)
			}
			if t, ok := hybrid.(interface{ SetNoRerank(bool) }); ok {
				t.SetNoRerank(noRerank)
			}
			col, _ := cmd.Flags().GetString("collection")
			results, err := hybrid.Search(context.Background(), query, searchIntent, col, limit)
			if err != nil {
				return err
			}
			return printResults(results, format)
		},
	}
	queryCmd.Flags().Int("limit", 10, "max results")
	queryCmd.Flags().String("format", "text", "output: text, json, csv, md, files")
	queryCmd.Flags().StringP("collection", "c", "", "collection name, or comma-separated list")

	// --- get ---
	getCmd := &cobra.Command{
		Use:   "get <ref>",
		Short: "Retrieve document by docid (#abc123) or path, with optional :from:count line-range suffix",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			doc, from, count, err := store.GetDocument(args[0])
			if err != nil {
				return err
			}
			if f, _ := cmd.Flags().GetInt("from"); f > 0 {
				from = f
			}
			if l, _ := cmd.Flags().GetInt("lines"); l > 0 {
				count = l
			}
			if searchFormat == "json" {
				b, _ := json.MarshalIndent(doc, "", "  ")
				fmt.Println(string(b))
				return nil
			}
			noNums, _ := cmd.Flags().GetBool("no-line-numbers")
			fullPath, _ := cmd.Flags().GetBool("full-path")
			fmt.Print(renderDocument(doc, from, count, !noNums, fullPath))
			return nil
		},
	}
	getCmd.Flags().Int("from", 0, "1-based first line to print (overrides ref suffix)")
	getCmd.Flags().Int("lines", 0, "number of lines to print")
	getCmd.Flags().Bool("no-line-numbers", false, "omit line-number prefixes")
	getCmd.Flags().Bool("full-path", false, "print the on-disk path in the header instead of the qmd:// URI")

	// --- status ---
	statusCmd := &cobra.Command{
		Use:   "status",
		Short: "Show index statistics",
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			cols, docs, chunks, _ := store.Stats()
			pending, _ := store.CountUnembedded(embedFingerprint(), "")
			fmt.Printf("collections: %d\ndocuments:   %d\nchunks:      %d\npending:     %d docs need (re-)embedding\nfingerprint: %s\ndatabase:    %s\n",
				cols, docs, chunks, pending, embedFingerprint(), dbPath(indexName))
			return nil
		},
	}

	// --- mcp ---
	mcpCmd := &cobra.Command{
		Use:   "mcp",
		Short: "Start MCP server (stdio or HTTP)",
		RunE: func(cmd *cobra.Command, args []string) error {
			httpAddr, _ := cmd.Flags().GetString("http")

			cfg, _, err := loadConfig(indexName)
			if err != nil {
				return err
			}
			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			engine := NewLLMEngine(cacheDir())
			server := NewMCPServer(store, engine, cfg)

			if httpAddr != "" {
				return server.ServeHTTP(httpAddr)
			}
			return server.ServeStdio()
		},
	}
	mcpCmd.Flags().String("http", "", "HTTP listen address (e.g., :8181)")

	// --- context add ---
	contextCmd := &cobra.Command{Use: "context", Short: "Manage context descriptions"}
	contextAddCmd := &cobra.Command{
		Use:   "add <uri> <description>",
		Short: "Attach context description to a qmd:// path",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, cfgPath, err := loadConfig(indexName)
			if err != nil {
				return err
			}
			cfg.Contexts = append(cfg.Contexts, ContextEntry{URI: args[0], Context: args[1]})
			return saveConfig(cfg, cfgPath)
		},
	}
	contextCmd.AddCommand(contextAddCmd)

	// --- model download / list ---
	modelCmd := &cobra.Command{Use: "model", Short: "Manage GGUF models"}
	modelDownloadCmd := &cobra.Command{
		Use:   "download [name]",
		Short: "Download GGUF model files (embedding, reranker, expansion, or all)",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			target := "all"
			if len(args) > 0 {
				target = args[0]
			}
			modelsDir := filepath.Join(cacheDir(), "models")

			for _, spec := range defaultModels {
				if target != "all" && spec.Name != target {
					continue
				}
				if err := ensureModel(modelsDir, spec.Name); err != nil {
					return err
				}
			}
			return nil
		},
	}
	modelListCmd := &cobra.Command{
		Use:   "list",
		Short: "List available and downloaded models",
		RunE: func(cmd *cobra.Command, args []string) error {
			modelsDir := filepath.Join(cacheDir(), "models")
			for _, spec := range defaultModels {
				status := "not downloaded"
				dest := filepath.Join(modelsDir, spec.Filename)
				if info, err := os.Stat(dest); err == nil {
					status = fmt.Sprintf("%.0f MB", float64(info.Size())/1024/1024)
				}
				fmt.Printf("  %-12s  %-40s  [%s]\n", spec.Name, spec.Purpose, status)
			}
			return nil
		},
	}
	modelCmd.AddCommand(modelDownloadCmd, modelListCmd)

	// --- embed-worker (hidden, used by subprocess orchestrator) ---
	var workerBatch int
	var workerCollection string
	embedWorkerCmd := &cobra.Command{
		Use:    "embed-worker",
		Short:  "Internal: embed a batch of documents (used by sync subprocess orchestrator)",
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return embedWorker(workerBatch, workerCollection)
		},
	}
	embedWorkerCmd.Flags().IntVar(&workerBatch, "batch", 500, "max documents to embed")
	embedWorkerCmd.Flags().StringVar(&workerCollection, "collection", "", "restrict to one collection")

	// --- export ---
	exportCmd := &cobra.Command{
		Use:   "export",
		Short: "Export index database and config to a tar.gz bundle",
		RunE: func(cmd *cobra.Command, args []string) error {
			output, _ := cmd.Flags().GetString("output")
			if output == "" {
				output = "picoqmd-export.tar.gz"
			}

			cfg, _, err := loadConfig(indexName)
			if err != nil {
				return err
			}

			dbFile := dbPath(indexName)
			if _, err := os.Stat(dbFile); err != nil {
				return fmt.Errorf("database not found: %s", dbFile)
			}

			f, err := os.Create(output)
			if err != nil {
				return err
			}
			defer f.Close()

			gw := gzip.NewWriter(f)
			defer gw.Close()
			tw := tar.NewWriter(gw)
			defer tw.Close()

			// Add database
			if err := addFileToTar(tw, dbFile, "index.sqlite"); err != nil {
				return fmt.Errorf("add database: %w", err)
			}

			// Rewrite config with relative paths
			exportCfg := *cfg
			for i := range exportCfg.Collections {
				exportCfg.Collections[i].Path = filepath.Base(exportCfg.Collections[i].Path)
			}
			cfgData, err := yaml.Marshal(&exportCfg)
			if err != nil {
				return err
			}
			if err := tw.WriteHeader(&tar.Header{
				Name: "index.yml",
				Size: int64(len(cfgData)),
				Mode: 0644,
			}); err != nil {
				return err
			}
			if _, err := tw.Write(cfgData); err != nil {
				return err
			}

			fmt.Printf("Exported to %s\n", output)
			return nil
		},
	}
	exportCmd.Flags().StringP("output", "o", "picoqmd-export.tar.gz", "output file path")

	// --- import ---
	importCmd := &cobra.Command{
		Use:   "import <file.tar.gz>",
		Short: "Import index database and config from a tar.gz bundle",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			f, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer f.Close()

			gz, err := gzip.NewReader(f)
			if err != nil {
				return err
			}
			defer gz.Close()

			tr := tar.NewReader(gz)
			for {
				hdr, err := tr.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					return fmt.Errorf("read archive: %w", err)
				}

				switch hdr.Name {
				case "index.sqlite":
					dest := dbPath(indexName)
					os.MkdirAll(filepath.Dir(dest), 0o755)
					out, err := os.Create(dest)
					if err != nil {
						return err
					}
					io.Copy(out, tr)
					out.Close()
					fmt.Printf("  Database → %s\n", dest)

				case "index.yml":
					name := "index"
					if indexName != "" {
						name = indexName
					}
					dest := filepath.Join(configDir(), name+".yml")
					os.MkdirAll(filepath.Dir(dest), 0o755)
					out, err := os.Create(dest)
					if err != nil {
						return err
					}
					io.Copy(out, tr)
					out.Close()
					fmt.Printf("  Config   → %s\n", dest)
				}
			}

			fmt.Println("Import complete. Search with: picoqmd search \"your query\"")
			return nil
		},
	}

	root.AddCommand(topAddCmd, syncCmd, collectionCmd, updateCmd, embedCmd, searchCmd, vsearchCmd, queryCmd, getCmd, statusCmd, mcpCmd, contextCmd, modelCmd, embedWorkerCmd, exportCmd, importCmd,
		newDoctorCmd(func() string { return dbPath(indexName) }),
		newCleanupCmd(func() string { return dbPath(indexName) }),
		newMigrateVectorsCmd(func() string { return dbPath(indexName) }),
		newBenchCmd(func() string { return dbPath(indexName) }))

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

// pathWithLine returns the result's path with `:L<n>` appended when the line
// number is known. Mirrors the convention used by `picoqmd get path:line`.
func pathWithLine(r SearchResult) string {
	if r.Line > 0 {
		return fmt.Sprintf("%s:L%d", r.Path, r.Line)
	}
	return r.Path
}

func printResults(results []SearchResult, format string) error {
	switch format {
	case "json":
		b, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(b))
	case "csv":
		fmt.Println("docid,score,path,context")
		for _, r := range results {
			fmt.Printf("%s,%.4f,%s,%s\n", r.DocID, r.Score, pathWithLine(r), r.Context)
		}
	case "files":
		for _, r := range results {
			fmt.Println(pathWithLine(r))
		}
	case "md":
		for i, r := range results {
			fmt.Printf("### %d. %s (`#%s` — %.4f)\n", i+1, r.Title, r.DocID, r.Score)
			fmt.Printf("`%s`\n\n", pathWithLine(r))
			if r.Snippet != "" {
				fmt.Printf("> %s\n\n", r.Snippet)
			}
		}
	default: // text
		for i, r := range results {
			fmt.Printf("%d. [#%s] %s (%.4f)\n", i+1, r.DocID, r.Title, r.Score)
			fmt.Printf("   %s\n", pathWithLine(r))
			if r.Snippet != "" {
				fmt.Printf("   %s\n", r.Snippet)
			}
		}
	}
	if len(results) == 0 {
		fmt.Println("no results")
	}
	return nil
}
