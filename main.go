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
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
	"zombiezen.com/go/sqlite"       // Pure Go SQLite — no CGO
	"zombiezen.com/go/sqlite/sqlitex"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	version           = "0.2.0"
	defaultDB         = "index.sqlite"
	chunkTarget       = 900  // target tokens per chunk
	chunkLookback     = 200  // tokens to look back for break points
	chunkOverlap      = 135  // 15% of chunkTarget — overlap tokens between adjacent chunks
	rrfK              = 60   // RRF fusion constant
	maxRerank         = 30   // candidates sent to reranker
	maxIndexFileBytes = 1 << 20 // 1 MB file size limit for indexing
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
	ID      int64
	Path    string
	Title   string
	DocID   string // 6-char content hash
	Hash    string // full content hash for change detection
	Active  bool
	Content string
	Context string // inherited from collection/context tree
}

type SearchResult struct {
	DocID   string  `json:"docid"`
	Path    string  `json:"path"`
	Title   string  `json:"title"`
	Score   float64 `json:"score"`
	Snippet string  `json:"snippet,omitempty"`
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
	pool, err := sqlitex.NewPool(dbPath, sqlitex.PoolOptions{PoolSize: 4})
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

	return sqlitex.ExecuteScript(conn, `
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

	// Upsert document row
	err = sqlitex.Execute(conn,
		`INSERT INTO documents (col_id, path, title, docid, hash, active)
		 VALUES (?, ?, ?, ?, ?, 1)
		 ON CONFLICT(col_id, path) DO UPDATE SET title=excluded.title, docid=excluded.docid, hash=excluded.hash, active=1`,
		&sqlitex.ExecOptions{Args: []any{colID, relPath, title, docid, hash}})
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

// SearchBM25 performs FTS5 BM25 ranked search.
func (s *Store) SearchBM25(query string, limit int) ([]SearchResult, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	ftsQuery := toFTS5Query(query)
	var results []SearchResult

	err = sqlitex.Execute(conn, `
		SELECT d.docid, d.path, d.title,
		       -rank AS score,
		       snippet(documents_fts, 1, '>>>', '<<<', '...', 40) AS snip,
		       c.context
		FROM documents_fts f
		JOIN documents d ON d.id = f.rowid
		JOIN collections c ON c.id = d.col_id
		WHERE documents_fts MATCH ?
		  AND d.active = 1
		ORDER BY rank
		LIMIT ?
	`, &sqlitex.ExecOptions{
		Args: []any{ftsQuery, limit},
		ResultFunc: func(stmt *sqlite.Stmt) error {
			results = append(results, SearchResult{
				DocID:   stmt.ColumnText(0),
				Path:    stmt.ColumnText(1),
				Title:   stmt.ColumnText(2),
				Score:   stmt.ColumnFloat(3),
				Snippet: stmt.ColumnText(4),
				Context: stmt.ColumnText(5),
			})
			return nil
		},
	})
	return results, err
}

// SearchVector performs optimized cosine similarity search over stored embeddings.
func (s *Store) SearchVector(queryVec []float32, limit int) ([]SearchResult, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	bestByHash := make(map[string]float64)

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

	err = sqlitex.Execute(conn, `SELECT hash, vec FROM content_vectors WHERE vec IS NOT NULL`,
		&sqlitex.ExecOptions{
			ResultFunc: func(stmt *sqlite.Stmt) error {
				h := stmt.ColumnText(0)
				vecLen := stmt.ColumnLen(1)
				if vecLen != vecDim*4 {
					return nil
				}
				raw := make([]byte, vecLen)
				stmt.ColumnBytes(1, raw)
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

				if sim > bestByHash[h] {
					bestByHash[h] = sim
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
	}
	sorted_ := make([]scored, 0, len(bestByHash))
	for h, sc := range bestByHash {
		sorted_ = append(sorted_, scored{h, sc})
	}
	sort.Slice(sorted_, func(i, j int) bool { return sorted_[i].score > sorted_[j].score })
	if len(sorted_) > limit {
		sorted_ = sorted_[:limit]
	}

	var results []SearchResult
	for _, s2 := range sorted_ {
		var r SearchResult
		r.Score = s2.score
		err = sqlitex.Execute(conn, `
			SELECT d.docid, d.path, d.title, c.context
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
					return nil
				},
			})
		if err != nil {
			continue
		}
		if r.DocID != "" {
			results = append(results, r)
		}
	}
	return results, nil
}

// GetDocument retrieves a single document by docid or path.
func (s *Store) GetDocument(ref string) (*Document, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	ref = strings.TrimPrefix(ref, "#")
	var doc Document
	found := false

	query := `SELECT d.id, d.path, d.title, d.docid, d.hash, c.context
	           FROM documents d JOIN collections c ON c.id = d.col_id
	           WHERE (d.docid = ? OR d.path = ?) AND d.active = 1 LIMIT 1`

	err = sqlitex.Execute(conn, query, &sqlitex.ExecOptions{
		Args: []any{ref, ref},
		ResultFunc: func(stmt *sqlite.Stmt) error {
			found = true
			doc.ID = stmt.ColumnInt64(0)
			doc.Path = stmt.ColumnText(1)
			doc.Title = stmt.ColumnText(2)
			doc.DocID = stmt.ColumnText(3)
			doc.Hash = stmt.ColumnText(4)
			doc.Context = stmt.ColumnText(5)
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

// MultiGet retrieves documents matching a glob pattern or comma-separated list of paths.
func (s *Store) MultiGet(pattern string) ([]Document, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var docs []Document
	patterns := strings.Split(pattern, ",")
	for _, p := range patterns {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		likePattern := strings.ReplaceAll(p, "*", "%")
		err = sqlitex.Execute(conn, `
			SELECT d.id, d.path, d.title, d.docid, d.hash, c.context
			FROM documents d JOIN collections c ON c.id = d.col_id
			WHERE d.active = 1 AND d.path LIKE ?
			ORDER BY d.path`, &sqlitex.ExecOptions{
			Args: []any{likePattern},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				docs = append(docs, Document{
					ID:      stmt.ColumnInt64(0),
					Path:    stmt.ColumnText(1),
					Title:   stmt.ColumnText(2),
					DocID:   stmt.ColumnText(3),
					Hash:    stmt.ColumnText(4),
					Context: stmt.ColumnText(5),
					Active:  true,
				})
				return nil
			},
		})
		if err != nil {
			return nil, err
		}
	}
	return docs, nil
}

// UnembeddedHashes returns content hashes that haven't been embedded yet.
func (s *Store) UnembeddedHashes() ([]string, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var hashes []string
	err = sqlitex.Execute(conn, `
		SELECT DISTINCT d.hash FROM documents d
		WHERE d.active = 1
		  AND d.hash NOT IN (SELECT DISTINCT hash FROM content_vectors WHERE vec IS NOT NULL)
	`, &sqlitex.ExecOptions{
		ResultFunc: func(stmt *sqlite.Stmt) error {
			hashes = append(hashes, stmt.ColumnText(0))
			return nil
		},
	})
	return hashes, err
}

// CountUnembedded returns the number of active documents without embeddings.
func (s *Store) CountUnembedded() (int, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return 0, err
	}
	defer s.pool.Put(conn)

	var count int
	err = sqlitex.Execute(conn, `
		SELECT COUNT(DISTINCT d.hash) FROM documents d
		WHERE d.active = 1
		  AND d.hash NOT IN (SELECT DISTINCT hash FROM content_vectors WHERE vec IS NOT NULL)
	`, &sqlitex.ExecOptions{
		ResultFunc: func(stmt *sqlite.Stmt) error {
			count = stmt.ColumnInt(0)
			return nil
		},
	})
	return count, err
}

// SkipNextUnembedded marks the next unembedded document as embedded (with empty
// vector) so the orchestrator can make progress past problematic documents.
func (s *Store) SkipNextUnembedded() error {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	var hash string
	err = sqlitex.Execute(conn, `
		SELECT DISTINCT d.hash FROM documents d
		WHERE d.active = 1
		  AND d.hash NOT IN (SELECT DISTINCT hash FROM content_vectors WHERE vec IS NOT NULL)
		LIMIT 1
	`, &sqlitex.ExecOptions{
		ResultFunc: func(stmt *sqlite.Stmt) error {
			hash = stmt.ColumnText(0)
			return nil
		},
	})
	if err != nil || hash == "" {
		return err
	}

	dummyVec := make([]byte, 4)
	return sqlitex.Execute(conn, `
		INSERT OR IGNORE INTO content_vectors (hash, seq, text, vec) VALUES (?, 0, '[skipped]', ?)
	`, &sqlitex.ExecOptions{
		Args: []any{hash, dummyVec},
	})
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

// StoreVector writes the embedding vector for a chunk.
func (s *Store) StoreVector(hash string, seq int, vec []float32) error {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return err
	}
	defer s.pool.Put(conn)

	return sqlitex.Execute(conn,
		`UPDATE content_vectors SET vec=? WHERE hash=? AND seq=?`,
		&sqlitex.ExecOptions{Args: []any{float32ToBytes(vec), hash, seq}})
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
func (s *Store) UnembeddedChunks(hash string) ([]struct{ Seq int; Text string }, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	var chunks []struct{ Seq int; Text string }
	err = sqlitex.Execute(conn,
		`SELECT seq, text FROM content_vectors WHERE hash = ? AND vec IS NULL ORDER BY seq`,
		&sqlitex.ExecOptions{
			Args: []any{hash},
			ResultFunc: func(stmt *sqlite.Stmt) error {
				chunks = append(chunks, struct{ Seq int; Text string }{
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
	searchSchema := map[string]any{"type": "object", "properties": map[string]any{
		"query":      map[string]any{"type": "string", "description": "Search query"},
		"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
		"collection": map[string]any{"type": "string", "description": "Filter to a specific collection"},
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
				"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
				"collection": map[string]any{"type": "string", "description": "Filter to a specific collection"},
				"minScore":   map[string]any{"type": "number", "description": "Minimum score threshold (default 0.3)"},
				"maxChars":   map[string]any{"type": "integer", "description": "Truncate total response to this many characters"},
				"note":       map[string]any{"type": "string", "description": "Save an observation linked to the top result"},
			}, "required": []string{"query"}}})
		tools = append(tools, map[string]any{
			"name": "deep_search", "description": "Full hybrid pipeline: auto-expands query into variations, searches each by keyword and meaning, reranks for top hits",
			"inputSchema": searchSchema})
		tools = append(tools, map[string]any{
			"name": "research", "description": "Composite search: runs BM25 + vector in parallel, deduplicates by docid via RRF, and merges within a token budget. One call instead of two.",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"query":      map[string]any{"type": "string", "description": "Search query"},
				"limit":      map[string]any{"type": "integer", "description": "Max results (default 10)"},
				"collection": map[string]any{"type": "string", "description": "Filter to a specific collection"},
				"minScore":   map[string]any{"type": "number", "description": "Minimum score threshold (default 0)"},
				"maxChars":   map[string]any{"type": "integer", "description": "Truncate total response to this many characters (default: no limit)"},
				"note":       map[string]any{"type": "string", "description": "Save an observation linked to the top result"},
			}, "required": []string{"query"}}})
	}

	tools = append(tools,
		map[string]any{"name": "get", "description": "Retrieve a single document by path or docid (#abc123). Supports line offset (file.md:100).",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"ref":      map[string]any{"type": "string", "description": "File path, docid (#abc123), or path:line"},
				"maxLines": map[string]any{"type": "integer", "description": "Maximum lines to return"},
				"maxChars": map[string]any{"type": "integer", "description": "Truncate response to this many characters"},
			}, "required": []string{"ref"}}},
		map[string]any{"name": "multi_get", "description": "Retrieve multiple documents by glob pattern or comma-separated list",
			"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
				"pattern":  map[string]any{"type": "string", "description": "Glob pattern (e.g., docs/*.md) or comma-separated paths"},
				"maxBytes": map[string]any{"type": "integer", "description": "Skip files over this size (default 10240)"},
				"maxChars": map[string]any{"type": "integer", "description": "Truncate total response to this many characters"},
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
		Query      string  `json:"query"`
		Ref        string  `json:"ref"`
		Pattern    string  `json:"pattern"`
		Limit      int     `json:"limit"`
		Collection string  `json:"collection"`
		MinScore   float64 `json:"minScore"`
		MaxBytes   int     `json:"maxBytes"`
		MaxLines   int     `json:"maxLines"`
		MaxChars   int     `json:"maxChars"`
		Note       string  `json:"note"`
	}
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		return nil, fmt.Errorf("invalid tool arguments: %w", err)
	}
	if args.Limit == 0 {
		args.Limit = 10
	}
	if args.MaxBytes == 0 {
		args.MaxBytes = 10240
	}

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
		results, err := m.store.SearchBM25(args.Query, args.Limit)
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
		results, err := m.store.SearchVector(qvec, args.Limit)
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
		results, err := m.hybrid.Search(context.Background(), args.Query, args.Limit)
		if err != nil {
			return nil, err
		}
		filtered := filterMinScore(results, args.MinScore)
		saveNote(filtered, args.Note)
		return applyMaxChars(toolResult(filtered), args.MaxChars), nil

	case "research":
		// Composite: BM25 + vector search, deduplicated via RRF
		bm25Results, _ := m.store.SearchBM25(args.Query, args.Limit*2)
		var vecResults []SearchResult
		if qvec, err := m.engine.Embed(args.Query, true); err == nil {
			vecResults, _ = m.store.SearchVector(qvec, args.Limit*2)
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
		doc, err := m.store.GetDocument(args.Ref)
		if err != nil {
			return nil, err
		}
		return applyMaxChars(toolResult(doc), args.MaxChars), nil

	case "multi_get":
		docs, err := m.store.MultiGet(args.Pattern)
		if err != nil {
			return nil, err
		}
		return applyMaxChars(toolResult(docs), args.MaxChars), nil

	case "status":
		cols, docs, chunks, _ := m.store.Stats()
		unembedded, _ := m.store.UnembeddedHashes()
		stale := getStaleObservations(m.observationsPath(), m.store)
		return toolResult(map[string]any{
			"collections":       cols,
			"documents":         docs,
			"chunks":            chunks,
			"needsEmbedding":    len(unembedded),
			"hasVectorIndex":    len(unembedded) == 0 && chunks > 0,
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
	doc, err := m.store.GetDocument(docid)
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
		doc, err := store.GetDocument(obs.DocID)
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
	return map[string]any{
		"content": []map[string]any{
			{"type": "text", "text": string(b)},
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

func toFTS5Query(query string) string {
	words := strings.Fields(query)
	if len(words) == 0 {
		return query
	}

	var parts []string
	inPhrase := false
	var phrase []string

	for _, w := range words {
		if !inPhrase && strings.HasPrefix(w, `"`) {
			inPhrase = true
			phrase = []string{strings.TrimPrefix(w, `"`)}
			if strings.HasSuffix(w, `"`) && len(w) > 1 {
				// Single-word quoted: "word"
				inPhrase = false
				phrase[0] = strings.TrimSuffix(phrase[0], `"`)
				parts = append(parts, `"`+strings.Join(phrase, " ")+`"`)
			}
			continue
		}
		if inPhrase {
			if strings.HasSuffix(w, `"`) {
				phrase = append(phrase, strings.TrimSuffix(w, `"`))
				parts = append(parts, `"`+strings.Join(phrase, " ")+`"`)
				inPhrase = false
			} else {
				phrase = append(phrase, w)
			}
			continue
		}
		// Unquoted word — use prefix matching
		w = strings.Trim(w, `"'`)
		if w != "" {
			parts = append(parts, w+"*")
		}
	}

	// Unclosed quote — treat remaining words as prefix terms
	if inPhrase {
		for _, w := range phrase {
			parts = append(parts, w+"*")
		}
	}

	return strings.Join(parts, " AND ")
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

func main() {
	var indexName string
	var searchLimit int
	var searchFormat string
	var remoteAddr string

	// --- smartSearch dispatches to the best available pipeline ---
	smartSearch := func(query string, store *Store, engine Embedder, limit int, format string) error {
		hasEmbed := modelExists(engine.ModelsDir(), defaultModels[0].Filename)
		hasRerank := modelExists(engine.ModelsDir(), defaultModels[1].Filename)
		hasExpand := modelExists(engine.ModelsDir(), defaultModels[2].Filename)

		// Full hybrid: all 3 models available
		if hasEmbed && hasRerank && hasExpand {
			hybrid := newHybridSearcher(store, engine)
			results, err := hybrid.Search(context.Background(), query, limit)
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
				vecResults, _ := store.SearchVector(qvec, limit*2)
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
			return smartSearch(query, store, engine, searchLimit, searchFormat)
		},
	}
	root.PersistentFlags().StringVar(&indexName, "index", "", "named index (separate DB + config)")
	root.PersistentFlags().IntVar(&searchLimit, "limit", 10, "max search results")
	root.PersistentFlags().StringVar(&searchFormat, "format", "text", "output: text, json, csv, md, files")
	root.PersistentFlags().StringVar(&remoteAddr, "remote", "", "forward searches to remote picoqmd MCP server (host:port)")

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
		Short: "Generate embeddings (alias for sync)",
		RunE:  syncRunE,
	}
	embedCmd.Flags().Bool("no-embed", false, "skip embedding (BM25-only re-index)")

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

			results, err := store.SearchBM25(query, limit)
			if err != nil {
				return err
			}
			return printResults(results, format)
		},
	}
	searchCmd.Flags().Int("limit", 10, "max results")
	searchCmd.Flags().String("format", "text", "output: text, json, csv, md, files")

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

			results, err := store.SearchVector(qvec, limit)
			if err != nil {
				return err
			}
			return printResults(results, format)
		},
	}
	vsearchCmd.Flags().Int("limit", 10, "max results")
	vsearchCmd.Flags().String("format", "text", "output: text, json, csv, md, files")

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
			results, err := hybrid.Search(context.Background(), query, limit)
			if err != nil {
				return err
			}
			return printResults(results, format)
		},
	}
	queryCmd.Flags().Int("limit", 10, "max results")
	queryCmd.Flags().String("format", "text", "output: text, json, csv, md, files")

	// --- get ---
	getCmd := &cobra.Command{
		Use:   "get <ref>",
		Short: "Retrieve document by docid (#abc123) or path",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			store, err := NewStore(dbPath(indexName))
			if err != nil {
				return err
			}
			defer store.Close()

			doc, err := store.GetDocument(args[0])
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(doc, "", "  ")
			fmt.Println(string(b))
			return nil
		},
	}

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
			fmt.Printf("collections: %d\ndocuments:   %d\nchunks:      %d\ndatabase:    %s\n",
				cols, docs, chunks, dbPath(indexName))
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
	embedWorkerCmd := &cobra.Command{
		Use:    "embed-worker",
		Short:  "Internal: embed a batch of documents (used by sync subprocess orchestrator)",
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return embedWorker(workerBatch)
		},
	}
	embedWorkerCmd.Flags().IntVar(&workerBatch, "batch", 500, "max documents to embed")

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

	root.AddCommand(topAddCmd, syncCmd, collectionCmd, updateCmd, embedCmd, searchCmd, vsearchCmd, queryCmd, getCmd, statusCmd, mcpCmd, contextCmd, modelCmd, embedWorkerCmd, exportCmd, importCmd)

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

func printResults(results []SearchResult, format string) error {
	switch format {
	case "json":
		b, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(b))
	case "csv":
		fmt.Println("docid,score,path,context")
		for _, r := range results {
			fmt.Printf("%s,%.4f,%s,%s\n", r.DocID, r.Score, r.Path, r.Context)
		}
	case "files":
		for _, r := range results {
			fmt.Println(r.Path)
		}
	case "md":
		for i, r := range results {
			fmt.Printf("### %d. %s (`#%s` — %.4f)\n", i+1, r.Title, r.DocID, r.Score)
			if r.Snippet != "" {
				fmt.Printf("> %s\n\n", r.Snippet)
			}
		}
	default: // text
		for i, r := range results {
			fmt.Printf("%d. [#%s] %s (%.4f)\n", i+1, r.DocID, r.Title, r.Score)
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
