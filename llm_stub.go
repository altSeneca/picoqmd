//go:build !((freebsd || linux || windows || darwin) && (amd64 || arm64))

package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
)

// ---------------------------------------------------------------------------
// stubEngine — Embedder implementation for platforms without llama.cpp
// ---------------------------------------------------------------------------

type stubEngine struct {
	modelsDir string
}

func NewLLMEngine(cache string) Embedder {
	modelsDir := filepath.Join(cache, "models")
	os.MkdirAll(modelsDir, 0o755)
	return &stubEngine{modelsDir: modelsDir}
}

func (e *stubEngine) Embed(text string, isQuery bool) ([]float32, error) {
	return nil, fmt.Errorf("embedding not supported on this platform (requires amd64 or arm64)")
}

func (e *stubEngine) BatchEmbed(texts []string) ([][]float32, error) {
	return nil, fmt.Errorf("embedding not supported on this platform")
}

func (e *stubEngine) ExpandQuery(query string) ([]QueryExpansion, error) {
	return []QueryExpansion{
		{Query: query, Type: "lex"},
		{Query: query, Type: "vec"},
	}, nil
}

func (e *stubEngine) Rerank(query string, candidates []string) ([]float64, error) {
	scores := make([]float64, len(candidates))
	for i := range scores {
		scores[i] = 0.5
	}
	return scores, nil
}

func (e *stubEngine) Close()            {}
func (e *stubEngine) ModelsDir() string { return e.modelsDir }
func (e *stubEngine) EnsureLib() error  { return nil }

// ---------------------------------------------------------------------------
// BM25OnlySearcher — uses precomputed embeddings when available
// ---------------------------------------------------------------------------

type BM25OnlySearcher struct {
	store *Store
}

func newHybridSearcher(store *Store, engine Embedder) Searcher {
	return &BM25OnlySearcher{store: store}
}

func (s *BM25OnlySearcher) Search(ctx context.Context, query string, limit int) ([]SearchResult, error) {
	// 1. BM25 seed
	bm25Results, err := s.store.SearchBM25(query, 20)
	if err != nil {
		return nil, err
	}
	if len(bm25Results) == 0 {
		return nil, nil
	}

	// 2. Load stored chunk vectors for those DocIDs
	docIDs := make([]string, len(bm25Results))
	for i, r := range bm25Results {
		docIDs[i] = r.DocID
	}
	vecs, err := s.store.EmbeddingsForDocIDs(docIDs)
	if err != nil || len(vecs) == 0 {
		// Fallback to BM25 only
		if len(bm25Results) > limit {
			bm25Results = bm25Results[:limit]
		}
		return bm25Results, nil
	}

	// 3. Centroid (mean) of loaded vectors → pseudo query vector
	dim := len(vecs[0])
	centroid := make([]float32, dim)
	count := 0
	for _, v := range vecs {
		if len(v) != dim {
			continue
		}
		for i, val := range v {
			centroid[i] += val
		}
		count++
	}
	if count == 0 {
		if len(bm25Results) > limit {
			bm25Results = bm25Results[:limit]
		}
		return bm25Results, nil
	}
	n := float32(count)
	for i := range centroid {
		centroid[i] /= n
	}

	// Normalize centroid
	var norm float64
	for _, v := range centroid {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range centroid {
			centroid[i] = float32(float64(centroid[i]) / norm)
		}
	}

	// 4. SearchVector with centroid for ranking
	vecResults, err := s.store.SearchVector(centroid, limit)
	if err != nil || len(vecResults) == 0 {
		if len(bm25Results) > limit {
			bm25Results = bm25Results[:limit]
		}
		return bm25Results, nil
	}

	// 5. Simple RRF merge
	return simpleRRF(bm25Results, vecResults, limit), nil
}

// ---------------------------------------------------------------------------
// Stub functions
// ---------------------------------------------------------------------------

func embedWorker(maxDocs int) error {
	return fmt.Errorf("embedding not supported on this platform (requires amd64 or arm64)")
}

func syncAll(store *Store, engine Embedder, cfg *Config, skipEmbed bool) error {
	for _, col := range cfg.Collections {
		if err := indexCollection(store, col); err != nil {
			log.Printf("error indexing %s: %v", col.Name, err)
		}
	}
	// Stub platforms don't support local embedding
	return nil
}
