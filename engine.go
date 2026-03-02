package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// Interfaces — platform files (llm.go / llm_stub.go) provide implementations
// ---------------------------------------------------------------------------

// Embedder abstracts the LLM engine for embedding, query expansion, and reranking.
type Embedder interface {
	Embed(text string, isQuery bool) ([]float32, error)
	BatchEmbed(texts []string) ([][]float32, error)
	ExpandQuery(query string) ([]QueryExpansion, error)
	Rerank(query string, candidates []string) ([]float64, error)
	Close()
	ModelsDir() string
	EnsureLib() error
}

// Searcher abstracts hybrid/vector search pipelines.
type Searcher interface {
	Search(ctx context.Context, query string, limit int) ([]SearchResult, error)
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

type QueryExpansion struct {
	Query string `json:"query"`
	Type  string `json:"type"` // "lex" or "vec"
}

type ModelSpec struct {
	Name     string
	Filename string
	URL      string
	Purpose  string
}

var defaultModels = []ModelSpec{
	{
		Name:     "embedding",
		Filename: "embeddinggemma-300M-Q8_0.gguf",
		URL:      "https://huggingface.co/tobi/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf",
		Purpose:  "Document and query embeddings",
	},
	{
		Name:     "reranker",
		Filename: "qwen3-reranker-0.6b-q8_0.gguf",
		URL:      "https://huggingface.co/tobi/qwen3-reranker-0.6b-GGUF/resolve/main/qwen3-reranker-0.6b-q8_0.gguf",
		Purpose:  "Cross-encoder re-ranking",
	},
	{
		Name:     "expansion",
		Filename: "qmd-query-expansion-1.7B-q4_k_m.gguf",
		URL:      "https://huggingface.co/tobi/qmd-query-expansion-GGUF/resolve/main/qmd-query-expansion-1.7B-q4_k_m.gguf",
		Purpose:  "Fine-tuned query expansion",
	},
}

// ---------------------------------------------------------------------------
// Model management — platform-independent
// ---------------------------------------------------------------------------

func modelExists(modelsDir, filename string) bool {
	_, err := os.Stat(filepath.Join(modelsDir, filename))
	return err == nil
}

func ensureModel(modelsDir, name string) error {
	for _, spec := range defaultModels {
		if spec.Name != name {
			continue
		}
		dest := filepath.Join(modelsDir, spec.Filename)
		if _, err := os.Stat(dest); err == nil {
			return nil
		}

		os.MkdirAll(modelsDir, 0o755)

		resp, err := http.Get(spec.URL)
		if err != nil {
			return fmt.Errorf("download %s: %w", name, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return fmt.Errorf("download %s: HTTP %d", name, resp.StatusCode)
		}

		f, err := os.Create(dest + ".tmp")
		if err != nil {
			return err
		}

		pw := &progressWriter{name: spec.Name + " model", total: resp.ContentLength}
		_, err = io.Copy(f, io.TeeReader(resp.Body, pw))
		f.Close()
		pw.finish()
		if err != nil {
			os.Remove(dest + ".tmp")
			return fmt.Errorf("download %s: %w", name, err)
		}

		return os.Rename(dest+".tmp", dest)
	}
	return fmt.Errorf("unknown model: %s", name)
}

// ---------------------------------------------------------------------------
// Progress writer — terminal progress bar for downloads
// ---------------------------------------------------------------------------

type progressWriter struct {
	name    string
	total   int64
	written int64
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n := len(p)
	pw.written += int64(n)
	barLen := 30
	if pw.total > 0 {
		pct := float64(pw.written) / float64(pw.total) * 100
		filled := int(pct / 100 * float64(barLen))
		if filled > barLen {
			filled = barLen
		}
		bar := strings.Repeat("=", filled) + strings.Repeat(" ", barLen-filled)
		fmt.Fprintf(os.Stderr, "\r  Downloading %s (%dMB)... [%s] %.0f%%",
			pw.name, pw.total/1024/1024, bar, pct)
	} else {
		fmt.Fprintf(os.Stderr, "\r  Downloading %s... %dMB",
			pw.name, pw.written/1024/1024)
	}
	return n, nil
}

func (pw *progressWriter) finish() {
	fmt.Fprintln(os.Stderr)
}
