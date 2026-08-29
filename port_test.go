package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// parseScope — collection scope strings
// ---------------------------------------------------------------------------

func TestParseScope(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"memory", []string{"memory"}},
		{"shared-src,ios-src", []string{"shared-src", "ios-src"}},
		{" a , b ,", []string{"a", "b"}},
		{",,", nil},
	}
	for _, c := range cases {
		got := parseScope(c.in)
		if len(got) != len(c.want) {
			t.Fatalf("parseScope(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("parseScope(%q) = %v, want %v", c.in, got, c.want)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// modelFileHash — sidecar caching
// ---------------------------------------------------------------------------

func TestModelFileHashSidecar(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(p, []byte("model bytes v1"), 0o644); err != nil {
		t.Fatal(err)
	}

	h1 := modelFileHash(p)
	if len(h1) != 16 {
		t.Fatalf("hash length = %d, want 16", len(h1))
	}
	// Sidecar written and reused
	if _, err := os.Stat(p + ".sha256"); err != nil {
		t.Fatalf("sidecar not written: %v", err)
	}
	if h2 := modelFileHash(p); h2 != h1 {
		t.Fatalf("cached hash %q != original %q", h2, h1)
	}

	// Changed bytes (same length forces mtime/size check to notice content
	// via mtime) must produce a different hash.
	if err := os.WriteFile(p, []byte("model bytes v2"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Bump mtime explicitly in case the writes land in the same second.
	st, _ := os.Stat(p)
	os.Chtimes(p, st.ModTime().Add(2e9), st.ModTime().Add(2e9))
	if h3 := modelFileHash(p); h3 == h1 {
		t.Fatal("hash unchanged after model bytes changed")
	}

	// Missing file → empty (legacy name-only fingerprint fallback).
	if h := modelFileHash(filepath.Join(dir, "absent.gguf")); h != "" {
		t.Fatalf("missing file hash = %q, want empty", h)
	}
}

// ---------------------------------------------------------------------------
// truncateMRL — Matryoshka truncation
// ---------------------------------------------------------------------------

func TestTruncateMRL(t *testing.T) {
	vec := []float32{3, 4, 0, 0, 5, 12}
	out := truncateMRL(vec, 2)
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2", len(out))
	}
	// [3,4] has norm 5 → renormalized to [0.6, 0.8]
	if out[0] != 0.6 || out[1] != 0.8 {
		t.Fatalf("got %v, want [0.6 0.8]", out)
	}
	var norm float64
	for _, x := range out {
		norm += float64(x) * float64(x)
	}
	if norm < 0.999 || norm > 1.001 {
		t.Fatalf("norm = %v, want 1", norm)
	}

	// No-op cases: dim 0 and already-small vectors return the input.
	if got := truncateMRL(vec, 0); len(got) != len(vec) {
		t.Fatal("dim 0 should be a no-op")
	}
	if got := truncateMRL(vec, 10); len(got) != len(vec) {
		t.Fatal("dim > len should be a no-op")
	}
	// Zero vector: no NaNs.
	z := truncateMRL([]float32{0, 0, 0, 0}, 2)
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("zero vector mangled: %v", z)
	}
}

// ---------------------------------------------------------------------------
// bench — matching and metrics
// ---------------------------------------------------------------------------

func TestBenchMatch(t *testing.T) {
	r := SearchResult{DocID: "abc123", Path: "memory/project_picoqmd.md", Title: "picoqmd"}
	if !benchMatch(r, "#abc123") {
		t.Fatal("docid match failed")
	}
	if benchMatch(r, "#zzz") {
		t.Fatal("docid mismatch matched")
	}
	if !benchMatch(r, "project_picoqmd") {
		t.Fatal("path substring failed")
	}
	if benchMatch(r, "nope") {
		t.Fatal("unrelated substring matched")
	}
}

func TestBenchScore(t *testing.T) {
	results := []SearchResult{
		{DocID: "a", Path: "x/alpha.md"},
		{DocID: "b", Path: "x/beta.md"},
		{DocID: "c", Path: "x/gamma.md"},
	}
	s := &benchScore{}
	// First relevant at rank 2 → MRR 0.5; one of two expectations found.
	s.add(results, []string{"beta", "missing-doc"}, 3)
	if s.hits != 1 {
		t.Fatalf("hits = %d, want 1", s.hits)
	}
	if s.mrr != 0.5 {
		t.Fatalf("mrr = %v, want 0.5", s.mrr)
	}
	if s.recall != 0.5 {
		t.Fatalf("recall = %v, want 0.5", s.recall)
	}
	// precision = 1 relevant / k=3
	if diff := s.precision - 1.0/3.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("precision = %v, want 1/3", s.precision)
	}

	// Total miss.
	s.add(results, []string{"nothing"}, 3)
	if s.hits != 1 || s.cases != 2 {
		t.Fatalf("after miss: hits=%d cases=%d", s.hits, s.cases)
	}
	if !strings.Contains(s.row("test"), "hit@k  50.0%") {
		t.Fatalf("row output unexpected: %q", s.row("test"))
	}
}

// ---------------------------------------------------------------------------
// scoped BM25 — multi-collection RRF vs single
// ---------------------------------------------------------------------------

func TestSearchBM25Scoped(t *testing.T) {
	dir := t.TempDir()
	store, err := NewStore(filepath.Join(dir, "test.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	mk := func(col, name, body string) {
		colDir := filepath.Join(dir, col)
		os.MkdirAll(colDir, 0o755)
		p := filepath.Join(colDir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mk("colA", "zebra.md", "# Zebra notes\nthe zebra migration pattern")
	mk("colB", "zebra2.md", "# More zebras\nzebra habitat and zebra stripes and zebra herds")

	for _, col := range []string{"colA", "colB"} {
		if _, err := store.UpsertCollection(col, filepath.Join(dir, col), "**/*.md", ""); err != nil {
			t.Fatal(err)
		}
		if err := indexCollection(store, CollectionConfig{Name: col, Path: filepath.Join(dir, col), Glob: "**/*.md"}); err != nil {
			t.Fatal(err)
		}
	}

	single, err := store.SearchBM25Scoped("zebra", "colA", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(single) != 1 || single[0].Path != "zebra.md" {
		t.Fatalf("single scope: %+v", single)
	}

	multi, err := store.SearchBM25Scoped("zebra", "colA,colB", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(multi) != 2 {
		t.Fatalf("multi scope returned %d results, want 2", len(multi))
	}

	all, err := store.SearchBM25Scoped("zebra", "", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2 {
		t.Fatalf("empty scope returned %d results, want 2", len(all))
	}
}
