package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// toFTS5Query — sanitization
// ---------------------------------------------------------------------------

func TestToFTS5Query(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"hello", `"hello"*`},
		// Only the LAST term gets prefix expansion — starring every term made
		// long queries enumerate huge posting ranges; porter stemming covers
		// interior-term morphology.
		{"hello world", `"hello" AND "world"*`},
		{`"exact phrase" loose`, `"exact phrase" AND "loose"*`},
		{`"single"`, `"single"`},
		// Dots: version strings must not produce FTS5 syntax errors
		{"v3.9.7", `"v3.9.7"*`},
		{"2026.4.10", `"2026.4.10"*`},
		// Hyphens
		{"real-time sync", `"real-time" AND "sync"*`},
		// FTS5 operator words must be neutralized. NOT is not a stopword.
		{"cats NOT dogs", `"cats" AND "NOT" AND "dogs"*`},
		// Stopwords are dropped when meaningful terms remain…
		{"how does the state of calm improve suggestions", `"how" AND "does" AND "state" AND "calm" AND "improve" AND "suggestions"*`},
		{"deploy to the server", `"deploy" AND "server"*`},
		// …but an all-stopword query keeps everything.
		{"to be or", `"to" AND "be" AND "or"*`},
		{"AND", `"AND"*`},
		// Punctuation-only tokens are dropped
		{"foo ---", `"foo"*`},
		{"...", ""},
		{"", ""},
		// Embedded double quotes are escaped by doubling
		{`say"cheese`, `"say""cheese"*`},
	}
	for _, c := range cases {
		if got := toFTS5Query(c.in); got != c.want {
			t.Errorf("toFTS5Query(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ---------------------------------------------------------------------------
// parseRefRange
// ---------------------------------------------------------------------------

func TestParseRefRange(t *testing.T) {
	cases := []struct {
		in    string
		base  string
		from  int
		count int
	}{
		{"file.md", "file.md", 0, 0},
		{"file.md:100", "file.md", 100, 0},
		{"file.md:120:40", "file.md", 120, 40},
		{"#abc123:5", "#abc123", 5, 0},
		{"dir/file.md:1:10", "dir/file.md", 1, 10},
		// negative from is parsed; renderDocument clamps it
		{"file.md:-5", "file.md", -5, 0},
		// colon in the name but no numeric suffix
		{"weird:name.md", "weird:name.md", 0, 0},
	}
	for _, c := range cases {
		base, from, count := parseRefRange(c.in)
		if base != c.base || from != c.from || count != c.count {
			t.Errorf("parseRefRange(%q) = (%q,%d,%d), want (%q,%d,%d)",
				c.in, base, from, count, c.base, c.from, c.count)
		}
	}
}

// ---------------------------------------------------------------------------
// renderDocument
// ---------------------------------------------------------------------------

func TestRenderDocument(t *testing.T) {
	doc := &Document{
		Path:       "notes.md",
		DocID:      "abc123",
		Collection: "test",
		AbsPath:    "/tmp/notes.md",
		Content:    "alpha\nbravo\ncharlie\ndelta\n",
	}

	out := renderDocument(doc, 0, 0, true, false)
	if !strings.HasPrefix(out, "qmd://test/notes.md #abc123 (lines 1-4 of 4)\n") {
		t.Errorf("unexpected header: %q", out)
	}
	if !strings.Contains(out, "2→bravo\n") {
		t.Errorf("missing numbered line: %q", out)
	}

	out = renderDocument(doc, 2, 2, true, false)
	if !strings.Contains(out, "(lines 2-3 of 4)") || strings.Contains(out, "alpha") || strings.Contains(out, "delta") {
		t.Errorf("range slice wrong: %q", out)
	}

	// Negative/overshoot from clamps instead of silently returning tail content
	out = renderDocument(doc, -7, 0, true, false)
	if !strings.Contains(out, "(lines 1-4 of 4)") {
		t.Errorf("negative from not clamped: %q", out)
	}
	out = renderDocument(doc, 99, 0, true, false)
	if !strings.Contains(out, "(lines 4-4 of 4)") {
		t.Errorf("overshoot from not clamped: %q", out)
	}

	// no line numbers + full path
	out = renderDocument(doc, 0, 0, false, true)
	if !strings.HasPrefix(out, "/tmp/notes.md #abc123") || strings.Contains(out, "1→") {
		t.Errorf("fullPath/no-numbers wrong: %q", out)
	}
}

// ---------------------------------------------------------------------------
// Store-level tests against a real temp SQLite index
// ---------------------------------------------------------------------------

func newTestStore(t *testing.T) *Store {
	t.Helper()
	store, err := NewStore(filepath.Join(t.TempDir(), "test.sqlite"))
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(store.Close)
	return store
}

// writeDoc creates a file on disk and indexes it under the collection.
func writeDoc(t *testing.T, store *Store, colID int64, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertDocument(colID, name, extractTitle(content, name), content); err != nil {
		t.Fatalf("UpsertDocument(%s): %v", name, err)
	}
}

func TestSearchBM25SpecialCharQueries(t *testing.T) {
	store := newTestStore(t)
	dir := t.TempDir()
	colID, err := store.UpsertCollection("test", dir, "**/*.md", "")
	if err != nil {
		t.Fatal(err)
	}
	writeDoc(t, store, colID, dir, "release.md", "# Release v3.9.7\n\nThe v3.9.7 build ships real-time sync and 2026.4.10 support.\n")
	writeDoc(t, store, colID, dir, "other.md", "# Unrelated\n\nNothing to see here.\n")

	// Every one of these previously produced an FTS5 syntax error.
	for _, q := range []string{"v3.9.7", "real-time", "2026.4.10", "real-time AND sync", "v3.9.7:"} {
		results, err := store.SearchBM25(q, 10)
		if err != nil {
			t.Fatalf("SearchBM25(%q) errored: %v", q, err)
		}
		if len(results) == 0 {
			t.Errorf("SearchBM25(%q) found nothing", q)
		} else if results[0].Path != "release.md" {
			t.Errorf("SearchBM25(%q) top hit = %s, want release.md", q, results[0].Path)
		}
	}

	// Punctuation-only query must not error
	if _, err := store.SearchBM25("...", 10); err != nil {
		t.Errorf("punctuation-only query errored: %v", err)
	}
}

func TestGetDocumentReadsContentFromDisk(t *testing.T) {
	store := newTestStore(t)
	dir := t.TempDir()
	colID, err := store.UpsertCollection("kb", dir, "**/*.md", "")
	if err != nil {
		t.Fatal(err)
	}
	writeDoc(t, store, colID, dir, "guide.md", "# Guide\n\nline three\nline four\nline five\n")

	doc, from, count, err := store.GetDocument("guide.md")
	if err != nil {
		t.Fatalf("GetDocument: %v", err)
	}
	if doc.Content == "" || !strings.Contains(doc.Content, "line three") {
		t.Errorf("content not loaded from disk: %q", doc.Content)
	}
	if doc.Collection != "kb" || from != 0 || count != 0 {
		t.Errorf("metadata wrong: col=%s from=%d count=%d", doc.Collection, from, count)
	}

	// docid ref + range suffix
	doc2, from, count, err := store.GetDocument("#" + doc.DocID + ":3:2")
	if err != nil {
		t.Fatalf("GetDocument by docid+range: %v", err)
	}
	if from != 3 || count != 2 || doc2.Path != "guide.md" {
		t.Errorf("range parse wrong: from=%d count=%d path=%s", from, count, doc2.Path)
	}

	// qmd:// URI form
	if _, _, _, err := store.GetDocument("qmd://kb/guide.md"); err != nil {
		t.Errorf("qmd:// ref failed: %v", err)
	}
}

func TestMultiGetContentAndSkips(t *testing.T) {
	store := newTestStore(t)
	dir := t.TempDir()
	colID, err := store.UpsertCollection("kb", dir, "**/*.md", "")
	if err != nil {
		t.Fatal(err)
	}
	writeDoc(t, store, colID, dir, "small.md", "# Small\n\ntiny\n")
	writeDoc(t, store, colID, dir, "big.md", "# Big\n\n"+strings.Repeat("x", 2000)+"\n")

	docs, skipped, err := store.MultiGet("*.md", 100)
	if err != nil {
		t.Fatalf("MultiGet: %v", err)
	}
	if len(docs) != 1 || docs[0].Path != "small.md" || !strings.Contains(docs[0].Content, "tiny") {
		t.Errorf("docs wrong: %+v", docs)
	}
	if len(skipped) != 1 || skipped[0].Path != "big.md" || skipped[0].Size < 2000 {
		t.Errorf("skipped wrong: %+v", skipped)
	}

	// comma list + dedup between overlapping patterns
	docs, _, err = store.MultiGet("small.md, *.md", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Errorf("expected 2 deduped docs, got %d", len(docs))
	}
}

func TestPendingEmbedTracking(t *testing.T) {
	store := newTestStore(t)
	dir := t.TempDir()
	colID, err := store.UpsertCollection("kb", dir, "**/*.md", "")
	if err != nil {
		t.Fatal(err)
	}
	content := "# Doc\n\nbody\n"
	writeDoc(t, store, colID, dir, "doc.md", content)
	hash := contentHash(content)
	fp := embedFingerprint()

	assertPending := func(want int, msg string) {
		t.Helper()
		n, err := store.CountUnembedded(fp, "")
		if err != nil {
			t.Fatal(err)
		}
		if n != want {
			t.Errorf("%s: pending = %d, want %d", msg, n, want)
		}
	}

	assertPending(1, "no chunks yet")

	chunks := []Chunk{
		{Hash: hash, Seq: 0, Pos: 0, Text: "chunk zero"},
		{Hash: hash, Seq: 1, Pos: 10, Text: "chunk one"},
	}
	if err := store.StoreChunks(hash, chunks); err != nil {
		t.Fatal(err)
	}
	assertPending(1, "chunked but not embedded")

	vec := []float32{1, 2, 3}
	if err := store.StoreVector(hash, 0, vec, fp); err != nil {
		t.Fatal(err)
	}
	// This was the v0.3.0 bug: one embedded chunk made the whole doc count
	// as embedded and it never resumed.
	assertPending(1, "partially embedded")

	if err := store.StoreVector(hash, 1, vec, fp); err != nil {
		t.Fatal(err)
	}
	assertPending(0, "fully embedded")

	// A different fingerprint (model/chunker change) marks it stale again.
	if n, err := store.CountUnembedded("other-model|cv9", ""); err != nil || n != 1 {
		t.Errorf("stale fingerprint not detected: n=%d err=%v", n, err)
	}
	stale, err := store.HasStaleFP(hash, "other-model|cv9")
	if err != nil || !stale {
		t.Errorf("HasStaleFP = %v,%v want true,nil", stale, err)
	}

	// UnembeddedChunks under the current fp returns nothing…
	uc, err := store.UnembeddedChunks(hash)
	if err != nil {
		t.Fatal(err)
	}
	if len(uc) != 0 {
		t.Errorf("expected no unembedded chunks, got %d", len(uc))
	}
}

func TestSkipNextUnembeddedProgresses(t *testing.T) {
	store := newTestStore(t)
	dir := t.TempDir()
	colID, err := store.UpsertCollection("kb", dir, "**/*.md", "")
	if err != nil {
		t.Fatal(err)
	}
	content := "# Stuck\n\nproblem doc\n"
	writeDoc(t, store, colID, dir, "stuck.md", content)
	hash := contentHash(content)
	fp := embedFingerprint()

	// Scoped skip must not touch documents outside its collection
	if h, err := store.SkipNextUnembedded("other-collection"); err != nil || h != "" {
		t.Errorf("skip in empty scope should be a no-op, got hash=%q err=%v", h, err)
	}
	if n, _ := store.CountUnembedded(fp, ""); n != 1 {
		t.Errorf("out-of-scope skip touched the pending doc")
	}

	// Shape 1: no chunks at all
	if h, err := store.SkipNextUnembedded("kb"); err != nil || h == "" {
		t.Fatalf("skip failed: hash=%q err=%v", h, err)
	}
	if n, _ := store.CountUnembedded(fp, ""); n != 0 {
		t.Errorf("skip did not clear chunkless doc: pending=%d", n)
	}

	// Shape 2: chunks exist but one vector is missing
	if err := store.StoreChunks(hash, []Chunk{
		{Hash: hash, Seq: 0, Pos: 0, Text: "a"},
		{Hash: hash, Seq: 1, Pos: 5, Text: "b"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.StoreVector(hash, 0, []float32{1}, fp); err != nil {
		t.Fatal(err)
	}
	if h, err := store.SkipNextUnembedded(""); err != nil || h != hash {
		t.Fatalf("skip failed: hash=%q err=%v", h, err)
	}
	if n, _ := store.CountUnembedded(fp, ""); n != 0 {
		t.Errorf("skip did not clear partially embedded doc: pending=%d", n)
	}
}

// TestBM25SnippetsFromDisk: the FTS table is contentless, so FTS5's own
// snippet() always returned "" — BM25 snippets must instead be extracted
// from the on-disk file, with a line number for the citation.
func TestBM25SnippetsFromDisk(t *testing.T) {
	store := newTestStore(t)
	dir := t.TempDir()
	colID, err := store.UpsertCollection("kb", dir, "**/*.md", "")
	if err != nil {
		t.Fatal(err)
	}
	writeDoc(t, store, colID, dir, "snip.md", "# Snippet\n\nthe quick brown fox jumps over the lazy dog\n")

	results, err := store.SearchBM25("quick brown", 5)
	if err != nil {
		t.Fatalf("SearchBM25: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}
	r := results[0]
	if !strings.Contains(r.Snippet, ">>>quick<<<") {
		t.Errorf("snippet missing highlighted term: %q", r.Snippet)
	}
	if r.Line != 3 {
		t.Errorf("snippet line = %d, want 3", r.Line)
	}
}
