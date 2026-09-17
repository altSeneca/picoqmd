package main

import (
	"strings"
	"testing"
)

func TestDetectLanguage(t *testing.T) {
	cases := map[string]SupportedLanguage{
		"a.ts":  LangTypeScript,
		"a.tsx": LangTSX,
		"a.js":  LangJavaScript,
		"a.jsx": LangTSX,
		"a.py":  LangPython,
		"a.go":  LangGo,
		"a.rs":  LangRust,
		"a.md":  "",
		"a.txt": "",
		"a":     "",
	}
	for path, want := range cases {
		if got := detectLanguage(path); got != want {
			t.Errorf("detectLanguage(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestGetASTBreakPointsGo(t *testing.T) {
	src := "package main\n\nimport \"fmt\"\n\ntype Server struct {\n\tport int\n}\n\nfunc main() {}\n"
	points := getASTBreakPoints(src, "a.go")
	if len(points) == 0 {
		t.Fatal("expected break points, got none")
	}
	// import scores 60, type scores 80, func scores 90 (qmd SCORE_MAP)
	byType := map[string]int{}
	for _, p := range points {
		byType[p.Type] = p.Score
		if !strings.HasPrefix(p.Type, "ast:") {
			t.Errorf("type %q missing ast: prefix", p.Type)
		}
	}
	if byType["ast:import"] != 60 {
		t.Errorf("ast:import score = %d, want 60", byType["ast:import"])
	}
	if byType["ast:type"] != 80 {
		t.Errorf("ast:type score = %d, want 80", byType["ast:type"])
	}
	if byType["ast:func"] != 90 {
		t.Errorf("ast:func score = %d, want 90", byType["ast:func"])
	}
}

func TestGetASTBreakPointScores(t *testing.T) {
	cases := []struct {
		path string
		src  string
		want map[string]int
	}{
		{"a.ts", "export class Foo {}\nexport function bar() {}",
			map[string]int{"ast:export": 90}},
		{"a.py", "class Foo:\n    pass\n\ndef bar():\n    pass",
			map[string]int{"ast:class": 100, "ast:func": 90}},
		{"a.go", "package main\n\ntype Server struct {\n    port int\n}\n\nfunc main() {}",
			map[string]int{"ast:type": 80, "ast:func": 90}},
		{"a.rs", "enum State {\n    On,\n    Off,\n}\n\nfn main() {}",
			map[string]int{"ast:enum": 80, "ast:func": 90}},
		{"a.js", "export function foo() {}\nconst bar = () => {};",
			map[string]int{"ast:export": 90, "ast:func": 90}},
	}
	for _, c := range cases {
		points := getASTBreakPoints(c.src, c.path)
		if len(points) == 0 {
			t.Errorf("%s: no break points", c.path)
			continue
		}
		byType := map[string]int{}
		for _, p := range points {
			byType[p.Type] = p.Score
		}
		for typ, score := range c.want {
			if byType[typ] != score {
				t.Errorf("%s: %s score = %d, want %d (all: %v)", c.path, typ, byType[typ], score, byType)
			}
		}
	}
}

func TestGetASTBreakPointsUnsupported(t *testing.T) {
	if pts := getASTBreakPoints("# hello\n\nworld\n", "readme.md"); len(pts) != 0 {
		t.Errorf("markdown should yield no AST points, got %v", pts)
	}
	if pts := getASTBreakPoints("x", "file.txt"); len(pts) != 0 {
		t.Errorf("unknown ext should yield no AST points, got %v", pts)
	}
}

func TestMergeBreakPoints(t *testing.T) {
	a := []BreakPoint{{Pos: 10, Score: 20, Type: "blank"}, {Pos: 50, Score: 1, Type: "newline"}}
	b := []BreakPoint{{Pos: 10, Score: 90, Type: "ast:func"}, {Pos: 75, Score: 100, Type: "ast:class"}}
	merged := mergeBreakPoints(a, b)
	if len(merged) != 3 {
		t.Fatalf("len = %d, want 3", len(merged))
	}
	for _, bp := range merged {
		if bp.Pos == 10 && bp.Score != 90 {
			t.Errorf("pos 10 score = %d, want 90 (AST wins)", bp.Score)
		}
	}
	if !(merged[0].Pos < merged[1].Pos && merged[1].Pos < merged[2].Pos) {
		t.Errorf("not sorted: %v", merged)
	}
}

func TestParseChunkStrategy(t *testing.T) {
	if s, err := parseChunkStrategy("auto"); err != nil || s != "auto" {
		t.Errorf("auto -> %q,%v", s, err)
	}
	if s, err := parseChunkStrategy(""); err != nil || s != "regex" {
		t.Errorf("empty -> %q,%v", s, err)
	}
	if s, err := parseChunkStrategy("REGEX"); err != nil || s != "regex" {
		t.Errorf("REGEX -> %q,%v", s, err)
	}
	if _, err := parseChunkStrategy("bogus"); err == nil {
		t.Error("bogus should error")
	}
}

func TestChunkerVersionFor(t *testing.T) {
	if chunkerVersionFor("regex") != chunkerVersion {
		t.Error("regex must stay on cv1 so existing indexes never re-embed on upgrade")
	}
	if chunkerVersionFor("auto") != "cv2" {
		t.Error("auto must use cv2 so opting in marks docs pending once")
	}
}

// countSplitFunctions counts how many of the numbered handler functions
// are spread across more than one chunk.
func countSplitFunctions(t *testing.T, src string, chunks []Chunk, n int) int {
	t.Helper()
	splits := 0
	// Locate each function by its unique name.
	for i := 0; i < n; i++ {
		name := "handler" + itoa(i) + "("
		funcStart := strings.Index(src, name)
		if funcStart < 0 {
			t.Fatalf("func %d not found", i)
		}
		// Function extends to the next function or EOF.
		funcEnd := len(src)
		if i+1 < n {
			next := strings.Index(src, "handler"+itoa(i+1)+"(")
			if next > funcStart {
				funcEnd = next
			}
		}
		seen := map[int]bool{}
		for ci := range chunks {
			cStart := chunks[ci].Pos
			cEnd := cStart + len(chunks[ci].Text)
			if cStart < funcEnd && cEnd > funcStart {
				seen[ci] = true
			}
		}
		if len(seen) > 1 {
			splits++
		}
		// Advance src window so Index finds the next function: names are
		// unique per i, so no window bookkeeping is needed.
		_ = funcEnd
	}
	return splits
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [8]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}

func largeGoFile(n int) string {
	var sb strings.Builder
	sb.WriteString("package main\n\n")
	for i := 0; i < n; i++ {
		// Small functions (~10 short lines each) so several fit per
		// 900-token chunk — cuts should land on boundaries, not mid-body.
		sb.WriteString("func handler" + itoa(i) + "(req int) int {\n")
		sb.WriteString("\tx := req + " + itoa(i) + "\n")
		sb.WriteString("\tif x > 0 {\n\t\treturn x\n\t}\n")
		sb.WriteString("\ty := x * 2\n")
		sb.WriteString("\treturn y - " + itoa(i) + "\n}\n\n")
	}
	return sb.String()
}

func TestChunkASTSplitsFewerFunctions(t *testing.T) {
	const n = 60
	src := largeGoFile(n)
	regexChunks := ChunkDocumentForPath(src, "handlers.go", "regex")
	astChunks := ChunkDocumentForPath(src, "handlers.go", "auto")
	if len(astChunks) == 0 || len(regexChunks) == 0 {
		t.Fatalf("no chunks: regex=%d ast=%d", len(regexChunks), len(astChunks))
	}
	regexSplits := countSplitFunctions(t, src, regexChunks, n)
	astSplits := countSplitFunctions(t, src, astChunks, n)
	t.Logf("regex splits=%d/%d across %d chunks; ast splits=%d/%d across %d chunks",
		regexSplits, n, len(regexChunks), astSplits, n, len(astChunks))
	if astSplits > regexSplits {
		t.Errorf("AST splits more functions (%d) than regex (%d)", astSplits, regexSplits)
	}
}

func TestChunkMarkdownIdenticalInAuto(t *testing.T) {
	var sb strings.Builder
	for i := 0; i < 15; i++ {
		sb.WriteString("# Section " + itoa(i) + "\n\nLorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor. " +
			"Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo. " +
			"Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla.\n\n")
	}
	md := sb.String()
	regexChunks := ChunkDocument(md)
	autoChunks := ChunkDocumentForPath(md, "readme.md", "auto")
	if len(autoChunks) != len(regexChunks) {
		t.Fatalf("markdown chunk count differs: regex=%d auto=%d", len(regexChunks), len(autoChunks))
	}
	for i := range regexChunks {
		if autoChunks[i].Text != regexChunks[i].Text {
			t.Errorf("chunk %d text differs", i)
		}
	}
}

func TestChunkSmallFileSingleChunk(t *testing.T) {
	for _, strategy := range []string{"regex", "auto"} {
		chunks := ChunkDocumentForPath("export const x = 1;", "s.ts", strategy)
		if len(chunks) != 1 {
			t.Errorf("strategy %s: got %d chunks, want 1", strategy, len(chunks))
		}
	}
}

func TestChunkRegexPathUnchanged(t *testing.T) {
	src := largeGoFile(12)
	a := ChunkDocument(src)
	b := ChunkDocumentForPath(src, "handlers.go", "regex")
	if len(a) != len(b) {
		t.Fatalf("regex path chunk count differs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].Text != b[i].Text || a[i].Pos != b[i].Pos {
			t.Fatalf("regex path chunk %d differs", i)
		}
	}
}
