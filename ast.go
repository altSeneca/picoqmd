// AST-aware chunking support via dcosson/treesitter-go (pure Go, no cgo).
//
// Port of qmd's src/ast.ts + chunking core (src/store.ts): language
// detection, per-language S-expression queries, score map, break-point
// extraction and merging, and a byte-offset chunker shared by both
// regex-only and AST-aware paths. All functions degrade gracefully —
// parse failures or unsupported languages fall back to regex-only
// chunking (byte-identical to ChunkDocument).
package main

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	ts "github.com/dcosson/treesitter-go"
	iparser "github.com/dcosson/treesitter-go/parser"
	"github.com/dcosson/treesitter-go/languages/golang"
	"github.com/dcosson/treesitter-go/languages/javascript"
	"github.com/dcosson/treesitter-go/languages/python"
	"github.com/dcosson/treesitter-go/languages/rust"
	"github.com/dcosson/treesitter-go/languages/tsx"
	"github.com/dcosson/treesitter-go/languages/typescript"
)

// SupportedLanguage mirrors qmd's SupportedLanguage.
type SupportedLanguage string

const (
	LangTypeScript SupportedLanguage = "typescript"
	LangTSX        SupportedLanguage = "tsx"
	LangJavaScript SupportedLanguage = "javascript"
	LangPython     SupportedLanguage = "python"
	LangGo         SupportedLanguage = "go"
	LangRust       SupportedLanguage = "rust"
)

var extensionMap = map[string]SupportedLanguage{
	".ts":  LangTypeScript,
	".tsx": LangTSX,
	".js":  LangJavaScript,
	".jsx": LangTSX,
	".mts": LangTypeScript,
	".cts": LangTypeScript,
	".mjs": LangJavaScript,
	".cjs": LangJavaScript,
	".py":  LangPython,
	".go":  LangGo,
	".rs":  LangRust,
}

// detectLanguage returns the tree-sitter language for a file path,
// or "" for unsupported extensions (including .md).
func detectLanguage(path string) SupportedLanguage {
	ext := strings.ToLower(filepath.Ext(path))
	if lang, ok := extensionMap[ext]; ok {
		return lang
	}
	return ""
}

// languageQueries ports qmd's LANGUAGE_QUERIES verbatim.
var languageQueries = map[SupportedLanguage]string{
	LangTypeScript: `
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (interface_declaration) @iface
    (type_alias_declaration) @type
    (enum_declaration) @enum
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  `,
	LangTSX: `
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (interface_declaration) @iface
    (type_alias_declaration) @type
    (enum_declaration) @enum
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  `,
	LangJavaScript: `
    (export_statement) @export
    (class_declaration) @class
    (function_declaration) @func
    (method_definition) @method
    (import_statement) @import
    (lexical_declaration (variable_declarator value: (arrow_function))) @func
    (lexical_declaration (variable_declarator value: (function_expression))) @func
  `,
	LangPython: `
    (class_definition) @class
    (function_definition) @func
    (decorated_definition) @decorated
    (import_statement) @import
    (import_from_statement) @import
  `,
	LangGo: `
    (type_declaration) @type
    (function_declaration) @func
    (method_declaration) @method
    (import_declaration) @import
  `,
	LangRust: `
    (struct_item) @struct
    (impl_item) @impl
    (function_item) @func
    (trait_item) @trait
    (enum_item) @enum
    (use_declaration) @import
    (type_item) @type
    (mod_item) @mod
  `,
}

// scoreMap ports qmd's SCORE_MAP verbatim.
var scoreMap = map[string]int{
	"class":     100,
	"iface":     100,
	"struct":    100,
	"trait":     100,
	"impl":      100,
	"mod":       100,
	"export":    90,
	"func":      90,
	"method":    90,
	"decorated": 90,
	"type":      80,
	"enum":      80,
	"import":    60,
}

// BreakPoint is a candidate chunk boundary. Pos is a byte offset into
// the source; Score follows qmd's scale so the existing distance-decay
// in the chunker works unchanged.
type BreakPoint struct {
	Pos   int
	Score int
	Type  string
}

func languageFor(lang SupportedLanguage) *ts.Language {
	switch lang {
	case LangTypeScript:
		return typescript.Language()
	case LangTSX:
		return tsx.Language()
	case LangJavaScript:
		return javascript.Language()
	case LangPython:
		return python.Language()
	case LangGo:
		return golang.Language()
	case LangRust:
		return rust.Language()
	default:
		return nil
	}
}

// getASTBreakPoints parses content and returns break points at AST node
// boundaries. Returns an empty slice for unsupported languages or on any
// parse/query failure. Never returns an error — callers fall back to
// regex-only chunking.
func getASTBreakPoints(content, path string) []BreakPoint {
	lang := detectLanguage(path)
	if lang == "" {
		return nil
	}
	tsLang := languageFor(lang)
	if tsLang == nil {
		return nil
	}
	qsrc, ok := languageQueries[lang]
	if !ok {
		return nil
	}

	ctx := context.Background()
	p := iparser.NewParser()
	p.SetLanguage(tsLang)
	tree := p.ParseString(ctx, []byte(content))
	if tree == nil {
		return nil
	}
	root := tree.RootNode()

	q, err := ts.NewQuery(tsLang, qsrc)
	if err != nil {
		return nil
	}
	cursor := ts.NewQueryCursor(q)
	cursor.Exec(root)

	seen := make(map[int]BreakPoint)
	for {
		m, ok := cursor.NextMatch()
		if !ok {
			break
		}
		for _, c := range m.Captures {
			name := q.CaptureNameForID(c.Index)
			score, ok := scoreMap[name]
			if !ok {
				score = 20
			}
			pos := int(c.Node.StartByte())
			if existing, dup := seen[pos]; !dup || score > existing.Score {
				seen[pos] = BreakPoint{Pos: pos, Score: score, Type: "ast:" + name}
			}
		}
	}

	out := make([]BreakPoint, 0, len(seen))
	for _, bp := range seen {
		out = append(out, bp)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Pos < out[j].Pos })
	return out
}

// mergeBreakPoints combines two break-point sets, keeping the highest
// score at each position. Result is sorted by position.
func mergeBreakPoints(a, b []BreakPoint) []BreakPoint {
	seen := make(map[int]BreakPoint, len(a)+len(b))
	for _, bp := range a {
		if existing, ok := seen[bp.Pos]; !ok || bp.Score > existing.Score {
			seen[bp.Pos] = bp
		}
	}
	for _, bp := range b {
		if existing, ok := seen[bp.Pos]; !ok || bp.Score > existing.Score {
			seen[bp.Pos] = bp
		}
	}
	out := make([]BreakPoint, 0, len(seen))
	for _, bp := range seen {
		out = append(out, bp)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Pos < out[j].Pos })
	return out
}

// ---------------------------------------------------------------------------
// Chunk strategy — "regex" (default, legacy output) or "auto" (AST for code)
// ---------------------------------------------------------------------------

// chunkerVersionAST is the fingerprint version for AST-cut chunks. Regex
// cuts stay on chunkerVersion (cv1) so existing indexes never re-embed
// on upgrade; opting into auto moves to cv2 and marks docs pending once.
const chunkerVersionAST = "cv2"

// activeChunkStrategy returns "auto" only when PICOQMD_CHUNK_STRATEGY=auto
// (case-insensitive, trimmed). Anything else — unset, empty, or invalid —
// is "regex", reproducing the pre-AST behavior exactly.
func activeChunkStrategy() string {
	if s := strings.TrimSpace(os.Getenv("PICOQMD_CHUNK_STRATEGY")); strings.EqualFold(s, "auto") {
		return "auto"
	}
	return "regex"
}

// parseChunkStrategy validates a --chunk-strategy flag value.
func parseChunkStrategy(s string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "regex":
		return "regex", nil
	case "auto":
		return "auto", nil
	default:
		return "", &chunkStrategyError{s}
	}
}

type chunkStrategyError struct{ got string }

func (e *chunkStrategyError) Error() string {
	return `--chunk-strategy must be "auto" or "regex" (got "` + e.got + `")`
}

// chunkerVersionFor returns the fingerprint version for a strategy.
func chunkerVersionFor(strategy string) string {
	if strategy == "auto" {
		return chunkerVersionAST
	}
	return chunkerVersion
}

// ---------------------------------------------------------------------------
// Byte-offset chunking core — port of qmd's chunkDocumentWithBreakPoints
// ---------------------------------------------------------------------------

// codeFence is a byte range [start, end) covering a fenced block. Cuts
// inside a fence are forbidden (qmd's CodeFenceRegion).
type codeFence struct{ start, end int }

// scanBreakPoints assigns every line start a break score via breakScore
// (same 100..1 scale qmd uses), returned as byte-offset break points.
func scanBreakPoints(content string) []BreakPoint {
	var out []BreakPoint
	off := 0
	for _, line := range strings.SplitAfter(content, "\n") {
		body := strings.TrimSuffix(line, "\n")
		out = append(out, BreakPoint{Pos: off, Score: breakScore(body), Type: "line"})
		off += len(line)
	}
	return out
}

// findCodeFences pairs ``` lines into fenced regions (byte offsets).
// An unclosed fence extends to end of document.
func findCodeFences(content string) []codeFence {
	var out []codeFence
	off := 0
	inFence := false
	fenceStart := 0
	for _, line := range strings.SplitAfter(content, "\n") {
		if strings.HasPrefix(strings.TrimSpace(strings.TrimSuffix(line, "\n")), "```") {
			if !inFence {
				fenceStart = off
				inFence = true
			} else {
				out = append(out, codeFence{start: fenceStart, end: off + len(line)})
				inFence = false
			}
		}
		off += len(line)
	}
	if inFence {
		out = append(out, codeFence{start: fenceStart, end: len(content)})
	}
	return out
}

func insideCodeFence(pos int, fences []codeFence) bool {
	for _, f := range fences {
		if pos > f.start && pos < f.end {
			return true
		}
	}
	return false
}

// findBestCutoff picks the highest decay-adjusted break point at or
// before target within window — qmd's squared-distance decay verbatim.
func findBestCutoff(bps []BreakPoint, target, window int, decay float64, fences []codeFence) int {
	windowStart := target - window
	bestScore := -1.0
	bestPos := target
	for _, bp := range bps {
		if bp.posBefore(windowStart) {
			continue
		}
		if bp.Pos > target {
			break
		}
		if insideCodeFence(bp.Pos, fences) {
			continue
		}
		nd := float64(target-bp.Pos) / float64(window)
		finalScore := float64(bp.Score) * (1.0 - nd*nd*decay)
		if finalScore > bestScore {
			bestScore = finalScore
			bestPos = bp.Pos
		}
	}
	return bestPos
}

func (bp BreakPoint) posBefore(windowStart int) bool { return bp.Pos < windowStart }

// backToRuneBoundary moves pos down to a UTF-8 character boundary so
// slicing never splits a multi-byte rune.
func backToRuneBoundary(s string, pos int) int {
	for pos > 0 && pos < len(s) && !utf8.RuneStart(s[pos]) {
		pos--
	}
	return pos
}

// chunkWithBreakPoints cuts content into ~chunkTarget-token pieces at the
// best break points with 15% overlap — qmd's algorithm on byte offsets.
func chunkWithBreakPoints(content string, bps []BreakPoint, fences []codeFence) []Chunk {
	hash := contentHash(content)
	maxChars := chunkTarget * 4
	overlapChars := chunkOverlap * 4
	windowChars := chunkLookback * 4
	if len(content) <= maxChars {
		return []Chunk{{Hash: hash, Seq: 0, Pos: 0, Text: content}}
	}
	var chunks []Chunk
	charPos := 0
	for charPos < len(content) {
		targetEnd := charPos + maxChars
		if targetEnd > len(content) {
			targetEnd = len(content)
		}
		endPos := targetEnd
		if endPos < len(content) {
			if cut := findBestCutoff(bps, targetEnd, windowChars, 0.7, fences); cut > charPos && cut <= targetEnd {
				endPos = cut
			} else {
				endPos = backToRuneBoundary(content, targetEnd)
			}
		}
		if endPos <= charPos {
			endPos = backToRuneBoundary(content, min(charPos+maxChars, len(content)))
			if endPos <= charPos {
				endPos = len(content)
			}
		}
		chunks = append(chunks, Chunk{Hash: hash, Seq: len(chunks), Pos: charPos, Text: content[charPos:endPos]})
		if endPos >= len(content) {
			break
		}
		charPos = endPos - overlapChars
		if charPos <= chunks[len(chunks)-1].Pos {
			charPos = endPos
		}
	}
	return chunks
}

// ChunkDocumentForPath chunks content like ChunkDocument, except with
// strategy "auto" code files are cut at AST function/class/import
// boundaries instead of arbitrary line positions. Markdown and unknown
// types always use the regex path; strategy "" means "regex".
func ChunkDocumentForPath(content, path, strategy string) []Chunk {
	if strategy == "" {
		strategy = "regex"
	}
	if strategy == "auto" && detectLanguage(path) != "" {
		if astPoints := getASTBreakPoints(content, path); len(astPoints) > 0 {
			merged := mergeBreakPoints(scanBreakPoints(content), astPoints)
			return chunkWithBreakPoints(content, merged, findCodeFences(content))
		}
	}
	return ChunkDocument(content)
}
