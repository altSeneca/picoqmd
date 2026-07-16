//go:build (freebsd || linux || windows || darwin) && (amd64 || arm64)

package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// ---------------------------------------------------------------------------
// LLM Engine — local GGUF model inference via yzma (hybridgroup/yzma)
// ---------------------------------------------------------------------------

const (
	idleTimeout     = 5 * time.Minute
	llamaCppRelease = "b8121"
)

type LLMEngine struct {
	mu        sync.Mutex
	modelsDir string
	libPath   string
	inited    bool

	embedModel  llama.Model
	embedCtx    llama.Context
	embedCount  int
	rerankModel llama.Model
	rerankCtx   llama.Context
	expandModel llama.Model
	expandCtx   llama.Context

	lastUsed  time.Time
	idleTimer *time.Timer
}

func NewLLMEngine(cache string) Embedder {
	modelsDir := filepath.Join(cache, "models")
	os.MkdirAll(modelsDir, 0o755)

	libPath := os.Getenv("PICOQMD_LIB")
	if libPath == "" {
		libPath = os.Getenv("YZMA_LIB")
	}

	return &LLMEngine{modelsDir: modelsDir, libPath: libPath}
}

func (e *LLMEngine) ModelsDir() string { return e.modelsDir }
func (e *LLMEngine) EnsureLib() error  { return e.ensureLib() }

// init loads the llama.cpp shared library (once).
func (e *LLMEngine) init() error {
	if e.inited {
		return nil
	}
	if e.libPath == "" {
		if err := e.ensureLib(); err != nil {
			return err
		}
	}
	if e.libPath == "" {
		return fmt.Errorf("llama.cpp library not found: set PICOQMD_LIB or YZMA_LIB env var, or run: picoqmd add <path>")
	}
	loadPath := e.libPath
	if info, err := os.Stat(loadPath); err == nil && !info.IsDir() {
		loadPath = filepath.Dir(loadPath)
	}
	if err := llama.Load(loadPath); err != nil {
		return fmt.Errorf("unable to load llama.cpp library: %w", err)
	}
	llama.LogSet(llama.LogSilent())
	llama.Init()
	e.inited = true
	return nil
}

func (e *LLMEngine) ensureEmbedModel() error {
	if e.embedModel != 0 {
		e.resetIdleTimer()
		return nil
	}

	if err := e.init(); err != nil {
		return err
	}

	modelPath := filepath.Join(e.modelsDir, defaultModels[0].Filename)
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("embedding model not found at %s — download with: picoqmd model download embedding", modelPath)
	}

	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("failed to load embedding model: %w", err)
	}
	if model == 0 {
		return fmt.Errorf("failed to load embedding model from %s", modelPath)
	}

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 2048
	ctxParams.NBatch = 2048
	ctxParams.NUbatch = 2048
	ctxParams.Embeddings = 1

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		llama.ModelFree(model)
		return fmt.Errorf("failed to create embedding context: %w", err)
	}

	e.embedModel = model
	e.embedCtx = ctx
	e.resetIdleTimer()
	return nil
}

func (e *LLMEngine) ensureRerankModel() error {
	if e.rerankModel != 0 {
		e.resetIdleTimer()
		return nil
	}

	if err := e.init(); err != nil {
		return err
	}

	modelPath := filepath.Join(e.modelsDir, defaultModels[1].Filename)
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("reranker model not found at %s", modelPath)
	}

	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("failed to load reranker model: %w", err)
	}

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 4096
	ctxParams.NBatch = 512

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		llama.ModelFree(model)
		return fmt.Errorf("failed to create reranker context: %w", err)
	}

	e.rerankModel = model
	e.rerankCtx = ctx
	e.resetIdleTimer()
	return nil
}

func (e *LLMEngine) ensureExpandModel() error {
	if e.expandModel != 0 {
		e.resetIdleTimer()
		return nil
	}

	if err := e.init(); err != nil {
		return err
	}

	modelPath := filepath.Join(e.modelsDir, defaultModels[2].Filename)
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("expansion model not found at %s", modelPath)
	}

	model, err := llama.ModelLoadFromFile(modelPath, llama.ModelDefaultParams())
	if err != nil {
		return fmt.Errorf("failed to load expansion model: %w", err)
	}

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 2048
	ctxParams.NBatch = 512

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		llama.ModelFree(model)
		return fmt.Errorf("failed to create expansion context: %w", err)
	}

	e.expandModel = model
	e.expandCtx = ctx
	e.resetIdleTimer()
	return nil
}

func (e *LLMEngine) resetIdleTimer() {
	e.lastUsed = time.Now()
	if e.idleTimer != nil {
		e.idleTimer.Stop()
	}
	e.idleTimer = time.AfterFunc(idleTimeout, func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		if time.Since(e.lastUsed) >= idleTimeout {
			e.freeModels()
		}
	})
}

func (e *LLMEngine) freeModels() {
	if e.embedCtx != 0 {
		llama.Free(e.embedCtx)
		e.embedCtx = 0
	}
	if e.embedModel != 0 {
		llama.ModelFree(e.embedModel)
		e.embedModel = 0
	}
	if e.rerankCtx != 0 {
		llama.Free(e.rerankCtx)
		e.rerankCtx = 0
	}
	if e.rerankModel != 0 {
		llama.ModelFree(e.rerankModel)
		e.rerankModel = 0
	}
	if e.expandCtx != 0 {
		llama.Free(e.expandCtx)
		e.expandCtx = 0
	}
	if e.expandModel != 0 {
		llama.ModelFree(e.expandModel)
		e.expandModel = 0
	}
}

func (e *LLMEngine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.idleTimer != nil {
		e.idleTimer.Stop()
	}
	e.freeModels()
	if e.inited {
		llama.Close()
		e.inited = false
	}
}

// ---------------------------------------------------------------------------
// Embed / BatchEmbed / ExpandQuery / Rerank
// ---------------------------------------------------------------------------

func (e *LLMEngine) Embed(text string, isQuery bool) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.ensureEmbedModel(); err != nil {
		return nil, err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	e.embedCount++
	if e.embedCount%100 == 0 {
		runtime.GC()
		llama.Free(e.embedCtx)
		ctxParams := llama.ContextDefaultParams()
		ctxParams.NCtx = 2048
		ctxParams.NBatch = 2048
		ctxParams.NUbatch = 2048
		ctxParams.Embeddings = 1
		ctx, err := llama.InitFromModel(e.embedModel, ctxParams)
		if err != nil {
			return nil, fmt.Errorf("failed to recreate embedding context: %w", err)
		}
		e.embedCtx = ctx
	}

	var input string
	if isQuery {
		input = fmt.Sprintf("task: search result | query: %s", text)
	} else {
		input = text
	}

	vocab := llama.ModelGetVocab(e.embedModel)
	tokens := llama.Tokenize(vocab, input, true, true)
	runtime.KeepAlive(input)
	runtime.KeepAlive(vocab)

	maxTokens := int(llama.NCtx(e.embedCtx))
	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}

	if mem, err := llama.GetMemory(e.embedCtx); err == nil && mem != 0 {
		llama.MemoryClear(mem, false)
	}

	batch := llama.BatchGetOne(tokens)
	if _, err := llama.Decode(e.embedCtx, batch); err != nil {
		return nil, fmt.Errorf("embedding decode failed: %w", err)
	}
	runtime.KeepAlive(tokens)

	nEmbd := llama.ModelNEmbd(e.embedModel)
	vec, err := llama.GetEmbeddingsSeq(e.embedCtx, 0, nEmbd)
	if err != nil {
		return nil, fmt.Errorf("get embeddings failed: %w", err)
	}
	if vec == nil {
		return nil, fmt.Errorf("no embeddings returned (model may not support embedding mode)")
	}

	result := make([]float32, len(vec))
	copy(result, vec)
	return result, nil
}

func (e *LLMEngine) BatchEmbed(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, text := range texts {
		vec, err := e.Embed(text, false)
		if err != nil {
			return nil, fmt.Errorf("batch embed [%d]: %w", i, err)
		}
		results[i] = vec
	}
	return results, nil
}

func (e *LLMEngine) ExpandQuery(query, intent string) ([]QueryExpansion, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.ensureExpandModel(); err != nil {
		return []QueryExpansion{
			{Query: query, Type: "lex"},
			{Query: query, Type: "vec"},
		}, nil
	}

	var prompt string
	if intent != "" {
		prompt = fmt.Sprintf("Expand this search query into two alternative queries that match the stated intent.\nIntent: %s\nQuery: %s\n", intent, query)
	} else {
		prompt = fmt.Sprintf("Expand this search query into two alternative queries.\nQuery: %s\n", query)
	}

	vocab := llama.ModelGetVocab(e.expandModel)
	tokens := llama.Tokenize(vocab, prompt, true, false)
	batch := llama.BatchGetOne(tokens)

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())
	defer llama.SamplerFree(sampler)

	var output strings.Builder
	maxTokens := int32(128)

	for i := int32(0); i < maxTokens; i++ {
		if _, err := llama.Decode(e.expandCtx, batch); err != nil {
			break
		}

		token := llama.SamplerSample(sampler, e.expandCtx, -1)
		if llama.VocabIsEOG(vocab, token) {
			break
		}

		buf := make([]byte, 64)
		n := llama.TokenToPiece(vocab, token, buf, 0, false)
		if n > 0 {
			output.Write(buf[:n])
		}

		batch = llama.BatchGetOne([]llama.Token{token})
	}

	expansions := parseExpansions(output.String(), query)
	return expansions, nil
}

func parseExpansions(output, original string) []QueryExpansion {
	lines := strings.Split(strings.TrimSpace(output), "\n")
	var expansions []QueryExpansion

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == original {
			continue
		}
		cleaned := line
		for _, prefix := range []string{"1.", "2.", "- ", "lex:", "vec:", "Lex:", "Vec:"} {
			cleaned = strings.TrimPrefix(cleaned, prefix)
		}
		cleaned = strings.TrimSpace(cleaned)
		if cleaned == "" || cleaned == original {
			continue
		}

		qtype := "lex"
		if len(expansions) > 0 {
			qtype = "vec"
		}
		expansions = append(expansions, QueryExpansion{Query: cleaned, Type: qtype})
		if len(expansions) >= 2 {
			break
		}
	}

	for len(expansions) < 2 {
		qtype := "lex"
		if len(expansions) == 1 {
			qtype = "vec"
		}
		expansions = append(expansions, QueryExpansion{Query: original, Type: qtype})
	}

	return expansions
}

func (e *LLMEngine) Rerank(query, intent string, candidates []string) ([]float64, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.ensureRerankModel(); err != nil {
		scores := make([]float64, len(candidates))
		for i := range scores {
			scores[i] = 0.5
		}
		return scores, nil
	}

	vocab := llama.ModelGetVocab(e.rerankModel)
	scores := make([]float64, len(candidates))

	for i, candidate := range candidates {
		var input string
		if intent != "" {
			input = fmt.Sprintf("Intent: %s\nQuery: %s\nDocument: %s\nRelevant:", intent, query, candidate)
		} else {
			input = fmt.Sprintf("Query: %s\nDocument: %s\nRelevant:", query, candidate)
		}

		tokens := llama.Tokenize(vocab, input, true, false)
		batch := llama.BatchGetOne(tokens)

		if _, err := llama.Decode(e.rerankCtx, batch); err != nil {
			scores[i] = 0.5
			continue
		}

		nVocab := llama.VocabNTokens(vocab)
		logits, err := llama.GetLogitsIth(e.rerankCtx, int32(len(tokens)-1), int(nVocab))
		if err != nil || logits == nil {
			scores[i] = 0.5
			continue
		}

		yesTokens := llama.Tokenize(vocab, "yes", false, false)
		noTokens := llama.Tokenize(vocab, "no", false, false)

		if len(yesTokens) > 0 && len(noTokens) > 0 {
			yesLogit := float64(logits[yesTokens[0]])
			noLogit := float64(logits[noTokens[0]])

			maxLogit := math.Max(yesLogit, noLogit)
			yesExp := math.Exp(yesLogit - maxLogit)
			noExp := math.Exp(noLogit - maxLogit)
			scores[i] = yesExp / (yesExp + noExp)
		} else {
			scores[i] = 0.5
		}
	}

	return scores, nil
}

// ---------------------------------------------------------------------------
// Hybrid Search Pipeline
// ---------------------------------------------------------------------------

type HybridSearcher struct {
	store    *Store
	engine   Embedder
	noExpand bool // force-disable ExpandQuery (CLI/MCP override for benchmarking)
	noRerank bool // force-disable Rerank (constrained hardware / benchmarking)
}

func newHybridSearcher(store *Store, engine Embedder) Searcher {
	return &HybridSearcher{store: store, engine: engine}
}

// SetNoExpand forces the LLM ExpandQuery stage off regardless of strong-signal
// detection. Mainly used for benchmarking and the --no-expand CLI flag.
func (h *HybridSearcher) SetNoExpand(v bool) { h.noExpand = v }

// SetNoRerank skips the LLM reranking stage and returns RRF-fused order
// directly. Useful on constrained hardware and for benchmarking.
func (h *HybridSearcher) SetNoRerank(v bool) { h.noRerank = v }

// hasStrongSignal returns true when the top BM25 result dominates the rest by
// at least strongSignalRatio× the score at strongSignalRankN. Scale-free, so
// it works against both raw FTS5 BM25 scores and RRF-normalized scores.
func hasStrongSignal(results []SearchResult) bool {
	if len(results) < strongSignalRankN {
		return false
	}
	top := results[0].Score
	cmp := results[strongSignalRankN-1].Score
	if top <= 0 || cmp <= 0 {
		return false
	}
	return top >= strongSignalRatio*cmp
}

func (h *HybridSearcher) Search(ctx context.Context, query, intent, collection string, limit int) ([]SearchResult, error) {
	// Probe BM25 first. Cheap (~ms) and lets us decide whether to spend
	// 150–400ms running the LLM ExpandQuery model.
	var probeBM25 []SearchResult
	var probeErr error
	if collection != "" {
		probeBM25, probeErr = h.store.SearchBM25InCollection(query, collection, strongSignalProbeN)
	} else {
		probeBM25, probeErr = h.store.SearchBM25Normalized(query, strongSignalProbeN)
	}
	if probeErr != nil {
		probeBM25 = nil
	}

	skipExpand := h.noExpand || hasStrongSignal(probeBM25)

	var expansions []QueryExpansion
	if !skipExpand {
		var err error
		expansions, err = h.engine.ExpandQuery(query, intent)
		if err != nil {
			expansions = nil
		}
	}

	type weightedQuery struct {
		query  string
		weight float64
		types  string
	}
	queries := []weightedQuery{
		{query: query, weight: 2.0, types: "both"},
	}
	for _, exp := range expansions {
		queries = append(queries, weightedQuery{query: exp.Query, weight: 1.0, types: exp.Type})
	}

	type rankedList struct {
		results []SearchResult
		weight  float64
	}
	var allLists []rankedList
	var listsMu sync.Mutex
	var wg sync.WaitGroup

	// Seed the lex side of the original query from the probe — saves one BM25
	// call on every hybrid search.
	probeUsed := false

	for _, wq := range queries {
		wq := wq
		if wq.types == "both" || wq.types == "lex" {
			if !probeUsed && wq.query == query {
				probeUsed = true
				if probeBM25 != nil {
					allLists = append(allLists, rankedList{results: probeBM25, weight: wq.weight})
				}
			} else {
				wg.Add(1)
				go func() {
					defer wg.Done()
					var results []SearchResult
					var err error
					if collection != "" {
						results, err = h.store.SearchBM25InCollection(wq.query, collection, strongSignalProbeN)
					} else {
						results, err = h.store.SearchBM25Normalized(wq.query, strongSignalProbeN)
					}
					if err != nil {
						return
					}
					listsMu.Lock()
					allLists = append(allLists, rankedList{results: results, weight: wq.weight})
					listsMu.Unlock()
				}()
			}
		}
		if wq.types == "both" || wq.types == "vec" {
			wg.Add(1)
			go func() {
				defer wg.Done()
				qvec, err := h.engine.Embed(wq.query, true)
				if err != nil {
					return
				}
				// Anchor snippets on the user's original query+intent, not on
				// the expansion variant — the expansion drives the embedding,
				// but highlights should reflect the user's actual words.
				snippetText := combineForSnippet(query, intent)
				var results []SearchResult
				if collection != "" {
					results, err = h.store.SearchVectorInCollection(snippetText, qvec, collection, 20)
				} else {
					results, err = h.store.SearchVector(snippetText, qvec, 20)
				}
				if err != nil {
					return
				}
				listsMu.Lock()
				allLists = append(allLists, rankedList{results: results, weight: wq.weight})
				listsMu.Unlock()
			}()
		}
	}
	wg.Wait()

	// RRF fusion
	rrfScores := make(map[string]float64)
	docByID := make(map[string]SearchResult)

	for _, list := range allLists {
		for rank, r := range list.results {
			score := list.weight / float64(rrfK+rank+1)
			switch rank {
			case 0:
				score += 0.05
			case 1, 2:
				score += 0.02
			}
			rrfScores[r.DocID] += score
			if _, exists := docByID[r.DocID]; !exists {
				docByID[r.DocID] = r
			}
		}
	}

	type rrfEntry struct {
		docID string
		score float64
	}
	var rrfRanked []rrfEntry
	for id, sc := range rrfScores {
		rrfRanked = append(rrfRanked, rrfEntry{id, sc})
	}
	sort.Slice(rrfRanked, func(i, j int) bool { return rrfRanked[i].score > rrfRanked[j].score })

	if len(rrfRanked) > maxRerank {
		rrfRanked = rrfRanked[:maxRerank]
	}

	// LLM re-ranking
	var candidateTexts []string
	for _, entry := range rrfRanked {
		doc := docByID[entry.docID]
		candidateTexts = append(candidateTexts, doc.Title+" "+doc.Snippet)
	}

	var rerankScores []float64
	var err error
	if h.noRerank {
		err = fmt.Errorf("rerank disabled")
	} else {
		rerankScores, err = h.engine.Rerank(query, intent, candidateTexts)
	}
	if err != nil {
		var results []SearchResult
		for _, entry := range rrfRanked {
			r := docByID[entry.docID]
			r.Score = entry.score
			results = append(results, r)
		}
		if len(results) > limit {
			results = results[:limit]
		}
		return results, nil
	}

	// Position-aware blend
	var results []SearchResult
	for i, entry := range rrfRanked {
		rrfNorm := entry.score
		rerank := rerankScores[i]

		var rrfWeight, rerankWeight float64
		switch {
		case i < 3:
			rrfWeight, rerankWeight = 0.75, 0.25
		case i < 10:
			rrfWeight, rerankWeight = 0.60, 0.40
		default:
			rrfWeight, rerankWeight = 0.40, 0.60
		}

		r := docByID[entry.docID]
		r.Score = rrfNorm*rrfWeight + rerank*rerankWeight
		results = append(results, r)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

// ---------------------------------------------------------------------------
// Auto-download: llama.cpp shared library
// ---------------------------------------------------------------------------

func (e *LLMEngine) ensureLib() error {
	libDir := filepath.Join(filepath.Dir(e.modelsDir), "lib")
	libName := "libllama.so"
	if runtime.GOOS == "darwin" {
		libName = "libllama.dylib"
	}
	libPath := filepath.Join(libDir, libName)
	if _, err := os.Stat(libPath); err == nil {
		e.libPath = libPath
		return nil
	}

	os.MkdirAll(libDir, 0o755)

	url, err := llamaReleaseURL()
	if err != nil {
		return err
	}

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("download llama.cpp: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download llama.cpp: HTTP %d", resp.StatusCode)
	}

	ext := ".tar.gz"
	if strings.HasSuffix(url, ".zip") {
		ext = ".zip"
	}
	tmpFile, err := os.CreateTemp("", "llama-*"+ext)
	if err != nil {
		return err
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	pw := &progressWriter{name: "llama.cpp runtime", total: resp.ContentLength}
	if _, err := io.Copy(tmpFile, io.TeeReader(resp.Body, pw)); err != nil {
		tmpFile.Close()
		return err
	}
	tmpFile.Close()
	pw.finish()

	if strings.HasSuffix(url, ".tar.gz") {
		if err := extractLibsFromTarGz(tmpPath, libDir); err != nil {
			return err
		}
	} else {
		if err := extractLibsFromZip(tmpPath, libDir); err != nil {
			return err
		}
	}

	if _, err := os.Stat(libPath); err != nil {
		return fmt.Errorf("llama.cpp library not found in downloaded archive — try setting PICOQMD_LIB manually")
	}
	e.libPath = libPath
	return nil
}

func llamaReleaseURL() (string, error) {
	base := "https://github.com/ggml-org/llama.cpp/releases/download/" +
		llamaCppRelease + "/llama-" + llamaCppRelease + "-bin-"
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm64":
			return base + "macos-arm64.tar.gz", nil
		case "amd64":
			return base + "macos-x64.tar.gz", nil
		}
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			if _, err := os.Stat("/usr/local/cuda"); err == nil {
				return base + "ubuntu-x64-cuda.tar.gz", nil
			}
			return base + "ubuntu-x64.tar.gz", nil
		case "arm64":
			return base + "ubuntu-arm64.tar.gz", nil
		}
	}
	return "", fmt.Errorf("unsupported platform: %s/%s — install llama.cpp manually and set PICOQMD_LIB", runtime.GOOS, runtime.GOARCH)
}

func extractLibsFromZip(zipPath, destDir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	libExt := ".so"
	if runtime.GOOS == "darwin" {
		libExt = ".dylib"
	}

	extracted := 0
	for _, f := range r.File {
		name := filepath.Base(f.Name)
		if !strings.HasSuffix(name, libExt) {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("open %s in zip: %w", f.Name, err)
		}
		destPath := filepath.Join(destDir, name)
		out, err := os.Create(destPath)
		if err != nil {
			rc.Close()
			return fmt.Errorf("create %s: %w", destPath, err)
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			rc.Close()
			os.Remove(destPath)
			return fmt.Errorf("extract %s: %w", name, err)
		}
		out.Close()
		rc.Close()
		os.Chmod(destPath, 0o755)
		extracted++
	}
	if extracted == 0 {
		return fmt.Errorf("no %s files found in archive", libExt)
	}
	return nil
}

func extractLibsFromTarGz(tgzPath, destDir string) error {
	f, err := os.Open(tgzPath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gz.Close()

	tr := tar.NewReader(gz)

	libExt := ".so"
	if runtime.GOOS == "darwin" {
		libExt = ".dylib"
	}

	extracted := 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}
		name := filepath.Base(hdr.Name)
		if !strings.Contains(name, libExt) {
			continue
		}
		destPath := filepath.Join(destDir, name)

		// llama.cpp releases ship version chains as symlinks
		// (libggml.dylib -> libggml.0.9.7.dylib). Symlink entries have no
		// body — writing them as regular files produces 0-byte dylibs that
		// dlopen rejects. Recreate them as symlinks; everything lands flat
		// in destDir so the target basename is the correct link.
		if hdr.Typeflag == tar.TypeSymlink || hdr.Typeflag == tar.TypeLink {
			os.Remove(destPath)
			if err := os.Symlink(filepath.Base(hdr.Linkname), destPath); err != nil {
				return fmt.Errorf("symlink %s: %w", name, err)
			}
			extracted++
			continue
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		out, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("create %s: %w", destPath, err)
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			os.Remove(destPath)
			return fmt.Errorf("extract %s: %w", name, err)
		}
		out.Close()
		os.Chmod(destPath, 0o755)
		extracted++
	}
	if extracted == 0 {
		return fmt.Errorf("no %s files found in archive", libExt)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Embedding orchestration
// ---------------------------------------------------------------------------

// embedAll embeds every pending document, or only those in `collection`
// when non-empty.
func embedAll(store *Store, collection string) error {
	fp := embedFingerprint()
	total, err := store.CountUnembedded(fp, collection)
	if err != nil {
		return err
	}
	if total == 0 {
		infof("  All documents already embedded\n")
		return nil
	}

	infof("  Embedding %d documents...\n", total)
	start := time.Now()
	embedded := 0

	selfBin, err := os.Executable()
	if err != nil {
		return fmt.Errorf("cannot find own binary: %w", err)
	}

	// When quiet, discard the worker subprocess's stdout/stderr instead
	// of inheriting the parent's. Otherwise the worker's per-chunk
	// progress lines would still flow through to a launchd-captured log
	// even though embedAll itself is silent.
	var workerOut, workerErr io.Writer = os.Stdout, os.Stderr
	if quiet {
		workerOut, workerErr = io.Discard, io.Discard
	}

	const batchSize = 500
	for {
		remaining, err := store.CountUnembedded(fp, collection)
		if err != nil {
			return err
		}
		if remaining == 0 {
			break
		}

		args := []string{"embed-worker", "--batch", strconv.Itoa(batchSize)}
		if collection != "" {
			args = append(args, "--collection", collection)
		}
		if quiet {
			args = append(args, "--quiet")
		}
		cmd := exec.Command(selfBin, args...)
		cmd.Stdout = workerOut
		cmd.Stderr = workerErr
		cmd.Env = os.Environ()

		err = cmd.Run()
		if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == embedEngineUnavailableExit {
			return fmt.Errorf("embedding engine unavailable — aborting instead of skipping documents (check 'picoqmd model list' and the llama.cpp lib in the cache dir)")
		}
		newRemaining, countErr := store.CountUnembedded(fp, collection)
		if countErr != nil {
			return countErr
		}
		batchDone := remaining - newRemaining
		embedded += batchDone

		if err != nil {
			// Worker errors go to the log facility (rate-limited by
			// the runtime, not per-line) rather than direct stdout, so
			// they're useful for debugging without becoming the source
			// of unbounded log growth on chronic failures.
			log.Printf("  worker exited (%v), embedded %d docs in this batch", err, batchDone)
		}
		if batchDone == 0 {
			infof("  No progress — skipping problematic document\n")
			if skipErr := store.SkipNextUnembedded(); skipErr != nil {
				return fmt.Errorf("failed to skip document: %w", skipErr)
			}
			continue
		}
		infof("  %d/%d documents embedded\n", total-newRemaining, total)
	}

	// Final summary always prints — one line per invocation, bounded.
	fmt.Printf("  Embedding done: %d documents (%s)\n", embedded, time.Since(start).Round(time.Second))
	return nil
}

func embedWorker(maxDocs int, collection string) error {
	cfg, _, err := loadConfig("")
	if err != nil {
		return err
	}
	store, err := NewStore(dbPath(""))
	if err != nil {
		return err
	}
	defer store.Close()

	engine := NewLLMEngine(cacheDir())
	defer engine.Close()

	// Probe the engine before touching any documents: a chronic failure
	// (missing or corrupt llama.cpp lib, missing model) must abort the run
	// instead of letting the orchestrator mark every document as skipped.
	if _, err := engine.Embed("startup probe", true); err != nil {
		fmt.Fprintf(os.Stderr, "embedding engine unavailable: %v\n", err)
		os.Exit(embedEngineUnavailableExit)
	}

	fp := embedFingerprint()
	hashes, err := store.UnembeddedHashes(fp, collection)
	if err != nil {
		return err
	}
	if maxDocs > 0 && len(hashes) > maxDocs {
		hashes = hashes[:maxDocs]
	}

	_ = cfg

	var skipCount, embedErrors int
	for i, hash := range hashes {
		title, absPath, err := store.DocForHash(hash)
		if err != nil {
			log.Printf("  skip %s: %v", hash[:6], err)
			skipCount++
			continue
		}

		hasChunks, err := store.HasChunks(hash)
		if err != nil {
			return err
		}
		// A stale fingerprint means the model or chunker changed — existing
		// chunks may be cut differently now, so re-chunk from scratch
		// (StoreChunks deletes the old rows) and re-embed everything.
		if hasChunks {
			stale, err := store.HasStaleFP(hash, fp)
			if err != nil {
				return err
			}
			if stale {
				hasChunks = false
			}
		}
		if !hasChunks {
			data, err := os.ReadFile(absPath)
			if err != nil {
				log.Printf("  skip %s: cannot read %s: %v", hash[:6], absPath, err)
				skipCount++
				continue
			}
			chunks := ChunkDocument(string(data))
			if err := store.StoreChunks(hash, chunks); err != nil {
				return err
			}
		}

		toEmbed, err := store.UnembeddedChunks(hash)
		if err != nil {
			return err
		}

		for _, chunk := range toEmbed {
			input := fmt.Sprintf("title: %s | text: %s", title, chunk.Text)
			vec, err := engine.Embed(input, false)
			if err != nil {
				log.Printf("  embed error for %s seq %d: %v", hash[:6], chunk.Seq, err)
				embedErrors++
				continue
			}
			if err := store.StoreVector(hash, chunk.Seq, vec, fp); err != nil {
				return err
			}
		}

		if (i+1)%10 == 0 {
			infof("  [worker] %d/%d docs\n", i+1, len(hashes))
		}
	}

	// One-shot summary — always prints. Bounded at one line per worker invocation.
	embedded := len(hashes) - skipCount
	if quiet {
		// Discard mode: even the summary goes nowhere, since the parent
		// already logs the per-batch totals on its own bounded line.
		return nil
	}
	fmt.Printf("  [worker] done: %d docs", embedded)
	if embedErrors > 0 {
		fmt.Printf(" (%d chunk errors)", embedErrors)
	}
	fmt.Println()
	return nil
}

// syncAll re-indexes all collections then embeds any unembedded documents.
func syncAll(store *Store, engine Embedder, cfg *Config, skipEmbed bool) error {
	for _, col := range cfg.Collections {
		if err := indexCollection(store, col); err != nil {
			log.Printf("error indexing %s: %v", col.Name, err)
		}
	}
	if skipEmbed {
		return nil
	}
	return embedAll(store, "")
}
