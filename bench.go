// bench.go — search-quality evaluation against a JSON fixture. (qmd 2.1)
//
// A fixture pins queries to the documents that should surface, so retrieval
// changes (model swaps, Matryoshka truncation, binary quantization, chunker
// edits) can be measured instead of eyeballed:
//
//	{
//	  "k": 10,
//	  "cases": [
//	    {"query": "envelope shadowing mask chain", "collection": "memory",
//	     "expect": ["masking-experiment", "#a1b2c3"]}
//	  ]
//	}
//
// An `expect` entry starting with '#' matches a docid exactly; anything else
// matches as a substring of the result path or title. Metrics per pipeline:
// hit@k (any relevant found), precision@k, recall@k, MRR.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

type benchCase struct {
	Query      string   `json:"query"`
	Intent     string   `json:"intent,omitempty"`
	Collection string   `json:"collection,omitempty"`
	Expect     []string `json:"expect"`
}

type benchFixture struct {
	K     int         `json:"k,omitempty"`
	Cases []benchCase `json:"cases"`
}

// benchMatch reports whether a result satisfies one expectation.
func benchMatch(r SearchResult, expect string) bool {
	if strings.HasPrefix(expect, "#") {
		return r.DocID == strings.TrimPrefix(expect, "#")
	}
	return strings.Contains(r.Path, expect) || strings.Contains(r.Title, expect)
}

type benchScore struct {
	hits      int     // cases with >=1 relevant result in top k
	precision float64 // sum over cases of relevant-in-topk / k
	recall    float64 // sum over cases of expectations-found / len(expect)
	mrr       float64 // sum over cases of 1/rank-of-first-relevant
	cases     int
}

func (b *benchScore) add(results []SearchResult, expect []string, k int) {
	b.cases++
	if len(results) > k {
		results = results[:k]
	}
	found := make(map[int]bool) // expectation index -> satisfied
	relevant := 0
	firstRank := 0
	for rank, r := range results {
		hit := false
		for ei, e := range expect {
			if benchMatch(r, e) {
				found[ei] = true
				hit = true
			}
		}
		if hit {
			relevant++
			if firstRank == 0 {
				firstRank = rank + 1
			}
		}
	}
	if firstRank > 0 {
		b.hits++
		b.mrr += 1.0 / float64(firstRank)
	}
	b.precision += float64(relevant) / float64(k)
	if len(expect) > 0 {
		b.recall += float64(len(found)) / float64(len(expect))
	}
}

func (b *benchScore) row(name string) string {
	n := float64(b.cases)
	if n == 0 {
		return fmt.Sprintf("  %-10s (no cases)", name)
	}
	return fmt.Sprintf("  %-10s hit@k %5.1f%%   precision %5.3f   recall %5.3f   MRR %5.3f",
		name, 100*float64(b.hits)/n, b.precision/n, b.recall/n, b.mrr/n)
}

func newBenchCmd(dbPathFn func() string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bench <fixture.json>",
		Short: "Evaluate search quality (hit@k, precision, recall, MRR) against a fixture",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			raw, err := os.ReadFile(args[0])
			if err != nil {
				return err
			}
			var fx benchFixture
			if err := json.Unmarshal(raw, &fx); err != nil {
				return fmt.Errorf("parse fixture: %w", err)
			}
			if fx.K == 0 {
				fx.K = 10
			}
			if len(fx.Cases) == 0 {
				return fmt.Errorf("fixture has no cases")
			}

			only, _ := cmd.Flags().GetString("pipeline")
			verbose, _ := cmd.Flags().GetBool("misses")

			store, err := NewStore(dbPathFn())
			if err != nil {
				return err
			}
			defer store.Close()
			engine := NewLLMEngine(cacheDir())

			hasEmbed := modelExists(engine.ModelsDir(), defaultModels[0].Filename)
			pipelines := []string{"bm25"}
			if hasEmbed {
				pipelines = append(pipelines, "vector", "research")
				if modelExists(engine.ModelsDir(), defaultModels[1].Filename) &&
					modelExists(engine.ModelsDir(), defaultModels[2].Filename) {
					pipelines = append(pipelines, "hybrid")
				}
			}
			if only != "" {
				pipelines = []string{only}
			}

			run := func(pipeline string, c benchCase) ([]SearchResult, error) {
				switch pipeline {
				case "bm25":
					return store.SearchBM25Scoped(c.Query, c.Collection, fx.K)
				case "vector":
					qvec, err := engine.Embed(c.Query, true)
					if err != nil {
						return nil, err
					}
					return store.SearchVectorScoped(combineForSnippet(c.Query, c.Intent), qvec, c.Collection, fx.K)
				case "research":
					bm25Results, _ := store.SearchBM25Scoped(c.Query, c.Collection, fx.K*2)
					var vecResults []SearchResult
					if qvec, err := engine.Embed(c.Query, true); err == nil {
						vecResults, _ = store.SearchVectorScoped(combineForSnippet(c.Query, c.Intent), qvec, c.Collection, fx.K*2)
					}
					return simpleRRF(bm25Results, vecResults, fx.K), nil
				case "hybrid":
					hybrid := newHybridSearcher(store, engine)
					return hybrid.Search(context.Background(), c.Query, c.Intent, c.Collection, fx.K)
				default:
					return nil, fmt.Errorf("unknown pipeline %q (bm25|vector|research|hybrid)", pipeline)
				}
			}

			fmt.Printf("bench: %d cases, k=%d, fingerprint %s\n\n", len(fx.Cases), fx.K, embedFingerprint())
			for _, p := range pipelines {
				score := &benchScore{}
				var misses []string
				for _, c := range fx.Cases {
					results, err := run(p, c)
					if err != nil {
						return fmt.Errorf("%s %q: %w", p, c.Query, err)
					}
					before := score.hits
					score.add(results, c.Expect, fx.K)
					if verbose && score.hits == before {
						misses = append(misses, c.Query)
					}
				}
				fmt.Println(score.row(p))
				for _, m := range misses {
					fmt.Printf("             miss: %s\n", m)
				}
			}
			return nil
		},
	}
	cmd.Flags().String("pipeline", "", "run only one pipeline: bm25, vector, research, hybrid (default: all available)")
	cmd.Flags().Bool("misses", false, "list queries with zero relevant results per pipeline")
	return cmd
}
