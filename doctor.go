// doctor.go — index/runtime diagnostics and vector hygiene.
//
// `picoqmd doctor` reports what `status` can't see: per-fingerprint vector
// distribution (stale-model detection), dummy/skipped vectors, and orphaned
// vector rows. Motivated by two real incidents the old tooling missed:
// v0.4.1's dummy-vector poisoning and the 2026-08-29 silent model-file swap
// (same filename, different bytes → every search returned noise).
//
// `picoqmd cleanup` deletes stale-fingerprint and orphaned vectors so the
// next embed pass regenerates them; `--dry-run` only reports. (qmd 2.8.3)

package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"zombiezen.com/go/sqlite"
	"zombiezen.com/go/sqlite/sqlitex"
)

type fpBucket struct {
	fp      string
	total   int // rows with this fp
	real    int // vec present and full-size
	dummy   int // 4-byte placeholder written for skipped docs
	pending int // vec IS NULL
}

type vectorDiagnostics struct {
	buckets []fpBucket
	orphans int // rows whose hash no active document references
}

const orphanVectorCond = `hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)`

func (s *Store) VectorDiagnostics() (*vectorDiagnostics, error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return nil, err
	}
	defer s.pool.Put(conn)

	d := &vectorDiagnostics{}
	err = sqlitex.Execute(conn, `
		SELECT fp, COUNT(*),
		       SUM(CASE WHEN vec IS NOT NULL AND LENGTH(vec) > 4 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN LENGTH(vec) = 4 THEN 1 ELSE 0 END),
		       SUM(CASE WHEN vec IS NULL THEN 1 ELSE 0 END)
		FROM content_vectors GROUP BY fp ORDER BY COUNT(*) DESC`,
		&sqlitex.ExecOptions{
			ResultFunc: func(stmt *sqlite.Stmt) error {
				d.buckets = append(d.buckets, fpBucket{
					fp:      stmt.ColumnText(0),
					total:   stmt.ColumnInt(1),
					real:    stmt.ColumnInt(2),
					dummy:   stmt.ColumnInt(3),
					pending: stmt.ColumnInt(4),
				})
				return nil
			},
		})
	if err != nil {
		return nil, err
	}

	err = sqlitex.Execute(conn, `SELECT COUNT(*) FROM content_vectors WHERE `+orphanVectorCond,
		&sqlitex.ExecOptions{
			ResultFunc: func(stmt *sqlite.Stmt) error {
				d.orphans = stmt.ColumnInt(0)
				return nil
			},
		})
	return d, err
}

// CleanupVectors deletes stale-fingerprint vectors (embedded under a
// different model/chunker than the current one) and orphaned vector rows.
// Returns (stale, orphans) counts — counted only when dryRun, deleted counts
// otherwise.
func (s *Store) CleanupVectors(currentFP string, dryRun bool) (stale, orphans int, err error) {
	conn, err := s.pool.Take(context.Background())
	if err != nil {
		return 0, 0, err
	}
	defer s.pool.Put(conn)

	count := func(sql string, args []any) (int, error) {
		n := 0
		err := sqlitex.Execute(conn, sql, &sqlitex.ExecOptions{
			Args: args,
			ResultFunc: func(stmt *sqlite.Stmt) error {
				n = stmt.ColumnInt(0)
				return nil
			},
		})
		return n, err
	}

	staleCond := `vec IS NOT NULL AND fp <> ?`
	if stale, err = count(`SELECT COUNT(*) FROM content_vectors WHERE `+staleCond, []any{currentFP}); err != nil {
		return
	}
	if orphans, err = count(`SELECT COUNT(*) FROM content_vectors WHERE `+orphanVectorCond, nil); err != nil {
		return
	}
	if dryRun {
		return
	}
	if err = sqlitex.Execute(conn, `DELETE FROM content_vectors WHERE `+staleCond,
		&sqlitex.ExecOptions{Args: []any{currentFP}}); err != nil {
		return
	}
	err = sqlitex.Execute(conn, `DELETE FROM content_vectors WHERE `+orphanVectorCond, nil)
	return
}

func newDoctorCmd(dbPathFn func() string) *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Diagnose index health: model identity, vector fingerprints, stale/orphaned vectors",
		RunE: func(cmd *cobra.Command, args []string) error {
			problems := 0

			fmt.Printf("picoqmd %s\n\n", version)

			// --- database ---
			dbp := dbPathFn()
			fmt.Printf("database:  %s", dbp)
			if st, err := os.Stat(dbp); err == nil {
				fmt.Printf(" (%.1f MB)", float64(st.Size())/1024/1024)
			}
			fmt.Println()

			// --- models ---
			modelsDir := filepath.Join(cacheDir(), "models")
			fmt.Println("\nmodels:")
			for _, spec := range defaultModels {
				p := filepath.Join(modelsDir, spec.Filename)
				st, err := os.Stat(p)
				if err != nil {
					fmt.Printf("  %-10s MISSING  %s\n", spec.Name+":", spec.Filename)
					continue
				}
				fmt.Printf("  %-10s %s  %.0f MB  sha256:%s\n",
					spec.Name+":", spec.Filename, float64(st.Size())/1024/1024, modelFileHash(p))
			}

			currentFP := embedFingerprint()
			fmt.Printf("\nembedding fingerprint: %s\n", currentFP)

			store, err := NewStore(dbp)
			if err != nil {
				return err
			}
			defer store.Close()

			cols, docs, chunks, _ := store.Stats()
			fmt.Printf("\nindex: %d collections, %d documents, %d chunks\n", cols, docs, chunks)

			// --- vector fingerprint distribution ---
			d, err := store.VectorDiagnostics()
			if err != nil {
				return err
			}
			fmt.Println("\nvectors by fingerprint:")
			for _, b := range d.buckets {
				label := ""
				if b.fp != currentFP && b.real > 0 {
					label = "  <-- STALE (will not match query embeddings)"
					problems++
				}
				fp := b.fp
				if fp == "" {
					fp = "(none)"
				}
				fmt.Printf("  %-52s %6d embedded, %d dummy, %d pending%s\n",
					fp, b.real, b.dummy, b.pending, label)
			}
			if d.orphans > 0 {
				fmt.Printf("\norphaned vector rows (no active document): %d  <-- run `picoqmd cleanup`\n", d.orphans)
				problems++
			}

			if problems == 0 {
				fmt.Println("\nOK: no problems found")
				return nil
			}
			fmt.Printf("\n%d problem(s) found. `picoqmd cleanup` removes stale/orphaned vectors,\nthen `picoqmd sync` re-embeds what's missing.\n", problems)
			// Non-zero exit so scripts (launchd refresh, CI) can gate on health.
			os.Exit(1)
			return nil
		},
	}
}

// newMigrateVectorsCmd truncates stored higher-dimension vectors to the
// current Matryoshka target in place. For MRL-trained models (EmbeddingGemma)
// truncate+renormalize is exactly what embedding at the lower dimension
// produces, so this is a seconds-long migration instead of a full re-embed.
func newMigrateVectorsCmd(dbPathFn func() string) *cobra.Command {
	return &cobra.Command{
		Use:   "migrate-vectors",
		Short: "Truncate stored MRL vectors to the current target dimension (instant alternative to re-embedding)",
		RunE: func(cmd *cobra.Command, args []string) error {
			dim := embedTargetDim()
			if dim <= 0 {
				return fmt.Errorf("PICOQMD_EMBED_DIM is 0 (full dimension); nothing to truncate to")
			}
			fp := embedFingerprint()
			dbp := dbPathFn()
			sizeBefore := int64(0)
			if st, err := os.Stat(dbp); err == nil {
				sizeBefore = st.Size()
			}

			store, err := NewStore(dbp)
			if err != nil {
				return err
			}
			defer store.Close()

			conn, err := store.pool.Take(context.Background())
			if err != nil {
				return err
			}
			defer store.pool.Put(conn)

			type row struct {
				hash string
				seq  int
				vec  []byte
			}
			var rows []row
			err = sqlitex.Execute(conn,
				`SELECT hash, seq, vec FROM content_vectors WHERE LENGTH(vec) > ?`,
				&sqlitex.ExecOptions{
					Args: []any{dim * 4},
					ResultFunc: func(stmt *sqlite.Stmt) error {
						b := make([]byte, stmt.ColumnLen(2))
						stmt.ColumnBytes(2, b)
						rows = append(rows, row{hash: stmt.ColumnText(0), seq: stmt.ColumnInt(1), vec: b})
						return nil
					},
				})
			if err != nil {
				return err
			}
			if len(rows) == 0 {
				fmt.Printf("no vectors above %d dims; nothing to migrate\n", dim)
				return nil
			}

			if err := sqlitex.Execute(conn, "BEGIN", nil); err != nil {
				return err
			}
			for _, r := range rows {
				src := make([]float32, len(r.vec)/4)
				for i := range src {
					bits := uint32(r.vec[i*4]) | uint32(r.vec[i*4+1])<<8 |
						uint32(r.vec[i*4+2])<<16 | uint32(r.vec[i*4+3])<<24
					src[i] = math.Float32frombits(bits)
				}
				out := truncateMRL(src, dim)
				if err := sqlitex.Execute(conn,
					`UPDATE content_vectors SET vec = ?, fp = ? WHERE hash = ? AND seq = ?`,
					&sqlitex.ExecOptions{Args: []any{float32ToBytes(out), fp, r.hash, r.seq}}); err != nil {
					sqlitex.Execute(conn, "ROLLBACK", nil)
					return err
				}
			}
			// Re-stamp dummy placeholders too, so they don't read as stale.
			if err := sqlitex.Execute(conn,
				`UPDATE content_vectors SET fp = ? WHERE LENGTH(vec) = 4 AND fp <> ?`,
				&sqlitex.ExecOptions{Args: []any{fp, fp}}); err != nil {
				sqlitex.Execute(conn, "ROLLBACK", nil)
				return err
			}
			if err := sqlitex.Execute(conn, "COMMIT", nil); err != nil {
				return err
			}
			fmt.Printf("truncated %d vectors to %d dims (fingerprint %s)\n", len(rows), dim, fp)

			fmt.Println("reclaiming space (VACUUM)...")
			if err := sqlitex.Execute(conn, "VACUUM", nil); err != nil {
				return fmt.Errorf("vacuum: %w", err)
			}
			// Flush the WAL so the main-file size below reflects the vacuum.
			sqlitex.Execute(conn, "PRAGMA wal_checkpoint(TRUNCATE)", nil)
			if st, err := os.Stat(dbp); err == nil && sizeBefore > 0 {
				fmt.Printf("index: %.0f MB -> %.0f MB\n",
					float64(sizeBefore)/1048576, float64(st.Size())/1048576)
			}
			return nil
		},
	}
}

func newCleanupCmd(dbPathFn func() string) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cleanup",
		Short: "Delete stale-fingerprint and orphaned vectors (re-embed with `picoqmd sync` after)",
		RunE: func(cmd *cobra.Command, args []string) error {
			dryRun, _ := cmd.Flags().GetBool("dry-run")
			store, err := NewStore(dbPathFn())
			if err != nil {
				return err
			}
			defer store.Close()

			stale, orphans, err := store.CleanupVectors(embedFingerprint(), dryRun)
			if err != nil {
				return err
			}
			verb := "deleted"
			if dryRun {
				verb = "would delete"
			}
			fmt.Printf("%s %d stale vector(s), %d orphaned row(s)\n", verb, stale, orphans)
			if !dryRun && stale+orphans > 0 {
				fmt.Println("run `picoqmd sync` to re-embed pending chunks")
			}
			return nil
		},
	}
	cmd.Flags().Bool("dry-run", false, "report what would be deleted without deleting")
	return cmd
}
