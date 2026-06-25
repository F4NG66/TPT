"""
Overlap analysis: 6 downstream datasets vs. GMSC pretraining corpus.

Two levels of check:
  1. Exact match  -- pure Python, no external tools, runs instantly.
     Finds sequences that appear character-for-character in the
     pretraining FASTA.

  2. Near-identity (≥90% sequence identity) -- uses CD-HIT or MMseqs2
     if available on the server. This is the standard redundancy-
     reduction threshold used in the field (and in the original GMSC
     paper). If neither tool is found the script skips this step and
     prints instructions.

Datasets:
  Original 4 (in ./benchmark/ or root -- adjust paths below):
    AMP   : AMP_amplify.csv
    TOX   : tox.csv
    CRISP : sequences_length_le100_CRISPR.csv
    MOES  : short_sequences_MOES.csv

  New 2 (in root):
    QSP   : QSP.csv
    CPP   : CPP.csv

GMSC pretraining corpus: ./benchmark/sample_10M.fasta
  (the 10M-sequence sample used for TPT pretraining)

Output:
  ./log/overlap_report.txt   -- human-readable summary
  ./log/overlap_exact.csv    -- per-dataset exact-match details
  ./log/downstream_all.fasta -- merged fasta of all 6 datasets
                                (input for CD-HIT / MMseqs2 if you
                                 want to run it manually)
"""

import os, time, sys, subprocess, shutil
import pandas as pd
import numpy as np
from Bio import SeqIO

os.makedirs("./log", exist_ok=True)

# ─── Dataset paths ─────────────────────────────────────────────────────────────
DATASETS = {
    "AMP":   "AMP_amplify.csv",
    "TOX":   "tox.csv",
    "CRISP": "./sequences_length_le100_CRISPR.csv",
    "MOES":  "./short_sequences_MOES.csv",
    "QSP":   "QSP.csv",
    "CPP":   "CPP.csv",
}

GMSC_FASTA = "/home/data/temp/GMSC10.90AA.faa"   # pretraining corpus

# ─── Load all downstream sequences ────────────────────────────────────────────
print("Loading downstream datasets...")
dataset_seqs = {}   # name -> list of sequences (upper-cased)
for name, path in DATASETS.items():
    df = pd.read_csv(path)
    seqs = df["Sequence"].str.upper().str.strip().tolist()
    dataset_seqs[name] = seqs
    print(f"  {name}: {len(seqs)} sequences "
          f"(len range {min(len(s) for s in seqs)}–{max(len(s) for s in seqs)})")

all_downstream = []
for seqs in dataset_seqs.values():
    all_downstream.extend(seqs)

downstream_set = set(all_downstream)
print(f"\nTotal downstream sequences : {len(all_downstream)}")
print(f"Unique downstream sequences: {len(downstream_set)}")

# ─── Write merged downstream FASTA (for CD-HIT / MMseqs2 later) ──────────────
merged_fasta = "./log/downstream_all.fasta"
with open(merged_fasta, "w") as f:
    idx = 0
    for ds_name, seqs in dataset_seqs.items():
        for seq in seqs:
            f.write(f">{ds_name}_{idx}\n{seq}\n")
            idx += 1
print(f"Merged downstream FASTA written to: {merged_fasta}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Exact match
# Stream through GMSC fasta, check every sequence against downstream set.
# Memory-efficient: only the downstream set (small) is held in RAM.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("STEP 1: Exact match vs GMSC pretraining corpus")
print(f"  Source: {GMSC_FASTA}")
print("  Streaming sequences (this may take a few minutes for 10M seqs)...")

# per-dataset hit sets
exact_hits = {name: set() for name in DATASETS}
gmsc_total  = 0
t0 = time.time()

with open(GMSC_FASTA) as fh:
    cur_seq = []
    for line in fh:
        line = line.rstrip()
        if line.startswith(">"):
            if cur_seq:
                seq = "".join(cur_seq).upper()
                gmsc_total += 1
                if seq in downstream_set:
                    # find which dataset(s) this belongs to
                    for name, seqs in dataset_seqs.items():
                        if seq in set(seqs):
                            exact_hits[name].add(seq)
                cur_seq = []
        else:
            cur_seq.append(line)
    # last record
    if cur_seq:
        seq = "".join(cur_seq).upper()
        gmsc_total += 1
        if seq in downstream_set:
            for name, seqs in dataset_seqs.items():
                if seq in set(seqs):
                    exact_hits[name].add(seq)

elapsed = time.time() - t0
print(f"  Scanned {gmsc_total:,} GMSC sequences in {elapsed:.1f}s")

# ─── Exact-match report ───────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("Exact match results:")
print(f"{'Dataset':<8} {'Total':>8} {'Exact hits':>12} {'Overlap %':>12}")
print("─" * 46)

exact_rows = []
for name in DATASETS:
    n_total = len(dataset_seqs[name])
    n_hit   = len(exact_hits[name])
    pct     = 100 * n_hit / n_total if n_total else 0
    print(f"{name:<8} {n_total:>8} {n_hit:>12} {pct:>11.2f}%")
    exact_rows.append({
        "dataset":    name,
        "n_seqs":     n_total,
        "exact_hits": n_hit,
        "overlap_pct": round(pct, 4),
    })

df_exact = pd.DataFrame(exact_rows)
df_exact.to_csv("./log/overlap_exact.csv", index=False)
print(f"\nSaved to ./log/overlap_exact.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Near-identity check via CD-HIT or MMseqs2
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print("STEP 2: Near-identity check (≥90% sequence identity)")

cdhit_bin   = shutil.which("cd-hit")
mmseqs_bin  = shutil.which("mmseqs")

if cdhit_bin:
    print(f"  Found CD-HIT: {cdhit_bin}")
    print("  Running CD-HIT on merged downstream FASTA + GMSC combined input...")

    # CD-HIT approach: combine downstream + GMSC, cluster at 90%, count
    # how many downstream seqs land in a cluster containing a GMSC seq.
    # For 10M sequences this is heavy; we instead run cd-hit-est-2d
    # (two-dataset mode): database = GMSC, query = downstream.
    cdhit2d = shutil.which("cd-hit-2d") or shutil.which("cd-hit-est-2d")
    if cdhit2d:
        out_prefix = "./log/cdhit2d_out"
        cmd = [
            cdhit2d,
            "-i",  GMSC_FASTA,        # database (larger set)
            "-i2", merged_fasta,       # query (smaller set = downstream)
            "-o",  out_prefix,
            "-c",  "0.90",             # 90% identity
            "-n",  "5",                # word size for 90%
            "-T",  "8",                # threads
            "-M",  "16000",            # memory MB
            "-d",  "0",
        ]
        print(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # parse .clstr to count downstream seqs that share a cluster
            # with at least one GMSC sequence
            clstr_path = out_prefix + ".clstr"
            hits_in_shared = 0
            with open(clstr_path) as f:
                cluster_has_db  = False
                cluster_q_seqs  = []
                for line in f:
                    if line.startswith(">Cluster"):
                        # process previous cluster
                        if cluster_has_db:
                            hits_in_shared += len(cluster_q_seqs)
                        cluster_has_db = False
                        cluster_q_seqs = []
                    else:
                        if "*" in line or "at" in line:
                            if "db" in line.lower() or GMSC_FASTA in line:
                                cluster_has_db = True
                            else:
                                cluster_q_seqs.append(line)
            print(f"\n  CD-HIT-2D: {hits_in_shared} downstream seqs share a cluster "
                  f"with a GMSC sequence at ≥90% identity")
            print(f"  ({100*hits_in_shared/len(all_downstream):.2f}% of all downstream seqs)")
        else:
            print("  CD-HIT-2D run failed:")
            print(result.stderr[:500])
    else:
        print("  cd-hit-2d not found (only cd-hit available). "
              "Use cd-hit-2d for two-set comparison.")

elif mmseqs_bin:
    print(f"  Found MMseqs2: {mmseqs_bin}")
    print("  Running MMseqs2 easy-search (downstream vs GMSC at 90% identity)...")

    mmseqs_out = "./log/mmseqs_hits.tsv"
    tmp_dir    = "./log/mmseqs_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        mmseqs_bin, "easy-search",
        merged_fasta,          # query
        GMSC_FASTA,            # target (GMSC)
        mmseqs_out,
        tmp_dir,
        "--min-seq-id", "0.90",
        "--cov-mode",   "0",
        "-c",           "0.80",
        "--threads",    "8",
        "-v",           "1",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        hits_df     = pd.read_csv(mmseqs_out, sep="\t", header=None,
                                  names=["query","target","pident","alnlen",
                                         "mismatch","gapopen","qstart","qend",
                                         "tstart","tend","evalue","bitscore"])
        hit_queries = set(hits_df["query"].str.split("_").str[:-1].str.join("_"))
        print(f"\n  MMseqs2: {hits_df['query'].nunique()} downstream seqs "
              f"have ≥90% identity hit in GMSC")
        print(f"  ({100*hits_df['query'].nunique()/len(all_downstream):.2f}% of all downstream seqs)")

        # per-dataset breakdown
        print(f"\n  Per-dataset MMseqs2 breakdown:")
        for ds_name in DATASETS:
            ds_hits = hits_df[hits_df["query"].str.startswith(ds_name + "_")]["query"].nunique()
            n_total = len(dataset_seqs[ds_name])
            print(f"    {ds_name:<8} {ds_hits:>6} / {n_total:<6} "
                  f"({100*ds_hits/n_total:.2f}%)")
    else:
        print("  MMseqs2 run failed:")
        print(result.stderr[:500])

else:
    print("  Neither CD-HIT nor MMseqs2 found on PATH.")
    print("\n  To run near-identity check manually, install one of:")
    print("    conda install -c bioconda cd-hit")
    print("    conda install -c bioconda mmseqs2")
    print("\n  Then run (MMseqs2 recommended for 10M seqs, much faster):")
    print(f"    mmseqs easy-search {merged_fasta} {GMSC_FASTA} \\")
    print( "      ./log/mmseqs_hits.tsv ./log/mmseqs_tmp \\")
    print( "      --min-seq-id 0.90 -c 0.80 --cov-mode 0 --threads 8")
    print("\n  Or with CD-HIT-2D:")
    print(f"    cd-hit-2d -i {GMSC_FASTA} -i2 {merged_fasta} \\")
    print( "      -o ./log/cdhit2d_out -c 0.90 -n 5 -T 8 -M 16000")

# ─── Write full text report ───────────────────────────────────────────────────
report_path = "./log/overlap_report.txt"
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("GMSC Pretraining vs Downstream Dataset Overlap Report\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"GMSC corpus : {GMSC_FASTA}\n")
    f.write(f"GMSC seqs   : {gmsc_total:,}\n\n")
    f.write("Downstream datasets:\n")
    for name, seqs in dataset_seqs.items():
        f.write(f"  {name:<8}: {len(seqs)} sequences\n")
    f.write(f"\nTotal downstream : {len(all_downstream)}\n")
    f.write(f"Unique downstream: {len(downstream_set)}\n\n")
    f.write("─" * 60 + "\n")
    f.write("Exact match results:\n")
    f.write(f"{'Dataset':<8} {'Total':>8} {'Exact hits':>12} {'Overlap %':>12}\n")
    f.write("─" * 46 + "\n")
    for row in exact_rows:
        f.write(f"{row['dataset']:<8} {row['n_seqs']:>8} "
                f"{row['exact_hits']:>12} {row['overlap_pct']:>11.2f}%\n")
    f.write("\n")
    f.write("Near-identity (≥90%): see MMseqs2/CD-HIT output above.\n")

print(f"\n{'─'*60}")
print(f"Report saved to: {report_path}")
print("Done.")
