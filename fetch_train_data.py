"""
fetch_train_data.py - download + merge the 3 VCPI contest training releases.

Pulls tvc-bhr-009, tvc-kdl-010, tvc-qnu-012 via vcpi-client, filters each to
the contest condition (THP-1, 24h, 10 uM library + DMSO), and writes:

  train_counts.parquet     gene_id index, well-sequenced_id columns (Int32)
  train_metadata.parquet   one row per kept well
  train_chemistry.parquet  one row per unique compound (deduplicated)

Uses polars throughout (the spec's pandas-concat recipe peaks ~22 GB; this
streams + lazy-joins, peak ~5 GB).

Requires TVC_TOKEN env var (set in your shell, NOT hardcoded here).
"""
import os
import sys

if not os.environ.get("TVC_TOKEN"):
    sys.exit("ERROR: TVC_TOKEN not set. Run: $env:TVC_TOKEN = 'your-token'")

import polars as pl
import vcpi

JOBS = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]

counts_pieces, metadata_pieces, chemistry_pieces = [], [], []

for job in JOBS:
    print(f"\n[fetch] {job}: downloading...", flush=True)
    exp = vcpi.load_experiment(job)
    meta_full = exp["metadata"]
    print(f"  raw metadata rows: {meta_full.height:,}", flush=True)

    meta = meta_full.filter(
        (pl.col("cell_line") == "THP-1")
        & (pl.col("timepoint") == "24h")
        & (
            ((pl.col("compound_concentration") == 10_000)
             & (pl.col("compound_concentration_unit") == "nM"))
            | (pl.col("user_compound_id") == "DMSO")
        )
    )
    keep = set(meta["sequenced_id"].cast(pl.Utf8).to_list())
    cols_keep = ["gene_id"] + [c for c in exp["data"].columns
                                if c != "gene_id" and c in keep]
    data = exp["data"].select(cols_keep)
    print(f"  after filter: {meta.height:,} wells   "
          f"counts cols (incl gene_id): {data.shape[1]:,}", flush=True)

    counts_pieces.append(data)
    metadata_pieces.append(meta)
    chemistry_pieces.append(exp["chemistry"])
    del exp, meta_full

print(f"\n[merge] joining {len(counts_pieces)} counts pieces on gene_id...", flush=True)
counts = counts_pieces[0]
for i, piece in enumerate(counts_pieces[1:], 2):
    print(f"  step {i}/{len(counts_pieces)}: outer-joining...", flush=True)
    counts = counts.join(piece, on="gene_id", how="full", coalesce=True)
counts = counts.fill_null(0)
non_gene = [c for c in counts.columns if c != "gene_id"]
counts = counts.with_columns([pl.col(c).cast(pl.Int32) for c in non_gene])
print(f"  merged counts shape: {counts.shape}", flush=True)

metadata = pl.concat(metadata_pieces, how="vertical_relaxed")
chemistry = (pl.concat(chemistry_pieces, how="vertical_relaxed")
             .unique(subset=["compound"]))
print(f"  metadata shape:  {metadata.shape}", flush=True)
print(f"  chemistry shape: {chemistry.shape}", flush=True)

print("\n[write] saving parquets...", flush=True)
counts.write_parquet("train_counts.parquet")
metadata.write_parquet("train_metadata.parquet")
chemistry.write_parquet("train_chemistry.parquet")
print("[done] wrote train_counts.parquet, train_metadata.parquet, "
      "train_chemistry.parquet", flush=True)
