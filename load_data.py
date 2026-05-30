import os
import pandas as pd
import polars as pl
import vcpi
from vcpi_prediction_contest import load_weights_matrix
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Downloading weights matrix...")
    load_weights_matrix().to_parquet("weights.parquet")
    print("Saved weights.parquet (12,995 genes x 14,031 compounds)")

    JOBS = ["tvc-bhr-009", "tvc-kdl-010", "tvc-qnu-012"]
    counts_pieces, metadata_pieces, chemistry_pieces = [], [], []

    for job in JOBS:
        print(f"Loading experiment: {job}")
        exp = vcpi.load_experiment(job)
        meta = exp["metadata"].filter(
            (pl.col("cell_line") == "THP-1")
            & (pl.col("timepoint") == "24h")
            & (
                ((pl.col("compound_concentration") == 10_000)
                 & (pl.col("compound_concentration_unit") == "nM"))
                | (pl.col("user_compound_id") == "DMSO")
            )
        )
        keep = set(meta["sequenced_id"].cast(pl.Utf8).to_list())
        data = exp["data"].select(
            ["gene_id", *[c for c in exp["data"].columns if c != "gene_id" and c in keep]]
        )
        counts_pieces.append(data.to_pandas().set_index("gene_id"))
        metadata_pieces.append(meta.to_pandas())
        chemistry_pieces.append(exp["chemistry"].to_pandas())
        del exp, meta, data

    print("Assembling final dataframes...")
    counts = (
        pd.concat(counts_pieces, axis=1, join="outer")
        .fillna(0)
        .astype("int32")
        .reset_index()
    )
    metadata = pd.concat(metadata_pieces, ignore_index=True)
    chemistry = (
        pd.concat(chemistry_pieces, ignore_index=True)
        .drop_duplicates(subset=["compound"])
        .reset_index(drop=True)
    )

    counts.to_parquet("train_counts.parquet")
    print(f"Saved train_counts.parquet ({counts.shape[0]} genes x {counts.shape[1]-1} samples)")

    metadata.to_parquet("train_metadata.parquet")
    print(f"Saved train_metadata.parquet ({metadata.shape[0]} samples x {metadata.shape[1]} columns)")

    chemistry.to_parquet("train_chemistry.parquet")
    print(f"Saved train_chemistry.parquet ({chemistry.shape[0]} compounds x {chemistry.shape[1]} columns)")

    print("Done! All data saved.")


if __name__ == "__main__":
    main()
