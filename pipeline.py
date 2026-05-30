"""
pipeline.py - shared data substrate for the VCPI compound expression-prediction contest.

Both the XGBoost (teammate) and GNN (you) paths import from this module so they
see the same data, use the same train/val split, and score with the same wMSE.

Contest target (per official spec):
  predict, for every test compound c and every gene g in the scored gene set,
  a single number: mean of log2(CPM + 1) across replicates of (c, g),
  at 10 uM, 24 h, THP-1 cells.

Pipeline (called by main() at the bottom):
  1. load_all()           read the 3 input files
  2. filter_contest()     keep wells matching contest condition (10 uM library + DMSO)
  3. counts_to_expression (official helper) -> per-compound mean log2(CPM+1)
  4. restrict to gene_filter.csv (12,995 scored genes)
  5. split off DMSO  -> global baseline (Series, 12995 genes)
  6. scaffold_split() 85/15 by Bemis-Murcko, COMPOUND-level
  7. assemble()           per-compound (compound x gene) matrices for train/val
  8. score_predictions    wrap official score_compounds + Mejia weights
  9. write_submission     long-format parquet matching the contest submission spec
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from vcpi_prediction_contest import (
    counts_to_expression,
    load_gene_filter,
    load_test_compounds,
    score_compounds,
)

RDLogger.DisableLog("rdApp.*")


# ----- CONFIG --------------------------------------------------------------
DATA_DIR = Path(__file__).parent

# Merged 3-release files written by fetch_train_data.py. If those don't exist,
# fall back to the qnu-012-only files the hackathon shipped initially.
_MERGED = (DATA_DIR / "train_counts.parquet").exists()
COMPOUNDS_FILE = (DATA_DIR / "train_chemistry.parquet" if _MERGED
                  else DATA_DIR / "compounds-tvc-qnu-012-2026-05-30.csv")
METADATA_FILE  = (DATA_DIR / "train_metadata.parquet"  if _MERGED
                  else DATA_DIR / "metadata-tvc-qnu-012.csv")
COUNTS_FILE    = (DATA_DIR / "train_counts.parquet"    if _MERGED
                  else DATA_DIR / "vcpi_tvc-qnu-012_counts.parquet")

# Contest condition: library compounds at 10 uM (= 10000 nM) + DMSO controls.
CONTEST_CONCENTRATION_NM = 10_000

# Compound IDs in metadata that should serve as the DMSO baseline.
DMSO_USER_ID = "DMSO"

# Compound-level join key (contest's canonical key).
USER_COMPOUND_ID_COL = "user_compound_id"
COMPOUND_UUID_COL    = "compound"
SMILES_COL           = "smiles"
SEQUENCED_ID_COL     = "sequenced_id"
CONCENTRATION_COL    = "compound_concentration"
CONCENTRATION_UNIT   = "compound_concentration_unit"
IS_CONTROL_COL       = "is_control"

VAL_FRAC   = 0.15
SPLIT_SEED = 42


# ----- LOAD ----------------------------------------------------------------

def _read_table(path):
    """Dispatch on suffix: parquet or csv."""
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def load_compounds(path=COMPOUNDS_FILE):
    df = _read_table(path)
    df = df.drop(columns=["purity_pct"], errors="ignore")  # 100% NaN
    df = df.drop_duplicates(subset=[COMPOUND_UUID_COL])
    return df


def load_metadata(path=METADATA_FILE):
    df = _read_table(path)
    # confirmed 100% NaN or single-valued, no information
    df = df.drop(columns=["condition", "seeded_cell_count"], errors="ignore")
    return df


def load_counts(path=COUNTS_FILE, columns=None):
    """rows=genes (indexed by gene_id), cols=well sequenced_id as STRING.

    counts_to_expression joins counts column names against metadata.sequenced_id
    cast to str, so we keep them as strings here to avoid a re-cast.

    If *columns* is given (set/list of well IDs as strings), only those
    columns plus gene_id are read from the parquet file.  This avoids
    loading the full 9+ GiB matrix when only a subset of wells is needed.
    """
    if columns is not None:
        import pyarrow.parquet as pq
        schema = pq.read_schema(str(path))
        available = set(schema.names)
        read_cols = ["gene_id"] + [c for c in columns if c in available]
        print(f"  [load_counts] selective read: {len(read_cols)-1} / "
              f"{len(available)-1} columns")
        df = pd.read_parquet(path, columns=read_cols)
    else:
        df = pd.read_parquet(path)
    if "gene_id" not in df.columns:
        raise ValueError("expected 'gene_id' column in counts parquet")
    df.set_index("gene_id", inplace=True)
    df.columns = df.columns.astype(str)
    return df


def load_all():
    print("[load] reading 3 input files...")
    return load_compounds(), load_metadata(), load_counts()


# ----- CONTEST FILTER ------------------------------------------------------

def filter_contest(metadata, counts):
    """
    Keep only the wells the contest scores against:
      - library compounds at 10 uM (10000 nM)
      - DMSO controls (any volume)

    Drops all other doses (0.03 / 0.1 / 0.3 / 1 / 3 uM) and any non-DMSO
    controls that aren't at 10 uM. Other 10 uM controls (Staurosporine,
    Brefeldin A, Rigosertib, Trichostatin A) are kept; they're not in
    chemistry.csv so they'll be skipped at the join step naturally.
    """
    is_10um = (
        (metadata[CONCENTRATION_COL] == CONTEST_CONCENTRATION_NM)
        & (metadata[CONCENTRATION_UNIT] == "nM")
    )
    is_dmso = metadata[USER_COMPOUND_ID_COL] == DMSO_USER_ID
    kept = metadata[is_10um | is_dmso].copy()

    well_str = kept[SEQUENCED_ID_COL].astype(str)
    cols_keep = [c for c in counts.columns if c in set(well_str)]
    counts_kept = counts[cols_keep]

    print(f"[filter_contest] wells: {len(metadata):,} -> {len(kept):,}  "
          f"(10uM library + DMSO)")
    print(f"  unique user_compound_ids in kept set: "
          f"{kept[USER_COMPOUND_ID_COL].nunique():,}")
    print(f"  DMSO wells: {int(is_dmso[is_10um | is_dmso].sum()):,}")
    return kept, counts_kept


# ----- EXPRESSION (use official helper) ------------------------------------

def compute_expression(counts, metadata, gene_filter):
    """
    Per-compound mean log2(CPM+1), via the official counts_to_expression helper.
    Restricted to the scored gene set (gene_filter.csv).

    Returns wide DataFrame: rows=user_compound_id, cols=gene_id (gene_filter order),
    values = mean log2(CPM+1) across replicates.
    """
    print("[compute_expression] counts -> long expression via contest helper...")
    # restrict counts rows to scored genes FIRST (cuts memory ~6x)
    scored = [g for g in gene_filter if g in counts.index]
    missing = len(gene_filter) - len(scored)
    if missing:
        print(f"  WARNING: {missing} scored genes absent from this dataset; "
              f"these will be filled with the DMSO baseline at submission time.")
    counts_scored = counts.loc[scored]
    # counts_to_expression needs gene_id as a column, not the index
    counts_long = counts_scored.reset_index()

    expr = counts_to_expression(
        counts_long, metadata,
        sample_col=SEQUENCED_ID_COL,
        compound_col=USER_COMPOUND_ID_COL,
        gene_col="gene_id",
    )
    print(f"  long form: {expr.shape}  unique compounds: "
          f"{expr[USER_COMPOUND_ID_COL].nunique():,}")

    # Manual dense fill rather than pandas pivot - pivot of a 100M+ row frame
    # peaks at many GB; this builds the (n_compounds x n_genes) matrix directly.
    compounds_order = sorted(expr[USER_COMPOUND_ID_COL].unique())
    c_idx = {c: i for i, c in enumerate(compounds_order)}
    g_idx = {g: i for i, g in enumerate(gene_filter)}

    mat = np.zeros((len(compounds_order), len(gene_filter)), dtype=np.float32)
    ci = expr[USER_COMPOUND_ID_COL].map(c_idx).to_numpy()
    gi = expr["gene_id"].map(g_idx).to_numpy()
    # filter rows whose gene isn't in the scored set (defensive; should be empty)
    valid = ~pd.isna(gi)
    mat[ci[valid].astype(int), gi[valid].astype(int)] = expr["expression"].values[valid]

    wide = pd.DataFrame(mat, index=compounds_order, columns=gene_filter)
    wide.index.name = USER_COMPOUND_ID_COL
    print(f"  wide expression: {wide.shape} (compounds x scored_genes)")
    return wide


def split_off_baseline(expr_wide):
    """Pop the DMSO row out as the global baseline; rest is the training pool."""
    if DMSO_USER_ID not in expr_wide.index:
        raise ValueError("DMSO row missing from expression matrix.")
    baseline = expr_wide.loc[DMSO_USER_ID].astype(np.float32)
    rest = expr_wide.drop(index=DMSO_USER_ID)
    print(f"[baseline] DMSO global baseline shape={baseline.shape}, "
          f"per-gene mean={baseline.mean():.3f}, std={baseline.std():.3f}")
    return baseline, rest


# ----- SCAFFOLD SPLIT ------------------------------------------------------

def _murcko_scaffold(smiles):
    if not isinstance(smiles, str) or not smiles:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=False) or ""
    except Exception:
        return ""


def scaffold_split(compounds_df, available_user_ids, val_frac=VAL_FRAC, seed=SPLIT_SEED):
    """
    Bemis-Murcko scaffold split, 85/15 by compound count, COMPOUND-level.
    Only considers compounds in `available_user_ids` (the ones we actually
    have 10 uM expression data for).

    Returns sets of user_compound_id strings.
    """
    pool = compounds_df[compounds_df[USER_COMPOUND_ID_COL].isin(available_user_ids)]
    print(f"[scaffold_split] computing scaffolds for "
          f"{len(pool):,} compounds with 10uM data...")

    groups = {}
    for _, row in pool.iterrows():
        scaf = _murcko_scaffold(row[SMILES_COL])
        groups.setdefault(scaf, []).append(str(row[USER_COMPOUND_ID_COL]))

    rng = random.Random(seed)
    group_list = list(groups.values())
    group_list.sort(key=lambda g: (-len(g), rng.random()))

    total = sum(len(g) for g in group_list)
    train_target = total - int(round(total * val_frac))

    train_ids, val_ids = [], []
    for g in group_list:
        if len(train_ids) + len(g) <= train_target:
            train_ids.extend(g)
        else:
            val_ids.extend(g)

    print(f"  {len(groups):,} unique scaffolds -> "
          f"train={len(train_ids):,} ({len(train_ids)/total:.1%})  "
          f"val={len(val_ids):,} ({len(val_ids)/total:.1%})")
    return set(train_ids), set(val_ids)


# ----- ASSEMBLE ------------------------------------------------------------

def assemble(expr_wide, baseline, compounds_df, train_ids, val_ids):
    """
    Per-compound (compound x gene) tables for train and val, plus the
    chemistry rows the model will consume.

    Returns dict with:
      train_y       (n_train x 12995)  log2(CPM+1) per compound
      val_y         (n_val   x 12995)  same for val
      train_delta   (n_train x 12995)  train_y - baseline
      val_delta     (n_val   x 12995)  val_y - baseline
      train_chem    (n_train x ...)    chemistry rows (smiles + descriptors)
      val_chem      (n_val   x ...)    same for val
      baseline      (12995,)           the global DMSO mean
    """
    train_idx = [c for c in train_ids if c in expr_wide.index]
    val_idx   = [c for c in val_ids   if c in expr_wide.index]

    train_y = expr_wide.loc[train_idx]
    val_y   = expr_wide.loc[val_idx]
    train_delta = (train_y - baseline).astype(np.float32)
    val_delta   = (val_y   - baseline).astype(np.float32)

    chem = compounds_df.set_index(USER_COMPOUND_ID_COL)
    # user_compound_id in chemistry is sometimes int; cast index to str to match
    chem.index = chem.index.astype(str)
    train_chem = chem.loc[[i for i in train_idx if i in chem.index]]
    val_chem   = chem.loc[[i for i in val_idx   if i in chem.index]]

    print(f"[assemble] train: y{train_y.shape}  chem{train_chem.shape}")
    print(f"           val:   y{val_y.shape}    chem{val_chem.shape}")
    return dict(
        train_y=train_y, val_y=val_y,
        train_delta=train_delta, val_delta=val_delta,
        train_chem=train_chem, val_chem=val_chem,
        baseline=baseline,
    )


# ----- SCORERS & SUBMISSION ------------------------------------------------

def score_predictions(pred_wide, truth_wide, gene_filter, weights=None):
    """
    Official wMSE via score_compounds(). Inputs are wide (compound x gene);
    we melt to long for the contest scorer and aggregate.

    Pass `weights=load_weights_matrix()` to match the leaderboard exactly.
    If None, the scorer uses variance-of-truth weights (cheaper sanity check).
    """
    def _to_long(df, value_col):
        long = df.reset_index().melt(
            id_vars=df.index.name or "user_compound_id",
            var_name="gene_id",
            value_name=value_col,
        )
        long = long.rename(columns={
            df.index.name or "user_compound_id": "compound"
        })
        long["compound"] = long["compound"].astype(str)
        return long

    truth_long = _to_long(truth_wide, "expression")
    pred_long  = _to_long(pred_wide,  "predicted_expression")

    per_compound = score_compounds(
        truth_long, pred_long,
        gene_filter=gene_filter, weights=weights,
    )
    return per_compound


def write_submission(pred_wide, gene_filter, out_path):
    """Long-format parquet exactly as the contest expects."""
    pred_wide = pred_wide.reindex(columns=gene_filter)
    long = pred_wide.reset_index().melt(
        id_vars=pred_wide.index.name or "user_compound_id",
        var_name="gene_id",
        value_name="predicted_expression",
    )
    long = long.rename(columns={
        pred_wide.index.name or "user_compound_id": "compound"
    })
    long["compound"] = long["compound"].astype(str)
    # contest requires non-negative
    long["predicted_expression"] = long["predicted_expression"].clip(lower=0.0).astype(np.float32)
    expected = len(pred_wide) * len(gene_filter)
    if len(long) != expected:
        raise ValueError(f"submission rows {len(long)} != expected {expected}")
    long[["compound", "gene_id", "predicted_expression"]].to_parquet(out_path, index=False)
    print(f"[write_submission] {len(long):,} rows -> {out_path}")
    return long


# ----- MAIN ----------------------------------------------------------------

def main():
    # Load metadata & compounds first (small); use metadata to figure out
    # which wells match the contest condition so we can read ONLY those
    # columns from the huge counts parquet (avoids 9+ GiB full load).
    print("[load] reading input files...")
    compounds = load_compounds()
    metadata  = load_metadata()

    is_10um = (
        (metadata[CONCENTRATION_COL] == CONTEST_CONCENTRATION_NM)
        & (metadata[CONCENTRATION_UNIT] == "nM")
    )
    is_dmso = metadata[USER_COMPOUND_ID_COL] == DMSO_USER_ID
    keep_wells = set(metadata.loc[is_10um | is_dmso, SEQUENCED_ID_COL].astype(str))
    counts = load_counts(columns=keep_wells)

    gene_filter = load_gene_filter()
    print(f"[contest] {len(gene_filter):,} scored genes loaded")

    metadata_f, counts_f = filter_contest(metadata, counts)
    del metadata, counts  # free memory

    expr = compute_expression(counts_f, metadata_f, gene_filter)
    del counts_f  # the largest in-memory object now released

    baseline, expr_train_pool = split_off_baseline(expr)

    train_ids, val_ids = scaffold_split(
        compounds,
        available_user_ids=set(expr_train_pool.index.astype(str)),
    )
    splits = assemble(expr_train_pool, baseline, compounds, train_ids, val_ids)

    # also load the bundled test compounds + Mejia weights for downstream use
    test_compounds = load_test_compounds()
    test_compounds["compound"] = test_compounds["compound"].astype(str)
    print(f"[contest] test compounds: {len(test_compounds):,}")

    print("\n[main] ready:")
    print(f"  train_y     {splits['train_y'].shape}")
    print(f"  train_delta {splits['train_delta'].shape}")
    print(f"  val_y       {splits['val_y'].shape}")
    print(f"  val_delta   {splits['val_delta'].shape}")
    print(f"  train_chem  {splits['train_chem'].shape}")
    print(f"  baseline    {splits['baseline'].shape}")
    print(f"  test set    {len(test_compounds)} compounds to predict")
    print("  -> model target: train_delta (= train_y - baseline)")
    print("  -> submission: predicted_delta + baseline, long-format")

    return {
        **splits,
        "test_compounds": test_compounds,
        "gene_filter": gene_filter,
    }


if __name__ == "__main__":
    main()
