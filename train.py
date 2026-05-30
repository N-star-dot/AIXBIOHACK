"""
Train pipeline for VCPI Prediction Contest.
Predicts per-compound mean log2(CPM+1) expression for 12,995 scored genes.

Strategy:
  1. Predict delta from DMSO baseline
  2. Rich molecular features (Morgan FP + MACCS + descriptors)
  3. Ridge regression (fast, stable, handles 12,995 targets natively)
  4. Evaluate with contest wMSE metric

Usage:  python3 train.py
Output: predictions.parquet
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.linear_model import Ridge
from pathlib import Path

DATA_DIR = Path(".")
CONTEST_DIR = Path("vcpi-prediction-contest-2026/src/vcpi_prediction_contest/data_files")
GENE_FILTER_PATH = CONTEST_DIR / "gene_filter.csv"
TEST_COMPOUNDS_PATH = CONTEST_DIR / "test_compounds.csv"
WEIGHTS_PATH = DATA_DIR / "weights.parquet"
FP_RADIUS = 3
FP_BITS = 2048
MIN_UMI = 50_000
MAX_MITO = 20.0


# ─── Data ─────────────────────────────────────────────────────

def load_data():
    print("Loading parquet files...")
    counts = pd.read_parquet(DATA_DIR / "train_counts.parquet")
    metadata = pd.read_parquet(DATA_DIR / "train_metadata.parquet")
    chemistry = pd.read_parquet(DATA_DIR / "train_chemistry.parquet")
    return counts, metadata, chemistry


def qc_filter(metadata):
    n_before = len(metadata)
    mask = (metadata["total_umi_count"] >= MIN_UMI) & (metadata["percent_mitochondrial"] <= MAX_MITO)
    metadata = metadata[mask].copy()
    print(f"QC: {n_before} -> {len(metadata)} samples")
    return metadata


def compute_log2cpm(counts, sample_ids):
    gene_ids = counts["gene_id"].values
    sample_cols = [str(s) for s in sample_ids if str(s) in counts.columns]
    mat = counts.set_index("gene_id")[sample_cols].values.astype(np.float64)
    lib_sizes = mat.sum(axis=0)
    cpm = mat / lib_sizes[np.newaxis, :] * 1e6
    return np.log2(cpm + 1), gene_ids, sample_cols


def aggregate_per_compound(log2cpm, gene_ids, sample_cols, metadata):
    sid_str = metadata["sequenced_id"].astype(str)
    ucid_str = metadata["user_compound_id"].astype(str)
    sample_set = set(sample_cols)
    mask = sid_str.isin(sample_set)
    sample_to_compound = dict(zip(sid_str[mask], ucid_str[mask]))

    compound_groups, dmso_indices = {}, []
    for i, s in enumerate(sample_cols):
        cid = sample_to_compound.get(s)
        if cid == "DMSO":
            dmso_indices.append(i)
        elif cid:
            compound_groups.setdefault(cid, []).append(i)

    dmso_mean = log2cpm[:, dmso_indices].mean(axis=1) if dmso_indices else np.zeros(len(gene_ids))
    compounds = sorted(compound_groups.keys())
    expr = np.column_stack([log2cpm[:, compound_groups[c]].mean(axis=1) for c in compounds])

    print(f"Aggregated {len(compounds)} compounds, {len(dmso_indices)} DMSO controls")
    return expr, compounds, dmso_mean


# ─── Features ─────────────────────────────────────────────────

DESCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumRotatableBonds, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.NumAromaticRings, Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    Descriptors.RingCount, Descriptors.NumHeteroatoms, Descriptors.BertzCT,
]


def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and not pd.isna(smiles) else None
    if mol is None:
        return np.zeros(FP_BITS + 167 + len(DESCS))
    morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS))
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
    desc_vals = []
    for func in DESCS:
        try:
            v = func(mol)
            desc_vals.append(v if np.isfinite(v) else 0.0)
        except Exception:
            desc_vals.append(0.0)
    return np.concatenate([morgan, maccs, desc_vals])


def build_features(compounds, chemistry):
    chem_lookup = dict(zip(chemistry["user_compound_id"].astype(str), chemistry["smiles"]))
    try:
        test_df = pd.read_csv(TEST_COMPOUNDS_PATH, dtype={"compound": str})
        for _, row in test_df.iterrows():
            if row["compound"] not in chem_lookup and pd.notna(row.get("smiles")):
                chem_lookup[row["compound"]] = row["smiles"]
    except Exception:
        pass
    X = np.vstack([mol_features(chem_lookup.get(c, "")) for c in compounds])
    print(f"Features: {X.shape} ({(X.sum(axis=1) > 0).sum()}/{len(compounds)} valid)")

    # Scale continuous descriptors (last 12 cols) to similar range as binary features
    n_binary = FP_BITS + 167
    desc_block = X[:, n_binary:]
    means = desc_block.mean(axis=0)
    stds = desc_block.std(axis=0)
    stds[stds == 0] = 1.0
    X[:, n_binary:] = (desc_block - means) / stds
    return X, means, stds


def apply_feature_scaling(X, means, stds):
    X = X.copy()
    n_binary = FP_BITS + 167
    X[:, n_binary:] = (X[:, n_binary:] - means) / stds
    return X


# ─── Weights & Eval ──────────────────────────────────────────

def load_gene_weights(scored_genes):
    if not WEIGHTS_PATH.exists():
        print("No weights.parquet — uniform weights")
        return np.ones(len(scored_genes)) / len(scored_genes)
    print("Loading Mejia weights...")
    W = pd.read_parquet(WEIGHTS_PATH)
    gene_to_idx = {g: i for i, g in enumerate(scored_genes)}
    weights = np.zeros(len(scored_genes))
    for g in scored_genes:
        if g in W.index:
            weights[gene_to_idx[g]] = W.loc[g].mean()
    total = weights.sum()
    weights = weights / total if total > 0 else np.ones(len(scored_genes)) / len(scored_genes)
    print(f"Weights for {(weights > 0).sum()}/{len(scored_genes)} genes")
    return weights


def wmse(preds, truth, weights):
    sq_err = (preds - truth) ** 2
    return (sq_err * weights[np.newaxis, :]).sum(axis=1).mean()


# ─── Main ─────────────────────────────────────────────────────

def main():
    counts, metadata, chemistry = load_data()
    gene_filter = pd.read_csv(GENE_FILTER_PATH)["gene_id"].tolist()
    test_df = pd.read_csv(TEST_COMPOUNDS_PATH, dtype={"compound": str})
    test_ids = test_df["compound"].tolist()
    print(f"Genes: {len(gene_filter)}, Test compounds: {len(test_ids)}")

    metadata = qc_filter(metadata)

    print("Computing log2(CPM+1)...")
    log2cpm, all_genes, sample_cols = compute_log2cpm(counts, metadata["sequenced_id"].values)
    gene_mask = np.isin(all_genes, gene_filter)
    log2cpm = log2cpm[gene_mask]
    scored_genes = all_genes[gene_mask]
    print(f"Scored genes: {len(scored_genes)}")

    expr, train_compounds, dmso = aggregate_per_compound(log2cpm, scored_genes, sample_cols, metadata)
    delta = expr - dmso[:, np.newaxis]
    print(f"Delta range: [{delta.min():.2f}, {delta.max():.2f}]")

    weights = load_gene_weights(scored_genes)

    print("Building features...")
    X_train, desc_means, desc_stds = build_features(train_compounds, chemistry)

    # Build test features — merge SMILES from chemistry + test_compounds.csv
    chem_lookup = dict(zip(chemistry["user_compound_id"].astype(str), chemistry["smiles"]))
    test_smiles_lookup = dict(zip(test_df["compound"], test_df["smiles"]))
    for c in test_ids:
        if c not in chem_lookup and c in test_smiles_lookup:
            chem_lookup[c] = test_smiles_lookup[c]
    X_test_raw = np.vstack([mol_features(chem_lookup.get(c, "")) for c in test_ids])
    X_test = apply_feature_scaling(X_test_raw, desc_means, desc_stds)
    print(f"X_test: {X_test.shape}")

    # ── Train/val split ──
    n = len(train_compounds)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    tr, va = idx[:split], idx[split:]

    # ── Sweep Ridge alpha ──
    print("\n── Ridge alpha sweep ──")
    best_alpha, best_score = 100.0, float("inf")
    for alpha in [10, 50, 100, 500, 1000, 5000]:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train[tr], delta[:, tr].T)
        preds = np.clip(model.predict(X_train[va]) + dmso[np.newaxis, :], 0, 20)
        score = wmse(preds, expr[:, va].T, weights)
        mse_val = np.mean((preds - expr[:, va].T) ** 2)
        print(f"  alpha={alpha:>5} → MSE: {mse_val:.6f}, wMSE: {score:.6f}")
        if score < best_score:
            best_alpha, best_score = alpha, score

    # ── DMSO baseline comparison ──
    dmso_preds = np.tile(dmso, (len(va), 1))
    dmso_score = wmse(dmso_preds, expr[:, va].T, weights)
    print(f"  DMSO baseline → wMSE: {dmso_score:.6f}")
    print(f"Best alpha: {best_alpha} (wMSE: {best_score:.6f})")

    # ── Final model on all data ──
    print(f"\nTraining final Ridge (alpha={best_alpha}) on all {n} compounds...")
    final = Ridge(alpha=best_alpha, fit_intercept=True)
    final.fit(X_train, delta.T)

    # ── Predict ──
    test_delta = final.predict(X_test)
    test_preds = np.clip(test_delta + dmso[np.newaxis, :], 0, 20)

    # ── Format submission ──
    df = pd.DataFrame(test_preds, columns=scored_genes)
    df["compound"] = test_ids
    df_long = df.melt(id_vars=["compound"], var_name="gene_id", value_name="predicted_expression")
    df_long = df_long[["compound", "gene_id", "predicted_expression"]]
    df_long.to_parquet("predictions.parquet", index=False)

    print(f"\nSaved predictions.parquet: {df_long.shape[0]:,} rows")
    print(f"({len(test_ids)} compounds × {len(scored_genes)} genes)")
    print("Done!")


if __name__ == "__main__":
    main()
