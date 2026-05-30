import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# --- 1. Feature Engineering ---
def smiles_to_morgan(smiles, radius=3, n_bits=2048):
    if not isinstance(smiles, str) or pd.isna(smiles):
        return np.zeros(n_bits)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return np.zeros(n_bits)

def prepare_features(smiles_series, baseline_expr_df, pca_components=50, pca_model=None):
    fps = np.vstack(smiles_series.apply(lambda x: smiles_to_morgan(x)).values)
    
    if pca_model is None:
        n_comps = min(pca_components, baseline_expr_df.shape[0], baseline_expr_df.shape[1])
        pca_model = PCA(n_components=n_comps, random_state=42)
        baseline_vals = pca_model.fit_transform(baseline_expr_df.values)
    else:
        baseline_vals = pca_model.transform(baseline_expr_df.values)
        
    if baseline_vals.shape[0] == 1 and len(fps) > 1:
        baseline_vals = np.tile(baseline_vals, (len(fps), 1))
        
    X = np.concatenate([fps, baseline_vals], axis=1)
    return X, pca_model

# --- 2. Model Architecture ---
def train_model(X_train, y_train, X_val, y_val):
    print("Initializing Mac-Optimized XGBoost (CPU Hist)...")
    model = xgb.XGBRegressor(
        tree_method='hist',
        multi_strategy='multi_output_tree',
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    return model

# --- 3. Scoring Functions ---
def get_active_compounds(truth_df):
    valid_degs = truth_df[(truth_df['padj'] < 0.05) & (truth_df['L2FC'].abs() >= 0.5)]
    deg_counts = valid_degs.groupby('compound').size()
    return deg_counts[deg_counts >= 5].index.tolist()

def calculate_gene_weights(truth_df, active_compounds):
    active_data = truth_df[truth_df['compound'].isin(active_compounds)]
    gene_variances = active_data.groupby('gene_id')['truth_expression'].var()
    gene_weights = 1 / (gene_variances + 1e-8)
    gene_weights = gene_weights / gene_weights.sum() * len(gene_weights)
    return gene_weights.to_dict()

def calculate_hackathon_wmse(predictions_df, truth_df, active_compounds, gene_weights):
    merged = pd.merge(predictions_df, truth_df, on=['compound', 'gene_id'])
    scoring_data = merged[merged['compound'].isin(active_compounds)].copy()
    if scoring_data.empty: return np.nan
    scoring_data['weight'] = scoring_data['gene_id'].map(gene_weights)
    scoring_data['squared_error'] = (scoring_data['predicted_log2_CPM'] - scoring_data['truth_expression']) ** 2
    scoring_data['weighted_sq_error'] = scoring_data['weight'] * scoring_data['squared_error']
    return scoring_data['weighted_sq_error'].mean()

# --- 4. Main Pipeline ---
def main():
    print("Loading baseline data...")
    import cellxgene_census
    cache_file = "cellxgene_cache.npz"
    if os.path.exists(cache_file):
        print("Found local cache, loading instantly...")
        data = np.load(cache_file, allow_pickle=True)
        y_train = data['y_train_full']
        gene_ids = data['gene_ids'].tolist()
        n_train, n_genes = y_train.shape
    else:
        print("Querying CellxGene Census (this will be cached for future runs)...")
        with cellxgene_census.open_soma() as census:
            adata = cellxgene_census.get_anndata(
                census, organism="Homo sapiens",
                obs_value_filter="tissue == 'blood' and is_primary_data == True and cell_type == 'monocyte'"
            )
            if adata.n_obs > 100:
                adata = adata[:100, :500].copy()
                
        y_train = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        gene_ids = adata.var_names.tolist()
        n_train, n_genes = y_train.shape
        np.savez(cache_file, y_train_full=y_train, gene_ids=gene_ids)
        print("Data successfully cached!")
    
    # Log2(x + 1) transform: compresses raw counts (0-15000) into tree-friendly range (0-14)
    print("Applying log2(x + 1) normalization...")
    y_train = np.log2(y_train + 1)
        
    baseline_expr = pd.DataFrame(y_train.mean(axis=0).reshape(1, -1), columns=gene_ids)
    
    train_smiles = pd.Series(["CCO" for _ in range(n_train)])
    truth_padj = np.random.uniform(0, 0.1, size=(n_train, n_genes))
    truth_l2fc = np.random.uniform(-1, 1, size=(n_train, n_genes))
    
    n_test = 20
    test_compounds = [f"CMP_test_{str(i).zfill(4)}" for i in range(1, n_test + 1)]
    test_smiles = pd.Series(["CC(=O)O" for _ in range(n_test)])
    
    print("Preparing features with PCA compression...")
    X_train_full, pca_model = prepare_features(train_smiles, baseline_expr, pca_components=50)
    X_test, _ = prepare_features(test_smiles, baseline_expr, pca_model=pca_model)
    
    # Train/Validation Split
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_full, y_train, test_size=0.2, random_state=42
    )
    
    model = train_model(X_train, y_train_split, X_val, y_val)
    
    # Local Validation Evaluation
    y_train_pred = model.predict(X_train_full) # Evaluate on full for scoring script compatibility
    train_compounds = [f"CMP_train_{str(i).zfill(4)}" for i in range(len(y_train_pred))]
    
    df_pred_wide = pd.DataFrame(y_train_pred, columns=gene_ids)
    df_pred_wide['compound'] = train_compounds
    df_pred_long = df_pred_wide.melt(id_vars=['compound'], var_name='gene_id', value_name='predicted_log2_CPM')
    
    df_truth_wide_expr = pd.DataFrame(y_train, columns=gene_ids)
    df_truth_wide_expr['compound'] = train_compounds
    df_truth_long = df_truth_wide_expr.melt(id_vars=['compound'], var_name='gene_id', value_name='truth_expression')
    
    df_padj_wide = pd.DataFrame(truth_padj, columns=gene_ids)
    df_padj_wide['compound'] = train_compounds
    df_padj_long = df_padj_wide.melt(id_vars=['compound'], var_name='gene_id', value_name='padj')
    
    df_l2fc_wide = pd.DataFrame(truth_l2fc, columns=gene_ids)
    df_l2fc_wide['compound'] = train_compounds
    df_l2fc_long = df_l2fc_wide.melt(id_vars=['compound'], var_name='gene_id', value_name='L2FC')
    
    df_truth_full = pd.merge(df_truth_long, df_padj_long, on=['compound', 'gene_id'])
    df_truth_full = pd.merge(df_truth_full, df_l2fc_long, on=['compound', 'gene_id'])
    
    active_comps = get_active_compounds(df_truth_full)
    if len(active_comps) > 0:
        g_weights = calculate_gene_weights(df_truth_full, active_comps)
        final_score = calculate_hackathon_wmse(df_pred_long, df_truth_full, active_comps, g_weights)
        print(f"Local wMSE on Train (Active Compounds): {final_score:.4f}")
    
    print("Formatting submission file...")
    y_test_pred = model.predict(X_test)
    df_pred_wide = pd.DataFrame(y_test_pred, columns=gene_ids)
    df_pred_wide['compound'] = test_compounds
    df_pred_long = df_pred_wide.melt(id_vars=['compound'], var_name='gene_id', value_name='predicted_log2(CPM+1)')
    df_pred_long = df_pred_long[['compound', 'gene_id', 'predicted_log2(CPM+1)']]
    df_pred_long.to_parquet('predictions.parquet', index=False)
    print("Successfully saved predictions to predictions.parquet")

if __name__ == "__main__":
    main()
