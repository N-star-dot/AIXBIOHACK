import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, Dropout
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cellxgene_census

# --- 1. Graph Construction ---
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs()
    ]

def smiles_to_graph(smiles, target_expr=None, context_expr=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Data(x=torch.zeros((1, 6), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    context=torch.tensor(context_expr, dtype=torch.float).unsqueeze(0) if context_expr is not None else None,
                    y=torch.tensor(target_expr, dtype=torch.float).unsqueeze(0) if target_expr is not None else None)
        
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    if context_expr is not None: data.context = torch.tensor(context_expr, dtype=torch.float).unsqueeze(0)
    if target_expr is not None: data.y = torch.tensor(target_expr, dtype=torch.float).unsqueeze(0)
    return data

class CompoundGeneDataset(Dataset):
    def __init__(self, smiles_list, context_matrix, target_matrix=None):
        super().__init__(None, None, None)
        self.smiles_list = smiles_list
        self.context_matrix = context_matrix
        self.target_matrix = target_matrix

    def len(self): return len(self.smiles_list)
    def get(self, idx):
        return smiles_to_graph(self.smiles_list[idx], 
                               self.target_matrix[idx] if self.target_matrix is not None else None, 
                               self.context_matrix[idx])

# --- 2. Model Architecture ---
class GNNPredictor(torch.nn.Module):
    def __init__(self, node_dim=6, hidden_dim=128, context_dim=50, output_dim=500):
        super(GNNPredictor, self).__init__()
        # Graph Convolution Layers with BatchNorm & Dropout
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.dropout = Dropout(0.2)
        
        # Regression Head
        self.mlp = Sequential(
            Linear(hidden_dim + context_dim, 256),
            BatchNorm1d(256),
            ReLU(),
            Dropout(0.3),
            Linear(256, output_dim)
        )

    def forward(self, x, edge_index, batch, context):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        
        context = context.view(x.size(0), -1)
        x = torch.cat([x, context], dim=1)
        
        return self.mlp(x)

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

# --- 4. Main Loop ---
def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    print("Loading baseline data...")
    cache_file = "cellxgene_cache.npz"
    if os.path.exists(cache_file):
        print("Found local cache, loading instantly...")
        data = np.load(cache_file, allow_pickle=True)
        y_train_full = data['y_train_full']
        gene_ids = data['gene_ids'].tolist()
        n_train, n_genes = y_train_full.shape
    else:
        print("Querying CellxGene Census (this will be cached for future runs)...")
        with cellxgene_census.open_soma() as census:
            adata = cellxgene_census.get_anndata(
                census, organism="Homo sapiens",
                obs_value_filter="tissue == 'blood' and is_primary_data == True and cell_type == 'monocyte'"
            )
            if adata.n_obs > 100: adata = adata[:100, :500].copy()
                
        y_train_full = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        gene_ids = adata.var_names.tolist()
        n_train, n_genes = y_train_full.shape
        np.savez(cache_file, y_train_full=y_train_full, gene_ids=gene_ids)
        print("Data successfully cached!")
    
    # Log2(x + 1) transform: compresses raw counts (0-15000) into NN-friendly range (0-14)
    print("Applying log2(x + 1) normalization...")
    y_train_full = np.log2(y_train_full + 1)
        
    baseline_expr_df = pd.DataFrame(y_train_full.mean(axis=0).reshape(1, -1), columns=gene_ids)
    
    train_smiles_full = ["CCO" for _ in range(n_train)]
    truth_padj = np.random.uniform(0, 0.1, size=(n_train, n_genes))
    truth_l2fc = np.random.uniform(-1, 1, size=(n_train, n_genes))
    
    n_test = 20
    test_compounds = [f"CMP_test_{str(i).zfill(4)}" for i in range(1, n_test + 1)]
    test_smiles = ["CC(=O)O" for _ in range(n_test)]

    # Split Data (80/20)
    train_smiles, val_smiles, y_train, y_val = train_test_split(
        train_smiles_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print("Preparing Context Features...")
    pca_components = 50
    n_comps = min(pca_components, baseline_expr_df.shape[0], baseline_expr_df.shape[1])
    pca = PCA(n_components=n_comps, random_state=42)
    scaler = StandardScaler()
    
    train_ctx_raw = np.tile(baseline_expr_df.values, (len(train_smiles), 1))
    val_ctx_raw = np.tile(baseline_expr_df.values, (len(val_smiles), 1))
    test_ctx_raw = np.tile(baseline_expr_df.values, (n_test, 1))
    
    train_ctx = scaler.fit_transform(pca.fit_transform(train_ctx_raw))
    val_ctx = scaler.transform(pca.transform(val_ctx_raw))
    test_ctx = scaler.transform(pca.transform(test_ctx_raw))
    
    # Loaders
    train_loader = DataLoader(CompoundGeneDataset(train_smiles, train_ctx, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(CompoundGeneDataset(val_smiles, val_ctx, y_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(CompoundGeneDataset(test_smiles, test_ctx, None), batch_size=32, shuffle=False)
    
    model = GNNPredictor(context_dim=n_comps, output_dim=n_genes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    max_grad_norm = 1.0  # Gradient clipping threshold
    
    # Training Loop (full 100 epochs, best weights saved)
    from tqdm import tqdm
    epochs = 100
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(epochs), desc="GNN Training")
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, batch.context)
            out = out.clamp(-5, 20)  # Clamp outputs to valid log2(CPM+1) range
            loss = criterion(out, batch.y.view(out.size()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, batch.context)
                out = out.clamp(-5, 20)  # Same clamp on validation
                val_loss += criterion(out, batch.y.view(out.size())).item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epoch_pbar.set_postfix({'Train': f"{train_loss/len(train_loader):.4f}", 'Val': f"{val_loss:.4f}"})
        
        # Checkpoint Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_gnn.pt')
                
    # Load Best Weights for Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load('best_gnn.pt', weights_only=True))
    model.eval()
    
    # Local Validation Evaluation
    full_loader = DataLoader(CompoundGeneDataset(train_smiles_full, 
                                                 scaler.transform(pca.transform(np.tile(baseline_expr_df.values, (len(train_smiles_full), 1)))), 
                                                 y_train_full), batch_size=32, shuffle=False)
    all_train_preds = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.context)
            all_train_preds.append(out.clamp(-5, 20).cpu().numpy())
            
    y_train_pred = np.vstack(all_train_preds)
    
    train_compounds = [f"CMP_train_{str(i).zfill(4)}" for i in range(len(y_train_pred))]
    df_pred_long = pd.DataFrame(y_train_pred, columns=gene_ids).assign(compound=train_compounds).melt(id_vars=['compound'], var_name='gene_id', value_name='predicted_log2_CPM')
    df_truth_long = pd.DataFrame(y_train_full, columns=gene_ids).assign(compound=train_compounds).melt(id_vars=['compound'], var_name='gene_id', value_name='truth_expression')
    df_padj_long = pd.DataFrame(truth_padj, columns=gene_ids).assign(compound=train_compounds).melt(id_vars=['compound'], var_name='gene_id', value_name='padj')
    df_l2fc_long = pd.DataFrame(truth_l2fc, columns=gene_ids).assign(compound=train_compounds).melt(id_vars=['compound'], var_name='gene_id', value_name='L2FC')
    
    df_truth_full = pd.merge(pd.merge(df_truth_long, df_padj_long, on=['compound', 'gene_id']), df_l2fc_long, on=['compound', 'gene_id'])
    
    active_comps = get_active_compounds(df_truth_full)
    if len(active_comps) > 0:
        g_weights = calculate_gene_weights(df_truth_full, active_comps)
        final_score = calculate_hackathon_wmse(df_pred_long, df_truth_full, active_comps, g_weights)
        print(f"Local wMSE on Full Train Split (Active Compounds): {final_score:.4f}")
    
    # Test Predictions
    print("Formatting submission file...")
    all_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.context)
            all_test_preds.append(out.clamp(-5, 20).cpu().numpy())
            
    df_pred_long = pd.DataFrame(np.vstack(all_test_preds), columns=gene_ids).assign(compound=test_compounds).melt(id_vars=['compound'], var_name='gene_id', value_name='predicted_log2(CPM+1)')
    df_pred_long[['compound', 'gene_id', 'predicted_log2(CPM+1)']].to_parquet('gnn_predictions.parquet', index=False)
    print("Successfully saved GNN predictions to gnn_predictions.parquet")

if __name__ == "__main__":
    main()
