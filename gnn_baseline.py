"""
gnn_baseline.py - molecular-graph GNN with a LOW-RANK regression head.

Architecture:
  SMILES -> molecular graph -> 2 GCN layers -> mean+max pool ->
  MLP -> 128 latent components -> reconstruct delta via SVD basis ->
  add DMSO baseline -> per-gene log2(CPM+1) prediction.

Why low-rank:
  The contest target is 12,995 genes. Regressing each gene independently
  ignores correlations across genes (and the GNN's tiny head would have to
  emit a 12,995-dim vector). SVD decomposes train_delta into ~128 principal
  axes of compound-response variation. The GNN predicts coordinates in that
  basis; multiplying by V_k.T recovers all 12,995 genes. This both regularizes
  the output (low-rank = less overfit) and speeds training (128 << 12,995).

Run order:
  1. import substrate from pipeline.py
  2. fit TruncatedSVD on train_delta -> V_k (12995 x 128)
  3. project: Z_train = train_delta @ V_k    (8721 x 128)
  4. build PyG graphs for every train/val/test compound (SMILES -> graph)
  5. train GNN to minimise MSE(z_pred, Z_train)
  6. each val epoch: latent-MSE; every K epochs full wMSE via official scorer
  7. final: predict test latents -> delta -> expression -> submission parquet
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger
from sklearn.decomposition import TruncatedSVD
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from tqdm import tqdm

from vcpi_prediction_contest import load_weights_matrix

from pipeline import main as build_splits
from pipeline import score_predictions, write_submission

RDLogger.DisableLog("rdApp.*")


# ----- CONFIG --------------------------------------------------------------
LATENT_DIM     = 512         # SVD components — 512 captures ~65% of delta variance
HIDDEN_DIM     = 128
DROPOUT        = 0.2
BATCH_SIZE     = 64
EPOCHS         = 50
LR             = 0.0005
WEIGHT_DECAY   = 1e-4
GRAD_CLIP      = 1.0
WMSE_EVAL_EVERY = 5
SEED           = 42

# Chemistry descriptor columns from the compounds table fed as graph context.
CHEM_FEATURES = [
    "molecular_weight", "log_p", "tpsa",
    "num_rotatable_bonds", "num_h_acceptors", "num_h_donors",
    "num_atoms", "num_bonds",
]

# Smoke-test knob: set to None to use the full data.
SMOKE_TRAIN_N  = None
SMOKE_VAL_N    = None
SMOKE_EPOCHS   = 3

SUBMISSION_PATH = "gnn_predictions.parquet"
CHECKPOINT_PATH = "best_gnn.pt"


# ----- GRAPH CONSTRUCTION --------------------------------------------------

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
    ]
ATOM_DIM = 6


def smiles_to_graph(smiles, y=None):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        # placeholder so the loader doesn't crash; this compound will train
        # toward zero, which still beats NaN.
        data = Data(
            x=torch.zeros((1, ATOM_DIM), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
    else:
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],
                         dtype=torch.float)
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append((i, j)); edges.append((j, i))
        edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                      if edges else torch.zeros((2, 0), dtype=torch.long))
        data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float).unsqueeze(0)
    return data


class CompoundDataset(Dataset):
    """One graph per compound. Optionally carries a chemistry feature vector."""
    def __init__(self, compound_ids, smiles, z_targets=None, chem_feats=None):
        super().__init__(None, None, None)
        self.ids       = list(compound_ids)
        self.smiles    = list(smiles)
        self.z         = z_targets   # numpy (N, LATENT_DIM) or None for test
        self.chem      = chem_feats  # numpy (N, n_chem_features) or None

    def len(self):
        return len(self.ids)

    def get(self, idx):
        y = self.z[idx] if self.z is not None else None
        g = smiles_to_graph(self.smiles[idx], y=y)
        if self.chem is not None:
            g.chem = torch.tensor(self.chem[idx], dtype=torch.float).unsqueeze(0)
        return g


# ----- MODEL ---------------------------------------------------------------

class GNNLowRank(torch.nn.Module):
    def __init__(self, node_dim=ATOM_DIM, hidden=HIDDEN_DIM, latent=LATENT_DIM,
                 n_chem=len(CHEM_FEATURES), dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden)
        self.bn1   = BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2   = BatchNorm1d(hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn3   = BatchNorm1d(hidden)
        self.dropout = Dropout(dropout)

        # mean + max pool (2*hidden) + chemistry descriptor vector
        # chemistry is normalized before being concatenated
        self.chem_norm = BatchNorm1d(n_chem)
        in_dim = 2 * hidden + n_chem
        self.mlp = Sequential(
            Linear(in_dim, hidden * 2),
            BatchNorm1d(hidden * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden * 2, hidden),
            ReLU(),
            Linear(hidden, latent),
        )

    def forward(self, x, edge_index, batch, chem):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        mean_p = global_mean_pool(x, batch)
        max_p  = global_max_pool(x, batch)
        chem_n = self.chem_norm(chem.view(mean_p.size(0), -1))
        g = torch.cat([mean_p, max_p, chem_n], dim=1)
        g = self.dropout(g)
        return self.mlp(g)


# ----- TRAIN / EVAL --------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch_label=""):
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=epoch_label, leave=False, ncols=100)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        z_pred = model(batch.x, batch.edge_index, batch.batch, batch.chem)
        z_true = batch.y.view(z_pred.size())
        loss = F.mse_loss(z_pred, z_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        bsz = z_pred.size(0)
        total += loss.item() * bsz
        n += bsz
        pbar.set_postfix({"batch_mse": f"{loss.item():.4f}", "avg": f"{total/n:.4f}"})
    return total / n


@torch.no_grad()
def predict_latent(model, loader, device, desc="predict"):
    model.eval()
    out = []
    for batch in tqdm(loader, desc=desc, leave=False, ncols=100):
        batch = batch.to(device)
        z = model(batch.x, batch.edge_index, batch.batch, batch.chem)
        out.append(z.cpu().numpy())
    return np.vstack(out)


def _banner(label):
    bar = "=" * 70
    print(f"\n{bar}\n  {label}\n{bar}", flush=True)


def reconstruct_expression(z_pred, V_k, baseline):
    """latent (N x K) -> delta (N x G) -> expression (N x G), clamped >= 0."""
    delta = z_pred @ V_k.T
    expr  = delta + baseline.values[None, :]
    return np.clip(expr, 0.0, None).astype(np.float32)


# ----- MAIN ----------------------------------------------------------------

def main():
    t_total = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = SMOKE_TRAIN_N is not None

    _banner("STAGE 0  CONFIG")
    print(f"  device       : {device}", flush=True)
    print(f"  latent_dim   : {LATENT_DIM}", flush=True)
    print(f"  hidden_dim   : {HIDDEN_DIM}", flush=True)
    print(f"  batch_size   : {BATCH_SIZE}", flush=True)
    print(f"  epochs       : {SMOKE_EPOCHS if smoke else EPOCHS}", flush=True)
    print(f"  wmse_every   : {WMSE_EVAL_EVERY} epochs", flush=True)
    print(f"  smoke_mode   : {smoke}", flush=True)

    _banner("STAGE 1  LOAD DATA (pipeline.main)")
    t0 = time.time()
    splits = build_splits()
    print(f"  -> pipeline ready ({time.time()-t0:.1f}s)", flush=True)

    _banner("STAGE 2  LOAD MEJIA WEIGHTS")
    t0 = time.time()
    mejia_weights = load_weights_matrix()
    print(f"  weights matrix: {mejia_weights.shape} ({time.time()-t0:.1f}s)", flush=True)
    train_delta = splits["train_delta"].astype(np.float32)
    val_delta   = splits["val_delta"].astype(np.float32)
    baseline    = splits["baseline"]
    gene_filter = splits["gene_filter"]
    train_chem  = splits["train_chem"]
    val_chem    = splits["val_chem"]
    test_comp   = splits["test_compounds"]
    val_y_true  = splits["val_y"]

    if smoke:
        train_delta = train_delta.iloc[:SMOKE_TRAIN_N]
        val_delta   = val_delta.iloc[:SMOKE_VAL_N]
        train_chem  = train_chem.loc[train_delta.index]
        val_chem    = val_chem.loc[val_delta.index]
        val_y_true  = val_y_true.loc[val_delta.index]

    _banner("STAGE 3  SVD ON train_delta")
    print(f"  input shape  : {train_delta.shape}", flush=True)
    print(f"  k components : {LATENT_DIM}", flush=True)
    t0 = time.time()
    svd = TruncatedSVD(n_components=LATENT_DIM, random_state=SEED)
    Z_train = svd.fit_transform(train_delta.values)
    V_k     = svd.components_.T.astype(np.float32)
    var_explained = svd.explained_variance_ratio_.sum()
    Z_val = (val_delta.values @ V_k)
    Z_train = Z_train.astype(np.float32)
    Z_val   = Z_val.astype(np.float32)
    # SVD may produce fewer components than LATENT_DIM when n_train < LATENT_DIM
    actual_latent = Z_train.shape[1]
    print(f"  Z_train shape: {Z_train.shape}", flush=True)
    print(f"  Z_val shape  : {Z_val.shape}", flush=True)
    print(f"  actual latent: {actual_latent} (requested {LATENT_DIM})", flush=True)
    print(f"  variance kept: {var_explained:.1%}  ({time.time()-t0:.1f}s)", flush=True)

    _banner("STAGE 4  BUILD PYG DATASETS")
    t0 = time.time()

    # Extract and impute chemistry descriptor matrix (fill NaN with column median).
    def get_chem_matrix(chem_df):
        mat = chem_df[CHEM_FEATURES].values.astype(np.float32)
        col_medians = np.nanmedian(mat, axis=0)
        nan_mask = np.isnan(mat)
        mat[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return mat

    train_chem_mat = get_chem_matrix(train_chem)
    val_chem_mat   = get_chem_matrix(val_chem)

    train_ds = CompoundDataset(train_chem.index, train_chem["smiles"], Z_train, train_chem_mat)
    val_ds   = CompoundDataset(val_chem.index,   val_chem["smiles"],   Z_val,   val_chem_mat)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  train graphs : {len(train_ds):,}", flush=True)
    print(f"  val graphs   : {len(val_ds):,}", flush=True)
    print(f"  chem features: {len(CHEM_FEATURES)} {CHEM_FEATURES}", flush=True)
    print(f"  batches/epoch: {len(train_loader):,} @ batch_size={BATCH_SIZE}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    _banner("STAGE 5  BUILD MODEL")
    model = GNNLowRank(latent=actual_latent).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GNNLowRank: {n_params:,} parameters", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    _banner("STAGE 6  TRAIN")
    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    best_val_latent_mse = float("inf")
    best_val_wmse = float("inf")
    best_epoch = 0
    print(f"  epochs={epochs}  lr={LR}  wd={WEIGHT_DECAY}  clip={GRAD_CLIP}\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_mse = train_one_epoch(model, train_loader, optimizer, device,
                                    epoch_label=f"  epoch {epoch:3d}/{epochs} train")
        z_val_pred = predict_latent(model, val_loader, device,
                                    desc=f"  epoch {epoch:3d}/{epochs} val  ")
        val_latent_mse = float(((z_val_pred - Z_val) ** 2).mean())
        scheduler.step(val_latent_mse)
        elapsed = time.time() - t0

        line = (f"  epoch {epoch:3d}/{epochs} | "
                f"train_z_mse={train_mse:.4f}  "
                f"val_z_mse={val_latent_mse:.4f}  "
                f"({elapsed:.1f}s)")

        if epoch % WMSE_EVAL_EVERY == 0 or epoch == epochs:
            expr_pred = reconstruct_expression(z_val_pred, V_k, baseline)
            pred_wide = pd.DataFrame(expr_pred,
                                     index=val_delta.index,
                                     columns=gene_filter)
            per_comp = score_predictions(pred_wide, val_y_true, gene_filter,
                                         weights=mejia_weights)
            wmse_mean = per_comp["wmse"].mean() if "wmse" in per_comp else per_comp.iloc[:, -1].mean()
            line += f"  *** val_wMSE(Mejia)={wmse_mean:.4f} ***"
            if wmse_mean < best_val_wmse:
                best_val_wmse = wmse_mean
                best_epoch = epoch
        print(line, flush=True)

        if val_latent_mse < best_val_latent_mse:
            best_val_latent_mse = val_latent_mse
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    _banner("STAGE 7  FINAL EVAL + TEST INFERENCE")
    print(f"  best val_z_mse  : {best_val_latent_mse:.4f}", flush=True)
    print(f"  best val_wMSE   : {best_val_wmse:.4f}  (epoch {best_epoch})", flush=True)
    print(f"  loading best checkpoint from {CHECKPOINT_PATH}...", flush=True)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    # Test compounds also need chem features; fill missing with train median.
    test_chem_mat = get_chem_matrix(
        test_comp.reindex(columns=CHEM_FEATURES).fillna(
            dict(zip(CHEM_FEATURES, np.nanmedian(train_chem_mat, axis=0)))
        )
    )
    test_ds = CompoundDataset(test_comp["compound"], test_comp["smiles"],
                              chem_feats=test_chem_mat)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  test graphs: {len(test_ds):,}", flush=True)
    z_test = predict_latent(model, test_loader, device, desc="  test     ")
    expr_test = reconstruct_expression(z_test, V_k, baseline)
    pred_wide = pd.DataFrame(expr_test,
                             index=test_comp["compound"].astype(str),
                             columns=gene_filter)
    pred_wide.index.name = "compound"

    _banner("STAGE 8  WRITE SUBMISSION")
    write_submission(pred_wide, gene_filter, SUBMISSION_PATH)

    _banner("DONE")
    print(f"  total wall-clock: {time.time()-t_total:.1f}s "
          f"({(time.time()-t_total)/60:.1f} min)", flush=True)
    print(f"  best val_wMSE   : {best_val_wmse:.4f}", flush=True)
    print(f"  submission file : {SUBMISSION_PATH}", flush=True)


if __name__ == "__main__":
    main()
