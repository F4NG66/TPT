"""
Standalone 5-fold CV: AMP Scanner v2 and AMPlify on AMP and TOX.

Both trained end-to-end from scratch (no pretraining).

Early stopping monitors val_ACC (not val_loss) and restores best weights,
matching the original papers' stated criterion:
  - AMP Scanner v2: "model weights from the epoch with the best validation accuracy"
    (implied by their cross-validation description)
  - AMPlify: "model weights from the epoch with the best validation accuracy"
    (explicitly stated in paper)
max_epochs=200, patience=10 for both.
"""
import os, time, random, logging, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from Utils.PerformanceMonitor import EfficiencyMonitor

os.makedirs("./log", exist_ok=True)
logging.basicConfig(
    filename='./log/AMPScanner_AMPlify_Compare.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()

MAX_LEN  = 200
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
aa2int   = {aa: i+1 for i, aa in enumerate(AA_ORDER)}  # 1-20; 0=pad/unknown

DATASETS = {"AMP": "AMP_amplify.csv", "TOX": "tox.csv"}
HARDWARE = "NVIDIA RTX 5090, 2TB RAM"


# ── Encoding ─────────────────────────────────────────────────────────────────
def encode_int(seq, max_len=MAX_LEN):
    ids = [aa2int.get(aa.upper(), 0) for aa in seq][:max_len]
    return ids + [0] * (max_len - len(ids))

def encode_onehot(seq, max_len=MAX_LEN):
    aa2idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    mat = np.zeros((max_len, 20), dtype=np.float32)
    for t, aa in enumerate(seq.upper()[:max_len]):
        if aa in aa2idx:
            mat[t, aa2idx[aa]] = 1.0
    return mat


# ── AMP Scanner v2 (Veltri et al., Bioinformatics 2018) ──────────────────────
# Embedding(21→128) → Conv1d(64,k=16,relu) → MaxPool(5)
# → Dropout(0.1) → LSTM(100) → Dense(1,sigmoid)
class AMPScannerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(21, 128, padding_idx=0)
        self.conv  = nn.Conv1d(128, 64, 16, padding=8)
        self.pool  = nn.MaxPool1d(5)
        self.drop  = nn.Dropout(0.1)     # LSTM input dropout (paper: dropout=0.1)
        self.lstm  = nn.LSTM(64, 100, batch_first=True)
        self.out   = nn.Linear(100, 1)

    def forward(self, x):                          # x: (B, L) int
        h = F.relu(self.conv(self.embed(x).permute(0, 2, 1)))
        h = self.pool(h).permute(0, 2, 1)
        _, (hn, _) = self.lstm(self.drop(h))
        return self.out(hn.squeeze(0)).squeeze(-1) # (B,) raw logit


# ── AMPlify (Li et al., BMC Genomics 2022) ───────────────────────────────────
# one-hot(20) → Dropout(0.5) → BiLSTM(512×2=1024)
# → MultiHeadAttn(32 heads) → Dropout(0.2) → ContextAttn → Dense(1,sigmoid)
class ContextAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.u = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask=None):               # mask: (B,L) True=padding
        score = self.u(torch.tanh(self.W(h)))      # (B, L, 1)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(-1), float('-inf'))
        alpha = torch.softmax(score, dim=1)
        return (alpha * h).sum(dim=1)              # (B, D)

class AMPlify(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_drop = nn.Dropout(0.5)             # dropout on BiLSTM input
        self.bilstm  = nn.LSTM(20, 512, batch_first=True, bidirectional=True)
        self.mha     = nn.MultiheadAttention(1024, 32, batch_first=True)
        self.drop    = nn.Dropout(0.2)
        self.ctx     = ContextAttention(1024)
        self.out     = nn.Linear(1024, 1)

    def forward(self, x, pad_mask=None):           # x: (B, L, 20) float
        h, _ = self.bilstm(self.in_drop(x))
        h, _ = self.mha(h, h, h, key_padding_mask=pad_mask)
        h    = self.drop(h)
        s    = self.ctx(h, mask=pad_mask)
        return self.out(s).squeeze(-1)             # (B,) raw logit


# ── Shared training loop ──────────────────────────────────────────────────────
def train_one_fold(model, X_tr, y_tr, X_val, y_val,
                   max_epochs=200, patience=10, batch_size=32, lr=1e-3,
                   is_amplify=False):
    model = model.cuda()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader  = DataLoader(TensorDataset(X_tr, y_tr.float()),
                         batch_size=batch_size, shuffle=True)

    # Monitor val_ACC and restore best weights -- matches original papers'
    # "model weights from the epoch with the best validation accuracy"
    best_acc   = -1.0
    best_f1    = 0.0
    best_auc   = 0.0
    best_state = None
    bad        = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            if is_amplify:
                pad  = (xb.sum(-1) == 0)
                loss = loss_fn(model(xb, pad_mask=pad), yb)
            else:
                loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            if is_amplify:
                pad    = (X_val.cuda().sum(-1) == 0)
                logits = model(X_val.cuda(), pad_mask=pad)
            else:
                logits = model(X_val.cuda())
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

        yn  = y_val.numpy()
        acc = accuracy_score(yn, preds)
        f1  = f1_score(yn, preds, zero_division=0)
        auc = roc_auc_score(yn, probs)

        if acc > best_acc + 1e-4:
            best_acc   = acc
            best_f1    = f1
            best_auc   = auc
            best_state = copy.deepcopy(model.state_dict())
            bad        = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # Restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, best_f1, best_auc


# ── 5-fold CV ─────────────────────────────────────────────────────────────────
def cross_validate(model_name, sequences, labels, monitor):
    print(f"\n===== {model_name} =====")
    is_amplify = (model_name == "AMPlify")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    training_time = 0.0

    for fold, (tr_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"  Fold {fold+1}")
        seq_tr  = [sequences[i] for i in tr_idx]
        seq_val = [sequences[i] for i in val_idx]
        y_tr    = torch.tensor(labels[tr_idx])
        y_val   = torch.tensor(labels[val_idx])

        if is_amplify:
            X_tr  = torch.tensor(np.stack([encode_onehot(s) for s in seq_tr]))
            X_val = torch.tensor(np.stack([encode_onehot(s) for s in seq_val]))
            model = AMPlify()
        else:
            X_tr  = torch.tensor(np.stack([encode_int(s) for s in seq_tr]))
            X_val = torch.tensor(np.stack([encode_int(s) for s in seq_val]))
            model = AMPScannerV2()

        t0 = time.time()
        acc, f1, auc = train_one_fold(
            model, X_tr, y_tr, X_val, y_val,
            max_epochs=200, patience=10, batch_size=32, lr=1e-3,
            is_amplify=is_amplify
        )
        training_time = time.time() - t0

        # Per-sample inference timing
        model.eval()
        with torch.no_grad():
            for i in range(len(seq_val)):
                t1 = time.time()
                if is_amplify:
                    xi  = torch.tensor(encode_onehot(seq_val[i])).unsqueeze(0).cuda()
                    pad = (xi.sum(-1) == 0)
                    model(xi, pad_mask=pad)
                else:
                    xi = torch.tensor(encode_int(seq_val[i])).unsqueeze(0).cuda()
                    model(xi)
                monitor.inference_times.append(time.time() - t1)

        accs.append(acc); f1s.append(f1); aucs.append(auc)
        del model; torch.cuda.empty_cache()
        print(f"    acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")

    am, as_ = np.mean(accs), np.std(accs)
    fm, fs  = np.mean(f1s),  np.std(f1s)
    um, us  = np.mean(aucs), np.std(aucs)

    report = monitor.get_efficiency_report(training_time, HARDWARE)
    print(report)
    logging.info(f"\n{model_name} efficiency:\n{report}")

    line = (f"{model_name}: ACC={am:.4f}±{as_:.4f}  "
            f"F1={fm:.4f}±{fs:.4f}  AUC={um:.4f}±{us:.4f}")
    print(f"\n{line}")
    logging.info(line)
    logging.info(f"  All ACC: {accs}")
    logging.info(f"  All F1:  {f1s}")
    logging.info(f"  All AUC: {aucs}")

    return {"model": model_name, "acc": am, "f1": fm, "auc": um}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_results = []

    for dataset_name, csv_path in DATASETS.items():
        print(f"\n{'#'*10} Dataset: {dataset_name} {'#'*10}")
        df        = pd.read_csv(csv_path)
        labels    = df["label"].values
        sequences = df["Sequence"].tolist()

        for model_name in ["AMPScannerV2", "AMPlify"]:
            monitor = EfficiencyMonitor()
            result  = cross_validate(model_name, sequences, labels, monitor)
            result["dataset"] = dataset_name
            all_results.append(result)

    print("\n==== Final Results Summary ====")
    df_res = pd.DataFrame(all_results)
    print(df_res)
    df_res.to_csv("./log/AMPScanner_AMPlify_summary.csv", index=False)
    print("Saved to ./log/AMPScanner_AMPlify_summary.csv")

if __name__ == "__main__":
    main()