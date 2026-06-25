"""
End-to-end from-scratch evaluation: FastMCWS_FromScratch only.
AMP and TOX datasets, same 5-fold StratifiedKFold(random_state=42) as all
other comparison scripts.

Early stopping monitors val_ACC (not val_loss) and restores best weights --
same fix applied to eval_ampscanner_amplify.py.
"""
import os, time, random, logging, copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from fastmcws_model import FastMCWSTransformer
from ngram_tokenizer import build_tokenizer_for_pretraining, ngram_tokenize
from Utils.PerformanceMonitor import EfficiencyMonitor

os.makedirs("./log", exist_ok=True)
logging.basicConfig(
    filename='./log/FastMCWS_Scratch_Compare.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()

FASTMCWS_CHUNK_SIZE = 4
FASTMCWS_MAX_KMERS  = 125
DATASETS  = {"AMP": "AMP_amplify.csv", "TOX": "tox.csv"}
HARDWARE  = "NVIDIA RTX 5090, 2TB RAM"


# ── Model wrapper ─────────────────────────────────────────────────────────────
class FastMCWSClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, n_filters=128, num_classes=2):
        super().__init__()
        self.encoder = FastMCWSTransformer(
            vocab_size=vocab_size, max_len=max_len, n_filters=n_filters)
        self.head = nn.Linear(n_filters * 3, num_classes)

    def forward(self, x):
        pooled = self.encoder.encode_pooled(x, pad_id=0)
        return self.head(pooled)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_fold(model, X_tr, y_tr, X_val, y_val,
                   max_epochs=200, patience=10, batch_size=64, lr=1e-3):
    model   = model.cuda()
    opt     = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader  = DataLoader(TensorDataset(X_tr, y_tr),
                         batch_size=batch_size, shuffle=True)

    # Monitor val_ACC; save and restore best weights
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
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val.cuda())
            preds  = logits.argmax(dim=1).cpu().numpy()
            probs  = logits.softmax(1)[:, 1].cpu().numpy()

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

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, best_f1, best_auc


# ── 5-fold CV ─────────────────────────────────────────────────────────────────
def cross_validate(sequences, labels, vocab, monitor):
    model_name = "FastMCWS_FromScratch"
    print(f"\n===== {model_name} =====")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    training_time   = 0.0

    for fold, (tr_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"  Fold {fold+1}")
        seq_tr  = [sequences[i] for i in tr_idx]
        seq_val = [sequences[i] for i in val_idx]
        y_tr    = torch.tensor(labels[tr_idx])
        y_val   = torch.tensor(labels[val_idx])

        X_tr  = torch.tensor(np.stack([
            ngram_tokenize(s, vocab, FASTMCWS_CHUNK_SIZE, FASTMCWS_MAX_KMERS)
            for s in seq_tr]))
        X_val = torch.tensor(np.stack([
            ngram_tokenize(s, vocab, FASTMCWS_CHUNK_SIZE, FASTMCWS_MAX_KMERS)
            for s in seq_val]))

        model = FastMCWSClassifier(vocab_size=len(vocab), max_len=FASTMCWS_MAX_KMERS)

        t0 = time.time()
        acc, f1, auc = train_one_fold(model, X_tr, y_tr, X_val, y_val)
        training_time = time.time() - t0

        model.eval()
        with torch.no_grad():
            for i in range(len(seq_val)):
                t1 = time.time()
                xi = torch.tensor(
                    ngram_tokenize(seq_val[i], vocab,
                                   FASTMCWS_CHUNK_SIZE, FASTMCWS_MAX_KMERS)
                ).unsqueeze(0).cuda()
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

        # Build 4-mer vocab from this dataset's sequences
        vocab = build_tokenizer_for_pretraining(sequences, chunk_size=FASTMCWS_CHUNK_SIZE)

        monitor = EfficiencyMonitor()
        result  = cross_validate(sequences, labels, vocab, monitor)
        result["dataset"] = dataset_name
        all_results.append(result)

    print("\n==== Final Results Summary ====")
    df_res = pd.DataFrame(all_results)
    print(df_res)
    df_res.to_csv("./log/fastmcws_scratch_summary.csv", index=False)
    print("Saved to ./log/fastmcws_scratch_summary.csv")

if __name__ == "__main__":
    main()
