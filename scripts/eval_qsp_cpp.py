"""
New task evaluation: QSP and CPP datasets.

Part 1 (matches doc78 ablation script):
  TPT, TPT_NoCL, TPT_Gated, TPT_MeanPool, TPT_NoCNN
  + ESM2-35M, ESM2-150M, ProtBERT

Part 2 (matches doc79 hidden-dim script):
  Hidden dim 320 / 480 / 640 / 800

Both parts use the same cross-validation protocol as the original scripts:
  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  train_classifier: AdamW lr=1e-3, batch=64, 5 epochs
  Metrics: ACC, F1, AUC

Datasets:
  QSP.csv  (440 seqs, balanced 220/220)
  CPP.csv  (5479 seqs, 1399 pos / 4080 neg)
"""

import os, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from Utils.PerformanceMonitor import EfficiencyMonitor

# Ablation model imports -- same as doc78
from Ablation.gate_tpt import GateTinyProteinTransformer
from Ablation.tpt import TinyProteinTransformer as TinyProteinTransformerLoss
from Ablation.tpt_without_attenpool import TinyProteinTransformerWithoutAttentionPooling
from Ablation.tpt_withoutCNN import TinyProteinTransformerWithoutCNN

import logging
os.makedirs("./log", exist_ok=True)
logging.basicConfig(
    filename='./log/NewTasks_QSP_CPP.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ─── Seeds ────────────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ─── Tokenizer (identical to both original scripts) ───────────────────────────
AA = "ACDEFGHIKLMNPQRSTVWYBXZ"

def build_tokenizer():
    tok = {aa: i+1 for i, aa in enumerate(AA)}
    tok["PAD"] = 0
    tok["MASK"] = len(tok)
    return tok

tokenizer_tiny = build_tokenizer()

def tokenize_seq_tiny(seq, max_len=128):
    ids = [tokenizer_tiny.get(a, tokenizer_tiny["X"]) for a in seq]
    ids = ids[:max_len]
    return ids + [tokenizer_tiny["PAD"]] * (max_len - len(ids))


# ─── TinyProteinTransformer (identical to both original scripts) ───────────────
class TinyProteinTransformer(nn.Module):
    def __init__(self, vocab_size, num_classes=None, hidden_dim=640,
                 num_layers=20, num_heads=10, max_len=128,
                 cnn_kernel_sizes=(3,5,7,9), cnn_out=256, dropout=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos   = nn.Embedding(max_len, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, cnn_out, k, padding=k//2)
            for k in cnn_kernel_sizes
        ])
        self.conv_proj = nn.Linear(cnn_out * len(cnn_kernel_sizes), hidden_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4, batch_first=True, norm_first=True
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.mlm_head  = nn.Linear(hidden_dim, vocab_size)
        self.cls_head  = nn.Linear(hidden_dim, num_classes) if num_classes else None
        self.dropout   = nn.Dropout(dropout)
        self.ln        = nn.LayerNorm(hidden_dim)

    def encode(self, x):
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) + self.pos(pos_ids)
        h = self.dropout(h)
        cnn_in  = h.permute(0, 2, 1)
        convs   = [F.relu(conv(cnn_in)) for conv in self.convs]
        conv_cat = torch.cat(convs, dim=1).permute(0, 2, 1)
        conv_out = self.conv_proj(conv_cat)
        h = self.ln(h + conv_out)
        return self.encoder(h)

    def attention_pool(self, h):
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return (attn * h).sum(dim=1)

    def forward(self, mlm_input=None, aug1=None, aug2=None):
        out = {}
        if mlm_input is not None:
            out["mlm_logits"] = self.mlm_head(self.encode(mlm_input))
        if aug1 is not None and aug2 is not None:
            out["h1"] = self.attention_pool(self.encode(aug1))
            out["h2"] = self.attention_pool(self.encode(aug2))
        return out


# ─── train_classifier: identical to doc78 ─────────────────────────────────────
def train_classifier(X_train, y_train, X_val, y_val,
                     epochs=5, lr=1e-3, batch_size=64):
    clf      = nn.Linear(X_train.size(1), 2).cuda()
    opt      = torch.optim.AdamW(clf.parameters(), lr=lr)
    loss_fn  = nn.CrossEntropyLoss()
    train_ds = TensorDataset(X_train, y_train)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_acc = best_f1 = best_auc = -1

    for ep in range(epochs):
        clf.train()
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            opt.zero_grad()
            loss_fn(clf(xb), yb).backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            logits = clf(X_val.cuda())
            preds  = logits.argmax(dim=1).cpu().numpy()
            probs  = logits.softmax(1)[:, 1].cpu().numpy()

        acc = accuracy_score(y_val.cpu().numpy(), preds)
        f1  = f1_score(y_val.cpu().numpy(), preds)
        auc = roc_auc_score(y_val.cpu().numpy(), probs)
        print(f"Epoch {ep}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

        if acc > best_acc:
            best_acc, best_f1, best_auc = acc, f1, auc

    return best_acc, best_f1, best_auc


# ─── cross_validate_model: identical to doc78 (with EfficiencyMonitor) ────────
def cross_validate_model(encode_fn, model_name, sequences, labels):
    print(f"\n===== Running 5-Fold CV for {model_name} =====")
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    hardware_info   = "NVIDIA RTX 5090, 2TB RAM"
    monitor         = EfficiencyMonitor()

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"Fold {fold+1}")
        seq_train = [sequences[i] for i in train_idx]
        seq_val   = [sequences[i] for i in val_idx]
        y_train   = torch.tensor(labels[train_idx]).cuda()
        y_val     = torch.tensor(labels[val_idx]).cuda()

        X_train = torch.stack([encode_fn(seq=s, monitor=monitor) for s in tqdm(seq_train)])
        X_val   = torch.stack([encode_fn(seq=s, monitor=monitor) for s in tqdm(seq_val)])

        Start = time.time()
        acc, f1, auc = train_classifier(X_train, y_train, X_val, y_val)
        training_time = time.time() - Start

        accs.append(acc); f1s.append(f1); aucs.append(auc)

    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean,  f1_std  = np.mean(f1s),  np.std(f1s)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)

    report = monitor.get_efficiency_report(training_time, hardware_info)
    logging.info(f"\n{model_name}  efficiency_report:\n{report}")
    print(report)

    print(f"\n{model_name} Results:")
    print(f"ACC: {accs}")
    print(f"F1 : {f1s}")
    print(f"AUC: {aucs}")
    logging.info(f"\n{model_name} Results:")
    logging.info(f"ACC: {accs}")
    logging.info(f"F1 : {f1s}")
    logging.info(f"AUC: {aucs}")

    print(f"Avg Results:")
    print(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1 : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    logging.info(f"Avg Results:")
    logging.info(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    logging.info(f"F1 : {f1_mean:.4f} ± {f1_std:.4f}")
    logging.info(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    return {"model": model_name, "acc": acc_mean, "f1": f1_mean, "auc": auc_mean}


# ─── cross_validate_model_hd: identical to doc79 (no monitor, for hidden dim) ─
def cross_validate_model_hd(encode_fn, model_name, sequences, labels):
    print(f"\n===== Running 5-Fold CV for {model_name} =====")
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"Fold {fold+1}")
        seq_train = [sequences[i] for i in train_idx]
        seq_val   = [sequences[i] for i in val_idx]
        y_train   = torch.tensor(labels[train_idx]).cuda()
        y_val     = torch.tensor(labels[val_idx]).cuda()

        X_train = torch.stack([encode_fn(s) for s in tqdm(seq_train)])
        X_val   = torch.stack([encode_fn(s) for s in tqdm(seq_val)])

        acc, f1, auc = train_classifier(X_train, y_train, X_val, y_val)
        accs.append(acc); f1s.append(f1); aucs.append(auc)

    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean,  f1_std  = np.mean(f1s),  np.std(f1s)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)

    print(f"\n{model_name} Results:")
    print(f"ACC: {accs}")
    print(f"F1 : {f1s}")
    print(f"AUC: {aucs}")
    logging.info(f"\n{model_name} Results:")
    logging.info(f"ACC: {accs}")
    logging.info(f"F1 : {f1s}")
    logging.info(f"AUC: {aucs}")

    print(f"Avg Results:")
    print(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1 : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    logging.info(f"Avg Results:")
    logging.info(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    logging.info(f"F1 : {f1_mean:.4f} ± {f1_std:.4f}")
    logging.info(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    return {"model": model_name, "acc": acc_mean, "f1": f1_mean, "auc": auc_mean}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: loop over QSP and CPP
# ══════════════════════════════════════════════════════════════════════════════
DataSetList = {
    "QSP": "QSP.csv",
    "CPP": "CPP.csv",
}

for datasetusing, csv_path in DataSetList.items():
    print(f"\n{'#'*60}")
    print(f"#  Dataset: {datasetusing}")
    print(f"{'#'*60}")

    df        = pd.read_csv(csv_path)
    labels    = df["label"].values
    sequences = df["Sequence"].tolist()
    print(df.head())

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1 — Ablation + ESM + ProtBERT  (mirrors doc78 exactly)
    # ──────────────────────────────────────────────────────────────────────────

    # ── Load ablation models ──────────────────────────────────────────────────
    tiny_model_Loss  = TinyProteinTransformerLoss(vocab_size=len(tokenizer_tiny), num_classes=None)
    tiny_model_Gate  = GateTinyProteinTransformer(vocab_size=len(tokenizer_tiny))
    tiny_model_WOAtt = TinyProteinTransformerWithoutAttentionPooling(vocab_size=len(tokenizer_tiny))
    tiny_model_WOCNN = TinyProteinTransformerWithoutCNN(vocab_size=len(tokenizer_tiny))
    tiny_model       = TinyProteinTransformer(vocab_size=len(tokenizer_tiny), num_classes=None)

    tiny_model_Loss.load_state_dict( torch.load("./Ablation/TPT_Contrast_Weight/best_pretrain_10M.pt"))
    tiny_model_Gate.load_state_dict( torch.load("./Ablation/gate_TPT_Weight/best_pretrain.pt"))
    tiny_model_WOAtt.load_state_dict(torch.load("./Ablation/TPT_Without_Attenpool_weight/best_pretrain_10M.pt"))
    tiny_model_WOCNN.load_state_dict(torch.load("./Ablation/TPT_Without_CNN_Weight/best_pretrain_10M.pt"))
    tiny_model.load_state_dict(      torch.load("./Ablation/TPT_Weight/best_pretrain.pt"))

    for m in [tiny_model, tiny_model_Loss, tiny_model_Gate, tiny_model_WOAtt, tiny_model_WOCNN]:
        m.cuda().eval()
        for p in m.parameters():
            p.requires_grad = False

    # ── Load ESM2 models ──────────────────────────────────────────────────────
    import esm, argparse
    torch.serialization.add_safe_globals([argparse.Namespace])

    esm_models = {
        "esm2_t12_35M":  "./PreTrain_model/ESM35/esm2_t12_35m/esm2_t12_35M_UR50D.pt",
        "esm2_t30_150M": "./PreTrain_model/ESM150/esm2_t30_150m/esm2_t30_150M_UR50D.pt",
    }
    esm_encoders = {}
    for name, tag in esm_models.items():
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(tag)
        model = model.cuda().eval()
        for p in model.parameters(): p.requires_grad = False
        esm_encoders[name] = (model, alphabet)

    # ── Load ProtBERT ─────────────────────────────────────────────────────────
    from transformers import BertTokenizer, BertModel
    protbert_tokenizer = BertTokenizer.from_pretrained("./PreTrain_model/Probert", do_lower_case=False)
    protbert_model     = BertModel.from_pretrained("./PreTrain_model/Probert")
    protbert_model.cuda().eval()
    for p in protbert_model.parameters(): p.requires_grad = False

    # ── Encode functions (identical to doc78) ─────────────────────────────────
    @torch.no_grad()
    def encode_tiny(seq, monitor):
        t = time.time()
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model.encode(x)
        pooled = tiny_model.attention_pool(h)
        monitor.inference_times.append(time.time() - t)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_Loss(seq, monitor):
        t = time.time()
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model_Loss.encode(x)
        pooled = tiny_model_Loss.attention_pool(h)
        monitor.inference_times.append(time.time() - t)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_Gate(seq, monitor):
        t = time.time()
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model_Gate.encode(x)
        pooled = tiny_model_Gate.attention_pool(h)
        monitor.inference_times.append(time.time() - t)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_WOAtt(seq, monitor):
        t = time.time()
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model_WOAtt.encode(x)
        pooled = h.mean(1)
        monitor.inference_times.append(time.time() - t)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_WOCNN(seq, monitor):
        t = time.time()
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model_WOCNN.encode(x)
        pooled = tiny_model_WOCNN.attention_pool(h)
        monitor.inference_times.append(time.time() - t)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_esm(model, alphabet, seq, monitor):
        t = time.time()
        data = alphabet.get_batch_converter()([(0, seq)])
        _, _, toks = data
        toks = toks.cuda()
        out  = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
        monitor.inference_times.append(time.time() - t)
        return out.mean(1).squeeze(0)

    @torch.no_grad()
    def encode_protbert(seq, monitor):
        t          = time.time()
        seq_spaced = " ".join(list(seq))
        toks       = protbert_tokenizer(seq_spaced, return_tensors="pt", padding=True)
        for k in toks: toks[k] = toks[k].cuda()
        out = protbert_model(**toks).last_hidden_state
        monitor.inference_times.append(time.time() - t)
        return out.mean(1).squeeze(0)

    # ── Run Part 1 ────────────────────────────────────────────────────────────
    results_ablation = []
    logging.info(f"\n{'='*40}")
    logging.info(f"PART 1 — Ablation + Baselines | Dataset: {datasetusing}")
    logging.info(f"{'='*40}")

    results_ablation.append(cross_validate_model(encode_tiny,            "TinyProteinTransformer", sequences, labels))
    results_ablation.append(cross_validate_model(encode_tiny_model_Loss, "tiny_model_Loss",        sequences, labels))
    results_ablation.append(cross_validate_model(encode_tiny_model_Gate, "tiny_model_Gate",        sequences, labels))
    results_ablation.append(cross_validate_model(encode_tiny_model_WOAtt,"tiny_model_WOAtt",       sequences, labels))
    results_ablation.append(cross_validate_model(encode_tiny_model_WOCNN,"tiny_model_WOCNN",       sequences, labels))

    for name, (m, alphabet) in esm_encoders.items():
        enc_fn = lambda seq, monitor, m=m, a=alphabet: encode_esm(m, a, seq, monitor)
        results_ablation.append(cross_validate_model(enc_fn, name, sequences, labels))

    results_ablation.append(cross_validate_model(encode_protbert, "ProtBERT", sequences, labels))

    print(f"\n==== Ablation Results Summary | {datasetusing} ====")
    logging.info(f"==== Ablation Results Summary {datasetusing} ====")
    logging.info(f"{results_ablation}")
    df_abl = pd.DataFrame(results_ablation)
    print(df_abl)
    df_abl.to_csv(f"./log/ablation_{datasetusing}_summary.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2 — Hidden Dim Ablation  (mirrors doc79 exactly)
    # ──────────────────────────────────────────────────────────────────────────
    logging.info(f"\n{'='*40}")
    logging.info(f"PART 2 — Hidden Dim | Dataset: {datasetusing}")
    logging.info(f"{'='*40}")

    tiny_model_320 = TinyProteinTransformer(vocab_size=len(tokenizer_tiny), num_classes=None, hidden_dim=320)
    tiny_model_480 = TinyProteinTransformer(vocab_size=len(tokenizer_tiny), num_classes=None, hidden_dim=480)
    tiny_model_640 = TinyProteinTransformer(vocab_size=len(tokenizer_tiny), num_classes=None, hidden_dim=640)
    tiny_model_800 = TinyProteinTransformer(vocab_size=len(tokenizer_tiny), num_classes=None, hidden_dim=800)

    tiny_model_320.load_state_dict(torch.load("./Ablation/Hidden_dim/320/best_pretrain.pt"))
    tiny_model_480.load_state_dict(torch.load("./Ablation/Hidden_dim/480/best_pretrain.pt"))
    tiny_model_640.load_state_dict(torch.load("./Ablation/Hidden_dim/640/best_pretrain.pt"))
    tiny_model_800.load_state_dict(torch.load("./Ablation/Hidden_dim/800/best_pretrain.pt"))

    for m in [tiny_model_320, tiny_model_480, tiny_model_640, tiny_model_800]:
        m.cuda().eval()
        for p in m.parameters(): p.requires_grad = False

    @torch.no_grad()
    def encode_tiny_320(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        return tiny_model_320.attention_pool(tiny_model_320.encode(x)).squeeze(0)

    @torch.no_grad()
    def encode_tiny_480(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        return tiny_model_480.attention_pool(tiny_model_480.encode(x)).squeeze(0)

    @torch.no_grad()
    def encode_tiny_640(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        return tiny_model_640.attention_pool(tiny_model_640.encode(x)).squeeze(0)

    @torch.no_grad()
    def encode_tiny_800(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        return tiny_model_800.attention_pool(tiny_model_800.encode(x)).squeeze(0)

    results_hd = []
    results_hd.append(cross_validate_model_hd(encode_tiny_320, f"HiddenDim320DataSet{datasetusing}", sequences, labels))
    results_hd.append(cross_validate_model_hd(encode_tiny_480, f"HiddenDim480DataSet{datasetusing}", sequences, labels))
    results_hd.append(cross_validate_model_hd(encode_tiny_640, f"HiddenDim640DataSet{datasetusing}", sequences, labels))
    results_hd.append(cross_validate_model_hd(encode_tiny_800, f"HiddenDim800DataSet{datasetusing}", sequences, labels))

    print(f"\n==== Hidden Dim Results Summary | {datasetusing} ====")
    logging.info(f"==== Hidden Dim Results Summary {datasetusing} ====")
    logging.info(f"{results_hd}")
    df_hd = pd.DataFrame(results_hd)
    print(df_hd)
    df_hd.to_csv(f"./log/hiddendim_{datasetusing}_summary.csv", index=False)

    # free GPU memory before next dataset
    del (tiny_model, tiny_model_Loss, tiny_model_Gate, tiny_model_WOAtt, tiny_model_WOCNN,
         tiny_model_320, tiny_model_480, tiny_model_640, tiny_model_800,
         protbert_model)
    for _, (m, _) in esm_encoders.items():
        del m
    torch.cuda.empty_cache()

print("\nAll done. Logs saved to ./log/NewTasks_QSP_CPP.log")
