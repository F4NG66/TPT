
###############################################################
#                        AMP Task
###############################################################


TPT_Origin_640

Hidden_dim_320

Hidden_dim_480

Hidden_dim_800






###############################################################
#                        TOXIC Task
###############################################################


TPT_Origin_640

Hidden_dim_320

Hidden_dim_480

Hidden_dim_800

"""
import logging
logging.basicConfig(filename='./log/HiddenDim.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')







##### import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset
import random


class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len=128, mode="pretrain"):
        self.seqs = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

#    def mlm(self, seq_ids):
 #       seq = seq_ids[:]
  #      labels = [-100] * len(seq)

   #     for i in range(len(seq)):
    #        if random.random() < 0.15:
     #           labels[i] = seq[i]
      #          seq[i] = self.tokenizer["MASK"]
#
 #       return seq, labels
    
    def mlm(self, seq_ids, mask_prob=0.15, span_prob=0.3, span_len_range=(2,5)):
        seq = seq_ids[:]
        labels = [-100] * len(seq)
        L = len(seq)

    # decide if we do span masking or per-token masking
        if random.random() < span_prob:
            # ---- span masking ----
            num_to_mask = int(L * mask_prob)
        
            while num_to_mask > 0:
                span_len = random.randint(*span_len_range)
                start = random.randint(0, max(0, L - span_len))
                for i in range(start, min(start + span_len, L)):
                    labels[i] = seq[i]
                    seq[i] = self.tokenizer["MASK"]
                num_to_mask -= span_len

        else:
            # ---- per-token masking ---- (your original)
            for i in range(L):
                if random.random() < mask_prob:
                    labels[i] = seq[i]
                    seq[i] = self.tokenizer["MASK"]

        return seq, labels

    def __getitem__(self, idx):
        seq = self.seqs[idx]

        # token IDs
        ids = tokenize_seq(seq, self.tokenizer, self.max_len)

        if self.mode == "pretrain":
            # MLM
            mlm_in, mlm_target = self.mlm(ids)

            # Contrastive augs
            aug1 = tokenize_seq(augment_sequence(seq), self.tokenizer, self.max_len)
            aug2 = tokenize_seq(augment_sequence(seq), self.tokenizer, self.max_len)

            return {
                "mlm_input": torch.tensor(mlm_in),
                "mlm_target": torch.tensor(mlm_target),
                "aug1": torch.tensor(aug1),
                "aug2": torch.tensor(aug2),
            }

        return {"input": torch.tensor(ids)}

    def __len__(self):
        return len(self.seqs)

def load_fasta(path):
    seqs = []
    with open(path) as f:
        cur = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur != "":
                    seqs.append(cur)
                    cur = ""
            else:
                cur += line
        if cur != "":
            seqs.append(cur)
    return seqs


AA = "ACDEFGHIKLMNPQRSTVWYBXZ"  # with X/B/Z
def build_tokenizer():
    tok = {aa: i+1 for i, aa in enumerate(AA)}
    tok["PAD"] = 0
    tok["MASK"] = len(tok)
    return tok


def tokenize_seq(seq, tokenizer, max_len):
    ids = [tokenizer.get(a, tokenizer["X"]) for a in seq]
    ids = ids[:max_len]
    return ids + [tokenizer["PAD"]] * (max_len - len(ids))


# protein augment
def augment_sequence(seq):
    import random

    seq = list(seq)

    # random delete
    if random.random() < 0.3:
        cut = random.randint(2, 5)
        st = random.randint(0, max(1, len(seq)-cut))
        del seq[st:st+cut]

    # similar AA replace
    aa_group = {
        "A":"AGS", "C":"C", "D":"DEN", "E":"EDNQ", "F":"FWY",
        "G":"GAS","H":"H", "I":"IVL","K":"KRH","L":"LIV",
        "M":"M","N":"NQDE","P":"P","Q":"QNE",
        "R":"RKH","S":"STAG","T":"TSAG","V":"VIL",
        "W":"WFY","Y":"YWF"
    }

    if random.random() < 0.15:
        i = random.randint(0,len(seq)-1)
        if seq[i] in aa_group:
            seq[i] = random.choice(aa_group[seq[i]])

    return "".join(seq)

class TinyProteinTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=None,
        hidden_dim=640,
        num_layers=20,
        num_heads=10,
        max_len=128,
        cnn_kernel_sizes=(3,5,7,9),
        cnn_out=256,
        dropout=0.05
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Embedding(max_len, hidden_dim)

        # CNN Motif
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, cnn_out, k, padding=k//2)
            for k in cnn_kernel_sizes
        ])
        self.conv_proj = nn.Linear(cnn_out * len(cnn_kernel_sizes), hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention Pooling
        self.attn_pool = nn.Linear(hidden_dim, 1)

        # Heads
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)
        self.cls_head = nn.Linear(hidden_dim, num_classes) if num_classes else None

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    def encode(self, x):
        B, L = x.size()
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        h = self.embed(x) + self.pos(pos_ids)
        h = self.dropout(h)

        cnn_in = h.permute(0, 2, 1)
        convs = [F.relu(conv(cnn_in)) for conv in self.convs]
        conv_cat = torch.cat(convs, dim=1).permute(0, 2, 1)
        conv_out = self.conv_proj(conv_cat)

        h = self.ln(h + conv_out)
        return self.encoder(h)

    def forward_mlm(self, x):
        return self.mlm_head(self.encode(x))

    def attention_pool(self, h):
        attn = torch.softmax(self.attn_pool(h), dim=1)
        return (attn * h).sum(dim=1)

    def forward_contrast(self, x):
        return self.attention_pool(self.encode(x))

    # Unified forward for DataParallel
    def forward(self, mlm_input=None, aug1=None, aug2=None):
        out = {}
        if mlm_input is not None:
            out["mlm_logits"] = self.forward_mlm(mlm_input)
        if aug1 is not None and aug2 is not None:
            h1 = self.forward_contrast(aug1)
            h2 = self.forward_contrast(aug2)
            out["h1"] = h1
            out["h2"] = h2
        return out





##### import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
#from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -----------------------
# Fix seeds
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

###############################################################
#                         Change Data Set
##############################################################
DataSetList = ["AMP","TOX","CRISP","MOES"]
# DataSetList = ["MOES"]
# datasetusing = "CRISP"
for  datasetusing in  DataSetList:
    if datasetusing == "AMP":
        df = pd.read_csv("AMP_amp.csv")  
    if datasetusing == "TOX":
        df = pd.read_csv("tox.csv")      
    if datasetusing == "CRISP":
        df = pd.read_csv("Acr.csv") 
    if datasetusing == "MOES":
        df = pd.read_csv("BCN.csv")

    print(df.head())

    labels = df["label"].values
    sequences = df["Sequence"].tolist()

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

    #######################################################
    # Modify loading Model
    #######################################################
    
    # for hidden_dim in hidden_dim_list:
    hidden_dim_list = [320,480,640,800]
    # tiny_model = TinyProteinTransformer(
    #     vocab_size=len(tokenizer_tiny),
    #     num_classes=None
    # )

    tiny_model_320 = TinyProteinTransformer(
        vocab_size=len(tokenizer_tiny),
        num_classes=None,
        hidden_dim = 320
    )
    tiny_model_480 = TinyProteinTransformer(
        vocab_size=len(tokenizer_tiny),
        num_classes=None,
        hidden_dim = 480
    )
    tiny_model_640 = TinyProteinTransformer(
        vocab_size=len(tokenizer_tiny),
        num_classes=None,
        hidden_dim = 640
    )
    tiny_model_800 = TinyProteinTransformer(
        vocab_size=len(tokenizer_tiny),
        num_classes=None,
        hidden_dim = 800
    )
    


    ################
    #Load Model
    #################

    # hidden_dim_list = [320,480,800]
    # # num_heads=10 embed_dim must be divisible by num_heads   
    #tiny_model.load_state_dict(torch.load("./Ablation/Hidden_dim/320/best_pretrain.pt")) 
    #tiny_model.load_state_dict(torch.load("./Ablation/Hidden_dim/480/best_pretrain.pt")) 
    #tiny_model.load_state_dict(torch.load("./Ablation/Hidden_dim/800/best_pretrain.pt"))        
  
        
    tiny_model_320.load_state_dict(torch.load(f"./Ablation/Hidden_dim/320/best_pretrain.pt")) 
    tiny_model_480.load_state_dict(torch.load(f"./Ablation/Hidden_dim/480/best_pretrain.pt")) 
    tiny_model_640.load_state_dict(torch.load(f"./Ablation/Hidden_dim/640/best_pretrain.pt")) 
    tiny_model_800.load_state_dict(torch.load(f"./Ablation/Hidden_dim/800/best_pretrain.pt")) 
     
    # tiny_model.load_state_dict(torch.load(f"./Ablation/TPT_Weight/best_pretrain_10M.pt")) #640
    tiny_model_320.cuda()
    tiny_model_320.eval()
    
    tiny_model_480.cuda()
    tiny_model_480.eval()
    
    tiny_model_640.cuda()
    tiny_model_640.eval()

    tiny_model_800.cuda()
    tiny_model_800.eval()

    for p in tiny_model_320.parameters():
        p.requires_grad = False

    for p in tiny_model_480.parameters():
        p.requires_grad = False
    
    for p in tiny_model_640.parameters():
        p.requires_grad = False
    
    for p in tiny_model_800.parameters():
        p.requires_grad = False

    # ============================================================
    # 4) Load ESM2 models
    # ============================================================
    import esm
    import argparse

    torch.serialization.add_safe_globals([argparse.Namespace])


    # ============================================================
    # 6) Encode function for each backbone
    # ============================================================
    @torch.no_grad()
    def encode_tiny_320(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_320.encode(x)               # B, L, D
        pooled = tiny_model_320.attention_pool(h)  # B, D
        
        return pooled.squeeze(0)
    
    @torch.no_grad()
    def encode_tiny_480(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_480.encode(x)               # B, L, D
        pooled = tiny_model_480.attention_pool(h)  # B, D
        
        return pooled.squeeze(0)
    
    @torch.no_grad()
    def encode_tiny_640(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_640.encode(x)               # B, L, D
        pooled = tiny_model_640.attention_pool(h)  # B, D
        
        return pooled.squeeze(0)
    
    @torch.no_grad()
    def encode_tiny_800(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_800.encode(x)               # B, L, D
        pooled = tiny_model_800.attention_pool(h)  # B, D
        
        return pooled.squeeze(0)

    # ============================================================
    # 7) Training loop (linear classifier)
    # ============================================================

    from torch.utils.data import DataLoader, TensorDataset
    # X_train, y_train, X_val, y_val, epochs=5, lr=1e-3, batch_size=64 Oringin
    def train_classifier(X_train, y_train, X_val, y_val, epochs=5, lr=1e-3, batch_size=64):
        clf = nn.Linear(X_train.size(1), 2).cuda()
        opt = torch.optim.AdamW(clf.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # ---- build dataloader ----
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_acc = -1
        best_f1 = -1
        best_auc = -1

        for ep in range(epochs):
            clf.train()

            for xb, yb in train_loader:
                xb = xb.cuda()
                yb = yb.cuda()

                opt.zero_grad()
                out = clf(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

            # ---- eval for this epoch ----
            clf.eval()
            with torch.no_grad():
                logits = clf(X_val.cuda())
                preds = logits.argmax(dim=1).cpu().numpy()
                probs = logits.softmax(1)[:,1].cpu().numpy()

            acc = accuracy_score(y_val.cpu().numpy(), preds)
            f1 = f1_score(y_val.cpu().numpy(), preds)
            auc = roc_auc_score(y_val.cpu().numpy(), probs)
            print(f"Epoch {ep}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

            # ---- keep best ----
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_auc = auc

        return best_acc, best_f1, best_auc

    # ============================================================
    # 8) 5-Fold CV wrapper
    # ============================================================


    def cross_validate_model(encode_fn, model_name):
        print(f"\n===== Running 5-Fold CV for {model_name} =====")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # n_splits=5, shuffle=True, random_state=42 Oringin

        accs, f1s, aucs = [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
            print(f"Fold {fold+1}")

            seq_train = [sequences[i] for i in train_idx]
            seq_val   = [sequences[i] for i in val_idx]
            y_train = torch.tensor(labels[train_idx]).cuda()
            y_val   = torch.tensor(labels[val_idx]).cuda()

            # Encode all sequences (frozen encoder)
            X_train = torch.stack([encode_fn(s) for s in tqdm(seq_train)])
            X_val   = torch.stack([encode_fn(s) for s in tqdm(seq_val)])

            acc, f1, auc = train_classifier(X_train, y_train, X_val, y_val)
            accs.append(acc); f1s.append(f1); aucs.append(auc)

        acc_mean, acc_std = np.mean(accs), np.std(accs)
        f1_mean,  f1_std  = np.mean(f1s), np.std(f1s)
        auc_mean, auc_std = np.mean(aucs), np.std(aucs)

        print(f"\n{model_name} Results:")
        print(f"ALL Results:")
        print(f"ACC: {accs} ")
        print(f"F1 : {f1s}")
        print(f"AUC: {aucs}")
        
        logging.info(f"\n{model_name} Results:")
        logging.info(f"ACC: {accs} ")
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


        return {
            "model": model_name,
            "acc": np.mean(accs),
            "f1": np.mean(f1s),
            "auc": np.mean(aucs)
        }
        


    TinyProteinTransformer
    # ============================================================
    # 9) Run all models
    # ============================================================
    results = []
    
    # def encode_tiny_320(seq):
    # def encode_tiny_480(seq):
    # def encode_tiny_640(seq):
    # def encode_tiny_800(seq):

    # TinyProteinTransformer
    results.append(cross_validate_model(encode_tiny_320, f"HiddenDim320DataSet{datasetusing}"))
    results.append(cross_validate_model(encode_tiny_480, f"HiddenDim480DataSet{datasetusing}"))
    results.append(cross_validate_model(encode_tiny_640, f"HiddenDim640DataSet{datasetusing}"))
    results.append(cross_validate_model(encode_tiny_800, f"HiddenDim800DataSet{datasetusing}"))


    # ============================================================
    # 10) Show result table
    # ============================================================
    print(f"\n==== Final Results Summary  ====")
    logging.info(f"==== Final Results Summary {datasetusing} ====")
    logging.info(f"{results}")
    df_res = pd.DataFrame(results)
    print(df_res)


    # ============================================================
    # 11) 统计测试
    # ============================================================
    def statistical_test(col1, col2, paired=False, parametric=True):
        """
        Parameters
        ----------
        col1, col2 : list or array
        paired : bool
        parametric : bool
        -------
        Returns
        p_value : float
        method : str
        """
        from scipy import stats
        if paired:
            if parametric:
                t_stat, p = stats.ttest_rel(col1, col2)
                method = "paired t-test"
            else:
                w_stat, p = stats.wilcoxon(col1, col2)
                method = "Wilcoxon signed-rank test"
        else:
            if parametric:
                t_stat, p = stats.ttest_ind(col1, col2)
                method = "independent t-test"
            else:
                u_stat, p = stats.mannwhitneyu(col1, col2, alternative='two-sided')
                method = "Mann-Whitney U test"
        return p, method

