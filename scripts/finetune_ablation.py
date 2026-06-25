#消融实验
# (8)The manuscript does not evaluate the individual contribution of key components (multi-scale CNN, contrastive learning, attention pooling, gated attention). Ablation studies are necessary to justify architectural design. :

# 推理速度记录 吞吐量 显存等等 还有对比方法的
# (9)The efficiency claims are promising but require more rigorous and standardized reporting. The authors should clearly describe the benchmarking setup and include additional metrics such as memory usage or throughput. :
"""
###############################################################
#                         AMP Task
###############################################################


Gate_TPT

TPT

TPT_WO_CNN


TPT_WO_AttenPool

esm2_t12_35M


esm2_t30_150M


protbert


###############################################################
#                        TOXIC Task
###############################################################

Gate_TPT

TPT

TPT_WO_CNN


TPT_WO_AttenPool


esm2_t12_35M


esm2_t30_150M


protbert

###############################################################
#                         CRISP Task
###############################################################

Gate_TPT

TPT

TPT_WO_CNN


TPT_WO_AttenPool


esm2_t12_35M


esm2_t30_150M


protbert

###############################################################
#                         MOES Task
###############################################################

Gate_TPT

TPT

TPT_WO_CNN


TPT_WO_AttenPool


esm2_t12_35M


esm2_t30_150M


protbert
"""

##### import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
from torch.utils.data import Dataset
import random
from Utils.PerformanceMonitor import EfficiencyMonitor

#消融模型
from Ablation.gate_tpt import  GateTinyProteinTransformer #门控已经有权重不用跑
from Ablation.tpt import TinyProteinTransformer as TinyProteinTransformerLoss  # 对比学习在此 消融 消除loss
from Ablation.tpt_without_attenpool import TinyProteinTransformerWithoutAttentionPooling #在推理阶段替换 或者在训练里替换为平均池化
from Ablation.tpt_withoutCNN import TinyProteinTransformerWithoutCNN # 消融 去除CNN

import logging
logging.basicConfig(filename='./log/Abalation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
###############################################################
# DataSetList = ["AMP","TOX","CRISP","MOES"]
DataSetList = ["TOX","MOES"]
# DataSetList = ["MOES"]
for  datasetusing in DataSetList:
    if datasetusing == "AMP":
        df = pd.read_csv("AMP_amplify.csv")  
    if datasetusing == "TOX":
        df = pd.read_csv("tox.csv")      
    if datasetusing == "CRISP":
        df = pd.read_csv("./sequences_length_le100_CRISPR.csv") # # all_sequences_CRISPR.csv
    if datasetusing == "MOES":
        df = pd.read_csv("./short_sequences_MOES.csv") # all_sequences_MOES.csv


    print(df.head())

    labels = df["label"].values
    sequences = df["Sequence"].tolist()
    
    # print(f"{len(sequences)}")

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


    #####################################################################################
    #Change Model
    #####################################################################################
    tiny_model_Loss = TinyProteinTransformer(vocab_size=len(tokenizer_tiny),num_classes=None)
    tiny_model_Gate = GateTinyProteinTransformer(vocab_size=len(tokenizer_tiny)) #Gate TPT
    tiny_model_WOAtt = TinyProteinTransformerWithoutAttentionPooling(vocab_size=len(tokenizer_tiny)) #Att
    tiny_model_WOCNN = TinyProteinTransformerWithoutCNN(vocab_size=len(tokenizer_tiny)) #CNN

    tiny_model = TinyProteinTransformer(
        vocab_size=len(tokenizer_tiny),
        num_classes=None
    ) #normer
    #####################################################################################
    #Load Model
    #####################################################################################
    # Ablation Model
    #tiny_model.load_state_dict(torch.load("./Ablation/TPT_Contrast_Weight/best_pretrain_1M.pt"))
    #tiny_model.load_state_dict(torch.load("./Ablation/gate_TPT_Weight/best_pretrain_1M.pt"))
    #tiny_model.load_state_dict(torch.load("./Ablation/TPT_Without_Attenpool_weight/best_pretrain_1M.pt"))
    #tiny_model.load_state_dict(torch.load("./Ablation/TPT_Without_CNN_Weight/best_pretrain_1M.pt"))

    tiny_model_Loss.load_state_dict(torch.load("./Ablation/TPT_Contrast_Weight/best_pretrain_10M.pt"))
    tiny_model_Gate.load_state_dict(torch.load("./Ablation/gate_TPT_Weight/best_pretrain_10M.pt"))
    tiny_model_WOAtt.load_state_dict(torch.load("./Ablation/TPT_Without_Attenpool_weight/best_pretrain_10M.pt"))
    tiny_model_WOCNN.load_state_dict(torch.load("./Ablation/TPT_Without_CNN_Weight/best_pretrain_10M.pt"))

    tiny_model.load_state_dict(torch.load("./Ablation/TPT_Weight/best_pretrain_10M.pt"))

    tiny_model.cuda()
    tiny_model.eval()

    tiny_model_Loss.cuda()
    tiny_model_Loss.eval()

    tiny_model_Gate.cuda()
    tiny_model_Gate.eval()

    tiny_model_WOAtt.cuda()
    tiny_model_WOAtt.eval()

    tiny_model_WOCNN.cuda()
    tiny_model_WOCNN.eval()

    for p in tiny_model.parameters():
        p.requires_grad = False
        
    for p in tiny_model_Loss.parameters():
        p.requires_grad = False
        
    for p in tiny_model_Gate.parameters():
        p.requires_grad = False
        
    for p in tiny_model_WOAtt.parameters():
        p.requires_grad = False

    for p in tiny_model_WOCNN.parameters():
        p.requires_grad = False


    # ============================================================
    # 4) Load ESM2 models
    # ============================================================
    import esm
    import argparse

    # 将 argparse.Namespace 添加到 PyTorch 的全局安全列表中
    torch.serialization.add_safe_globals([argparse.Namespace])

    #  "esm2_t12_35M": "./PreTrain_model/ESM35/esm2_t12_35m/model.pt",
    #   "esm2_t30_150M": "./PreTrain_model/ESM150/esm2_t30_150m/model.safetensors",
    esm_models = {  
        "esm2_t12_35M": "./PreTrain_model/ESM35/esm2_t12_35m/esm2_t12_35M_UR50D.pt",
        "esm2_t30_150M": "./PreTrain_model/ESM150/esm2_t30_150m/esm2_t30_150M_UR50D.pt"
    }

    esm_encoders = {}
    for name, tag in esm_models.items():
        #model, alphabet = esm.pretrained.load_model_and_alphabet(tag)
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(tag)
        model = model.cuda().eval()
        for p in model.parameters(): p.requires_grad = False
        esm_encoders[name] = (model, alphabet)


    from transformers import BertTokenizer, BertModel

    enable_protbert = True

    if enable_protbert:
        protbert_tokenizer = BertTokenizer.from_pretrained("./PreTrain_model/Probert", do_lower_case=False)
        protbert_model = BertModel.from_pretrained("./PreTrain_model/Probert")
        protbert_model.cuda().eval()
        for p in protbert_model.parameters():
            p.requires_grad = False

    # ============================================================
    # 6) Encode function for each backbone
    # ============================================================

    @torch.no_grad()
    def encode_tiny_model_Loss(seq,monitor):
        start_time = time.time() 
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_Loss.encode(x)               # B, L, D
        pooled = tiny_model_Loss.attention_pool(h)  # B, D
        
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_Gate(seq,monitor):
        start_time = time.time() 
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_Gate.encode(x)               # B, L, D
        pooled = tiny_model_Gate.attention_pool(h)  # B, D
        
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_WOAtt(seq,monitor):
        start_time = time.time() 
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_WOAtt.encode(x)               # B, L, D
        pooled = h.mean(1)  # B, 1, D
        
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        return pooled.squeeze(0)

    @torch.no_grad()
    def encode_tiny_model_WOCNN(seq,monitor):
        start_time = time.time() 
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        
        h = tiny_model_WOCNN.encode(x)               # B, L, D
        pooled = tiny_model_WOCNN.attention_pool(h)  # B, D
        
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        return pooled.squeeze(0)


    @torch.no_grad()
    def encode_tiny(seq,monitor):
        start_time = time.time() 
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model.encode(x)               # B, L, D
        pooled = tiny_model.attention_pool(h)  # B, D
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        return pooled.squeeze(0)


    @torch.no_grad()
    def encode_esm(model, alphabet, seq, monitor):
        batch = [(0, seq)]
        
        start_time = time.time()
        
        data = alphabet.get_batch_converter()(batch)
        _, _, toks = data
        toks = toks.cuda()
        out = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
        
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        
        return out.mean(1).squeeze(0)  # mean pooling

    @torch.no_grad()
    def encode_protbert(seq,monitor):
        seq_spaced = " ".join(list(seq))   
        
        start_time = time.time()
        toks = protbert_tokenizer(seq_spaced, return_tensors="pt", padding=True)
        for k in toks:
            toks[k] = toks[k].cuda()

        out = protbert_model(**toks).last_hidden_state
        inference_time = time.time() - start_time
        monitor.inference_times.append(inference_time)
        
        return out.mean(1).squeeze(0)        

    # ============================================================
    # 7) Training loop (linear classifier)
    # ============================================================

    from torch.utils.data import DataLoader, TensorDataset
    # X_train, y_train, X_val, y_val, epochs=5, lr=1e-3, batch_size=64 Oringin #(X_train, y_train, X_val, y_val, epochs=1, lr=1e-3, batch_size=128)
    def train_classifier(X_train, y_train, X_val, y_val, epochs=5, lr=1e-3, batch_size=64):  ############################################
        clf = nn.Linear(X_train.size(1), 2).cuda()
        opt = torch.optim.AdamW(clf.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # ---- build dataloader ----
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_acc = -1
        best_f1 = -1
        best_auc = -1
        
        # hardware_info = "NVIDIA RTX 5090, 2TB RAM"
        # monitor = EfficiencyMonitor()



        for ep in range(epochs):
            clf.train()
            
            # print("Starting training...")
            # monitor.start_training_timer()
            
            for xb, yb in train_loader:
                xb = xb.cuda()
                yb = yb.cuda()

                opt.zero_grad()
                out = clf(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
            
            #training_time = monitor.end_training_timer()
            

            # ---- eval for this epoch ----
            clf.eval()
            with torch.no_grad():
                #start_time = time.time()
                
                logits = clf(X_val.cuda())
                
                #inference_time = time.time() - start_time
                
                preds = logits.argmax(dim=1).cpu().numpy()
                probs = logits.softmax(1)[:,1].cpu().numpy()

            acc = accuracy_score(y_val.cpu().numpy(), preds)
            f1 = f1_score(y_val.cpu().numpy(), preds)
            auc = roc_auc_score(y_val.cpu().numpy(), probs)
            #get_efficiency_report(training_time,hardware_info)
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
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) ############################################
        # n_splits=5, shuffle=True, random_state=42 Oringin

        accs, f1s, aucs = [], [], []
        
        hardware_info = "NVIDIA RTX 5090, 2TB RAM"
        monitor = EfficiencyMonitor()

        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
            print(f"Fold {fold+1}")

            seq_train = [sequences[i] for i in train_idx]
            seq_val   = [sequences[i] for i in val_idx]
            y_train = torch.tensor(labels[train_idx]).cuda()
            y_val   = torch.tensor(labels[val_idx]).cuda()

            # Encode all sequences (frozen encoder)
            X_train = torch.stack([encode_fn(seq = s,monitor= monitor) for s in tqdm(seq_train)])
            X_val   = torch.stack([encode_fn(seq = s,monitor= monitor) for s in tqdm(seq_val)])
            
            Strat = time.time()
            acc, f1, auc = train_classifier(X_train, y_train, X_val, y_val)
            End = time.time()
            training_time = End - Strat
        
        
            
            accs.append(acc); f1s.append(f1); aucs.append(auc)
            
        

        acc_mean, acc_std = np.mean(accs), np.std(accs)
        f1_mean,  f1_std  = np.mean(f1s), np.std(f1s)
        auc_mean, auc_std = np.mean(aucs), np.std(aucs)
        
        report = monitor.get_efficiency_report(training_time, hardware_info)
        logging.info(f"\n{model_name}  efficiency_report:")
        logging.info(f"{report}")
        print(report)

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

    #TinyProteinTransformer
    results.append(cross_validate_model(encode_tiny, "TinyProteinTransformer"))
    
    results.append(cross_validate_model(encode_tiny_model_Loss, "tiny_model_Loss"))

    results.append(cross_validate_model(encode_tiny_model_Gate, "tiny_model_Gate"))

    results.append(cross_validate_model(encode_tiny_model_WOAtt, "tiny_model_WOAtt"))

    results.append(cross_validate_model(encode_tiny_model_WOCNN, "tiny_model_WOCNN"))

    # def encode_tiny_model_Loss(seq,monitor):
    # def encode_tiny_model_Gate(seq,monitor):
    # def encode_tiny_model_WOAtt(seq,monitor):
    # def encode_tiny_model_WOCNN(seq,monitor):


    # # ESM models
    for name, (m, alphabet) in esm_encoders.items():
        encode_fn = lambda seq, monitor, m=m, a=alphabet,  : encode_esm(m, a, seq,monitor)
        results.append(cross_validate_model(encode_fn, name))

    if enable_protbert:
        results.append(cross_validate_model(encode_protbert, "ProtBERT"))






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
        计算两列数据的统计检验 p 值
        Parameters
        ----------
        col1, col2 : list or array
            两列数值
        paired : bool
            是否为配对样本
        parametric : bool
            是否使用参数检验（t检验）。False 时使用非参数检验。
        -------
        Returns
        p_value : float
            检验的双侧 p 值
        method : str
            使用的检验方法名称
        一般使用"Wilcoxon signed-rank test"
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

    # # 示例
    # p, method = statistical_test(col1, col2, paired=False, parametric=False)
    # print(f"{method} 的 p 值 = {p:.4f}")