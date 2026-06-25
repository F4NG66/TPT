
###############################################################
#                         AMP Task
###############################################################
Silhouette Scores → Tiny: 0.0195, ESM35: 0.0530, ESM150: 0.0536, Probert: 0.0911
###############################################################
#                        TOXIC Task
###############################################################
Silhouette Scores → Tiny: 0.0425, ESM35: 0.0616, ESM150: 0.0575, Probert: 0.0677
###############################################################
#                        CRISP Task
###############################################################
Silhouette Scores → Tiny: 0.0024, ESM35: -0.0399, ESM150: -0.0337, Probert: -0.0495
###############################################################
#                        MOES Task
###############################################################
Silhouette Scores → Tiny: 0.0060, ESM35: 0.1094, ESM150: 0.0956, Probert: 0.1398
"""

# ======================================================
# =============  0. Imports and Setup  =================
# ======================================================
from sklearn.metrics import silhouette_score
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from umap.umap_ import UMAP

import esm
from transformers import T5EncoderModel, T5Tokenizer, BertTokenizer, BertModel
from Ablation.gate_tpt import  GateTinyProteinTransformer 

plt.style.use("seaborn-v0_8")
torch.set_grad_enabled(False)

# ======================================================
# =============  1. Fix Seeds  =========================
# ======================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
# DataSetList = ["AMP","TOX","CRISP","MOES"]
# DataSetList = ["CRISP","MOES"]
DataSetList = ["CRISP","MOES"]

for datasetusing in DataSetList:
    # ======================================================
    # =============  2. Load AMP dataset  ==================
    # ======================================================
    if datasetusing == "AMP":
        df = pd.read_csv("./AMP.csv")
        sequences = df["Sequence"].tolist()
        labels = df["label"].values


    # ======================================================
    # =============  2. Load TOX dataset  ==================
    # ======================================================
    if datasetusing == "TOX":
        df = pd.read_csv("./tox.csv")
        sequences = df["Sequence"].tolist()
        labels = df["label"].values

    # ======================================================
    # =============  2. Load CRISP dataset  ==================
    # ======================================================
    if datasetusing == "CRISP":
        df = pd.read_csv("./Acr.csv")
        # all_sequences_CRISPR.csv
        sequences = df["Sequence"].tolist()
        labels = df["label"].values

    # ======================================================
    # =============  2. Load MOES dataset  ==================
    # ======================================================
    if datasetusing == "MOES":
        df = pd.read_csv("./BCN.csv")
        # all_sequences_MOES.csv
        sequences = df["Sequence"].tolist()
        labels = df["label"].values

        print(f"Loaded {datasetusing} dataset:", df.shape)


    # ======================================================
    # =============  3. Tokenizer & Tiny Model  ============
    # ======================================================
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


    # ---------------- TinyProteinTransformer ----------------
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
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.pos   = nn.Embedding(max_len, hidden_dim)

            self.convs = nn.ModuleList([
                nn.Conv1d(hidden_dim, cnn_out, k, padding=k//2)
                for k in cnn_kernel_sizes
            ])
            self.conv_proj = nn.Linear(cnn_out * len(cnn_kernel_sizes), hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.attn_pool = nn.Linear(hidden_dim, 1)
            self.dropout   = nn.Dropout(dropout)
            self.ln = nn.LayerNorm(hidden_dim)

        def encode(self, x):
            B, L = x.size()
            pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

            h = self.embed(x) + self.pos(pos_ids)
            h = self.dropout(h)

            cnn_in = h.permute(0,2,1)
            convs = [F.relu(conv(cnn_in)) for conv in self.convs]
            conv_cat = torch.cat(convs, dim=1).permute(0,2,1)
            conv_out = self.conv_proj(conv_cat)

            h = self.ln(h + conv_out)
            return self.encoder(h)

        def attention_pool(self, h):
            att = torch.softmax(self.attn_pool(h), dim=1)
            return (att * h).sum(dim=1)


    # -------- Load pretrained Tiny --------
    tiny_model = TinyProteinTransformer(vocab_size=len(tokenizer_tiny))
    tiny_model.load_state_dict(torch.load("./Ablation/TPT_Weight/best_pretrain.pt"),strict=False)
    tiny_model.cuda().eval()
    for p in tiny_model.parameters():
        p.requires_grad=False

    VALID = set("ACDEFGHIKLMNPQRSTVWYBXZ")

    def clean_seq(seq):
        return "".join([a if a in VALID else "X" for a in seq])
        
    # ======================================================
    # =============  4. Load ESM Models  ===================
    # ======================================================
    # esm_models = {
    #     "esm2_t30_150M": "esm2_t30_150M_UR50D",
    #     "esm2_t33_650M": "esm2_t33_650M_UR50D",
    # }

    import argparse

    torch.serialization.add_safe_globals([argparse.Namespace])

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


    # ======================================================
    # =============  5. Encode functions  ==================
    # ======================================================
    @torch.no_grad()
    def encode_tiny(seq):
        x = torch.tensor(tokenize_seq_tiny(seq)).unsqueeze(0).cuda()
        h = tiny_model.encode(x)
        return tiny_model.attention_pool(h).squeeze(0)


    @torch.no_grad()
    def encode_esm(model, alphabet, seq):
        batch = [(0, seq)]
        data = alphabet.get_batch_converter()(batch)
        _, _, toks = data
        toks = toks.cuda()
        out = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
        return out.mean(1).squeeze(0)
    
    
    @torch.no_grad()
    def encode_protbert(seq):
        seq_spaced = " ".join(list(seq))   
        

        toks = protbert_tokenizer(seq_spaced, return_tensors="pt", padding=True)
        for k in toks:
            toks[k] = toks[k].cuda()
        out = protbert_model(**toks).last_hidden_state


        
        return out.mean(1).squeeze(0)       


    # ======================================================
    # =============  6. Build visualization subset =========
    # ======================================================
    #N = 7513
    N = int(0.9*len(sequences))
    #8347
    # get 0.9
    idx = np.random.choice(len(sequences), N, replace=False)
    seq_vis  = [sequences[i] for i in idx]
    lab_vis  = np.array([labels[i] for i in idx])


    # ======================================================
    # =============  7. Batch Encoding  ====================
    # ======================================================
    def batch_encode(seqs, fn):
        embs = []
        for s in tqdm(seqs):
            embs.append(fn(s).cpu().numpy())
        return np.array(embs)

    print("Encoding Tiny...")
    emb_tiny = batch_encode(seq_vis, encode_tiny)

    print("Encoding ESM35...")
    emb_35 = batch_encode(seq_vis, lambda s: encode_esm(*esm_encoders["esm2_t12_35M"], clean_seq(s)))

    print("Encoding ESM150...")
    emb_150 = batch_encode(seq_vis, lambda s: encode_esm(*esm_encoders["esm2_t30_150M"], clean_seq(s)))

    print("Encoding Probert...")
    emb_probert = batch_encode(seq_vis, encode_protbert)

    # ======================================================
    # =============  8. Dimensionality Reduction  ==========
    # ======================================================
    emb_tiny32 = emb_tiny.astype(np.float32)
    emb_35_32 = emb_35.astype(np.float32)
    emb_150_32 = emb_150.astype(np.float32)
    emb_probert_32 = emb_probert.astype(np.float32)


    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    t_tiny = tsne.fit_transform(emb_tiny32)
    t_35 = tsne.fit_transform(emb_35_32)
    t_150  = tsne.fit_transform(emb_150_32)
    t_probert = tsne.fit_transform(emb_probert_32)




    # ======================================================
    sil_tiny = silhouette_score(emb_tiny, lab_vis)
    sil_35   = silhouette_score(emb_35,   lab_vis)
    sil_150  = silhouette_score(emb_150,  lab_vis)
    sil_probert  = silhouette_score(t_probert,  lab_vis)
    print(f"Silhouette Scores → Tiny: {sil_tiny:.4f}, ESM35: {sil_35:.4f}, ESM150: {sil_150:.4f}, Probert: {sil_probert:.4f}")

    # ======================================================


    def plot_embed1(embed, labels, title, modelname, datasetname):
        plt.figure(figsize=(7,5))
        plt.scatter(embed[:,0], embed[:,1],
                    c=labels, cmap="coolwarm",
                    s=14, alpha=0.8, edgecolor="none")
        plt.title(title, fontsize=16)
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./VisulPlot/tsne_{datasetname}_{modelname}.png",
                    format="png",
                    bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_embed2(embed, labels, title):
        plt.figure(figsize=(7,5))
        plt.scatter(embed[:,0], embed[:,1],
                    c=labels, cmap="coolwarm",
                    s=14, alpha=0.8, edgecolor="none")
        plt.title(title, fontsize=16)
        plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./VisulPlot/tsne_{datasetname}_{modelname}.png",
                    format="png",
                    bbox_inches="tight")
        plt.show()
        plt.close()
        

        
    # ---------------- t-SNE ----------------
    plot_embed1(t_tiny, lab_vis, f"t-SNE – TinyProteinTransformer ({datasetusing})","TPT",datasetusing)
    plot_embed1(t_35, lab_vis, f"t-SNE – ESM2-35M ({datasetusing})","ESM35",datasetusing)
    plot_embed1(t_150, lab_vis, f"t-SNE – ESM2-150M ({datasetusing})","ESM150",datasetusing)
    plot_embed1(t_probert, lab_vis, f"t-SNE – Probert ({datasetusing})","Probert",datasetusing)


    
