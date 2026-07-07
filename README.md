# TinyProteinTransformer (TPT)

A lightweight CNN-Transformer hybrid encoder for microbial smORF-encoded small proteins, pretrained on the Global Microbial smORF Catalog (GMSC; >280M sequences) with masked language modeling and contrastive learning.
---

```
╔═══════════════════════════════════════════╗
║                                           ║
║        _______ _____ _______              ║
║       |__   __|  __ \__   __|             ║
║          | |  | |__) | | |                ║
║          | |  |  ___/  | |                ║
║          | |  | |      | |                ║
║          |_|  |_|      |_|                ║
║                                           ║
║     ▸ TINY  PROTEIN  TRANSFORMER ◂        ║
║                                           ║
╚═══════════════════════════════════════════╝
```

## Repository Structure

```                 
├── weights/
│   ├── best_pretrain.pt         # Pretrained on GMSC10.90
│   └── Ablation/
│       ├── TPT_Weight/               # Baseline pretrained weights 
│       ├── TPT_Contrast_Weight/      # w/o contrastive loss
│       ├── TPT_Without_Attenpool_weight/
│       └── TPT_Without_CNN_Weight/
├── data/
│   └── GMSC10.90AA.faa               # GMSC pretraining corpus (too large to be placed here)
├── pretrain.py                       # Main pretraining on full GMSC corpus
│   
└── scripts/            
    ├── pretrain_ablation.py          # Ablation pretraining (no CL / attn pool / CNN)
    ├── pretrain_dim_sweep.py         # Hidden dim sweep (320 / 480 / 800)
    ├── finetune_ablation.py          # Downstream eval for ablation variants
    ├── finetune_dim_sweep.py         # Downstream eval for dim sweep variants
    ├── eval_ampscanner_amplify.py    # AMP Scanner v2 + AMPlify baselines
    ├── eval_fastmcws.py              # FastMCWS baseline
    ├── eval_qsp_cpp.py               # additional QSP CPP eval
    ├── check_overlap.py              # GMSC vs downstream overlap (exact + MMseqs2)
    ├── tsne_viz.py                   # t-SNE visualization + silhouette scores
    └── mcws_analysis.py              # MCWS-Transformer reference implementation
```

---
## Architecture

<p align="center"><img src="arch.jpeg" width="700"></p>

**(a)** Embedding layer feeds three parallel branches into stacked Transformer blocks (Multi-Head Attention + FeedForward, ×N layers) with Dropout and Layer Normalization. **(b)** Pretraining objectives: masked language modeling (MLM) and contrastive learning, optimized jointly via cross-entropy and InfoNCE loss.




---

## Weights

Download from Hugging Face: https://huggingface.co/ffbond/TinyProteinTransformer 

---

## Performance

| Category | Model | AMP (AUC) | Acr (AUC) | TOX (AUC) | BCN (AUC) | QSP (AUC) | CPP (AUC) |
|---|---|---|---|---|---|---|---|
| Ours | **TPT** | 0.9295 ± 0.0077 | 0.6962 ± 0.0706 | 0.9299 ± 0.0065 | 0.9219 ± 0.0446 | 0.9225 ± 0.0206 | 0.9380 ± 0.0088 |
| Ours | TPT (Gated) | 0.9173 ± 0.0050 | 0.6923 ± 0.0977 | 0.9257 ± 0.0105 | 0.9471 ± 0.0450 | 0.9005 ± 0.0289 | 0.9371 ± 0.0091 |
| Baseline | ESM-2 (35M) | 0.9185 ± 0.0046 | 0.4629 ± 0.0363 | 0.9151 ± 0.0068 | 0.7822 ± 0.0936 | 0.8902 ± 0.0368 | 0.9265 ± 0.0095 |
| Baseline | ESM-2 (150M) | 0.9280 ± 0.0044 | 0.4280 ± 0.1319 | 0.9251 ± 0.0088 | 0.9551 ± 0.0398 | 0.8912 ± 0.0351 | 0.9221 ± 0.0078 |
| Baseline | ProtBERT | 0.9073 ± 0.0058 | 0.3215 ± 0.0493 | 0.8946 ± 0.0073 | 0.9369 ± 0.0943 | 0.8500 ± 0.0319 | 0.9180 ± 0.0082 |

<details>
<summary>Full results (ACC / F1 / AUC, all six tasks)</summary>

| Category | Model | AMP | | | Acr | | | TOX | | | BCN | | | QSP | | | CPP | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | ACC | F1 | AUC | ACC | F1 | AUC | ACC | F1 | AUC | ACC | F1 | AUC | ACC | F1 | AUC | ACC | F1 | AUC |
| Ours | TPT | 0.8591 | 0.8596 | 0.9295 | 0.7073 | 0.3822 | 0.6962 | 0.8885 | 0.8029 | 0.9299 | 0.9421 | 0.9695 | 0.9219 | 0.8682 | 0.8723 | 0.9225 | 0.9007 | 0.7930 | 0.9380 |
| Ours | TPT (Gated) | 0.8444 | 0.8467 | 0.9173 | 0.7584 | 0.3708 | 0.6923 | 0.8877 | 0.7976 | 0.9257 | 0.9668 | 0.9825 | 0.9471 | 0.8455 | 0.8499 | 0.9005 | 0.9002 | 0.7908 | 0.9371 |
| Baseline | ESM-2 (35M) | 0.8402 | 0.8349 | 0.9185 | 0.8402 | 0.0000 | 0.4629 | 0.8859 | 0.7891 | 0.9151 | 0.9463 | 0.9724 | 0.7822 | 0.8273 | 0.8465 | 0.8902 | 0.8887 | 0.7506 | 0.9265 |
| Baseline | ESM-2 (150M) | 0.8469 | 0.8421 | 0.9280 | 0.8402 | 0.0000 | 0.4280 | 0.8905 | 0.8015 | 0.9251 | 0.9505 | 0.9745 | 0.9551 | 0.8455 | 0.8581 | 0.8912 | 0.8888 | 0.7563 | 0.9221 |
| Baseline | ProtBERT | 0.8243 | 0.8197 | 0.9073 | 0.8402 | 0.0000 | 0.3215 | 0.8443 | 0.7040 | 0.8946 | 0.9545 | 0.9766 | 0.9369 | 0.7750 | 0.8048 | 0.8500 | 0.8794 | 0.7312 | 0.9180 |

</details>

---

## Pretraining

**GMSC** 

```bash
# Place corpus at data/GMSC10.90AA.faa
python scripts/pretrain.py
```

Checkpoints saved to `checkpoint.pt`; best weights to `best_pretrain_288M.pt`.

**Ablation pretraining**:

```bash
# Trains three variants: w/o contrastive loss, w/o attention pooling, w/o CNN
python scripts/pretrain_ablation.py

# Hidden dim sweep: 320, 480, 800 (default is 640)
python scripts/pretrain_dim_sweep.py
```

Ablation weights saved to `Ablation/<variant>/best_pretrain.pt`.

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Hidden dim | 640 |
| Transformer layers | 20 |
| CNN kernel sizes | 3, 5, 7, 9 |
| Pretraining LR | 2e-5 (AdamW) |
| Contrastive loss weight λ | 0.05 |
| Contrastive temperature τ | 0.1 |
| Max sequence length | 128 |
| Batch size | 200 / 64 (ablation)|

---

## Inference

```python
import torch
from model import TinyProteinTransformer
from utils import build_tokenizer

tokenizer = build_tokenizer()
model = TinyProteinTransformer(vocab_size=len(tokenizer))
model.load_state_dict(torch.load("weights/best_pretrain_288M.pt"))
model.cuda().eval()

def encode(seq, max_len=128):
    ids = [tokenizer.get(aa, tokenizer["X"]) for aa in seq][:max_len]
    ids += [tokenizer["PAD"]] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0).cuda()

with torch.no_grad():
    x = encode("MKVLILACLVVVTITVS")
    h = model.encode(x)              # (1, L, 640) residue-level representations
    embedding = model.attention_pool(h)  # (1, 640) sequence embedding
```

For classification, attach a additional classifier on top of `embedding`.

---

## Fine-tuning / Downstream Evaluation

All downstream tasks use **5-fold stratified CV** (`StratifiedKFold(random_state=42)`), frozen encoder + single linear head, AdamW (lr=1e-3), 5 epochs. Input CSV format:

```
Sequence,label
MKVLIL...,1
ACDEFG...,0
```

```bash

python scripts/finetune_ablation.py

python scripts/finetune_dim_sweep.py

python scripts/eval_qsp_cpp.py
```

Supported datasets: `AMP_amplify.csv`, `tox.csv`, `QSP.csv`, `CPP.csv`, `Acr.csv`, `BCN.csv`.

---

## Baseline Comparisons

All baselines trained from scratch with the same 5-fold CV protocol, early stopping on val ACC (patience=10, max 200 epochs):

```bash
# AMP Scanner v2 (Veltri et al., 2018) + AMPlify (Li et al., 2022)
python scripts/eval_ampscanner_amplify.py

# FastMCWS (Mahala et al., 2025)
python scripts/eval_fastmcws.py
```

Results logged to `./log/` as CSV files.

---

## Data Overlap Analysis

Check for sequence leakage between GMSC pretraining and downstream datasets (exact match + MMseqs2 at ≥90% identity):

```bash
# Requires MMseqs2 on PATH (conda install -c bioconda mmseqs2)
python scripts/check_overlap.py
```

Outputs: `./log/overlap_report.txt`, `./log/overlap_exact.csv`.

---

## Visualization

t-SNE projections and silhouette scores comparing TPT, ESM2-35M, ESM2-150M, ProtBERT:

```bash
python scripts/tsne_viz.py
```

Plots saved to `./VisulPlot/`.


---

## Requirements

```bash
pip install torch numpy pandas scikit-learn biopython matplotlib tqdm transformers fair-esm umap-learn
conda install -c bioconda mmseqs2  # for overlap analysis only
```

---

## Citation

```bibtex
@article{sheng2026tpt,
  title={TPT: A Compact CNN-Transformer Encoder for Efficient Microbial Small Protein Modeling},
  author={Sheng, Fang and Zhang, Junhe and Zhu, Chengkai},
  journal={Frontiers in },
  year={2026}
}
```
