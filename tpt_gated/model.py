# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedTransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer with SDPA gating (G1):
    Y' = Y * sigmoid(W_g(X))
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # GATING MODULE (head-specific)
        # Produces (B, L, H) gating scores
        self.gate_linear = nn.Linear(d_model, nhead)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def sdpa_gated(self, x):
        """
        Apply SDPA + G1 gating:
        attn_output: (B, L, D)
        gate:         (B, L, H) → broadcast to (B, L, D)
        """
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)

        # ---- GATING (G1) ----
        gate = torch.sigmoid(self.gate_linear(x))  # (B, L, H)

        # reshape attention output into head format
        B, L, D = attn_output.shape
        H = self.nhead
        d = D // H
        attn_head = attn_output.view(B, L, H, d)

        # apply gate per head
        gate = gate.unsqueeze(-1)  # (B, L, H, 1)
        gated_head = attn_head * gate

        # merge back
        gated_output = gated_head.reshape(B, L, D)

        return gated_output

    def forward(self, src,src_mask=None, src_key_padding_mask=None, is_causal=False):
        # ---- SDPA + G1 gating ----
        x = src + self.dropout(self.sdpa_gated(src))
        x = self.norm1(x)

        # ---- FFN ----
        ff = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(ff)
        x = self.norm2(x)

        return x

class TinyProteinTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=None,
        hidden_dim=1280,
        num_layers=30,
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
        encoder_layer = GatedTransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
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
