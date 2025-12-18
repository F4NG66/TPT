# dataset.py
import torch
from torch.utils.data import Dataset
import random
from utils import augment_sequence, tokenize_seq, load_fasta


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
