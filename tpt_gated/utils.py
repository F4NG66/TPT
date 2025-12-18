# utils.py
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
