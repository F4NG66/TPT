import os
import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import TinyProteinTransformer
from dataset import ProteinDataset
from utils import load_fasta, build_tokenizer

from Ablation.gate_tpt import  GateTinyProteinTransformer 

from Ablation.tpt import TinyProteinTransformer
from Ablation.tpt_without_attenpool import TinyProteinTransformerWithoutAttentionPooling
from Ablation.tpt_withoutCNN import TinyProteinTransformerWithoutCNN 
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:1'


def contrastive_loss(z1, z2, tau=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def save_loss_plot(log_dict, fname="pretrain_loss.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(log_dict["mlm"], label="MLM Loss")
    plt.plot(log_dict["contrastive"], label="Contrastive Loss")
    plt.plot(log_dict["total"], label="Total Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Pretraining Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved loss plot to {fname}")


def main():
    xiaorong = ["TPT_Contrast_Weight","TPT_Without_Attenpool_weight","TPT_Without_CNN_Weight"]
    # hidden_dim_list = [256,512 ,1024]
    for xr in  xiaorong:

    
        seqs = load_fasta("./benchmark/sample_10M.fasta")
        tokenizer = build_tokenizer()

        dataset = ProteinDataset(sequences=seqs, tokenizer=tokenizer, mode="pretrain")
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

        print("Building Model...")
        

        if xr == "TPT_Contrast_Weight":
             model = TinyProteinTransformer(vocab_size=len(tokenizer)) 
        if xr == "TPT_Without_Attenpool_weight":
             model = TinyProteinTransformerWithoutAttentionPooling(vocab_size=len(tokenizer)) 
        if xr == "TPT_Without_CNN_Weight":
             model = TinyProteinTransformerWithoutCNN(vocab_size=len(tokenizer)) 
        
        print("Now Training Model is {xr}...")
       

     

        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        #     model = torch.nn.DataParallel(model,device_ids=[0,1])

        model = model.cuda()

        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scaler = GradScaler()

        start_epoch = 0
        best_loss = float("inf")

        # if os.path.exists("checkpoint.pt"):
        #     print("Loading checkpoint.pt ...")
        #     ckpt = torch.load("checkpoint.pt", map_location="cpu")

        #     if isinstance(model, torch.nn.DataParallel):
        #         model.module.load_state_dict(ckpt["model"])
        #     else:
        #         model.load_state_dict(ckpt["model"])
        #     optim.load_state_dict(ckpt["optim"])
        #     scaler.load_state_dict(ckpt["scaler"])

        #     start_epoch = ckpt["epoch"] + 1
        #     best_loss = ckpt["best_loss"]

        if os.path.exists("loss_log.json"):
            with open("loss_log.json", "r") as f:
                log_dict = json.load(f)
        else:
            log_dict = {"mlm": [], "contrastive": [], "total": []}

        print(f"Starting Pretraining from epoch {start_epoch}")

        EPOCHS = 1

        for epoch in range(start_epoch, EPOCHS):

            progress = tqdm(loader, ncols=120)
            total_epoch_loss = 0

            for step, batch in enumerate(progress):

                mlm_in = batch["mlm_input"].cuda()
                mlm_tg = batch["mlm_target"].cuda()
                a1 = batch["aug1"].cuda()
                a2 = batch["aug2"].cuda()

                with autocast():
                    out = model(mlm_input=mlm_in, aug1=a1, aug2=a2)

                    mlm_logits = out["mlm_logits"]
                    h1 = out["h1"]
                    h2 = out["h2"]

                    loss_mlm = F.cross_entropy(mlm_logits.permute(0, 2, 1), mlm_tg)
                    loss_con = contrastive_loss(h1, h2)
                    if xr == "TPT_Contrast_Weight":
                      loss = loss_mlm 
                    else :
                      loss = loss_mlm + 0.05 * loss_con

                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                log_dict["mlm"].append(loss_mlm.item())
                log_dict["contrastive"].append(loss_con.item())
                log_dict["total"].append(loss.item())
                total_epoch_loss += loss.item()

                progress.set_postfix({
                    "MLM": f"{loss_mlm.item():.3f}",
                    "Con": f"{loss_con.item():.3f}",
                    "Total": f"{loss.item():.3f}"
                })

            avg_loss = total_epoch_loss / len(loader)
            print(f"./log/Epoch {epoch} finished. Avg Loss = {avg_loss:.4f}")

            with open(f"loss_log{xr}.json", "w") as f:
                json.dump(log_dict, f)

            save_loss_plot(log_dict,fname=f"./VisulPlot/{xr}_pretrain_loss.png")

            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "optim": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss
            }

            torch.save(state, f"./Ablation/{xr}/checkpoint.pt")
            print("Saved checkpoint.pt")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(state["model"], f"./Ablation/{xr}/best_pretrain.pt")
                print("Updated best_pretrain.pt")

        print("Pretraining Completed")


if __name__ == "__main__":
    main()
