#!/usr/bin/env python3
import argparse, json, os, csv
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from collections import defaultdict

DATA_ROOT = "/mnt/data/lsy/ZZQ/cifar100-LT-IF100"

# =====  =====
def build_model_and_load(ckpt_path):
    #  ResNet-50  100 
    model = models.resnet18(num_classes=100)
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)          # 
    state = {k.replace("module.", ""): v for k, v in state.items()}  # DDP
    # fc
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model

def make_test_loader(sel_ids, batch_size=256, num_workers=4):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=tf)
    idxs = [i for i, (_, y) in enumerate(test.samples) if y in set(sel_ids)]
    sub = Subset(test, idxs)
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers), test

def per_class_acc(model, loader, device, sel_ids):
    model.eval()
    correct = defaultdict(int); total = defaultdict(int)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for cid in sel_ids:
                m = (y == cid)
                if m.any():
                    total[cid]   += int(m.sum().item())
                    correct[cid] += int((pred[m] == cid).sum().item())
    return {cid: (correct[cid] / total[cid] if total[cid] > 0 else 0.0) for cid in sel_ids}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hmt_json", default="hmt_and_selected9.json")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    sel = json.load(open(args.hmt_json))
    ids9, names9 = sel["selected_ids"], sel["selected_names"]
    head, med, tail = set(sel["groups"]["head"]), set(sel["groups"]["medium"]), set(sel["groups"]["tail"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_and_load(args.ckpt).to(device)

    test_loader, _ = make_test_loader(ids9)
    acc_te = per_class_acc(model, test_loader, device, ids9)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "test_acc"])
        for cid, name in zip(ids9, names9):
            w.writerow([name, f"{acc_te[cid]:.4f}"])

    # H/M/T 93+ Tail
    ids_head = [i for i in ids9 if i in head]
    ids_med  = [i for i in ids9 if i in med]
    ids_tail = [i for i in ids9 if i in tail]
    summary = {
        "H_mean_test": sum(acc_te[i] for i in ids_head)/len(ids_head),
        "M_mean_test": sum(acc_te[i] for i in ids_med)/len(ids_med),
        "T_mean_test": sum(acc_te[i] for i in ids_tail)/len(ids_tail),
        "Tail_worst_test": min(acc_te[i] for i in ids_tail),
    }
    with open(args.out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
