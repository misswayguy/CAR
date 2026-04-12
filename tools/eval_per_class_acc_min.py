#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def build_eval_tf(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

@torch.no_grad()
def per_class_acc(model: nn.Module, root: str, img_size: int, batch_size: int, workers: int):
    ds = datasets.ImageFolder(root, transform=build_eval_tf(img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    n_cls = len(ds.classes)
    device = next(model.parameters()).device
    model.eval()

    correct = [0] * n_cls
    total   = [0] * n_cls

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(imgs).argmax(dim=1)
        for p, t in zip(preds.tolist(), targets.tolist()):
            total[t]   += 1
            if p == t: correct[t] += 1

    rows = []
    for c in range(n_cls):
        acc = (correct[c] / total[c]) if total[c] > 0 else 0.0
        rows.append({"class_id": c, "class_name": ds.classes[c], "acc": f"{acc:.6f}"})
    return rows, ds.classes

def load_model(ckpt_path: str, num_classes: int, override_model: str = "", device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model_name = override_model or cfg.get("model_name", "resnet18")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(device)

    sd = ckpt.get("model", ckpt)        #  {state_dict}  
    sd = {k.replace("module.", ""): v for k, v in sd.items()}  # DP
    model.load_state_dict(sd, strict=False)
    return model

def write_csv(rows, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class_id","class_name","acc"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"✓ Saved: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, default="")
    ap.add_argument("--val-dir", type=str, default="")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, required=True, help="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = args.train_dir or args.val_dir
    if not probe:
        raise ValueError(" --train-dir  --val-dir ")

    #  probe 
    n_cls = len(datasets.ImageFolder(probe).classes)
    model = load_model(args.ckpt, num_classes=n_cls, override_model=args.model, device=device)

    if args.train_dir:
        rows, _ = per_class_acc(model, args.train_dir, args.img_size, args.batch_size, args.workers)
        write_csv(rows, args.out + "_train_acc.csv")

    if args.val_dir:
        rows, _ = per_class_acc(model, args.val_dir, args.img_size, args.batch_size, args.workers)
        #  val-dir  test  test
        write_csv(rows, args.out + "_test_acc.csv")

if __name__ == "__main__":
    main()
