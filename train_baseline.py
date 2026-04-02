

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy



# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_by_class(imagefolder_dataset) -> np.ndarray:
    """Count #samples per class from an ImageFolder dataset."""
    if hasattr(imagefolder_dataset, "targets"):
        targets = imagefolder_dataset.targets
    else:
        targets = [y for _, y in imagefolder_dataset.samples]
    nclass = len(imagefolder_dataset.classes)
    hist = np.zeros(nclass, dtype=np.int64)
    for y in targets:
        hist[y] += 1
    return hist


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str):

    assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
    print(f"==> Loading pretrained from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: sd = ckpt["state_dict"]
        elif "model" in ckpt:    sd = ckpt["model"]
        else:                    sd = ckpt
    else:
        sd = ckpt

    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    fixed = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("classifier."):
            k2 = k2.replace("classifier.", "head.")
        if k2.startswith("head.weight"):
            k2 = k2.replace("head.weight", "head.fc.weight")
        if k2.startswith("head.bias"):
            k2 = k2.replace("head.bias", "head.fc.bias")
        fixed[k2] = v

    msd = model.state_dict()
    filtered, dropped = {}, []

    for k, v in fixed.items():
        if k in msd and v.shape == msd[k].shape:
            filtered[k] = v
        else:
            dropped.append(k)


    for head_key in list(filtered.keys()):
        if head_key.startswith("head.") or head_key.startswith("fc.") or head_key.startswith("classifier."):
            dropped.append(head_key)
            filtered.pop(head_key, None)

    print(f"Filtered {len(filtered)} keys to load; dropped {len(dropped)} keys (mismatch/cls head).")
    msg = model.load_state_dict(filtered, strict=False)
    print(f"Loaded with missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")


@dataclass
class TrainConfig:
    train_dir: str
    val_dir: str
    model_name: str = "resnet50"
    img_size: int = 224
    batch_size: int = 64
    workers: int = 8
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    opt: str = "adamw"  # or "sgd"
    momentum: float = 0.9
    label_smoothing: float = 0.0
    # dataloader
    use_balanced_sampler: bool = False
    use_rw: bool = False        
    use_cb: bool = False          
    cb_beta: float = 0.999        
    cb_loss: str = "ce"           
    cb_gamma: float = 2.0         
    use_bsm: bool = False
    use_ldam: bool=False
    ldam_margin: float=0.5
    ldam_scale: float=30.0
    mixup_alpha: float = 0.0   
    cutmix_alpha: float = 0.0  
    mix_prob: float = 1.0      
    switch_prob: float = 0.0   
    # misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "checkpoint_baseline.pth"


# -----------------------------
# Data
# -----------------------------

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf



def build_dataloaders(cfg: TrainConfig):
    train_tf, val_tf = build_transforms(cfg.img_size)
    train_set = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    val_set   = datasets.ImageFolder(cfg.val_dir,   transform=val_tf)
    num_classes = len(train_set.classes)

    device = torch.device(cfg.device)


    counts_np = count_by_class(train_set)                                   # numpy [C]
    counts_t  = torch.tensor(counts_np, dtype=torch.float32, device=device) # tensor [C]
    counts_t  = counts_t.clamp_min(1.0)                                     


    prior     = counts_t / counts_t.sum()
    log_prior = prior.log()                                                 # [C] on device


    m_raw         = counts_t.pow(-0.25)                                     # 1 / n^(1/4)
    ldam_margins  = m_raw * (cfg.ldam_margin / m_raw.max())                 


    rw_weights_per_class = None
    if cfg.use_rw:
        rw_weights_per_class = (counts_t.sum() / counts_t)                  # ∝ 1/n
        rw_weights_per_class = rw_weights_per_class / rw_weights_per_class.mean()


    cb_weights_per_class = None
    if cfg.use_cb:
        beta = cfg.cb_beta

        effective_num = (1.0 - torch.pow(torch.tensor(beta, device=device), counts_t)) / (1.0 - beta)
        cb_weights_per_class = 1.0 / effective_num
        cb_weights_per_class = cb_weights_per_class / cb_weights_per_class.mean()


    if cfg.use_balanced_sampler:
        class_weights_np = 1.0 / np.clip(counts_np, 1, None)
        class_weights_np = class_weights_np / class_weights_np.sum()
        if hasattr(train_set, "targets"):
            targets = train_set.targets
        else:
            targets = [y for _, y in train_set.samples]
        sample_weights = [class_weights_np[y] for y in targets]
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
        

    drop_last = (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=shuffle,
                              num_workers=cfg.workers, pin_memory=True, sampler=sampler,
                              drop_last=drop_last,          
                             )
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.workers, pin_memory=True,
                              drop_last=False,              
                              )


    return train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins



# -----------------------------
# Train / Eval
# -----------------------------

def train_one_epoch_baseline(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    label_smoothing: float = 0.0,
    rw_weights_per_class: Optional[torch.Tensor] = None,  
    cb_weights_per_class: Optional[torch.Tensor] = None,   
    cb_loss: str = "ce",                                    # "ce" or "focal"
    cb_gamma: float = 2.0,                                  # focal 的 gamma
    use_bsm: bool = False,
    log_prior: Optional[torch.Tensor] = None,
    use_ldam: bool = False,
    ldam_margins: Optional[torch.Tensor] = None,
    ldam_scale: float = 30.0,
    mixup_fn: Optional[Mixup] = None
):
    model.train()
    total_loss = 0.0
    
    if use_bsm:
        assert log_prior is not None, "use_bsm=True 时必须提供 log_prior（shape=[C]）"


    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)  

        logits = model(images)

        if mixup_fn is not None:

            loss = SoftTargetCrossEntropy()(logits, targets)
        else:
            if use_bsm:
                loss = F.cross_entropy(
                    logits + log_prior,     # (B,C) + (C,)
                    targets,
                    label_smoothing=label_smoothing,
                    reduction="mean"
                )
            else:
                if use_ldam:
                    assert ldam_margins is not None, "use_ldam=True 时必须提供 ldam_margins（shape=[C]）"
   
                    logits_m = logits.clone()
                    index = torch.zeros_like(logits_m, dtype=torch.bool)
                    index.scatter_(1, targets.unsqueeze(1), True)           
                    margins = ldam_margins.to(device)[targets]               # [B]
                    logits_m[index] = logits_m[index] - margins              

                    s = ldam_scale
                    per_sample_loss = F.cross_entropy(
                        s * logits_m, targets, reduction="none",
                        label_smoothing=label_smoothing
                    )  # [B]
                elif (cb_weights_per_class is not None) and (cb_loss == "focal"):
                    # CB + Focal： per-sample focal * CE
                    with torch.no_grad():
                        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
                    focal_factor = torch.pow(1.0 - pt, cb_gamma)  # [B]
                    base_ce = F.cross_entropy(
                        logits, targets, reduction="none", label_smoothing=label_smoothing
                    )  # [B]
                    per_sample_loss = focal_factor * base_ce  # [B]
                else:
                    
                    per_sample_loss = F.cross_entropy(
                        logits, targets, reduction="none", label_smoothing=label_smoothing
                    )  # [B]

  
                if cb_weights_per_class is not None:
                
                    w = cb_weights_per_class[targets]  # [B]
                    loss = (w * per_sample_loss).mean()
                elif rw_weights_per_class is not None:
                   
                    w = rw_weights_per_class[targets]  # [B]
                    loss = (w * per_sample_loss).mean()
                else:
                  
                    loss = per_sample_loss.mean()
                


        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str) -> Dict[str, float]:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # [pred, true]
    total, correct = 0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.numel()
        for p, t in zip(preds.view(-1), targets.view(-1)):
            cm[p.item(), t.item()] += 1

    # per-class stats
    per_class = {}
    eps = 1e-12
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = cm[:, c].sum().item() - tp
        fp = cm[c, :].sum().item() - tp

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)            # sensitivity
        f1 = 2 * prec * rec / (prec + rec + eps)
        acc_c = tp / (tp + fn + eps)

        per_class[c] = dict(precision=prec, recall=rec, f1=f1, acc=acc_c)

    macro_f1 = np.mean([v["f1"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])  # Macro-Sensitivity
    worst_acc = np.min([v["acc"] for v in per_class.values()])
    overall_acc = correct / total

    return dict(
        acc=overall_acc,
        macro_f1=macro_f1,
        macro_sensitivity=macro_rec,
        worst_class_acc=worst_acc
    )


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint_baseline.pth")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init", type=str, default="")
    parser.add_argument("--use-rw", action="store_true")
    parser.add_argument("--use-cb", action="store_true")
    parser.add_argument("--cb-beta", type=float, default=0.999)
    parser.add_argument("--cb-loss", type=str, default="ce", choices=["ce","focal"])
    parser.add_argument("--cb-gamma", type=float, default=2.0)
    parser.add_argument("--use-bsm", action="store_true")
    parser.add_argument("--use-ldam", action="store_true")
    parser.add_argument("--ldam-margin", type=float, default=0.5)
    parser.add_argument("--ldam-scale", type=float, default=30.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--mix-switch-prob", type=float, default=0.0)


    args = parser.parse_args()
    
    if args.use_rw and args.balanced_sampler:
        print("[WARN] 你同时启用了 RS(--balanced-sampler) 与 RW(--use-rw)。如果只想要纯 RS，请去掉 --use-rw；只想要纯 RW，请去掉 --balanced-sampler。")
    if args.use_cb and (args.use_rw or args.balanced_sampler):
        print("[WARN] CB 与 RW/RS 不建议同时启用。建议只开一个以便公平对比。")
    if args.use_ldam and (args.use_rw or args.use_cb or args.balanced_sampler or args.use_bsm):
        print("[WARN] LDAM 通常单独使用；请不要与 RS/RW/CB/BSM 同时开启。")



    cfg = TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt=args.opt,
        momentum=args.momentum,
        label_smoothing=args.label_smoothing,
        use_balanced_sampler=args.balanced_sampler,
        use_rw=args.use_rw,          # ✅ 
        use_cb=args.use_cb,
        cb_beta=args.cb_beta,
        cb_loss=args.cb_loss,
        cb_gamma=args.cb_gamma,
        seed=args.seed,
        save_path=args.save,
        use_bsm=args.use_bsm,
        use_ldam=args.use_ldam,
        ldam_margin=args.ldam_margin,
        ldam_scale=args.ldam_scale,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        mix_prob=args.mix_prob,
        switch_prob=args.mix_switch_prob,
    )
    

    use_mix = (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0)
    if use_mix and (cfg.use_rw or cfg.use_cb or cfg.use_ldam or cfg.use_bsm or cfg.use_balanced_sampler):
        print("[WARN] MixUp/CutMix 建议单独使用；将忽略 RS/RW/CB/LDAM/BSM。")
        cfg.use_balanced_sampler = False
        cfg.use_rw = cfg.use_cb = cfg.use_ldam = cfg.use_bsm = False

    set_seed(cfg.seed)


    device = cfg.device
    train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins = build_dataloaders(cfg)
    

    mixup_fn = None
    if (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0):
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mix_prob,
            switch_prob=cfg.switch_prob,
            mode="batch",
            label_smoothing=cfg.label_smoothing,  
            num_classes=num_classes,
        )


    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)

    if args.init:
        load_checkpoint_flex(model, args.init)


    if cfg.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay, nesterov=True)

    best_val = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch_baseline(
            model, train_loader, optimizer, device,
            label_smoothing=cfg.label_smoothing,
            rw_weights_per_class=(rw_weights_per_class if cfg.use_rw else None),
            cb_weights_per_class=(cb_weights_per_class if cfg.use_cb else None),
            cb_loss=cfg.cb_loss, 
            cb_gamma=cfg.cb_gamma,
            use_bsm=cfg.use_bsm,                
            log_prior=(log_prior if cfg.use_bsm else None),
            use_ldam=cfg.use_ldam,
            ldam_margins=ldam_margins,
            ldam_scale=cfg.ldam_scale,
            mixup_fn=mixup_fn,          
        )

        metrics = evaluate(model, val_loader, num_classes, device)
        log = (f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | "
               f"Acc {metrics['acc']*100:.2f}% | MacroF1 {metrics['macro_f1']*100:.2f}% | "
               f"MacroSens {metrics['macro_sensitivity']*100:.2f}% | "
               f"WorstClassAcc {metrics['worst_class_acc']*100:.2f}%")
        print(log)


        score = (metrics['macro_f1'] + metrics['macro_sensitivity']) / 2.0
        if score > best_val:
            best_val = score
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "metrics": metrics,
            }, cfg.save_path)
            print(f"✓ Saved best checkpoint to {cfg.save_path}")


if __name__ == "__main__":
    main()
