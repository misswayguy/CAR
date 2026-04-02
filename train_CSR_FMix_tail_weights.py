# -*- coding: utf-8 -*-
"""
CSR + FMix (ICLR'21) for long-tailed medical image classification
- Phase A: CLEAN soft confusion -> SVD -> gm
- Phase B: FMix (frequency-domain mask) + SoftTarget CE + fairness branch guided by gm
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler  # 放到 import timm 旁边即可

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_by_class(imagefolder_dataset):
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
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

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
        if k2.startswith("classifier."): k2 = k2.replace("classifier.", "head.")
        if k2.startswith("head.weight"): k2 = k2.replace("head.weight", "head.fc.weight")
        if k2.startswith("head.bias"):   k2 = k2.replace("head.bias", "head.fc.bias")
        fixed[k2] = v

    msd = model.state_dict()
    filtered, dropped = {}, []
    for k, v in fixed.items():
        if k in msd and v.shape == msd[k].shape: filtered[k] = v
        else: dropped.append(k)

    for head_key in list(filtered.keys()):
        if head_key.startswith(("head.","fc.","classifier.")):
            dropped.append(head_key); filtered.pop(head_key, None)

    print(f"Filtered {len(filtered)} keys; dropped {len(dropped)}.")
    msg = model.load_state_dict(filtered, strict=False)
    print(f"Loaded with missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

# -----------------------------
# Config
# -----------------------------

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
    opt: str = "adamw"
    momentum: float = 0.9
    label_smoothing: float = 0.0
    # fairness
    alpha: float = 0.3
    fair_mode: str = "kl"  # or "wce"
    # soft confusion
    tau: float = 0.0
    ema_mu: float = 0.9
    gm_update_every: int = 1
    # sampler
    use_balanced_sampler: bool = False
    # FMix
    use_fmix: bool = True
    fmix_alpha: float = 1.0     # Beta(alpha, alpha) for target area
    fmix_decay: float = 3.0     # 1/f^decay spectrum
    fmix_prob: float = 1.0
    # misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "checkpoint.pth"
    hmt: bool = False
    head_th: int = 100
    tail_th: int = 20

# -----------------------------
# Data
# -----------------------------

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

def build_dataloaders(cfg: TrainConfig):
    train_tf, val_tf = build_transforms(cfg.img_size)
    train_set = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    val_set   = datasets.ImageFolder(cfg.val_dir,   transform=val_tf)
    num_classes = len(train_set.classes)
    
    counts_np = count_by_class(train_set)   # [C] (已存在)
    
    hmt_groups = None
    if cfg.hmt:
        head = [i for i,c in enumerate(counts_np) if c > cfg.head_th]
        tail = [i for i,c in enumerate(counts_np) if c < cfg.tail_th]
        medium = [i for i,c in enumerate(counts_np) if (cfg.tail_th <= c <= cfg.head_th)]
        hmt_groups = {"head": head, "medium": medium, "tail": tail, "counts": counts_np.tolist()}
        print(f"[HMT] head={len(head)}, medium={len(medium)}, tail={len(tail)}  (th: >{cfg.head_th} / {cfg.tail_th}–{cfg.head_th} / <{cfg.tail_th})")

    if cfg.use_balanced_sampler:
        counts = count_by_class(train_set)
        class_weights = 1.0 / np.clip(counts, 1, None)
        class_weights = class_weights / class_weights.sum()
        targets = train_set.targets if hasattr(train_set, "targets") else [y for _, y in train_set.samples]
        sample_weights = [class_weights[y] for y in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=shuffle,
                              num_workers=cfg.workers, pin_memory=True, sampler=sampler, drop_last=True)
    train_loader_eval = DataLoader(datasets.ImageFolder(cfg.train_dir, transform=val_tf),
                                   batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True)

    return train_loader, train_loader_eval, val_loader, num_classes, hmt_groups

# -----------------------------
# Phase A: Soft Confusion -> gm
# -----------------------------

@torch.no_grad()
def compute_soft_confusion_matrix(model, loader, num_classes, device, tau=0.0):
    model.eval()
    S = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        probs = model(images).softmax(dim=1)
        if tau > 0:
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            probs = probs * (margin <= tau).float().unsqueeze(1)
        for cls in range(num_classes):
            idx = (targets == cls)
            if idx.any():
                S[:, cls] += probs[idx].sum(dim=0).to(torch.float64)
                counts[cls] += idx.sum()
    counts = counts.clamp_min(1.0)
    S = S / counts.unsqueeze(0)
    S.fill_diagonal_(0.0)
    return S.to(torch.float32)

def gm_from_S(S: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(S, full_matrices=False)
    u1 = U[:, 0]; v1 = Vh.transpose(-1, -2)[:, 0]
    gm = torch.outer(u1, v1)
    gm_min, gm_max = gm.min(), gm.max(); eps = 1e-8
    gm = 2.0 * (gm - gm_min) / (gm_max - gm_min + eps) + 0.01
    return gm

# -----------------------------
# FMix helpers
# -----------------------------

def fmix_masks(B: int, H: int, W: int, alpha: float = 1.0, decay: float = 3.0, device: str = "cuda"):
    """
    生成 B 个 FMix 二值 mask，形状更自然；每个 mask 的均值≈采样的 lam。
    - 过程：构造 1/f^decay 频谱 -> IFFT 得到连续图 -> 量化到阈值，保留 top-lam 部分为1
    """
    masks = []
    for _ in range(B):
        # 频域坐标
        fy = torch.fft.fftfreq(H, d=1.0).abs().unsqueeze(1).to(device)
        fx = torch.fft.fftfreq(W, d=1.0).abs().unsqueeze(0).to(device)
        f = (fx**2 + fy**2).sqrt().clamp_(min=1e-6)
        amp = (1.0 / (f ** decay))                       # 1/f^decay

        # 随机相位 + 幅值
        noise = torch.randn(H, W, device=device)
        spec = torch.fft.fft2(noise * amp)
        field = torch.fft.ifft2(spec).real               # 连续场
        field = (field - field.min()) / (field.max() - field.min() + 1e-8)

        lam = np.random.beta(alpha, alpha)
        thr = torch.quantile(field.flatten(), 1 - lam)
        m = (field >= thr).float()                       # 二值 mask，mean≈lam
        masks.append(m)
    masks = torch.stack(masks, dim=0)                    # [B,H,W]
    return masks

def one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets, num_classes=num_classes).float()

# -----------------------------
# Phase B: Train (CSR + FMix)
# -----------------------------

def soft_confusion_from_soft_targets(logits: torch.Tensor,
                                     t_soft: torch.Tensor,  # [B,C]，每样本对真类列的权重（CutMix 的软 one-hot）
                                     tau: float = 1.0) -> torch.Tensor:
    # logits: [B,C], t_soft: [B,C]，行和可以<1（常为1），列表示“属于真类 j 的权重”
    probs = F.softmax(logits / tau, dim=1)        # [B,C]
    C_num = probs.transpose(0,1) @ t_soft         # [C,C]，第 j 列累加到真类 j
    C_den = t_soft.sum(0, keepdim=True).clamp(min=1.0)  # [1,C]
    C_t = C_num / C_den                           # 列归一
    C_t.fill_diagonal_(0.0)
    return C_t

def spectral_loss_from_C(C: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    Cw = C @ torch.diag(w.to(C.device))
    _, svals, _ = torch.linalg.svd(Cw, full_matrices=False)
    return svals[0]


ce_soft = SoftTargetCrossEntropy()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_classes: int,
    w: torch.Tensor,
    cbar: Optional[torch.Tensor],
    lambda_spec: float = 0.3,
    beta: float = 0.9,
    temp: float = 1.0,
    use_fmix: bool = True,
    fmix_alpha: float = 1.0,
    fmix_decay: float = 3.0,
    fmix_prob: float = 1.0,
):
    """
    FMix + SoftTarget CE 作为主损失；
    同时用 FMix 产生的软标签 y_soft 构造“真类软指示” t_soft 来统计 C_t；
    C_bar 做 EMA；谱正则项用 || C_bar @ diag(w) ||_2。
    """
    model.train()
    total_loss = 0.0
    w = w.to(device)

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        B, C, H, W = images.shape

        # ---- FMix -> 软标签 y_soft，同时构造 t_soft（= y_soft.detach()） ----
        if use_fmix and (np.random.rand() <= fmix_prob):
            perm = torch.randperm(B, device=device)
            masks = fmix_masks(B, H, W, alpha=fmix_alpha, decay=fmix_decay, device=device)  # [B,H,W]
            masks = masks.unsqueeze(1)  # [B,1,H,W]
            x_mix = masks * images + (1.0 - masks) * images[perm]
            lam = masks.mean(dim=(1,2,3))  # [B]
            y_soft = lam.unsqueeze(1) * one_hot(targets, num_classes) + \
                     (1.0 - lam).unsqueeze(1) * one_hot(targets[perm], num_classes)

            logits = model(x_mix)
            loss_ce = ce_soft(logits, y_soft)
            t_soft = y_soft.detach()
        else:
            logits = model(images)
            # 非 FMix 时，用硬标签 one-hot 当软标签
            y_onehot = one_hot(targets, num_classes)
            loss_ce = ce_soft(logits, y_onehot)
            t_soft = y_onehot.detach()

        # ---- 可导软混淆 C_t（带温度） + EMA 累计 C_bar ----
        C_t = soft_confusion_from_soft_targets(logits, t_soft, tau=temp)
        if cbar is None:
            cbar = C_t.detach()
        else:
            cbar = beta * cbar.detach() + (1.0 - beta) * C_t

        # ---- 频次加权的谱正则 || C_bar @ diag(w) ||_2 ----
        spec = spectral_loss_from_C(cbar, w)

        # ---- 总损失：CE + λ_spec * spec ----
        loss = loss_ce + lambda_spec * spec

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset), cbar


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str,
             groups: Optional[dict] = None) -> Dict[str, float]:
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

    # per-class
    eps = 1e-12
    per_class = {}
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = cm[:, c].sum().item() - tp
        fp = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        acc_c = tp / (tp + fn + eps)   # class-accuracy（=recall）
        per_class[c] = dict(precision=prec, recall=rec, f1=f1, acc=acc_c)

    macro_f1  = np.mean([v["f1"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])
    worst_acc = np.min([v["acc"] for v in per_class.values()])
    overall_acc = correct / total

    out = dict(acc=overall_acc, macro_f1=macro_f1,
               macro_sensitivity=macro_rec, worst_class_acc=worst_acc)

    # ---- Group metrics (H/M/T) ----
    if groups is not None:
        def group_macro(vkey, idxs):
            if len(idxs)==0: return float("nan")
            return float(np.mean([per_class[i][vkey] for i in idxs]))
        def group_overall_acc(idxs):
            if len(idxs)==0: return float("nan")
            tp = sum(int(cm[i, i].item()) for i in idxs)
            tot = sum(int(cm[:, i].sum().item()) for i in idxs)
            return tp / (tot + eps)

        for tag, idxs in [("H", groups["head"]), ("M", groups["medium"]), ("T", groups["tail"])]:
            out[f"{tag}_acc"]  = group_overall_acc(idxs)
            out[f"{tag}_f1"]   = group_macro("f1", idxs)
            out[f"{tag}_sens"] = group_macro("recall", idxs)
            out[f"{tag}_wca"]  = float(np.min([per_class[i]["acc"] for i in idxs])) if idxs else float("nan")

    return out


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
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw","sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--fair-mode", type=str, default="kl", choices=["kl","wce"])
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--ema-mu", type=float, default=0.9)
    parser.add_argument("--gm-update-every", type=int, default=1)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint.pth")

    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init", type=str, default="")

    # FMix args
    parser.add_argument("--no-fmix", action="store_true", help="disable FMix (use pure CSR)")
    parser.add_argument("--fmix-alpha", type=float, default=1.0)
    parser.add_argument("--fmix-decay", type=float, default=3.0)
    parser.add_argument("--fmix-prob", type=float, default=1.0)
    parser.add_argument("--hmt", action="store_true", help="按每类样本数统计 Head/Medium/Tail 指标（基于 train 计数）")
    parser.add_argument("--head-th", type=int, default=100, help="Head 阈值：> head_th")
    parser.add_argument("--tail-th", type=int, default=20,  help="Tail 阈值：< tail_th")
    parser.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--spec-lambda", type=float, default=0.3, help="λ for spectral regularization")
    parser.add_argument("--spec-beta",   type=float, default=0.9, help="EMA β for running soft confusion C_bar")
    parser.add_argument("--spec-temp",   type=float, default=1.0, help="temperature τ in softmax(z/τ) for C_t")
    parser.add_argument("--lambda0",     type=float, default=10.0, help="λ0 in w_j = 1/sqrt(m_j + λ0)")


    args = parser.parse_args()

    cfg = TrainConfig(
        train_dir=args.train_dir, val_dir=args.val_dir, model_name=args.model,
        img_size=args.img_size, batch_size=args.batch_size, workers=args.workers,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        opt=args.opt, momentum=args.momentum, label_smoothing=args.label_smoothing,
        alpha=args.alpha, fair_mode=args.fair_mode, tau=args.tau, ema_mu=args.ema_mu,
        gm_update_every=args.gm_update_every, use_balanced_sampler=args.balanced_sampler,
        seed=args.seed, save_path=args.save,
        use_fmix=(not args.no_fmix), fmix_alpha=args.fmix_alpha, fmix_decay=args.fmix_decay, fmix_prob=args.fmix_prob,
        hmt=args.hmt, head_th=args.head_th, tail_th=args.tail_th,
    )

    set_seed(cfg.seed)
    device = cfg.device

    train_loader, train_loader_eval, val_loader, num_classes, hmt_groups = build_dataloaders(cfg)
    
    # === class-frequency weights: w_j = 1/sqrt(m_j + λ0) ===
    counts_np = count_by_class(train_loader.dataset)         # [C]
    counts_t  = torch.tensor(counts_np, dtype=torch.float32) # on CPU
    w = 1.0 / torch.sqrt(counts_t + args.lambda0)
    w = (w / w.mean()).clamp(min=1e-3)  # 可留作数值稳定


    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)
    if args.init:
        load_checkpoint_flex(model, args.init)

    if cfg.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay, nesterov=True)
        
    # === NEW: LR Scheduler ===
    scheduler = None
    if args.sched == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.epochs,            # 以 epoch 为单位的总长度
            lr_min=args.min_lr,              # 余弦最低学习率
            warmup_t=args.warmup_epochs,     # warmup 的 epoch 数
            warmup_lr_init=max(args.lr * 0.01, 1e-7),  # warmup 起始 lr
            k_decay=1.0,
        )

    S_ema: Optional[torch.Tensor] = None
    gm: Optional[torch.Tensor] = None
    C_bar: Optional[torch.Tensor] = None

    best_val = -1.0

    for epoch in range(1, cfg.epochs + 1):
        # Phase A: CLEAN -> gm
        # if (epoch % cfg.gm_update_every == 1) or (gm is None):
        #     S_cur = compute_soft_confusion_matrix(model, train_loader_eval, num_classes, device, tau=cfg.tau)
        #     S_ema = S_cur if S_ema is None else cfg.ema_mu * S_ema + (1.0 - cfg.ema_mu) * S_cur
        #     gm = gm_from_S(S_ema)

        # Phase B: CSR + FMix
        train_loss, C_bar = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=num_classes, w=w, cbar=C_bar,
            lambda_spec=args.spec_lambda, beta=args.spec_beta, temp=args.spec_temp,
            use_fmix=cfg.use_fmix, fmix_alpha=cfg.fmix_alpha, fmix_decay=cfg.fmix_decay, fmix_prob=cfg.fmix_prob
        )


        metrics = evaluate(model, val_loader, num_classes, device, groups=(hmt_groups if cfg.hmt else None))

        log = (f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | "
            f"Acc {metrics['acc']*100:.2f}% | MacroF1 {metrics['macro_f1']*100:.2f}% | "
            f"MacroSens {metrics['macro_sensitivity']*100:.2f}% | "
            f"WorstClassAcc {metrics['worst_class_acc']*100:.2f}%")

        if cfg.hmt:
            log += (f" || H Acc/F1/Sens/WCA: {metrics['H_acc']*100:.2f}/{metrics['H_f1']*100:.2f}/"
                    f"{metrics['H_sens']*100:.2f}/{metrics['H_wca']*100:.2f} | "
                    f"M {metrics['M_acc']*100:.2f}/{metrics['M_f1']*100:.2f}/{metrics['M_sens']*100:.2f}/{metrics['M_wca']*100:.2f} | "
                    f"T {metrics['T_acc']*100:.2f}/{metrics['T_f1']*100:.2f}/{metrics['T_sens']*100:.2f}/{metrics['T_wca']*100:.2f}")
        print(log)
        
        if scheduler is not None:
            # timm 的 step 接受 epoch；纯 torch 用 scheduler.step()
            scheduler.step(epoch)

        score = (metrics['macro_f1'] + metrics['macro_sensitivity']) / 2.0
        if score > best_val:
            best_val = score
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "epoch": epoch, "metrics": metrics}, cfg.save_path)
            print(f"✓ Saved best checkpoint to {cfg.save_path}")

if __name__ == "__main__":
    main()
