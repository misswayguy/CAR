# -*- coding: utf-8 -*-
"""
CSR + CutMix for long-tailed medical image classification
- Phase A: build soft confusion matrix on CLEAN data -> SVD -> gm
- Phase B: CutMix + CE(soft) + fairness branch (KL/WCE) guided by gm
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
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str):
    """
    安全加载本地预训练：自动剔除分类头、以及所有“形状不匹配”的键；
    同时兼容常见命名差异（head/head.fc/classifier），并去掉 DataParallel 前缀。
    """
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

    print(f"Filtered {len(filtered)} keys to load; dropped {len(dropped)} keys.")
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
    opt: str = "adamw"  # or "sgd"
    momentum: float = 0.9
    label_smoothing: float = 0.0
    # fairness branch
    alpha: float = 0.3       # weight for fairness loss
    fair_mode: str = "kl"    # "kl" or "wce"
    # soft confusion matrix options
    tau: float = 0.0         # only accumulate samples with (top1-top2) <= tau
    ema_mu: float = 0.9
    gm_update_every: int = 1
    # sampler
    use_balanced_sampler: bool = False
    # cutmix
    use_cutmix: bool = True
    cutmix_beta: float = 1.0     # Beta(beta, beta)
    cutmix_prob: float = 1.0     # prob to apply cutmix per batch
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
        if hasattr(train_set, "targets"):
            targets = train_set.targets
        else:
            targets = [y for _, y in train_set.samples]
        sample_weights = [class_weights[y] for y in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=shuffle,
                              num_workers=cfg.workers, pin_memory=True, sampler=sampler, drop_last=True)
    # eval on CLEAN transform
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
def compute_soft_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str,
    tau: float = 0.0,
) -> torch.Tensor:
    """
    软混淆矩阵 S \in R^{C x C}（列=真类j，行为softmax预测到各类的概率之和）
    - 仅在 CLEAN 样本上统计；可选 margin 过滤聚焦易混淆样本；列归一；对角清零。
    """
    model.eval()
    S = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = logits.softmax(dim=1)

        if tau > 0:
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            mask = (margin <= tau).float().unsqueeze(1)
            probs = probs * mask

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
    U, Svals, Vh = torch.linalg.svd(S, full_matrices=False)
    u1 = U[:, 0]
    v1 = Vh.transpose(-1, -2)[:, 0]
    gm = torch.outer(u1, v1)  # [C, C]
    gm_min, gm_max = gm.min(), gm.max()
    eps = 1e-8
    gm = 2.0 * (gm - gm_min) / (gm_max - gm_min + eps) + 0.01  # ~ [0.01, 2.01]
    return gm

# -----------------------------
# CutMix helpers
# -----------------------------

def one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets, num_classes=num_classes).float()

def rand_bbox(W: int, H: int, lam: float):
    """随机生成 CutMix 矩形框 (x1, y1, x2, y2)。"""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # 中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_aug(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    beta: float,
    apply_prob: float,
):
    """
    CutMix：矩形贴补，返回 (images_mix, y_mix, y_a, y_b, lam)
    - 若未触发（概率）或 beta<=0，则返回原图 + onehot 标签。
    """
    B, C, H, W = images.size()
    device = images.device

    if (np.random.rand() > apply_prob) or (beta <= 0.0):
        y = one_hot(targets, num_classes)
        lam = torch.tensor(1.0, device=device)
        return images, y, targets, targets, lam

    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(B, device=device)

    y_a = targets
    y_b = targets[rand_index]

    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    # 贴 patch
    images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    # 修正 lam 为实际区域比例
    lam_eff = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    lam = torch.tensor(lam_eff, dtype=torch.float32, device=device)

    # 软标签
    y_onehot_a = one_hot(y_a, num_classes)
    y_onehot_b = one_hot(y_b, num_classes)
    y_mix = lam * y_onehot_a + (1.0 - lam) * y_onehot_b
    return images, y_mix, y_a, y_b, lam

# -----------------------------
# Phase B: Train one epoch (CSR + CutMix)
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

def train_one_epoch(model, loader, optimizer, device,
                    num_classes: int, w: torch.Tensor,
                    cbar: Optional[torch.Tensor],
                    lambda_spec: float = 0.3, beta: float = 0.9, temp: float = 1.0,
                    use_cutmix: bool = True, cutmix_beta: float = 1.0, cutmix_prob: float = 1.0):
    model.train()
    total_loss = 0.0
    w = w.to(device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # —— CutMix -> 软标签 y_mix（作为 CE 标签），同时构造 “真类软指示” t_soft —— #
        if use_cutmix:
            # 你已有的 cutmix_aug，可直接复用
            images_cm, y_mix, y_a, y_b, lam = cutmix_aug(
                images, targets, num_classes, beta=cutmix_beta, apply_prob=cutmix_prob
            )
            logits = model(images_cm)
            loss_ce = ce_soft(logits, y_mix)
            t_soft = y_mix.detach()   # 作为 “属于真类列 j 的权重”，参与 \tilde C_t 的统计
        else:
            logits = model(images)
            y_onehot = F.one_hot(targets, num_classes=num_classes).float()
            loss_ce = ce_soft(logits, y_onehot)
            t_soft = y_onehot.detach()

        # —— 可导软混淆矩阵（带温度）+ EMA —— #
        C_t = soft_confusion_from_soft_targets(logits, t_soft, tau=temp)
        if cbar is None:
            cbar = C_t.detach()
        else:
            cbar = beta * cbar.detach() + (1.0 - beta) * C_t

        # —— 频率加权谱正则 —— #
        spec = spectral_loss_from_C(cbar, w)
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

    # offline pretrained
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--init", type=str, default="")

    # CutMix args
    parser.add_argument("--no-cutmix", action="store_true", help="disable CutMix (use pure CSR)")
    parser.add_argument("--cutmix-beta", type=float, default=1.0)
    parser.add_argument("--cutmix-prob", type=float, default=1.0)
    parser.add_argument("--hmt", action="store_true", help="按每类样本数统计 Head/Medium/Tail 指标（基于 train 计数）")
    parser.add_argument("--head-th", type=int, default=100, help="Head 阈值：> head_th")
    parser.add_argument("--tail-th", type=int, default=20,  help="Tail 阈值：< tail_th")
    parser.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--spec-lambda", type=float, default=0.3, help="λ for spectral regularization")
    parser.add_argument("--spec-beta",   type=float, default=0.9, help="EMA β for running soft confusion C_bar")
    parser.add_argument("--spec-temp",   type=float, default=1.0, help="temperature τ in softmax(z/τ)")
    parser.add_argument("--lambda0",     type=float, default=10.0, help="λ0 in w_j = 1/sqrt(m_j + λ0)")





    args = parser.parse_args()

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
        alpha=args.alpha,
        fair_mode=args.fair_mode,
        tau=args.tau,
        ema_mu=args.ema_mu,
        gm_update_every=args.gm_update_every,
        use_balanced_sampler=args.balanced_sampler,
        seed=args.seed,
        save_path=args.save,
        use_cutmix=(not args.no_cutmix),
        cutmix_beta=args.cutmix_beta,
        cutmix_prob=args.cutmix_prob,
        hmt=args.hmt, head_th=args.head_th, tail_th=args.tail_th,
    )

    set_seed(cfg.seed)
    device = cfg.device

    train_loader, train_loader_eval, val_loader, num_classes, hmt_groups = build_dataloaders(cfg)
    
    # === class-frequency weights: w_j = 1/sqrt(m_j + λ0) ===
    counts_np = count_by_class(train_loader.dataset)         # [C]
    counts_t  = torch.tensor(counts_np, dtype=torch.float32) # on CPU
    w = 1.0 / torch.sqrt(counts_t + args.lambda0)
    w = (w / w.mean()).clamp(min=1e-3)  # 可保留，防极端尺度


    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)

    if args.init:
        load_checkpoint_flex(model, args.init)

    if cfg.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=args.momentum,
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
    
    # 新 CSR_weights 的“持续”软混淆 EMA
    C_bar: Optional[torch.Tensor] = None

    best_val = -1.0

    for epoch in range(1, cfg.epochs + 1):
        # ---------- Phase A: CLEAN stats -> gm ----------
        # if (epoch % cfg.gm_update_every == 1) or (gm is None):
        #     S_cur = compute_soft_confusion_matrix(
        #         model, train_loader_eval, num_classes, device, tau=cfg.tau
        #     )
        #     if S_ema is None:
        #         S_ema = S_cur
        #     else:
        #         S_ema = cfg.ema_mu * S_ema + (1.0 - cfg.ema_mu) * S_cur
        #     gm = gm_from_S(S_ema)

        # ---------- Phase B: CSR + CutMix ----------
        # ---------- Phase B: CSR_weights + CutMix ----------
        train_loss, C_bar = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=num_classes, w=w, cbar=C_bar,
            lambda_spec=args.spec_lambda, beta=args.spec_beta, temp=args.spec_temp,
            use_cutmix=cfg.use_cutmix, cutmix_beta=cfg.cutmix_beta, cutmix_prob=cfg.cutmix_prob,
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
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "metrics": metrics,
            }, cfg.save_path)
            print(f"✓ Saved best checkpoint to {cfg.save_path}")

if __name__ == "__main__":
    main()
