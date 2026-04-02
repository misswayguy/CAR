# -*- coding: utf-8 -*-
"""
Clean-training with Confusional Spectral Regularization (non-adversarial version)
- Works on medical long-tailed datasets with pre-split train/val folders.
- Backbones from timm (resnet / swin / convnext / vit ...).
- Phase A: build a "soft confusion matrix" on clean data -> SVD -> gm (direction to shrink spectral norm).
- Phase B: standard CE + fairness branch (KL or weighted-CE) guided by gm.
- Metrics: Accuracy, Macro-F1, Macro-Sensitivity(Recall), Worst-class Accuracy.
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
from collections import defaultdict

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
    # torchvision>=0.13 keeps targets in .targets, older in .samples
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

    # 取 state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: sd = ckpt["state_dict"]
        elif "model" in ckpt:    sd = ckpt["model"]
        else:                    sd = ckpt
    else:
        sd = ckpt

    # 去掉 DataParallel 前缀
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # 规范一些常见的分类头命名
    fixed = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("classifier."):            # HF/timm旧版
            k2 = k2.replace("classifier.", "head.")
        if k2.startswith("head.weight"):            # 有的实现是 head.weight/bias
            k2 = k2.replace("head.weight", "head.fc.weight")
        if k2.startswith("head.bias"):
            k2 = k2.replace("head.bias", "head.fc.bias")
        fixed[k2] = v

    msd = model.state_dict()
    filtered = {}
    dropped = []

    # 只保留：1) 模型里存在的键；2) 形状完全一致
    for k, v in fixed.items():
        if k in msd and v.shape == msd[k].shape:
            filtered[k] = v
        else:
            dropped.append(k)

    # 额外强制丢弃分类头（即使形状凑巧一致也不加载）
    for head_key in list(filtered.keys()):
        if head_key.startswith("head.") or head_key.startswith("fc.") or head_key.startswith("classifier."):
            # 保守：训练时让分类头按当前num_classes随机初始化
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
    # fairness branch
    alpha: float = 0.3       # weight for fairness loss
    fair_mode: str = "kl"    # "kl" or "wce"
    # soft confusion matrix options
    tau: float = 0.0         # margin filter (0=off). Only use samples with top1-top2 <= tau when accumulating S
    ema_mu: float = 0.9      # EMA for S across epochs
    gm_update_every: int = 1 # update gm every N epochs
    # sampler
    use_balanced_sampler: bool = False
    # misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "checkpoint.pth"

# -----------------------------
# Data
# -----------------------------

def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # normalize to ImageNet stats (works well for timm pretrained)
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

    if cfg.use_balanced_sampler:
        counts = count_by_class(train_set)
        # inverse frequency as sampling weight
        class_weights = 1.0 / np.clip(counts, 1, None)
        # normalize
        class_weights = class_weights / class_weights.sum()
        # expand to per-sample weight
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
                              num_workers=cfg.workers, pin_memory=True, sampler=sampler)
    train_loader_eval = DataLoader(datasets.ImageFolder(cfg.train_dir, transform=val_tf),
                                   batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True)

    return train_loader, train_loader_eval, val_loader, num_classes

def make_lts_splits_from_counts(counts: np.ndarray, head_thr: int = 700, tail_thr: int = 70):
    """根据训练集每类样本数做长尾分组：Head(>head_thr), Tail(<tail_thr), 其余为 Medium。"""
    head = [i for i, c in enumerate(counts) if c > head_thr]
    tail = [i for i, c in enumerate(counts) if c < tail_thr]
    medium = [i for i in range(len(counts)) if i not in head and i not in tail]
    return {"head": head, "medium": medium, "tail": tail}

def summarize_group_metrics(cm: torch.Tensor, per_class: Dict[int, Dict], idxs: list):
    """
    用评估阶段得到的混淆矩阵(cm: [pred,true])和逐类指标(per_class)，汇总一个分组的：
    - Acc（样本加权的整体准确率）
    - Macro-F1（组内类别平均）
    - Macro-Sensitivity（组内类别平均）
    """
    if len(idxs) == 0:
        return dict(acc=float("nan"), macro_f1=float("nan"),
                    macro_sensitivity=float("nan"), num_classes=0)

    # 该分组的总样本 = 该分组各真类在 cm 的列和
    total = int(cm[:, idxs].sum().item())
    correct = int(sum(cm[c, c].item() for c in idxs))
    acc = correct / (total + 1e-12)

    macro_f1 = float(np.mean([per_class[c]["f1"] for c in idxs]))
    macro_sens = float(np.mean([per_class[c]["recall"] for c in idxs]))

    return dict(acc=acc, macro_f1=macro_f1,
                macro_sensitivity=macro_sens, num_classes=len(idxs))



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
    构造“软混淆矩阵” S \in R^{C x C} （列=真类j，行为softmax预测到各类的概率之和）
    - 与论文不同：我们用干净样本，不做对抗
    - 做法：对每个样本的 softmax p，把 p 累加到第 j 列（真类 j）
    - 可选：只在小 margin (top1 - top2 <= tau) 的样本上累加，更聚焦“易混淆”
    - 对角线清零（只关心错分）；最后对每列做归一（消除长尾样本数影响）
    """
    model.eval()
    S = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)  # 用双精度累计更稳
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = logits.softmax(dim=1)  # [B, C]

        if tau > 0:
            top2 = torch.topk(probs, k=2, dim=1).values  # [B,2]
            margin = top2[:, 0] - top2[:, 1]
            mask = (margin <= tau).float().unsqueeze(1)  # [B,1]
            probs = probs * mask  # 只累计易混淆样本；其余样本相当于贡献为0

        # 按列累加：S[:, j] += p
        for cls in range(num_classes):
            idx = (targets == cls)
            if idx.any():
                S[:, cls] += probs[idx].sum(dim=0).to(torch.float64)
                counts[cls] += idx.sum()

    # 列归一（避免长尾导致列值偏大）
    counts = counts.clamp_min(1.0)  # 防除0
    S = S / counts.unsqueeze(0)     # 每列除以该真类样本数

    # 只关心错分：对角清零
    S.fill_diagonal_(0.0)

    return S.to(torch.float32)

def gm_from_S(S: torch.Tensor) -> torch.Tensor:
    """
    从软混淆矩阵 S 提取“谱范数下降方向” gm：
    - SVD: S = U diag(s) V^T
    - 取首左右奇异向量 u[:,0], v[:,0]，gm = u v^T
    - min-max 归一到正区间，作为“(预测i, 真类j)”的权重
    """
    # torch.linalg.svd 更稳；full_matrices=False 更快
    U, Svals, Vh = torch.linalg.svd(S, full_matrices=False)
    u1 = U[:, 0]
    v1 = Vh.transpose(-1, -2)[:, 0]  # Vh是V^T
    gm = torch.outer(u1, v1)         # [C, C]

    # ######################################### compute gradients for the spectral norm ##### the first term in (11) ######################################
    # 这里的 gm ≈ 谱范数 ||S||_2 的梯度方向（当最大奇异值主导时，∇||S||_2 ≈ u1 v1^T）
    # 我们把它 min-max 到正区间，作为“(预测类 i, 真类 j)”的样本权重：
    gm_min, gm_max = gm.min(), gm.max()
    eps = 1e-8
    gm = 2.0 * (gm - gm_min) / (gm_max - gm_min + eps) + 0.01  # ≈ [0.01, 2.01]
    # #####################################################################################################################################################

    return gm

# -----------------------------
# Phase B: Train one epoch
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    gm: torch.Tensor,
    alpha: float = 0.3,
    fair_mode: str = "kl",
    label_smoothing: float = 0.0,
):
    model.train()
    gm = gm.to(device)
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)  # [B, C]

        # 标准 CE（可带 label smoothing）
        loss_ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction="mean")

        # --- 公平分支：沿 gm 方向罚“最强错分通道” ---
        with torch.no_grad():
            preds = logits.argmax(dim=1)  # [B]
            weights = gm[preds, targets]  # [B]，每个样本一个权重

        if fair_mode == "wce":
            # Weighted Cross-Entropy：最直观
            per_sample_ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)  # [B]
            loss_fair = (weights * per_sample_ce).mean()
        else:
            # KL soft-target：更贴论文“可微替身”的做法（只在真类位置施压，并按weights缩放）
            yy = F.one_hot(targets, num_classes=logits.size(1)).float()  # [B, C]
            yy = yy * weights.unsqueeze(1)                               # 只放大真类维度
            loss_fair = F.kl_div(F.log_softmax(logits, dim=1), yy, reduction="batchmean")

        loss = (1.0 - alpha) * loss_ce + alpha * loss_fair

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

# -----------------------------
# Evaluation
# -----------------------------

# @torch.no_grad()
# def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str) -> Dict[str, float]:
#     model.eval()
#     cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # [pred, true]
#     total, correct = 0, 0
#     for images, targets in loader:
#         images, targets = images.to(device), targets.to(device)
#         logits = model(images)
#         preds = logits.argmax(dim=1)
#         correct += (preds == targets).sum().item()
#         total += targets.numel()
#         for p, t in zip(preds.view(-1), targets.view(-1)):
#             cm[p.item(), t.item()] += 1

#     # per-class stats
#     per_class = {}
#     eps = 1e-12
#     for c in range(num_classes):
#         tp = cm[c, c].item()
#         fn = cm[:, c].sum().item() - tp
#         fp = cm[c, :].sum().item() - tp
#         tn = cm.sum().item() - tp - fn - fp

#         prec = tp / (tp + fp + eps)
#         rec  = tp / (tp + fn + eps)  # sensitivity
#         f1   = 2 * prec * rec / (prec + rec + eps)
#         acc_c = tp / (tp + fn + eps)

#         per_class[c] = dict(precision=prec, recall=rec, f1=f1, acc=acc_c)

#     macro_f1  = np.mean([v["f1"] for v in per_class.values()])
#     macro_rec = np.mean([v["recall"] for v in per_class.values()])  # Macro-Sensitivity
#     worst_acc = np.min([v["acc"] for v in per_class.values()])
#     overall_acc = correct / total

#     return dict(
#         acc=overall_acc,
#         macro_f1=macro_f1,
#         macro_sensitivity=macro_rec,
#         worst_class_acc=worst_acc
#     )
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str,
             group_splits: Optional[Dict[str, list]] = None) -> Dict[str, float]:
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # [pred, true]
    total, correct = 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
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
        tn = cm.sum().item() - tp - fn - fp

        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)  # sensitivity
        f1   = 2 * prec * rec / (prec + rec + eps)
        acc_c = tp / (tp + fn + eps)

        per_class[c] = dict(precision=prec, recall=rec, f1=f1, acc=acc_c)

    macro_f1  = np.mean([v["f1"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])  # Macro-Sensitivity
    worst_acc = np.min([v["acc"] for v in per_class.values()])
    overall_acc = correct / total

    out = dict(
        acc=overall_acc,
        macro_f1=macro_f1,
        macro_sensitivity=macro_rec,
        worst_class_acc=worst_acc
    )

    # 组内指标
    if group_splits is not None:
        groups = {}
        for name, idxs in group_splits.items():
            groups[name] = summarize_group_metrics(cm, per_class, idxs)
        out["groups"] = groups

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

    # —— 仅为“加载本地权重 / 禁止联网预训练”新增的两个参数 ——
    parser.add_argument("--no-pretrained", action="store_true",
                        help="不从网络下载 timm 预训练（离线时必须开）")
    parser.add_argument("--init", type=str, default="",
                        help="本地预训练权重路径（.pth/.bin），与 --no-pretrained 搭配使用")
    parser.add_argument("--head-threshold", type=int, default=100,
                    help="Head 分组阈值：训练集中该类样本数 > 此值 视为 Head")
    parser.add_argument("--tail-threshold", type=int, default=20,
                    help="Tail 分组阈值：训练集中该类样本数 < 此值 视为 Tail")


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
    )

    set_seed(cfg.seed)

    device = cfg.device
    train_loader, train_loader_eval, val_loader, num_classes = build_dataloaders(cfg)
    
    # —— 基于“训练集频次”做 Head/Medium/Tail 分组（FoPro-KD 的做法）——
    tmp_train_for_count = datasets.ImageFolder(cfg.train_dir)  # 仅用于计数与类名
    counts = count_by_class(tmp_train_for_count)               # 每类训练样本数
    class_names = tmp_train_for_count.classes

    splits = make_lts_splits_from_counts(
        counts, head_thr=args.head_threshold, tail_thr=args.tail_threshold
    )
    # 打印一次分组信息（类数与样本占比），便于核对
    def _sum_samples(idxs): return int(np.sum(counts[idxs])) if len(idxs) else 0
    print(f"[LTS split] Head>( {args.head_threshold} ), Tail<( {args.tail_threshold} )")
    for k in ["head","medium","tail"]:
        print(f"  - {k.capitalize():6s}: {len(splits[k]):2d} classes, "
              f"{_sum_samples(splits[k])} / {int(counts.sum())} samples")

    # —— 这里仅改一处：是否联网下载预训练 —— 
    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)

    # —— 若提供本地权重，则加载 —— 
    if args.init:
        load_checkpoint_flex(model, args.init)

    # Optimizer
    if cfg.opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay, nesterov=True)

    # 初始化软混淆矩阵的 EMA
    S_ema: Optional[torch.Tensor] = None
    gm: Optional[torch.Tensor] = None

    best_val = -1.0
    for epoch in range(1, cfg.epochs + 1):
        # ---------------- Phase A: 用干净样本估计软混淆矩阵 S，并得到 gm ----------------
        if (epoch % cfg.gm_update_every == 1) or (gm is None):
        # if (epoch == 1) or (((epoch - 1) % cfg.gm_update_every) == 0):
            S_cur = compute_soft_confusion_matrix(
                model, train_loader_eval, num_classes, device, tau=cfg.tau
            )
            if S_ema is None:
                S_ema = S_cur
            else:
                # EMA 跨epoch累计，平滑抖动
                S_ema = cfg.ema_mu * S_ema + (1.0 - cfg.ema_mu) * S_cur
            gm = gm_from_S(S_ema)

        # ---------------- Phase B: 沿谱范数方向训练（CE + 公平分支） ----------------
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            gm=gm, alpha=cfg.alpha, fair_mode=cfg.fair_mode,
            label_smoothing=cfg.label_smoothing
        )

        # ---------------- Evaluation ----------------
        # metrics = evaluate(model, val_loader, num_classes, device)
        
                # log = (f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | "
        #        f"Acc {metrics['acc']*100:.2f}% | MacroF1 {metrics['macro_f1']*100:.2f}% | "
        #        f"MacroSens {metrics['macro_sensitivity']*100:.2f}% | "
        #        f"WorstClassAcc {metrics['worst_class_acc']*100:.2f}%")
        # print(log)
        
        metrics = evaluate(model, val_loader, num_classes, device, group_splits=splits)
        
        log = (f"Epoch {epoch:03d} | "
               f"ALL Acc {metrics['acc']*100:.2f}% | "
               f"ALL MacroF1 {metrics['macro_f1']*100:.2f}% | "
               f"ALL MacroSens {metrics['macro_sensitivity']*100:.2f}%")
        print(log)

        if "groups" in metrics:
            g = metrics["groups"]
            def _fmt(m): 
                return f"Acc {m['acc']*100:.2f}% | F1 {m['macro_f1']*100:.2f}% | Sens {m['macro_sensitivity']*100:.2f}%"
            print("  Head  :", _fmt(g["head"]))
            print("  Medium:", _fmt(g["medium"]))
            print("  Tail  :", _fmt(g["tail"]))


        # Save best by MacroF1（也可按 WorstClassAcc）
        score = (metrics['macro_f1'] + metrics['macro_sensitivity']) / 2.0
        if score > best_val:
            best_val = score
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "metrics": metrics,
                "class_names": class_names,  # 方便复现实验
                "counts": counts,
            }, cfg.save_path)
            print(f"✓ Saved best checkpoint to {cfg.save_path}")

if __name__ == "__main__":
    main()
