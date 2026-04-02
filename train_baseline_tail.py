# -*- coding: utf-8 -*-
"""
Baseline fine-tuning (no CSR)
- Works on medical long-tailed datasets with pre-split train/val folders.
- Backbones from timm (resnet / swin / convnext / vit ...).
- Loss: standard CrossEntropy (optionally with label smoothing).
- Metrics: Accuracy, Macro-F1, Macro-Sensitivity(Recall), Worst-class Accuracy.
"""

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

from timm.scheduler import CosineLRScheduler  # 放到 import timm 旁边即可
import time

# -----------------------------
# Utils
# -----------------------------
# ==== ACL 所需辅助模块 ====
class ProjectionHead(nn.Module):
    """MLP投影头，用于提取对比特征"""
    def __init__(self, in_dim=768, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

def get_classifier_module(model: nn.Module) -> nn.Module:
    """兼容 timm：优先 get_classifier；否则尝试 head/fc/classifier。"""
    if hasattr(model, "get_classifier"):
        clf = model.get_classifier()
        if clf is not None:
            return clf
    for name in ["head", "fc", "classifier"]:
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Module):
            return getattr(model, name)
    raise RuntimeError("Cannot find classifier module (head/fc/classifier).")

@torch.no_grad()
def wb_renorm_classifier(clf: nn.Module, target_norm: float = None, alpha: float = 1.0, max_norm: float = None, eps: float = 1e-12):
    """
    将分类器权重重整到统一尺度：
    - 若给 max_norm：先把每行权重的L2范数截到 max_norm；
    - 再做“软等范数”重整：w_c <- w_c * (target_norm / ||w_c||)^alpha  (alpha∈[0,1]).
    """
    if not hasattr(clf, "weight"):
        return
    W = clf.weight  # [C, D]
    if W is None:
        return
    # max-norm（硬约束）
    if max_norm is not None:
        norms = W.detach().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(eps)  # [C,1]
        scale = (max_norm / norms).clamp(max=1.0)
        W.mul_(scale)
    # “软等范数”重整
    if target_norm is not None:
        norms = W.detach().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(eps)
        scale = (float(target_norm) / norms).pow(alpha)  # [C,1]
        W.mul_(scale)


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


def load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str, keep_head: bool = False):
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

    # 规范一些常见的分类头命名（把 ckpt 的键名映射到 timm 常见命名）
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

    # 只加载形状匹配的键
    for k, v in fixed.items():
        if k in msd and v.shape == msd[k].shape:
            filtered[k] = v
        else:
            dropped.append(k)

    # ⚠️ 只在“不需要旧头”时才丢弃分类头；GML 第二阶段要 keep_head=True
    if not keep_head:
        for head_key in list(filtered.keys()):
            if head_key.startswith(("head.", "fc.", "classifier.")):
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
    use_rw: bool = False        # ✅ 补上这个
    use_cb: bool = False          # ✅
    cb_beta: float = 0.999        # ✅
    cb_loss: str = "ce"           # ✅
    cb_gamma: float = 2.0         # ✅
    use_bsm: bool = False
    use_ldam: bool=False
    ldam_margin: float=0.5
    ldam_scale: float=30.0
    mixup_alpha: float = 0.0   # 典型 0.2
    cutmix_alpha: float = 0.0  # 典型 1.0
    mix_prob: float = 1.0      # 进行 mix 的概率
    switch_prob: float = 0.0   # MixUp 与 CutMix 切换概率（两者都>0时才有用）
    # misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "checkpoint_baseline.pth"
    hmt: bool = False
    head_th: int = 100
    tail_th: int = 20
    gml: bool = False
    gml_epochs: int = 10
    ensemble: bool = False
    t_old: float = 1.0
    t_new: float = 1.0



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
    
    counts_np = count_by_class(train_set)   # [C] (已存在)
    
    hmt_groups = None
    if cfg.hmt:
        head = [i for i,c in enumerate(counts_np) if c > cfg.head_th]
        tail = [i for i,c in enumerate(counts_np) if c < cfg.tail_th]
        medium = [i for i,c in enumerate(counts_np) if (cfg.tail_th <= c <= cfg.head_th)]
        hmt_groups = {"head": head, "medium": medium, "tail": tail, "counts": counts_np.tolist()}
        print(f"[HMT] head={len(head)}, medium={len(medium)}, tail={len(tail)}  (th: >{cfg.head_th} / {cfg.tail_th}–{cfg.head_th} / <{cfg.tail_th})")

    # === 统一一次计数 ===
    counts_t  = torch.tensor(counts_np, dtype=torch.float32, device=device) # tensor [C]
    counts_t  = counts_t.clamp_min(1.0)                                     # 防 0

    # === BSM: 先验 log_prior ===
    prior     = counts_t / counts_t.sum()
    log_prior = prior.log()                                                 # [C] on device

    # === LDAM: per-class margins ===
    m_raw         = counts_t.pow(-0.25)                                     # 1 / n^(1/4)
    ldam_margins  = m_raw * (cfg.ldam_margin / m_raw.max())                 # 最大缩放到 C

    # === RW: 1 / n （均值归一到 1）===
    rw_weights_per_class = None
    if cfg.use_rw:
        rw_weights_per_class = (counts_t.sum() / counts_t)                  # ∝ 1/n
        rw_weights_per_class = rw_weights_per_class / rw_weights_per_class.mean()

    # === CB: 1 / E(n) （均值归一到 1）===
    cb_weights_per_class = None
    if cfg.use_cb:
        beta = cfg.cb_beta
        # 注意：pow 的底数是标量 beta，指数是 counts_t（在 device 上）
        effective_num = (1.0 - torch.pow(torch.tensor(beta, device=device), counts_t)) / (1.0 - beta)
        cb_weights_per_class = 1.0 / effective_num
        cb_weights_per_class = cb_weights_per_class / cb_weights_per_class.mean()

    # === 采样器（RS）：用 numpy 版计数 ===
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
        
    # 是否在训练时丢弃最后一个不足 batch_size 的 batch
    drop_last = (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=shuffle,
                              num_workers=cfg.workers, pin_memory=True, sampler=sampler,
                              drop_last=drop_last,          # ← 新增
                             )
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.workers, pin_memory=True,
                              drop_last=False,              # 验证集保持完整
                              )

    # 返回签名不变
    # return train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins
    return train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins, hmt_groups


# -----------------------------
# Train / Eval
# -----------------------------

def aligned_contrastive_loss(z, y, t=0.07):
    """
    轻量版 ACL：带标签的对比损失，正样本来自同类，负样本来自不同类。
    """
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / t        # [B, B] 相似度矩阵
    labels = y.unsqueeze(0) == y.unsqueeze(1)  # [B, B] 正样本掩码
    mask_pos = labels.fill_diagonal_(False)    # 自身不算正样本

    # 只考虑正样本
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * labels).sum(1)
    all_sum = exp_sim.sum(1)
    loss = -torch.log((pos_sum + 1e-9) / (all_sum + 1e-9))
    return loss.mean()


def train_one_epoch_baseline(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    label_smoothing: float = 0.0,
    rw_weights_per_class: Optional[torch.Tensor] = None,  # <—— 新增
    cb_weights_per_class: Optional[torch.Tensor] = None,   # CB 权重（按 1/E(n)）
    cb_loss: str = "ce",                                    # "ce" or "focal"
    cb_gamma: float = 2.0,                                  # focal 的 gamma
    use_bsm: bool = False,
    log_prior: Optional[torch.Tensor] = None,
    use_ldam: bool = False,
    ldam_margins: Optional[torch.Tensor] = None,
    ldam_scale: float = 30.0,
    mixup_fn: Optional[Mixup] = None,
    use_wb: bool = False,
    wb_max_norm: Optional[float] = None,
    wb_renorm_alpha: float = 1.0,
    wb_renorm_target: str = "mean",
    proj_head=None,
    use_acl=False,
    acl_lambda=0.1,
    acl_temp=0.07,
):
    model.train()
    total_loss = 0.0
    
    if use_bsm:
        assert log_prior is not None, "use_bsm=True 时必须提供 log_prior（shape=[C]）"
        # 广播到 [B, C] 用： (B,C) + (C,) 会自动广播；也可手动 log_prior.unsqueeze(0)
        log_prior = log_prior.to(device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)  # targets 变成 [B, C] 软标签

        logits = model(images)
        # loss = F.cross_entropy(logits, targets,
        #                        label_smoothing=label_smoothing,
        #                        reduction="mean")
        # -------- 先构造“基损失” per-sample --------
        if mixup_fn is not None:
            # 软标签损失
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
                    # 复制 logits 并对真类位置减去 margin
                    logits_m = logits.clone()
                    index = torch.zeros_like(logits_m, dtype=torch.bool)
                    index.scatter_(1, targets.unsqueeze(1), True)           # 真类为 True
                    margins = ldam_margins.to(device)[targets]               # [B]
                    logits_m[index] = logits_m[index] - margins              # y 类减 m_y

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
                    # 普通 CE（用于：纯 CE / RW / CB-CE）
                    per_sample_loss = F.cross_entropy(
                        logits, targets, reduction="none", label_smoothing=label_smoothing
                    )  # [B]

                # -------- 再乘“类别权重” --------
                if cb_weights_per_class is not None:
                    # ✅ CB：权重来自 1 / E(n)；（已在 build_dataloaders 里均值归一）
                    w = cb_weights_per_class[targets]  # [B]
                    loss = (w * per_sample_loss).mean()
                elif rw_weights_per_class is not None:
                    # ✅ RW：权重来自 1 / n；（已在 build_dataloaders 里均值归一）
                    w = rw_weights_per_class[targets]  # [B]
                    loss = (w * per_sample_loss).mean()
                else:
                    # ✅ 纯 CE
                    loss = per_sample_loss.mean()
                    
        if use_acl and proj_head is not None and (mixup_fn is None):
            if hasattr(model, "forward_features"):
                feats = model.forward_features(images)   # 可能是 [B, C] / [B, 197, C] / [B, C, H, W]
                # —— 统一压成 [B, D] 向量 ——
                if feats.ndim == 3:          # ViT: [B, N, C]（N=197等）
                    # 用 CLS（更贴合 ViT 分类）或平均都可；推荐 CLS：
                    feats = feats[:, 0, :]   # 取 CLS token
                    # 若想平均： feats = feats.mean(dim=1)
                elif feats.ndim == 4:        # CNN: [B, C, H, W]
                    feats = feats.mean(dim=(2, 3))   # GAP
                # else: 已经是 [B, D]
            else:
                # 无 forward_features 就退一步：用分类头前的 logits 的梯度会更 noisy，这里只兜底
                feats = logits.detach()

            z = proj_head(feats)                         # [B, acl_dim]
            loss_acl = aligned_contrastive_loss(z, targets, t=acl_temp)
            loss = loss + acl_lambda * loss_acl        

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if use_wb and wb_renorm_target != "none":
            clf = get_classifier_module(model)
            # 计算 target_norm
            if hasattr(clf, "weight") and clf.weight is not None:
                with torch.no_grad():
                    norms = clf.weight.detach().pow(2).sum(dim=1).sqrt()
                    if wb_renorm_target == "mean":
                        target = float(norms.mean().item())
                    elif wb_renorm_target == "median":
                        target = float(norms.median().item())
                    else:
                        target = None
                wb_renorm_classifier(
                    clf,
                    target_norm=target,
                    alpha=wb_renorm_alpha,
                    max_norm=(wb_max_norm if (wb_max_norm is not None and wb_max_norm > 0) else None),
                )

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

def gml_loss_from_logits(logits: torch.Tensor,
                         targets: torch.Tensor,
                         log_counts: torch.Tensor,
                         num_classes: int,
                         eps: float = 1e-12) -> torch.Tensor:
    """
    实现论文 GML：
      L = -(1/C) * sum_c log( pbar_c )
      pbar_c = mean_j  tilde_p_y(j)  over samples with class c in this mini-batch
      tilde_p = softmax(logits + log(N))   # N 为训练集中每类样本数
    参考: CVPR'23 NOLB（Eq. (1)(2)(3)）。 
    """
    # 若模型输出带额外维度（如 [B,C,H,W] / [B,T,C]），压成 [B,C]
    if logits.ndim > 2:
        # 对除 batch 与 class 之外的维做平均更稳（大部分头部最终 H=W=1，等价）
        dims = tuple(range(2, logits.ndim))
        logits = logits.mean(dim=dims)

    # reweighted softmax: softmax(o + log N)
    logits_rw = logits + log_counts.view(1, -1)               # [B,C]
    p_tilde = torch.softmax(logits_rw, dim=1)                  # [B,C]

    # 按 batch 内类别聚合 p_{y}
    B, C = logits.shape
    present_classes = targets.unique()
    pbar_list = []
    for c in present_classes:
        mask = (targets == c)
        # 取该类样本的 p_tilde 对应真类概率
        pc = p_tilde[mask, c]                                  # [Nc]
        # 避免 log(0)
        pbar_c = pc.mean().clamp_min(eps)
        pbar_list.append(pbar_c)

    if len(pbar_list) == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # 论文原式使用 1/C；工程上常用 1/|present| 稳定梯度
    # 你也可以改成除以 C 与论文完全一致：
    loss = -torch.log(torch.stack(pbar_list)).mean()           # / |present|
    # 若想完全按论文： loss = -torch.log(torch.stack(pbar_list)).sum() / float(num_classes)
    return loss


@torch.no_grad()
def extract_features(model, x):
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    raise NotImplementedError("This model lacks forward_features")


def finetune_classifier_with_gml(model: nn.Module,
                                 train_loader: DataLoader,
                                 counts_np: np.ndarray,
                                 num_classes: int,
                                 device: str,
                                 epochs: int = 10,
                                 lr: float = 1e-3,
                                 weight_decay: float = 0.0):
    """
    冻结 backbone，重置分类头，并用 GML 训练若干 epoch。
    返回：(new_head, old_head) 方便后续集成。
    """
    model.eval()
    # 1) 取旧的分类头副本（用于 ensemble）
    old_head = None
    if hasattr(model, "get_classifier"):
        old_head = nn.Sequential(*[m for m in [model.get_classifier()] if m is not None])
    else:
        # 大多数 timm 模型都有 head / fc / classifier 之一
        old_head = nn.Sequential(*[getattr(model, k) for k in ["head", "fc", "classifier"] if hasattr(model, k)])

    # 2) 冻结 backbone
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    # 3) 重置分类头（timm 提供）
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        new_head = model.get_classifier()
    else:
        # 兜底：假定有名为 "head" 的线性层
        in_dim = model.get_classifier().in_features
        setattr(model, "head", nn.Linear(in_dim, num_classes))
        new_head = model.head

    # 只训练新头
    for p in new_head.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(new_head.parameters(), lr=lr, weight_decay=weight_decay)

    # 预先计算 log(N)
    counts = torch.from_numpy(np.clip(counts_np, 1, None)).float().to(device)
    log_counts = counts.log()

    model.to(device)
    new_head.to(device)
    if old_head is not None:
        old_head = old_head.to(device)

    model.train()
    for ep in range(1, epochs + 1):
        tot, nimg = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)  # 若 reset_classifier 正常，这里会用新头
            loss = gml_loss_from_logits(logits, targets, log_counts, num_classes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            tot += loss.item() * bs
            nimg += bs
        print(f"[GML] Epoch {ep:02d}/{epochs} | Loss {tot / max(nimg,1):.4f}")

    return new_head, old_head



@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int, device: str,
             groups: Optional[dict] = None) -> Dict[str, float]:
    model.eval()
    
    def _softmax_temp(logits, t):
        return torch.softmax(logits / max(t, 1e-6), dim=1)

    use_ens = getattr(evaluate, "_use_ensemble", False)
    t_old = getattr(evaluate, "_t_old", 1.0)
    t_new = getattr(evaluate, "_t_new", 1.0)
    old_head = getattr(evaluate, "_old_head", None)

    
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # [pred, true]
    total, correct = 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # logits = model(images)
        # preds = logits.argmax(dim=1)
        if use_ens and (old_head is not None) and hasattr(model, "forward_features") and hasattr(model, "get_classifier"):
            # 温度集成：提取特征 -> 分别过新头与旧头 -> 温度softmax -> 平均
            feats = model.forward_features(images)
            logits_new = model.get_classifier()(feats)   # 新头
            logits_old = old_head(feats)                 # 旧头（在 GML 第二阶段前保存的副本）
            probs = 0.5 * _softmax_temp(logits_new, t_new) + 0.5 * _softmax_temp(logits_old, t_old)
            preds = probs.argmax(dim=1)
        else:
            # 原始单头推理
            logits = model(images)
            preds = logits.argmax(dim=1)
                # —— 你的分支结束后，加这一行“一刀切兜底” ——
        if preds.ndim > 1:
            preds = preds.argmax(dim=1)


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
    parser.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="checkpoint_baseline.pth")
    # 仅与加载预训练相关
    parser.add_argument("--no-pretrained", action="store_true",
                        help="不从网络下载 timm 预训练（离线时必须开）")
    parser.add_argument("--init", type=str, default="",
                        help="本地预训练权重路径（.pth/.bin），与 --no-pretrained 搭配使用")
    parser.add_argument("--use-rw", action="store_true",
                    help="启用重加权(Re-Weighting)，按类频倒数做加权 CE。")
    parser.add_argument("--use-cb", action="store_true",
                    help="启用 Class-Balanced Loss（与 --use-rw / --balanced-sampler 互斥）")
    parser.add_argument("--cb-beta", type=float, default=0.999, 
                    help="CB loss 的 beta，长尾常用 0.99~0.9999")
    parser.add_argument("--cb-loss", type=str, default="ce", choices=["ce","focal"],
                    help="CB 搭配的基损失：普通 CE 或 Focal")
    parser.add_argument("--cb-gamma", type=float, default=2.0,
                    help="CB+Focal 时的 gamma")
    parser.add_argument("--use-bsm", action="store_true",
                        help="Balanced Softmax loss: CE(logits + log(count)). "
                            "与 RS/RW/CB 互斥。")
    parser.add_argument("--use-ldam", action="store_true",
                    help="启用 LDAM（不带 DRW）")
    parser.add_argument("--ldam-margin", type=float, default=0.5,
                        help="LDAM 的最大 margin C（常用 0.5）")
    parser.add_argument("--ldam-scale", type=float, default=30.0,
                        help="LDAM 的缩放 s（常用 30）")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, >0 启用")
    parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha, >0 启用")
    parser.add_argument("--mix-prob", type=float, default=1.0, help="执行 mix 的概率")
    parser.add_argument("--mix-switch-prob", type=float, default=0.0, help="两种策略切换概率")
    parser.add_argument("--hmt", action="store_true", help="按每类样本数统计 Head/Medium/Tail 指标（基于 train 计数）")
    parser.add_argument("--head-th", type=int, default=100, help="Head 阈值：> head_th")
    parser.add_argument("--tail-th", type=int, default=20,  help="Tail 阈值：< tail_th")
    parser.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--gml", action="store_true", help="启用 NOLB 第二阶段（GML）微调分类头")
    parser.add_argument("--gml-epochs", type=int, default=10, help="GML 微调轮数（论文建议 few epochs）")
    parser.add_argument("--ensemble", action="store_true", help="推理时使用 old/new 双头温度集成")
    parser.add_argument("--t-old", type=float, default=1.0, help="old 头温度")
    parser.add_argument("--t-new", type=float, default=1.0, help="new 头温度")
    parser.add_argument("--use-wb", action="store_true",
                        help="启用 Weight Balancing：分类头更强WD + 权重范数重整。")
    parser.add_argument("--wb-wd-mult", type=float, default=5.0,
                        help="分类头相对基座的 weight_decay 倍数（例如 5x）。")
    parser.add_argument("--wb-max-norm", type=float, default=3.0,
                        help="分类头每类权重的最大L2范数（None或<=0 表示不启用max-norm）。")
    parser.add_argument("--wb-renorm-alpha", type=float, default=1.0,
                        help="范数重整强度 alpha∈[0,1]；1为完全等范数，0为不变。")
    parser.add_argument("--wb-renorm-target", type=str, default="mean",
                        choices=["mean","median","none"],
                        help="target范数：用所有类权重范数的mean/median，或不做等范数。")
    parser.add_argument("--wb-renorm-freq", type=str, default="step",
                        choices=["step","epoch","none"],
                        help="在每个step或每个epoch末进行一次重整，或关闭。")
    parser.add_argument("--use-acl", action="store_true",
                        help="启用 Aligned Contrastive Loss (ACL)：CE + 对比学习联合优化。")
    parser.add_argument("--acl-lambda", type=float, default=0.1,
                        help="ACL 损失权重 (λ)，建议 0.05~0.2。")
    parser.add_argument("--acl-temp", type=float, default=0.07,
                        help="ACL 温度系数 temperature。")
    parser.add_argument("--acl-dim", type=int, default=128,
                        help="ACL 投影特征维度。")





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
        use_rw=args.use_rw,          # ✅ 传进来
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
        hmt=args.hmt, head_th=args.head_th, tail_th=args.tail_th,
        gml=args.gml,
        gml_epochs=args.gml_epochs,
        ensemble=args.ensemble,
        t_old=args.t_old,
        t_new=args.t_new,

    )
    
    # ===== 如果启用了 MixUp/CutMix，先关掉会冲突/不必要的选项 =====
    use_mix = (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0)
    if use_mix and (cfg.use_rw or cfg.use_cb or cfg.use_ldam or cfg.use_bsm or cfg.use_balanced_sampler):
        print("[WARN] MixUp/CutMix 建议单独使用；将忽略 RS/RW/CB/LDAM/BSM。")
        cfg.use_balanced_sampler = False
        cfg.use_rw = cfg.use_cb = cfg.use_ldam = cfg.use_bsm = False

    set_seed(cfg.seed)


    device = cfg.device
    # train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins = build_dataloaders(cfg)
    train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins, hmt_groups = build_dataloaders(cfg)
    
    # ===== 现在才构造 mixup_fn（已有 num_classes）=====
    mixup_fn = None
    if (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0):
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mix_prob,
            switch_prob=cfg.switch_prob,
            mode="batch",
            label_smoothing=cfg.label_smoothing,  # 一般配合 MixUp 设 0 或很小
            num_classes=num_classes,
        )

    # 构建模型（可禁用在线预训练）
    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {n_params/1e6:.2f}M | Trainable: {n_trainable/1e6:.2f}M")
    
    # ===== ACL：构建投影头（仅在启用时）=====
    proj_head = None
    if args.use_acl:
        # 兼容不同 backbone，自动取分类头输入维度
        # feat_dim = getattr(model.get_classifier(), "in_features", 768)
        clf = get_classifier_module(model)
        feat_dim = getattr(clf, "in_features", None)
        if feat_dim is None and hasattr(clf, "weight"):
            feat_dim = clf.weight.shape[1]
        if feat_dim is None:
            feat_dim = 768
        proj_head = ProjectionHead(in_dim=feat_dim, out_dim=args.acl_dim).to(device)

    # 加载本地权重（可选）
    if args.init:
        load_checkpoint_flex(model, args.init, keep_head=True)
        
        # ======= 如果只想跑第二阶段 GML（推荐按论文做法：加载第一阶段/外部已训的 backbone） =======
    if cfg.gml:
        # 获取全局计数：优先从 hmt_groups 复用，否则现算一次（开销很小）
        if hmt_groups is not None and "counts" in hmt_groups:
            counts_np = np.array(hmt_groups["counts"], dtype=np.int64)
        else:
            counts_np = count_by_class(datasets.ImageFolder(cfg.train_dir))

        # 跑 GML，仅训练新分类头
        new_head, old_head = finetune_classifier_with_gml(
            model, train_loader, counts_np, num_classes, device,
            epochs=cfg.gml_epochs, lr=max(cfg.lr, 1e-3), weight_decay=0.0
        )

        # （可选）推理温度集成
        if cfg.ensemble:
            evaluate._use_ensemble = True
            evaluate._old_head = old_head.eval() if old_head is not None else None
            evaluate._t_old = cfg.t_old
            evaluate._t_new = cfg.t_new

        # 评估并保存
        metrics = evaluate(model, val_loader, num_classes, device, groups=(hmt_groups if cfg.hmt else None))
        print(f"[GML{' + Ensemble' if cfg.ensemble else ''}] "
              f"Acc {metrics['acc']*100:.2f}% | MacroF1 {metrics['macro_f1']*100:.2f}% | "
              f"MacroSens {metrics['macro_sensitivity']*100:.2f}% | WCA {metrics['worst_class_acc']*100:.2f}%")

        torch.save({
            "model": model.state_dict(),
            "new_head": (model.get_classifier().state_dict() if hasattr(model, 'get_classifier') else None),
            "old_head": (old_head.state_dict() if old_head is not None else None),
            "cfg": cfg.__dict__,
            "metrics": metrics,
        }, cfg.save_path.replace(".pth", "_gml.pth"))
        print(f"✓ Saved GML checkpoint to {cfg.save_path.replace('.pth','_gml.pth')}")
        return


    # 优化器
    # if cfg.opt == "adamw":
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
    #                                 weight_decay=cfg.weight_decay, nesterov=True)
    
    # ===== 优化器（Weight Balancing：分类头单独 param group）=====
    # 1) 选优化器类型
    if cfg.opt == "adamw":
        OptimCls = torch.optim.AdamW
    elif cfg.opt == "sgd":
        OptimCls = torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer: {cfg.opt}")

    if args.use_wb:
        # 2) 按参数名把“分类头”与“其它层”分开
        head_prefixes = ("head", "fc", "classifier")  # timm 常见分类头命名
        base_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(head_prefixes):
                head_params.append(p)
            else:
                base_params.append(p)

        if len(head_params) == 0:
            # 兜底：如果某些 timm 模型分类头名字不同，这里就全部放进 base_params，避免崩溃
            base_params = [p for p in model.parameters() if p.requires_grad]
            head_params = []

        # 3) 分类头用更强的 weight decay（wb-wd-mult 倍）
        param_groups = []
        if len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay})
        if len(head_params) > 0:
            param_groups.append({"params": head_params, "lr": cfg.lr,
                                "weight_decay": cfg.weight_decay * args.wb_wd_mult})

        if cfg.opt == "adamw":
            optimizer = OptimCls(param_groups)
        else:
            optimizer = OptimCls(param_groups, momentum=cfg.momentum, nesterov=True)

    else:
        # 不启用 WB：保持原来“整模一个 param group”的做法
        if cfg.opt == "adamw":
            optimizer = OptimCls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            optimizer = OptimCls(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay, nesterov=True)
            
            
    # ===== 把 ACL 投影头参数加入优化器 =====
    if args.use_acl and proj_head is not None:
        optimizer.add_param_group({"params": proj_head.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay})



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
    t_total0 = time.perf_counter()
    best_val = -1.0
    for epoch in range(1, cfg.epochs + 1):
        t_epoch0 = time.perf_counter()
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.reset_peak_memory_stats()
        t_train0 = time.perf_counter()
        train_loss = train_one_epoch_baseline(
            model, train_loader, optimizer, device,
            label_smoothing=cfg.label_smoothing,
            rw_weights_per_class=(rw_weights_per_class if cfg.use_rw else None),
            cb_weights_per_class=(cb_weights_per_class if cfg.use_cb else None),
            cb_loss=cfg.cb_loss, 
            cb_gamma=cfg.cb_gamma,
            use_bsm=cfg.use_bsm,                # <<< 这里打开 BSM
            log_prior=(log_prior if cfg.use_bsm else None),
            use_ldam=cfg.use_ldam,
            ldam_margins=ldam_margins,
            ldam_scale=cfg.ldam_scale,
            mixup_fn=mixup_fn,           # <<< 新增参数
            use_wb=args.use_wb and (args.wb_renorm_freq == "step"),
            wb_max_norm=(args.wb_max_norm if args.wb_max_norm is not None and args.wb_max_norm > 0 else None),
            wb_renorm_alpha=args.wb_renorm_alpha,
            wb_renorm_target=args.wb_renorm_target,
            use_acl=args.use_acl,
            proj_head=proj_head,
            acl_lambda=args.acl_lambda,
            acl_temp=args.acl_temp
        )
        
        t_train1 = time.perf_counter()
        
        t_eval0 = time.perf_counter()

        metrics = evaluate(model, val_loader, num_classes, device, groups=(hmt_groups if cfg.hmt else None))
        
        t_eval1 = time.perf_counter()
        
        t_epoch1 = time.perf_counter()
        train_time = t_train1 - t_train0
        eval_time  = t_eval1 - t_eval0
        epoch_time = t_epoch1 - t_epoch0

        # 训练吞吐（按“看过的训练样本数 / train_time”近似）
        # 注意：如果你启用了 WeightedRandomSampler (RS)，len(dataset) 仍可作为本 epoch 采样数，因为你 num_samples=len(sample_weights)
        train_imgs = len(train_loader.dataset)
        ips = train_imgs / max(train_time, 1e-9)

        # 显存峰值（GB）
        mem_gb = None
        if torch.cuda.is_available() and "cuda" in str(device):
            mem_gb = torch.cuda.max_memory_allocated() / (1024**3)


        log = (f"Epoch {epoch:03d} | "
            f"Time Train/Eval/All: {train_time:.1f}/{eval_time:.1f}/{epoch_time:.1f}s | "
            f"Throughput: {ips:.1f} img/s | "
            f"TrainLoss {train_loss:.4f} | "
            f"Acc {metrics['acc']*100:.2f}% | MacroF1 {metrics['macro_f1']*100:.2f}% | "
            f"MacroSens {metrics['macro_sensitivity']*100:.2f}% | "
            f"WorstClassAcc {metrics['worst_class_acc']*100:.2f}%")

        if mem_gb is not None:
            log += f" | MaxMem {mem_gb:.2f} GB"

        # HMT 指标保持你原来的拼接逻辑
        if cfg.hmt:
            log += (f" || H Acc/F1/Sens/WCA: {metrics['H_acc']*100:.2f}/{metrics['H_f1']*100:.2f}/"
                    f"{metrics['H_sens']*100:.2f}/{metrics['H_wca']*100:.2f} | "
                    f"M {metrics['M_acc']*100:.2f}/{metrics['M_f1']*100:.2f}/{metrics['M_sens']*100:.2f}/{metrics['M_wca']*100:.2f} | "
                    f"T {metrics['T_acc']*100:.2f}/{metrics['T_f1']*100:.2f}/{metrics['T_sens']*100:.2f}/{metrics['T_wca']*100:.2f}")

        print(log)
        
        if scheduler is not None:
            # timm 的 step 接受 epoch；纯 torch 用 scheduler.step()
            scheduler.step(epoch)
        
                # ===== WB: epoch 级重整（若选择 epoch 频率）=====
        if args.use_wb and args.wb_renorm_freq == "epoch" and args.wb_renorm_target != "none":
            clf = get_classifier_module(model)
            if hasattr(clf, "weight") and clf.weight is not None:
                with torch.no_grad():
                    norms = clf.weight.detach().pow(2).sum(dim=1).sqrt()
                    if args.wb_renorm_target == "mean":
                        target = float(norms.mean().item())
                    elif args.wb_renorm_target == "median":
                        target = float(norms.median().item())
                    else:
                        target = None
                wb_renorm_classifier(
                    clf,
                    target_norm=target,
                    alpha=args.wb_renorm_alpha,
                    max_norm=(args.wb_max_norm if (args.wb_max_norm is not None and args.wb_max_norm > 0) else None),
                )


        # 保存 best（以 MacroF1+MacroSens 的平均为准；也可改为 WorstClassAcc）
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
