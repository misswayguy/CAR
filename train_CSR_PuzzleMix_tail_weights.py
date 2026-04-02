# -*- coding: utf-8 -*-
"""
CSR + PuzzleMix (grid OT approximation)
- Phase A: CLEAN soft confusion -> SVD -> gm
- Phase B: PuzzleMix (saliency-guided grid patch transport) + SoftTarget CE + gm-guided fairness
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

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
def infer_backbone(model) -> str:
    name = model.__class__.__name__.lower()
    if 'swin' in name:
        return 'swin'
    if 'vit' in name or 'visiontransformer' in name or (hasattr(model, 'blocks') and hasattr(model, 'cls_token')):
        return 'vit'
    return 'resnet'

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
    # PuzzleMix
    use_pmix: bool = True
    pmix_prob: float = 1.0         # prob per batch
    pmix_grid: int = 7             # GxG grid in feature map space
    pmix_ratio: float = 0.4        # total pasted area ratio (0~1)
    pmix_layer: str = "layer4"     # feature layer for saliency (ResNet: layer4)
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
# Saliency hooks
# -----------------------------
def get_classifier_weight(model: nn.Module) -> torch.Tensor:
    # 兼容 timm：resnet(fc) / head / classifier
    if hasattr(model, "fc") and hasattr(model.fc, "weight"):
        return model.fc.weight.detach()
    if hasattr(model, "classifier") and hasattr(model.classifier, "weight"):
        return model.classifier.weight.detach()
    if hasattr(model, "head"):
        head = model.head
        if hasattr(head, "weight"):
            return head.weight.detach()
        if hasattr(head, "fc") and hasattr(head.fc, "weight"):
            return head.fc.weight.detach()
    if hasattr(model, "get_classifier"):
        clf = model.get_classifier()
        if hasattr(clf, "weight"):
            return clf.weight.detach()
    raise RuntimeError("Cannot find classifier weight on the model.")

@torch.no_grad()
def vit_cam_nograd(model, images, targets, clf_w=None):
    """基于 ViT patch tokens 的 CAM，输出 [B,1,Hf,Wf]，按样本归一化。"""
    model.eval()
    B = images.size(0)
    x = model.patch_embed(images)                 # [B, N, C]
    cls = model.cls_token.expand(B, 1, -1)        # [B, 1, C]
    x = torch.cat((cls, x), dim=1)                # [B, 1+N, C]
    if getattr(model, 'pos_embed', None) is not None:
        x = x + model.pos_embed
    x = model.pos_drop(x) if hasattr(model, 'pos_drop') else x
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x) if hasattr(model, 'norm') else x
    tokens = x[:, 1:, :]                          # [B, N, C]

    # 网格大小
    if hasattr(model.patch_embed, 'grid_size'):
        Hf, Wf = model.patch_embed.grid_size
    else:
        N = tokens.shape[1]; Hf = Wf = int(N ** 0.5)

    clf_w = get_classifier_weight(model).to(images.device) if clf_w is None else clf_w.to(images.device)
    w = clf_w[targets]                            # [B, C]
    cam = torch.relu((tokens * w.unsqueeze(1)).sum(dim=2)).view(B, 1, Hf, Wf)  # [B,1,Hf,Wf]
    cam = cam / cam.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6).view(B,1,1,1)
    return cam

@torch.no_grad()
def swin_cam_nograd(model, images, targets, clf_w=None):
    """
    兼容 timm Swin 的多种返回：tokens [B,L,C] / fmap (NCHW) [B,C,H,W] / fmap (NHWC) [B,H,W,C] / (x,H,W) tuple。
    输出 CAM: [B,1,Hf,Wf]，每样本归一化到总和=1。
    """
    model.eval()

    # 1) patch_embed 可能返回 x 或 (x,H,W)
    pe = model.patch_embed(images)
    if isinstance(pe, (tuple, list)):
        x = pe[0]; H = pe[1] if len(pe) > 1 else None; W = pe[2] if len(pe) > 2 else None
    else:
        x, H, W = pe, None, None

    # 2) pos_drop（有就用）
    x = model.pos_drop(x) if hasattr(model, 'pos_drop') else x

    # 3) 逐层前向；layer 也可能返回 (x,H,W)
    for layer in model.layers:
        out = layer(x) if H is None else layer(x)
        if isinstance(out, (tuple, list)):
            x = out[0]; H = out[1] if len(out) > 1 else H; W = out[2] if len(out) > 2 else W
        else:
            x = out

    # 4) norm（有就用）
    x = model.norm(x) if hasattr(model, 'norm') else x

    # 5) 统一成 fmap: [B,C,Hf,Wf]
    if x.ndim == 4:
        # 可能是 NHWC 或 NCHW
        if x.shape[1] < x.shape[-1]:       # NHWC
            B, Hf, Wf, C = x.shape
            fmap = x.permute(0, 3, 1, 2).contiguous()
        else:                              # NCHW
            B, C, Hf, Wf = x.shape
            fmap = x
    elif x.ndim == 3:
        # tokens [B,L,C]，还原网格
        B, L, C = x.shape
        if H is None or W is None:
            if hasattr(model.layers[-1], 'input_resolution'):
                Hf, Wf = model.layers[-1].input_resolution
            elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'grid_size'):
                Hf, Wf = model.patch_embed.grid_size
            else:
                Hf = Wf = int(L ** 0.5)
        else:
            Hf, Wf = H, W
        fmap = x.transpose(1, 2).reshape(B, C, Hf, Wf)
    else:
        raise RuntimeError(f"swin_cam_nograd: unexpected shape {tuple(x.shape)}")

    # 6) 无反传 CAM
    if clf_w is None:
        clf_w = get_classifier_weight(model).to(images.device)   # [K,C]
    else:
        clf_w = clf_w.to(images.device)
    w = clf_w[targets]                                           # [B,C]
    cam = torch.relu((fmap * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True))  # [B,1,Hf,Wf]
    cam = cam / cam.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6).view(fmap.size(0), 1, 1, 1)
    return cam


def register_fmap_hook(model: nn.Module, layer_name: str = "layer4"):
    container = {}
    layer = dict([*model.named_modules()]).get(layer_name, None)
    if layer is None:
        raise RuntimeError(f"Cannot find layer '{layer_name}'.")
    def _hook(module, inp, out):
        container["fmap"] = out.detach()  # [B,C,Hf,Wf]
    handle = layer.register_forward_hook(_hook)
    return container, handle

@torch.no_grad()
def feature_saliency(fmap: torch.Tensor) -> torch.Tensor:
    sal = fmap.relu_().mean(dim=1, keepdim=True)  # [B,1,Hf,Wf]
    ssum = sal.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6)
    sal = sal / ssum.view(-1,1,1,1)
    return sal

# -----------------------------
# PuzzleMix (grid OT approx)
# -----------------------------

def grid_boxes(H: int, W: int, Gh: int, Gw: int):
    """Return list of pixel boxes [y1,x1,y2,x2] for a GhxGw uniform grid."""
    ys = torch.linspace(0, H, steps=Gh+1).round().long()
    xs = torch.linspace(0, W, steps=Gw+1).round().long()
    boxes = []
    for i in range(Gh):
        for j in range(Gw):
            y1,y2 = ys[i].item(), ys[i+1].item()
            x1,x2 = xs[j].item(), xs[j+1].item()
            if y2<=y1: y2=y1+1
            if x2<=x1: x2=x1+1
            boxes.append((y1,x1,y2,x2))
    return boxes  # len = Gh*Gw

def saliency_grid_scores(sal: torch.Tensor, Gh: int, Gw: int):
    """Average saliency per grid cell on saliency map [B,1,Hf,Wf]."""
    B, _, Hf, Wf = sal.shape
    kh, kw = Hf // Gh, Wf // Gw
    # adaptive avg pool to GhxGw
    pooled = F.adaptive_avg_pool2d(sal, output_size=(Gh, Gw)).view(B, -1)  # [B,Gh*Gw]
    return pooled  # per-cell mean

def puzzlemix_augment(
    images: torch.Tensor, saliency: torch.Tensor,
    grid: int = 7, ratio: float = 0.4
):
    """
    Grid OT approx:
    - Source: top cells by saliency
    - Target: bottom cells by saliency
    - Paste source cells into target cells (1-1), keep total pasted area ≈ ratio
    """
    B, C, H, W = images.shape
    _, _, Hf, Wf = saliency.shape
    device = images.device

    Gh = Gw = max(1, grid)
    ncell = Gh * Gw
    boxes_pix = grid_boxes(H, W, Gh, Gw)              # pixel boxes
    scores = saliency_grid_scores(saliency, Gh, Gw)   # [B, ncell]

    perm = torch.randperm(B, device=device)

    images_mix = images.clone()
    lam = torch.ones(B, device=device, dtype=images.dtype)  # portion kept
    area_total = float(H * W)
    cell_area = [(y2-y1)*(x2-x1) for (y1,x1,y2,x2) in boxes_pix]
    cell_area = torch.tensor(cell_area, device=device, dtype=images.dtype)  # [ncell]
    target_cells_per_b = int(round(ratio * area_total / (area_total / ncell)))  # ~ratio*ncell
    target_cells_per_b = max(1, min(ncell, target_cells_per_b))

    for b in range(B):
        src_b = perm[b].item()
        # 排序
        src_rank = torch.argsort(scores[src_b], descending=True)   # high -> low
        tgt_rank = torch.argsort(scores[b], descending=False)      # low -> high

        # 选择前 k 个
        k = target_cells_per_b
        src_sel = src_rank[:k]
        tgt_sel = tgt_rank[:k]

        # 逐格贴补
        pasted_area = 0.0
        for s_idx, t_idx in zip(src_sel.tolist(), tgt_sel.tolist()):
            y1s,x1s,y2s,x2s = boxes_pix[s_idx]
            y1t,x1t,y2t,x2t = boxes_pix[t_idx]
            patch = images[src_b:src_b+1, :, y1s:y2s, x1s:x2s]
            ht, wt = (y2t-y1t), (x2t-x1t)
            if patch.shape[-2:] != (ht, wt):
                patch = F.interpolate(patch, size=(ht, wt), mode="bilinear", align_corners=False)
            images_mix[b, :, y1t:y2t, x1t:x2t] = patch[0]
            pasted_area += float(ht * wt)

        lam[b] = 1.0 - pasted_area / area_total

    return images_mix, perm, lam

def one_hot(y: torch.Tensor, C: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=C).float()

# -----------------------------
# Train one epoch
# -----------------------------
def soft_confusion_from_soft_targets(logits: torch.Tensor,
                                     t_soft: torch.Tensor,
                                     tau: float = 1.0) -> torch.Tensor:
    """
    用 batch 内预测 + 软真类指示 t_soft 统计可导软混淆 C_t（列=真类软权重）。
    logits: [B,C], t_soft: [B,C]（来自 PuzzleMix 的 y_soft）
    """
    probs = F.softmax(logits / tau, dim=1)       # [B,C]
    C_num = probs.transpose(0,1) @ t_soft        # [C,C]
    C_den = t_soft.sum(0, keepdim=True).clamp(min=1.0)
    C_t = C_num / C_den
    C_t.fill_diagonal_(0.0)
    return C_t

def spectral_loss_from_C(C: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    频次加权谱正则：|| C @ diag(w) ||_2 的最大奇异值
    """
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
    use_pmix: bool,
    pmix_prob: float,
    pmix_grid: int,
    pmix_ratio: float,
    fmap_hook: dict,
    backbone: str,
    clf_w: torch.Tensor,
    # === CSR_weights 相关 ===
    w: torch.Tensor,
    cbar: Optional[torch.Tensor],
    lambda_spec: float = 0.3,
    beta: float = 0.9,
    temp: float = 1.0,
):
    """
    损失 = SoftTarget CE（来自 PuzzleMix 的 y_soft） + λ_spec * || C_bar @ diag(w) ||_2
    其中 C_bar 是 batch 内 C_t 的 EMA（带温度）；C_t 用 y_soft 作为“真类软指示”聚合。
    """
    model.train()
    total_loss = 0.0
    w = w.to(device)

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ---- 计算 saliency（PuzzleMix 需要）----
        if use_pmix and (np.random.rand() <= pmix_prob):
            with torch.no_grad():
                clf_w_batch = get_classifier_weight(model).to(device)
                if backbone == 'resnet':
                    _ = model(images)
                    fmap = fmap_hook.get("fmap", None) if fmap_hook is not None else None
                    if fmap is None:
                        raise RuntimeError("Feature map not captured. Check --pmix-layer (ResNet only).")
                    sal = feature_saliency(fmap)  # [B,1,Hf,Wf]
                elif backbone == 'vit':
                    sal = vit_cam_nograd(model, images, targets, clf_w=clf_w_batch)  # [B,1,Hf,Wf]
                elif backbone == 'swin':
                    sal = swin_cam_nograd(model, images, targets, clf_w=clf_w_batch) # [B,1,Hf,Wf]
                else:
                    raise RuntimeError(f"Unknown backbone: {backbone}")

            # ---- PuzzleMix: 生成混合样本 + 软标签 ----
            x_mix, perm, lam = puzzlemix_augment(images, saliency=sal, grid=pmix_grid, ratio=pmix_ratio)
            y_a, y_b = targets, targets[perm]
            y_soft = lam.unsqueeze(1) * one_hot(y_a, num_classes) + (1.0 - lam).unsqueeze(1) * one_hot(y_b, num_classes)

            logits = model(x_mix)
            loss_ce = ce_soft(logits, y_soft)
            t_soft = y_soft.detach()  # 只用于统计 C_t/EMA
        else:
            logits = model(images)
            y_onehot = one_hot(targets, num_classes)
            loss_ce = ce_soft(logits, y_onehot)
            t_soft  = y_onehot.detach()

        # ---- 可导软混淆矩阵 C_t（带温度） + EMA 得到 C_bar ----
        C_t = soft_confusion_from_soft_targets(logits, t_soft, tau=temp)
        if cbar is None:
            cbar = C_t.detach()
        else:
            cbar = beta * cbar.detach() + (1.0 - beta) * C_t

        # ---- 谱正则 || C_bar @ diag(w) ||_2 ----
        spec = spectral_loss_from_C(cbar, w)

        # ---- 总损失 ----
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

    # PuzzleMix args
    parser.add_argument("--no-pmix", action="store_true", help="disable PuzzleMix (use pure CSR)")
    parser.add_argument("--pmix-prob", type=float, default=1.0)
    parser.add_argument("--pmix-grid", type=int, default=7)
    parser.add_argument("--pmix-ratio", type=float, default=0.4)
    parser.add_argument("--pmix-layer", type=str, default="layer4")
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
        use_pmix=(not args.no_pmix), pmix_prob=args.pmix_prob, pmix_grid=args.pmix_grid,
        pmix_ratio=args.pmix_ratio, pmix_layer=args.pmix_layer,
        hmt=args.hmt, head_th=args.head_th, tail_th=args.tail_th,
    )

    set_seed(cfg.seed)
    device = cfg.device

    train_loader, train_loader_eval, val_loader, num_classes, hmt_groups = build_dataloaders(cfg)
    
    # === class-frequency weights: w_j = 1/sqrt(m_j + λ0) ===
    counts_np = count_by_class(train_loader.dataset)         # [C]
    counts_t  = torch.tensor(counts_np, dtype=torch.float32) # on CPU
    w = 1.0 / torch.sqrt(counts_t + args.lambda0)
    w = (w / w.mean()).clamp(min=1e-3)  # 稳定一下尺度，保留下限

    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)
    if args.init:
        load_checkpoint_flex(model, args.init)

    backbone = infer_backbone(model)

    # saliency hook
    # 只有 ResNet 需要 hook；ViT/Swin 不注册
    fmap_hook, hook_handle = (None, None)
    if backbone == 'resnet':
        fmap_hook, hook_handle = register_fmap_hook(model, layer_name=cfg.pmix_layer)

    # 分类器权重（给 ViT/Swin 的 cam 用；ResNet不强制用）
    clf_w = get_classifier_weight(model)

    # optimizer
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

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_loss, C_bar = train_one_epoch(
                model, train_loader, optimizer, device,
                num_classes=num_classes,
                use_pmix=cfg.use_pmix, pmix_prob=cfg.pmix_prob,
                pmix_grid=cfg.pmix_grid, pmix_ratio=cfg.pmix_ratio,
                fmap_hook=fmap_hook, backbone=backbone, clf_w=clf_w,
                # CSR_weights:
                w=w, cbar=C_bar,
                lambda_spec=args.spec_lambda, beta=args.spec_beta, temp=args.spec_temp,
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
    finally:
        if hook_handle is not None:
            hook_handle.remove()

if __name__ == "__main__":
    main()
