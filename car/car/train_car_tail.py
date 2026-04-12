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

from timm.scheduler import CosineLRScheduler  #  import timm 

import time


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_global_grad_norm(model: nn.Module) -> float:
    """
    Compute global L2 (Frobenius) norm of gradients over all parameters.
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if len(grads) == 0:
        return 0.0
    g = torch.cat(grads)
    return torch.norm(g, p=2).item()


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
    “”
    head/head.fc/classifier DataParallel 
    """
    assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
    print(f"==> Loading pretrained from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    #  state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: sd = ckpt["state_dict"]
        elif "model" in ckpt:    sd = ckpt["model"]
        else:                    sd = ckpt
    else:
        sd = ckpt

    #  DataParallel 
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # 
    fixed = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("classifier."):            # HF/timm
            k2 = k2.replace("classifier.", "head.")
        if k2.startswith("head.weight"):            #  head.weight/bias
            k2 = k2.replace("head.weight", "head.fc.weight")
        if k2.startswith("head.bias"):
            k2 = k2.replace("head.bias", "head.fc.bias")
        fixed[k2] = v

    msd = model.state_dict()
    filtered = {}
    dropped = []

    # 1) 2) 
    for k, v in fixed.items():
        if k in msd and v.shape == msd[k].shape:
            filtered[k] = v
        else:
            dropped.append(k)

    # 
    for head_key in list(filtered.keys()):
        if head_key.startswith("head.") or head_key.startswith("fc.") or head_key.startswith("classifier."):
            # num_classes
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
    
    counts_np = count_by_class(train_set)   # [C] ()
    
    hmt_groups = None
    if cfg.hmt:
        head = [i for i,c in enumerate(counts_np) if c > cfg.head_th]
        tail = [i for i,c in enumerate(counts_np) if c < cfg.tail_th]
        medium = [i for i,c in enumerate(counts_np) if (cfg.tail_th <= c <= cfg.head_th)]
        hmt_groups = {"head": head, "medium": medium, "tail": tail, "counts": counts_np.tolist()}
        print(f"[HMT] head={len(head)}, medium={len(medium)}, tail={len(tail)}  (th: >{cfg.head_th} / {cfg.tail_th}–{cfg.head_th} / <{cfg.tail_th})")

    if cfg.use_balanced_sampler:
        counts = count_by_class(train_set)
        num_classes = len(counts)
        # counts_t = torch.tensor(counts, dtype=torch.float32)   # CPU tensor

        # # w_j = 1 / sqrt(m_j + lambda0)
        # w = 1.0 / torch.sqrt(counts_t + args.lambda0)          # shape [C]

        # # 
        # w = (w / w.mean()).clamp(min=1e-3)

        # # DDP local_rank  device
        # w = w.to(device)
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

    # return train_loader, train_loader_eval, val_loader, num_classes
    return train_loader, val_loader,train_loader_eval, num_classes, hmt_groups



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
    “” S \in R^{C x C} =jsoftmax
    - 
    -  softmax p p  j  j
    -  margin (top1 - top2 <= tau) “”
    - 
    """
    model.eval()
    S = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)  # 
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
            probs = probs * mask  # 0

        # S[:, j] += p
        for cls in range(num_classes):
            idx = (targets == cls)
            if idx.any():
                S[:, cls] += probs[idx].sum(dim=0).to(torch.float64)
                counts[cls] += idx.sum()

    # 
    counts = counts.clamp_min(1.0)  # 0
    S = S / counts.unsqueeze(0)     # 

    # 
    S.fill_diagonal_(0.0)

    return S.to(torch.float32)

def gm_from_S(S: torch.Tensor) -> torch.Tensor:
    """
     S “” gm
    - SVD: S = U diag(s) V^T
    -  u[:,0], v[:,0]gm = u v^T
    - min-max “(i, j)”
    """
    # torch.linalg.svd full_matrices=False 
    U, Svals, Vh = torch.linalg.svd(S, full_matrices=False)
    u1 = U[:, 0]
    v1 = Vh.transpose(-1, -2)[:, 0]  # VhV^T
    gm = torch.outer(u1, v1)         # [C, C]

    # ######################################### compute gradients for the spectral norm ##### the first term in (11) ######################################
    #  gm ≈  ||S||_2 ∇||S||_2 ≈ u1 v1^T
    #  min-max “( i,  j)”
    gm_min, gm_max = gm.min(), gm.max()
    eps = 1e-8
    gm = 2.0 * (gm - gm_min) / (gm_max - gm_min + eps) + 0.01  # ≈ [0.01, 2.01]
    # #####################################################################################################################################################

    return gm

# -----------------------------
# Phase B: Train one epoch
# -----------------------------
# def soft_confusion_from_batch(logits: torch.Tensor, targets: torch.Tensor,
#                               num_classes: int, tau: float = 1.0) -> torch.Tensor:
#     """
#      C_t:  C[i,j] = avg_{y=j}( softmax_i(z/τ) )
#     :  i, :  j
#     """
#     B, C = logits.shape
#     probs = F.softmax(logits / tau, dim=1)          # [B,C]
#     one_hot = F.one_hot(targets, C).float()         # [B,C]
#     C_num = probs.T @ one_hot                       # [C,C], j
#     C_den = one_hot.sum(0, keepdim=True).clamp(min=1.0)  # [1,C]
#     C_t = C_num / C_den                             # 
#     C_t.fill_diagonal_(0.0)                         # 
#     return C_t

# def soft_confusion_from_batch(
#     logits: torch.Tensor,
#     targets: torch.Tensor,
#     num_classes: int,
#     tau: float = 1.0,
#     detach_gate_delta: bool = True,
#     detach_baseline_in_softmax: bool = True,
# ) -> torch.Tensor:
#     B, C = logits.shape
#     device = logits.device

#     #  j  logit [B,1]
#     f_j = logits.gather(1, targets.view(-1,1))

#     # === soft argmax over non-j: softmax((f - f_j)/τ) ===
#     rel = (logits - (f_j.detach() if detach_baseline_in_softmax else f_j)) / tau
#     S_nonj = F.softmax(rel, dim=1)                            # [B,C]

#     #  j True  j
#     mask_nonj = ~F.one_hot(targets, C).to(device).bool()      # [B,C]
#     S_nonj = S_nonj * mask_nonj.float()                       #  scatter_ in-place

#     # === soft margin gate: σ(f_i - f_j) (γ=0) ===
#     delta = logits - f_j                                      # [B,C]
#     if detach_gate_delta:
#         delta = delta.detach()
#     gate = torch.sigmoid(delta)
#     gate = gate * mask_nonj.float()                           #  gate.scatter_

#     # ===  j ===
#     contrib = gate * S_nonj                                   # [B,C]
#     one_hot = F.one_hot(targets, C).to(device).float()        # [B,C]
#     C_num = contrib.transpose(0,1) @ one_hot                  # [C,C]
#     C_den = one_hot.sum(0, keepdim=True).clamp(min=1.0)       # [1,C]
#     C_t = C_num / C_den                                       # 
#     C_t.fill_diagonal_(0.0)
#     return C_t

def soft_confusion_from_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    tau: float = 1.0,
    detach_gate_delta: bool = True,
    detach_baseline_in_softmax: bool = True,

    # ===== ablations =====
    ablate_no_sigmoid: bool = False,   # Ablation A
    ablate_no_softmax: bool = False,   # Ablation B
    ablate_no_both: bool = False,      # Ablation C (overrides A/B)
    gamma: float = 0.0,                # only for Ablation C
) -> torch.Tensor:
    """
    Return soft confusion matrix C_t in shape [C,C], row=pred i, col=true j.
    By default: uses Eq.(7)-like surrogate: sigmoid gate * softmax competition.
    Ablations:
      - A: remove sigmoid gate -> gate=1(non-j)
      - B: remove softmax -> ReLU-normalized competition
      - C: remove both -> ReLU margin gate * ReLU-normalized competition
    """

    B, C = logits.shape
    device = logits.device

    # true-class logit f_j: [B,1]
    f_j = logits.gather(1, targets.view(-1, 1))

    # mask for non-j entries
    mask_nonj = ~F.one_hot(targets, C).to(device).bool()   # [B,C]
    mask_f = mask_nonj.float()

    # if ablate_no_both: it overrides other two switches
    if ablate_no_both:
        ablate_no_sigmoid = True
        ablate_no_softmax = True

    # -----------------------------
    # 1) Competition distribution S_nonj
    # -----------------------------
    # rel = (f_i - f_j)/tau
    baseline = f_j.detach() if detach_baseline_in_softmax else f_j
    rel = (logits - baseline) / max(tau, 1e-12)            # [B,C]

    if not ablate_no_softmax:
        # ==== Default: softmax competition (Eq.7's S(·)) ====
        S_nonj = F.softmax(rel, dim=1) * mask_f            # [B,C]
    else:
        # ==== Ablation B/C: remove softmax -> ReLU-normalized distribution ====
        # pos >= 0, then normalize across classes (per sample)
        pos = F.relu(rel) * mask_f                         # [B,C]
        denom = pos.sum(dim=1, keepdim=True).clamp(min=1e-12)
        S_nonj = pos / denom                               # [B,C]

    # -----------------------------
    # 2) Gate term
    # -----------------------------
    if not ablate_no_sigmoid:
        # ==== Default: sigmoid gate (Eq.7's σ(·)) ====
        delta = logits - f_j                                # [B,C]
        if detach_gate_delta:
            delta = delta.detach()
        gate = torch.sigmoid(delta) * mask_f
    else:
        if not ablate_no_both:
            # ==== Ablation A: remove sigmoid -> constant gate ====
            gate = mask_f                                   # gate=1 for non-j
        else:
            # ==== Ablation C: remove sigmoid -> ReLU margin gate ====
            # gate = relu(f_i - f_j + gamma)  (non-saturating)
            delta = (logits - f_j)                           # [B,C]
            if detach_gate_delta:
                delta = delta.detach()                       # keep consistent with your design
            gate = F.relu(delta + float(gamma)) * mask_f

    # -----------------------------
    # 3) Aggregate into C_t
    # -----------------------------
    contrib = gate * S_nonj                                  # [B,C]
    one_hot = F.one_hot(targets, C).to(device).float()       # [B,C]

    # accumulate into columns of true class j
    C_num = contrib.transpose(0, 1) @ one_hot                # [C,C]
    C_den = one_hot.sum(0, keepdim=True).clamp(min=1.0)      # [1,C]
    C_t = C_num / C_den                                      # [C,C]
    C_t.fill_diagonal_(0.0)
    return C_t


@torch.no_grad()
def compute_true_soft_confusion_full(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str,
    temp: float = 1.0,
    detach_gate_delta: bool = True,
    detach_softmax_baseline: bool = True,
    ablate_no_sigmoid: bool = False,
    ablate_no_softmax: bool = False,
    ablate_no_both: bool = False,
    gamma: float = 0.0,
) -> torch.Tensor:
    """
    Compute "true" soft confusion on the full training set (eval transform),
    using the same surrogate definition as training, but WITHOUT EMA.
    Returns C_true in shape [C,C], row=pred i, col=true j, with diagonal cleared.
    """
    model.eval()
    C_sum = torch.zeros(num_classes, num_classes, dtype=torch.float64, device=device)
    col_counts = torch.zeros(num_classes, dtype=torch.float64, device=device)  # count per true class (denominator)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)

        C_batch = soft_confusion_from_batch(
            logits, targets, num_classes,
            tau=temp,
            detach_gate_delta=detach_gate_delta,
            detach_baseline_in_softmax=detach_softmax_baseline,
            ablate_no_sigmoid=ablate_no_sigmoid,
            ablate_no_softmax=ablate_no_softmax,
            ablate_no_both=ablate_no_both,
            gamma=gamma,
        ).to(torch.float64)  # [C,C], already column-normalized within batch (by batch counts)

        # IMPORTANT: batch-level column-normalization makes each batch equally weighted.
        # For a full-dataset "true" estimate, we should re-weight by batch true-class counts:
        # reconstruct numerator by multiplying back the batch denominators.
        one_hot = F.one_hot(targets, num_classes).to(device).float()     # [B,C]
        # den = one_hot.sum(0).to(torch.float64).clamp(min=1.0)           # [C]
        # C_num = C_batch * den.unsqueeze(0) 
        
        den = one_hot.sum(0).to(torch.float64)          # [C], can be 0
        C_num = C_batch * den.unsqueeze(0)
        # col_counts += den                              # [C,C] numerator in this batch

        C_sum += C_num
        col_counts += den

    col_counts = col_counts.clamp_min(1.0)
    C_true = (C_sum / col_counts.unsqueeze(0)).to(torch.float32)
    C_true.fill_diagonal_(0.0)
    return C_true



def spectral_loss_from_C(C: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    L_spec = || C @ diag(w) ||_2 = 
    """
    Cw = C @ torch.diag(w.to(C.device))
    # svd 
    _, svals, _ = torch.linalg.svd(Cw, full_matrices=False)
    return svals[0]

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
    label_smoothing: float = 0.0,
    detach_gate_delta: bool = True,           # ← 
    detach_softmax_baseline: bool = True,     # ← 
    ablate_no_sigmoid: bool = False,
    ablate_no_softmax: bool = False,
    ablate_no_both: bool = False,
    gamma: float = 0.0,
    record_grad_norm: bool = False,
    grad_norm_recorder: list = None,
):

    train_time = 0.0
    spec_time = 0.0
    """
     + EMA  epoch
    - cbar:  C_barNone  batch 
    """
    model.train()

    if record_grad_norm and grad_norm_recorder is None:
        grad_norm_recorder = []
    total_loss = 0.0
    w = w.to(device)

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # -------- start iter timing --------
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_iter0 = time.perf_counter()

        logits = model(images)  # [B,C]

        #  CE
        loss_ce = F.cross_entropy(
            logits, targets,
            label_smoothing=label_smoothing,
            reduction="mean"
        )

        # -------- spectral part timing start --------
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_spec0 = time.perf_counter()

        C_t = soft_confusion_from_batch(
            logits, targets, num_classes,
            tau=temp,
            detach_gate_delta=detach_gate_delta,
            detach_baseline_in_softmax=detach_softmax_baseline,
            ablate_no_sigmoid=ablate_no_sigmoid,
            ablate_no_softmax=ablate_no_softmax,
            ablate_no_both=ablate_no_both,
            gamma=gamma,
        )

        # EMA: C_bar = β*C_bar(stopped) + (1-β)*C_t
        if cbar is None:
            cbar = C_t.detach()
        else:
            cbar = beta * cbar.detach() + (1.0 - beta) * C_t

        spec = spectral_loss_from_C(cbar, w)

        # ================= Gradient Diagnostic (CSR only) =================
        if record_grad_norm:
            # 
            optimizer.zero_grad(set_to_none=True)

            #  regularizer  CE
            spec.backward(retain_graph=True)

            #  global F-norm
            grad_norm = compute_global_grad_norm(model)

            # 
            grad_norm_recorder.append(grad_norm)

            # 
            optimizer.zero_grad(set_to_none=True)
        # ==================================================================


        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_spec1 = time.perf_counter()
        spec_time += (t_spec1 - t_spec0)
        # -------- spectral part timing end --------

        loss = loss_ce + lambda_spec * spec

        

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_iter1 = time.perf_counter()
        train_time += (t_iter1 - t_iter0)
        # -------- end iter timing --------



    return total_loss / len(loader.dataset), cbar, train_time, spec_time


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
        acc_c = tp / (tp + fn + eps)   # class-accuracy=recall
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

t_train0 = time.perf_counter()

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

    # —— “ / ” ——
    parser.add_argument("--no-pretrained", action="store_true",
                        help=" timm ")
    parser.add_argument("--init", type=str, default="",
                        help=".pth/.bin --no-pretrained ")
    
    parser.add_argument("--hmt", action="store_true", help=" Head/Medium/Tail  train ")
    parser.add_argument("--head-th", type=int, default=100, help="Head > head_th")
    parser.add_argument("--tail-th", type=int, default=20,  help="Tail < tail_th")
    parser.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument('--use-wj', action='store_true',
                        help='Use class-frequency weights w_j in CSR (C @ diag(w))')
    parser.add_argument("--spec-lambda", type=float, default=0.3,
                    help="λ for spectral norm regularization")
    parser.add_argument("--spec-beta", type=float, default=0.9,
                        help="EMA factor β for running soft confusion C_bar")
    parser.add_argument("--spec-temp", type=float, default=1.0,
                        help="temperature τ in softmax(z/τ) for soft confusion")
    parser.add_argument("--wj-norm", action=argparse.BooleanOptionalAction, default=True,
                    help=" w  (w /= mean(w))")
    parser.add_argument("--wj-min", type=float, default=1e-3,
                    help="w  0 ")
    parser.add_argument("--no-w", action="store_true",
        help="Disable class-weighting in spectral term (use identity W).")
    #  argparse 
    parser.add_argument("--detach-gate-delta", action=argparse.BooleanOptionalAction, default=True,
                    help="Detach Δ=f_i−f_j in gating σ(·). Default: True")
    parser.add_argument("--detach-softmax-baseline", action=argparse.BooleanOptionalAction, default=True,
                        help="Detach baseline f_j in softmax(f−f_j). Default: True")
    #  r0 0~0.5
    parser.add_argument("--r0", type=float, default=0.2,
        help="Smoothing radius on class frequency m_j in w_j = 1/sqrt(m_j + r0). Use 0~0.5.")

    #  --lambda0 help  deprecated
    parser.add_argument('--lambda0', type=float, default=None,
        help='[DEPRECATED] Old count-scale smoothing for w_j. Do not use together with --r0.')
    
    parser.add_argument("--ablate-no-sigmoid", action="store_true",
                        help="Ablation A: remove sigmoid gate, use constant gate=1 for non-j.")
    parser.add_argument("--ablate-no-softmax", action="store_true",
                        help="Ablation B: remove softmax; use ReLU-normalized competition distribution.")
    parser.add_argument("--ablate-no-both", action="store_true",
                        help="Ablation C: remove both sigmoid and softmax; use ReLU margin gate + ReLU-normalized competition.")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Margin offset used in Ablation C (ReLU margin gate): relu(f_i - f_j + gamma).")



    from collections import defaultdict
    grad_log = defaultdict(list)

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
        hmt=args.hmt, head_th=args.head_th, tail_th=args.tail_th,
    )

    set_seed(cfg.seed)

    device = cfg.device
    # train_loader, train_loader_eval, val_loader, num_classes = build_dataloaders(cfg)
    # train_loader, val_loader, num_classes,  hmt_groups = build_dataloaders(cfg)
    train_loader, val_loader, train_loader_eval, num_classes, hmt_groups = build_dataloaders(cfg)

    # counts_np = count_by_class(train_loader.dataset)        # [C]
    # counts_t  = torch.tensor(counts_np, dtype=torch.float32)
    # w = 1.0 / torch.sqrt(counts_t + args.lambda0)  # w_j = 1/sqrt(m_j+λ0)
    
    counts_np = count_by_class(train_loader.dataset)          # [C]
    counts_t  = torch.tensor(counts_np, dtype=torch.float32)  # [C]
    N = float(counts_t.sum().item()) + 1e-12                  # 
    m = counts_t / N                                          #  m_j ∈ [0,1]

    #  r0
    if args.r0 is not None:
        w = 1.0 / torch.sqrt(m + float(args.r0) + 1e-12)

    #  --lambda0“+”
    elif args.lambda0 is not None:
        w = 1.0 / torch.sqrt(counts_t + float(args.lambda0) + 1e-12)

    #  r0
    else:
        w = 1.0 / torch.sqrt(m + 0.2 + 1e-12)

    #  & 
    if args.wj_norm:
        w = w / w.mean()
    if args.wj_min > 0:
        w = w.clamp(min=args.wj_min)

    if args.no_w:
        w = torch.ones_like(w)


    # ——  —— 
    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)

    # ——  —— 
    if args.init:
        load_checkpoint_flex(model, args.init)

    # Optimizer
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
            t_initial=cfg.epochs,            #  epoch 
            lr_min=args.min_lr,              # 
            warmup_t=args.warmup_epochs,     # warmup  epoch 
            warmup_lr_init=max(args.lr * 0.01, 1e-7),  # warmup  lr
            k_decay=1.0,
        )


    #  EMA
    S_ema: Optional[torch.Tensor] = None
    gm: Optional[torch.Tensor] = None
    
    C_bar: Optional[torch.Tensor] = None

    best_val = -1.0
    for epoch in range(1, cfg.epochs + 1):
        #  20%  epoch  reviewer 
        record_grad = epoch >= int(0.8 * cfg.epochs)
        grad_recorder = []
        t0 = time.perf_counter()
        # # ---------------- Phase A:  S gm ----------------
        # if (epoch % cfg.gm_update_every == 1) or (gm is None):
        # # if (gm is None) or (((epoch - 1) % cfg.gm_update_every) == 0):
        # # if (epoch == 1) or (((epoch - 1) % cfg.gm_update_every) == 0):
        #     #  + EMA
        #     S_cur = compute_soft_confusion_matrix(model, train_loader_eval, num_classes, tau=args.tau, device=device)
        #     S_ema = S_cur if S_ema is None else args.ema_mu * S_ema + (1 - args.ema_mu) * S_cur

        #     # >>>  diag(w)  SVD
        #     if args.use_wj:
        #         # S_eff = S_ema @ diag(w)    = || C^{col-norm} W ||_2 
        #         S_eff = S_ema @ torch.diag(w.to(S_ema.device))
        #         gm = gm_from_S(S_eff)
        #     else:
        #         gm = gm_from_S(S_ema)
        
        train_loss, C_bar, train_time, spec_time = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=num_classes, w=w,
            cbar=C_bar,
            lambda_spec=args.spec_lambda,
            beta=args.spec_beta,
            temp=args.spec_temp,
            label_smoothing=cfg.label_smoothing,
            detach_gate_delta=args.detach_gate_delta,                  # ←  args 
            detach_softmax_baseline=args.detach_softmax_baseline,       # ←  args 
            ablate_no_sigmoid=args.ablate_no_sigmoid,
            ablate_no_softmax=args.ablate_no_softmax,
            ablate_no_both=args.ablate_no_both,
            gamma=args.gamma,
            record_grad_norm=record_grad,
            grad_norm_recorder=grad_recorder,
        )


        # # ---------------- Phase B: CE +  ----------------
        # train_loss = train_one_epoch(
        #     model, train_loader, optimizer, device,
        #     gm=gm, alpha=cfg.alpha, fair_mode=cfg.fair_mode,
        #     label_smoothing=cfg.label_smoothing
        # )

        # ---------------- Evaluation ----------------
        # metrics = evaluate(model, val_loader, num_classes, device, groups=(hmt_groups if cfg.hmt else None))

        # log = (f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | "
        #     f"Acc {metrics['acc']*100:.2f}% | MacroF1 {metrics['macro_f1']*100:.2f}% | "
        #     f"MacroSens {metrics['macro_sensitivity']*100:.2f}% | "
        #     f"WorstClassAcc {metrics['worst_class_acc']*100:.2f}%")

        # if cfg.hmt:
        #     log += (f" || H Acc/F1/Sens/WCA: {metrics['H_acc']*100:.2f}/{metrics['H_f1']*100:.2f}/"
        #             f"{metrics['H_sens']*100:.2f}/{metrics['H_wca']*100:.2f} | "
        #             f"M {metrics['M_acc']*100:.2f}/{metrics['M_f1']*100:.2f}/{metrics['M_sens']*100:.2f}/{metrics['M_wca']*100:.2f} | "
        #             f"T {metrics['T_acc']*100:.2f}/{metrics['T_f1']*100:.2f}/{metrics['T_sens']*100:.2f}/{metrics['T_wca']*100:.2f}")
        # print(log)
        
        # if scheduler is not None:
        #     # timm  step  epoch torch  scheduler.step()
        #     scheduler.step(epoch)


        # # Save best by MacroF1 WorstClassAcc
        # score = (metrics['macro_f1'] + metrics['macro_sensitivity']) / 2.0
        # if score > best_val:
        #     best_val = score
        #     torch.save({
        #         "model": model.state_dict(),
        #         "cfg": cfg.__dict__,
        #         "epoch": epoch,
        #         "metrics": metrics,
        #     }, cfg.save_path)
        #     print(f"✓ Saved best checkpoint to {cfg.save_path}")
        
        # ---------------- Evaluation ----------------
        # 1)  train_loader_eval

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_eval0 = time.perf_counter()


        train_metrics = evaluate(
            model, train_loader_eval, num_classes, device,
            groups=(hmt_groups if cfg.hmt else None)
        )

        # 2) 
        val_metrics = evaluate(
            model, val_loader, num_classes, device,
            groups=(hmt_groups if cfg.hmt else None)
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_eval1 = time.perf_counter()
        eval_time = t_eval1 - t_eval0
        
        t1 = time.perf_counter()
        epoch_time = t1 - t0

        #  Train  Val  WCA Acc / Macro 
        log = (f"Epoch {epoch:03d} | TrainLoss {train_loss:.4f} | "
            f"Train Acc {train_metrics['acc']*100:.2f}% | "
            f"Train MacroF1 {train_metrics['macro_f1']*100:.2f}% | "
            f"Train MacroSens {train_metrics['macro_sensitivity']*100:.2f}% | "
            f"Train WCA {train_metrics['worst_class_acc']*100:.2f}% || "
            f"Val Acc {val_metrics['acc']*100:.2f}% | "
            f"Val MacroF1 {val_metrics['macro_f1']*100:.2f}% | "
            f"Val MacroSens {val_metrics['macro_sensitivity']*100:.2f}% | "
            f"Val WCA {val_metrics['worst_class_acc']*100:.2f}%")
        spec_ratio = spec_time / max(train_time, 1e-12)
        log += f" | Time: train {train_time:.1f}s (spec {spec_time:.1f}s, {spec_ratio*100:.1f}%), eval {eval_time:.1f}s, total {epoch_time:.1f}s"

        if cfg.hmt:
            log += (f" || [Train H/M/T WCA] "
                    f"{train_metrics['H_wca']*100:.2f}/{train_metrics['M_wca']*100:.2f}/{train_metrics['T_wca']*100:.2f} | "
                    f"[Val H/M/T WCA] "
                    f"{val_metrics['H_wca']*100:.2f}/{val_metrics['M_wca']*100:.2f}/{val_metrics['T_wca']*100:.2f}")

        print(log)

        # ---------------- CM gap: EMA-CM vs full-train "true" CM ----------------
        cm_gap_abs = float("nan")
        cm_gap_rel = float("nan")

        if C_bar is not None:
            C_true = compute_true_soft_confusion_full(
                model, train_loader_eval, num_classes, device,
                temp=args.spec_temp,
                detach_gate_delta=args.detach_gate_delta,
                detach_softmax_baseline=args.detach_softmax_baseline,
                ablate_no_sigmoid=args.ablate_no_sigmoid,
                ablate_no_softmax=args.ablate_no_softmax,
                ablate_no_both=args.ablate_no_both,
                gamma=args.gamma,
            )

            diff = (C_bar.detach() - C_true).float()
            cm_gap_abs = torch.norm(diff, p="fro").item()
            denom = torch.norm(C_true.float(), p="fro").clamp(min=1e-12).item()
            cm_gap_rel = cm_gap_abs / denom

            print(f"[CMGap] ||C_ema - C_true||_F = {cm_gap_abs:.3e} | rel = {cm_gap_rel:.3e}")
        # ------------------------------------------------------------------------


        # 
        if scheduler is not None:
            scheduler.step(epoch)

        if record_grad and len(grad_recorder) > 0:
            g_mean = float(np.mean(grad_recorder))
            g_min  = float(np.min(grad_recorder))
            g_max  = float(np.max(grad_recorder))
            grad_log["spec_grad_norm"].append({"epoch": epoch, "mean": g_mean, "min": g_min, "max": g_max})
            print(f"[GradDiag] spec-only grad ||g||_2: mean={g_mean:.3e}, min={g_min:.3e}, max={g_max:.3e}")


        # Save best by MacroF1
        save_dict = {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "epoch": epoch,
            "metrics": val_metrics,
            "train_metrics": train_metrics,
            "C_bar_ema": (C_bar.detach().cpu() if C_bar is not None else None),
            "cm_gap_abs_fro": cm_gap_abs,
            "cm_gap_rel_fro": cm_gap_rel,
        }
        torch.save(save_dict, cfg.save_path)
        print(f"✓ Saved best checkpoint to {cfg.save_path}")


if __name__ == "__main__":
    main()
