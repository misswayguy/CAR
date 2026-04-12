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

from timm.scheduler import CosineLRScheduler  #  import timm 
import time

# -----------------------------
# Utils
# -----------------------------
# ==== ACL  ====
class ProjectionHead(nn.Module):
    """MLP"""
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
    """ timm get_classifier head/fc/classifier"""
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
    
    -  max_normL2 max_norm
    - “”w_c <- w_c * (target_norm / ||w_c||)^alpha  (alpha∈[0,1]).
    """
    if not hasattr(clf, "weight"):
        return
    W = clf.weight  # [C, D]
    if W is None:
        return
    # max-norm
    if max_norm is not None:
        norms = W.detach().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(eps)  # [C,1]
        scale = (max_norm / norms).clamp(max=1.0)
        W.mul_(scale)
    # “”
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

    #  state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt: sd = ckpt["state_dict"]
        elif "model" in ckpt:    sd = ckpt["model"]
        else:                    sd = ckpt
    else:
        sd = ckpt

    #  DataParallel 
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    #  ckpt  timm 
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

    # 
    for k, v in fixed.items():
        if k in msd and v.shape == msd[k].shape:
            filtered[k] = v
        else:
            dropped.append(k)

    # ⚠️ “”GML  keep_head=True
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
    use_rw: bool = False        # ✅ 
    use_cb: bool = False          # ✅
    cb_beta: float = 0.999        # ✅
    cb_loss: str = "ce"           # ✅
    cb_gamma: float = 2.0         # ✅
    use_bsm: bool = False
    use_ldam: bool=False
    ldam_margin: float=0.5
    ldam_scale: float=30.0
    mixup_alpha: float = 0.0   #  0.2
    cutmix_alpha: float = 0.0  #  1.0
    mix_prob: float = 1.0      #  mix 
    switch_prob: float = 0.0   # MixUp  CutMix >0
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
    
    counts_np = count_by_class(train_set)   # [C] ()
    
    hmt_groups = None
    if cfg.hmt:
        head = [i for i,c in enumerate(counts_np) if c > cfg.head_th]
        tail = [i for i,c in enumerate(counts_np) if c < cfg.tail_th]
        medium = [i for i,c in enumerate(counts_np) if (cfg.tail_th <= c <= cfg.head_th)]
        hmt_groups = {"head": head, "medium": medium, "tail": tail, "counts": counts_np.tolist()}
        print(f"[HMT] head={len(head)}, medium={len(medium)}, tail={len(tail)}  (th: >{cfg.head_th} / {cfg.tail_th}–{cfg.head_th} / <{cfg.tail_th})")

    # ===  ===
    counts_t  = torch.tensor(counts_np, dtype=torch.float32, device=device) # tensor [C]
    counts_t  = counts_t.clamp_min(1.0)                                     #  0

    # === BSM:  log_prior ===
    prior     = counts_t / counts_t.sum()
    log_prior = prior.log()                                                 # [C] on device

    # === LDAM: per-class margins ===
    m_raw         = counts_t.pow(-0.25)                                     # 1 / n^(1/4)
    ldam_margins  = m_raw * (cfg.ldam_margin / m_raw.max())                 #  C

    # === RW: 1 / n  1===
    rw_weights_per_class = None
    if cfg.use_rw:
        rw_weights_per_class = (counts_t.sum() / counts_t)                  # ∝ 1/n
        rw_weights_per_class = rw_weights_per_class / rw_weights_per_class.mean()

    # === CB: 1 / E(n)  1===
    cb_weights_per_class = None
    if cfg.use_cb:
        beta = cfg.cb_beta
        # pow  beta counts_t device 
        effective_num = (1.0 - torch.pow(torch.tensor(beta, device=device), counts_t)) / (1.0 - beta)
        cb_weights_per_class = 1.0 / effective_num
        cb_weights_per_class = cb_weights_per_class / cb_weights_per_class.mean()

    # === RS numpy  ===
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
        
    #  batch_size  batch
    drop_last = (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=shuffle,
                              num_workers=cfg.workers, pin_memory=True, sampler=sampler,
                              drop_last=drop_last,          # ← 
                             )
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.workers, pin_memory=True,
                              drop_last=False,              # 
                              )

    # 
    # return train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins
    return train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins, hmt_groups


# -----------------------------
# Train / Eval
# -----------------------------

def aligned_contrastive_loss(z, y, t=0.07):
    """
     ACL
    """
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / t        # [B, B] 
    labels = y.unsqueeze(0) == y.unsqueeze(1)  # [B, B] 
    mask_pos = labels.fill_diagonal_(False)    # 

    # 
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
    rw_weights_per_class: Optional[torch.Tensor] = None,  # <—— 
    cb_weights_per_class: Optional[torch.Tensor] = None,   # CB  1/E(n)
    cb_loss: str = "ce",                                    # "ce" or "focal"
    cb_gamma: float = 2.0,                                  # focal  gamma
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
        assert log_prior is not None, "use_bsm=True  log_priorshape=[C]"
        #  [B, C]  (B,C) + (C,)  log_prior.unsqueeze(0)
        log_prior = log_prior.to(device)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)  # targets  [B, C] 

        logits = model(images)
        # loss = F.cross_entropy(logits, targets,
        #                        label_smoothing=label_smoothing,
        #                        reduction="mean")
        # -------- “” per-sample --------
        if mixup_fn is not None:
            # 
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
                    assert ldam_margins is not None, "use_ldam=True  ldam_marginsshape=[C]"
                    #  logits  margin
                    logits_m = logits.clone()
                    index = torch.zeros_like(logits_m, dtype=torch.bool)
                    index.scatter_(1, targets.unsqueeze(1), True)           #  True
                    margins = ldam_margins.to(device)[targets]               # [B]
                    logits_m[index] = logits_m[index] - margins              # y  m_y

                    s = ldam_scale
                    per_sample_loss = F.cross_entropy(
                        s * logits_m, targets, reduction="none",
                        label_smoothing=label_smoothing
                    )  # [B]
                elif (cb_weights_per_class is not None) and (cb_loss == "focal"):
                    # CB + Focal per-sample focal * CE
                    with torch.no_grad():
                        pt = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
                    focal_factor = torch.pow(1.0 - pt, cb_gamma)  # [B]
                    base_ce = F.cross_entropy(
                        logits, targets, reduction="none", label_smoothing=label_smoothing
                    )  # [B]
                    per_sample_loss = focal_factor * base_ce  # [B]
                else:
                    #  CE CE / RW / CB-CE
                    per_sample_loss = F.cross_entropy(
                        logits, targets, reduction="none", label_smoothing=label_smoothing
                    )  # [B]

                # -------- “” --------
                if cb_weights_per_class is not None:
                    # ✅ CB 1 / E(n) build_dataloaders 
                    w = cb_weights_per_class[targets]  # [B]
                    loss = (w * per_sample_loss).mean()
                elif rw_weights_per_class is not None:
                    # ✅ RW 1 / n build_dataloaders 
                    w = rw_weights_per_class[targets]  # [B]
                    loss = (w * per_sample_loss).mean()
                else:
                    # ✅  CE
                    loss = per_sample_loss.mean()
                    
        if use_acl and proj_head is not None and (mixup_fn is None):
            if hasattr(model, "forward_features"):
                feats = model.forward_features(images)   #  [B, C] / [B, 197, C] / [B, C, H, W]
                # ——  [B, D]  ——
                if feats.ndim == 3:          # ViT: [B, N, C]N=197
                    #  CLS ViT  CLS
                    feats = feats[:, 0, :]   #  CLS token
                    #  feats = feats.mean(dim=1)
                elif feats.ndim == 4:        # CNN: [B, C, H, W]
                    feats = feats.mean(dim=(2, 3))   # GAP
                # else:  [B, D]
            else:
                #  forward_features  logits  noisy
                feats = logits.detach()

            z = proj_head(feats)                         # [B, acl_dim]
            loss_acl = aligned_contrastive_loss(z, targets, t=acl_temp)
            loss = loss + acl_lambda * loss_acl        

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if use_wb and wb_renorm_target != "none":
            clf = get_classifier_module(model)
            #  target_norm
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
     GML
      L = -(1/C) * sum_c log( pbar_c )
      pbar_c = mean_j  tilde_p_y(j)  over samples with class c in this mini-batch
      tilde_p = softmax(logits + log(N))   # N 
    : CVPR'23 NOLBEq. (1)(2)(3) 
    """
    #  [B,C,H,W] / [B,T,C] [B,C]
    if logits.ndim > 2:
        #  batch  class  H=W=1
        dims = tuple(range(2, logits.ndim))
        logits = logits.mean(dim=dims)

    # reweighted softmax: softmax(o + log N)
    logits_rw = logits + log_counts.view(1, -1)               # [B,C]
    p_tilde = torch.softmax(logits_rw, dim=1)                  # [B,C]

    #  batch  p_{y}
    B, C = logits.shape
    present_classes = targets.unique()
    pbar_list = []
    for c in present_classes:
        mask = (targets == c)
        #  p_tilde 
        pc = p_tilde[mask, c]                                  # [Nc]
        #  log(0)
        pbar_c = pc.mean().clamp_min(eps)
        pbar_list.append(pbar_c)

    if len(pbar_list) == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    #  1/C 1/|present| 
    #  C 
    loss = -torch.log(torch.stack(pbar_list)).mean()           # / |present|
    #  loss = -torch.log(torch.stack(pbar_list)).sum() / float(num_classes)
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
     backbone GML  epoch
    (new_head, old_head) 
    """
    model.eval()
    # 1)  ensemble
    old_head = None
    if hasattr(model, "get_classifier"):
        old_head = nn.Sequential(*[m for m in [model.get_classifier()] if m is not None])
    else:
        #  timm  head / fc / classifier 
        old_head = nn.Sequential(*[getattr(model, k) for k in ["head", "fc", "classifier"] if hasattr(model, k)])

    # 2)  backbone
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    # 3) timm 
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        new_head = model.get_classifier()
    else:
        #  "head" 
        in_dim = model.get_classifier().in_features
        setattr(model, "head", nn.Linear(in_dim, num_classes))
        new_head = model.head

    # 
    for p in new_head.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(new_head.parameters(), lr=lr, weight_decay=weight_decay)

    #  log(N)
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
            logits = model(images)  #  reset_classifier 
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
            #  ->  -> softmax -> 
            feats = model.forward_features(images)
            logits_new = model.get_classifier()(feats)   # 
            logits_old = old_head(feats)                 #  GML 
            probs = 0.5 * _softmax_temp(logits_new, t_new) + 0.5 * _softmax_temp(logits_old, t_old)
            preds = probs.argmax(dim=1)
        else:
            # 
            logits = model(images)
            preds = logits.argmax(dim=1)
                # —— “” ——
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
    # 
    parser.add_argument("--no-pretrained", action="store_true",
                        help=" timm ")
    parser.add_argument("--init", type=str, default="",
                        help=".pth/.bin --no-pretrained ")
    parser.add_argument("--use-rw", action="store_true",
                    help="(Re-Weighting) CE")
    parser.add_argument("--use-cb", action="store_true",
                    help=" Class-Balanced Loss --use-rw / --balanced-sampler ")
    parser.add_argument("--cb-beta", type=float, default=0.999, 
                    help="CB loss  beta 0.99~0.9999")
    parser.add_argument("--cb-loss", type=str, default="ce", choices=["ce","focal"],
                    help="CB  CE  Focal")
    parser.add_argument("--cb-gamma", type=float, default=2.0,
                    help="CB+Focal  gamma")
    parser.add_argument("--use-bsm", action="store_true",
                        help="Balanced Softmax loss: CE(logits + log(count)). "
                            " RS/RW/CB ")
    parser.add_argument("--use-ldam", action="store_true",
                    help=" LDAM DRW")
    parser.add_argument("--ldam-margin", type=float, default=0.5,
                        help="LDAM  margin C 0.5")
    parser.add_argument("--ldam-scale", type=float, default=30.0,
                        help="LDAM  s 30")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, >0 ")
    parser.add_argument("--cutmix", type=float, default=0.0, help="cutmix alpha, >0 ")
    parser.add_argument("--mix-prob", type=float, default=1.0, help=" mix ")
    parser.add_argument("--mix-switch-prob", type=float, default=0.0, help="")
    parser.add_argument("--hmt", action="store_true", help=" Head/Medium/Tail  train ")
    parser.add_argument("--head-th", type=int, default=100, help="Head > head_th")
    parser.add_argument("--tail-th", type=int, default=20,  help="Tail < tail_th")
    parser.add_argument("--sched", type=str, default="cosine", choices=["none","cosine"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--gml", action="store_true", help=" NOLB GML")
    parser.add_argument("--gml-epochs", type=int, default=10, help="GML  few epochs")
    parser.add_argument("--ensemble", action="store_true", help=" old/new ")
    parser.add_argument("--t-old", type=float, default=1.0, help="old ")
    parser.add_argument("--t-new", type=float, default=1.0, help="new ")
    parser.add_argument("--use-wb", action="store_true",
                        help=" Weight BalancingWD + ")
    parser.add_argument("--wb-wd-mult", type=float, default=5.0,
                        help=" weight_decay  5x")
    parser.add_argument("--wb-max-norm", type=float, default=3.0,
                        help="L2None<=0 max-norm")
    parser.add_argument("--wb-renorm-alpha", type=float, default=1.0,
                        help=" alpha∈[0,1]10")
    parser.add_argument("--wb-renorm-target", type=str, default="mean",
                        choices=["mean","median","none"],
                        help="targetmean/median")
    parser.add_argument("--wb-renorm-freq", type=str, default="step",
                        choices=["step","epoch","none"],
                        help="stepepoch")
    parser.add_argument("--use-acl", action="store_true",
                        help=" Aligned Contrastive Loss (ACL)CE + ")
    parser.add_argument("--acl-lambda", type=float, default=0.1,
                        help="ACL  (λ) 0.05~0.2")
    parser.add_argument("--acl-temp", type=float, default=0.07,
                        help="ACL  temperature")
    parser.add_argument("--acl-dim", type=int, default=128,
                        help="ACL ")





    args = parser.parse_args()
    
    if args.use_rw and args.balanced_sampler:
        print("[WARN]  RS(--balanced-sampler)  RW(--use-rw) RS --use-rw RW --balanced-sampler")
    if args.use_cb and (args.use_rw or args.balanced_sampler):
        print("[WARN] CB  RW/RS ")
    if args.use_ldam and (args.use_rw or args.use_cb or args.balanced_sampler or args.use_bsm):
        print("[WARN] LDAM  RS/RW/CB/BSM ")



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
        hmt=args.hmt, head_th=args.head_th, tail_th=args.tail_th,
        gml=args.gml,
        gml_epochs=args.gml_epochs,
        ensemble=args.ensemble,
        t_old=args.t_old,
        t_new=args.t_new,

    )
    
    # =====  MixUp/CutMix/ =====
    use_mix = (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0)
    if use_mix and (cfg.use_rw or cfg.use_cb or cfg.use_ldam or cfg.use_bsm or cfg.use_balanced_sampler):
        print("[WARN] MixUp/CutMix  RS/RW/CB/LDAM/BSM")
        cfg.use_balanced_sampler = False
        cfg.use_rw = cfg.use_cb = cfg.use_ldam = cfg.use_bsm = False

    set_seed(cfg.seed)


    device = cfg.device
    # train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins = build_dataloaders(cfg)
    train_loader, val_loader, num_classes, rw_weights_per_class, cb_weights_per_class, log_prior, ldam_margins, hmt_groups = build_dataloaders(cfg)
    
    # =====  mixup_fn num_classes=====
    mixup_fn = None
    if (cfg.mixup_alpha > 0.0) or (cfg.cutmix_alpha > 0.0):
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mix_prob,
            switch_prob=cfg.switch_prob,
            mode="batch",
            label_smoothing=cfg.label_smoothing,  #  MixUp  0 
            num_classes=num_classes,
        )

    # 
    model = build_model(cfg.model_name, num_classes=num_classes,
                        pretrained=(not args.no_pretrained)).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {n_params/1e6:.2f}M | Trainable: {n_trainable/1e6:.2f}M")
    
    # ===== ACL=====
    proj_head = None
    if args.use_acl:
        #  backbone
        # feat_dim = getattr(model.get_classifier(), "in_features", 768)
        clf = get_classifier_module(model)
        feat_dim = getattr(clf, "in_features", None)
        if feat_dim is None and hasattr(clf, "weight"):
            feat_dim = clf.weight.shape[1]
        if feat_dim is None:
            feat_dim = 768
        proj_head = ProjectionHead(in_dim=feat_dim, out_dim=args.acl_dim).to(device)

    # 
    if args.init:
        load_checkpoint_flex(model, args.init, keep_head=True)
        
        # =======  GML/ backbone =======
    if cfg.gml:
        #  hmt_groups 
        if hmt_groups is not None and "counts" in hmt_groups:
            counts_np = np.array(hmt_groups["counts"], dtype=np.int64)
        else:
            counts_np = count_by_class(datasets.ImageFolder(cfg.train_dir))

        #  GML
        new_head, old_head = finetune_classifier_with_gml(
            model, train_loader, counts_np, num_classes, device,
            epochs=cfg.gml_epochs, lr=max(cfg.lr, 1e-3), weight_decay=0.0
        )

        # 
        if cfg.ensemble:
            evaluate._use_ensemble = True
            evaluate._old_head = old_head.eval() if old_head is not None else None
            evaluate._t_old = cfg.t_old
            evaluate._t_new = cfg.t_new

        # 
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


    # 
    # if cfg.opt == "adamw":
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
    #                                 weight_decay=cfg.weight_decay, nesterov=True)
    
    # ===== Weight Balancing param group=====
    # 1) 
    if cfg.opt == "adamw":
        OptimCls = torch.optim.AdamW
    elif cfg.opt == "sgd":
        OptimCls = torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer: {cfg.opt}")

    if args.use_wb:
        # 2) “”“”
        head_prefixes = ("head", "fc", "classifier")  # timm 
        base_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(head_prefixes):
                head_params.append(p)
            else:
                base_params.append(p)

        if len(head_params) == 0:
            #  timm  base_params
            base_params = [p for p in model.parameters() if p.requires_grad]
            head_params = []

        # 3)  weight decaywb-wd-mult 
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
        #  WB“ param group”
        if cfg.opt == "adamw":
            optimizer = OptimCls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            optimizer = OptimCls(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay, nesterov=True)
            
            
    # =====  ACL  =====
    if args.use_acl and proj_head is not None:
        optimizer.add_param_group({"params": proj_head.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay})



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
            use_bsm=cfg.use_bsm,                # <<<  BSM
            log_prior=(log_prior if cfg.use_bsm else None),
            use_ldam=cfg.use_ldam,
            ldam_margins=ldam_margins,
            ldam_scale=cfg.ldam_scale,
            mixup_fn=mixup_fn,           # <<< 
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

        # “ / train_time”
        #  WeightedRandomSampler (RS)len(dataset)  epoch  num_samples=len(sample_weights)
        train_imgs = len(train_loader.dataset)
        ips = train_imgs / max(train_time, 1e-9)

        # GB
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

        # HMT 
        if cfg.hmt:
            log += (f" || H Acc/F1/Sens/WCA: {metrics['H_acc']*100:.2f}/{metrics['H_f1']*100:.2f}/"
                    f"{metrics['H_sens']*100:.2f}/{metrics['H_wca']*100:.2f} | "
                    f"M {metrics['M_acc']*100:.2f}/{metrics['M_f1']*100:.2f}/{metrics['M_sens']*100:.2f}/{metrics['M_wca']*100:.2f} | "
                    f"T {metrics['T_acc']*100:.2f}/{metrics['T_f1']*100:.2f}/{metrics['T_sens']*100:.2f}/{metrics['T_wca']*100:.2f}")

        print(log)
        
        if scheduler is not None:
            # timm  step  epoch torch  scheduler.step()
            scheduler.step(epoch)
        
                # ===== WB: epoch  epoch =====
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


        #  best MacroF1+MacroSens  WorstClassAcc
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
