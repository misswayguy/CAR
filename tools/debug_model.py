#!/usr/bin/env python3
import argparse, os, csv
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

def debug_model_loading(ckpt_path, arch="resnet18", num_classes=100):
    """"""
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    
    print("=" * 50)
    print("MODEL LOADING DEBUG INFO")
    print("=" * 50)
    
    # checkpoint
    sd = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint keys: {list(sd.keys())}")
    
    # state_dict
    state_dict = None
    if 'state_dict' in sd:
        state_dict = sd['state_dict']
        print("Using 'state_dict' from checkpoint")
    elif 'model' in sd:
        state_dict = sd['model'] 
        print("Using 'model' from checkpoint")
    else:
        state_dict = sd
        print("Using root level as state_dict")
    
    print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")
    
    # key
    state_dict_clean = {}
    for k, v in state_dict.items():
        # prefix
        new_k = k.replace('module.', '').replace('model.', '').replace('backbone.', '')
        state_dict_clean[new_k] = v
    
    print("Cleaned keys (first 10):", list(state_dict_clean.keys())[:10])
    
    # 
    try:
        missing, unexpected = model.load_state_dict(state_dict_clean, strict=True)
        print("✓ Strict loading successful!")
    except:
        print("✗ Strict loading failed, trying non-strict...")
        missing, unexpected = model.load_state_dict(state_dict_clean, strict=False)
    
    print(f"Missing keys: {len(missing)}")
    if missing:
        print(f"First 5 missing: {list(missing)[:5]}")
    print(f"Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"First 5 unexpected: {list(unexpected)[:5]}")
    
    # 
    print("\nWeight statistics:")
    for name, param in model.named_parameters():
        if 'weight' in name and param.numel() > 0:
            print(f"{name:30} mean: {param.data.mean().item():10.6f} std: {param.data.std().item():10.6f}")
            break
    
    model.eval()
    return model

def main_debug():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--data_root", default="/mnt/data/lsy/ZZQ/cifar100-LT-IF100")
    args = ap.parse_args()
    
    # 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = debug_model_loading(args.ckpt).to(device)
    
    # 
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nDummy input output: {output[0][:5]}")  # 5logits
        print(f"Output softmax: {torch.softmax(output[0], dim=0)[:5]}")

if __name__ == "__main__":
    main_debug()