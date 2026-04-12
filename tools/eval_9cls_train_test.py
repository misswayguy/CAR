#!/usr/bin/env python3
import argparse, os, csv
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm

DEFAULT_DATA_ROOT = "/mnt/data/lsy/ZZQ/cifar100-LT-IF100"

CIFAR100_NAMES = [
 'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
 'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
 'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
 'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard',
 'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
 'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
 'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
 'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
 'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
 'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

# ----------------- timm +  -----------------
def _strip_prefix(state, prefix):
    if not prefix: return state
    plen = len(prefix)
    return { (k[plen:] if k.startswith(prefix) else k): v for k, v in state.items() }

def _normalize_head_keys(k: str) -> str:
    k2 = k
    k2 = k2.replace("classifier.", "head.")
    k2 = k2.replace("fc.", "head.fc.")
    if k2.startswith("head.weight"): k2 = k2.replace("head.weight", "head.fc.weight")
    if k2.startswith("head.bias"):   k2 = k2.replace("head.bias",   "head.fc.bias")
    return k2

def _best_match_state(model_keys, raw_state):
    prefixes = [
        "", "state_dict.", "model.", "net.", "backbone.", "encoder.",
        "module.", "module.model.", "module.net.", "module.backbone.", "module.encoder.",
        "model.backbone.", "module.model.backbone.",
    ]
    best = None; best_n = -1
    for p in prefixes:
        s = _strip_prefix(raw_state, p)
        s = { _normalize_head_keys(k): v for k, v in s.items() }
        n = sum(1 for k in s if k in model_keys)
        if n > best_n:
            best_n = n; best = (p, s)
    return best

def build_model_and_load(ckpt_path, model_name="resnet18", num_classes=100):
    m = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items() }
    used_prefix, stripped = _best_match_state(set(m.state_dict().keys()), sd)
    msd = m.state_dict()
    filtered = {k:v for k,v in stripped.items() if k in msd and v.shape == msd[k].shape}
    msg = m.load_state_dict(filtered, strict=False)
    matched = sum(1 for k in filtered if k in msd)
    cover = matched / len(msd) * 100
    print(f"[load] model={model_name} prefix_used='{used_prefix}' "
          f"matched={matched}/{len(msd)} ({cover:.1f}%) "
          f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    m.eval()
    return m

# ----------------- “ID” -----------------
def parse_ids(ids_str):
    ids = [int(x) for x in ids_str.replace(' ','').split(',') if x!='']
    for i in ids: assert 0 <= i < 100, f"class id {i} out of range [0,99]"
    return ids

def make_loader(split, sel_official_ids, data_root, batch_size=256, num_workers=4,
                input_preset="imagenet"):
    """
    ImageFolder  ''  'ID'
     input_preset 
    """
    root = os.path.join(data_root, split)

    if input_preset == "imagenet":
        # 
        img_size = 224
        tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
    elif input_preset == "cifar":
        # CIFAR 32x32 + CIFAR100  mean/std
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    else:
        raise ValueError(f"Unknown input_preset: {input_preset}")

    ds_raw = datasets.ImageFolder(root, transform=tf)  # y 
    idx2official = [int(name) for name in ds_raw.classes]  #  -> ID

    # ID
    sel_set = set(sel_official_ids)
    keep = [i for i, (_, y_letter) in enumerate(ds_raw.samples)
            if idx2official[y_letter] in sel_set]

    # ID
    ds_raw.target_transform = lambda y_letter: idx2official[y_letter]

    sub = Subset(ds_raw, keep)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ID
    from collections import Counter
    cnt = Counter([idx2official[y] for _, y in [ds_raw.samples[i] for i in keep]])
    print(f"[{split}] filtered samples: {len(keep)}; official_ids={sel_official_ids}")
    print("totals(official): {", ", ".join(f"{k}: {cnt.get(k,0)}" for k in sel_official_ids), "}")
    return loader


# ----------------- ID -----------------
@torch.no_grad()
def per_class_acc(model, loader, device, sel_official_ids):
    correct = defaultdict(int); total = defaultdict(int)
    sel_set = set(sel_official_ids)
    for x, y in loader:              #  y “ID”
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        eq = (pred == y)
        for cid in sel_set:
            m = (y == cid)
            if m.any():
                total[cid]   += int(m.sum().item())
                correct[cid] += int(eq[m].sum().item())
    print("totals(official, counted):", {cid: total[cid] for cid in sel_official_ids})
    return {cid: (correct[cid] / total[cid] if total[cid] > 0 else 0.0) for cid in sel_official_ids}

# -----------------  -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", required=True, help='OFFICIAL numeric IDs, e.g. "24,9,27,32,49,64,72,84,95"')
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--arch", default="resnet18")  # timm 
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--with_names", action="store_true")
    ap.add_argument("--input_preset", default="imagenet", choices=["imagenet","cifar"])

    args = ap.parse_args()

    sel_official = parse_ids(args.ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_and_load(args.ckpt, model_name=args.arch, num_classes=100).to(device)

    tr_loader = make_loader("train", sel_official, args.data_root, args.batch_size,
                            input_preset=args.input_preset)
    te_loader = make_loader("test",  sel_official, args.data_root, args.batch_size,
                            input_preset=args.input_preset)

    acc_tr = per_class_acc(model, tr_loader, device, sel_official)
    acc_te = per_class_acc(model, te_loader, device, sel_official)

    # ID
    with open(args.out_csv, "w", newline="") as f:
        hdr = ["class_id","train_acc","test_acc"] + (["class_name"] if args.with_names else [])
        w = csv.writer(f); w.writerow(hdr)
        for cid in sel_official:
            row = [cid, f"{acc_tr.get(cid,0.0):.4f}", f"{acc_te.get(cid,0.0):.4f}"]
            if args.with_names: row.append(CIFAR100_NAMES[cid])
            w.writerow(row)
    print("Saved:", args.out_csv)
    
    
    # 
    print(f"Device: {device}")
    print(f"Selected class IDs: {sel_official}")
    print(f"Checkpoint path: {args.ckpt}")
    print(f"Data root: {args.data_root}")
    
    # 
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")
    print(f"Train dir exists: {os.path.exists(train_dir)}")
    print(f"Test dir exists: {os.path.exists(test_dir)}")
    
    # 
    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        print(f"Checkpoint keys: {ckpt.keys() if isinstance(ckpt, dict) else 'Not a dict'}")
    else:
        print(f"Checkpoint file not found: {args.ckpt}")
        return

if __name__ == "__main__":
    main()
