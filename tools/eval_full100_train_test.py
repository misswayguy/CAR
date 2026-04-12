#!/usr/bin/env python3
import argparse, os, csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# CIFAR-100 =ID
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

def build_model(model_name: str, num_classes: int):
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

def smart_load_ckpt(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for k in ["model", "state_dict", "model_state", "module", "net"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]; break
    #  DataParallel  'module.' 
    state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
    msd = model.state_dict()
    filt = {k:v for k,v in state.items() if k in msd and v.shape == msd[k].shape}
    msg = model.load_state_dict(filt, strict=False)
    print(f"[load] matched={len(filt)}/{len(msd)}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model

def make_loader(root, img_size, batch_size, workers):
    #  CIFAR-10 224 + ImageNet 
    tfm = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root=root, transform=tfm)
    # ImageFolder “”“ID(0..99)”
    idx2official = [int(name) for name in ds.classes]   # e.g. ['0','1','10',...] -> [0,1,10,...]
    ds.target_transform = lambda y_letter: idx2official[y_letter]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return loader

@torch.no_grad()
def eval_split(model, loader, num_classes=100):
    device = next(model.parameters()).device
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)  # [gt, pred]
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)   # “ID”
        pred = model(x).argmax(1)
        for t, p in zip(y, pred):
            cm[t, p] += 1
    gt_per_class = cm.sum(1).clamp(min=1)
    acc_per_class = (cm.diag().float() / gt_per_class.float()).cpu().numpy()
    overall = (cm.diag().sum().float() / cm.sum().clamp(min=1).float()).item()
    counts = gt_per_class.cpu().numpy()
    return acc_per_class, overall, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--val-dir',   required=True)
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--model',     default='resnet18')   # timm 
    ap.add_argument('--img-size',  type=int, default=224)
    ap.add_argument('--batch-size',type=int, default=128)
    ap.add_argument('--workers',   type=int, default=4)
    ap.add_argument('--out',       required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(args.model, num_classes=100).to(device)
    model  = smart_load_ckpt(model, args.ckpt)

    train_loader = make_loader(args.train_dir, args.img_size, args.batch_size, args.workers)
    test_loader  = make_loader(args.val_dir,  args.img_size, args.batch_size, args.workers)

    tr_acc, tr_overall, tr_cnt = eval_split(model, train_loader, 100)
    te_acc, te_overall, te_cnt = eval_split(model, test_loader,  100)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id','class_name','train_acc','test_acc','train_count','test_count'])
        for cid in range(100):
            w.writerow([cid, CIFAR100_NAMES[cid],
                        f'{tr_acc[cid]:.4f}', f'{te_acc[cid]:.4f}',
                        int(tr_cnt[cid]), int(te_cnt[cid])])
        w.writerow(['OVERALL','-', f'{tr_overall:.4f}', f'{te_overall:.4f}', '-', '-'])
    print(f'[done] saved → {args.out}')
    print(f'overall (train/test): {tr_overall:.4f} / {te_overall:.4f}')

if __name__ == '__main__':
    main()
