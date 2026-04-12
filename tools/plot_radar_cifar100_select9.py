#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, csv, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

# CIFAR-100  fine labels 0..99
CIFAR100_FINE = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]

def id2name(s: str):
    """ '23' -> CIFAR100_FINE[23]"""
    s_ = s.strip()
    if s_.isdigit():
        i = int(s_)
        if 0 <= i < 100:
            return CIFAR100_FINE[i]
    return s_

def read_acc_csv(path):
    # class_id,class_name,acc
    d = {}
    with open(path, 'r') as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if not row or len(row) < 3: 
                continue
            cls_name = row[1].strip()
            acc = float(row[2])
            d[cls_name] = acc
    return d

def class_counts_from_train_dir(train_dir):
    ds = datasets.ImageFolder(train_dir)
    counts = defaultdict(int)
    for _, y in ds.samples:
        counts[ds.classes[y]] += 1
    return [(c, counts[c]) for c in ds.classes]

def split_head_medium_tail(count_list):
    arr = sorted(count_list, key=lambda x: x[1], reverse=True)
    n = len(arr)
    n_head = n // 3
    n_tail = n // 3
    n_med  = n - n_head - n_tail
    head = [x[0] for x in arr[:n_head]]
    medium = [x[0] for x in arr[n_head:n_head+n_med]]
    tail = [x[0] for x in arr[n_head+n_med:]]
    return head, medium, tail

def pick_worst_classes(groups, train_acc, test_acc, k=3):
    selected = []
    for g in groups:
        cand = []
        for name in g:
            if name in train_acc and name in test_acc:
                cand.append((name, train_acc[name], test_acc[name]))
        # accacc“”
        cand.sort(key=lambda x: (x[1], x[2]))
        selected.extend([x[0] for x in cand[:k]])
    return selected

def radar_plot(labels, train_vals, test_vals, out_path, title=''):
    plt.rcParams['figure.constrained_layout.use'] = False
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, [theta[0]]])
    train = np.concatenate([train_vals, [train_vals[0]]])
    test  = np.concatenate([test_vals,  [test_vals[0]]])

    fig = plt.figure(figsize=(6.4, 5.8), dpi=220)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.72)

    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    l1, = ax.plot(theta_closed, train, linewidth=2.0, marker='o', label='Acc on Training Set')
    l2, = ax.plot(theta_closed, test,  linewidth=2.0, marker='o', label='Acc on Test Set')

    imin_tr = int(np.argmin(train_vals))
    imin_te = int(np.argmin(test_vals))
    ax.plot(theta[imin_tr], train_vals[imin_tr], marker='*', markersize=12)
    ax.plot(theta[imin_te], test_vals[imin_te],  marker='*', markersize=12)

    ax.set_thetagrids(np.rad2deg(theta), labels, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0,0.2,0.4,0.6,0.8])
    ax.set_yticklabels(['0.0','0.2','0.4','0.6','0.8'], fontsize=9)
    ax.grid(True, alpha=0.6)

    title_artist = None
    if title:
        title_artist = fig.suptitle(title, fontsize=14, y=0.96)

    legend = fig.legend(handles=[l1, l2],
                        labels=['Acc on Training Set', 'Acc on Test Set'],
                        loc='center right',
                        bbox_to_anchor=(1.02, 0.5),
                        bbox_transform=fig.transFigure,
                        frameon=True, fontsize=10)
    legend.get_frame().set_alpha(0.85)
    legend.get_frame().set_linewidth(0.8)

    out_dir = os.path.dirname(out_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    extras = [legend] + ([title_artist] if title_artist else [])
    fig.savefig(out_path, bbox_inches='tight', bbox_extra_artists=extras)
    print(f"Saved figure to: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--test-csv',  required=True)
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--out',       required=True)
    ap.add_argument('--title',     default='')  #  "RS"  RS
    args = ap.parse_args()

    tr_acc = read_acc_csv(args.train_csv)
    te_acc = read_acc_csv(args.test_csv)

    counts = class_counts_from_train_dir(args.train_dir)
    head, med, tail = split_head_medium_tail(counts)
    selected = pick_worst_classes([head, med, tail], tr_acc, te_acc, k=3)

    labels, tr_vals, te_vals = [], [], []
    for nm in selected:
        labels.append(id2name(nm))          # <- 
        tr_vals.append(tr_acc[nm])
        te_vals.append(te_acc[nm])

    # 9CSV
    sel_csv = os.path.splitext(args.out)[0] + "_selected9.csv"
    with open(sel_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class_name','train_acc','test_acc'])
        for nm, tr, te in zip(labels, tr_vals, te_vals):
            w.writerow([nm, f"{tr:.6f}", f"{te:.6f}"])
    print(f"Saved selected CSV to: {sel_csv}")

    radar_plot(labels, np.array(tr_vals), np.array(te_vals), args.out, title=args.title)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr); sys.exit(1)
