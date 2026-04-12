#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, csv, sys
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-100 fine labels0..99
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
FINE2ID = {n:i for i,n in enumerate(CIFAR100_FINE)}

def read_acc_csv(path):
    """ per-class acc dict[key]->acckey  CSV """
    d = {}
    with open(path, 'r') as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if not row or len(row) < 3: continue
            key = row[1].strip()
            d[key] = float(row[2])
    return d

def resolve_key(token, keys_set):
    """
     token  CSV  key
    1) 
    2) key
    3)  ->  fine name  name  key
    4)  fine name ->  id 
    """
    t = token.strip()
    if t in keys_set: return t
    if t.isdigit() and t in keys_set: return t
    if t.isdigit():
        i = int(t)
        if 0 <= i < 100:
            nm = CIFAR100_FINE[i]
            if nm in keys_set: return nm
    # fine name -> id string
    if t in FINE2ID:
        id_str = str(FINE2ID[t])
        if id_str in keys_set: return id_str
        if t in keys_set: return t
    #  not found
    return t

def to_display_name(key):
    """ CSV  key fine name"""
    k = key.strip()
    if k.isdigit():
        i = int(k)
        if 0 <= i < 100: return CIFAR100_FINE[i]
    #  fine name
    return k

def radar_plot(labels, train_vals, test_vals, out_path, title=''):
    plt.rcParams['figure.constrained_layout.use'] = False
    N = len(labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta_closed = np.concatenate([theta, [theta[0]]])
    tr = np.concatenate([train_vals, [train_vals[0]]])
    te = np.concatenate([test_vals,  [test_vals[0]]])

    fig = plt.figure(figsize=(6.4, 5.8), dpi=220)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.72)

    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    l1, = ax.plot(theta_closed, tr, linewidth=2.0, marker='o', label='Acc on Training Set')
    l2, = ax.plot(theta_closed, te, linewidth=2.0, marker='o', label='Acc on Test Set')

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
    ap.add_argument('--train-csv', required=True, help='..._train_acc.csv')
    ap.add_argument('--test-csv',  required=True, help='..._test_acc.csv')
    ap.add_argument('--classes',   required=True,
                    help='9IDfine name"baby,boy,bear,otter,possum,plate,woman,whale,turtle"  "2,11,3,55,66,63,98,96,92"')
    ap.add_argument('--out',       required=True)
    ap.add_argument('--title',     default='')  # RS / CE
    args = ap.parse_args()

    tr = read_acc_csv(args.train_csv)
    te = read_acc_csv(args.test_csv)
    keys_set = set(tr.keys()) & set(te.keys())
    if len(keys_set) == 0:
        print("[error] train/test CSV ", file=sys.stderr); sys.exit(1)

    tokens = [t for t in args.classes.split(',') if t.strip()]
    if len(tokens) != 9:
        print("[error] 9--classes", file=sys.stderr); sys.exit(1)

    #  CSV key
    resolved_keys, display_names = [], []
    not_found = []
    for t in tokens:
        k = resolve_key(t, keys_set)
        if k not in keys_set:
            not_found.append(t)
        else:
            resolved_keys.append(k)
            display_names.append(to_display_name(k))
    if not_found:
        print(f"[error] CSV{not_found}", file=sys.stderr); sys.exit(1)

    tr_vals = np.array([tr[k] for k in resolved_keys], dtype=float)
    te_vals = np.array([te[k] for k in resolved_keys], dtype=float)

    #  CSV
    sel_csv = os.path.splitext(args.out)[0] + "_fixed9.csv"
    with open(sel_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class_name','train_acc','test_acc'])
        for nm, a, b in zip(display_names, tr_vals, te_vals):
            w.writerow([nm, f"{a:.6f}", f"{b:.6f}"])
    print(f"Saved fixed-classes CSV to: {sel_csv}")

    radar_plot(display_names, tr_vals, te_vals, args.out, title=args.title)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr); sys.exit(1)
