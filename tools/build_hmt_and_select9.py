#!/usr/bin/env python3
import json

# CIFAR-100 
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

with open("class_counts.json") as f:
    counts = {int(k): int(v) for k, v in json.load(f).items()}

head = [i for i,c in counts.items() if c > 100]
med  = [i for i,c in counts.items() if 20 <= c <= 100]
tail = [i for i,c in counts.items() if c < 20]

def pick3(ids):
    ids_sorted = sorted(ids, key=lambda i: counts[i], reverse=True)
    pos = [0.1, 0.5, 0.9]
    idxs = [ids_sorted[int(p*(len(ids_sorted)-1))] for p in pos]
    return idxs

sel_head = pick3(head)
sel_med  = pick3(med)
sel_tail = pick3(tail)
selected = sel_head + sel_med + sel_tail

out = {
  "threshold": { "head": ">100", "medium": "20-100", "tail": "<20" },
  "groups": { "head": head, "medium": med, "tail": tail },
  "selected_ids": selected,
  "selected_names": [CIFAR100_NAMES[i] for i in selected]
}
with open("hmt_and_selected9.json", "w") as f:
    json.dump(out, f, indent=2)

print("H/M/T sizes:", len(head), len(med), len(tail))
print("Selected 9:", out["selected_names"])
