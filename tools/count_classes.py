#!/usr/bin/env python3
import json, os
from collections import Counter
from torchvision import datasets

DATA_ROOT = "/mnt/data/lsy/ZZQ/cifar100-LT-IF100"
train = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"))

cnt = Counter([y for _, y in train.samples])  # y0..99
counts = {int(k): int(v) for k, v in cnt.items()}
with open("class_counts.json", "w") as f:
    json.dump(counts, f, indent=2)

print("Saved class_counts.json. Seen classes:", len(counts))
