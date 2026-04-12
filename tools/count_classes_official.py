#!/usr/bin/env python3
import json, os
from collections import Counter
from torchvision import datasets

DATA_ROOT = "/mnt/data/lsy/ZZQ/cifar100-LT-IF100"

def main():
    train = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"))
    # ImageFolder  '0'..'99'
    #  ->  int ID
    idx2official = [int(name) for name in train.classes]  # len=100

    # ID
    counts = Counter()
    for _, y_letter in train.samples:      # y_letter: 
        cid = idx2official[y_letter]       # 
        counts[cid] += 1

    counts = {int(k): int(v) for k, v in counts.items()}
    with open("class_counts.json", "w") as f:
        json.dump(counts, f, indent=2)
    print("Saved class_counts.json using OFFICIAL indices. Seen classes:", len(counts))

if __name__ == "__main__":
    main()
