"""Per-class accuracy diagnostic: isolate whether low accuracy is a real model
limitation (uniform) or a class-index mapping bug (a few classes near zero)."""

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

WNID_TO_IMAGENET_IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}
WNID_NAME = {
    "n01440764": "tench", "n02102040": "springer", "n02979186": "cassette",
    "n03000684": "chainsaw", "n03028079": "church", "n03394916": "french_horn",
    "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball",
    "n03888257": "parachute",
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = mobilenet_v2(weights=weights).to(device).eval()

ds = datasets.ImageFolder(args.data, transform=weights.transforms())
idx_to_wnid = {v: k for k, v in ds.class_to_idx.items()}
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

cats = weights.meta["categories"]
hit = defaultdict(int)
tot = defaultdict(int)
# Track what the model MOST often predicts per folder, to spot a wrong mapping.
votes = defaultdict(lambda: defaultdict(int))

with torch.no_grad():
    for images, folder_labels in loader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        for p, fl in zip(preds, folder_labels):
            wnid = idx_to_wnid[int(fl)]
            tot[wnid] += 1
            if int(p) == WNID_TO_IMAGENET_IDX[wnid]:
                hit[wnid] += 1
            votes[wnid][int(p)] += 1

print(f"{'class':16s} {'acc':>6s}   most-predicted index (count) -> name")
for wnid in WNID_TO_IMAGENET_IDX:
    acc = hit[wnid] / tot[wnid] if tot[wnid] else 0
    top_idx = max(votes[wnid], key=votes[wnid].get)
    print(f"{WNID_NAME[wnid]:16s} {acc*100:5.1f}%   expected {WNID_TO_IMAGENET_IDX[wnid]:4d}, "
          f"model says {top_idx:4d} ({votes[wnid][top_idx]}x) = {cats[top_idx]}")