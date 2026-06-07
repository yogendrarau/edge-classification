"""Phase 2b control: FP32 fine-tuned baseline (no quantization).

QAT reached 97% but that mixes two effects: fine-tuning AND quantization. This script
isolates the fine-tuning effect by training the SAME model on the SAME data for the
SAME epochs WITHOUT quantization. The gap between this and QAT-INT8 is the true
quantization cost (expected: near zero).

Mirrors phase2_qat.py exactly except: no prepare_qat_fx, no convert_fx.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

WNID_TO_IMAGENET_IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


def build_loaders(root, preprocess, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=preprocess)
    f2i_train = {train_ds.class_to_idx[w]: i for w, i in WNID_TO_IMAGENET_IDX.items()
                 if w in train_ds.class_to_idx}
    f2i_val = {val_ds.class_to_idx[w]: i for w, i in WNID_TO_IMAGENET_IDX.items()
               if w in val_ds.class_to_idx}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, f2i_train, f2i_val


@torch.no_grad()
def evaluate(model, loader, f2i, device):
    model.eval()
    top1 = top5 = total = 0
    for images, folder_labels in loader:
        logits = model(images.to(device)).cpu()
        true_idx = torch.tensor([f2i[int(l)] for l in folder_labels])
        top1 += (logits.argmax(1) == true_idx).sum().item()
        top5 += (logits.topk(5, 1).indices == true_idx.unsqueeze(1)).any(1).sum().item()
        total += len(folder_labels)
    return top1 / total, top5 / total


def measure_latency(model, device, n_warmup=20, n_runs=200):
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            t = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000.0)
    times = np.array(times)
    return {"median_ms": float(np.median(times)), "fps": 1000.0 / float(np.median(times))}


def measure_size(model, path="_tmp.pt"):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train device {device}")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    train_loader, val_loader, f2i_train, f2i_val = build_loaders(args.data, preprocess, args.batch_size)

    model = mobilenet_v2(weights=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, folder_labels in train_loader:
            images = images.to(device)
            targets = torch.tensor([f2i_train[int(l)] for l in folder_labels], device=device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
        top1, top5 = evaluate(model, val_loader, f2i_val, device)
        print(f"epoch {epoch}/{args.epochs} | loss {running/len(train_loader):.3f} "
              f"| val top1 {top1*100:.2f}% top5 {top5*100:.2f}%")
        if top1 > best_top1:
            best_top1 = top1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"\nbest val top1 (FP32 fine-tuned): {best_top1*100:.2f}%")

    print("\n== FP32 fine-tuned ==")
    size = measure_size(model)
    # measure latency on CPU too, to compare fairly against the INT8-CPU numbers
    model.cpu().eval()
    lat_cpu = measure_latency(model, torch.device("cpu"))
    top1, top5 = evaluate(model, val_loader, f2i_val, torch.device("cpu"))
    print(f"size {size:.2f} MB | latency CPU median {lat_cpu['median_ms']:.2f} ms "
          f"| {lat_cpu['fps']:.1f} FPS | top1 {top1*100:.2f}% top5 {top5*100:.2f}%")

    torch.save(model.state_dict(), "mobilenetv2_fp32_finetuned.pt")
    print("saved mobilenetv2_fp32_finetuned.pt")


if __name__ == "__main__":
    main()