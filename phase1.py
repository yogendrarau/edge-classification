"""Phase 1: baseline metrics for unmodified MobileNetV2 — accuracy, latency, size.

Dataset: Imagenette (10-class ImageNet subset, full-size images). The model still
predicts over all 1000 ImageNet classes; we map each Imagenette folder to its
ImageNet class index and score a hit when that index wins.
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Imagenette folder (WordNet) IDs -> the ImageNet class index that folder represents.
# Order matches Imagenette's directory layout.
WNID_TO_IMAGENET_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def get_loader(data_dir, preprocess, batch_size=32):
    # ImageFolder assigns folder names to 0..9 alphabetically; remap to ImageNet indices.
    ds = datasets.ImageFolder(data_dir, transform=preprocess)
    folder_to_imagenet = {
        ds.class_to_idx[wnid]: imagenet_idx
        for wnid, imagenet_idx in WNID_TO_IMAGENET_IDX.items()
        if wnid in ds.class_to_idx
    }
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, folder_to_imagenet


def measure_accuracy(model, loader, folder_to_imagenet, device):
    # Top-1 is the strict headline metric; top-5 is the sanity check that absorbs
    # ImageNet near-synonym classes (cassette/tape player, church/monastery).
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, folder_labels in loader:
            images = images.to(device)
            logits = model(images).cpu()
            true_idx = torch.tensor([folder_to_imagenet[int(l)] for l in folder_labels])

            top1 += (logits.argmax(dim=1) == true_idx).sum().item()
            top5_idx = logits.topk(5, dim=1).indices
            top5 += (top5_idx == true_idx.unsqueeze(1)).any(dim=1).sum().item()
            total += len(folder_labels)
    return top1 / total, top5 / total


def measure_latency(model, device, n_warmup=20, n_runs=200):
    # Single-image latency is the number that maps to real-time use.
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):  # warm caches / lazy CUDA init before timing
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()  # GPU calls are async; wait before stopping clock
            times.append((time.perf_counter() - start) * 1000.0)  # ms

    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(times.mean()),
        "fps": 1000.0 / float(np.median(times)),
    }


def measure_size(model, path="_tmp_size.pt"):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to imagenette val dir (folders = classes)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    print(f"device {device}")

    size_mb = measure_size(model)
    print(f"size   {size_mb:.2f} MB")

    lat = measure_latency(model, device)
    print(f"latency median {lat['median_ms']:.2f} ms | p95 {lat['p95_ms']:.2f} ms | {lat['fps']:.1f} FPS")

    loader, folder_to_imagenet = get_loader(args.data, preprocess, args.batch_size)
    top1, top5 = measure_accuracy(model, loader, folder_to_imagenet, device)
    print(f"accuracy top1 {top1 * 100:.2f}% | top5 {top5 * 100:.2f}%")


if __name__ == "__main__":
    main()