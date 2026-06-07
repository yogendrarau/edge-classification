"""Phase 2b: quantization-aware training (QAT) of MobileNetV2.

PTQ gave a ~6pt top-1 drop. QAT inserts fake-quant ops during a short fine-tune so
the model learns weights robust to INT8 rounding, recovering most of that drop.

Flow: prepare_qat_fx (x86 qconfig, reduce_range=True) -> fine-tune on Imagenette
train split on GPU -> validate each epoch, keep best -> convert to INT8 -> measure on CPU.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

WNID_TO_IMAGENET_IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}
# The 10 ImageNet indices Imagenette covers, for remapping the 1000-way head.
IMAGENETTE_INDICES = sorted(WNID_TO_IMAGENET_IDX.values())


def build_loaders(root, preprocess, batch_size):
    # Light augmentation on train helps QAT generalize; val uses the eval transform.
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


def remap_targets(folder_labels, f2i, device):
    # ImageFolder gives 0..9; map to the true ImageNet index for loss/accuracy.
    return torch.tensor([f2i[int(l)] for l in folder_labels], device=device)


@torch.no_grad()
def evaluate(model, loader, f2i, device):
    model.eval()
    top1 = top5 = total = 0
    for images, folder_labels in loader:
        images = images.to(device)
        logits = model(images).cpu()
        true_idx = torch.tensor([f2i[int(l)] for l in folder_labels])
        top1 += (logits.argmax(1) == true_idx).sum().item()
        top5 += (logits.topk(5, 1).indices == true_idx.unsqueeze(1)).any(1).sum().item()
        total += len(folder_labels)
    return top1 / total, top5 / total


def measure_latency_cpu(model, n_warmup=10, n_runs=100):
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        times = []
        for _ in range(n_runs):
            t = time.perf_counter()
            model(x)
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
    ap.add_argument("--data", required=True, help="imagenette root (contains train/ and val/)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)  # small LR: we're fine-tuning, not training
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train device {device}")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    train_loader, val_loader, f2i_train, f2i_val = build_loaders(args.data, preprocess, args.batch_size)

    fp32 = mobilenet_v2(weights=weights)

    # Prepare for QAT: insert fake-quant + observers. x86 qconfig -> reduce_range=True,
    # the setting that fixed our oneDNN saturation on this non-VNNI CPU.
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    example_input = torch.randn(1, 3, 224, 224)
    qat_model = prepare_qat_fx(fp32, qconfig_mapping, example_input).to(device)

    optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        qat_model.train()
        running = 0.0
        for images, folder_labels in train_loader:
            images = images.to(device)
            targets = remap_targets(folder_labels, f2i_train, device)
            optimizer.zero_grad()
            loss = criterion(qat_model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item()

        # validate (fake-quant model on GPU approximates the final INT8 accuracy)
        top1, top5 = evaluate(qat_model, val_loader, f2i_val, device)
        print(f"epoch {epoch}/{args.epochs} | loss {running/len(train_loader):.3f} "
              f"| val top1 {top1*100:.2f}% top5 {top5*100:.2f}%")
        if top1 > best_top1:
            best_top1 = top1
            best_state = copy.deepcopy(qat_model.state_dict())

    # restore best epoch before converting
    qat_model.load_state_dict(best_state)
    print(f"\nbest val top1 during QAT: {best_top1*100:.2f}%")

    # convert to real INT8 and measure on CPU (fair comparison vs PTQ)
    qat_model.to("cpu").eval()
    int8 = convert_fx(qat_model)

    print("\n== INT8 QAT (CPU) ==")
    size = measure_size(int8)
    lat = measure_latency_cpu(int8)
    top1, top5 = evaluate(int8, val_loader, f2i_val, torch.device("cpu"))
    print(f"size {size:.2f} MB | latency median {lat['median_ms']:.2f} ms "
          f"| {lat['fps']:.1f} FPS | top1 {top1*100:.2f}% top5 {top5*100:.2f}%")

    torch.save(int8.state_dict(), "mobilenetv2_int8_qat.pt")
    print("saved mobilenetv2_int8_qat.pt")


if __name__ == "__main__":
    main()