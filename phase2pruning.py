"""Phase 2c: structured channel pruning of the FP32 fine-tuned MobileNetV2.

Structured (not unstructured) pruning: removes whole channels so the model is
genuinely smaller/faster on normal hardware — unlike weight masking, which only
helps with sparse kernels the browser won't use.

Torch-Pruning's DependencyGraph handles MobileNetV2's depthwise convs (whose channel
counts are coupled via `groups`); naive pruning throws "divisible by groups" errors.
We ignore the classifier head (pruning it would change the class count).

Flow: load fine-tuned FP32 -> prune ~30% channels -> retrain to recover -> measure.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

WNID_TO_IMAGENET_IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


def build_loaders(root, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(232), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=eval_tf)
    f2i_train = {train_ds.class_to_idx[w]: i for w, i in WNID_TO_IMAGENET_IDX.items() if w in train_ds.class_to_idx}
    f2i_val = {val_ds.class_to_idx[w]: i for w, i in WNID_TO_IMAGENET_IDX.items() if w in val_ds.class_to_idx}
    return (DataLoader(train_ds, batch_size, shuffle=True, num_workers=4),
            DataLoader(val_ds, batch_size, shuffle=False, num_workers=4), f2i_train, f2i_val)


@torch.no_grad()
def evaluate(model, loader, f2i, device):
    model.eval()
    top1 = total = 0
    for images, folder_labels in loader:
        logits = model(images.to(device)).cpu()
        true_idx = torch.tensor([f2i[int(l)] for l in folder_labels])
        top1 += (logits.argmax(1) == true_idx).sum().item()
        total += len(folder_labels)
    return top1 / total


def train_epochs(model, loader, f2i, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, folder_labels in loader:
            images = images.to(device)
            targets = torch.tensor([f2i[int(l)] for l in folder_labels], device=device)
            opt.zero_grad()
            loss = crit(model(images), targets)
            loss.backward()
            opt.step()
            running += loss.item()
        yield ep, running / len(loader)


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
    return float(np.median(times))


def measure_size(model, path="_tmp.pt"):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="imagenette root (train/ and val/)")
    ap.add_argument("--weights", default="mobilenetv2_fp32_finetuned.pt", help="fine-tuned FP32 weights")
    ap.add_argument("--ratio", type=float, default=0.3, help="channel pruning ratio")
    ap.add_argument("--recover-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    # load the fine-tuned dense model as the pruning starting point
    model = mobilenet_v2()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device)

    train_loader, val_loader, f2i_train, f2i_val = build_loaders(args.data, batch_size=32)

    base_size = measure_size(model)
    base_macs, base_params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, 224, 224).to(device))
    base_acc = evaluate(model, val_loader, f2i_val, device)
    print(f"\n== before pruning ==")
    print(f"size {base_size:.2f} MB | params {base_params/1e6:.2f}M | MACs {base_macs/1e6:.0f}M | top1 {base_acc*100:.2f}%")

    # --- structured pruning ---
    # never prune the classifier head: its output dim must stay = num classes
    ignored = [m for m in model.modules() if isinstance(m, nn.Linear)]
    example = torch.randn(1, 3, 224, 224).to(device)
    pruner = tp.pruner.MagnitudePruner(   # prune lowest-L1-norm channels
        model, example,
        importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=args.ratio,
        ignored_layers=ignored,
        global_pruning=False,  # per-layer ratio is safer for depthwise MobileNetV2
    )
    pruner.step()  # actually removes the channels and reshapes coupled layers

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example)
    pruned_acc_before_recover = evaluate(model, val_loader, f2i_val, device)
    print(f"\n== after pruning (before recovery) ==")
    print(f"params {pruned_params/1e6:.2f}M | MACs {pruned_macs/1e6:.0f}M | top1 {pruned_acc_before_recover*100:.2f}%")

    # --- recovery fine-tune ---
    print(f"\n== recovery fine-tune ({args.recover_epochs} epochs) ==")
    best_acc = 0.0
    best_state = None
    for ep, loss in train_epochs(model, train_loader, f2i_train, device, args.recover_epochs, args.lr):
        acc = evaluate(model, val_loader, f2i_val, device)
        print(f"epoch {ep}/{args.recover_epochs} | loss {loss:.3f} | val top1 {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    pruned_size = measure_size(model)
    model.cpu()
    lat = measure_latency_cpu(model)
    final_acc = evaluate(model, val_loader, f2i_val, torch.device("cpu"))

    print(f"\n== pruned + recovered (final) ==")
    print(f"size {pruned_size:.2f} MB | params {pruned_params/1e6:.2f}M | MACs {pruned_macs/1e6:.0f}M "
          f"| latency CPU {lat:.2f} ms | top1 {final_acc*100:.2f}%")
    print(f"\nsize {base_size/pruned_size:.2f}x smaller | MACs {base_macs/pruned_macs:.2f}x fewer "
          f"| top1 {(final_acc-base_acc)*100:+.2f} pts vs dense fine-tuned ({base_acc*100:.2f}%)")

    torch.save(model, "mobilenetv2_pruned.pt")  # save whole model: structure changed
    print("saved mobilenetv2_pruned.pt")


if __name__ == "__main__":
    main()