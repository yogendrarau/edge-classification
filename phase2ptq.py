"""Phase 2a: post-training INT8 quantization of MobileNetV2.

PyTorch INT8 inference is CPU-only (fbgemm/qnnpack backends), so EVERYTHING here
is measured on CPU. The honest comparison is FP32-CPU vs INT8-CPU — never against
the GPU baseline from Phase 1.

Pipeline: load FP32 -> fuse conv+bn+relu -> insert observers -> calibrate on a few
hundred images -> convert to INT8 -> measure accuracy / size / latency.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import MobileNet_V2_Weights
# purpose-built quantizable variant: exposes fuse_model() for clean conv+bn+relu folding
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2

WNID_TO_IMAGENET_IDX = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


def get_loader(data_dir, preprocess, batch_size, subset=None, shuffle=False):
    ds = datasets.ImageFolder(data_dir, transform=preprocess)
    folder_to_imagenet = {
        ds.class_to_idx[w]: i for w, i in WNID_TO_IMAGENET_IDX.items() if w in ds.class_to_idx
    }
    if subset is not None:
        # fixed stride sample for a representative calibration set
        idxs = list(range(0, len(ds), max(1, len(ds) // subset)))[:subset]
        ds = Subset(ds, idxs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0), folder_to_imagenet


def measure_accuracy(model, loader, folder_to_imagenet):
    model.eval()
    top1 = top5 = total = 0
    with torch.no_grad():
        for images, folder_labels in loader:
            logits = model(images)
            # Subset wraps labels the same way ImageFolder does
            true_idx = torch.tensor([folder_to_imagenet[int(l)] for l in folder_labels])
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
    return {"median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "fps": 1000.0 / float(np.median(times))}


def measure_size(model, path="_tmp.pt"):
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="imagenette val dir")
    ap.add_argument("--calib-size", type=int, default=300, help="images for calibration")
    args = ap.parse_args()

    # Pick whatever quantized backend this build supports. fbgemm=x86 (legacy),
    # onednn=x86 (Intel oneDNN kernels), qnnpack=ARM (also runs on x86).
    supported = torch.backends.quantized.supported_engines
    engine = next((e for e in ("fbgemm", "onednn", "qnnpack") if e in supported), None)
    if engine is None:
        raise SystemExit(
            f"No quantized backend available (supported: {supported}). "
            "This PyTorch build lacks quantization support — use a CPU-only build for Phase 2.")
    torch.backends.quantized.engine = engine
    print(f"quantized engine: {engine}")
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # --- FP32 baseline on CPU (the fair comparison point) ---
    # quantize=False gives the float model with the SAME architecture we'll quantize,
    # so the FP32 vs INT8 comparison is apples-to-apples.
    fp32 = q_mobilenet_v2(weights=weights, quantize=False).eval()
    val_loader, f2i = get_loader(args.data, preprocess, batch_size=32)

    print("== FP32 (CPU) ==")
    fp32_size = measure_size(fp32)
    fp32_lat = measure_latency_cpu(fp32)
    fp32_acc = measure_accuracy(fp32, val_loader, f2i)
    print(f"size {fp32_size:.2f} MB | latency median {fp32_lat['median_ms']:.2f} ms "
          f"| {fp32_lat['fps']:.1f} FPS | top1 {fp32_acc[0]*100:.2f}% top5 {fp32_acc[1]*100:.2f}%")

    # --- INT8 via PTQ ---
    qmodel = copy.deepcopy(fp32)
    qmodel.fuse_model()  # fold conv+bn+relu — required for good INT8 accuracy
    qmodel.qconfig = get_default_qconfig(engine)
    prepare(qmodel, inplace=True)  # insert observers

    calib_loader, _ = get_loader(args.data, preprocess, batch_size=32, subset=args.calib_size)
    with torch.no_grad():
        for images, _ in calib_loader:  # calibration: observe activation ranges
            qmodel(images)

    convert(qmodel, inplace=True)  # fold observers, swap to INT8 kernels

    print("\n== INT8 PTQ (CPU) ==")
    int8_size = measure_size(qmodel)
    int8_lat = measure_latency_cpu(qmodel)
    int8_acc = measure_accuracy(qmodel, val_loader, f2i)
    print(f"size {int8_size:.2f} MB | latency median {int8_lat['median_ms']:.2f} ms "
          f"| {int8_lat['fps']:.1f} FPS | top1 {int8_acc[0]*100:.2f}% top5 {int8_acc[1]*100:.2f}%")

    print(f"\nsize {fp32_size/int8_size:.2f}x smaller | "
          f"top1 drop {(fp32_acc[0]-int8_acc[0])*100:.2f} pts")


if __name__ == "__main__":
    main()