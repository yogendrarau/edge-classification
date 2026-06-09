# edge-classification

This project explores what happens when a MobileNetV2 image classifier is optimized for edge deployment. I applied quantization and pruning to the model, measured how each change affected accuracy and performance, and then deployed it in the browser to see how it performed in a real environment.

The model runs entirely client-side using ONNX Runtime Web, so images never leave the device and no server is involved during inference.

**Live Demo:** https://yogendrarau.github.io/edge-classification/

## Overview

Modern image classification models can achieve impressive accuracy, but they are often too large or computationally expensive for edge devices. Running models directly on a user's device has several advantages, including lower latency, improved privacy, offline functionality, and no server-side inference costs. The challenge is that reducing model size and computation often comes at the expense of accuracy.

The goal of this project was to better understand that tradeoff. Starting with MobileNetV2, I applied common edge optimization techniques one at a time and measured their impact on accuracy, model size, compute requirements, and latency. Rather than proposing a new optimization method, the focus was on understanding the deployment process and evaluating the practical effects of each technique.

## Results

All accuracy measurements were collected on Imagenette, a 10-class subset of ImageNet. Top-5 accuracy is included alongside Top-1 because some classes are visually similar and can be difficult to distinguish even when the prediction is reasonable.

| Model                 | Top-1 | Top-5 | Size    | MACs |
| --------------------- | ----- | ----- | ------- | ---- |
| FP32, zero-shot       | 78.8% | 95.5% | 13.6 MB | 320M |
| FP32, fine-tuned      | 97.7% | 99.8% | 13.6 MB | 320M |
| INT8 PTQ (zero-shot)  | 72.8% | 91.8% | 3.74 MB | —    |
| INT8 QAT (fine-tuned) | 97.4% | 99.9% | 3.74 MB | —    |
| Pruned 20% (FP32)     | 95.0% | —     | 9.58 MB | 210M |
| Pruned 30% (FP32)     | 92.2% | —     | 7.83 MB | 165M |

For browser deployment, the FP32 ONNX model achieved roughly 17 ms inference time using WebGPU and around 50 ms using WASM. Measurements were taken as the median of 20 runs after warm-up on an NVIDIA RTX 4060 laptop running Chrome.

## What I Learned

### Quantization provided the biggest benefit

Quantization was easily the most effective optimization. Converting the model from FP32 to INT8 reduced its size from 13.6 MB to 3.74 MB, about a 3.6× reduction. After quantization-aware training, the model maintained nearly the same accuracy as the full-precision version, dropping only from 97.7% to 97.4%.

### PTQ vs. QAT needs a fair comparison

One thing I learned quickly was that comparing post-training quantization (PTQ) directly to quantization-aware training (QAT) can be misleading. The QAT model was also fine-tuned on Imagenette, while the PTQ model was not.

To separate the effects of fine-tuning from quantization, I trained a fine-tuned FP32 model as a control. That model reached 97.7% accuracy, showing that the actual cost of quantization was only the small difference between FP32 and QAT, not the much larger gap between PTQ and QAT.

### Pruning helped less than expected

Since MobileNetV2 is already designed to be lightweight, there wasn't much unnecessary structure left to remove. After retraining, 20% channel pruning recovered to 95.0% accuracy, but increasing pruning to 30% caused accuracy to fall to 92.2%.

The results suggest that moderate pruning is possible, but the accuracy tradeoff becomes much steeper as more channels are removed.

### Lower compute didn't necessarily mean faster inference

Pruning reduced MACs by roughly 1.5–1.9×, but CPU inference time did not improve much. Although the model performed fewer operations on paper, the resulting architecture was less efficient for the hardware to execute.

This was a good reminder that reducing FLOPs or MACs does not automatically translate into lower latency in practice.

## A Bug That Took a While to Track Down

During post-training quantization, accuracy suddenly dropped to around 12%, which was obviously wrong.

The issue turned out to be related to PyTorch's oneDNN INT8 backend. On CPUs without AVX-512 VNNI support, the default quantization configuration can suffer from activation saturation. Switching to the x86 qconfig, which enables `reduce_range=True`, fixed the problem by using 7-bit activations instead of 8-bit activations.

I also switched from eager mode quantization to FX graph mode because eager mode was silently skipping observers on some layers.

## Methodology

* Latency measurements use batch size 1.
* Warm-up runs are discarded before timing begins.
* Results are reported using the median of multiple runs rather than a single measurement.
* GPU timing uses `torch.cuda.synchronize()` before stopping the timer.
* Quantized models are compared against FP32 CPU results since PyTorch INT8 inference is CPU-only.
* A per-class accuracy diagnostic was used to verify class-index mappings before reporting final accuracy values.
* ONNX exports were validated against PyTorch outputs before deployment, with a maximum observed difference of approximately 5e-6.

## Project Structure

| File               | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| `phase0.py`        | Basic model loading and image classification test   |
| `phase1.py`        | Baseline model evaluation                           |
| `diagnose.py`      | Per-class accuracy diagnostics                      |
| `phase2ptqfx.py`   | Post-training INT8 quantization using FX graph mode |
| `phase2qat.py`     | Quantization-aware training                         |
| `phase2fp32.py`    | Fine-tuned FP32 control model                       |
| `phase2pruning.py` | Structured channel pruning                          |
| `phase3onnx.py`    | ONNX export and numerical verification              |
| `docs/`            | Browser deployment files                            |

## Running the Project

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python phase1.py --data path/to/imagenette2-320/val
python phase2ptqfx.py --data path/to/imagenette2-320/val
python phase2fp32.py --data path/to/imagenette2-320 --epochs 5
python phase2qat.py --data path/to/imagenette2-320 --epochs 5
python phase2pruning.py --data path/to/imagenette2-320 --ratio 0.2 --recover-epochs 10
python phase3onnx.py
```

Dataset: Imagenette (imagenette2-320)

