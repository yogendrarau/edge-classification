"""Phase 3a: export the fine-tuned FP32 MobileNetV2 to ONNX and verify correctness.

Verification is the whole point of this script: we confirm the ONNX model produces
numerically identical outputs to PyTorch BEFORE building any browser code. If the
browser later gives wrong predictions, we'll know it's the web layer, not the export.

Steps: load weights -> export to ONNX -> check with onnx.checker -> run both PyTorch
and onnxruntime on the same input and compare outputs.
"""

import argparse

import numpy as np
import torch
from torchvision.models import mobilenet_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="mobilenetv2_fp32_finetuned.pt")
    ap.add_argument("--out", default="mobilenetv2_fp32.onnx")
    args = ap.parse_args()

    model = mobilenet_v2()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    # torch 2.12's dynamo exporter is natively opset 18; request 18 directly to avoid
    # the (failed-but-recovered) downgrade-to-17 conversion warnings.
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"exported {args.out}")

    # Re-save as a SINGLE self-contained file. The dynamo exporter splits weights into
    # an external .onnx.data sidecar, which onnxruntime-web can't load in the browser.
    # Loading and re-saving without external data packs everything into one .onnx.
    import onnx
    m = onnx.load(args.out)  # load_external_data=True by default, pulls in the sidecar
    onnx.save_model(m, args.out, save_as_external_data=False)
    print(f"re-saved {args.out} as single self-contained file")

    # structural validity
    import onnx
    onnx.checker.check_model(onnx.load(args.out))
    print("onnx.checker passed")

    # numerical parity: PyTorch vs onnxruntime on identical input
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_diff = np.abs(torch_out - ort_out).max()
    print(f"max abs diff PyTorch vs ONNX: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("PASS: outputs match (export is faithful)")
    else:
        print("WARN: outputs diverge more than expected — investigate before deploying")

    # confirm both pick the same class
    print(f"argmax  PyTorch={torch_out.argmax()}  ONNX={ort_out.argmax()}")


if __name__ == "__main__":
    main()