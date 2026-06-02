"""Phase 0: confirm MobileNetV2 loads with pretrained weights and predicts sanely."""

import argparse
import sys

import torch
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def report_env():
    print(f"torch          {torch.__version__}")
    print(f"cuda available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu            {torch.cuda.get_device_name(0)}")


def load_model():
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights).eval()
    # transforms() ships the exact resize/crop/normalize these weights were trained with
    return model, weights.transforms(), weights.meta["categories"]


def classify(image_path, model, preprocess, categories, topk=5):
    img = Image.open(image_path).convert("RGB")
    batch = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        probs = model(batch).softmax(dim=1)

    top = torch.topk(probs, topk)
    print(f"\n{image_path}")
    for score, idx in zip(top.values[0], top.indices[0]):
        print(f"  {categories[idx]:28s} {score.item():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--check-env", action="store_true")
    args = ap.parse_args()

    report_env()
    if args.check_env:
        return
    if not args.image:
        sys.exit("pass an image path, e.g. python phase0_sanity.py test.jpg")

    model, preprocess, categories = load_model()
    classify(args.image, model, preprocess, categories, args.topk)


if __name__ == "__main__":
    main()