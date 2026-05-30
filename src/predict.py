"""
predict.py
----------
Run inference on a single image or a folder of images.

Usage:
    python src/predict.py --image data/samples/test_patch.jpg
    python src/predict.py --folder data/samples/ --output results/predictions.csv
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import build_model

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

IDX_TO_CLASS = {0: "Forest", 1: "Deforested"}
CLASS_COLORS = {0: "#2d6a4f", 1: "#d62828"}   # green / red

TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str, model_type: str = "resnet18", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_type).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded from {checkpoint_path} (Val F1: {checkpoint.get('val_f1', 'N/A'):.4f})")
    return model, device


def predict_image(image_path: str, model, device) -> dict:
    """Predict class and confidence for a single image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = TRANSFORM(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "path":       str(image_path),
        "prediction": pred_class,
        "confidence": confidence,
        "prob_forest":      float(probs[0]),
        "prob_deforested":  float(probs[1]),
    }


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def visualize_predictions(results: list, save_path: str = "results/sample_predictions.png", max_show: int = 12):
    """Grid visualization of predictions with confidence bars."""
    n = min(len(results), max_show)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 4))
    fig.suptitle("Sample Predictions — Deforestation Detection", fontsize=14, fontweight="bold")
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for i, result in enumerate(results[:n]):
        ax = axes[i]
        img = Image.open(result["path"]).convert("RGB")
        ax.imshow(img)

        color = CLASS_COLORS[0 if result["prediction"] == "Forest" else 1]
        label = f"{result['prediction']}\n{result['confidence']:.1%}"
        ax.set_title(label, fontsize=10, color=color, fontweight="bold")
        ax.axis("off")

        # Thin colored border indicating prediction
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {save_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run deforestation detection inference")
    parser.add_argument("--image",      type=str, default=None, help="Path to a single image")
    parser.add_argument("--folder",     type=str, default=None, help="Path to a folder of images")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--model",      type=str, default="resnet18")
    parser.add_argument("--output",     type=str, default="results/predictions.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    model, device = load_model(args.checkpoint, model_type=args.model)

    results = []

    if args.image:
        result = predict_image(args.image, model, device)
        results.append(result)
        print(f"\n{'='*40}")
        print(f"  Image:      {result['path']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  P(Forest):      {result['prob_forest']:.4f}")
        print(f"  P(Deforested):  {result['prob_deforested']:.4f}")
        print(f"{'='*40}\n")

    elif args.folder:
        folder = Path(args.folder)
        image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        print(f"[Predict] Running on {len(image_paths)} images in {folder}")

        for img_path in image_paths:
            result = predict_image(img_path, model, device)
            results.append(result)
            label_icon = "🌿" if result["prediction"] == "Forest" else "🔴"
            print(f"  {label_icon} {img_path.name:<30} → {result['prediction']:<12} ({result['confidence']:.1%})")

        # Save CSV
        import pandas as pd
        df = pd.DataFrame(results)
        os.makedirs(Path(args.output).parent, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n[Results] Saved to {args.output}")

        # Visualize
        visualize_predictions(results)

        # Summary
        deforested_count = sum(1 for r in results if r["prediction"] == "Deforested")
        print(f"\n[Summary] {deforested_count}/{len(results)} patches classified as Deforested ({deforested_count/len(results):.1%})")

    else:
        print("[Error] Please provide --image or --folder")


if __name__ == "__main__":
    main()
