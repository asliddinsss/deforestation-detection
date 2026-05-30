"""
evaluate.py
-----------
Evaluation utilities:
  - compute_metrics: accuracy, precision, recall, F1
  - plot_confusion_matrix
  - plot_training_curves (from CSV log)
  - evaluate_model: full test-set evaluation with saved plots
"""

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

LABELS = ["Forest", "Deforested"]


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="binary", zero_division=0),
    }


# ─────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: str = "results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix — Deforestation Detection", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Counts", "Normalized"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=LABELS, yticklabels=LABELS, ax=ax,
            linewidths=0.5, linecolor="gray",
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────

def plot_training_curves(log_csv: str = "results/training_log.csv",
                         save_path: str = "results/training_curves.png"):
    df = pd.read_csv(log_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("loss",  "Loss",     "tab:blue",   "tab:orange"),
        ("acc",   "Accuracy", "tab:green",  "tab:red"),
        ("f1",    "F1 Score", "tab:purple", "tab:brown"),
    ]

    for ax, (metric, ylabel, c_train, c_val) in zip(axes, metrics):
        ax.plot(df["epoch"], df[f"train_{metric}"], color=c_train, label="Train", linewidth=2)
        ax.plot(df["epoch"], df[f"val_{metric}"],   color=c_val,   label="Val",   linewidth=2, linestyle="--")

        # Mark phase transition
        if "phase" in df.columns:
            phase_change = df[df["phase"] == "full"]["epoch"].min()
            if not np.isnan(phase_change):
                ax.axvline(x=phase_change, color="gray", linestyle=":", linewidth=1.5, label="Unfreeze")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# Full test-set evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, test_loader, device, results_dir: str = "results"):
    import torch
    model.eval()
    all_preds, all_labels, all_paths = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, paths = batch
                all_paths.extend(paths)
            else:
                images, labels = batch

            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)

    print("\n" + "="*50)
    print("  TEST SET RESULTS")
    print("="*50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print("="*50)
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, target_names=LABELS))

    plot_confusion_matrix(all_labels, all_preds, save_path=os.path.join(results_dir, "confusion_matrix.png"))

    return metrics, all_preds, all_labels
