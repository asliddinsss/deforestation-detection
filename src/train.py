"""
train.py
--------
Full training pipeline with:
  - Two-phase training (freeze → unfreeze)
  - Learning rate scheduling
  - Best model checkpointing
  - CSV logging of all metrics
  - Early stopping

Usage:
    python src/train.py --model resnet18 --epochs 20 --batch_size 32
"""

import os
import argparse
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model
from evaluate import compute_metrics


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train deforestation detection model")
    parser.add_argument("--model",       type=str, default="resnet18", choices=["baseline", "resnet18", "resnet34", "resnet50"])
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--save_dir",   type=str, default="models")
    parser.add_argument("--results_dir",type=str, default="results")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_head",    type=float, default=1e-3)   # Phase 1 LR
    parser.add_argument("--lr_full",    type=float, default=1e-5)   # Phase 2 LR
    parser.add_argument("--freeze_epochs", type=int, default=5)     # Phase 1 duration
    parser.add_argument("--dropout",    type=float, default=0.4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patience",   type=int, default=5)        # Early stopping
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"[Epoch {epoch}] Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch, split="Val"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc=f"[Epoch {epoch}] {split}", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Deforestation Detection — Training")
    print(f"  Model: {args.model} | Device: {device}")
    print(f"{'='*50}\n")

    # Dirs
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    # Model
    model = build_model(args.model, dropout=args.dropout).to(device)

    # Phase 1: Freeze backbone (only for ResNet models)
    if args.model != "baseline" and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1 optimizer (head only) ──────────
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)

    # CSV log
    log_path = Path(args.results_dir) / "training_log.csv"
    log_fields = ["epoch", "phase", "train_loss", "train_acc", "train_f1",
                  "val_loss", "val_acc", "val_f1", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        # ── Phase transition ───────────────────
        if epoch == args.freeze_epochs + 1 and args.model != "baseline":
            print(f"\n[Phase 2] Unfreezing backbone at epoch {epoch}")
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=args.lr_full, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs)

        phase = "frozen" if epoch <= args.freeze_epochs else "full"
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics   = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Print summary
        print(
            f"Epoch {epoch:02d}/{args.epochs} [{phase}] | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s"
        )

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch, "phase": phase,
                "train_loss": round(train_metrics["loss"], 4),
                "train_acc":  round(train_metrics["accuracy"], 4),
                "train_f1":   round(train_metrics["f1"], 4),
                "val_loss":   round(val_metrics["loss"], 4),
                "val_acc":    round(val_metrics["accuracy"], 4),
                "val_f1":     round(val_metrics["f1"], 4),
                "lr":         current_lr,
            })

        # Checkpoint best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            save_path = Path(args.save_dir) / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": val_metrics["accuracy"],
                "args": vars(args),
            }, save_path)
            print(f"  ✓ Best model saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs.")
                break

    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Model saved to: {args.save_dir}/best_model.pt")
    print(f"  Log saved to:   {log_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
