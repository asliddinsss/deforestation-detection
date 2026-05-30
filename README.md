#  Deforestation Detection from Satellite Imagery
### Using Deep Learning (CNN + Transfer Learning) to Monitor Forest Loss

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> **Real-world impact:** Every year, the world loses millions of hectares of forest — often in remote areas that go unmonitored for months. This project uses freely available satellite imagery and deep learning to automatically classify forest patches as *intact*, *degraded*, or *deforested*, enabling faster environmental response.

---

##  Problem Statement

Manual monitoring of deforestation is slow, expensive, and geographically limited. Remote sensing satellites (like Landsat and Sentinel-2) capture the entire Earth every few days — but analyzing this imagery at scale requires automation.

This project trains a CNN-based binary classifier on labeled satellite image patches to distinguish **forested** vs **deforested** land, achieving strong generalization across different geographic regions.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline CNN (scratch) | 83.2% | 0.81 | 0.84 | 0.82 |
| ResNet-18 (transfer learning) | **91.7%** | **0.90** | **0.93** | **0.91** |
| ResNet-18 + Augmentation | **93.4%** | **0.92** | **0.94** | **0.93** |

**Key finding:** The model struggled most with *edge cases* — areas undergoing selective logging, where partial canopy cover confused the classifier. Adding targeted augmentation (random erasing, brightness jitter) improved recall on degraded patches by ~8%.

---

##  Repository Structure

```
deforestation-detection/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA on satellite image dataset
│   ├── 02_baseline_cnn.ipynb           # Baseline CNN from scratch
│   └── 03_transfer_learning.ipynb      # ResNet-18 fine-tuning + results
│
├── src/
│   ├── dataset.py                      # Custom PyTorch Dataset class
│   ├── model.py                        # Model definitions (BaseCNN, ResNet)
│   ├── train.py                        # Training loop with logging
│   ├── evaluate.py                     # Evaluation metrics + confusion matrix
│   └── predict.py                      # Run inference on a new image
│
├── data/
│   ├── raw/                            # Original downloaded images
│   ├── processed/                      # Resized, normalized patches
│   └── samples/                        # Example images for quick testing
│
├── models/
│   └── best_model.pt                   # Saved best checkpoint (not tracked by git)
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── sample_predictions.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

This project uses the **[Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)** dataset (Kaggle), which contains 40,000+ multi-label satellite image chips from the Amazon rainforest.

For a lightweight demo, we also include 200 sample patches in `data/samples/` so you can run inference immediately without downloading the full dataset.

**Labels used (binary classification):**
- `0` → Forest (primary, agriculture with canopy)  
- `1` → Deforested (bare ground, slash/burn, logging)

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/asliddinsss/deforestation-detection.git
cd deforestation-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run inference on a sample image
```bash
python src/predict.py --image data/samples/test_patch.jpg
```

### 4. Train from scratch
```bash
python src/train.py --model resnet18 --epochs 20 --batch_size 32
```

### 5. Explore the notebooks (recommended)
```bash
jupyter notebook notebooks/
```
Open `01_data_exploration.ipynb` first, then follow the sequence.

---

## Model Architecture

We use **ResNet-18** pre-trained on ImageNet, with the final fully-connected layer replaced for binary classification:

```
ResNet-18 Backbone (frozen for first 5 epochs)
    ↓
Global Average Pooling
    ↓
FC(512 → 256) + ReLU + Dropout(0.4)
    ↓
FC(256 → 2) + Softmax
```

**Training strategy:**
- Phase 1 (epochs 1–5): Freeze backbone, train classifier head only
- Phase 2 (epochs 6–20): Unfreeze all layers, train end-to-end with lower LR

---

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-3 (head), 1e-5 (backbone) |
| Batch Size | 32 |
| Epochs | 20 |
| Image Size | 224×224 |
| Augmentation | HorizontalFlip, RandomRotation, ColorJitter, RandomErasing |
| Loss | CrossEntropyLoss |
| Hardware | Google Colab (T4 GPU) |

---

## Real-World Impact

This kind of system, deployed at scale with real-time satellite feeds, could:
- Alert environmental agencies within days of deforestation events (vs. months of manual review)
- Track illegal logging in protected zones across Central Asia and South America
- Provide objective, timestamped evidence for climate reporting

This project is part of my broader research into **AI for environmental monitoring**, building toward a system that integrates air quality, soil degradation, and vegetation loss into a unified early-warning dashboard for policymakers.

---

## Future Work

- [ ] Multi-class segmentation (not just classify patches, but pixel-level maps)
- [ ] Deploy as a web API so NGOs can submit their own satellite crops for analysis
- [ ] Extend to Sentinel-2 multispectral bands (beyond RGB) for better accuracy
- [ ] Test on Central Asian forests (Tian Shan region)

---

## Author

**Asliddin** — Grade 9, Presidential School, Namangan, Uzbekistan  
AI/ML Researcher | Climate Advocate | APIO Finalist 2025  
[LinkedIn](#) · [GitHub](#) · [YouTube](#)

---

## 📄 License

MIT License — feel free to use, modify, and build on this work.
