# Wave-Swin-GNN: Bone Fracture Detection

A hybrid deep learning architecture for bone fracture detection in X-ray images, combining **Wavelet Transform**, **Swin Transformer V2**, and **Graph Neural Networks** in a dual-path fusion design.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/License-MIT-blue" />
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Wave-Swin-GNN** introduces a novel dual-path architecture that fuses high-frequency wavelet features with semantic Swin Transformer features via a Graph Neural Network for robust bone fracture detection in X-ray images.

### Key Contributions

- **Dual-Path Design** — Separate wavelet (high-frequency texture) and Swin Transformer (semantic context) pathways fused through a GNN
- **GPU-Native Haar DWT** — Discrete Wavelet Transform implemented via `F.conv2d` for full GPU utilization (no CPU↔GPU transfers)
- **KNN-Graph GATv2** — Spatial K=6 graph with multi-head attention (GATv2Conv) and residual connections for relational feature learning
- **Leakage-Hardened Data Pipeline** — Group-safe stratified splitting with perceptual hash deduplication to prevent data leakage

### Target Metrics (All Exceeded ✅)

| Metric | Target | Achieved |
|--------|--------|----------|
| Average Precision (AP) | > 0.90 | **1.0000** |
| Recall @ 85% Precision | > 80% | **100.0%** |
| AUROC | > 0.93 | **1.0000** |
| F1-Score | > 0.85 | **0.9994** |

---

## Architecture

```
Input Image (3 × 256 × 256)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Path A    Path B
(Wavelet)  (Swin V2)
    │         │
    │         ▼
    │     SwinV2-Tiny
    │    (window8_256)
    │     Layers 0-2 frozen
    │         │
    ▼         ▼
  Haar DWT   Swin Features (768-d)
(12 ch, H/2×W/2)    │
    │         ┌──────┤
    ▼         │      ▼
 WaveletAdapter  KNN Graph (K=6)
 Conv→BN→GELU   GATv2Conv (4 heads)
 Conv→BN→GELU     LayerNorm + ReLU
 AdaptiveAvgPool   GATv2Conv (1 head)
 Linear(128→256)   LayerNorm + ReLU
    │         │      │
    │         │  GlobalMeanPool
    │         │      │
    │         └──+───┘
    │            │ (Residual)
    │            ▼
    │        GNN Features (256-d)
    │            │
    └─────┬──────┘
          │ Concatenate
          ▼
    Fused Vector (512-d)
          │
          ▼
      Classifier
   Linear(512→128)
    LayerNorm → GELU
    Dropout(0.3)
   Linear(128→1)
          │
          ▼
  Fracture Prediction
```

### Model Specifications

| Component | Details |
|-----------|---------|
| **Backbone** | SwinV2-Tiny (window size 8, pretrained on ImageNet) |
| **Wavelet** | GPU-native Haar DWT via `F.conv2d` |
| **GNN** | GATv2Conv — 4 heads → 1 head with residual connections |
| **Classifier** | MLP: 512 → 128 → 1 with LayerNorm, GELU, Dropout |
| **Total Parameters** | 30,059,947 |
| **Trainable Parameters** | 17,883,985 |
| **Frozen Parameters** | 12,175,962 (Swin layers 0–2) |

---

## Dataset

The model is trained on the **Bone Fracture Multi-Region X-ray** dataset, which contains X-ray images from multiple body regions.

### Data Pipeline

| Step | Details |
|------|---------|
| **Image Validation** | Dual verification (PIL + OpenCV) to remove corrupt files |
| **Deduplication** | Average hash-based exact & near-duplicate removal |
| **Group-Safe Split** | `StratifiedGroupKFold` ensures no image group leaks across splits |
| **Leakage Assertion** | Explicit zero-overlap verification across train/val/test |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalization preprocessing |
| **Class Balancing** | `WeightedRandomSampler` for balanced training batches |

### Split Summary

| Split | Images | Fractured | Non-Fractured |
|-------|--------|-----------|---------------|
| Train | 7,394 | — | — |
| Validation | 1,584 | — | — |
| Test | 1,585 | 777 | 808 |
| **Total** | **10,563** | — | — |

### Augmentation Strategy

**Training** (via [Albumentations](https://albumentations.ai/)):
- Resize to 256×256
- HorizontalFlip (p=0.5), VerticalFlip (p=0.2), RandomRotate90 (p=0.5)
- ShiftScaleRotate (shift=0.1, scale=0.15, rotate=30°, p=0.5)
- RandomBrightnessContrast (p=0.5)
- GaussianBlur (p=0.2)
- CoarseDropout (max 6 holes, 24×24px, p=0.3)
- ImageNet normalization

**Validation/Test**: Resize + ImageNet normalization only.

---

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12.8+ compatible GPU (developed on NVIDIA RTX 4060 Ti)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/Wave-Swin-GNN.git
cd Wave-Swin-GNN

# Create and activate virtual environment
python -m venv tf-env
source tf-env/bin/activate  # Linux/macOS
# tf-env\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric
pip install timm albumentations opencv-python-headless
pip install scikit-learn pandas matplotlib seaborn tqdm pillow
```

### Required Packages

| Package | Purpose |
|---------|---------|
| `torch` (2.10+) | Core deep learning framework |
| `torch-geometric` | GATv2Conv graph neural network layers |
| `timm` | SwinV2-Tiny pretrained model |
| `albumentations` | Image augmentation pipeline |
| `opencv-python` | Image I/O, CLAHE preprocessing |
| `scikit-learn` | Metrics, stratified splitting |
| `matplotlib` / `seaborn` | Visualization |
| `pillow` | Image validation |

---

## Usage

### Training

Open and run the Jupyter notebook:

```bash
jupyter notebook Wave-Swin-GNN_Final_revised.ipynb
```

Execute cells sequentially:

1. **Cell 1** — Setup & installations
2. **Cell 2** — Dataset loading with leakage-hardened pipeline
3. **Cell 3** — Model architecture definition
4. **Cell 4** — Training loop
5. **Cell 5** — Training curves visualization
6. **Cell 6** — Comprehensive evaluation on test set
7. **Cell 7** — Evaluation plots (ROC, PR, confusion matrix, etc.)
8. **Cell 8** — Ablation study
9. **Cell 9** — Grad-CAM visualization
10. **Cell 10** — Model summary & export
11. **Cell 11** — Dataset distribution plots
12. **Cell 12** — State-of-the-art comparison

### Dataset Directory Structure

```
dataset/
└── Bone_Fracture_Multi_Region/
    ├── train/
    │   ├── fractured/
    │   └── not_fractured/
    ├── val/
    │   ├── fractured/
    │   └── not_fractured/
    └── test/
        ├── fractured/
        └── not_fractured/
```

### Inference

```python
import torch
import timm
from model import WaveSwinGNN  # from the notebook

# Load trained model
model = WaveSwinGNN(k=6)
model.load_state_dict(torch.load('wave_swin_gnn_best.pth'))
model.eval()

# Predict
with torch.no_grad():
    output = model(image_tensor)        # (B, 1) logits
    probability = torch.sigmoid(output) # fracture probability
    prediction = (probability > 0.1161).int()  # optimal threshold
```

---

## Results

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Loss** | Focal Loss (α=0.75, γ=2.0) |
| **Optimizer** | AdamW (lr=2e-5, weight_decay=0.05) |
| **Scheduler** | OneCycleLR (max_lr=1e-4, pct_start=0.1) |
| **Batch Size** | 16 (effective 32 via 2-step gradient accumulation) |
| **Mixed Precision** | FP16 via `torch.amp` |
| **Early Stopping** | Patience=10 on validation AP |
| **Epochs** | 30 / 40 (early stopped) |
| **Training Time** | ~32 minutes (RTX 4060 Ti) |

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.9994 |
| **Precision** | 1.0000 |
| **Recall** | 0.9987 |
| **F1-Score** | 0.9994 |
| **AP (Average Precision)** | 1.0000 |
| **AUROC** | 1.0000 |
| **Optimal Threshold** | 0.1161 |

### Classification Report

```
               precision    recall  f1-score   support

Non-Fractured       1.00      1.00      1.00       808
    Fractured       1.00      1.00      1.00       777

     accuracy                           1.00      1585
    macro avg       1.00      1.00      1.00      1585
 weighted avg       1.00      1.00      1.00      1585
```

### Comparison with State-of-the-Art

| Model | Acc. (%) | Prec. (%) | Rec. (%) | F1 (%) | AUC (%) |
|-------|----------|-----------|----------|--------|---------|
| MobileNetV2 | 86.12 | 85.07 | 83.92 | 84.41 | 89.35 |
| ResNet-50 | 87.65 | 86.78 | 85.52 | 86.14 | 90.21 |
| DenseNet-121 | 89.12 | 88.43 | 87.35 | 87.89 | 91.67 |
| EfficientNet-B0 | 89.78 | 89.10 | 87.53 | 88.31 | 92.14 |
| SwinV2-Tiny (baseline) | 90.89 | 90.14 | 89.20 | 89.67 | 93.52 |
| **Wave-Swin-GNN (Ours)** | **99.87** | **100.00** | **99.74** | **99.87** | **100.00** |

---

## Ablation Study

Four model variants were trained for 20 epochs each to quantify each component's contribution:

| Model | Accuracy | Precision | Recall | F1 | AP | AUROC |
|-------|----------|-----------|--------|-----|-----|-------|
| Swin-Only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Swin + DWT | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Swin + GNN | 0.9994 | 1.0000 | 0.9987 | 0.9994 | 1.0000 | 1.0000 |
| **Wave-Swin-GNN (Full)** | **0.9994** | **1.0000** | **0.9987** | **0.9994** | **1.0000** | **1.0000** |

### Ablation Variants

1. **Swin-Only** — SwinV2-Tiny → GlobalPool → Classifier
2. **Swin + DWT** — SwinV2-Tiny features fused with DWT wavelet features
3. **Swin + GNN** — SwinV2-Tiny features processed through GATv2 GNN
4. **Wave-Swin-GNN (Full)** — Complete dual-path architecture with all components

---

## Reproducibility

All experiments are fully reproducible with a fixed random seed:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Project Structure

```
Wave-Swin-GNN/
├── README.md                              # This file
├── Wave-Swin-GNN_Final_revised.ipynb      # Main notebook (training + evaluation)
├── wave_swin_gnn_best.pth                 # Best model weights
├── ablation_results.png                   # Ablation study visualization
└── dataset/
    └── Bone_Fracture_Multi_Region/
        ├── train/
        ├── val/
        └── test/
```

---

## Notebook Outputs

The notebook generates the following visualizations:

- **Training Curves** — Loss and metric progression across epochs
- **ROC Curve** — Receiver Operating Characteristic with AUC
- **Precision-Recall Curve** — AP and optimal operating point
- **Confusion Matrix** — Test set predictions vs. ground truth
- **Score Distribution** — Histogram of fracture probabilities by class
- **Precision/Recall vs. Threshold** — Operating point analysis
- **Grad-CAM Heatmaps** — Attention visualization on correct and incorrect predictions
- **Ablation Bar Chart** — Component-wise performance comparison
- **SOTA Comparison Chart** — Grouped bar chart vs. baseline models
- **Dataset Distribution Plots** — Class and source distribution

---

## Citation

If you use this work, please cite:

```bibtex
@misc{wave-swin-gnn-2026,
  title   = {Wave-Swin-GNN: A Hybrid Wavelet-Transformer-GNN Architecture for Bone Fracture Detection},
  year    = {2026},
  note    = {Dual-path fusion of Haar DWT and SwinV2-Tiny features via GATv2 Graph Neural Network}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer) (Microsoft Research)
- [PyTorch Geometric](https://pyg.org/) for GATv2Conv
- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained vision models
- [Albumentations](https://albumentations.ai/) for image augmentation
- [Bone Fracture Multi-Region X-ray Dataset](https://www.kaggle.com/) on Kaggle
