# **Image Classification Benchmark – Skin Lesions & Facial Emotions**
[![Python](https://img.shields.io/badge/Language-Python-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Library-PyTorch-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Albumentations](https://img.shields.io/badge/Library-Albumentations-3F8FD2.svg?style=flat)](https://albumentations.ai/)
[![Torchvision](https://img.shields.io/badge/Library-Torchvision-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/vision/)
 
**Copyright © Springer Robert Stefan, 2026**
 
> A modular deep learning experimentation framework for multi-class image classification on two datasets: **skin lesion diagnosis** and **facial emotion recognition**.
>
> The project systematically compares architectural choices (MLP vs. CNN), regularization techniques (Dropout, BatchNorm), data augmentation strategies, class-imbalance handling, transfer learning with ResNet-18 and a self-paced curriculum learning approach.
 
## **Table of Contents**
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Experiments](#experiments)
- [Evaluation & Outputs](#evaluation--outputs)
- [Setup & Usage](#setup--usage)
## **Overview**
 
The framework trains and evaluates several models on two 7-class image datasets:
 
| Dataset | Task | Classes | Input Size |
|:---:|:---|:---|:---:|
| **Skin** | Skin lesion diagnosis | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` | 128×128 |
| **Emotion** | Facial expression recognition | `Surprise`, `Fear`, `Disgust`, `Happiness`, `Sadness`, `Anger`, `Neutral` | 100×100 |
 
Every experiment is reproducible through a single entry point and automatically produces metrics, confusion matrices and training curves.
 
## **System Architecture**
 
| Module | Purpose |
|:---:|:---|
| **`train.py`** | Central orchestrator: builds datasets, loaders and models, runs experiments and triggers the final evaluation. |
| **`dataset_loader.py`** | Custom `SkinDataset` / `EmotionDataset` classes and the dynamic Albumentations transform factory. |
| **`models.py`** | From-scratch architectures: `SimpleMLP` and `SimpleCNN`, with switchable Dropout / BatchNorm. |
| **`finetune2.py`** | ResNet-18 transfer learning with optional backbone freezing and LR warmup + cosine scheduling. |
| **`curriculum_learning.py`** | Self-paced (loss-based) curriculum learning compared against a standard baseline. |
| **`common.py`** | Shared training loop, evaluation, manual metrics (Accuracy, F1 Macro/Micro) and plotting utilities. |
| **`vizualize_augumentations.py`** | Generates original vs. augmented montages for each class. |
 
## **Data Pipeline**
 
1. **Custom Datasets**: Images are read from CSV manifests. Skin labels are mapped from diagnosis codes to indices; emotion labels are shifted from `1-7` to `0-6`.
2. **Dataset-specific Normalization**: Per-channel mean/std computed for each dataset (ImageNet statistics are used for pre-trained ResNet-18).
3. **Configurable Augmentations** (`aug_type` = `none` / `geometric` / `color` / `all`):
   
| Augmentation | Skin | Emotion |
|:---|:---:|:---:|
| Horizontal Flip | p=0.5 | p=0.5 |
| Vertical Flip | p=0.5 | – |
| Rotation | ±180° | ±15° |
| Color Jitter | brightness/contrast 0.1 | brightness/contrast 0.2 |
| Gaussian Noise | – | p=0.2 |
 
4. **Class Imbalance Fix**: Optional `WeightedRandomSampler` with inverse-frequency class weights, so minority classes are sampled more often.
## **Models**
 
### **SimpleMLP**
Fully-connected baseline: `Flatten → Linear(512) → BN → ReLU → Dropout(0.3) → Linear(256) → BN → ReLU → Dropout(0.3) → Linear(classes)`. Dropout can be disabled for ablation.
 
### **SimpleCNN**
Four convolutional blocks (`32 → 64 → 128 → 128` filters, 3×3 kernels) with optional BatchNorm, ReLU and Max Pooling, followed by Global Average Pooling and a compact classifier head (`128 → 64 → classes`, Dropout 0.3). Bias terms are dropped automatically when BatchNorm is enabled.
 
### **ResNet-18 (Transfer Learning)**
Pre-trained on ImageNet-1K, with the final layer replaced by a new `fc` head.
- **Frozen backbone**: only the head is trained (`lr=1e-3`).
- **Unfrozen backbone**: differential learning rates – `1e-4` for the backbone, `1e-3` for the head.
- **Optimizer**: Adam with `weight_decay=1e-4`.
- **Scheduler**: optional 3-epoch linear warmup followed by Cosine Annealing.
- **Input**: resize to 256×256 → random crop 224×224 → flip / rotation (±15°) → ImageNet normalization.
## **Experiments**
 
| Experiment | Variants Compared |
|:---|:---|
| **Dropout (MLP)** | With vs. without Dropout |
| **Batch Normalization (CNN)** | With vs. without BatchNorm |
| **Data Augmentation (CNN)** | `none` vs. `geometric` vs. `color` vs. `all` |
| **Class Imbalance (CNN)** | Standard shuffling vs. `WeightedRandomSampler` |
| **Fine-tuning (ResNet-18)** | Warmup on/off × Frozen/Unfrozen backbone (4 combinations) |
| **Curriculum Learning** | Standard training vs. self-paced learning (emotion dataset) |
 
### **Self-Paced Curriculum Learning**
At the start of every epoch, per-sample losses are computed with the current model. Only the "easiest" samples (lowest loss) are used for training, using a quantile threshold that starts at **30%** of the data and grows by **5%** each epoch until the full dataset is used. The per-epoch validation accuracy is then compared against an identical baseline trained on all the data.
 
## **Evaluation & Outputs**
 
Metrics are implemented from scratch in `common.py`:
- **Accuracy**
- **F1 Macro** (average of per-class F1 scores)
- **F1 Micro** (computed from global TP / FP / FN)
- **Confusion Matrix**
Each run saves:
 
| File | Content |
|:---|:---|
| `{dataset}_{experiment}_confusion_matrix.png` | Annotated confusion matrix of the validation set. |
| `{dataset}_{experiment}_curves.png` | Train / validation loss and accuracy curves. |
| `skin_augmentations.png`, `emotion_augmentations.png` | Original vs. augmented sample per class. |
 
## **Performance Results**
 
> Fill in the table below with the values printed at the end of each run.
 
| Model | Dataset | Accuracy | F1 Macro | F1 Micro |
|:---|:---:|:---:|:---:|:---:|
| **SimpleMLP** | Skin / Emotion | – | – | – |
| **SimpleCNN** | Skin / Emotion | – | – | – |
| **ResNet-18 (Frozen + Warmup)** | Skin / Emotion | – | – | – |
| **ResNet-18 (Unfrozen + Warmup)** | Skin / Emotion | – | – | – |
 
## **Setup & Usage**
 
### **Installation**
```bash
pip install torch torchvision numpy pandas matplotlib pillow albumentations
```
 
### **Expected Data Layout**
```
vai-de-pielea-mea/
├── train.csv
├── test.csv
├── train/
└── test/
you-re-on-candid-camera/
└── splits/
    ├── train.csv
    └── local_test.csv
```
 
### **How to Run**
1. **Full pipeline** (ResNet-18 fine-tuning grid on both datasets, 30 epochs):
```bash
   python train.py
```
2. **Single custom experiment**:
```python
   from train import run_experiment
   run_experiment(model_type='cnn', dataset_type='skin', epochs=30,
                  aug_type='all', use_imbalance_fix=True)
```
3. **Curriculum learning comparison**:
```bash
   python curriculum_learning.py
```
4. **Augmentation montages**: call `generate_augmentations()` from `vizualize_augumentations.py`.
The MLP / CNN ablation runs are available inside `run_all_combinations()` in `train.py` (currently commented out) and can be re-enabled as needed.
