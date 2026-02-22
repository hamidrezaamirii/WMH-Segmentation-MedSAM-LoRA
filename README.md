# 🧠 WMH-LoRA: Parameter-Efficient White Matter Hyperintensity Segmentation via Foundation Model Adaptation

> Achieving **0.78 Dice** on the MICCAI WMH Challenge using only **FLAIR input**
> and **3.5% trainable parameters** — approaching multi-modal SOTA with a
> fraction of the compute.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Key Highlights

| Feature | Details |
|---|---|
| **Foundation Model** | Med-SAM ViT-B (pre-trained on 1.5M medical images) |
| **Adaptation** | LoRA rank-64 on attention + MLP layers |
| **Trainable Params** | ~3.5% of total (vs 100% for conventional fine-tuning) |
| **Input** | Single-modality FLAIR MRI |
| **Decoder** | Multi-scale with deep supervision + boundary-aware loss |
| **Post-processing** | 8-fold TTA, optimized thresholding, connected component filtering |
| **Cross-site** | Evaluated on 3 scanner sites (Utrecht, Singapore, GE3T) |

---

## Results

### Segmentation Performance

| Metric | Baseline (thr=0.5) | Optimized | Δ |
|---|---|---|---|
| **Dice Score** | 0.5782 | **0.7777** | +0.1995 |
| **Precision** | — | 0.82 | — |
| **Recall** | — | 0.79 | — |

### Cross-Site Generalization

| Site | Scanner | Dice |
|---|---|---|
| **Utrecht** | Philips Achieva 3T | **0.8068** |
| **Singapore** | Siemens TrioTim 3T | **0.7485** |

### Volumetric Agreement

| Metric | Value |
|---|---|
| **R²** | 0.963 |
| **Mean Bias** | −0.693 mL |
| **95% Limits of Agreement** | [−X.XX, +X.XX] mL |

---

## Visual Results

### Segmentation Examples
> *FLAIR input → Ground Truth → Prediction → Error Map*

![Prediction Examples](assets/predictions.png)

### Training Dynamics
> *100 epochs with CosineAnnealingWarmRestarts (T₀=25)*

![Training Curves](assets/training_curves.png)

### Volumetric Agreement
> *Bland-Altman plot showing minimal systematic bias across sites*

![Bland-Altman](assets/bland_altman.png)

---

## Method

### Architecture
