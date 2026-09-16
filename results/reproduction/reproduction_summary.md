# WMH MedSAM-LoRA Reproduction Summary

## Overview

This repository provides a reproducible implementation of the final
MedSAM-LoRA configuration for white matter hyperintensity (WMH) segmentation.

The pipeline adapts MedSAM ViT-B using a custom LoRA implementation while
preserving the original patient splits, preprocessing, training configuration,
and evaluation protocols.

---

# Model

MedSAM ViT-B with custom LoRA adaptation.

## LoRA configuration

- LoRA layers: 80
- Rank: 4
- Trainable parameters: 673,792
- Total parameters: 94,409,264

---

# Dataset

WMH dataset:

- Train patients: 42
- Validation patients: 9
- Test patients: 9

Axial slices:

| Split | Slices |
|---|---:|
| Train | 2506 |
| Validation | 537 |
| Test | 537 |

Test cohort:

- Positive slices: 232
- Empty slices: 305

---

# Training Configuration

Configuration:

- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Epochs: 20
- Batch size: 2

The best checkpoint is selected using validation Dice with strict
validation improvement.

---

# Reference Run

Best epoch:

9

Best validation Dice:

0.8892

---

# Test Results

Overall test evaluation:

| Metric | Value |
|---|---:|
| Dice | 0.8739 |
| IoU | 0.8125 |
| Sensitivity | 0.8881 |
| Precision | 0.9113 |

Evaluation protocols:

- F144/F146 batch-level evaluation
- F148 site-wise slice-level reporting

---

# Site-wise Dice

| Site | Dice |
|---|---:|
| Amsterdam | 0.9219 |
| Singapore | 0.8477 |
| Utrecht | 0.8312 |

---

# Reproducibility Artifacts

The repository includes:

- reproducible training/evaluation notebook
- environment information
- observed package versions
- split records
- model configuration files
- evaluation outputs

---

# Limitations

This implementation evaluates a prompted segmentation setting where
box prompts are derived from ground-truth annotations.

Therefore, it should not be interpreted as fully automatic WMH segmentation.

---

# Future Work

Potential extensions include:

- additional external validation
- LoRA placement ablation studies
- automatic prompt generation
- qualitative error analysis
- model deployment studies
