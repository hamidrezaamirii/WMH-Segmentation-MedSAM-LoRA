# WMH MedSAM-LoRA

## Parameter-Efficient Adaptation of MedSAM for White Matter Hyperintensity Segmentation

This repository provides a reproducible implementation of MedSAM adaptation for white matter hyperintensity (WMH) segmentation using custom LoRA.

The project includes:
- MedSAM-LoRA training pipeline
- LoRA ablation study
- U-Net benchmark comparison
- Multi-site evaluation
- Statistical analysis
- Qualitative visualization

---

# Overview

Medical foundation models such as MedSAM provide strong segmentation capabilities but full fine-tuning can be computationally expensive.

This project investigates parameter-efficient adaptation of MedSAM ViT-B using LoRA, updating only a small fraction of model parameters while keeping the backbone frozen.

---

# Method

## Backbone

- Model: MedSAM ViT-B
- Adaptation: Custom LoRA
- LoRA layers: 80

Trainable parameters:
673,792

Total parameters:
94,409,264


Only LoRA parameters are optimized during training.

---

# LoRA Ablation

Two LoRA configurations were evaluated (preliminary protocol):

| Configuration | Validation Dice |
|---|---:|
| LoRA rank 4 | 0.8739 |
| LoRA rank 8 | 0.8768 |

Note: this ablation used a preliminary evaluation protocol. The final benchmark — paired protocol, 9 test patients, 537 slices (`results/benchmark/`, provenance in `benchmark_provenance.json`) — used the **LoRA rank-4** checkpoint (epoch 9, validation Dice 0.8892): test Dice **0.8777** vs U-Net **0.6683**.

---

# Dataset

The project uses the WMH dataset with patient-level separation.

| Split | Patients | Slices |
|---|---:|---:|
| Training | 42 | 2506 |
| Validation | 9 | 537 |
| Test | 9 | 537 |

Test cohort:

- 537 slices
- 232 positive slices
- 305 empty slices

Sites:

- Amsterdam
- Singapore
- Utrecht

---

# Training Configuration

- Optimizer: AdamW
- Epochs: 20
- Batch size: 2
- Learning rate: 1e-4
- Weight decay: 1e-4

Best validation Dice:
0.8892


---

# Benchmark: U-Net vs MedSAM-LoRA

Both models were evaluated on the identical held-out test cohort:

- same 537 slices
- same ground-truth masks
- same evaluation protocol

No retraining was performed during benchmarking.

| Model | Dice | IoU | Sensitivity | Precision |
|---|---:|---:|---:|---:|
| MedSAM-LoRA | 0.8777 | 0.8177 | 0.8985 | 0.8849 |
| U-Net | 0.6683 | 0.6125 | 0.8636 | 0.6999 |

Dice improvement:
+0.2094


---

# Site-wise Generalization

| Site | MedSAM-LoRA Dice | U-Net Dice |
|---|---:|---:|
| Amsterdam | 0.9220 | 0.6650 |
| Singapore | 0.8477 | 0.6390 |
| Utrecht | 0.8311 | 0.7033 |

---

# Statistical Analysis

Paired slice-level Wilcoxon signed-rank testing was performed on the complete test cohort.

| Metric | p-value |
|---|---:|
| Dice | 3.05e-36 |
| IoU | 1.08e-36 |
| Precision | 2.83e-27 |
| Sensitivity | 0.029 |

---

# Qualitative Results

Representative qualitative comparisons are provided in:
figures/


Main comparison:

- Input FLAIR
- Ground truth
- U-Net
- MedSAM-LoRA (rank 8 — ablation-phase figures; to be regenerated with the final rank-4 checkpoint)

Additional qualitative analyses include:
- Frozen MedSAM
- LoRA rank 4
- LoRA rank 8

---

# Reproducibility

Final weights (verified epoch-9 LoRA checkpoint, SHA256 `fef10131c00ee63b5bb2e82d7c803655923c42e5c07b8080512a2658fcfb0f48`) are published on the Hugging Face Hub: https://huggingface.co/hamidrezaamiri/wmh-medsam-lora-final

The repository includes:

- training notebooks
- benchmark notebooks
- configuration files
- checkpoint metadata
- evaluation outputs
- statistical analysis

Important files:
configs/
results/
figures/
src/
notebooks/
environment.yml
requirements.txt
CITATION.cff


---

# Limitations

This repository evaluates a box-prompted WMH segmentation setting where prompts are derived from ground-truth annotations.

The current implementation is not presented as a fully automatic WMH segmentation system.

Future work includes:

- automatic prompt generation
- external validation
- larger multi-center evaluation
- improved prompt strategies

---

# Citation

Please cite this repository using:
CITATION.cff
