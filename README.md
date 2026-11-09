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

# Fully Automatic Evaluation

A second, fully automatic setting was evaluated (preprint Section 4.5): a U-Net detector trained on all training slices proposes boxes without ground truth, and the same frozen MedSAM-LoRA checkpoint delineates within them. The detector is a standard 2D U-Net with the same architecture as the baseline, trained on all slices of the 42 training patients at 256x256 (BCE + soft Dice, AdamW lr 1e-4, batch 4, 20 epochs, seed 42); best checkpoint epoch 11 by validation Dice 0.7748.

| Method | Volumetric Dice | Volumetric IoU | Sensitivity | Precision |
|---|---:|---:|---:|---:|
| Automatic (detector + MedSAM-LoRA) | 0.6762 | 0.5311 | 0.7597 | 0.6829 |
| Detector alone | 0.6678 | 0.5177 | 0.7107 | 0.6983 |
| U-Net | 0.7044 | 0.5543 | 0.7882 | 0.6684 |
| Oracle reference (ground-truth boxes) | 0.7660 | 0.6246 | — | — |

Volumetric Dice 0.6762 vs 0.7044 for U-Net (mean paired difference -0.028, Wilcoxon p = 0.43, not significant). The detector alone scores 0.6678, so MedSAM adds only +0.008 on top of the detector: the bottleneck is localization, not delineation. A post-hoc particle filter (components <50 px) lifts Dice only to 0.6796 and is reported as a sensitivity analysis, not a primary result.

Lesion recall by size band (per-slice 2D components; same mm3 definition for both arms):

| Lesion size | Automatic recall | U-Net recall |
|---|---:|---:|
| Small (<50 mm3) | 0.653 (828/1,268) | 0.777 (985/1,268) |
| Medium (50-500 mm3) | 0.952 (376/395) | 0.972 (384/395) |
| Large (>500 mm3) | 0.911 (51/56) | 0.982 (55/56) |
| Total | 0.730 (1,255/1,719) | 0.828 (1,424/1,719) |

U-Net wins lesion recall in every size band. The automatic pipeline is a clean negative result: it does not beat the U-Net baseline.

---

# Statistical Analysis

The primary analysis aggregates to the patient level (n = 9): slices from the same patient are correlated, so the patient, not the slice, is the independent unit. Paired exact Wilcoxon signed-rank tests on per-patient mean metrics; the primary endpoint is per-patient mean Dice.

| Metric | MedSAM-LoRA mean | U-Net mean | Mean paired difference | p-value (two-sided) | Rank-biserial r |
|---|---:|---:|---:|---:|---:|
| Dice | 0.8669 | 0.6691 | +0.198 (95% CI 0.143-0.253) | 0.004 | 1.00 |
| IoU | 0.8025 | 0.6097 | +0.193 | 0.004 | 1.00 |
| Sensitivity | 0.8887 | 0.8452 | +0.044 | 0.020 | 0.87 |
| Precision | 0.8772 | 0.7110 | +0.166 | 0.004 | 1.00 |

Note: p = 0.004 is the smallest value a two-sided Wilcoxon test can return at n = 9. Slice-level tests (e.g. Dice p = 3.05e-36) are anti-conservative and are reported in `results/benchmark/statistical_analysis.md` for transparency only; they are not used for inference. Full per-patient results are in `results/benchmark/statistical_analysis_per_patient.csv` and `.md`.

---

# Qualitative Results

Representative qualitative comparisons are provided in:
figures/


Main comparison:

- Input FLAIR
- Ground truth
- U-Net
- MedSAM-LoRA (rank 4, final checkpoint; `figures/main_qualitative_figure_rank4.png`, same as preprint Figure 1)

Additional qualitative analyses include:
- Frozen MedSAM
- LoRA rank 4
- LoRA rank 8

---

# Reproducibility

Final weights (verified epoch-9 LoRA checkpoint, SHA256 `fef10131c00ee63b5bb2e82d7c803655923c42e5c07b8080512a2658fcfb0f48`) are published on the Hugging Face Hub: https://huggingface.co/hamidrezaamiri/wmh-medsam-lora-final

Per-patient results (Dice, IoU, sensitivity, precision for all 9 test patients) are in `results/benchmark/statistical_analysis_per_patient.csv`.

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

This repository evaluates a box-prompted WMH segmentation setting where prompts are derived from ground-truth annotations; the prompted results characterize a semi-automatic, human-in-the-loop workflow, not fully automatic segmentation.

A fully automatic evaluation is also reported (see above): with detector-proposed boxes, the pipeline reaches volumetric Dice 0.6762 against 0.7044 for U-Net (not significant) and trails U-Net on lesion recall in every size band. The bottleneck is localization, not delineation. A single outlier patient (Utrecht 23) drives much of the gap; without it the two arms tie (0.7273 vs 0.7265).

Future work includes:

- a stronger detector / prompt-free distillation
- external validation on an independent cohort
- larger multi-center evaluation
- improved prompt strategies

---

# Citation

Please cite this repository using:
CITATION.cff
