# WMH MedSAM-LoRA Benchmark Results

## U-Net vs MedSAM-LoRA

This benchmark evaluates both models on the identical held-out test cohort.

Evaluation protocol:

- 9 test patients
- 537 axial slices
- 232 positive slices
- 305 empty slices
- Sites: Amsterdam, Singapore, Utrecht

Both models were evaluated with identical target masks and identical scoring protocol.

---

# Overall Performance

| Model | Dice | IoU | Sensitivity | Precision |
|---|---:|---:|---:|---:|
| MedSAM-LoRA | 0.877692 | 0.817739 | 0.898520 | 0.884887 |
| U-Net | 0.668270 | 0.612527 | 0.863615 | 0.699936 |

MedSAM-LoRA improved Dice by:

**+0.209422**

compared with U-Net on the unified benchmark cohort.

---

# Site-wise Performance

| Site | MedSAM-LoRA Dice | U-Net Dice |
|---|---:|---:|
| Amsterdam | 0.921991 | 0.664951 |
| Singapore | 0.847664 | 0.638987 |
| Utrecht | 0.831122 | 0.703291 |

MedSAM-LoRA showed consistent improvement across all three independent sites.

---

# Positive Slice Performance

Only slices containing WMH annotations:

| Model | Dice |
|---|---:|
| MedSAM-LoRA | 0.716900 |
| U-Net | 0.611470 |

---

# Empty Slice Performance

Slices without WMH annotations:

| Model | Dice |
|---|---:|
| MedSAM-LoRA | 1.000000 |
| U-Net | 0.711475 |

---

# Reproducibility

The benchmark stores:

- checkpoint hashes
- dataset verification hashes
- paired slice identifiers
- per-slice predictions
- patient-level summaries

Generated files:

- benchmark_overall.csv
- benchmark_sitewise.csv
- benchmark_positive_empty.csv
- benchmark_per_slice.csv
- benchmark_paired_slices.csv
- benchmark_patient_slice_means.csv

No retraining or checkpoint modification was performed during benchmarking.