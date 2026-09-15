# Final Results — WMH MedSAM-LoRA

## Reference Run

Configuration:
- MedSAM ViT-B with custom LoRA adaptation
- LoRA rank: 4
- LoRA layers: 80
- Trainable parameters: 673,792
- Total parameters: 94,409,264

Training:
- Epochs: 20
- Batch size: 2
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4

Best checkpoint:
- Selected by validation Dice
- Best epoch: 9

---

# Validation Performance

Best validation Dice:

0.8892

---

# Test Performance

Test cohort:
- 537 slices
- Positive slices: 232
- Empty slices: 305

## Overall Test Metrics

| Metric | Value |
|---|---:|
| Dice | 0.8739 |
| IoU | 0.8125 |
| Sensitivity | 0.8881 |
| Precision | 0.9113 |

Evaluation protocol:
- F144/F146 historical batch-level evaluation

---

# Site-wise Performance

| Site | Dice |
|---|---:|
| Amsterdam | 0.9219 |
| Singapore | 0.8477 |
| Utrecht | 0.8312 |

---

# Historical Reference Comparison

The reproduced run achieved:

- Dice: 0.8739

Historical reference:

- Dice: 0.8706

Difference:

+0.0033 Dice

This confirms successful reproduction of the final configuration within expected run variation.

---

# Notes

This experiment evaluates a prompted segmentation setting where
box prompts are derived from ground-truth annotations.

The model is not presented as a fully automatic WMH segmentation system.