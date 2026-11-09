# Statistical Analysis (Per-Patient)

Paired Wilcoxon signed-rank comparison between MedSAM-LoRA and U-Net, using one paired observation per patient (patient-level means of the test slices). Slices from the same patient are correlated, so the patient (n=9), not the slice (n=537), is the independent unit of analysis. See `statistical_analysis.md` for the slice-level (optimistic) version, kept for transparency.

Cohort: 9 test patients (3 per site: Amsterdam, Singapore, Utrecht), with 48 or 83 test slices each (537 slices total).

| metric | n_patients | medsam_mean | unet_mean | mean_difference | median_difference | iqr_difference | wilcoxon_statistic | p_value | rank_biserial_effect_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dice | 9 | 0.866925 | 0.669077 | 0.197849 | 0.174587 | 0.104358 | 45 | 0.00195312 | 1 |
| iou | 9 | 0.802526 | 0.609691 | 0.192835 | 0.169747 | 0.099006 | 45 | 0.00195312 | 1 |
| sensitivity | 9 | 0.888717 | 0.84518 | 0.0435369 | 0.0235807 | 0.0723965 | 42 | 0.00976562 | 0.777778 |
| precision | 9 | 0.877243 | 0.711017 | 0.166226 | 0.154898 | 0.101064 | 45 | 0.00195312 | 1 |