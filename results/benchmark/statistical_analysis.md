# Statistical Analysis

Paired Wilcoxon signed-rank comparison between MedSAM-LoRA and U-Net.

Cohort: 537 paired slices from the unified benchmark.

| metric      |   n |   medsam_mean |   unet_mean |   mean_difference |   median_difference |   iqr_difference |   wilcoxon_statistic |     p_value |   rank_biserial_effect_size |
|:------------|----:|--------------:|------------:|------------------:|--------------------:|-----------------:|---------------------:|------------:|----------------------------:|
| dice        | 537 |      0.877692 |    0.66827  |         0.209422  |                   0 |         0.184479 |                46411 | 3.04737e-36 |                   0.625     |
| iou         | 537 |      0.817739 |    0.612527 |         0.205212  |                   0 |         0.193018 |                46547 | 1.07979e-36 |                   0.625     |
| sensitivity | 537 |      0.89852  |    0.863615 |         0.0349054 |                   0 |         0        |                14943 | 0.029013    |                  -0.0438596 |
| precision   | 537 |      0.884887 |    0.699936 |         0.184951  |                   0 |         0.136252 |                43007 | 2.82538e-27 |                   0.559748  |