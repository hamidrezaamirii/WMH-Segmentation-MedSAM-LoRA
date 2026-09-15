
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT = PROJECT_DIR / "results" / "benchmark" / "benchmark_paired_slices.csv"

OUTPUT_CSV = PROJECT_DIR / "results" / "benchmark" / "statistical_analysis.csv"
OUTPUT_MD = PROJECT_DIR / "results" / "benchmark" / "statistical_analysis.md"


df = pd.read_csv(INPUT)

print("Loaded:")
print(INPUT)

print("\nRows:", len(df))


def rank_biserial_effect(medsam, unet):
    diff = medsam - unet

    positive = np.sum(diff > 0)
    negative = np.sum(diff < 0)

    if positive + negative == 0:
        return 0.0

    return (positive - negative) / (positive + negative)


def analyze(metric):

    medsam = df[f"{metric}_medsam"].to_numpy()
    unet = df[f"{metric}_unet"].to_numpy()

    diff = medsam - unet

    statistic, p_value = wilcoxon(
        medsam,
        unet,
        alternative="greater"
    )

    return {
        "metric": metric,
        "n": len(diff),

        "medsam_mean": float(np.mean(medsam)),
        "unet_mean": float(np.mean(unet)),

        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),

        "iqr_difference": float(
            np.percentile(diff, 75)
            -
            np.percentile(diff, 25)
        ),

        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),

        "rank_biserial_effect_size": float(
            rank_biserial_effect(medsam, unet)
        )
    }


metrics = [
    "dice",
    "iou",
    "sensitivity",
    "precision"
]


results = []

for metric in metrics:
    results.append(
        analyze(metric)
    )


result_df = pd.DataFrame(results)


OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_CSV,
    index=False
)


with open(OUTPUT_MD, "w") as f:

    f.write("# Statistical Analysis\n\n")

    f.write(
        "Paired Wilcoxon signed-rank comparison "
        "between MedSAM-LoRA and U-Net.\n\n"
    )

    f.write(
        "Cohort: 537 paired slices from the unified benchmark.\n\n"
    )

    f.write(
        result_df.to_markdown(index=False)
    )


print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_MD)
