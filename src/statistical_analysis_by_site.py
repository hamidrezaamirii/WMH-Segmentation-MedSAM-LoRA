
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT = PROJECT_DIR / "results" / "benchmark" / "benchmark_paired_slices.csv"

OUTPUT_CSV = PROJECT_DIR / "results" / "benchmark" / "statistical_analysis_by_site.csv"
OUTPUT_MD = PROJECT_DIR / "results" / "benchmark" / "statistical_analysis_by_site.md"


df = pd.read_csv(INPUT)


def rank_biserial_effect(medsam, unet):

    diff = medsam - unet

    positive = np.sum(diff > 0)
    negative = np.sum(diff < 0)

    if positive + negative == 0:
        return 0.0

    return (positive - negative) / (positive + negative)


def analyze(site, metric):

    subset = df[df["site"] == site]

    medsam = subset[f"{metric}_medsam"].to_numpy()
    unet = subset[f"{metric}_unet"].to_numpy()

    diff = medsam - unet

    statistic, p_value = wilcoxon(
        medsam,
        unet,
        alternative="greater"
    )

    return {
        "site": site,
        "metric": metric,
        "n": len(diff),

        "medsam_mean": np.mean(medsam),
        "unet_mean": np.mean(unet),

        "mean_difference": np.mean(diff),

        "wilcoxon_statistic": statistic,
        "p_value": p_value,

        "rank_biserial_effect_size":
            rank_biserial_effect(medsam, unet)
    }


results = []


for site in sorted(df["site"].unique()):

    for metric in [
        "dice",
        "iou",
        "sensitivity",
        "precision"
    ]:

        results.append(
            analyze(site, metric)
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

    f.write("# Site-wise Statistical Analysis\n\n")

    f.write(
        "Paired Wilcoxon signed-rank comparison "
        "between MedSAM-LoRA and U-Net by site.\n\n"
    )

    f.write(
        result_df.to_markdown(index=False)
    )


print("Saved:")
print(OUTPUT_CSV)
print(OUTPUT_MD)
