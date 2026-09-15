
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

BENCHMARK_DIR = PROJECT_DIR / "results" / "benchmark"

FIG_DIR = PROJECT_DIR / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------
# Figure 1: Overall metrics
# -------------------------

overall = pd.read_csv(
    BENCHMARK_DIR / "benchmark_overall.csv"
)


metrics = [
    "dice",
    "iou",
    "sensitivity",
    "precision"
]


plt.figure(figsize=(8,5))

x = range(len(metrics))

medsam_values = [
    overall.loc[
        overall["model"]=="MedSAM-LoRA",
        m
    ].values[0]
    for m in metrics
]


unet_values = [
    overall.loc[
        overall["model"]=="U-Net",
        m
    ].values[0]
    for m in metrics
]


width = 0.35

plt.bar(
    [i-width/2 for i in x],
    medsam_values,
    width,
    label="MedSAM-LoRA"
)

plt.bar(
    [i+width/2 for i in x],
    unet_values,
    width,
    label="U-Net"
)


plt.xticks(
    x,
    metrics
)

plt.ylabel("Score")

plt.title(
    "Overall Benchmark Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR / "overall_metrics_comparison.png",
    dpi=300
)

plt.close()



# -------------------------
# Figure 2: Site-wise Dice
# -------------------------

site = pd.read_csv(
    BENCHMARK_DIR / "benchmark_sitewise.csv"
)


plt.figure(figsize=(8,5))


sites = sorted(
    site["site"].unique()
)


medsam_dice = []
unet_dice = []


for s in sites:

    medsam_dice.append(
        site[
            (site["site"]==s)
            &
            (site["model"]=="MedSAM-LoRA")
        ]["dice"].values[0]
    )

    unet_dice.append(
        site[
            (site["site"]==s)
            &
            (site["model"]=="U-Net")
        ]["dice"].values[0]
    )


x = range(len(sites))


plt.bar(
    [i-0.2 for i in x],
    medsam_dice,
    width=0.4,
    label="MedSAM-LoRA"
)


plt.bar(
    [i+0.2 for i in x],
    unet_dice,
    width=0.4,
    label="U-Net"
)


plt.xticks(
    x,
    sites
)

plt.ylabel("Dice")

plt.title(
    "Site-wise Dice Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR / "sitewise_dice_comparison.png",
    dpi=300
)

plt.close()



# -------------------------
# Figure 3: Dice difference
# -------------------------

paired = pd.read_csv(
    BENCHMARK_DIR / "benchmark_paired_slices.csv"
)


difference = paired[
    "dice_difference_medsam_minus_unet"
]


plt.figure(figsize=(8,5))


plt.hist(
    difference,
    bins=40
)


plt.xlabel(
    "Dice difference (MedSAM-LoRA - U-Net)"
)

plt.ylabel(
    "Number of slices"
)


plt.title(
    "Per-slice Dice Improvement Distribution"
)


plt.tight_layout()


plt.savefig(
    FIG_DIR / "dice_difference_distribution.png",
    dpi=300
)


plt.close()



print("Figures saved:")
for p in FIG_DIR.glob("*.png"):
    print(p)
