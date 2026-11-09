"""Per-patient statistical analysis: MedSAM-LoRA vs U-Net.

Why this exists
---------------
The original statistical_analysis.py runs the paired Wilcoxon signed-rank test
on individual slices (n=537). Slices from the same patient are correlated, so
the effective independent sample is the patient (n=9), not the slice. This
script first averages each metric per patient and then runs the paired test on
the 9 patient-level pairs -- the correct unit of analysis for patient-level
splits. The p-values here are honest; the slice-level ones are optimistic.

Usage: python src/statistical_analysis_per_patient.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT = PROJECT_DIR / "results" / "benchmark" / "benchmark_paired_slices.csv"

OUTPUT_CSV = PROJECT_DIR / "results" / "benchmark" / "statistical_analysis_per_patient.csv"
OUTPUT_MD = PROJECT_DIR / "results" / "benchmark" / "statistical_analysis_per_patient.md"


df = pd.read_csv(INPUT)

print("Loaded:", INPUT)
print("Slices:", len(df), "| Patients:", df["patient_id"].nunique())

# Patient-level means (one paired observation per patient).
patient_means = (
    df.groupby(["site", "patient_id"])
    [[f"{m}_medsam" for m in ("dice", "iou", "sensitivity", "precision")]
     + [f"{m}_unet" for m in ("dice", "iou", "sensitivity", "precision")]]
    .mean()
    .reset_index()
)

print("Patient rows:", len(patient_means))
print("Slices per patient:\n", df.groupby("patient_id").size().to_string())


def rank_biserial_effect(medsam, unet):
    diff = medsam - unet
    positive = np.sum(diff > 0)
    negative = np.sum(diff < 0)
    if positive + negative == 0:
        return 0.0
    return (positive - negative) / (positive + negative)


def analyze(metric):
    medsam = patient_means[f"{metric}_medsam"].to_numpy()
    unet = patient_means[f"{metric}_unet"].to_numpy()
    diff = medsam - unet
    statistic, p_value = wilcoxon(medsam, unet, alternative="greater")
    return {
        "metric": metric,
        "n_patients": len(diff),
        "medsam_mean": float(np.mean(medsam)),
        "unet_mean": float(np.mean(unet)),
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "iqr_difference": float(np.percentile(diff, 75) - np.percentile(diff, 25)),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "rank_biserial_effect_size": float(rank_biserial_effect(medsam, unet)),
    }


results = [analyze(m) for m in ("dice", "iou", "sensitivity", "precision")]
result_df = pd.DataFrame(results)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_CSV, index=False)

def _md_table(df_):
    cols = list(df_.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(" --- " for _ in cols) + "|"]
    for _, row in df_.iterrows():
        lines.append("| " + " | ".join(
            f"{v:.6g}" if isinstance(v, float) else str(v)
            for v in row
        ) + " |")
    return "\n".join(lines)


with open(OUTPUT_MD, "w") as f:
    f.write("# Statistical Analysis (Per-Patient)\n\n")
    f.write(
        "Paired Wilcoxon signed-rank comparison between MedSAM-LoRA and U-Net, "
        "using one paired observation per patient (patient-level means of the "
        "test slices). Slices from the same patient are correlated, so the "
        "patient (n=9), not the slice (n=537), is the independent unit of "
        "analysis. See `statistical_analysis.md` for the slice-level (optimistic) "
        "version, kept for transparency.\n\n"
    )
    f.write(
        "Cohort: 9 test patients (3 per site: Amsterdam, Singapore, Utrecht), "
        "with 48 or 83 test slices each (537 slices total).\n\n"
    )
    f.write(_md_table(result_df))

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_MD)
print("\n" + result_df[["metric", "medsam_mean", "unet_mean", "mean_difference", "p_value"]].to_string(index=False))
