from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = "dataset/fatigue_data.csv"
OUTPUT_DIRECTORY = Path("images/analysis")

FEATURES = [
    "mean_ear",
    "min_ear",
    "blink_rate",
    "mean_blink_duration",
    "max_blink_duration",
    "perclos"
]


# Load dataset
df = pd.read_csv(DATASET_PATH)


# --------------------------------------------------
# 1. Feature statistics by fatigue level
# --------------------------------------------------

print("\n==============================")
print("Feature Statistics by Class")
print("==============================")

print(
    df.groupby("fatigue_label")[FEATURES]
    .mean()
    .round(3)
)


# --------------------------------------------------
# 2. Feature distributions
# --------------------------------------------------

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

for feature in FEATURES:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="fatigue_label",
        y=feature
    )

    plt.title(f"{feature} vs Fatigue Level")
    plt.xlabel("Fatigue Level")
    plt.ylabel(feature)

    plt.tight_layout()

    output_path = OUTPUT_DIRECTORY / f"analysis_{feature}.png"
    plt.savefig(output_path, dpi=150)

    plt.close()


# --------------------------------------------------
# 3. Feature correlation
# --------------------------------------------------

plt.figure(figsize=(9, 7))

correlation = df[FEATURES].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.tight_layout()

correlation_path = OUTPUT_DIRECTORY / "feature_correlation.png"
plt.savefig(correlation_path, dpi=150)

plt.close()


print("\nAnalysis complete.")

print("\nGenerated files:")

for feature in FEATURES:
    print(OUTPUT_DIRECTORY / f"analysis_{feature}.png")

print(correlation_path)