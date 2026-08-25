import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
OUT = "figures"
import os
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("diabetes_raw.csv")

print("=== Shape ===")
print(df.shape)

print("\n=== Data types ===")
print(df.dtypes)

print("\n=== Duplicate rows ===")
print(df.duplicated().sum())

print("\n=== Summary statistics ===")
summary = df.describe().T
summary.to_csv("summary_statistics.csv")
print(summary)

# Columns where a physiological zero is impossible -> these zeros are
# disguised missing values, not real measurements.
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\n=== Count of biologically-impossible zeros (disguised missing values) ===")
zero_counts = (df[zero_invalid_cols] == 0).sum()
print(zero_counts)
zero_counts.to_csv("zero_value_counts.csv")

print("\n=== True NaN count (none expected, since missingness is encoded as 0) ===")
print(df.isnull().sum())

print("\n=== Target class distribution ===")
target_counts = df["Outcome"].value_counts()
target_pct = df["Outcome"].value_counts(normalize=True) * 100
print(target_counts)
print(target_pct)

# --- Plot: target class distribution ---
plt.figure(figsize=(5, 4))
sns.countplot(x="Outcome", data=df, palette=["#4C72B0", "#DD8452"])
plt.title("Target Class Distribution (0 = No Diabetes, 1 = Diabetes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUT}/target_distribution.png", dpi=150)
plt.close()

# --- Plot: feature distributions ---
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.ravel()
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=30, color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[i].set_title(col, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/feature_distributions.png", dpi=150)
plt.close()

# --- Plot: correlation heatmap ---
plt.figure(figsize=(9, 7))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=150)
plt.close()

# --- Plot: boxplots by outcome for top correlated features ---
top_feats = corr["Outcome"].abs().sort_values(ascending=False).index[1:5]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(top_feats):
    sns.boxplot(x="Outcome", y=col, data=df, ax=axes[i], palette=["#4C72B0", "#DD8452"])
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT}/boxplots_top_features.png", dpi=150)
plt.close()

print("\nEDA complete. Figures saved to ./figures/")
print("Top features correlated with Outcome:")
print(corr["Outcome"].abs().sort_values(ascending=False))
