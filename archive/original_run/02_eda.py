"""
Step 3: Exploratory Data Analysis (EDA)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.facecolor": "white", "font.size": 10})

df = pd.read_csv("./winequality_clean.csv")

print("Descriptive statistics:\n", df.describe().round(2).to_string())

# ---------- 1. Quality distribution ----------
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["quality"].value_counts().sort_index()
bars = ax.bar(counts.index.astype(str), counts.values, color="#722F37")
ax.set_xlabel("Quality Score")
ax.set_ylabel("Count")
ax.set_title("Distribution of Wine Quality Scores", fontweight="bold")
ax.bar_label(bars, padding=3)
plt.tight_layout()
plt.savefig("./eda_quality_distribution.png", dpi=150)
plt.close()

# ---------- 2. Feature distributions ----------
features = [c for c in df.columns if c != "quality"]
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for ax, col in zip(axes.flat, features):
    sns.histplot(df[col], kde=True, ax=ax, color="#722F37", bins=25)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
axes.flat[-1].axis("off")
plt.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("./eda_feature_distributions.png", dpi=150)
plt.close()

# ---------- 3. Correlation heatmap ----------
fig, ax = plt.subplots(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            square=True, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("./eda_correlation.png", dpi=150)
plt.close()

print("\nCorrelation with quality (sorted):")
print(corr["quality"].drop("quality").sort_values(ascending=False).round(3).to_string())

# ---------- 4. Boxplots: key features vs quality ----------
top_features = corr["quality"].drop("quality").abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, top_features):
    sns.boxplot(data=df, x="quality", y=col, ax=ax, color="#B0555A", hue="quality", legend=False)
    ax.set_title(f"{col} vs Quality", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/wine/eda_top_features_vs_quality.png", dpi=150)
plt.close()

print("\nSaved: eda_quality_distribution.png, eda_feature_distributions.png, eda_correlation.png, eda_top_features_vs_quality.png")
