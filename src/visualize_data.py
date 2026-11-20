
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate project directories
project_dir = Path(__file__).resolve().parents[1]
data_path = project_dir / "data" / "phishing_websites_clean.csv"
reports_dir = project_dir / "reports"

reports_dir.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(data_path)

sns.set_theme(style="whitegrid")

# 1. Target class distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="result",
    hue="result",
    palette="Set2",
    legend=False
)

plt.title("Phishing vs Legitimate Websites")
plt.xlabel("Website Classification")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["Phishing (-1)", "Legitimate (1)"])

plt.tight_layout()
plt.savefig(reports_dir / "class_distribution.png")
plt.show()


# 2. Feature distributions
selected_features = [
    "having_ip_address",
    "url_length",
    "sslfinal_state",
    "age_of_domain"
]

for feature in selected_features:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x=feature,
        hue=feature,
        palette="Set2",
        legend=False
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Records")

    plt.tight_layout()
    plt.savefig(reports_dir / f"{feature}_distribution.png")
    plt.show()


# 3. Feature correlation heatmap
plt.figure(figsize=(18, 14))

correlation = df.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    linewidths=0.2
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig(reports_dir / "correlation_heatmap.png")
plt.show()

print("\nAll visualizations generated successfully!")
print("Saved in:", reports_dir)
# 4. Features most correlated with the target

correlation_with_target = (
    df.corr()["result"]
    .drop("result")
    .sort_values(key=abs, ascending=False)
)

print("\n--- FEATURE CORRELATION WITH TARGET ---")
print(correlation_with_target)

print("\n--- TOP 10 FEATURES ---")
print(correlation_with_target.head(10))