
import pandas as pd
from pathlib import Path

# Locate and load the dataset
project_dir = Path(__file__).resolve().parents[1]
file_path = project_dir / "data" / "phishing_websites.csv"

df = pd.read_csv(file_path)

# 1. Dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 2. Column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 3. Data types and information
print("\n--- DATASET INFORMATION ---")
df.info()

# 4. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())

# 5. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())

# 6. Target distribution
print("\n--- TARGET DISTRIBUTION ---")
print(df["result"].value_counts())

# 7. Target percentages
print("\n--- TARGET PERCENTAGES ---")
print(df["result"].value_counts(normalize=True) * 100)

# 8. Investigate duplicate records

print("\n--- DUPLICATE INVESTIGATION ---")

# Separate input features and target
X = df.drop(columns=["result"])

# Check duplicate feature combinations
duplicate_features = X.duplicated().sum()

print("Duplicate feature combinations:", duplicate_features)

# Check whether identical features have conflicting labels
label_counts = df.groupby(
    list(X.columns)
)["result"].nunique()

conflicting_groups = (label_counts > 1).sum()

print("Feature groups with conflicting labels:", conflicting_groups)

# Check dataset size after removing exact duplicate rows
cleaned_df = df.drop_duplicates()

print("Original dataset shape:", df.shape)
print("Shape after removing exact duplicates:", cleaned_df.shape)

# 9. Measure conflicting feature groups

feature_columns = df.columns.drop("result")

# Remove exact duplicate rows first
deduplicated_df = df.drop_duplicates()

# Group identical feature combinations
grouped = deduplicated_df.groupby(
    list(feature_columns)
)

# Find groups having more than one target label
conflicting_groups = grouped.filter(
    lambda group: group["result"].nunique() > 1
)

print("\n--- CONFLICTING RECORD ANALYSIS ---")

print(
    "Conflicting feature groups:",
    conflicting_groups.groupby(
        list(feature_columns)
    ).ngroups
)

print(
    "Records involved in conflicts:",
    len(conflicting_groups)
)

print(
    "Records remaining after removing conflicts:",
    len(deduplicated_df) - len(conflicting_groups)
)

print("\nTarget distribution after exact deduplication:")
print(deduplicated_df["result"].value_counts())

print("\nTarget distribution of conflicting records:")
print(conflicting_groups["result"].value_counts())
