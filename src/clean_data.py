
import pandas as pd
from pathlib import Path

# Locate project directories
project_dir = Path(__file__).resolve().parents[1]

input_path = project_dir / "data" / "phishing_websites.csv"
output_path = project_dir / "data" / "phishing_websites_clean.csv"

# Load original dataset
df = pd.read_csv(input_path)

print("Original dataset:", df.shape)

# Separate feature columns from target
feature_columns = df.columns.drop("result")

# Step 1: Remove exact duplicate rows
df_clean = df.drop_duplicates().copy()

print("After exact deduplication:", df_clean.shape)

# Step 2: Identify conflicting feature combinations
label_counts = df_clean.groupby(
    list(feature_columns)
)["result"].nunique()

conflicting_keys = label_counts[label_counts > 1].index

# Step 3: Remove ambiguous feature combinations
df_clean = df_clean.set_index(
    list(feature_columns)
)

df_clean = df_clean.drop(
    index=conflicting_keys
).reset_index()

# Step 4: Save cleaned dataset
df_clean.to_csv(output_path, index=False)

# Step 5: Display cleaning results
print("\n--- CLEANING SUMMARY ---")

print("Final dataset shape:", df_clean.shape)

print("\nTarget distribution:")
print(df_clean["result"].value_counts())

print("\nTarget percentages:")
print(df_clean["result"].value_counts(normalize=True) * 100)

print("\nRemaining duplicate rows:", df_clean.duplicated().sum())

print("\nSaved cleaned dataset to:")
print(output_path)
