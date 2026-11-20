
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Locate project directories
project_dir = Path(__file__).resolve().parents[1]

data_path = project_dir / "data" / "phishing_websites_clean.csv"
data_dir = project_dir / "data"

# Load cleaned dataset
df = pd.read_csv(data_path)

# Separate input features and target
X = df.drop(columns=["result"])
y = df["result"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: divide temporary data equally
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Combine features and targets
train_data = X_train.copy()
train_data["result"] = y_train

val_data = X_val.copy()
val_data["result"] = y_val

test_data = X_test.copy()
test_data["result"] = y_test

# Save datasets
train_data.to_csv(data_dir / "train.csv", index=False)
val_data.to_csv(data_dir / "validation.csv", index=False)
test_data.to_csv(data_dir / "test.csv", index=False)

# Display results
print("\n--- DATA SPLITTING RESULTS ---")

print("Training:", train_data.shape)
print("Validation:", val_data.shape)
print("Testing:", test_data.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation class distribution:")
print(y_val.value_counts(normalize=True) * 100)

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True) * 100)

print("\nAll datasets saved successfully!")
