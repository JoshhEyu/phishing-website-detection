
from ucimlrepo import fetch_ucirepo
from pathlib import Path
import pandas as pd

# Download the UCI Phishing Websites dataset
print("Downloading dataset...")

phishing = fetch_ucirepo(id=327)

# Combine features and target
X = phishing.data.features
y = phishing.data.targets

data = pd.concat([X, y], axis=1)

# Locate the project's data folder
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Save dataset
file_path = data_dir / "phishing_websites.csv"
data.to_csv(file_path, index=False)

print("\nDownload completed!")
print("Dataset shape:", data.shape)
print("Saved to:", file_path)
print("\nFirst five records:")
print(data.head())
