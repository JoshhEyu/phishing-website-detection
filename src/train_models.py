
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Locate project directories
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
models_dir = project_dir / "models"

models_dir.mkdir(parents=True, exist_ok=True)

# Load training and validation data
train_data = pd.read_csv(data_dir / "train.csv")
val_data = pd.read_csv(data_dir / "validation.csv")

# Separate features and target
X_train = train_data.drop(columns=["result"])
y_train = train_data["result"]

X_val = val_data.drop(columns=["result"])
y_val = val_data["result"]

# Build classification pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train model
print("Training Logistic Regression...")
model.fit(X_train, y_train)

# Make validation predictions
y_pred = model.predict(X_val)

# Evaluate model
print("\n--- VALIDATION RESULTS ---")

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(
    y_val, y_pred, pos_label=-1
))
print("Recall:", recall_score(
    y_val, y_pred, pos_label=-1
))
print("F1-score:", f1_score(
    y_val, y_pred, pos_label=-1
))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred, labels=[-1, 1]))

# Save trained model
model_path = models_dir / "logistic_regression.pkl"
joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)
