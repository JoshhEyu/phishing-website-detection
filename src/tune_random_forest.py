
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer
)

# Project paths
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
models_dir = project_dir / "models"

# Load training and validation data
train_data = pd.read_csv(data_dir / "train.csv")
val_data = pd.read_csv(data_dir / "validation.csv")

X_train = train_data.drop(columns=["result"])
y_train = train_data["result"]

X_val = val_data.drop(columns=["result"])
y_val = val_data["result"]

# Hyperparameter search space
param_grid = {
    "n_estimators": [100, 150, 200, 250, 300],
    "max_depth": [None, 10, 15, 20, 25, 30],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Focus on the F1-score of the phishing class (-1)
phishing_f1 = make_scorer(f1_score, pos_label=-1)

# Initialize Random Forest
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Randomized hyperparameter search
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    scoring=phishing_f1,
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Train and tune
print("Starting Random Forest hyperparameter tuning...")

search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

print("\n--- BEST HYPERPARAMETERS ---")
print(search.best_params_)

print("\nBest cross-validation F1-score:", search.best_score_)

# Validation predictions
y_pred = best_model.predict(X_val)

print("\n--- TUNED MODEL VALIDATION RESULTS ---")

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, pos_label=-1))
print("Recall:", recall_score(y_val, y_pred, pos_label=-1))
print("F1-score:", f1_score(y_val, y_pred, pos_label=-1))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred, labels=[-1, 1]))

# Compare against the original Random Forest
baseline_path = models_dir / "random_forest.pkl"
baseline_model = joblib.load(baseline_path)

baseline_pred = baseline_model.predict(X_val)

print("\n--- BASELINE VS TUNED RANDOM FOREST ---")

print(
    "Baseline F1-score:",
    f1_score(y_val, baseline_pred, pos_label=-1)
)

print(
    "Tuned F1-score:",
    f1_score(y_val, y_pred, pos_label=-1)
)

# Save tuned model
model_path = models_dir / "tuned_random_forest.pkl"
joblib.dump(best_model, model_path)

print("\nTuned model saved successfully!")
print("Location:", model_path)
