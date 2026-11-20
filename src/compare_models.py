import pandas as pd
from pathlib import Path
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
models_dir = project_dir / "models"

train_data = pd.read_csv(data_dir / "train.csv")
val_data = pd.read_csv(data_dir / "validation.csv")

X_train = train_data.drop(columns=["result"])
y_train = train_data["result"]

X_val = val_data.drop(columns=["result"])
y_val = val_data["result"]

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, pos_label=-1),
        "Recall": recall_score(y_val, y_pred, pos_label=-1),
        "F1-score": f1_score(y_val, y_pred, pos_label=-1)
    })

    joblib.dump(model, models_dir / f"{name.lower().replace(' ', '_')}.pkl")

comparison = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(comparison.to_string(index=False))

print("\nModels saved successfully!")