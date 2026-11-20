
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Project paths
project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
models_dir = project_dir / "models"
reports_dir = project_dir / "reports"

reports_dir.mkdir(parents=True, exist_ok=True)

# Load the untouched test dataset
test_data = pd.read_csv(data_dir / "test.csv")

X_test = test_data.drop(columns=["result"])
y_test = test_data["result"]

# Load the selected tuned model
model_path = models_dir / "tuned_random_forest.pkl"
model = joblib.load(model_path)

# Make predictions
print("Evaluating the tuned Random Forest on test data...")

y_pred = model.predict(X_test)

# Treat phishing (-1) as the positive class
y_test_binary = (y_test == -1).astype(int)

phishing_index = list(model.classes_).index(-1)
y_score = model.predict_proba(X_test)[:, phishing_index]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=-1)
recall = recall_score(y_test, y_pred, pos_label=-1)
f1 = f1_score(y_test, y_pred, pos_label=-1)
roc_auc = roc_auc_score(y_test_binary, y_score)

print("\n--- FINAL TEST RESULTS ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Classification report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_test,
    y_pred,
    labels=[-1, 1],
    target_names=["Phishing", "Legitimate"]
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

print("--- CONFUSION MATRIX ---")
print(cm)

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    y_score
)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Phishing Website Detection")
plt.legend()
plt.grid(alpha=0.3)

roc_path = reports_dir / "final_roc_curve.png"
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Save final metrics
metrics = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "ROC-AUC": roc_auc
}])

metrics_path = reports_dir / "final_test_metrics.csv"
metrics.to_csv(metrics_path, index=False)

# Save classification report
report = classification_report(
    y_test,
    y_pred,
    labels=[-1, 1],
    target_names=["Phishing", "Legitimate"]
)

report_path = reports_dir / "final_classification_report.txt"
report_path.write_text(report, encoding="utf-8")

print("\nFinal evaluation completed!")
print("ROC curve:", roc_path)
print("Metrics:", metrics_path)
print("Classification report:", report_path)
