# Phishing Website Detection Using Machine Learning

## 1. Project Overview

This project develops a machine learning-based system for detecting phishing websites using extracted website characteristics.

The system analyzes 30 website features and classifies each website as either:

- Phishing (-1)
- Legitimate (1)

A Streamlit web application provides an interface for uploading CSV files, generating predictions, viewing results, and downloading prediction reports.

**Note:** The application accepts extracted website features in CSV format. It does not directly analyze raw website URLs.

## 2. Model Performance

The final selected model is a tuned Random Forest Classifier.

| Evaluation Metric | Test Result |
|---|---:|
| Accuracy | 96.97% |
| Precision | 97.72% |
| Recall | 96.40% |
| F1-Score | 97.05% |
| ROC-AUC | 99.41% |

## 3. Project Structure

```text
phishing_Website_Detection/
├── app/
│   └── app.py
├── data/
│   ├── phishing_websites.csv
│   ├── phishing_websites_clean.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── tuned_random_forest.pkl
├── notebooks/
├── reports/
│   ├── evidence/
│   └── [evaluation reports and visualizations]
├── src/
│   ├── clean_data.py
│   ├── compare_models.py
│   ├── download_data.py
│   ├── evaluate_final_model.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── tune_random_forest.py
│   └── visualize_data.py
├── .gitignore
├── requirements.txt
└── README.md
```
## 4. Live Demo

Try the deployed application:

[Open Phishing Website Detection App](https://phishing-website-detection-wndwdlz8r7gp6ckag6hcde.streamlit.app/)

## 5. Author & Copyright

**Project Author:** Eyasu Kebede

© 2026 Eyasu Kebede. All Rights Reserved.

This project is made available for viewing and personal or educational study only. Copying, redistribution, modification for distribution, and commercial use of the original project materials require prior written permission from the copyright holder.

For further details, see the [LICENSE](LICENSE) file.
