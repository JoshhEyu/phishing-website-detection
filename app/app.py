
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Locate project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "models" / "tuned_random_forest.pkl"
TRAIN_PATH = PROJECT_DIR / "data" / "train.csv"

# Page configuration
st.set_page_config(
    page_title="Phishing Website Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Load expected feature names
@st.cache_data
def load_features():
    columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
    return [column for column in columns if column != "result"]

feature_columns = load_features()

# Application interface
st.title("🛡️ Phishing Website Detection System")

st.write(
    "A Machine Learning application that classifies "
    "websites using extracted website features."
)

st.divider()

# Model performance
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Random Forest")

with col2:
    st.metric("Test Accuracy", "96.97%")

with col3:
    st.metric("ROC-AUC", "99.41%")

st.divider()

# CSV upload
st.subheader("Upload Website Feature Data")

st.info(
    "Upload a CSV file containing the 30 extracted website features. "
    "This application does not directly analyze raw website URLs."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        st.write("Uploaded dataset preview:")
        st.dataframe(uploaded_data.head())

        # Remove target column if present
        input_data = uploaded_data.drop(
            columns=["result"],
            errors="ignore"
        )

        # Validate feature columns
        missing_columns = [
            col for col in feature_columns
            if col not in input_data.columns
        ]

        extra_columns = [
            col for col in input_data.columns
            if col not in feature_columns
        ]

        if missing_columns or extra_columns:

            if missing_columns:
                st.error(f"Missing features: {missing_columns}")

            if extra_columns:
                st.error(f"Unexpected features: {extra_columns}")

        else:

            # Arrange features in training order
            input_data = input_data[feature_columns]

            # Validate numerical values
            input_data = input_data.apply(
                pd.to_numeric,
                errors="coerce"
            )

            if input_data.isnull().any().any():

                st.error(
                    "The uploaded data contains missing or "
                    "non-numeric feature values."
                )

            else:

                if st.button("Predict Website Classes"):

                    predictions = model.predict(input_data)

                    probabilities = model.predict_proba(input_data)

                    phishing_index = list(model.classes_).index(-1)

                    phishing_probabilities = probabilities[
                        :, phishing_index
                    ]

                    results = uploaded_data.copy()

                    results["Prediction"] = [
                        "Phishing" if value == -1 else "Legitimate"
                        for value in predictions
                    ]

                    results["Phishing_Probability"] = (
                        phishing_probabilities
                    )

                    st.success("Predictions completed!")

                    st.subheader("Prediction Results")

                    st.dataframe(results)

                    st.download_button(
                        label="Download Prediction Results",
                        data=results.to_csv(index=False),
                        file_name="phishing_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as error:

        st.error(f"An error occurred: {error}")

st.divider()

st.caption(
    "Educational demonstration. Predictions depend on the quality "
    "of the extracted website features and are not a guarantee "
    "of real-world website safety."
)

