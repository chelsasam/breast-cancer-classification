
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning classifiers using diagnostic "
    "features from the Breast Cancer Wisconsin (Diagnostic) dataset."
)

st.warning(
    "Educational project only. This application is not intended "
    "for medical diagnosis or clinical decision-making."
)


# --------------------------------------------------
# 2. DEFINE FILE LOCATIONS AND MODEL NAMES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "SVM": "svm.joblib"
}


# --------------------------------------------------
# 3. LOAD FEATURES AND TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_feature_columns():
    feature_file = MODEL_DIR / "feature_columns.json"

    with open(feature_file, "r") as file:
        columns = json.load(file)

    return columns


@st.cache_resource
def load_models():
    loaded = {}

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded[model_name] = joblib.load(model_path)

    return loaded


try:
    feature_columns = load_feature_columns()
    models = load_models()

except Exception as error:
    st.error(
        "The saved models could not be loaded. "
        "Check that the model folder and its files are present."
    )
    st.exception(error)
    st.stop()


st.success("All five trained models loaded successfully.")


# --------------------------------------------------
# 4. SELECT A MODEL
# --------------------------------------------------

selected_model_name = st.selectbox(
    "Choose a classification model",
    list(MODEL_FILES.keys())
)

selected_model = models[selected_model_name]

st.write(f"**Selected model:** {selected_model_name}")


# --------------------------------------------------
# 5. UPLOAD A CSV FILE
# --------------------------------------------------

st.subheader("Upload test data")

st.write(
    "Upload a CSV containing the 30 diagnostic feature columns. "
    "If it also contains a diagnosis column, the app will calculate "
    "evaluation metrics. Without diagnosis labels, it will show "
    "predictions only."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload test_data.csv to try the app with labelled test data."
    )
    st.stop()


# --------------------------------------------------
# 6. READ AND VALIDATE THE CSV
# --------------------------------------------------

try:
    input_df = pd.read_csv(uploaded_file)

except Exception as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()


# Remove accidental spaces from column names
input_df.columns = input_df.columns.astype(str).str.strip()

# Find the diagnosis column without depending on capitalization
diagnosis_column = next(
    (
        column for column in input_df.columns
        if column.strip().lower() == "diagnosis"
    ),
    None
)

has_labels = diagnosis_column is not None

st.write("**Uploaded data shape:**", input_df.shape)
st.write("**Preview of uploaded data:**")
st.dataframe(input_df.head())


# --------------------------------------------------
# 7. CHECK REQUIRED FEATURES
# --------------------------------------------------

missing_columns = [
    column for column in feature_columns
    if column not in input_df.columns
]

if missing_columns:
    st.error(
        "The uploaded CSV is missing required feature columns: "
        + ", ".join(missing_columns)
    )
    st.write(
        "Please upload the project's test_data.csv or a CSV "
        "containing all required feature columns."
    )
    st.stop()


# Select the features in the exact order used during training.
# Extra columns such as ID are ignored.
try:
    X_input = input_df[feature_columns].copy()

    for column in feature_columns:
        X_input[column] = pd.to_numeric(
            X_input[column],
            errors="raise"
        )

except Exception as error:
    st.error(
        "Feature columns must contain numeric values. "
        f"Details: {error}"
    )
    st.stop()


if X_input.empty:
    st.error("The uploaded CSV contains no data rows.")
    st.stop()

if X_input.isnull().any().any():
    st.error(
        "The uploaded feature data contains missing values. "
        "Please correct the CSV and upload it again."
    )
    st.stop()

if not np.isfinite(X_input.to_numpy(dtype=float)).all():
    st.error(
        "The uploaded features contain infinite or invalid numeric values."
    )
    st.stop()


# --------------------------------------------------
# 8. PREDICT DIAGNOSIS AND PROBABILITIES
# --------------------------------------------------

try:
    y_pred = selected_model.predict(X_input)

    if hasattr(selected_model, "predict_proba"):
        y_prob = selected_model.predict_proba(X_input)[:, 1]
    else:
        y_prob = None

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()


# Create a readable prediction table
prediction_df = pd.DataFrame({
    "Predicted Diagnosis": np.where(
        y_pred == 1,
        "Malignant",
        "Benign"
    )
})

if y_prob is not None:
    prediction_df["Malignant Probability"] = np.round(
        y_prob, 4
    )

st.subheader("Predictions")
st.dataframe(prediction_df, use_container_width=True)

st.write("**Prediction counts:**")
st.dataframe(
    prediction_df["Predicted Diagnosis"]
    .value_counts()
    .rename_axis("Diagnosis")
    .reset_index(name="Number of records"),
    use_container_width=True
)


# --------------------------------------------------
# 9. CALCULATE METRICS IF LABELS ARE PROVIDED
# --------------------------------------------------

if has_labels:

    st.subheader("Model Evaluation")

    raw_labels = (
        input_df[diagnosis_column]
        .astype(str)
        .str.strip()
    )

    # Support the labels B/M, Benign/Malignant, and 0/1.
    label_map = {
        "B": 0,
        "BENIGN": 0,
        "M": 1,
        "MALIGNANT": 1,
        "0": 0,
        "1": 1,
        "0.0": 0,
        "1.0": 1
    }

    y_true = raw_labels.str.upper().map(label_map)

    if y_true.isnull().any():
        st.error(
            "The diagnosis column contains unsupported labels. "
            "Use B/M, Benign/Malignant, or 0/1."
        )
        st.stop()

    y_true = y_true.astype(int)

    # Metric calculations
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC needs both classes and probability scores.
    if len(np.unique(y_true)) == 2 and y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = None

    # Show all six required metrics
    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "Accuracy", f"{accuracy:.4f}"
    )

    metric_columns[1].metric(
        "AUC",
        f"{auc:.4f}" if auc is not None else "N/A"
    )

    metric_columns[2].metric(
        "Precision", f"{precision:.4f}"
    )

    metric_columns[0].metric(
        "Recall", f"{recall:.4f}"
    )

    metric_columns[1].metric(
        "F1-score", f"{f1:.4f}"
    )

    metric_columns[2].metric(
        "MCC", f"{mcc:.4f}"
    )

    if auc is None:
        st.info(
            "AUC is unavailable because the uploaded labels do not "
            "contain both classes or probability scores are unavailable."
        )

    # ----------------------------------------------
    # 10. CONFUSION MATRIX
    # ----------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"]
    )

    st.dataframe(cm_df, use_container_width=True)

    # ----------------------------------------------
    # 11. CLASSIFICATION REPORT
    # ----------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

else:
    st.info(
        "No diagnosis column was found. Metrics, confusion matrix, "
        "and classification report are unavailable. Predictions "
        "are displayed above."
    )


# --------------------------------------------------
# 12. DOWNLOAD PREDICTIONS
# --------------------------------------------------

st.subheader("Download results")

output_df = input_df.copy()
output_df["Predicted Diagnosis"] = np.where(
    y_pred == 1,
    "Malignant",
    "Benign"
)

if y_prob is not None:
    output_df["Malignant Probability"] = np.round(
        y_prob, 4
    )

st.download_button(
    label="Download predictions as CSV",
    data=output_df.to_csv(index=False).encode("utf-8"),
    file_name="breast_cancer_predictions.csv",
    mime="text/csv"
)

st.caption(
    "Prediction results are generated by the selected saved model. "
    "This educational app is not a medical diagnostic tool."
)
