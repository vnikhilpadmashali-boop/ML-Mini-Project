import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Smart Digital Wellness Analyzer", page_icon="🌱", layout="wide")

DB_FILE = "wellness_history.db"

def save_result(input_values, predicted_label):
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_values TEXT NOT NULL,
                predicted_category TEXT NOT NULL
            )
        """)
        connection.execute(
            "INSERT INTO prediction_history (created_at, input_values, predicted_category) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), str(input_values), str(predicted_label))
        )

def read_history():
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_values TEXT NOT NULL,
                predicted_category TEXT NOT NULL
            )
        """)
        return pd.read_sql_query(
            "SELECT created_at, predicted_category FROM prediction_history ORDER BY id DESC LIMIT 30",
            connection
        )

st.title("🌱 Smart Digital Wellness Analyzer")
st.write("Explore smartphone habits and use a machine-learning model to estimate a usage category.")
st.caption("Educational mini project — predictions depend on the dataset and are not a medical assessment.")

sample_path = Path("data/smartphone_usage_sample.csv")
uploaded = st.sidebar.file_uploader("Upload smartphone usage CSV", type=["csv"])
try:
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif sample_path.exists():
        data = pd.read_csv(sample_path)
        st.sidebar.success("Using the included sample dataset.")
    else:
        st.info("Upload a CSV file to start.")
        st.stop()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

target = "Usage_Category"
if target not in data.columns:
    st.error("Dataset must contain a column named Usage_Category.")
    st.write("Available columns:", list(data.columns))
    st.stop()

data = data.dropna(subset=[target]).copy()
data[target] = data[target].astype(str).str.strip()
data = data[data[target] != ""]
if len(data) < 10:
    st.error("Please provide at least 10 labelled records.")
    st.stop()

# Dashboard overview
st.subheader("Dataset Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Records", len(data))
m2.metric("Input features", len(data.columns) - 1)
m3.metric("Usage categories", data[target].nunique())

left, right = st.columns(2)
with left:
    st.markdown("**Usage category distribution**")
    fig, ax = plt.subplots()
    data[target].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of records")
    st.pyplot(fig)
    plt.close(fig)
with right:
    if "Screen_Time" in data.columns:
        st.markdown("**Screen time distribution**")
        fig, ax = plt.subplots()
        pd.to_numeric(data["Screen_Time"], errors="coerce").dropna().plot(kind="hist", bins=10, ax=ax)
        ax.set_xlabel("Screen time (hours)")
        ax.set_ylabel("Number of records")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.markdown("**Dataset preview**")
        st.dataframe(data.head(), use_container_width=True)

st.divider()
st.subheader("Train Machine Learning Model")

features = data.drop(columns=[target])
id_columns = [col for col in features.columns if col.lower() in {"user_id", "id", "index"}]
features = features.drop(columns=id_columns, errors="ignore")
usable_columns = [col for col in features.columns if features[col].notna().any() and features[col].nunique(dropna=True) > 1]
features = features[usable_columns]

if features.empty:
    st.error("No usable feature columns found.")
    st.stop()

numeric_columns = features.select_dtypes(include="number").columns.tolist()
categorical_columns = [col for col in features.columns if col not in numeric_columns]
transformers = []
if numeric_columns:
    transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns))
if categorical_columns:
    transformers.append(("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_columns))

pipeline = Pipeline([
    ("preprocessing", ColumnTransformer(transformers=transformers)),
    ("model", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"))
])

labels = data[target]
stratify_labels = labels if labels.value_counts().min() >= 2 else None
try:
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    pipeline.fit(x_train, y_train)
    test_predictions = pipeline.predict(x_test)
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

accuracy = accuracy_score(y_test, test_predictions)
st.metric("Test accuracy", f"{accuracy:.1%}")
st.caption("This is the score on a held-out test split; it is not a guarantee of future prediction accuracy.")

with st.expander("View model evaluation"):
    st.text(classification_report(y_test, test_predictions, zero_division=0))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.subheader("Try a New Prediction")
st.write("Enter sample values below to estimate the usage category.")
user_values = {}
for column in features.columns:
    if column in numeric_columns:
        numbers = pd.to_numeric(features[column], errors="coerce").dropna()
        default_value = float(numbers.median()) if not numbers.empty else 0.0
        user_values[column] = st.number_input(column.replace("_", " "), value=default_value)
    else:
        choices = sorted(features[column].dropna().astype(str).unique().tolist())
        user_values[column] = st.selectbox(column.replace("_", " "), choices)

if st.button("Analyze My Usage", type="primary"):
    one_row = pd.DataFrame([user_values], columns=features.columns)
    prediction = pipeline.predict(one_row)[0]
    save_result(user_values, prediction)
    st.success(f"Estimated usage category: {prediction}")
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(one_row)[0]
        probability_table = pd.DataFrame({
            "Category": pipeline.named_steps["model"].classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.dataframe(probability_table, use_container_width=True)

st.divider()
st.subheader("Prediction History")
history = read_history()
if history.empty:
    st.info("No predictions saved yet. Run a prediction to create history.")
else:
    st.dataframe(history, use_container_width=True)

with st.expander("View dataset"):
    st.dataframe(data, use_container_width=True)
    st.download_button("Download dataset CSV", data.to_csv(index=False), "smartphone_usage_data.csv", "text/csv")
