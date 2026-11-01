
import pandas as pd
import numpy as np
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# Separate input features and target
X = df.drop(columns=["student_id", "placement_status"])
y = df["placement_status"]


# Identify column types
numeric_features = [
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "attendance_percentage",
    "aptitude_score",
    "technical_skills_score",
    "communication_skills_score"
]

categorical_features = [
    "internship_experience"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# Complete ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Train model
model.fit(X_train, y_train)

print("\nModel training completed!")


# Predictions
y_pred = model.predict(X_test)

# Probability of Placed class
placed_index = list(model.classes_).index("Placed")
y_probability = model.predict_proba(X_test)[:, placed_index]


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, pos_label="Placed"
)

recall = recall_score(
    y_test, y_pred, pos_label="Placed"
)

f1 = f1_score(
    y_test, y_pred, pos_label="Placed"
)

roc_auc = roc_auc_score(
    (y_test == "Placed").astype(int),
    y_probability
)


print("\n----- MODEL EVALUATION -----")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["Not Placed", "Placed"]
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Save trained model
MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)


print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)
