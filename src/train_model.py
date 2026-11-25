
import pandas as pd
import numpy as np
import pickle
import json

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.calibration import calibration_curve

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    brier_score_loss
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"

REPORT_PATH = BASE_DIR / "reports" / "model_evaluation.json"
COMPARISON_PATH = BASE_DIR / "reports" / "model_comparison.json"


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# Separate features and target
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

categorical_features = ["internship_experience"]


# Preprocessing
def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )


# Define model configurations
classifiers = {
    "Logistic Regression (Unbalanced)": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Logistic Regression (Balanced)": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest (Unbalanced)": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Random Forest (Balanced)": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree (Unbalanced)": DecisionTreeClassifier(
        random_state=42
    ),

    "Decision Tree (Balanced)": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )
}


# Create independent pipelines
models = {}

for name, classifier in classifiers.items():

    models[name] = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier)
        ]
    )


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# Compare all six configurations
comparison_results = []

print("\n----- SIX-CONFIGURATION MODEL COMPARISON -----")

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        (y_train == "Placed").astype(int),
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "model": name,
        "cv_accuracy": float(np.mean(scores["test_accuracy"])),
        "cv_precision": float(np.mean(scores["test_precision"])),
        "cv_recall": float(np.mean(scores["test_recall"])),
        "cv_f1": float(np.mean(scores["test_f1"])),
        "cv_roc_auc": float(np.mean(scores["test_roc_auc"])),
        "cv_f1_std": float(np.std(scores["test_f1"]))
    }

    comparison_results.append(result)

    print(f"\n{name}")
    print(f"Accuracy: {result['cv_accuracy']:.4f}")
    print(f"Precision: {result['cv_precision']:.4f}")
    print(f"Recall: {result['cv_recall']:.4f}")
    print(f"F1: {result['cv_f1']:.4f}")
    print(f"ROC-AUC: {result['cv_roc_auc']:.4f}")
    print(f"F1 Std: {result['cv_f1_std']:.4f}")


# Select configuration using CV F1
best_result = max(
    comparison_results,
    key=lambda result: (
        result["cv_f1"],
        result["cv_roc_auc"]
    )
)

best_model_name = best_result["model"]
best_model = models[best_model_name]

print("\nSelected Configuration:", best_model_name)


# Train selected model
best_model.fit(X_train, y_train)

print("\nSelected model training completed!")


# Test predictions
y_pred = best_model.predict(X_test)

placed_index = list(best_model.classes_).index("Placed")

y_probability = best_model.predict_proba(X_test)[:, placed_index]


# Evaluate selected model
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Placed",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Placed",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Placed",
    zero_division=0
)

roc_auc = roc_auc_score(
    (y_test == "Placed").astype(int),
    y_probability
)

# Probability calibration evaluation
y_test_binary = (y_test == "Placed").astype(int)

brier_score = brier_score_loss(
    y_test_binary,
    y_probability
)

fraction_positive, mean_predicted = calibration_curve(
    y_test_binary,
    y_probability,
    n_bins=5,
    strategy="quantile"
)

calibration_results = {
    "brier_score": float(brier_score),
    "calibration_curve": [
        {
            "mean_predicted_probability": float(predicted),
            "fraction_of_positives": float(actual)
        }
        for predicted, actual in zip(
            mean_predicted,
            fraction_positive
        )
    ]
}

print("\n----- PROBABILITY CALIBRATION -----")
print(f"Brier Score: {brier_score:.4f}")
print("\nCalibration Curve:")

for item in calibration_results["calibration_curve"]:
    print(item)

class_report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

conf_matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=["Not Placed", "Placed"]
)


print("\n----- SELECTED MODEL TEST PERFORMANCE -----")

print("Selected Configuration:", best_model_name)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Save evaluation report
evaluation_results = {
    "selected_model": best_model_name,
    "selection_metric": "Cross-validation F1-score",
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "calibration": calibration_results,
    "confusion_matrix": conf_matrix.tolist(),
    "classification_report": class_report
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(REPORT_PATH, "w") as file:
    json.dump(evaluation_results, file, indent=4)


# Save comparison report
comparison_report = {
    "selection_metric": "Cross-validation F1-score",
    "selected_model": best_model_name,
    "models": comparison_results
}

with open(COMPARISON_PATH, "w") as file:
    json.dump(comparison_report, file, indent=4)


# Save selected model
MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(best_model, file)


print("\nEvaluation report saved:", REPORT_PATH)
print("Comparison report saved:", COMPARISON_PATH)
print("Selected model saved:", MODEL_PATH)
