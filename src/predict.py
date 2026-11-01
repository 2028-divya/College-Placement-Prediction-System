
import pandas as pd
import pickle

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"


def load_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


def predict_placements(student_data):

    model = load_model()

    # Remove student ID before prediction
    features = student_data.drop(
        columns=["student_id", "placement_status"],
        errors="ignore"
    )

    predictions = model.predict(features)

    probabilities = model.predict_proba(features)

    placed_index = list(model.classes_).index("Placed")

    placed_probability = probabilities[:, placed_index]

    results = student_data.copy()

    results["predicted_placement"] = predictions

    results["placement_probability"] = (
        placed_probability * 100
    ).round(2)

    results["risk_level"] = results[
        "placement_probability"
    ].apply(
        lambda probability:
            "Low" if probability >= 70
            else "Medium" if probability >= 40
            else "High"
    )

    return results


if __name__ == "__main__":

    DATA_PATH = (
        BASE_DIR / "data" / "processed"
        / "cleaned_student_data.csv"
    )

    df = pd.read_csv(DATA_PATH)

    # Test predictions on first 10 students
    sample = df.head(10)

    results = predict_placements(sample)

    print("\nStudent Placement Predictions:\n")

    print(
        results[
            [
                "student_id",
                "predicted_placement",
                "placement_probability",
                "risk_level"
            ]
        ].to_string(index=False)
    )
