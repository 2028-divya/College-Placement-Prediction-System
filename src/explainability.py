
import pandas as pd
from pathlib import Path

from src.predict import load_model


FEATURE_LABELS = {
    "cgpa": "CGPA",
    "tenth_percentage": "10th Percentage",
    "twelfth_percentage": "12th Percentage",
    "backlogs": "Backlogs",
    "attendance_percentage": "Attendance",
    "aptitude_score": "Aptitude Score",
    "technical_skills_score": "Technical Skills",
    "communication_skills_score": "Communication Skills",
    "internship_experience_No": "No Internship",
    "internship_experience_Yes": "Internship Experience"
}


def explain_prediction(student_data):

    model = load_model()

    features = student_data.drop(
        columns=["student_id", "placement_status"],
        errors="ignore"
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_features = preprocessor.transform(features)

    feature_names = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    contributions = transformed_features * coefficients

    explanation = pd.DataFrame(
        contributions,
        columns=feature_names,
        index=student_data.index
    )

    # Remove preprocessing prefixes from feature names
    explanation.columns = [
        name.replace("numeric__", "")
            .replace("categorical__", "")
        for name in explanation.columns
    ]

    # Replace technical names with readable labels
    explanation = explanation.rename(
        columns=FEATURE_LABELS
    )

    return explanation


def get_top_contributors(explanation, top_n=3):

    results = []

    for index, row in explanation.iterrows():

        positive = row[row > 0].sort_values(
            ascending=False
        ).head(top_n)

        negative = row[row < 0].sort_values(
            ascending=True
        ).head(top_n)

        results.append({
            "student_index": index,
            "positive_factors": positive.to_dict(),
            "negative_factors": negative.to_dict()
        })

    return results


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parents[1]

    DATA_PATH = (
        BASE_DIR / "data" / "processed"
        / "cleaned_student_data.csv"
    )

    df = pd.read_csv(DATA_PATH)

    sample = df.head(5)

    explanation = explain_prediction(sample)

    print("\nReadable Feature Contributions:\n")
    print(explanation.round(3).to_string())

    print("\nTop Contributors:\n")

    for item in get_top_contributors(explanation):

        print(f"\nStudent Index: {item['student_index']}")

        print("Positive Factors:", item["positive_factors"])
        print("Negative Factors:", item["negative_factors"])
