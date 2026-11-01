
import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "student_id",
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "attendance_percentage",
    "aptitude_score",
    "technical_skills_score",
    "communication_skills_score",
    "internship_experience",
    "placement_status"
]


def load_dataset(file_path):

    df = pd.read_csv(file_path)

    return df


def validate_dataset(df):

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return True


def clean_dataset(df):

    df = df.copy()

    # Remove duplicate records
    df = df.drop_duplicates(
        subset=["student_id"]
    )

    # Convert numeric columns
    numeric_columns = [
        "cgpa",
        "tenth_percentage",
        "twelfth_percentage",
        "backlogs",
        "attendance_percentage",
        "aptitude_score",
        "technical_skills_score",
        "communication_skills_score"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Standardize categorical values
    df["internship_experience"] = (
        df["internship_experience"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df["placement_status"] = (
        df["placement_status"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    # Remove rows with missing required values
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Validate category values
    df = df[
        df["internship_experience"].isin(["Yes", "No"])
    ]

    df = df[
        df["placement_status"].isin(
            ["Placed", "Not Placed"]
        )
    ]

    # Validate numerical ranges
    valid_ranges = {
        "cgpa": (0, 10),
        "tenth_percentage": (0, 100),
        "twelfth_percentage": (0, 100),
        "backlogs": (0, 20),
        "attendance_percentage": (0, 100),
        "aptitude_score": (0, 100),
        "technical_skills_score": (0, 100),
        "communication_skills_score": (0, 100)
    }

    for col, (minimum, maximum) in valid_ranges.items():
        df = df[
            df[col].between(minimum, maximum)
        ]

    return df.reset_index(drop=True)


if __name__ == "__main__":

    file_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "raw"
        / "student_placement_data.csv"
    )

    data = load_dataset(file_path)

    print("Original dataset:", data.shape)

    validate_dataset(data)

    cleaned_data = clean_dataset(data)

    print("Cleaned dataset:", cleaned_data.shape)

    print("\nMissing values:")
    print(cleaned_data.isnull().sum())

    print("\nPlacement distribution:")
    print(cleaned_data["placement_status"].value_counts())

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed"
        / "cleaned_student_data.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned_data.to_csv(
        output_path,
        index=False
    )

    print("\nCleaned dataset saved successfully!")
