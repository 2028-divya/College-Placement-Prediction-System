
import pandas as pd
import numpy as np
from pathlib import Path

def generate_dataset(n_students=1000, random_seed=42):

    np.random.seed(random_seed)

    cgpa = np.round(np.random.normal(7.2, 1.0, n_students), 2)
    cgpa = np.clip(cgpa, 4, 10)

    tenth = np.round(np.random.normal(76, 10, n_students), 2)
    tenth = np.clip(tenth, 40, 100)

    twelfth = np.round(np.random.normal(73, 11, n_students), 2)
    twelfth = np.clip(twelfth, 40, 100)

    backlogs = np.random.poisson(0.5, n_students)
    backlogs = np.clip(backlogs, 0, 5)

    attendance = np.round(
        np.random.normal(82, 9, n_students), 2
    )
    attendance = np.clip(attendance, 50, 100)

    aptitude = np.round(
        np.random.normal(65, 15, n_students), 2
    )
    aptitude = np.clip(aptitude, 15, 100)

    technical = np.round(
        np.random.normal(66, 14, n_students), 2
    )
    technical = np.clip(technical, 15, 100)

    communication = np.round(
        np.random.normal(67, 13, n_students), 2
    )
    communication = np.clip(communication, 15, 100)

    internship = np.random.choice(
        ["Yes", "No"],
        n_students,
        p=[0.4, 0.6]
    )

    score = (
        0.62 * (cgpa - 7)
        + 0.018 * (aptitude - 60)
        + 0.018 * (technical - 60)
        + 0.010 * (communication - 60)
        + 0.012 * (attendance - 80)
        + 0.48 * (internship == "Yes")
        - 0.62 * backlogs
        + np.random.normal(0, 0.85, n_students)
    )

    probability = 1 / (1 + np.exp(-score))

    placement = np.random.binomial(
        1, probability, n_students
    )

    df = pd.DataFrame({
        "student_id": [
            f"STU{i:04d}" for i in range(1, n_students + 1)
        ],
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "attendance_percentage": attendance,
        "aptitude_score": aptitude,
        "technical_skills_score": technical,
        "communication_skills_score": communication,
        "internship_experience": internship,
        "placement_status": np.where(
            placement == 1, "Placed", "Not Placed"
        )
    })

    return df


if __name__ == "__main__":

    df = generate_dataset(1000)

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "raw" / "student_placement_data.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Dataset generated successfully!")
    print("Total students:", len(df))
    print("Dataset shape:", df.shape)

    print("\nPlacement distribution:")
    print(df["placement_status"].value_counts())

    print("\nFirst five records:")
    print(df.head())
