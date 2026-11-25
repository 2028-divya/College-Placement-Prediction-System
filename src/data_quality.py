
import pandas as pd


NUMERIC_RANGES = {
    "cgpa": (0, 10),
    "tenth_percentage": (0, 100),
    "twelfth_percentage": (0, 100),
    "backlogs": (0, 20),
    "attendance_percentage": (0, 100),
    "aptitude_score": (0, 100),
    "technical_skills_score": (0, 100),
    "communication_skills_score": (0, 100)
}

REQUIRED_FEATURES = list(NUMERIC_RANGES.keys()) + [
    "internship_experience"
]


def analyze_data_quality(df):
    """
    Analyze uploaded student data without modifying
    or deleting any records.
    """

    result = df.copy()

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_FEATURES
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Generate IDs if unavailable
    if "student_id" not in result.columns:
        result.insert(
            0,
            "student_id",
            [f"UPLOAD_{i + 1:04d}" for i in range(len(result))]
        )

    issues = pd.DataFrame(index=result.index)

    # Check missing values
    for column in REQUIRED_FEATURES:
        issues[f"{column}_missing"] = (
            result[column].isna()
        )

    # Check numerical values
    for column, (minimum, maximum) in NUMERIC_RANGES.items():

        numeric_values = pd.to_numeric(
            result[column],
            errors="coerce"
        )

        issues[f"{column}_invalid"] = (
            result[column].notna()
            & numeric_values.isna()
        )

        issues[f"{column}_out_of_range"] = (
            numeric_values.notna()
            & ~numeric_values.between(minimum, maximum)
        )

        if column == "backlogs":
            issues["backlogs_not_integer"] = (
                numeric_values.notna()
                & (numeric_values % 1 != 0)
            )

    # Check internship categories
    internship_values = (
        result["internship_experience"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    issues["internship_invalid"] = (
        ~internship_values.isin(["Yes", "No"])
        & internship_values.notna()
    )

    # Check missing and duplicate student IDs
    student_ids = result["student_id"].astype("string").str.strip()

    issues["missing_student_id"] = (
        student_ids.isna() | student_ids.eq("")
    )

    issues["duplicate_student_id"] = (
        student_ids.duplicated(keep=False)
        & ~issues["missing_student_id"]
    )

    # A row is invalid if any issue is present
    issues["has_issues"] = issues.any(axis=1)

    # Summary
    summary = {
        "total_records": len(result),
        "valid_records": int((~issues["has_issues"]).sum()),
        "invalid_records": int(issues["has_issues"].sum()),
        "duplicate_ids": int(
            issues["duplicate_student_id"].sum()
        ),
        "missing_values": int(
            issues.filter(like="_missing").sum().sum()
        ),
        "invalid_numeric_values": int(
            sum(
                issues[f"{column}_invalid"].sum()
                for column in NUMERIC_RANGES
            )
        ),
        "out_of_range_values": int(
            issues.filter(like="_out_of_range").sum().sum()
        ),
        "invalid_internship_values": int(
            issues["internship_invalid"].sum()
        ),
        "invalid_backlog_counts": int(
            issues["backlogs_not_integer"].sum()
        )
    }

    # Record-level report
    record_report = result[["student_id"]].copy()
    record_report["has_issues"] = issues["has_issues"]

    record_report["issue_count"] = (
    issues.drop(columns="has_issues").sum(axis=1)
    )

    issue_columns = [
        column for column in issues.columns
        if column != "has_issues"
    ]

    record_report["issue_details"] = issues[issue_columns].apply(
        lambda row: ", ".join(
            column.replace("_", " ")
            for column, has_issue in row.items()
            if has_issue
        ),
        axis=1
    )

    return summary, record_report
