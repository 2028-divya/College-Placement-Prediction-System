
import pandas as pd


def get_cgpa_distribution(df):
    """
    Groups students into CGPA ranges.
    """

    bins = [0, 5, 6, 7, 8, 9, 10]
    labels = [
        "Below 5",
        "5-6",
        "6-7",
        "7-8",
        "8-9",
        "9-10"
    ]

    result = df.copy()

    result["cgpa_range"] = pd.cut(
        result["cgpa"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return result["cgpa_range"].value_counts().sort_index()


def get_cgpa_placement_analysis(df):
    """
    Compares predicted placement outcomes across CGPA ranges.
    """

    bins = [0, 5, 6, 7, 8, 9, 10]
    labels = [
        "Below 5",
        "5-6",
        "6-7",
        "7-8",
        "8-9",
        "9-10"
    ]

    result = df.copy()

    result["cgpa_range"] = pd.cut(
        result["cgpa"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    analysis = pd.crosstab(
        result["cgpa_range"],
        result["predicted_placement"]
    )

    return analysis


def get_academic_performance_categories(df):
    """
    Categorizes students based on CGPA.
    """

    result = df.copy()

    result["academic_category"] = pd.cut(
        result["cgpa"],
        bins=[0, 6, 7.5, 10],
        labels=[
            "Needs Improvement",
            "Satisfactory",
            "Good"
        ],
        include_lowest=True
    )

    return result["academic_category"].value_counts().sort_index()
