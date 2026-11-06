
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
        labels=["Needs Improvement", "Satisfactory", "Good"],
        include_lowest=True
    )

    return result["academic_category"].value_counts().sort_index()

def get_skill_average_scores(df):
    """
    Calculates average scores for different skill areas.
    """

    skill_columns = [
        "aptitude_score",
        "technical_skills_score",
        "communication_skills_score"
    ]

    average_scores = df[skill_columns].mean()

    return average_scores

def get_skill_performance_categories(df):
    """
    Categorizes student performance in different skills.
    """

    skill_columns = [
        "aptitude_score",
        "technical_skills_score",
        "communication_skills_score"
    ]

    results = {}

    for skill in skill_columns:

        results[skill] = pd.cut(
            df[skill],
            bins=[-1, 50, 70, 101],
            labels=["Needs Improvement", "Average", "Good"],
            right=False
        ).value_counts().sort_index()

    return pd.DataFrame(results)

def get_skill_gap_analysis(df):
    """
    Counts students needing improvement in each skill.
    """

    skill_columns = [
        "aptitude_score",
        "technical_skills_score",
        "communication_skills_score"
    ]

    skill_gaps = {}

    for skill in skill_columns:
        skill_gaps[skill] = (df[skill] < 50).sum()

    return pd.Series(skill_gaps)

def get_skill_training_priority(df):
    """
    Ranks skill areas based on the number of students
    needing improvement.
    """

    gap_counts = get_skill_gap_analysis(df)

    priority = pd.DataFrame({
        "students_needing_improvement": gap_counts
    })

    priority["priority_rank"] = (
        priority["students_needing_improvement"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    priority = priority.sort_values(
        by="students_needing_improvement",
        ascending=False
    )

    return priority

def get_placement_probability_analysis(df):
    """
    Groups students according to their predicted
    placement probability.
    """

    result = df.copy()

    result["probability_range"] = pd.cut(
        result["placement_probability"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=[
            "0-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%"
        ],
        include_lowest=True
    )

    return result["probability_range"].value_counts().sort_index()