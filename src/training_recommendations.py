
import pandas as pd


def add_training_recommendations(df):
    """
    Add rule-based training recommendations to student records.
    """

    result = df.copy()

    def recommend(row):
        recommendations = []

        # Academic support
        if row["cgpa"] < 6.5:
            recommendations.append("Academic Improvement")

        if row["backlogs"] > 0:
            recommendations.append("Backlog Support")

        # Skill-based training
        if row["aptitude_score"] < 50:
            recommendations.append("Aptitude Training")

        if row["technical_skills_score"] < 50:
            recommendations.append("Technical Skills Training")

        if row["communication_skills_score"] < 50:
            recommendations.append("Communication Skills Training")

        if not recommendations:
            return "No Immediate Training Required"

        return ", ".join(recommendations)

    result["recommended_training"] = result.apply(
        recommend, axis=1
    )

    return result
