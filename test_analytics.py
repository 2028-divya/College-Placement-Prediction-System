import pandas as pd

from src.analytics import (
    get_cgpa_distribution,
    get_cgpa_placement_analysis,
    get_academic_performance_categories
)
from src.analytics import get_placement_probability_analysis

# Sample student data
data = {
    "student_id": [101, 102, 103, 104, 105, 106],
    "cgpa": [5.5, 6.5, 7.2, 8.1, 8.8, 9.5],
    "predicted_placement": [
        "Not Placed",
        "Not Placed",
        "Placed",
        "Placed",
        "Placed",
        "Placed"
    ]
}

df = pd.DataFrame(data)

# Test 1: CGPA Distribution
print("\n--- CGPA Distribution ---")
print(get_cgpa_distribution(df))

# Test 2: CGPA Placement Analysis
print("\n--- CGPA Placement Analysis ---")
print(get_cgpa_placement_analysis(df))

# Test 3: Academic Performance Categories
print("\n--- Academic Performance Categories ---")
print(get_academic_performance_categories(df))

from src.analytics import get_skill_average_scores

# Sample student data
data = {
    "aptitude_score": [60, 70, 80],
    "technical_skills_score": [40, 50, 60],
    "communication_skills_score": [70, 80, 90]
}

df = pd.DataFrame(data)

# Calculate average scores
result = get_skill_average_scores(df)

print("\nSkill Average Scores:")
print(result)

from src.analytics import get_skill_performance_categories

# Skill performance categories
result = get_skill_performance_categories(df)

print("\n--- Skill Performance Categories ---")
print(result)

from src.analytics import get_skill_gap_analysis

# Skill gap analysis
result = get_skill_gap_analysis(df)

print("\n--- Skill Gap Analysis ---")
print(result)

from src.analytics import get_skill_training_priority

result = get_skill_training_priority(df)

print("\n--- Skill Training Priority ---")
print(result)

# Test Placement Probability Analysis

sample_data = pd.DataFrame({
    "placement_probability": [
        10, 25, 35, 45, 55, 65, 75, 85, 95
    ]
})

print("\nPlacement Probability Analysis:")
print(get_placement_probability_analysis(sample_data))