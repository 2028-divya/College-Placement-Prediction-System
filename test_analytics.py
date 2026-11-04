import pandas as pd

from src.analytics import (
    get_cgpa_distribution,
    get_cgpa_placement_analysis,
    get_academic_performance_categories
)

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