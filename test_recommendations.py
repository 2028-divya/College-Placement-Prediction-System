
import pandas as pd
from src.training_recommendations import add_training_recommendations

# Sample student data
data = {
    "student_id": [101, 102, 103],
    "cgpa": [5.8, 7.5, 8.2],
    "backlogs": [2, 0, 0],
    "aptitude_score": [40, 75, 80],
    "technical_skills_score": [65, 45, 85],
    "communication_skills_score": [70, 40, 80]
}

df = pd.DataFrame(data)

# Generate recommendations
result = add_training_recommendations(df)

print(result[["student_id", "recommended_training"]])
