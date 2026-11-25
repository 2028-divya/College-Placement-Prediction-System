
import pandas as pd


DEFAULT_COMPANY_CRITERIA = {
    "min_cgpa": 7.0,
    "min_tenth_percentage": 60,
    "min_twelfth_percentage": 60,
    "max_backlogs": 0,
    "min_aptitude_score": 60,
    "min_technical_skills_score": 60,
    "min_communication_skills_score": 60,
    "internship_required": False
}


def validate_company_criteria(criteria):
    """Validate company eligibility requirements."""

    missing_keys = [
        key for key in DEFAULT_COMPANY_CRITERIA
        if key not in criteria
    ]

    if missing_keys:
        raise ValueError(f"Missing company criteria: {missing_keys}")

    if not 0 <= criteria["min_cgpa"] <= 10:
        raise ValueError("Minimum CGPA must be between 0 and 10.")

    percentage_fields = [
        "min_tenth_percentage",
        "min_twelfth_percentage",
        "min_aptitude_score",
        "min_technical_skills_score",
        "min_communication_skills_score"
    ]

    for field in percentage_fields:
        if not 0 <= criteria[field] <= 100:
            raise ValueError(f"{field} must be between 0 and 100.")

    if (
        not isinstance(criteria["max_backlogs"], int)
        or criteria["max_backlogs"] < 0
    ):
        raise ValueError("Maximum backlogs must be a non-negative integer.")

    if not isinstance(criteria["internship_required"], bool):
        raise ValueError("internship_required must be True or False.")

    return True


def check_student_eligibility(student, criteria):
    """Check one student's eligibility and return failed criteria."""

    failed_criteria = []

    checks = {
        "CGPA": (
            student["cgpa"] >= criteria["min_cgpa"],
            f"Minimum CGPA: {criteria['min_cgpa']}"
        ),
        "10th Percentage": (
            student["tenth_percentage"] >= criteria["min_tenth_percentage"],
            f"Minimum 10th percentage: {criteria['min_tenth_percentage']}%"
        ),
        "12th Percentage": (
            student["twelfth_percentage"] >= criteria["min_twelfth_percentage"],
            f"Minimum 12th percentage: {criteria['min_twelfth_percentage']}%"
        ),
        "Backlogs": (
            student["backlogs"] <= criteria["max_backlogs"],
            f"Maximum allowed backlogs: {criteria['max_backlogs']}"
        ),
        "Aptitude Score": (
            student["aptitude_score"] >= criteria["min_aptitude_score"],
            f"Minimum aptitude score: {criteria['min_aptitude_score']}"
        ),
        "Technical Skills": (
            student["technical_skills_score"] >= criteria["min_technical_skills_score"],
            f"Minimum technical score: {criteria['min_technical_skills_score']}"
        ),
        "Communication Skills": (
            student["communication_skills_score"] >= criteria["min_communication_skills_score"],
            f"Minimum communication score: {criteria['min_communication_skills_score']}"
        )
    }

    for criterion, (passed, requirement) in checks.items():
        if not passed:
            failed_criteria.append(f"{criterion} - {requirement}")

    if (
        criteria["internship_required"]
        and student["internship_experience"] != "Yes"
    ):
        failed_criteria.append("Internship Experience - Required")

    return failed_criteria


def evaluate_company_drive(student_data, criteria):
    """Evaluate eligibility and near-eligibility for all students."""

    validate_company_criteria(criteria)

    results = student_data.copy()

    failed_results = results.apply(
        lambda student: check_student_eligibility(student, criteria),
        axis=1
    )

    results["failed_criteria"] = failed_results

    results["eligibility_status"] = results["failed_criteria"].apply(
        lambda failures: "Eligible" if len(failures) == 0 else "Not Eligible"
    )

    near_eligible_details = results.apply(
        lambda student: get_near_eligible_details(student, criteria),
        axis=1
    )

    results["near_eligible"] = near_eligible_details.apply(
        lambda details: details is not None
    )

    results["improvement_area"] = near_eligible_details.apply(
        lambda details: details["improvement_area"] if details else None
    )

    results["improvement_gap"] = near_eligible_details.apply(
        lambda details: details["gap"] if details else None
    )

    return results

def get_company_drive_summary(results):
    """Generate summary statistics for a company placement drive."""

    total_students = len(results)

    eligible_students = int(
        (results["eligibility_status"] == "Eligible").sum()
    )

    not_eligible_students = total_students - eligible_students

    near_eligible_students = int(
        results["near_eligible"].sum()
    )

    eligibility_percentage = (
        round((eligible_students / total_students) * 100, 2)
        if total_students > 0
        else 0
    )

    return {
        "total_students": total_students,
        "eligible_students": eligible_students,
        "not_eligible_students": not_eligible_students,
        "near_eligible_students": near_eligible_students,
        "eligibility_percentage": eligibility_percentage
    }

def is_near_eligible(student, criteria):
    """
    Identify students who fail exactly one numeric criterion
    within its allowed gap, without failing backlog or
    mandatory internship requirements.
    """

    numeric_checks = [
        ("cgpa", "min_cgpa", 0.5),
        ("tenth_percentage", "min_tenth_percentage", 5),
        ("twelfth_percentage", "min_twelfth_percentage", 5),
        ("aptitude_score", "min_aptitude_score", 5),
        ("technical_skills_score", "min_technical_skills_score", 5),
        ("communication_skills_score", "min_communication_skills_score", 5)
    ]

    gaps = []

    for student_field, criteria_field, allowed_gap in numeric_checks:
        difference = criteria[criteria_field] - student[student_field]

        if difference > 0:
            gaps.append((difference, allowed_gap))

    if student["backlogs"] > criteria["max_backlogs"]:
        return False

    if (
        criteria["internship_required"]
        and student["internship_experience"] != "Yes"
    ):
        return False

    return (
        len(gaps) == 1
        and gaps[0][0] <= gaps[0][1]
    )

def get_near_eligible_details(student, criteria):
    """Return improvement details for near-eligible students."""

    if not is_near_eligible(student, criteria):
        return None

    numeric_checks = [
        ("cgpa", "min_cgpa", "CGPA"),
        ("tenth_percentage", "min_tenth_percentage", "10th Percentage"),
        ("twelfth_percentage", "min_twelfth_percentage", "12th Percentage"),
        ("aptitude_score", "min_aptitude_score", "Aptitude Score"),
        ("technical_skills_score", "min_technical_skills_score", "Technical Skills"),
        ("communication_skills_score", "min_communication_skills_score", "Communication Skills")
    ]

    for student_field, criteria_field, label in numeric_checks:
        required_value = criteria[criteria_field]
        actual_value = student[student_field]
        gap = required_value - actual_value

        if gap > 0:
            return {
                "improvement_area": label,
                "current_value": actual_value,
                "required_value": required_value,
                "gap": round(gap, 2)
            }

    return None

if __name__ == "__main__":

    test_student = {
        "student_id": "TEST101",
        "cgpa": 6.8,
        "tenth_percentage": 75,
        "twelfth_percentage": 78,
        "backlogs": 0,
        "aptitude_score": 70,
        "technical_skills_score": 75,
        "communication_skills_score": 72,
        "internship_experience": "Yes"
    }

    criteria = DEFAULT_COMPANY_CRITERIA.copy()

    failures = check_student_eligibility(test_student, criteria)

    print("Failed Criteria:", failures)
    print("Near Eligible:", is_near_eligible(test_student, criteria))
