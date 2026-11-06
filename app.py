
import streamlit as st
import pandas as pd
import sys
import json
import io
from pathlib import Path
from src.training_recommendations import add_training_recommendations
from src.analytics import (
    get_cgpa_distribution,
    get_cgpa_placement_analysis,
    get_academic_performance_categories,
    get_skill_average_scores,
    get_skill_performance_categories,
    get_skill_gap_analysis,
    get_skill_training_priority,
    get_placement_probability_analysis
)

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

REPORT_PATH = BASE_DIR / "reports" / "model_evaluation.json"

with open(REPORT_PATH, "r") as file:
    evaluation = json.load(file)

from src.predict import predict_placements

st.set_page_config(
    page_title="College Placement Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Placement Prediction System")
st.caption("Placement Cell Analytics Dashboard")

st.divider()

# Required model features
REQUIRED_COLUMNS = [
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "attendance_percentage",
    "aptitude_score",
    "technical_skills_score",
    "communication_skills_score",
    "internship_experience"
]

NUMERIC_COLUMNS = [
    "cgpa",
    "tenth_percentage",
    "twelfth_percentage",
    "backlogs",
    "attendance_percentage",
    "aptitude_score",
    "technical_skills_score",
    "communication_skills_score"
]

# Upload section
st.subheader("📁 Upload Student Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

try:
    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            uploaded_df = pd.read_csv(uploaded_file)
        else:
            uploaded_df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")

        # Validate columns
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        # Generate IDs if unavailable
        if "student_id" not in uploaded_df.columns:
            uploaded_df.insert(
                0,
                "student_id",
                [f"UPLOAD_{i+1:04d}" for i in range(len(uploaded_df))]
            )

        # Convert numerical columns
        for col in NUMERIC_COLUMNS:
            uploaded_df[col] = pd.to_numeric(
                uploaded_df[col],
                errors="coerce"
            )

        if uploaded_df[NUMERIC_COLUMNS].isnull().any().any():
            st.error("Some numerical columns contain missing or invalid values.")
            st.stop()

        if uploaded_df["internship_experience"].isnull().any():
            st.error("Internship experience contains missing values.")
            st.stop()

        st.write("Uploaded Dataset Preview")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        # Predict uploaded records
        results = predict_placements(uploaded_df)

    else:
        st.info("No file uploaded. Showing sample dataset predictions.")

        @st.cache_data
        def load_sample_data():
            data_path = BASE_DIR / "data" / "processed" / "cleaned_student_data.csv"
            df = pd.read_csv(data_path)
            return predict_placements(df)

        results = load_sample_data()


    # Generate academic training recommendations
    results = add_training_recommendations(results)

    # Download Prediction Reports

    st.divider()
    st.subheader("📥 Download Prediction Reports")

    report_columns = [
        "student_id",
        "predicted_placement",
        "placement_probability",
        "risk_level",
        "recommended_training"
    ]

    report_data = results[report_columns]

    csv_report = report_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_report,
        file_name="placement_prediction_report.csv",
        mime="text/csv"
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        report_data.to_excel(writer, index=False, sheet_name="Placement Report")

    st.download_button(
        label="⬇️ Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="placement_prediction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Dashboard KPIs
    total_students = len(results)

    predicted_placed = (
        results["predicted_placement"] == "Placed"
    ).sum()

    predicted_not_placed = (
        results["predicted_placement"] == "Not Placed"
    ).sum()

    placement_rate = (
        predicted_placed / total_students * 100
        if total_students > 0 else 0
    )

    st.divider()
    st.subheader("📊 Placement Overview")

    # Model Performance
    st.divider()
    st.subheader("🤖 Model Test Performance")

    st.caption(
        "Performance metrics evaluated on the held-out synthetic test dataset (200 records). "
        "These metrics do not represent accuracy on uploaded student data."
    )

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    metric1.metric(
        "Accuracy",
        f"{evaluation['accuracy'] * 100:.2f}%"
    )

    metric2.metric(
        "Precision",
        f"{evaluation['precision'] * 100:.2f}%"
    )

    metric3.metric(
        "Recall",
        f"{evaluation['recall'] * 100:.2f}%"
    )

    metric4.metric(
        "F1 Score",
        f"{evaluation['f1_score'] * 100:.2f}%"
    )

    metric5.metric(
        "ROC-AUC",
        f"{evaluation['roc_auc'] * 100:.2f}%"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Predicted Placed", predicted_placed)
    col3.metric("Predicted Not Placed", predicted_not_placed)
    col4.metric("Predicted Placement Rate", f"{placement_rate:.2f}%")

    st.divider()
    st.subheader("📋 Management Summary")

    # Students requiring support
    support_mask = (
        (results["risk_level"] != "Low") |
        (results["recommended_training"] != "No Immediate Training Required")
    )

    students_requiring_support = support_mask.sum()

    # High-priority students
    high_priority_students = (
        results["risk_level"] == "High"
    ).sum()

    # Students requiring academic support
    academic_support = results["recommended_training"].str.contains(
        "Academic Improvement|Backlog Support",
        regex=True
    ).sum()

    # Students needing skill improvement
    skill_columns = [
        "aptitude_score",
        "technical_skills_score",
        "communication_skills_score"
    ]

    skill_support = (
        results[skill_columns] < 50
    ).any(axis=1).sum()

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    summary_col1.metric(
        "Students Requiring Support",
        students_requiring_support
    )

    summary_col2.metric(
        "High-Priority Students",
        high_priority_students
    )

    summary_col3.metric(
        "Academic Support Needed",
        academic_support
    )

    summary_col4.metric(
        "Skill Improvement Needed",
        skill_support
    )

    st.divider()

    

    # Confusion Matrix
    st.write("### Confusion Matrix")

    confusion_data = pd.DataFrame(
        evaluation["confusion_matrix"],
        index=["Actual Not Placed", "Actual Placed"],
        columns=["Predicted Not Placed", "Predicted Placed"]
    )

    st.dataframe(
        confusion_data,
        use_container_width=True
    )

    # Classification Report
    st.write("### Class-wise Performance")

    classification_data = pd.DataFrame(
        evaluation["classification_report"]
    ).transpose()

    classification_data = classification_data.drop(
        index=["accuracy"]
    )

    st.dataframe(
        classification_data,
        use_container_width=True
    )

    # Class-wise Performance Chart
    st.write("### Class-wise Metrics Comparison")

    class_metrics = classification_data.loc[
        ["Not Placed", "Placed"],
        ["precision", "recall", "f1-score"]
    ]

    st.bar_chart(class_metrics)

    # Analytics charts
    st.subheader("📈 Placement Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Predicted Placement Distribution")
        st.bar_chart(results["predicted_placement"].value_counts())

    with chart_col2:
        st.write("Student Support Priority")
        st.bar_chart(results["risk_level"].value_counts())

    # Placement Probability Analysis
    st.write("### Placement Probability Distribution")

    probability_analysis = get_placement_probability_analysis(results)

    st.bar_chart(probability_analysis)

    st.divider()

    
    # Student Support Filter
    st.subheader("🎯 Student Support Analysis")

    priority_options = ["All", "High", "Medium", "Low"]

    selected_priority = st.selectbox(
        "Filter students by support priority",
        priority_options
    )

    if selected_priority == "All":
        filtered_results = results.copy()
    else:
        filtered_results = results[
            results["risk_level"] == selected_priority
        ]

    st.write(f"Students found: {len(filtered_results)}")

    display_columns = [
        "student_id",
        "cgpa",
        "predicted_placement",
        "placement_probability",
        "risk_level"
    ]

    st.dataframe(
        filtered_results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Academic Training Recommendations
    st.subheader("📚 Academic Training Recommendations")

    training_counts = (
        results["recommended_training"]
        .str.split(", ")
        .explode()
        .value_counts()
    )

    training_col1, training_col2 = st.columns(2)

    with training_col1:
        st.write("Training-wise Student Count")
        st.bar_chart(training_counts)

    with training_col2:
        selected_training = st.selectbox(
            "Select Training Area",
            ["All"] + sorted(
                results["recommended_training"]
                .str.split(", ")
                .explode()
                .unique()
                .tolist()
            )
        )

        if selected_training == "All":
            training_students = results[
                results["recommended_training"] !=
                "No Immediate Training Required"
            ]
        else:
            training_students = results[
                results["recommended_training"].str.contains(
                    selected_training,
                    regex=False
                )
            ]

        st.write(f"Students requiring this training: {len(training_students)}")

        st.dataframe(
            training_students[
                ["student_id", "cgpa", "recommended_training"]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("📊 Academic Performance Analytics")

    # 1. CGPA Distribution
    st.write("### CGPA-wise Student Distribution")

    cgpa_distribution = get_cgpa_distribution(results)

    st.bar_chart(cgpa_distribution)

    # 2. CGPA vs Predicted Placement
    st.write("### CGPA Range vs Predicted Placement")

    cgpa_placement = get_cgpa_placement_analysis(results)

    st.bar_chart(cgpa_placement)

    # 3. Academic Performance Categories
    st.write("### Academic Performance Categories")

    academic_categories = get_academic_performance_categories(results)

    st.bar_chart(academic_categories)

    st.divider()
    st.subheader("🛠️ Skill Gap Analytics")

    # 1. Average Skill Scores
    st.write("### Average Skill Scores")
    skill_averages = get_skill_average_scores(results)
    st.bar_chart(skill_averages)

    # 2. Skill Performance Categories
    st.write("### Skill Performance Categories")
    skill_categories = get_skill_performance_categories(results)
    st.bar_chart(skill_categories)

    # 3. Skill Gap Analysis
    st.write("### Students Needing Skill Improvement")
    skill_gaps = get_skill_gap_analysis(results)
    st.bar_chart(skill_gaps)

    # 4. Skill Training Priority
    st.write("### Skill Training Priority")
    skill_priority = get_skill_training_priority(results)
    st.dataframe(skill_priority)

except Exception as e:
    st.error(f"Something went wrong: {e}")

