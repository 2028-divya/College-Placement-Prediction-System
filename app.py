
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

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

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Predicted Placed", predicted_placed)
    col3.metric("Predicted Not Placed", predicted_not_placed)
    col4.metric("Predicted Placement Rate", f"{placement_rate:.2f}%")

    st.divider()

    # Analytics charts
    st.subheader("📈 Placement Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Predicted Placement Distribution")
        st.bar_chart(results["predicted_placement"].value_counts())

    with chart_col2:
        st.write("Student Support Priority")
        st.bar_chart(results["risk_level"].value_counts())

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

except Exception as e:
    st.error(f"Something went wrong: {e}")

