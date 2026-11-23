
import streamlit as st
import pandas as pd
import sys
import json
import io
import altair as alt
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
from src.explainability import (
    explain_prediction,
    get_top_contributors
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

st.markdown("""
<style>
[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    padding: 18px;
    border-radius: 12px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.18);
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 College Placement Prediction System")
st.caption("Placement Cell Analytics Dashboard")

st.divider()

# Dashboard Navigation
st.sidebar.title("🎓 Placement Dashboard")

selected_page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "Placement Intelligence",
        "What-If Simulator",
        "Student Support",
        "Training Recommendations",
        "Academic Analytics",
        "Skill Gap Analytics",
        "Model Performance"
    ]
)
st.sidebar.divider()

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
            uploaded_df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

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


    # ==========================================
    # GLOBAL INTERACTIVE FILTERS
    # ==========================================

    st.sidebar.title("🎛️ Dashboard Filters")
    st.sidebar.caption("Customize your student analysis")

    # 1. CGPA Range Filter
    min_cgpa = float(results["cgpa"].min())
    max_cgpa = float(results["cgpa"].max())

    selected_cgpa = st.sidebar.slider(
        "CGPA Range",
        min_value=min_cgpa,
        max_value=max_cgpa,
        value=(min_cgpa, max_cgpa),
        step=0.1
    )

    # 2. Placement Status Filter
    placement_options = ["Placed", "Not Placed"]

    selected_placement = st.sidebar.multiselect(
        "Predicted Placement",
        placement_options,
        default=placement_options
    )

    # 3. Support Priority Filter
    priority_options = ["High", "Medium", "Low"]

    selected_priority = st.sidebar.multiselect(
        "Support Priority",
        priority_options,
        default=priority_options
    )

    # 4. Training Area Filter
    training_options = sorted(
        results["recommended_training"]
        .str.split(", ")
        .explode()
        .unique()
        .tolist()
    )

    selected_global_training = st.sidebar.selectbox(
        "Training Area",
        ["All"] + training_options
    )

    # Apply Global Filters
    filtered_results = results[
        results["cgpa"].between(
            selected_cgpa[0],
            selected_cgpa[1]
        ) &
        results["predicted_placement"].isin(selected_placement) &
        results["risk_level"].isin(selected_priority)
    ].copy()

    if selected_global_training != "All":
        filtered_results = filtered_results[
            filtered_results["recommended_training"].str.contains(
                selected_global_training,
                regex=False
            )
        ]

    # Use filtered data throughout the dashboard
    results = filtered_results

    st.sidebar.divider()

    st.sidebar.metric(
        "Students After Filtering",
        len(results)
    )

    if results.empty:
        st.warning(
            "No students match the selected filters. "
            "Please adjust your filters."
        )
        st.stop()


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
    if selected_page == "Overview":
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



    if selected_page == "Placement Intelligence":
        # ==========================================
        # PLACEMENT INTELLIGENCE
        # ==========================================

        st.divider()

        st.subheader("🧠 Placement Intelligence")

        st.caption(
            "Understand the factors influencing an individual "
            "student's placement prediction."
        )

        # Select a student from the filtered dataset
        student_options = (
            results["student_id"].astype(str).drop_duplicates().tolist()
        )

        selected_student_id = st.selectbox(
            "Select Student",
            student_options,
            key="intelligence_student"
        )

        selected_student = results[
            results["student_id"].astype(str) == selected_student_id
        ].head(1)

        # Display prediction details
        student_col1, student_col2, student_col3 = st.columns(3)

        student_col1.metric(
            "Predicted Placement",
            selected_student["predicted_placement"].iloc[0]
        )

        student_col2.metric(
            "Placement Probability",
            f"{selected_student['placement_probability'].iloc[0]:.2f}%"
        )

        student_col3.metric(
            "Support Priority",
            selected_student["risk_level"].iloc[0]
        )

        # Calculate feature contributions
        explanation = explain_prediction(selected_student)

        top_factors = get_top_contributors(
            explanation,
            top_n=3
        )[0]

        positive_col, negative_col = st.columns(2)

        with positive_col:
            st.write("#### ✅ Factors Supporting Placement")

            positive_factors = pd.DataFrame(
                list(top_factors["positive_factors"].items()),
                columns=["Feature", "Contribution"]
            )

            if not positive_factors.empty:
                st.dataframe(
                    positive_factors,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No positive contributing factors found.")

        with negative_col:
            st.write("#### ⚠️ Factors Pulling Placement Prediction Down")

            negative_factors = pd.DataFrame(
                list(top_factors["negative_factors"].items()),
                columns=["Feature", "Contribution"]
            )

            if not negative_factors.empty:
                st.dataframe(
                    negative_factors,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No negative contributing factors found.")

        st.caption(
            "Contributions explain the model's log-odds, not direct "
            "percentage-point changes in placement probability. "
            "They show model associations, not proven causes."
        )


    # ==========================================
    # WHAT-IF PLACEMENT SIMULATOR
    # ==========================================
    if selected_page == "What-If Simulator":
        st.divider()
        st.write("### 🔬 What-If Placement Simulator")

        st.caption(
            "Adjust hypothetical student scores to compare "
            "the model's original and simulated predictions."
        )
        # Select a student for simulation
        sim_student_options = (
            results["student_id"].astype(str).drop_duplicates().tolist()
        )

        sim_selected_id = st.selectbox(
            "Select Student for Simulation",
            sim_student_options,
            key="sim_student_select"
        )

        selected_student = results[
            results["student_id"].astype(str) == sim_selected_id
        ].copy()

        # Current student values
        current = selected_student.iloc[0]

        cgpa_value = st.slider(
            "Hypothetical CGPA",
            min_value=0.0,
            max_value=10.0,
            value=float(current["cgpa"]),
            step=0.1,
            key="sim_cgpa"
        )

        aptitude_value = st.slider(
            "Hypothetical Aptitude Score",
            0,
            100,
            int(current["aptitude_score"]),
            key="sim_aptitude"
        )

        technical_value = st.slider(
            "Hypothetical Technical Skills",
            0,
            100,
            int(current["technical_skills_score"]),
            key="sim_technical"
        )

        communication_value = st.slider(
            "Hypothetical Communication Skills",
            0,
            100,
            int(current["communication_skills_score"]),
            key="sim_communication"
        )

        # Create modified student input
        simulated_student = selected_student[
            ["student_id"] + REQUIRED_COLUMNS
        ].copy()

        simulated_student["cgpa"] = cgpa_value
        simulated_student["aptitude_score"] = aptitude_value
        simulated_student["technical_skills_score"] = technical_value
        simulated_student["communication_skills_score"] = communication_value

        # Generate hypothetical prediction
        simulated_result = predict_placements(simulated_student)

        original_probability = float(
            current["placement_probability"]
        )

        simulated_probability = float(
            simulated_result["placement_probability"].iloc[0]
        )

        probability_change = (
            simulated_probability - original_probability
        )

        # Display comparison
        st.write("#### Original vs Simulated Prediction")

        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

        comparison_col1.metric(
            "Original Probability",
            f"{original_probability:.2f}%"
        )

        comparison_col2.metric(
            "Simulated Probability",
            f"{simulated_probability:.2f}%"
        )

        comparison_col3.metric(
            "Probability Change",
            f"{probability_change:+.2f}%",
            delta=f"{probability_change:+.2f}%"
        )

        st.info(
            "This is a hypothetical model simulation, not a guarantee "
            "of actual placement outcomes. It does not establish "
            "causation between score changes and placement."
        )

    if selected_page == "Model Performance":
        st.subheader("🤖 Model Test Performance")

        st.caption(
            "Metrics evaluated on the held-out synthetic test dataset (200 records)."
        )

        metric1, metric2, metric3, metric4, metric5 = st.columns(5)

        metric1.metric("Accuracy", f"{evaluation['accuracy'] * 100:.2f}%")
        metric2.metric("Precision", f"{evaluation['precision'] * 100:.2f}%")
        metric3.metric("Recall", f"{evaluation['recall'] * 100:.2f}%")
        metric4.metric("F1 Score", f"{evaluation['f1_score'] * 100:.2f}%")
        metric5.metric("ROC-AUC", f"{evaluation['roc_auc'] * 100:.2f}%")

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

        st.bar_chart(
            class_metrics,
            horizontal=True,
            height=270
        )

        # Analytics charts
        st.subheader("📈 Placement Analytics")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("Predicted Placement Distribution")

            placement_counts = (
                results["predicted_placement"]
                .value_counts()
                .rename_axis("Placement")
                .reset_index(name="Students")
            )

            placement_chart = alt.Chart(placement_counts).mark_bar().encode(
                x=alt.X(
                    "Placement:N",
                    axis=alt.Axis(
                        labelAngle=0,
                        title="Placement",
                        titleFontSize=15,
                        titlePadding=17,
                        titleFontWeight="normal"
                    )
                ),
                y=alt.Y(
                    "Students:Q",
                    title="Number of Students",
                    axis=alt.Axis(titleFontSize=14)
                ),
                tooltip=["Placement", "Students"]
            ).properties(height=300, padding={"bottom": 10})

            st.altair_chart(
                placement_chart,
                use_container_width=True
            )

        with chart_col2:
            st.write("Student Support Priority")

            risk_counts = (
                results["risk_level"]
                .value_counts()
                .rename_axis("Priority")
                .reset_index(name="Students")
            )

            # Add this code here
            risk_counts["Priority"] = pd.Categorical(
                risk_counts["Priority"],
                categories=["High", "Medium", "Low"],
                ordered=True
            )

            risk_counts = risk_counts.sort_values("Priority")

            # Existing chart code continues

            risk_chart = alt.Chart(risk_counts).mark_bar().encode(
                x=alt.X(
                    "Priority:N",
                    sort=["High", "Medium", "Low"],
                    axis=alt.Axis(
                        labelAngle=0,
                        title="Support Priority",
                        titleFontSize=15,
                        titlePadding=17,
                        titleFontWeight="normal"
                    )
                ),
                y=alt.Y(
                    "Students:Q",
                    title="Number of Students",
                    axis=alt.Axis(titleFontSize=14)
                ),
                tooltip=["Priority", "Students"]
            ).properties(height=300, padding={"bottom": 9})

            st.altair_chart(
                risk_chart,
                use_container_width=True
            )

        # Placement Probability Analysis
        st.write("### Placement Probability Distribution")

        probability_analysis = get_placement_probability_analysis(results)

        probability_df = probability_analysis.reset_index()

        probability_df.columns = ["Probability Range", "Students"]

        probability_chart = alt.Chart(probability_df).mark_bar().encode(
            x=alt.X(
                "Probability Range:N",
                axis=alt.Axis(
                    labelAngle=0,
                    title="Placement Probability",
                    titleFontSize=15,
                    titlePadding=17,
                    titleFontWeight="normal"
                )
            ),
            y=alt.Y(
                "Students:Q",
                title="Number of Students",
                axis=alt.Axis(titleFontSize=14)
            ),
            tooltip=["Probability Range", "Students"]
        ).properties(height=350)

        st.altair_chart(
            probability_chart,
            use_container_width=True
        )

    st.divider()

    if selected_page == "Student Support":
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

        display_df = filtered_results[display_columns].rename(
            columns={
                "student_id": "Student ID",
                "cgpa": "CGPA",
                "predicted_placement": "Placement Prediction",
                "placement_probability": "Placement Probability (%)",
                "risk_level": "Support Priority"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CGPA": st.column_config.NumberColumn(
                    "CGPA",
                    format="%.2f"
                ),
                "Placement Probability (%)": st.column_config.NumberColumn(
                    "Placement Probability (%)",
                    format="%.2f"
                )
            }
        )

        st.divider()

    if selected_page == "Training Recommendations":
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

            training_df = (
                training_counts
                .rename_axis("Training Area")
                .reset_index(name="Students")
            )

            training_chart = alt.Chart(training_df).mark_bar().encode(
                x=alt.X(
                    "Training Area:N",
                    axis=alt.Axis(
                        labelAngle=90,
                        title="Training Area",
                        titleFontSize=14
                    )
                ),
                y=alt.Y(
                    "Students:Q",
                    title="Number of Students"
                ),
                tooltip=["Training Area", "Students"]
            ).properties(height=350)

            st.altair_chart(
                training_chart,
                use_container_width=True
            )

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

            display_df = training_students[
            ["student_id", "cgpa", "recommended_training"]
        ].rename(
            columns={
                "student_id": "Student ID",
                "cgpa": "CGPA",
                "recommended_training": "Recommended Training"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CGPA": st.column_config.NumberColumn(
                    "CGPA",
                    format="%.2f"
                )
            }
        )
    # Academic Performance Analytics

    if selected_page == "Academic Analytics":
        st.divider()
        st.subheader("📊 Academic Performance Analytics")

        # 1. CGPA Distribution
        st.write("### CGPA-wise Student Distribution")
        cgpa_distribution = get_cgpa_distribution(results)
        st.bar_chart(
            cgpa_distribution,
            height=300
        )

        # 2. CGPA vs Predicted Placement
        st.write("### CGPA Range vs Predicted Placement")
        cgpa_placement = get_cgpa_placement_analysis(results)
        st.bar_chart(
            cgpa_placement,
            horizontal=True,
            height=450
        )

        # 3. Academic Performance Categories
        st.write("### Academic Performance Categories")
        academic_categories = get_academic_performance_categories(results)
        st.bar_chart(
            academic_categories,
            horizontal=True,
            height=280
        )

    if selected_page == "Skill Gap Analytics":
        st.divider()
        st.subheader("🛠️ Skill Gap Analytics")

        # 1. Average Skill Scores
        st.write("### Average Skill Scores")
        skill_averages = get_skill_average_scores(results)
        st.bar_chart(
            skill_averages,
            horizontal=True,
            height=250
        )

        # 2. Skill Performance Categories
        st.write("### Skill Performance Categories")
        skill_categories = get_skill_performance_categories(results)
        st.bar_chart(
            skill_categories,
            horizontal=True,
            height=350
        )

        # 3. Skill Gap Analysis
        st.write("### Students Needing Skill Improvement")
        skill_gaps = get_skill_gap_analysis(results)
        st.bar_chart(
            skill_gaps,
            horizontal=True,
            height=250
        )

        # 4. Skill Training Priority
        st.write("### Skill Training Priority")

        skill_priority = get_skill_training_priority(results)

        display_df = skill_priority.reset_index()

        display_df = display_df.rename(
            columns={
                "index": "Skill Area",
                "students_needing_improvement": "Students Needing Improvement",
                "priority_rank": "Priority Rank"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

except Exception as e:
    st.error(f"Something went wrong: {e}")