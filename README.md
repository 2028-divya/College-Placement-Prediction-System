# 🎓 College Placement Prediction System

A Machine Learning-based web application designed for College Placement Cells and Administrators to analyze student placement data, estimate placement outcomes, identify students who may benefit from additional support, and explore placement-related insights through an interactive analytics dashboard.

## 🚀 Features

### 1. Student Data Management
- Bulk student dataset upload using CSV and Excel files
- Dataset validation and preprocessing
- Automatic student ID generation when IDs are unavailable
- Uploaded dataset preview

### 2. Placement Prediction
- Binary placement prediction using Logistic Regression
- Placement probability estimation
- Student support priority classification (High, Medium, Low)
- Student-wise prediction results

### 3. Placement Intelligence
- Individual student prediction analysis
- Identification of factors contributing to model predictions
- Positive and negative prediction contributors

### 4. What-If Simulator
- Interactive simulation of changes in selected student attributes
- Comparison of original and simulated placement probabilities
- Exploration of hypothetical academic and skill improvements

### 5. Student Support and Training Recommendations
- Student support priority filtering
- Identification of students requiring additional support
- Academic training recommendations
- Skill improvement analysis and training priorities

### 6. Advanced Analytics Dashboard
- Placement distribution analysis
- Placement probability distribution
- Student support priority analysis
- CGPA-based academic analytics
- Skill gap analytics
- Academic performance categories

### 7. Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion matrix
- Class-wise performance comparison

### 8. Prediction Reports
- Downloadable prediction reports in CSV format
- Downloadable prediction reports in Excel format

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression)
- **Web Framework:** Streamlit
- **Data Visualization:** Streamlit Charts, Altair
- **Excel Support:** OpenPyXL
- **Model Storage:** Pickle

## 📂 Project Structure

```text
College-Placement-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   └── model_evaluation.json
│
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── analytics.py
│   ├── explainability.py
│   └── training_recommendations.py
│
├── test_analytics.py
└── test_recommendations.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/2028-divya/College-Placement-Prediction-System.git
```

### 2. Open the project folder

```bash
cd College-Placement-Prediction-System
```

### 3. Create a virtual environment

```bash
py -m venv .venv
```

### 4. Activate the virtual environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 6. Generate the dataset

```bash
py src/generate_dataset.py
```

### 7. Preprocess the dataset

```bash
py src/data_preprocessing.py
```

### 8. Train the Machine Learning model

```bash
py src/train_model.py
```

### 9. Run the application

```bash
py -m streamlit run app.py
```

## 📊 Machine Learning Model

The system uses **Logistic Regression**, a supervised machine learning algorithm for binary classification.

The model predicts two possible outcomes:

- Placed
- Not Placed

It also estimates placement probability and assigns a support priority category based on the predicted probability.

Model performance is evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and a confusion matrix.

## 🧭 Dashboard Pages

The application provides eight dedicated dashboard pages:

1. **Overview** – Key placement indicators and management summary.
2. **Placement Intelligence** – Individual student predictions and contributing factors.
3. **What-If Simulator** – Exploration of hypothetical changes in student attributes.
4. **Student Support** – Identification and filtering of students by support priority.
5. **Training Recommendations** – Academic training recommendations.
6. **Academic Analytics** – CGPA distribution and academic performance analysis.
7. **Skill Gap Analytics** – Skill performance, gaps, and training priorities.
8. **Model Performance** – Model evaluation metrics and placement analytics.

## ⚠️ Important Disclaimer

The current dataset contains synthetically generated student records for demonstration and development purposes.

The model's predictions are estimates and should support, not replace, human review and placement-cell decisions. The model has not yet been validated using real institutional placement data.

Training recommendations and support priority categories are based on predefined project rules and should not be treated as official institutional standards.

## 🎯 Project Purpose

Developed as a B.Tech third-year mini-project to demonstrate the application of Machine Learning and data analytics in college placement management.

The system aims to help placement-cell administrators explore placement trends, identify potential support requirements, and make more informed decisions using data-driven insights.