
# 🎓 College Placement Prediction System

A Machine Learning-based web application designed for College Placement Cells and Administrators to analyze student placement data, predict placement outcomes, and identify students who may benefit from additional academic support.

## 🚀 Features

- Bulk student dataset upload using CSV and Excel files
- Data validation and preprocessing
- Placement prediction using Logistic Regression
- Placement probability estimation
- Student support priority classification (High, Medium, Low)
- Interactive placement analytics dashboard
- Student-wise prediction reports
- Student support priority filtering
- Model evaluation using Accuracy, Precision, Recall, F1-Score and ROC-AUC

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- Matplotlib / Streamlit Charts

## 📂 Project Structure

```text
College-Placement-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
├── assets/
│
└── src/
    ├── __init__.py
    ├── generate_dataset.py
    ├── data_preprocessing.py
    ├── train_model.py
    ├── predict.py
    └── analytics.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
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

The system currently uses Logistic Regression for binary classification.

The model predicts:
- Placed
- Not Placed

It also estimates placement probability and assigns a student support priority category.

## 📌 Current Project Status

**Completed:**
- Project setup and environment configuration
- Synthetic dataset generation
- Data preprocessing
- Logistic Regression model training and evaluation
- Student placement prediction module
- Streamlit analytics dashboard
- CSV/Excel upload
- Predictions on uploaded datasets
- Student support priority filter

**In Progress:**
- Academic training recommendations
- Advanced analytics
- Final UI improvements
- Project documentation

## ⚠️ Important Disclaimer

The current dataset contains synthetically generated student records for demonstration and development purposes.

Predictions are estimates produced by a trained model. They should support, not replace, human review and placement-cell decisions. The model has not yet been validated on real institutional placement data.

## 👨‍💻 Project Purpose

Developed as a B.Tech third-year mini-project to demonstrate the application of Machine Learning and data analytics in college placement management.
