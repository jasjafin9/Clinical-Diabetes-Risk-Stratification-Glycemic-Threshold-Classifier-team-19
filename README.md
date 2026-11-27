# Clinical Diabetes Risk Stratification & Glycemic Threshold Classifier

**Project 19 – Python Programming**

## 1. Project Overview

This project focuses on analysing diabetes-related clinical data and identifying patterns associated with diabetes risk.

The project uses Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Plotly to perform data ingestion, cleaning, analysis, visualisation, and interactive dashboard creation.

## 2. Dataset

The project uses the Diabetes Prediction Dataset obtained from Kaggle.

The dataset includes the following features:

- Gender
- Age
- Hypertension
- Heart disease
- Smoking history
- BMI
- HbA1c level
- Blood glucose level
- Diabetes status

## 3. Project Objectives

- Load and validate the raw diabetes dataset.
- Clean the data and handle invalid records.
- Generate an audit log during data ingestion.
- Analyse diabetes-related clinical features.
- Evaluate HbA1c and blood glucose thresholds.
- Perform statistical analysis and regression.
- Visualise distributions and feature correlations.
- Create an interactive 3D dashboard.

## 4. Project Structure

```text
Clinical-Diabetes-Risk-Stratification-Glycemic-Threshold-Classifier-team-19/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── reports/
│   ├── figures/
│   └── interactive_dashboard.html
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── analysis.py
│   └── visualisation.py
│
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Git and GitHub

## 6. Installation

Clone the repository:

```bash
git clone https://github.com/jasjafin9/Clinical-Diabetes-Risk-Stratification-Glycemic-Threshold-Classifier-team-19.git
```

Move into the project directory:

```bash
cd Clinical-Diabetes-Risk-Stratification-Glycemic-Threshold-Classifier-team-19
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 7. Running the Project

Run the pipeline using:

```bash
python run_pipeline.py
```

If your pipeline requires command-line arguments, use the exact arguments defined in `run_pipeline.py`.

## 8. Expected Outputs

The project generates:

- Cleaned clinical dataset.
- Data ingestion audit log.
- Statistical analysis results.
- Sensitivity and specificity threshold plot.
- Clinical feature distribution plots.
- Correlation heatmap.
- Interactive Plotly dashboard.

The visualisation files are stored in the `reports/` directory.

## 9. Team Members

**Team 19**

- Jafin – Team Lead and Visualisation
- Vyshnav – Data Ingestion
- Pranav – Data Analysis

## 10. Disclaimer

This project is developed for academic and educational purposes. The risk scores and threshold analysis are exploratory and are not intended to replace professional medical diagnosis.