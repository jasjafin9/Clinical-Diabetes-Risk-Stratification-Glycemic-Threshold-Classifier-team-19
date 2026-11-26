
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go



RAW_FILE = "data/raw/diabetes_prediction_dataset.csv"
CLEAN_FILE = "data/processed/clean_diabetes_clinical.csv"

FIGURE_FOLDER = "reports/figures"
REPORT_FOLDER = "reports"

os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

#LOAD DATASET

if os.path.exists(CLEAN_FILE):
    data_file = CLEAN_FILE
else:
    data_file = RAW_FILE

df = pd.read_csv(data_file)

required_columns = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(subset=required_columns).copy()

df["diabetes"] = pd.to_numeric(
    df["diabetes"], errors="coerce"
)

df = df.dropna(subset=["diabetes"])
df = df[df["diabetes"].isin([0, 1])].copy()
df["diabetes"] = df["diabetes"].astype(int)

print("Dataset loaded successfully!")
print("Source:", data_file)
print("Total records:", len(df))

# Q10: SENSITIVITY-SPECIFICITY CURVE
actual = df["diabetes"].to_numpy()
hba1c = df["HbA1c_level"].to_numpy()

thresholds = np.arange(4.0, 10.01, 0.1)

sensitivities = []
specificities = []
youden_scores = []

for threshold in thresholds:

    predicted = (hba1c >= threshold).astype(int)

    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    sensitivity = (
        tp / (tp + fn) if (tp + fn) > 0 else 0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )

    youden_j = sensitivity + specificity - 1

    sensitivities.append(sensitivity)
    specificities.append(specificity)
    youden_scores.append(youden_j)


# Find optimal threshold

best_index = int(np.argmax(youden_scores))

best_threshold = thresholds[best_index]
best_sensitivity = sensitivities[best_index]
best_specificity = specificities[best_index]
best_youden = youden_scores[best_index]

false_positive_rates = 1 - np.array(specificities)


# Plot the curve

plt.figure(figsize=(10, 7))
plt.plot(
    false_positive_rates,
    sensitivities,
    linewidth=2,
    label="HbA1c Threshold Curve")
plt.scatter(
    false_positive_rates[best_index],
    best_sensitivity,
    color="red",
    s=90,
    label=f"Optimal Threshold = {best_threshold:.1f}")
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Reference Line")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("Sensitivity")
plt.title(
    "HbA1c Sensitivity-Specificity ROC curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURE_FOLDER + "/primary_envelope.png",
    dpi=300
)
plt.close()
print("\nQ10 completed successfully!")
print("Optimal HbA1c threshold:", round(best_threshold, 2))
print("Sensitivity:", round(best_sensitivity, 4))
print("Specificity:", round(best_specificity, 4))
print("Youden's J:", round(best_youden, 4))

# Q11: VIOLIN PLOTS AND CORRELATION HEATMAP

df["Diagnosis"] = df["diabetes"].map({
    0: "Non-diabetic",
    1: "Diabetic"
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6)
)

sns.violinplot(
    data=df,
    x="Diagnosis",
    y="HbA1c_level",
    hue="Diagnosis",
    ax=axes[0],
    inner="quartile",
    legend=False
)

axes[0].set_title(
    "HbA1c Distribution by Diabetes Status"
)
axes[0].set_xlabel("Diagnosis")
axes[0].set_ylabel("HbA1c Level")

sns.violinplot(
    data=df,
    x="Diagnosis",
    y="blood_glucose_level",
    hue="Diagnosis",
    ax=axes[1],
    inner="quartile",
    legend=False
)

axes[1].set_title(
    "Blood Glucose Distribution by Diabetes Status"
)

axes[1].set_xlabel("Diagnosis")
axes[1].set_ylabel("Blood Glucose Level")
plt.tight_layout()
plt.savefig(
    FIGURE_FOLDER + "/distribution_regimes.png",
    dpi=300
)
plt.close()
print("Q11 violin plots created successfully!")

# Q11: ANNOTATED CORRELATION HEATMAP

correlation_columns = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "hypertension",
    "heart_disease",
    "diabetes"
]

correlation_data = df[
    correlation_columns
].corr(numeric_only=True)

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_data,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

plt.title(
    "Correlation Heatmap of Clinical Variables"
)

plt.tight_layout()

plt.savefig(
    FIGURE_FOLDER + "/correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Q11 correlation heatmap created successfully!")


# Q12: INTERACTIVE 3D DASHBOARD(HTML)

dashboard_df = df.copy()
# Normalize clinical measurements for an
# exploratory composite score.
risk_columns = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

normalized_values = []

for column in risk_columns:

    minimum = dashboard_df[column].min()
    maximum = dashboard_df[column].max()

    if maximum == minimum:
        normalized = np.zeros(len(dashboard_df))
    else:
        normalized = (
            dashboard_df[column] - minimum
        ) / (maximum - minimum)

    normalized_values.append(normalized.to_numpy())


# Equal-weight exploratory score from 0 to 1

dashboard_df["risk_score"] = np.mean(
    normalized_values,
    axis=0
)


# Limit plotted records to keep the dashboard responsive

if len(dashboard_df) > 15000:
    dashboard_df = dashboard_df.sample(
        n=15000,
        random_state=42
    )

print(
    "Dashboard records:",
    len(dashboard_df)
)


# Separate diabetic and non-diabetic records

non_diabetic = dashboard_df[
    dashboard_df["diabetes"] == 0
]

diabetic = dashboard_df[
    dashboard_df["diabetes"] == 1
]


# Create a function to build each 3D trace

def create_trace(group, label, marker_color):

    custom_data = group[
        [
            "age",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level",
            "risk_score"
        ]
    ].to_numpy()

    return go.Scatter3d(
        x=group["age"],
        y=group["bmi"],
        z=group["HbA1c_level"],
        mode="markers",
        name=label,
        marker=dict(
            size=3,
            color=marker_color,
            opacity=0.65
        ),
        customdata=custom_data,
        hovertemplate=(
            "Age: %{customdata[0]:.1f}<br>"
            "BMI: %{customdata[1]:.2f}<br>"
            "HbA1c: %{customdata[2]:.2f}<br>"
            "Glucose: %{customdata[3]:.1f}<br>"
            "Risk Score: %{customdata[4]:.3f}"
            "<extra></extra>"
        )
    )


# Initial dashboard: show all records

fig = go.Figure()

fig.add_trace(
    create_trace(
        non_diabetic,
        "Non-diabetic",
        "blue"
    )
)

fig.add_trace(
    create_trace(
        diabetic,
        "Diabetic",
        "red"
    )
)


# Create slider frames for risk score thresholds

risk_thresholds = np.round(
    np.arange(0.0, 1.01, 0.1),
    1
)

frames = []

for threshold in risk_thresholds:

    filtered_non_diabetic = non_diabetic[
        non_diabetic["risk_score"] >= threshold
    ]

    filtered_diabetic = diabetic[
        diabetic["risk_score"] >= threshold
    ]

    frames.append(
        go.Frame(
            name=f"{threshold:.1f}",
            data=[
                create_trace(
                    filtered_non_diabetic,
                    "Non-diabetic",
                    "blue"
                ),
                create_trace(
                    filtered_diabetic,
                    "Diabetic",
                    "red"
                )
            ]
        )
    )

fig.frames = frames


# Diagnostic filter

fig.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            x=0.02,
            y=1.15,
            buttons=[
                dict(
                    label="All Patients",
                    method="update",
                    args=[
                        {"visible": [True, True]},
                        {"title": "All Patient Records"}
                    ]
                ),
                dict(
                    label="Diabetic Only",
                    method="update",
                    args=[
                        {"visible": [False, True]},
                        {"title": "Diabetic Patients"}
                    ]
                ),
                dict(
                    label="Non-diabetic Only",
                    method="update",
                    args=[
                        {"visible": [True, False]},
                        {"title": "Non-diabetic Patients"}
                    ]
                )
            ]
        )
    ],

    sliders=[
        dict(
            active=0,
            currentvalue=dict(
                prefix="Minimum Risk Score: "
            ),
            pad=dict(t=50),
            steps=[
                dict(
                    method="animate",
                    label=f"{threshold:.1f}",
                    args=[
                        [f"{threshold:.1f}"],
                        {
                            "mode": "immediate",
                            "frame": {
                                "duration": 0,
                                "redraw": True
                            },
                            "transition": {
                                "duration": 0
                            }
                        }
                    ]
                )
                for threshold in risk_thresholds
            ]
        )
    ],

    title="Interactive Clinical Diabetes Dashboard",
    scene=dict(
        xaxis_title="Age",
        yaxis_title="BMI",
        zaxis_title="HbA1c Level"
    ),

    height=750,
    template="plotly_white"
)
# Export interactive dashboard
fig.write_html(
    REPORT_FOLDER + "/interactive_dashboard.html"
)
print("Q12 interactive dashboard created successfully!")


# FINAL SUMMARY
# --------------------------------------------------

print("\n===================================")
print("CLINICAL DIABETES VISUALIZATION")
print("===================================")
print("Total records:", len(df))
print("Diabetic records:", (df["diabetes"] == 1).sum())
print("Non-diabetic records:", (df["diabetes"] == 0).sum())
print("\nQ10 Optimal HbA1c threshold:", round(best_threshold, 2))
print("Q10 Sensitivity:", round(best_sensitivity, 4))
print("Q10 Specificity:", round(best_specificity, 4))
print("\nGenerated files:")
print("1. reports/figures/primary_envelope.png")
print("2. reports/figures/distribution_regimes.png")
print("3. reports/figures/correlation_heatmap.png")
print("4. reports/interactive_dashboard.html")
print("\nALL VISUALIZATION TASKS COMPLETED!")
