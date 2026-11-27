import numpy as np
import pandas as pd


class DiabetesAnalyzer:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        required = [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level",
            "blood_glucose_level", "diabetes"
        ]

        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        numeric_columns = [
            "age", "bmi", "HbA1c_level", "blood_glucose_level",
            "hypertension", "heart_disease", "diabetes"
        ]

        for col in numeric_columns:
            self.df[col] = pd.to_numeric(
                self.df[col], errors="coerce"
            )

        self.df = self.df.dropna(
            subset=numeric_columns
        ).copy()

        self.df["patient_row"] = range(len(self.df))

    def calculate_risk_score(self, weights=(0.15, 0.15, 0.35, 0.35)):

        features = self.df[
            ["bmi", "age", "HbA1c_level", "blood_glucose_level"]
        ].to_numpy(dtype=float)

        if len(features) == 0:
            raise ValueError("No valid patient records available.")

        minimum = np.min(features, axis=0)
        maximum = np.max(features, axis=0)

        spread = maximum - minimum
        spread[spread == 0] = 1

        normalized = (features - minimum) / spread
        weights = np.asarray(weights, dtype=float)
        if (
            len(weights) != 4
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError(
                "Weights must be four non-negative values summing to 1."
            )
        scores = normalized @ weights
        manual_scores = np.sum(normalized * weights, axis=1)

        if not np.allclose(scores, manual_scores):
            raise ValueError("Risk score verification failed.")

        self.df["composite_risk_score"] = scores

        return self.df[["composite_risk_score"]]
    
    def analyze_high_risk_cohorts(self):

        hba1c_high = self.df["HbA1c_level"].to_numpy() >= 6.5

        glucose_high = (
            self.df["blood_glucose_level"].to_numpy() >= 140
        )

        diabetes_positive = self.df["diabetes"].to_numpy() == 1

        high_biomarker = hba1c_high & glucose_high

        high_risk_diabetic = high_biomarker & diabetes_positive

        hypertension = self.df["hypertension"].to_numpy() == 1

        heart_disease = self.df["heart_disease"].to_numpy() == 1

        hba1c_values = self.df["HbA1c_level"].to_numpy()

        glucose_values = self.df["blood_glucose_level"].to_numpy()

        result = {
            "hba1c_and_glucose_high": int(
                np.sum(high_biomarker)
            ),

            "high_biomarker_and_diabetes": int(
                np.sum(high_risk_diabetic)
            ),

            "overlap_with_hypertension": int(
                np.sum(high_risk_diabetic & hypertension)
            ),

            "overlap_with_heart_disease": int(
                np.sum(high_risk_diabetic & heart_disease)
            ),

            "mean_hba1c": float(np.mean(hba1c_values)),

            "std_hba1c": float(np.std(hba1c_values)),

            "mean_glucose": float(np.mean(glucose_values)),

            "std_glucose": float(np.std(glucose_values))
        }

        return result
    
    def perform_regression(self):

        x = self.df[
            ["HbA1c_level", "blood_glucose_level", "bmi"]
        ].to_numpy(dtype=float)

        y = self.df["diabetes"].to_numpy(dtype=float)

        x_with_intercept = np.column_stack(
            (np.ones(len(x)), x)
        )

        coefficients, residuals, rank, singular_values = (
            np.linalg.lstsq(
                x_with_intercept, y, rcond=None
            )
        )

        predictions = x_with_intercept @ coefficients

        ss_res = np.sum((y - predictions) ** 2)

        ss_total = np.sum((y - np.mean(y)) ** 2)

        r_squared = (
            1 - (ss_res / ss_total)
            if ss_total != 0
            else 0.0
        )

        y_std = np.std(y)

        if y_std != 0:
            standardized_coefficients = (
                coefficients[1:] * np.std(x, axis=0) / y_std
            )
        else:
            standardized_coefficients = np.zeros(3)

        result = {
            "intercept_b0": float(coefficients[0]),
            "hba1c_b1": float(coefficients[1]),
            "glucose_b2": float(coefficients[2]),
            "bmi_b3": float(coefficients[3]),
            "r_squared": float(r_squared),
            "standardized_hba1c": float(standardized_coefficients[0]),
            "standardized_glucose": float(standardized_coefficients[1]),
            "standardized_bmi": float(standardized_coefficients[2]),
            "rank": int(rank)
        }

        return result
    
    def demographic_analysis(self):

        grouped = (
            self.df.groupby(
                ["gender", "smoking_history"],
                dropna=False
            )
            .agg(
                patient_count=("diabetes", "size"),
                diabetic_count=("diabetes", "sum"),
                diabetes_prevalence=("diabetes", "mean")
            )
            .reset_index()
        )
        self.df["age_decile"] = pd.qcut(
            self.df["age"],
            q=10,
            duplicates="drop"
        )
        age_results = (
            self.df.groupby(
                "age_decile",
                observed=False
            )
            .agg(
                patient_count=("diabetes", "size"),
                diabetes_prevalence=("diabetes", "mean")
            )
            .reset_index()
        )
        gender_balance = self.df["gender"].value_counts(
            dropna=False
        )

        return grouped, age_results, gender_balance
    
    def age_bmi_analysis(self):
        self.df["age_category"] = pd.cut(
            self.df["age"],
            bins=[
                -np.inf,
                35,
                np.nextafter(55, np.inf),
                np.inf
            ],
            labels=["Young", "Middle", "Senior"],
            right=False
        )
        self.df["bmi_category"] = pd.cut(
            self.df["bmi"],
            bins=[-np.inf, 25, 30, np.inf],
            labels=["Normal", "Overweight", "Obese"],
            right=False
        )
        pivot = pd.pivot_table(
            self.df,
            values="diabetes",
            index="age_category",
            columns="bmi_category",
            aggfunc="mean",
            margins=True,
            margins_name="All",
            observed=False
        )

        pivot = pivot.reindex(
            index=["Young", "Middle", "Senior", "All"],
            columns=["Normal", "Overweight", "Obese", "All"]
        )

        return pivot
    
    def biomarker_ranking(self):
        if "composite_risk_score" not in self.df.columns:
            self.calculate_risk_score()

        biomarker_columns = [
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]
        long_data = pd.melt(
            self.df,
            id_vars=[
                "patient_row",
                "gender",
                "age",
                "diabetes",
                "composite_risk_score"
            ],
            value_vars=biomarker_columns,
            var_name="Biomarker",
            value_name="Value"
        )
        top_five = (
            long_data.sort_values(
                "composite_risk_score",
                ascending=False
            )
            .groupby(
                ["gender", "Biomarker"],
                group_keys=False
            )
            .head(5)
            .reset_index(drop=True)
        )
        return long_data, top_five
    
if __name__ == "__main__":

    input_file = "data/processed/clean_diabetes_clinical.csv"
    data = pd.read_csv(input_file)
    analyzer = DiabetesAnalyzer(data)

    print("\nQ4 - COMPOSITE RISK SCORE")
    print(analyzer.calculate_risk_score().head())

    print("\nQ5 - HIGH-RISK COHORTS")
    print(analyzer.analyze_high_risk_cohorts())

    print("\nQ6 - REGRESSION")
    print(analyzer.perform_regression())

    print("\nQ7 - DEMOGRAPHICS")

    grouped, age_results, gender_balance = (
        analyzer.demographic_analysis()
    )
    print("\nGender and smoking history:")
    print(grouped)

    print("\nAge deciles:")
    print(age_results)

    print("\nGender balance:")
    print(gender_balance)

    print("\nQ8 - AGE AND BMI PIVOT")
    print(analyzer.age_bmi_analysis())

    print("\nQ9 - TOP FIVE HIGH-RISK PATIENTS")

    long_data, top_five = analyzer.biomarker_ranking()
    print(
        top_five[
            [
                "patient_row",
                "gender",
                "age",
                "Biomarker",
                "Value",
                "diabetes",
                "composite_risk_score"
            ]
        ]
    )