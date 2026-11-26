

import csv
import os
from datetime import datetime


NUMERIC_RANGES = {
    "age": (0, 120),
    "bmi": (10, 80),
    "HbA1c_level": (3.5, 15.0),
    "blood_glucose_level": (50, 400),
}

BINARY_COLUMNS = ("hypertension", "heart_disease", "diabetes")


def validate_record(row):
 
    reasons = []

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        try:
            value = float(row[column])
            if not minimum <= value <= maximum:
                reasons.append(
                    f"{column}={value} outside [{minimum}, {maximum}]"
                )
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{column} is missing or not numeric")

    for column in BINARY_COLUMNS:
        try:
            value = int(row[column])
            if value not in (0, 1):
                reasons.append(f"{column} must be 0 or 1")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{column} is missing or not an integer")
    smoking = row.get("smoking_history", "")
    if smoking is None or not str(smoking).strip():
        reasons.append("smoking_history is missing")

    return len(reasons) == 0, reasons


def load_and_clean_data(input_file, output_file, log_file):
   
    valid_rows = []
    rejected_rows = []
    fieldnames = None

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("The input CSV has no header row.")

        for line_number, row in enumerate(reader, start=2):
            is_valid, reasons = validate_record(row)

            if is_valid:
                valid_rows.append(row)
            else:
                rejected_rows.append((line_number, reasons))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "w", encoding="utf-8") as file:
        file.write("DIABETES DATA INGESTION AUDIT\n")
        file.write(f"Generated: {timestamp}\n")
        file.write(f"Input file: {input_file}\n")
        file.write(f"Valid records: {len(valid_rows)}\n")
        file.write(f"Rejected records: {len(rejected_rows)}\n")
        file.write("\nREJECTED RECORD DETAILS\n")

        for line_number, reasons in rejected_rows:
            file.write(f"CSV line {line_number}: {'; '.join(reasons)}\n")

    print(f"Valid records: {len(valid_rows)}")
    print(f"Rejected records: {len(rejected_rows)}")
    print(f"Clean CSV saved to: {output_file}")
    print(f"Audit log saved to: {log_file}")

    return len(valid_rows), len(rejected_rows)


class DiabetesRiskStratifier:
  
    def __init__(self, hba1c_thresh=6.5, glucose_thresh=140):
        if hba1c_thresh < 0 or glucose_thresh < 0:
            raise ValueError("Thresholds cannot be negative.")

        self.hba1c_thresh = hba1c_thresh
        self.glucose_thresh = glucose_thresh
        self.data = []

    def load_from_csv(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        with open(filename, "r", newline="", encoding="utf-8-sig") as file:
            self.data = list(csv.DictReader(file))

        return self.data

    def evaluate_thresholds(self):
        results = []

        for row in self.data:
            hba1c = float(row["HbA1c_level"])
            glucose = float(row["blood_glucose_level"])

            results.append({
                "hba1c_high": hba1c >= self.hba1c_thresh,
                "glucose_high": glucose >= self.glucose_thresh,
                "both_high": (
                    hba1c >= self.hba1c_thresh
                    and glucose >= self.glucose_thresh
                ),
            })

        return results

    def summary_report(self):
        flags = self.evaluate_thresholds()
        total = len(flags)

        hba1c_count = sum(item["hba1c_high"] for item in flags)
        glucose_count = sum(item["glucose_high"] for item in flags)
        both_count = sum(item["both_high"] for item in flags)

        report = {
            "total_patients": total,
            "hba1c_above_threshold": hba1c_count,
            "glucose_above_threshold": glucose_count,
            "both_above_threshold": both_count,
        }

        print("THRESHOLD SUMMARY")
        for key, value in report.items():
            print(f"{key}: {value}")

        return report


if __name__ == "__main__":
    raw_file = "data/raw/diabetes_prediction_dataset.csv"
    clean_file = "data/processed/clean_diabetes_clinical.csv"
    audit_file = "data/processed/diabetes_ingestion_audit.log"

    load_and_clean_data(raw_file, clean_file, audit_file)

    stratifier = DiabetesRiskStratifier()
    stratifier.load_from_csv(clean_file)
    stratifier.summary_report()
