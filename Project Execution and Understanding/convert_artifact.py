
import joblib
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Input and output paths
p = Path(r"D:\AQI-Prediction-CNN-LSTM\artifacts\scaler.joblib")
out = p.parent

# Load only a trusted artifact
obj = joblib.load(p)

scaler = obj["scaler"]
features = obj["feature_columns"]

# 1. Feature-wise scaling details
data = {
    "Index": range(1, len(features) + 1),
    "Feature Name": features
}

# Include available statistics for each feature
for attr, label in [
    ("mean_", "Training Mean"),
    ("scale_", "Scaling Factor (Std Dev)"),
    ("var_", "Training Variance")
]:
    if hasattr(scaler, attr):
        values = np.asarray(getattr(scaler, attr)).reshape(-1)
        if len(values) == len(features):
            data[label] = values

df = pd.DataFrame(data)

# 2. Save feature details and complete readable report
txt_path = out / "scaler_readable.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("AQI PROJECT - SCALER ARTIFACT REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("DICTIONARY KEYS\n")
    f.write(str(list(obj.keys())) + "\n\n")

    f.write("FEATURE COUNT\n")
    f.write(str(len(features)) + "\n\n")

    f.write("FEATURE NAMES AND SCALING STATISTICS\n")
    f.write(df.to_string(index=False) + "\n\n")

    f.write("SCALER OBJECT ATTRIBUTES\n")
    for name, value in vars(scaler).items():
        f.write(f"\n{name}:\n")
        if isinstance(value, np.ndarray):
            f.write(np.array2string(value, threshold=value.size + 1))
        else:
            f.write(repr(value))
        f.write("\n")

    f.write("\nOTHER DICTIONARY CONTENTS\n")
    for key, value in obj.items():
        if key != "scaler":
            f.write(f"\n{key}:\n{repr(value)}\n")

# 3. Save an Excel workbook with separate sheets
xlsx_path = out / "scaler_readable.xlsx"

attributes = []
for name, value in vars(scaler).items():
    if isinstance(value, np.ndarray):
        value_text = np.array2string(
            value, threshold=value.size + 1
        )
    else:
        value_text = repr(value)

    attributes.append({
        "Attribute": name,
        "Value": value_text
    })

metadata = pd.DataFrame([
    {"Property": "Artifact file", "Value": str(p)},
    {"Property": "Dictionary keys", "Value": ", ".join(obj.keys())},
    {"Property": "Scaler class", "Value": type(scaler).__name__},
    {"Property": "Feature count", "Value": len(features)},
    {
        "Property": "Samples seen during fitting",
        "Value": repr(getattr(scaler, "n_samples_seen_", "Not available"))
    }
])

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Feature Details", index=False)
    pd.DataFrame(attributes).to_excel(
        writer, sheet_name="Scaler Attributes", index=False
    )
    metadata.to_excel(writer, sheet_name="Metadata", index=False)

print("Conversion completed!")
print("Text file:", txt_path)
print("Excel file:", xlsx_path)
print("Total features exported:", len(features))
