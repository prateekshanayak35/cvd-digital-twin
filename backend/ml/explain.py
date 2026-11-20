from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "cardio_clean.csv"
MODEL_FILE = BASE / "model.pkl"

# 1. Load the trained model and the data
saved = joblib.load(MODEL_FILE)
model = saved["model"]
FEATURES = saved["features"]
X = pd.read_csv(DATA_FILE)[FEATURES]

# 2. Create the SHAP explainer for the Random Forest
explainer = shap.TreeExplainer(model)


def class1_values(sv):
    """Return SHAP values for 'has heart disease' (class 1), any SHAP version."""
    if isinstance(sv, list):
        return np.array(sv[1])
    sv = np.array(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


# 3. GLOBAL explanation: which features matter most overall?
sample = X.sample(500, random_state=42)
values = class1_values(explainer.shap_values(sample))
importance = pd.Series(
    np.abs(values).mean(axis=0), index=FEATURES
).sort_values(ascending=False)

print("=== Overall feature importance (average impact on risk) ===")
for name, score in importance.items():
    print(f"{name:<12} {score * 100:5.1f} points")

# 4. LOCAL explanation: why did the model give THIS patient this risk?
patient = X.iloc[[0]]
risk = model.predict_proba(patient)[0][1]
base_value = float(np.atleast_1d(explainer.expected_value)[-1])
contrib = class1_values(explainer.shap_values(patient))[0]

print()
print("=== One patient ===")
print(patient.iloc[0].to_string())
print()
print(f"Average risk of all patients: {base_value * 100:.1f}%")
print(f"This patient's predicted risk: {risk * 100:.1f}%")
print()
print("What pushed this patient's risk up (+) or down (-):")
order = np.argsort(-np.abs(contrib))
for i in order:
    print(f"{FEATURES[i]:<12} value={patient.iloc[0, i]:<7} {contrib[i] * 100:+.1f} points")