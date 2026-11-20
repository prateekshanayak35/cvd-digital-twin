from pathlib import Path

import dice_ml
import joblib
import pandas as pd

print("Running counterfactual_dice.py (text-category version)")

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "cardio_clean.csv"
MODEL_FILE = BASE / "model.pkl"
TARGET = "cardio"

# 1. Load the trained model and the data
saved = joblib.load(MODEL_FILE)
model = saved["model"]
FEATURES = saved["features"]
df = pd.read_csv(DATA_FILE)


class ModelWrapper:
    """Makes sure DiCE always passes numbers in the right column order."""

    def __init__(self, model, features):
        self.model = model
        self.features = features

    def _prep(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
        return X[self.features].astype(float)

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))

    def predict(self, X):
        return self.model.predict(self._prep(X))


wrapped = ModelWrapper(model, FEATURES)

# 2. A sample high-risk patient (you can change these numbers later)
patient = pd.DataFrame([{
    "age_years": 55, "gender": 1, "height": 165, "weight": 85.0,
    "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 1,
    "smoke": 1, "alco": 0, "active": 0,
}])[FEATURES]

# DiCE needs the category columns as text (this avoids a data-type error)
CATEGORICAL = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
patient[CATEGORICAL] = patient[CATEGORICAL].astype(str)

row = patient.iloc[0]
base_risk = wrapped.predict_proba(patient)[0][1]
print(f"Current predicted risk: {base_risk * 100:.1f}%")

# 3. Decide what the patient is allowed to change
#    (age, gender and height stay fixed; only improvements are allowed)
permitted = {
    "ap_hi": [max(90, int(row["ap_hi"]) - 40), int(row["ap_hi"])],
    "ap_lo": [max(60, int(row["ap_lo"]) - 25), int(row["ap_lo"])],
    "weight": [round(float(row["weight"]) * 0.85, 1), float(row["weight"])],
}
features_to_vary = list(permitted)

lowest_value = {"cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0}
for col, lowest in lowest_value.items():
    options = [str(v) for v in range(lowest, int(row[col]) + 1)]
    if len(options) > 1:
        permitted[col] = options
        features_to_vary.append(col)

if int(row["active"]) == 0:
    permitted["active"] = ["0", "1"]
    features_to_vary.append("active")

# 4. Set up DiCE and ask for 3 lower-risk alternatives
continuous = ["age_years", "height", "weight", "ap_hi", "ap_lo"]
data = dice_ml.Data(
    dataframe=df.sample(20000, random_state=42),
    continuous_features=continuous,
    outcome_name=TARGET,
)
dice_model = dice_ml.Model(model=wrapped, backend="sklearn")
explainer = dice_ml.Dice(data, dice_model, method="random")

result = explainer.generate_counterfactuals(
    patient,
    total_CFs=3,
    desired_class=0,
    features_to_vary=features_to_vary,
    permitted_range=permitted,
    random_seed=42,
)

# 5. Show each option in plain language
cfs = result.cf_examples_list[0].final_cfs_df


def show(value):
    value = float(value)
    return str(int(value)) if value == int(value) else f"{value:.1f}"


if cfs is None or len(cfs) == 0:
    print("No counterfactuals were found. Try widening the permitted ranges.")
else:
    for n, (_, cf) in enumerate(cfs.iterrows(), start=1):
        cf = cf[FEATURES].astype(float)
        cf["ap_hi"] = round(cf["ap_hi"])
        cf["ap_lo"] = round(cf["ap_lo"])
        cf["weight"] = round(cf["weight"], 1)
        new_risk = wrapped.predict_proba(cf.to_frame().T)[0][1]
        print()
        print(f"Option {n}: risk {base_risk * 100:.1f}% -> {new_risk * 100:.1f}%")
        for col in FEATURES:
            if abs(cf[col] - float(row[col])) > 1e-9:
                print(f"   {col}: {show(row[col])} -> {show(cf[col])}")