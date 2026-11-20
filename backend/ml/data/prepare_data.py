from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW_FILE = BASE / "data" / "cardio_train.csv"
CLEAN_FILE = BASE / "data" / "cardio_clean.csv"

# 1. Load (this file uses ";" as the separator)
df = pd.read_csv(RAW_FILE, sep=";")
print("Rows before cleaning:", len(df))

# 2. Convert age from days to years
df["age_years"] = (df["age"] / 365.25).round().astype(int)

# 3. Remove impossible or extreme values
df = df[
    (df["ap_hi"].between(70, 250))
    & (df["ap_lo"].between(40, 150))
    & (df["ap_hi"] > df["ap_lo"])
    & (df["height"].between(120, 220))
    & (df["weight"].between(30, 200))
]

# 4. Keep only the columns we need
columns = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "cardio",
]
df = df[columns]

print("Rows after cleaning:", len(df))
print("Share with heart disease:", round(df["cardio"].mean(), 3))

# 5. Save the cleaned file
df.to_csv(CLEAN_FILE, index=False)
print("Saved:", CLEAN_FILE.name)