
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "cardio_clean.csv"
MODEL_FILE = BASE / "model.pkl"

TARGET = "cardio"

# 1. Load the cleaned data
df = pd.read_csv(DATA_FILE)
FEATURES = [col for col in df.columns if col != TARGET]
X = df[FEATURES]
y = df[TARGET]

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train the Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# 4. Evaluate on data the model has never seen
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))
print()
print(classification_report(y_test, pred))

# 5. Save the model together with its feature list
joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
print("Saved:", MODEL_FILE.name)