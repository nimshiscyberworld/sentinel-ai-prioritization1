import os
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Import data from preprocessing
from preprocessing import (
    X_train_smote,
    y_train_smote,
    X_test,
    y_test
)

print("====================================")
print("Training Microsoft Sentinel AI Model")
print("====================================")

# -----------------------------
# Create XGBoost Model
# -----------------------------
model = XGBClassifier(

    objective="multi:softprob",

    num_class=3,

    n_estimators=200,

    learning_rate=0.1,

    max_depth=6,

    random_state=42,

    eval_metric="mlogloss"
)

# -----------------------------
# Train Model
# -----------------------------
print("\nTraining XGBoost...")

model.fit(
    X_train_smote,
    y_train_smote
)

print("Training Completed Successfully!")

# -----------------------------
# Quick Accuracy
# -----------------------------
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/sentinel_incident_xgboost.pkl"
)

print("\nModel Saved Successfully!")

print("\nLocation")
print("models/sentinel_incident_xgboost.pkl")