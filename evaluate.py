import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import X_test, y_test

print("====================================")
print("Evaluating XGBoost Model")
print("====================================")

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/sentinel_incident_xgboost.pkl")

severity_encoder = joblib.load("models/severity_encoder.pkl")

# -----------------------------
# Prediction
# -----------------------------
pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy")
print(round(accuracy * 100, 2), "%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report")

print(
    classification_report(
        y_test,
        pred,
        target_names=severity_encoder.classes_
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------
print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        pred
    )
)