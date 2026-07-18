from flask import Flask, request, jsonify
import pandas as pd
import joblib

from shap_explainer import explain_prediction

app = Flask(__name__)

print("Loading AI Model...")

# -------------------------
# Load Model
# -------------------------

model = joblib.load("models/sentinel_incident_xgboost.pkl")
severity_encoder = joblib.load("models/severity_encoder.pkl")
feature_encoders = joblib.load("models/feature_encoders.pkl")

print("AI Model Loaded Successfully!")

# -------------------------
# Risk Score Calculation
# -------------------------

def calculate_risk_score(data):

    score = 0

    # Failed Login Count (30)
    failed = int(data["FailedLoginCount"])
    score += min((failed / 20) * 30, 30)

    # Tor IP (25)
    if data["TorIP"] == "Yes":
        score += 25

    # Network Risk (25)
    score += (int(data["NetworkRiskScore"]) / 100) * 25

    # Country Risk (10)
    country = {
        "Low": 2,
        "Medium": 6,
        "High": 10
    }

    score += country[data["CountryRisk"]]

    # User Risk (10)
    user = {
        "Low": 2,
        "Medium": 6,
        "High": 10
    }

    score += user[data["UserRisk"]]

    return round(score)

# -------------------------
# Home
# -------------------------

@app.route("/")
def home():

    return jsonify({
        "message": "Microsoft Sentinel AI Incident Prioritization API Running"
    })

# -------------------------
# Predict
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        df = pd.DataFrame([data])

        # Encode categorical columns
        categorical_columns = [
            "TorIP",
            "CountryRisk",
            "UserRisk",
            "IncidentType"
        ]

        for col in categorical_columns:
            df[col] = feature_encoders[col].transform(df[col])

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0]

        severity = severity_encoder.inverse_transform([prediction])[0]

        confidence = round(float(max(probability)) * 100, 2)

        # -------------------------
        # Risk Score
        # -------------------------

        risk_score = calculate_risk_score(data)

        # -------------------------
        # Priority + Reasons
        # -------------------------

        if severity == "High":

            priority = "P1"
            action = "Immediate Investigation Required"

            reasons = [
                "FailedLoginCount",
                "TorIP",
                "NetworkRiskScore",
                "CountryRisk",
                "UserRisk"
            ]

        elif severity == "Medium":

            priority = "P2"
            action = "Review Incident"

            reasons = [
                "FailedLoginCount",
                "CountryRisk"
            ]

        else:

            priority = "P3"
            action = "Monitor Only"

            reasons = [
                "FailedLoginCount"
            ]

        # Tree SHAP (for debugging / future use)
        shap_reasons = explain_prediction(df)
        print("Tree SHAP:", shap_reasons)

        return jsonify({

            "severity": severity,
            "confidence": confidence,
            "risk_score": risk_score,
            "priority": priority,
            "recommended_action": action,
            "top_reasons": reasons

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


if __name__ == "__main__":
    app.run(debug=True)