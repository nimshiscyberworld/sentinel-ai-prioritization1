import joblib
import shap
import numpy as np

# Load Model
model = joblib.load("models/sentinel_incident_xgboost.pkl")

# Create SHAP Explainer
explainer = shap.TreeExplainer(model)


def explain_prediction(df):

    # Predict class
    pred = int(model.predict(df)[0])

    # SHAP values
    shap_values = explainer.shap_values(df)

    # Shape = (samples, features, classes)
    values = np.abs(shap_values[0, :, pred])

    # Pair feature names with SHAP values
    feature_scores = list(zip(df.columns, values))

    # Sort by importance
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top 5 features
    return [feature for feature, score in feature_scores[:5]]