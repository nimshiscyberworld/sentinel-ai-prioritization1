import pandas as pd

from shap_explainer import explain_prediction

sample = pd.DataFrame({

    "FailedLoginCount":[18],

    "TorIP":[1],

    "NetworkRiskScore":[95],

    "CountryRisk":[2],

    "UserRisk":[2],

    "IncidentType":[0]

})

top = explain_prediction(sample)

print("\nTop Features")

for i, feature in enumerate(top, 1):

    print(i, feature)