import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

print("Loading Dataset...")

# ==============================
# Create models folder
# ==============================
os.makedirs("models", exist_ok=True)

# ==============================
# Load Dataset
# ==============================
df = pd.read_excel("sentinel_incident_dataset_500.xlsx")

print("Dataset Loaded Successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)

# ==============================
# Remove Empty Rows
# ==============================
df.dropna(how="all", inplace=True)

# Remove rows having empty Label
df = df[df["Label"].notna()]

# ==============================
# Fill Missing Values
# ==============================

# Numeric Columns
numeric_cols = [
    "FailedLoginCount",
    "NetworkRiskScore"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[numeric_cols] = df[numeric_cols].fillna(0)

# Categorical Columns
categorical_cols = [
    "TorIP",
    "CountryRisk",
    "UserRisk",
    "IncidentType"
]

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==============================
# Encode Categorical Features
# ==============================

feature_encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col].astype(str))

    feature_encoders[col] = encoder

joblib.dump(feature_encoders, "models/feature_encoders.pkl")

print("\nFeature Encoders Saved Successfully!")

# ==============================
# Encode Labels
# ==============================

severity_encoder = LabelEncoder()

df["Label"] = severity_encoder.fit_transform(df["Label"].astype(str))

joblib.dump(severity_encoder, "models/severity_encoder.pkl")

print("\nSeverity Encoder Saved Successfully!")

print("\nLabel Mapping")

for i, label in enumerate(severity_encoder.classes_):
    print(i, "=", label)

# ==============================
# Features and Target
# ==============================

X = df.drop(columns=["Label"])

y = df["Label"]

print("\nFeature Columns")
print(X.columns.tolist())

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

print("\nTraining Label Distribution")
print(y_train.value_counts())

# ==============================
# Apply SMOTE
# ==============================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE")
print(y_train_smote.value_counts())

print("\nPreprocessing Completed Successfully!")

print("\nTraining Data Shape :", X_train_smote.shape)
print("Testing Data Shape  :", X_test.shape)