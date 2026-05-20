import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# Column names from UCI documentation
COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose", "credit_amount",
    "savings_account", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age", "other_installments",
    "housing", "existing_credits", "job", "dependents", "telephone", "foreign_worker",
    "target"
]

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker"
]

def load_and_preprocess(data_path: str, save_dir: str = "data/processed"):
    """Load German Credit dataset, encode categoricals, scale numerics, split."""

    # Load — space separated, no header
    df = pd.read_csv(data_path, sep=" ", header=None, names=COLUMNS)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Target: 1=Good, 2=Bad → convert to 0=Good, 1=Bad (standard binary)
    df["target"] = df["target"].map({1: 0, 2: 1})
    print(f"Class distribution:\n{df['target'].value_counts()}")

    # Encode categorical columns
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    print(f"Shape after encoding: {df_encoded.shape}")

    # Split features and target
    X = df_encoded.drop(columns=["target"])
    y = df_encoded["target"]

    # Train/test split — stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    # Save processed data
    os.makedirs(save_dir, exist_ok=True)
    X_train_scaled.to_csv(f"{save_dir}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{save_dir}/X_test.csv",   index=False)
    y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{save_dir}/y_test.csv",   index=False)
    joblib.dump(scaler, f"{save_dir}/scaler.joblib")

    print(f"Processed data saved to {save_dir}/")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess("data/raw/german.data")