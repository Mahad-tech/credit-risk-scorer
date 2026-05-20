import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

def train(data_dir: str = "data/processed", model_dir: str = "models"):
    """Train XGBoost on preprocessed German Credit data."""

    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n=== Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['Good Credit', 'Bad Credit'])}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/credit_model.joblib")
    print(f"Model saved to {model_dir}/credit_model.joblib")

    return model, X_test, y_test

if __name__ == "__main__":
    train()