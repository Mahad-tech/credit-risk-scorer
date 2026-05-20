import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
import pytest

DATA_PATH  = "data/raw/german.data"
MODEL_PATH = "models/credit_model.joblib"
X_TEST     = "data/processed/X_test.csv"
Y_TEST     = "data/processed/y_test.csv"

def test_raw_data_shape():
    from src.data.preprocess import COLUMNS
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=COLUMNS)
    assert df.shape[0] == 1000
    assert df.shape[1] == 21

def test_no_nulls_after_preprocessing():
    X = pd.read_csv(X_TEST)
    assert X.isnull().sum().sum() == 0

def test_model_output_range():
    model  = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST)
    probs  = model.predict_proba(X_test)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()

def test_shap_values_shape():
    import shap
    model  = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    assert shap_values.shape == X_test.shape